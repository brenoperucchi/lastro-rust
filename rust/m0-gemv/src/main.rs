//! M0 do runtime MoE em Rust (plano `docs/PLANO-runtime-rust-moe-20261029.md`): GEMV BF16 de um expert em batch 1.
//!
//! Pergunta: um kernel Tile do cutile-rs chega perto do cuBLAS no produto matriz×vetor que domina o decode de um MoE
//! (`y = W·x`, W em BF16 com M×K de um expert, x em BF16, acumulação e saída em f32)? O critério sugerido na consulta
//! ask-5 é abaixo de ~60–70% do cuBLAS sem caminho de melhora.
//!
//! Medição: o tempo vem de eventos da GPU em volta de `ITERS` lançamentos seguidos, sem sincronizar entre eles. Os pesos
//! giram entre cópias distintas que somam mais que o L2 da 4090 (72 MB), para medir leitura da VRAM, como num decode em
//! que cada expert é lido uma vez por token. O tempo de CPU para enfileirar um lançamento é medido à parte: se passar do
//! tempo de GPU, o número é limitado pelo lançamento, não pelo kernel.
use cuda_core::{Event, Stream};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Reshape;
use std::ffi::c_void;
use std::time::Instant;
use m0_gemv::{host_values, kernels};

/// FFI mínima do cuBLAS (só o necessário para `cublasGemmEx` com n = 1).
mod cublas {
    use std::ffi::c_void;
    pub type Handle = *mut c_void;
    pub const OP_N: i32 = 0;
    pub const OP_T: i32 = 1;
    pub const R_32F: i32 = 0;
    pub const R_16BF: i32 = 14;
    pub const COMPUTE_32F: i32 = 68;
    pub const GEMM_DEFAULT: i32 = -1;
    extern "C" {
        pub fn cublasCreate_v2(handle: *mut Handle) -> i32;
        pub fn cublasDestroy_v2(handle: Handle) -> i32;
        pub fn cublasSetStream_v2(handle: Handle, stream: *mut c_void) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub fn cublasGemmEx(
            handle: Handle, transa: i32, transb: i32, m: i32, n: i32, k: i32,
            alpha: *const c_void, a: *const c_void, a_type: i32, lda: i32,
            b: *const c_void, b_type: i32, ldb: i32,
            beta: *const c_void, c: *mut c_void, c_type: i32, ldc: i32,
            compute_type: i32, algo: i32,
        ) -> i32;
    }
}

const L2_BYTES: usize = 72 << 20;
const ITERS: usize = 3000;
const WARMUP: usize = 200;
const CPU_BATCH: usize = 256;

struct Shape {
    name: &'static str,
    m: usize,
    k: usize,
}

const SHAPES: &[Shape] = &[
    Shape { name: "Qwen3-30B gate/up", m: 768, k: 2048 },
    Shape { name: "Qwen3-30B down", m: 2048, k: 768 },
    Shape { name: "OLMoE gate/up", m: 1024, k: 2048 },
    Shape { name: "OLMoE down", m: 2048, k: 1024 },
    Shape { name: "banda 4096²", m: 4096, k: 4096 },
];

/// Resultado de uma variante: µs por GEMV no tempo da GPU e µs de CPU por lançamento enfileirado.
struct Timing {
    gpu_us: f64,
    cpu_us: f64,
}

/// Enfileira `launch(i)` `n` vezes entre dois eventos; devolve µs de GPU e de CPU por lançamento.
fn time_launches(
    stream: &Arc<Stream>,
    ev: (&Event, &Event),
    n: usize,
    mut launch: impl FnMut(usize),
) -> Timing {
    for i in 0..WARMUP {
        launch(i);
    }
    unsafe { stream.synchronize() }.unwrap();
    ev.0.record(stream).unwrap();
    for i in 0..n {
        launch(i);
    }
    ev.1.record(stream).unwrap();
    ev.1.synchronize().unwrap();
    let gpu_ms = ev.0.elapsed_time(ev.1).unwrap() as f64;
    // Custo de enfileirar, com um lote menor que a fila de lançamentos do driver (~1024) para a CPU não bloquear.
    let t = Instant::now();
    for i in 0..CPU_BATCH {
        launch(i);
    }
    let cpu = t.elapsed().as_secs_f64();
    unsafe { stream.synchronize() }.unwrap();
    Timing { gpu_us: gpu_ms * 1e3 / n as f64, cpu_us: cpu * 1e6 / CPU_BATCH as f64 }
}

fn reference(w: &[bf16], x: &[bf16], m: usize, k: usize) -> Vec<f64> {
    (0..m)
        .map(|r| (0..k).map(|c| w[r * k + c].to_f64() * x[c].to_f64()).sum())
        .collect()
}

/// Maior erro relativo à norma infinita da referência.
fn max_rel_err(got: &[f32], want: &[f64]) -> f64 {
    let scale = want.iter().fold(0f64, |a, v| a.max(v.abs())).max(1e-30);
    got.iter().zip(want).fold(0f64, |a, (g, w)| a.max((*g as f64 - w).abs())) / scale
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let stream = device.new_stream()?;
    let ev = (device.new_event()?, device.new_event()?);
    let mut handle: cublas::Handle = std::ptr::null_mut();
    unsafe {
        assert_eq!(cublas::cublasCreate_v2(&mut handle), 0, "cublasCreate");
        assert_eq!(cublas::cublasSetStream_v2(handle, stream.cu_stream() as *mut c_void), 0, "cublasSetStream");
    }
    let only: Option<String> = std::env::args().nth(1);

    println!("forma; MxK; cópias; variante; µs GPU; µs CPU/lançamento; GB/s; erro rel. máx.");
    for (si, s) in SHAPES.iter().enumerate() {
        if only.as_deref().is_some_and(|o| !s.name.contains(o)) {
            continue;
        }
        let (m, k) = (s.m, s.k);
        let w_bytes = m * k * 2;
        let copies = (4 * L2_BYTES).div_ceil(w_bytes).max(2);
        let w_host = Arc::new(host_values(m * k, 0x9e37 + si as u64));
        let x_host = Arc::new(host_values(k, 0x51ed + si as u64));
        let want = reference(&w_host, &x_host, m, k);
        let mut ws: Vec<Tensor<bf16>> = Vec::with_capacity(copies);
        for _ in 0..copies {
            ws.push(api::copy_host_vec_to_device(&w_host).sync_on(&stream)?.reshape(&[m, k])?);
        }
        let x: Tensor<bf16> = api::copy_host_vec_to_device(&x_host).sync_on(&stream)?;
        let bytes = (w_bytes + k * 2 + m * 4) as f64;
        let row = |variant: &str, t: &Timing, err: f64| {
            println!(
                "{}; {m}x{k}; {copies}; {variant}; {:.2}; {:.2}; {:.0}; {err:.1e}",
                s.name,
                t.gpu_us,
                t.cpu_us,
                bytes / (t.gpu_us * 1e-6) / 1e9
            );
        };

        // cuBLAS: W em linha (M×K) é W^T em coluna (K×M, ld = K); y = op_T(A)·x.
        let y_blas: Arc<Tensor<f32>> = Arc::new(api::zeros::<f32>(&[m]).sync_on(&stream)?);
        let y_ptr = y_blas.device_pointer().cu_deviceptr() as *mut c_void;
        let x_ptr = x.device_pointer().cu_deviceptr() as *const c_void;
        let w_ptrs: Vec<*const c_void> =
            ws.iter().map(|w| w.device_pointer().cu_deviceptr() as *const c_void).collect();
        let (alpha, beta) = (1f32, 0f32);
        let blas = |i: usize| unsafe {
            let st = cublas::cublasGemmEx(
                handle, cublas::OP_T, cublas::OP_N, m as i32, 1, k as i32,
                &alpha as *const f32 as *const c_void, w_ptrs[i % copies], cublas::R_16BF, k as i32,
                x_ptr, cublas::R_16BF, k as i32,
                &beta as *const f32 as *const c_void, y_ptr, cublas::R_32F, m as i32,
                cublas::COMPUTE_32F, cublas::GEMM_DEFAULT,
            );
            assert_eq!(st, 0, "cublasGemmEx");
        };
        let t_blas = time_launches(&stream, (&ev.0, &ev.1), ITERS, blas);
        let got = api::copy_device_to_host_vec(&y_blas).sync_on(&stream)?;
        row("cuBLAS GemmEx", &t_blas, max_rel_err(&got, &want));

        for bm in [2usize, 4, 8, 16, 32] {
            for bk in [128usize, 256, 512, 1024] {
                if m % bm != 0 || k % bk != 0 {
                    continue;
                }
                let generics = vec![bm.to_string(), bk.to_string(), k.to_string()];
                let mut y = Some(api::zeros::<f32>(&[m]).sync_on(&stream)?.partition([bm]));
                let tile = |i: usize| {
                    // SAFETY: `unchecked_accesses`: M é múltiplo de BM e K de BK (conferido acima), então toda fatia
                    // carregada está dentro de W e de x.
                    let (out, _, _) = unsafe {
                        kernels::gemv(y.take().unwrap(), &ws[i % copies], &x)
                            .generics(generics.clone())
                            .async_on(&stream)
                    }
                    .unwrap();
                    y = Some(out);
                };
                let t = time_launches(&stream, (&ev.0, &ev.1), ITERS, tile);
                let got = y.take().unwrap().unpartition().to_host_vec().sync_on(&stream)?;
                row(&format!("Tile BM={bm} BK={bk}"), &t, max_rel_err(&got, &want));
            }
        }
    }
    unsafe { cublas::cublasDestroy_v2(handle) };
    Ok(())
}
