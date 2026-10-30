//! M0 do runtime MoE em Rust: cópia de experts RAM→VRAM e sobreposição com cálculo, na RTX 4090.
//!
//! Perguntas:
//! 1. Quanto custa trazer um expert inteiro (gate + up + down em BF16) da RAM, com origem paginável (`Vec`) e pinned?
//! 2. A cópia roda ao mesmo tempo que os GEMVs de experts residentes, em outra stream, e quanto um atrapalha o outro?
//!
//! A memória pinned vem do `PinnedHostBuffer` da API SIMT do cuda-core. O `CudaContext` dela e o `Device` do cutile retêm o
//! mesmo contexto primário da GPU, então o buffer serve às cópias nas streams do cutile. O `PinnedHostBuffer` não aceita
//! `bf16` (não implementa `DeviceCopy`), por isso o host guarda os bits como `u16`.
//!
//! Tempo por eventos da GPU. Na sobreposição, cada cópia de expert é enfileirada intercalada com `KPE` GEMVs, como num
//! decode que traz o próximo expert enquanto calcula os residentes.
use cuda_core::{CudaContext, Event, PinnedHostBuffer};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Reshape;
use m0_gemv::{host_values, kernels};
use std::time::Instant;

const L2_BYTES: usize = 72 << 20;
/// Experts distintos na origem (RAM), para a cópia não reler sempre o mesmo buffer.
const SRC_EXPERTS: usize = 16;
/// Slots de destino na VRAM (nesta medição nenhum kernel os lê, então podem ser sobrescritos em rodízio).
const DST_SLOTS: usize = 8;
const COPIES: usize = 64;

struct ExpertShape {
    name: &'static str,
    /// Linhas e colunas de gate/up (a down é a transposta); o expert tem 3·M·K elementos.
    m: usize,
    k: usize,
    /// Configuração do GEMV medida no item anterior para a matriz gate/up.
    bm: usize,
    bk: usize,
}

const SHAPES: &[ExpertShape] = &[
    ExpertShape { name: "Qwen3-30B", m: 768, k: 2048, bm: 4, bk: 1024 },
    ExpertShape { name: "OLMoE", m: 1024, k: 2048, bm: 2, bk: 1024 },
];

enum Source<'a> {
    Pageable(&'a [Vec<u16>]),
    Pinned(&'a [PinnedHostBuffer<u16>]),
}

impl Source<'_> {
    fn ptr(&self, i: usize) -> *const u16 {
        match self {
            Source::Pageable(v) => v[i % v.len()].as_ptr(),
            Source::Pinned(v) => v[i % v.len()].as_ptr(),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Source::Pageable(_) => "paginável",
            Source::Pinned(_) => "pinned",
        }
    }
}

fn gpu_ms(a: &Event, b: &Event) -> f64 {
    a.elapsed_time(b).unwrap() as f64
}

fn main() -> Result<(), Error> {
    let device = Device::new(0)?;
    let ctx = CudaContext::new(0).expect("contexto SIMT");
    let s_compute = device.new_stream()?;
    let s_copy = device.new_stream()?;
    let ev: Vec<Event> = (0..4).map(|_| device.new_event().unwrap()).collect();

    for (si, sh) in SHAPES.iter().enumerate() {
        let (m, k) = (sh.m, sh.k);
        let n = 3 * m * k;
        let expert_bytes = (n * 2) as f64;
        println!("\n== {} (expert {:.1} MB; GEMV {}x{} BM={} BK={}) ==", sh.name, expert_bytes / 1e6, m, k, sh.bm, sh.bk);

        // Origem: os mesmos bits em memória paginável e pinned.
        let pageable: Vec<Vec<u16>> = (0..SRC_EXPERTS)
            .map(|e| host_values(n, 0xa5a5 + (si * 100 + e) as u64).iter().map(|v| v.to_bits()).collect())
            .collect();
        let pinned: Vec<PinnedHostBuffer<u16>> =
            pageable.iter().map(|v| PinnedHostBuffer::from_slice(&ctx, v).expect("pinned")).collect();
        let slots: Vec<Arc<Tensor<bf16>>> = (0..DST_SLOTS)
            .map(|_| Ok(Arc::new(api::zeros::<bf16>(&[n]).sync_on(&s_copy)?)))
            .collect::<Result<_, Error>>()?;
        let dst: Vec<_> = slots.iter().map(|t| t.device_pointer().cu_deviceptr()).collect();
        let copy = |src: &Source, i: usize| unsafe {
            // SAFETY: cada slot tem `n` elementos de 2 bytes; a origem `i % SRC_EXPERTS` tem `n` u16 e vive até o fim de
            // `main`, depois de toda sincronização.
            cuda_core::memcpy_htod_async(dst[i % DST_SLOTS], src.ptr(i), n, &s_copy).unwrap();
        };

        // Conferência: o slot recebe exatamente os bits da origem (pinned).
        copy(&Source::Pinned(&pinned), 0);
        let back = api::copy_device_to_host_vec(&slots[0]).sync_on(&s_copy)?;
        assert!(back.iter().map(|v| v.to_bits()).eq(pageable[0].iter().copied()), "slot difere da origem");

        // 1. Cópia sozinha.
        let mut copy_alone = [0f64; 2];
        for (j, src) in [Source::Pageable(&pageable), Source::Pinned(&pinned)].iter().enumerate() {
            for i in 0..4 {
                copy(src, i);
            }
            unsafe { s_copy.synchronize() }.unwrap();
            ev[0].record(&s_copy).unwrap();
            let t = Instant::now();
            for i in 0..COPIES {
                copy(src, i);
            }
            let cpu = t.elapsed().as_secs_f64() * 1e6 / COPIES as f64;
            ev[1].record(&s_copy).unwrap();
            ev[1].synchronize().unwrap();
            let us = gpu_ms(&ev[0], &ev[1]) * 1e3 / COPIES as f64;
            copy_alone[j] = us;
            println!(
                "cópia {}: {:.0} µs/expert ({:.1} GB/s); CPU presa na chamada {:.0} µs/expert",
                src.name(),
                us,
                expert_bytes / us / 1e3,
                cpu
            );
        }

        // 2. GEMV sozinho, com pesos residentes que somam 4× o L2.
        let w_host = Arc::new(host_values(m * k, 0x77 + si as u64));
        let w_copies = (4 * L2_BYTES).div_ceil(m * k * 2);
        let ws: Vec<Tensor<bf16>> = (0..w_copies)
            .map(|_| api::copy_host_vec_to_device(&w_host).sync_on(&s_compute)?.reshape(&[m, k]))
            .collect::<Result<_, Error>>()?;
        let x: Tensor<bf16> = api::copy_host_vec_to_device(&Arc::new(host_values(k, 0x99))).sync_on(&s_compute)?;
        let generics = vec![sh.bm.to_string(), sh.bk.to_string(), k.to_string()];
        let mut y = Some(api::zeros::<f32>(&[m]).sync_on(&s_compute)?.partition([sh.bm]));
        let mut gemv = |i: usize| {
            // SAFETY: M é múltiplo de BM e K de BK, então toda fatia carregada está dentro de W e de x.
            let (out, _, _) = unsafe {
                kernels::gemv(y.take().unwrap(), &ws[i % w_copies], &x).generics(generics.clone()).async_on(&s_compute)
            }
            .unwrap();
            y = Some(out);
        };
        for i in 0..200 {
            gemv(i);
        }
        unsafe { s_compute.synchronize() }.unwrap();
        ev[0].record(&s_compute).unwrap();
        for i in 0..2000 {
            gemv(i);
        }
        ev[1].record(&s_compute).unwrap();
        ev[1].synchronize().unwrap();
        let gemv_us = gpu_ms(&ev[0], &ev[1]) * 1e3 / 2000.0;
        // GEMVs por cópia para os dois lados durarem o mesmo (cópia pinned como referência).
        let kpe = (copy_alone[1] / gemv_us).round() as usize;
        println!("GEMV sozinho: {gemv_us:.2} µs; {kpe} GEMVs por cópia de expert equilibram os dois lados");

        // 3. Sobreposição: cópia na stream de cópia, GEMVs na de cálculo, enfileirados intercalados.
        for (j, src) in [Source::Pageable(&pageable), Source::Pinned(&pinned)].into_iter().enumerate() {
            unsafe { s_compute.synchronize() }.unwrap();
            unsafe { s_copy.synchronize() }.unwrap();
            ev[0].record(&s_compute).unwrap();
            ev[2].record(&s_copy).unwrap();
            for e in 0..COPIES {
                copy(&src, e);
                for j in 0..kpe {
                    gemv(e * kpe + j);
                }
            }
            ev[1].record(&s_compute).unwrap();
            ev[3].record(&s_copy).unwrap();
            ev[1].synchronize().unwrap();
            ev[3].synchronize().unwrap();
            let t_c = gpu_ms(&ev[0], &ev[1]);
            let t_x = gpu_ms(&ev[2], &ev[3]);
            // Os dois pares começam quase juntos; o total vai do primeiro início ao último fim.
            let t0 = gpu_ms(&ev[0], &ev[2]).min(0.0);
            let total = gpu_ms(&ev[0], &ev[1]).max(gpu_ms(&ev[0], &ev[3])) - t0;
            let alone_c = gemv_us * (COPIES * kpe) as f64 / 1e3;
            let alone_x = copy_alone[j] * COPIES as f64 / 1e3;
            let serial = alone_c + alone_x;
            println!(
                "juntos, origem {}: total {:.2} ms (em série seriam {:.2} ms; o maior sozinho {:.2} ms) → sobreposição {:.0}%; \
                 GEMV {:+.0}% e cópia {:+.0}% mais lentos que sozinhos",
                src.name(),
                total,
                serial,
                alone_c.max(alone_x),
                (serial - total) / alone_c.min(alone_x) * 100.0,
                (t_c / alone_c - 1.0) * 100.0,
                (t_x / alone_x - 1.0) * 100.0,
            );
        }
    }
    Ok(())
}
