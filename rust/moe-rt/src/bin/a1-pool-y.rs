//! A1.3b: o `y` escrito pelas threads do `Pool` da CPU (critério em `qual-runs/A1-POOL-Y-20261101-b/criterio.txt`).
//! Repete o conferidor do quarto leitor (`a1-cutile-leitor`): o kernel cutile lê o `y` em memória host mapeada depois do
//! `cuStreamWaitValue32` da flag, mas quem escreve o `y` agora é `Cpu::fill_rows` (as threads do `Pool`), e a linha de
//! serviço grava a flag com `Release` só depois do retorno. O N3 grava a flag de dentro de uma tarefa do `Pool`, antes de
//! as outras terminarem.
//!
//!   CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd taskset -c 0-15 a1-pool-y
use cuda_core::{sys, Device};
use moe_rt::cpu::Cpu;
use cutile::prelude::*;
use cutile::tile_kernel::DevicePointer;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Palavras do `y`: a largura do 3B.
const W: usize = 1536;
const BK: usize = 512;

#[cutile::module]
pub mod leitor {
    use cutile::core::*;

    /// `dst[i] = src[i]` para `n` palavras, por vistas sobre ponteiros do device (aqui, memória host mapeada).
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn copy_y<const BK: i32>(src: *mut u32, dst: *mut u32, n: i32) {
        let s: Tensor<u32, { [-1] }> = tensor_from_ptr(src, n);
        let mut d: Tensor<u32, { [-1] }> = tensor_from_ptr(dst, n);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let t: Tile<u32, { [BK] }> = s.partition(const_shape![BK]).load([pid.0]);
        d.partition_mut(const_shape![BK]).store(t, [pid.0]);
    }

    unsafe fn tensor_from_ptr<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }
}

macro_rules! ck {
    ($e:expr) => {{
        let code = unsafe { $e };
        assert_eq!(code, sys::cudaError_enum_CUDA_SUCCESS, "{} devolveu {}", stringify!($e), code);
    }};
}

/// Sempre um `f32` finito (o expoente nunca é todo 1), para o `Pool` escrever como `f32` sem mexer nos bits.
fn padrao(i: u32, j: usize) -> u32 {
    (i.wrapping_mul(2_654_435_761) ^ (j as u32).wrapping_mul(0x9E37_79B9) ^ 0x5BD1_E995) & 0x7F7F_FFFF
}

/// `(int32_t)(a − b) >= 0`, o `GEQ` do `cuStreamWaitValue32` (`cuda.h:535`).
fn geq(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) as i32 >= 0
}

fn spin_us(us: u64) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_micros(us) {
        std::hint::spin_loop();
    }
}

#[derive(Clone, Copy)]
struct Mapped {
    host: *mut u32,
    dev: sys::CUdeviceptr,
}

// SAFETY: a memória vive até o fim do processo; o host só a toca por atômicos e acessos voláteis.
unsafe impl Send for Mapped {}

impl Mapped {
    fn new(words: usize) -> Self {
        let mut p = std::ptr::null_mut();
        ck!(sys::cuMemHostAlloc(&mut p, words * 4, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP));
        let host = p as *mut u32;
        for j in 0..words {
            unsafe { host.add(j).write_volatile(0) };
        }
        let mut dev = 0;
        ck!(sys::cuMemHostGetDevicePointer_v2(&mut dev, p, 0));
        Mapped { host, dev }
    }
    fn ptr(&self) -> *mut u32 {
        self.host
    }
    fn word(&self) -> &AtomicU32 {
        unsafe { AtomicU32::from_ptr(self.host) }
    }
    fn get(&self, j: usize) -> u32 {
        unsafe { self.host.add(j).read_volatile() }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Modo {
    Certo,
    /// N1: a GPU não espera a flag, e o host gira 20 µs antes de escrever o `y` em toda volta.
    SemEspera,
    /// N2: o host grava a flag antes de escrever o `y`.
    FlagAntes,
    /// N3: a flag sai de dentro de uma tarefa do `Pool`, na última linha da faixa 0, antes de as outras faixas terminarem.
    FlagNoPool,
}

struct Gpu {
    stream: std::sync::Arc<cuda_core::Stream>,
    raw: sys::CUstream,
    seq: Mapped,
    flag: Mapped,
    done: Mapped,
    y: Mapped,
    out: Mapped,
}

impl Gpu {
    fn lancar(&self) {
        // SAFETY: as duas áreas têm W palavras e vivem até o fim do processo; nada a segurar depois do lançamento.
        let k = unsafe {
            leitor::copy_y(DevicePointer::<u32>::from_cu_deviceptr(self.y.dev), DevicePointer::<u32>::from_cu_deviceptr(self.out.dev), W as i32)
        };
        // SAFETY: `async_on` não segura recursos; as áreas vivem até o fim do processo e a stream é sincronizada no fim do segmento.
        unsafe { k.generics(vec![BK.to_string()]).grid(((W / BK) as u32, 1, 1)).async_on(&self.stream) }.expect("lançamento do kernel cutile");
    }
    fn write(&self, m: Mapped, v: u32) {
        ck!(sys::cuStreamWriteValue32_v2(self.raw, m.dev, v, sys::CUstreamWriteValue_flags_enum_CU_STREAM_WRITE_VALUE_DEFAULT as _));
    }
    fn wait(&self, m: Mapped, v: u32) {
        ck!(sys::cuStreamWaitValue32_v2(self.raw, m.dev, v, sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ as _));
    }
    fn sync(&self) {
        ck!(sys::cuStreamSynchronize(self.raw));
    }
    fn reset(&self, t0: u32) {
        self.sync();
        for m in [self.seq, self.flag, self.done] {
            m.word().store(t0.wrapping_sub(1), Ordering::SeqCst);
        }
    }
}

/// Um segmento de `n` voltas a partir de `t0`; devolve quantas saídas vieram erradas.
fn segmento(g: &Gpu, cpu: &std::sync::Arc<Cpu>, t0: u32, n: usize, modo: Modo) -> usize {
    g.reset(t0);
    let cpu = cpu.clone();
    let (seq, flag, done, y, out) = (g.seq, g.flag, g.done, g.y, g.out);
    let servico = std::thread::spawn(move || {
        let mut erradas = 0;
        for k in 0..n {
            let i = t0.wrapping_add(k as u32);
            while !geq(seq.word().load(Ordering::Acquire), i) {
                std::hint::spin_loop();
            }
            let parada = match modo {
                Modo::Certo => i % 4 == 0,
                Modo::SemEspera => true,
                Modo::FlagAntes | Modo::FlagNoPool => false,
            };
            if parada {
                spin_us(20);
            }
            if modo == Modo::FlagAntes {
                flag.word().store(i, Ordering::Release);
                spin_us(20);
            }
            // SAFETY: `y.host` aponta W palavras de memória mapeada, `f32` e `u32` têm o mesmo tamanho e alinhamento; o host é o único
            // a tocar nelas até a flag, e a GPU só as lê depois.
            let ys = unsafe { std::slice::from_raw_parts_mut(y.ptr() as *mut f32, W) };
            let flag_addr = flag.ptr() as usize;
            if modo == Modo::FlagNoPool {
                cpu.fill_rows(ys, &|r| {
                    // Segura as faixas diferentes da 0 (20 µs no primeiro uso de cada uma), como o N1 e o N2 forçam a janela.
                    if r >= 16 && r % 16 == 0 {
                        spin_us(20);
                    }
                    if r == 15 {
                        // SAFETY: `flag_addr` é o endereço da palavra `flag`, viva até o fim do processo.
                        unsafe { AtomicU32::from_ptr(flag_addr as *mut u32) }.store(i, Ordering::Release);
                    }
                    f32::from_bits(padrao(i, r))
                });
            } else {
                cpu.fill_rows(ys, &|r| f32::from_bits(padrao(i, r)));
            }
            if modo != Modo::FlagAntes && modo != Modo::FlagNoPool {
                flag.word().store(i, Ordering::Release);
            }
            while !geq(done.word().load(Ordering::Acquire), i) {
                std::hint::spin_loop();
            }
            // No N3 só conta o erro que alcança uma faixa de outra tarefa (linhas >= 16); a linha 15 ainda é escrita depois do callback.
            let de = if modo == Modo::FlagNoPool { 16 } else { 0 };
            if (de..W).any(|j| out.get(j) != padrao(i, j)) {
                erradas += 1;
            }
        }
        erradas
    });
    for k in 0..n {
        let i = t0.wrapping_add(k as u32);
        g.write(seq, i);
        if modo != Modo::SemEspera {
            g.wait(flag, i);
        }
        g.lancar();
        g.write(done, i);
    }
    let erradas = servico.join().expect("a linha de serviço caiu");
    g.sync();
    erradas
}

fn main() {
    // O sha do binário em execução e o do fonte no disco, impressos pelo próprio binário (como no a1-doorbell).
    let fonte = concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/a1-pool-y.rs");
    for (nome, caminho) in [("binário", std::env::current_exe().unwrap().display().to_string()), ("fonte", fonte.to_string())] {
        let out = std::process::Command::new("sha256sum").arg(&caminho).output().expect("sha256sum");
        println!("sha256 do {nome}: {}", String::from_utf8_lossy(&out.stdout).split_whitespace().next().unwrap_or("?"));
    }
    let device = Device::new(0).expect("device 0");
    device.bind_to_thread().expect("contexto");
    let stream = device.new_stream().expect("stream");
    let raw = stream.cu_stream();
    let cpu = std::sync::Arc::new(Cpu::new(8));
    let g = Gpu { stream, raw, seq: Mapped::new(1), flag: Mapped::new(1), done: Mapped::new(1), y: Mapped::new(W), out: Mapped::new(W) };

    println!("== Correção (kernel cutile como leitor, y escrito pelo Pool; nas voltas i % 4 == 0 o host para 20 µs antes de escrever)");
    let mut certo = true;
    // As primeiras voltas compilam o kernel (JIT); também são conferidas e entram na conta.
    let jit = segmento(&g, &cpu, 1, 10, Modo::Certo);
    println!("  voltas de compilação (JIT)   10 voltas: saídas erradas {jit}");
    certo &= jit == 0;
    for (nome, t0, n) in [("aquecimento, de 1", 1u32, 1000), ("de 1", 1u32, 20_000), ("atravessando 2^31", (1u32 << 31) - 2500, 5000), ("atravessando o zero", 0u32.wrapping_sub(2500), 5000)] {
        let e = segmento(&g, &cpu, t0, n, Modo::Certo);
        println!("  {nome:22} {n} voltas: saídas erradas {e}");
        certo &= e == 0;
    }
    println!("\n== Controles negativos (2.000 voltas cada; o conferidor tem de acusar dado velho)");
    let mut negativos = true;
    for (nome, modo, minimo) in [("N1 sem a espera da flag", Modo::SemEspera, 1000), ("N2 flag antes do y", Modo::FlagAntes, 1000), ("N3 flag dentro do Pool", Modo::FlagNoPool, 1000)] {
        let e = segmento(&g, &cpu, 1, 2000, modo);
        println!("  {nome:24} saídas erradas {e} de 2000 ({:.1}%; mínimo {minimo})", e as f64 / 20.0);
        negativos &= e >= minimo;
    }
    println!("\n== Portão do A1.3b (y escrito pelo Pool)");
    println!("  2 correção:                  {}", if certo { "passa" } else { "NÃO PASSA" });
    println!("  3 controles negativos:       {}", if negativos { "passa" } else { "NÃO PASSA" });
    println!("portão: {}", if certo && negativos { "PASSA" } else { "NÃO PASSA" });
    std::process::exit(if certo && negativos { 0 } else { 1 });
}
