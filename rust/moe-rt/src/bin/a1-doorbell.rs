//! A1.0, o micro-teste do doorbell (plano, seção A0b, passo 1; critério em `qual-runs/A1-DOORBELL-20261031/criterio.txt`).
//! Antes de qualquer outra peça do A1, confere na 4090 o mecanismo da D1: a GPU avisa o host por `cuStreamWriteValue32`
//! numa palavra de memória host mapeada, espera a resposta com `cuStreamWaitValue32` (GEQ cíclico) e lê o `y` que o host
//! escreveu antes da flag. Usa a API do driver direto (`cuda_core::sys`), sem cutile, com dois kernels PTX escritos à mão
//! (um `ld.global` com o cache padrão e um `ld.global.nc`) e uma cópia pelo motor de cópia como os três leitores do `y`.
//!
//! A linha principal enfileira na stream; a de serviço só toca memória host, por atômicos e escritas voláteis, como na D1.
//! Rodar com `taskset -c 0-15` (P-cores).
//!
//!   a1-doorbell
use cuda_core::sys;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

/// Palavras do `y`: a largura do 3B.
const W: usize = 1536;
const BLOCK: u32 = 256;

const PTX: &str = r#"
.version 8.0
.target sm_89
.address_size 64

.visible .entry copy_ca(.param .u64 p_src, .param .u64 p_dst, .param .u32 p_n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd2, [p_dst];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra FIM;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.u32 %r6, [%rd4];
    st.global.u32 [%rd5], %r6;
FIM:
    ret;
}

.visible .entry copy_nc(.param .u64 p_src, .param .u64 p_dst, .param .u32 p_n)
{
    .reg .pred %p<2>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [p_src];
    ld.param.u64 %rd2, [p_dst];
    ld.param.u32 %r1, [p_n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra FIM;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.nc.u32 %r6, [%rd4];
    st.global.u32 [%rd5], %r6;
FIM:
    ret;
}
"#;

macro_rules! ck {
    ($e:expr) => {{
        let code = unsafe { $e };
        assert_eq!(code, sys::cudaError_enum_CUDA_SUCCESS, "{} devolveu {}", stringify!($e), code);
    }};
}

fn padrao(i: u32, j: usize) -> u32 {
    i.wrapping_mul(2_654_435_761) ^ (j as u32).wrapping_mul(0x9E37_79B9) ^ 0x5BD1_E995
}

/// `a` passou de `b` na comparação cíclica do `GEQ` (`(int32_t)(a − b) >= 0`, `cuda.h:535`).
fn geq(a: u32, b: u32) -> bool {
    a.wrapping_sub(b) as i32 >= 0
}

/// Espera `feito()` até `prazo`, pelo relógio `agora` (o tempo desde a gravação da flag). O relógio é lido depois de ver
/// a conclusão, então o instante devolvido é um limite superior dela; só conta se vier antes do prazo. Uma conclusão vista
/// depois do prazo, ou nunca, devolve `None` (rodada 54, achado 1 do rev-1).
fn espera_no_prazo(feito: impl Fn() -> bool, agora: impl Fn() -> Duration, prazo: Duration) -> Option<Duration> {
    loop {
        let ok = feito();
        let t = agora();
        if ok {
            return (t < prazo).then_some(t);
        }
        if t >= prazo {
            return None;
        }
        std::hint::spin_loop();
    }
}

fn spin_us(us: u64) {
    let t = Instant::now();
    while t.elapsed() < Duration::from_micros(us) {
        std::hint::spin_loop();
    }
}

/// Um bloco de memória host pinned e mapeada: o ponteiro do host e o do device.
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
    fn word(&self) -> &AtomicU32 {
        unsafe { AtomicU32::from_ptr(self.host) }
    }
    fn get(&self, j: usize) -> u32 {
        unsafe { self.host.add(j).read_volatile() }
    }
    fn set(&self, j: usize, v: u32) {
        unsafe { self.host.add(j).write_volatile(v) }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Modo {
    /// O desenho: a GPU espera a flag, e o host escreve o `y` antes dela.
    Certo,
    /// N1: a GPU não espera a flag.
    SemEspera,
    /// N2: o host grava a flag antes de escrever o `y`.
    FlagAntes,
    /// Latência: só o leitor (a), sem as paradas de 20 µs.
    Latencia,
}

struct Gpu {
    stream: sys::CUstream,
    copy_ca: sys::CUfunction,
    copy_nc: sys::CUfunction,
    seq: Mapped,
    flag: Mapped,
    done: Mapped,
    y: Mapped,
    out: [Mapped; 3],
}

impl Gpu {
    fn launch(&self, f: sys::CUfunction, src: sys::CUdeviceptr, dst: sys::CUdeviceptr) {
        let (mut s, mut d, mut n) = (src, dst, W as u32);
        let mut params = [&mut s as *mut _ as *mut c_void, &mut d as *mut _ as *mut c_void, &mut n as *mut _ as *mut c_void];
        let grid = (W as u32).div_ceil(BLOCK);
        ck!(sys::cuLaunchKernel(f, grid, 1, 1, BLOCK, 1, 1, 0, self.stream, params.as_mut_ptr(), std::ptr::null_mut()));
    }
    fn write(&self, m: Mapped, v: u32) {
        ck!(sys::cuStreamWriteValue32_v2(self.stream, m.dev, v, sys::CUstreamWriteValue_flags_enum_CU_STREAM_WRITE_VALUE_DEFAULT as _));
    }
    fn wait(&self, m: Mapped, v: u32) {
        ck!(sys::cuStreamWaitValue32_v2(self.stream, m.dev, v, sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ as _));
    }
    fn sync(&self) {
        ck!(sys::cuStreamSynchronize(self.stream));
    }
    /// Zera o estado para um segmento que começa em `t0`: as três palavras em `t0 − 1`.
    fn reset(&self, t0: u32) {
        self.sync();
        for m in [self.seq, self.flag, self.done] {
            m.word().store(t0.wrapping_sub(1), Ordering::SeqCst);
        }
    }
}

struct Seg {
    voltas: usize,
    erradas: [usize; 3],
    voltas_erradas: usize,
    /// µs da gravação da flag até ver o done.
    volta_us: Vec<f64>,
}

/// Um segmento de `n` voltas a partir de `t0`, no modo dado. A linha principal enfileira; a de serviço responde e confere.
fn segmento(g: &Gpu, t0: u32, n: usize, modo: Modo) -> Seg {
    g.reset(t0);
    let (seq, flag, done, y, out) = (g.seq, g.flag, g.done, g.y, g.out);
    let servico = std::thread::spawn(move || {
        let leitores = if modo == Modo::Latencia { 1 } else { 3 };
        let mut s = Seg { voltas: n, erradas: [0; 3], voltas_erradas: 0, volta_us: Vec::with_capacity(n) };
        for k in 0..n {
            let i = t0.wrapping_add(k as u32);
            while !geq(seq.word().load(Ordering::Acquire), i) {
                std::hint::spin_loop();
            }
            let parada = match modo {
                Modo::Certo => i % 4 == 0,
                Modo::SemEspera => true,
                Modo::FlagAntes | Modo::Latencia => false,
            };
            if parada {
                spin_us(20);
            }
            if modo == Modo::FlagAntes {
                flag.word().store(i, Ordering::Release);
                spin_us(20);
            }
            for j in 0..W {
                y.set(j, padrao(i, j));
            }
            if modo != Modo::FlagAntes {
                flag.word().store(i, Ordering::Release);
            }
            let t = Instant::now();
            while !geq(done.word().load(Ordering::Acquire), i) {
                std::hint::spin_loop();
            }
            s.volta_us.push(t.elapsed().as_secs_f64() * 1e6);
            let mut errou = false;
            for (r, o) in out.iter().take(leitores).enumerate() {
                if (0..W).any(|j| o.get(j) != padrao(i, j)) {
                    s.erradas[r] += 1;
                    errou = true;
                }
            }
            s.voltas_erradas += errou as usize;
        }
        s
    });
    for k in 0..n {
        let i = t0.wrapping_add(k as u32);
        g.write(seq, i);
        if modo != Modo::SemEspera {
            g.wait(flag, i);
        }
        g.launch(g.copy_ca, y.dev, out[0].dev);
        if modo != Modo::Latencia {
            g.launch(g.copy_nc, y.dev, out[1].dev);
            ck!(sys::cuMemcpyDtoDAsync_v2(out[2].dev, y.dev, W * 4, g.stream));
        }
        g.write(done, i);
    }
    let s = servico.join().expect("a linha de serviço caiu");
    g.sync();
    s
}

fn quantis(v: &mut [f64]) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
    format!("mediana {:6.2}  p90 {:6.2}  p99 {:6.2}  máx {:8.2} µs", q(0.5), q(0.9), q(0.99), v[v.len() - 1])
}

fn main() {
    // O sha do binário em execução e o do fonte no disco (rodada 54, achado 2 do rev-2).
    let fonte = concat!(env!("CARGO_MANIFEST_DIR"), "/src/bin/a1-doorbell.rs");
    for (nome, caminho) in [("binário", std::env::current_exe().unwrap().display().to_string()), ("fonte", fonte.to_string())] {
        let out = std::process::Command::new("sha256sum").arg(&caminho).output().expect("sha256sum");
        println!("sha256 do {nome}: {}", String::from_utf8_lossy(&out.stdout).split_whitespace().next().unwrap_or("?"));
    }
    ck!(sys::cuInit(0));
    let mut dev = 0;
    ck!(sys::cuDeviceGet(&mut dev, 0));
    let mut ctx = std::ptr::null_mut();
    ck!(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev));
    ck!(sys::cuCtxSetCurrent(ctx));

    let attr = |a: sys::CUdevice_attribute| {
        let mut v = 0;
        ck!(sys::cuDeviceGetAttribute(&mut v, a, dev));
        v
    };
    let map_host = attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY);
    let uva = attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING);
    println!("== Atributos do device 0");
    println!("  CAN_MAP_HOST_MEMORY {map_host}, UNIFIED_ADDRESSING {uva}, CAN_USE_STREAM_MEM_OPS_V1 {}, CAN_FLUSH_REMOTE_WRITES {}, HOST_NATIVE_ATOMIC_SUPPORTED {}",
        attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1),
        attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES),
        attr(sys::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED));

    let src = std::ffi::CString::new(PTX).unwrap();
    let mut module = std::ptr::null_mut();
    ck!(sys::cuModuleLoadData(&mut module, src.as_ptr() as *const c_void));
    let func = |name: &str| {
        let name = std::ffi::CString::new(name).unwrap();
        let mut f = std::ptr::null_mut();
        ck!(sys::cuModuleGetFunction(&mut f, module, name.as_ptr()));
        f
    };
    let mut stream = std::ptr::null_mut();
    ck!(sys::cuStreamCreate(&mut stream, sys::CUstream_flags_enum_CU_STREAM_NON_BLOCKING as _));
    let g = Gpu {
        stream,
        copy_ca: func("copy_ca"),
        copy_nc: func("copy_nc"),
        seq: Mapped::new(1),
        flag: Mapped::new(1),
        done: Mapped::new(1),
        y: Mapped::new(W),
        out: [Mapped::new(W), Mapped::new(W), Mapped::new(W)],
    };

    let nomes = ["(a) ld.global", "(b) ld.global.nc", "(c) motor de cópia"];
    let mut certo = map_host == 1 && uva == 1;
    println!("\n== Correção (leitores (a), (b) e (c); nas voltas i % 4 == 0 o host para 20 µs antes de escrever)");
    for (nome, t0, n) in [("aquecimento, de 1", 1u32, 1000), ("de 1", 1u32, 20_000), ("atravessando 2^31", (1u32 << 31) - 2500, 5000), ("atravessando o zero", 0u32.wrapping_sub(2500), 5000)] {
        let s = segmento(&g, t0, n, Modo::Certo);
        println!("  {nome:22} {} voltas: saídas erradas {:?} ({}), voltas com erro {}", s.voltas, s.erradas, nomes.join(", "), s.voltas_erradas);
        certo &= s.voltas_erradas == 0;
    }

    // Soltura: a GPU espera t + 1, o host escreve t + 2^30.
    let t = 1_000_000u32;
    g.reset(t);
    g.wait(g.flag, t + 1);
    g.write(g.done, t + 1);
    std::thread::sleep(Duration::from_millis(1));
    // `reset(t)` deixa o done em t − 1; a espera segura se ele ainda não chegou a t + 1. (Na primeira execução, a conferência
    // comparava com t e dava falso sempre: `resultado-execucao1.txt`.)
    let antes = g.done.word().load(Ordering::Acquire);
    let segurou = !geq(antes, t + 1);
    let inicio = Instant::now();
    g.flag.word().store(t.wrapping_add(1 << 30), Ordering::Release);
    let soltou = espera_no_prazo(|| geq(g.done.word().load(Ordering::Acquire), t + 1), || inicio.elapsed(), Duration::from_secs(1));
    match soltou {
        Some(d) => println!("  soltura com t + 2^30 (t = {t}): done antes da flag {antes}, a espera segurou {segurou}; soltou em {:.1} µs (prazo 1 s)",
            d.as_secs_f64() * 1e6),
        None => {
            // A stream não terminou no prazo: sair com falha, sem o `cuStreamSynchronize`, que esperaria para sempre.
            println!("  soltura com t + 2^30 (t = {t}): done antes da flag {antes}, a espera segurou {segurou}; NÃO soltou em 1 s");
            println!("portão: NÃO PASSA (soltura)");
            std::process::exit(1);
        }
    }
    certo &= segurou;
    g.sync();

    println!("\n== Controles negativos (2.000 voltas cada; o conferidor tem de acusar dado velho)");
    let mut negativos = true;
    for (nome, modo) in [("N1 sem a espera da flag", Modo::SemEspera), ("N2 flag antes do y", Modo::FlagAntes)] {
        let s = segmento(&g, 1, 2000, modo);
        let frac = s.voltas_erradas as f64 / s.voltas as f64;
        println!("  {nome:24} saídas erradas {:?}, voltas com erro {} ({:.1}%)", s.erradas, s.voltas_erradas, frac * 100.0);
        negativos &= frac >= 0.5;
    }

    println!("\n== Latência (20.000 voltas, só o leitor (a))");
    let aq = segmento(&g, 1, 1000, Modo::Latencia);
    let mut db = segmento(&g, 1, 20_000, Modo::Latencia);
    // O leitor (a) também é conferido nestas voltas, e um erro aqui reprova a correção (rodada 54, achado 2 do rev-2).
    println!("  voltas com erro no leitor (a): aquecimento {} de {}, medida {} de {}", aq.voltas_erradas, aq.voltas, db.voltas_erradas, db.voltas);
    certo &= aq.voltas_erradas == 0 && db.voltas_erradas == 0;
    let mut ev = std::ptr::null_mut();
    ck!(sys::cuEventCreate(&mut ev, 0));
    let mut evt = Vec::with_capacity(20_000);
    for k in 0..21_000 {
        let t0 = Instant::now();
        g.launch(g.copy_ca, g.y.dev, g.out[0].dev);
        ck!(sys::cuEventRecord(ev, g.stream));
        ck!(sys::cuEventSynchronize(ev));
        if k >= 1000 {
            evt.push(t0.elapsed().as_secs_f64() * 1e6);
        }
    }
    let med = |v: &mut Vec<f64>| {
        let mut w = v.clone();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap());
        w[w.len() / 2]
    };
    let (m_db, m_ev) = (med(&mut db.volta_us), med(&mut evt));
    println!("  doorbell (flag gravada → done visto)     {}", quantis(&mut db.volta_us));
    println!("  evento (lançar → cuEventSynchronize)     {}", quantis(&mut evt));
    let lat = m_db <= 20.0 && m_db <= 1.5 * m_ev;
    println!("  razão das medianas doorbell/evento: {:.2}", m_db / m_ev);

    println!("\n== Portão do A1.0");
    println!("  1+2 atributos, chamadas e correção: {}", if certo { "passa" } else { "NÃO PASSA" });
    println!("  3 controles negativos acusados:     {}", if negativos { "passa" } else { "NÃO PASSA" });
    println!("  4 latência (≤ 20 µs e ≤ 1,5× evento): {}", if lat { "passa" } else { "NÃO PASSA" });
    println!("portão: {}", if certo && negativos && lat { "PASSA" } else { "NÃO PASSA" });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    /// Relógio falso: cada leitura avança 1 ms; a conclusão aparece na leitura `quando` do `feito` (ou nunca).
    fn caso(quando: Option<u32>) -> Option<Duration> {
        let (lidas, ms) = (Cell::new(0u32), Cell::new(0u64));
        let feito = || {
            lidas.set(lidas.get() + 1);
            quando.is_some_and(|q| lidas.get() >= q)
        };
        let agora = || {
            ms.set(ms.get() + 1);
            Duration::from_millis(ms.get())
        };
        espera_no_prazo(feito, agora, Duration::from_millis(10))
    }

    #[test]
    fn conclusao_antes_do_prazo_conta() {
        assert_eq!(caso(Some(3)), Some(Duration::from_millis(3)));
    }

    #[test]
    fn conclusao_vista_depois_do_prazo_reprova() {
        assert_eq!(caso(Some(10)), None);
        assert_eq!(caso(Some(50)), None);
    }

    #[test]
    fn sem_conclusao_reprova_e_termina() {
        assert_eq!(caso(None), None);
    }

    #[test]
    fn geq_e_ciclico() {
        assert!(geq(0, u32::MAX));
        assert!(geq(1 << 31, (1 << 31) - 1));
        assert!(!geq(u32::MAX, 0));
        assert!(geq(7u32.wrapping_add(1 << 30), 8));
    }
}
