//! A1-DIAG10, harness I: intervalos DESCRITIVOS da linha de serviço (criterio.txt v6, seção 2). Sem a feature `diag` tudo aqui é
//! vazio e o compilador o remove: o binário N não tem relógio extra. Com a feature, cada intervalo vai para acumuladores da
//! thread de serviço (uma só escritora) e, ao fim de cada token, uma linha vai para um buffer pré-alocado; nada é impresso no
//! caminho quente. Os intervalos NÃO entram em nenhuma regra de decisão.
//!
//!   entrada_com_seq  o seq já estava gravado na primeira leitura da camada (NÃO indica atraso da GPU)
//!   T_espera         do começo da espera ao seq visto, em toda camada (folga do host)
//!   T_servico        do seq visto até a flag gravada (depois de `answer`), por camada com falta
//!   T_cpu            só a chamada `Cpu::experts` (as GEMV); T_conv é o resto de `CpuServe::compute` (conversões e montagem)
//!   T_livro          do fim de `answer` ao fim de pin_layer, escolha de vítima e copy_after_reader, por camada com falta

#[cfg(feature = "diag")]
pub use on::*;
#[cfg(not(feature = "diag"))]
pub use off::*;

#[cfg(not(feature = "diag"))]
mod off {
    #[derive(Clone, Copy)]
    pub struct Stamp;
    #[inline(always)]
    pub fn now() -> Stamp {
        Stamp
    }
    #[inline(always)]
    pub fn wait_done(_since: Stamp, _first: bool) {}
    #[inline(always)]
    pub fn compute_parts(_t0: Stamp, _t1: Stamp, _t2: Stamp, _t3: Stamp) {}
    #[inline(always)]
    pub fn answered() {}
    #[inline(always)]
    pub fn layer_end(_has_cpu: bool) {}
    #[inline(always)]
    pub fn token_end() {}
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Site {
        Cpu,
        Livro,
        AntesDoWait,
    }
    #[inline(always)]
    pub fn inject(_site: Site) {}
}

#[cfg(feature = "diag")]
mod on {
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::time::Instant;

    pub type Stamp = Instant;

    #[inline(always)]
    pub fn now() -> Stamp {
        Instant::now()
    }

    /// Uma linha por token medido; tempos em ms somados sobre as camadas do token.
    #[derive(Clone, Copy, Default, Debug)]
    pub struct Rec {
        pub layers: u32,
        pub faltas: u32,
        pub entrada_com_seq: u32,
        pub t_espera: f64,
        pub t_servico: f64,
        pub t_cpu: f64,
        pub t_conv: f64,
        pub t_livro: f64,
        /// Espera injetada antes do wait do seq (acumulador próprio; só no teste de fidelidade).
        pub t_inj: f64,
    }

    fn epoch() -> Instant {
        static E: OnceLock<Instant> = OnceLock::new();
        *E.get_or_init(Instant::now)
    }

    fn ns(t: Instant) -> u64 {
        t.saturating_duration_since(epoch()).as_nanos() as u64
    }

    /// Controle de fidelidade (criterio.txt, seção 7): espera ativa injetada por camada com falta, num único ponto por execução.
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Site {
        Cpu,
        Livro,
        AntesDoWait,
    }

    static INJ_SITE: AtomicU64 = AtomicU64::new(u64::MAX);
    static INJ_NS: AtomicU64 = AtomicU64::new(0);

    /// Liga a injeção de `us` microssegundos em `site`; `None` desliga.
    pub fn set_injection(site: Option<(Site, u64)>) {
        match site {
            Some((s, us)) => {
                INJ_NS.store(us * 1000, Ordering::Release);
                INJ_SITE.store(s as u64, Ordering::Release);
            }
            None => INJ_SITE.store(u64::MAX, Ordering::Release),
        }
    }

    /// Espera ativa no ponto, se for o escolhido. Chamado só em camadas com falta.
    #[inline]
    pub fn inject(site: Site) {
        if INJ_SITE.load(Ordering::Relaxed) != site as u64 {
            return;
        }
        let d = INJ_NS.load(Ordering::Relaxed);
        let t0 = Instant::now();
        while (t0.elapsed().as_nanos() as u64) < d {
            std::hint::spin_loop();
        }
        if site == Site::AntesDoWait {
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            ACC.with(|a| a.borrow_mut().tok.t_inj += ms);
        }
    }

    static RESET_NS: AtomicU64 = AtomicU64::new(0);
    static RECS: Mutex<Vec<Rec>> = Mutex::new(Vec::new());

    /// Prepara o buffer (capacidade para `cap` tokens, sem alocar depois) e zera. Chamado pela thread principal depois do aquecimento.
    pub fn reset(cap: usize) {
        epoch();
        let mut r = RECS.lock().unwrap_or_else(|e| e.into_inner());
        r.clear();
        r.reserve(cap);
        RESET_NS.store(ns(Instant::now()), Ordering::Release);
    }

    /// Os registros desde o último `reset`.
    pub fn take() -> Vec<Rec> {
        std::mem::take(&mut *RECS.lock().unwrap_or_else(|e| e.into_inner()))
    }

    #[derive(Default)]
    struct Acc {
        tok: Rec,
        /// Começo da espera da primeira camada do token; um token que começou antes do `reset` é descartado.
        tok_start: Option<Instant>,
        // da camada corrente
        espera: f64,
        seen: Option<Instant>,
        servico: f64,
        cpu: f64,
        conv: f64,
        answered: Option<Instant>,
    }

    thread_local! {
        static ACC: RefCell<Acc> = RefCell::new(Acc::default());
    }

    fn ms(a: Instant, b: Instant) -> f64 {
        b.saturating_duration_since(a).as_secs_f64() * 1e3
    }

    pub fn wait_done(since: Stamp, first: bool) {
        let t = Instant::now();
        ACC.with(|a| {
            let a = &mut *a.borrow_mut();
            if a.tok_start.is_none() {
                a.tok_start = Some(since);
            }
            a.espera = ms(since, t);
            a.seen = Some(t);
            a.tok.entrada_com_seq += first as u32;
        });
    }

    /// Os quatro instantes de `CpuServe::compute`: começo, antes e depois de `Cpu::experts`, e o fim.
    pub fn compute_parts(t0: Stamp, t1: Stamp, t2: Stamp, t3: Stamp) {
        ACC.with(|a| {
            let a = &mut *a.borrow_mut();
            a.cpu = ms(t1, t2);
            a.conv = ms(t0, t1) + ms(t2, t3);
        });
    }

    /// Chamado logo depois de `answer` (a flag já foi gravada).
    pub fn answered() {
        let t = Instant::now();
        ACC.with(|a| {
            let a = &mut *a.borrow_mut();
            if let Some(s) = a.seen {
                a.servico = ms(s, t);
            }
            a.answered = Some(t);
        });
    }

    pub fn layer_end(has_cpu: bool) {
        let t = Instant::now();
        ACC.with(|a| {
            let a = &mut *a.borrow_mut();
            a.tok.layers += 1;
            a.tok.t_espera += a.espera;
            if has_cpu {
                a.tok.faltas += 1;
                a.tok.t_servico += a.servico;
                a.tok.t_cpu += a.cpu;
                a.tok.t_conv += a.conv;
                if let Some(x) = a.answered {
                    a.tok.t_livro += ms(x, t);
                }
            }
            a.espera = 0.0;
            a.servico = 0.0;
            a.cpu = 0.0;
            a.conv = 0.0;
            a.seen = None;
            a.answered = None;
        });
    }

    pub fn token_end() {
        ACC.with(|a| {
            let a = &mut *a.borrow_mut();
            let rec = std::mem::take(&mut a.tok);
            let start = a.tok_start.take();
            if start.is_some_and(|s| ns(s) >= RESET_NS.load(Ordering::Acquire)) {
                RECS.lock().unwrap_or_else(|e| e.into_inner()).push(rec);
            }
        });
    }
}

#[cfg(all(test, feature = "diag"))]
mod testes {
    use super::*;
    use std::time::Duration;

    /// Os registros são globais: os testes que os usam rodam um de cada vez.
    static TRAVA: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn camada(espera_us: u64, cpu_us: u64, com_falta: bool, primeira_leitura: bool) {
        let ini = now();
        std::thread::sleep(Duration::from_micros(espera_us));
        wait_done(ini, primeira_leitura);
        let t0 = now();
        let t1 = now();
        std::thread::sleep(Duration::from_micros(cpu_us));
        let t2 = now();
        compute_parts(t0, t1, t2, now());
        answered();
        layer_end(com_falta);
    }

    #[test]
    fn um_token_vira_uma_linha_com_os_cinco_intervalos() {
        let _g = TRAVA.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(|| {
            reset(4);
            camada(500, 1000, true, false);
            camada(0, 0, false, true);
            token_end();
        })
        .join()
        .unwrap();
        let r = take();
        assert_eq!(r.len(), 1);
        let r = r[0];
        assert_eq!((r.layers, r.faltas, r.entrada_com_seq), (2, 1, 1));
        assert!(r.t_espera >= 0.5 && r.t_espera < 5.0, "{r:?}");
        assert!(r.t_cpu >= 1.0 && r.t_cpu < 5.0, "{r:?}");
        // a camada sem falta não soma servico, cpu, conv nem livro
        assert!(r.t_servico >= r.t_cpu, "{r:?}");
    }

    #[test]
    fn token_que_comecou_antes_do_reset_e_descartado() {
        let _g = TRAVA.lock().unwrap_or_else(|e| e.into_inner());
        std::thread::spawn(|| {
            let ini = now();
            reset(4);
            wait_done(ini, false);
            layer_end(true);
            token_end();
        })
        .join()
        .unwrap();
        assert!(take().is_empty());
    }

    #[test]
    fn a_injecao_sobe_so_o_intervalo_que_a_contem() {
        let _g = TRAVA.lock().unwrap_or_else(|e| e.into_inner());
        set_injection(Some((Site::AntesDoWait, 200)));
        std::thread::spawn(|| {
            reset(4);
            camada(0, 0, true, true);
            inject(Site::AntesDoWait);
            inject(Site::Cpu); // outro ponto: não faz nada
            token_end();
        })
        .join()
        .unwrap();
        set_injection(None);
        let r = take();
        assert!(r[0].t_inj >= 0.2 && r[0].t_inj < 2.0, "{:?}", r[0]);
        assert!(r[0].t_cpu < 0.2, "{:?}", r[0]);
    }
}
