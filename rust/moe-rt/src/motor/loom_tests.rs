//! Modelos do `loom` para as ordens de memória do protocolo D1 (`--cfg motor_loom`; plano, seção A1.1). Sem GPU.
//!
//! Dois modelos pequenos, de duas threads (a linha de serviço do host e a "GPU"), que chamam as mesmas peças de produção
//! de `line.rs`: `seq_reached` (a leitura de `seq` com `Acquire`), `write_answer` (a saída da CPU e a flag com `Release`) e
//! `release_layer` (a regra de soltura da cópia guardada). O `Engine` inteiro com o backend falso fica com o `proptest`
//! (`tests.rs`): em três threads com espera ativa o `loom` estoura o limite de ramos (achado P2-2 da rodada
//! `lastro-rust-1`).
//!
//! - Doorbell: a GPU escreve ids e `x` (`Relaxed`) e depois `seq` (`Release`); o host espera `seq`, lê ids e `x`, escreve
//!   `y` e a flag; a GPU espera a flag e lê `y`. `SeqRelaxed` quebra a primeira metade e `FlagBeforeY` a segunda.
//! - Encerramento: a principal percorre a lista `close_order` de produção; a linha de serviço enfileira uma cópia e drena. A
//!   ordem antiga e a saída sem drenagem liberariam a memória com cópia pendente.
//! - Cópia guardada: o slot é um `loom::cell::UnsafeCell`. A GPU lê o slot no lançamento da camada da vítima e só depois
//!   grava `seq` da camada seguinte; o host escreve o slot ao soltar a cópia. `HeldOneLayerEarly` solta cedo: o `loom`
//!   acusa o acesso concorrente.
//!
//! Um mutante só conta como acusado se o panic tiver a marca `MOTOR_VIOLACAO` (asserção deste arquivo) ou for a violação de
//! causalidade do próprio `loom`; o estouro de ramos e qualquer outro panic reprovam o teste.
//!
//! Rodar: `RUSTFLAGS="--cfg motor_loom" CARGO_TARGET_DIR=target/loom taskset -c 16-27 cargo test --lib motor`.
//! `LOOM_MAX_PREEMPTIONS` (padrão 3) é impresso com cada resultado.
#![allow(unsafe_code)] // só o modelo: o `UnsafeCell` do `loom` para o slot do banco

use super::backend::LayerIo;
use super::engine::{close_order, signal_done, CloseStep};
use super::line::{exit_service, ExitSkip, held_due, read_layer, read_step, release_flags, release_layer, write_answer};
use super::sync::{thread, Arc, AtomicBool, AtomicU32, Ordering};
use loom::cell::UnsafeCell;
use loom::model::Builder;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// A violação do protocolo vista pelo modelo.
fn viola(ok: bool, o_que: &str) {
    if !ok {
        panic!("MOTOR_VIOLACAO: {o_que}");
    }
}

fn builder() -> Builder {
    let mut b = Builder::new();
    b.preemption_bound = Some(std::env::var("LOOM_MAX_PREEMPTIONS").ok().and_then(|s| s.parse().ok()).unwrap_or(3));
    b
}

fn espera(f: impl Fn() -> bool) {
    while !f() {
        thread::yield_now();
    }
}

/// O modelo do doorbell de uma camada: `seq_relaxed` e `flag_first` são os mutantes.
fn doorbell(seq_relaxed: bool, flag_first: bool) {
    const T: u32 = 7;
    let seq = Arc::new(AtomicU32::new(0));
    let flag = Arc::new(AtomicU32::new(0));
    let io = Arc::new(LayerIo::new(1, 2));

    let gpu = {
        let (seq, flag, io) = (seq.clone(), flag.clone(), io.clone());
        thread::spawn(move || {
            io.ids[0].store(3, Ordering::Relaxed);
            io.x[0].store(10, Ordering::Relaxed);
            io.x[1].store(20, Ordering::Relaxed);
            seq.store(T, Ordering::Release);
            espera(|| super::backend::geq(flag.load(Ordering::Acquire), T));
            LayerIo::load(&io.y)
        })
    };

    let (ids, x) = loop {
        if let Some(r) = read_layer(&seq, &io, T, seq_relaxed) {
            break r;
        }
        thread::yield_now();
    };
    viola(ids == [3] && x == [10, 20], &format!("o host leu ids/x velhos: {ids:?} {x:?}"));
    let y: Vec<u32> = x.iter().map(|v| v + ids[0]).collect();
    write_answer(&io.y, &flag, T, &y, flag_first);

    let visto = gpu.join().expect("thread da GPU");
    viola(visto == [13, 23], &format!("a GPU leu y velho: {visto:?}"));
}

/// O doorbell de um passo de `M = 2` posições com a palavra `m` no fim da área: a GPU escreve ids e `x` das `m` posições, depois a palavra
/// `m` (`Relaxed`), depois `seq` (`Release`); o host lê pelo helper real `read_step` (a palavra `m` só depois do `Acquire`), escreve o `y`
/// de todas as posições e a flag; a GPU lê todo o `y`. `seq_relaxed` e `flag_first` são os mutantes de sempre; `m_depois` grava a palavra
/// `m` depois de `seq` (o host pode ver a palavra velha ou nenhuma).
fn doorbell_passo(seq_relaxed: bool, flag_first: bool, m_depois: bool) {
    const T: u32 = 7;
    const M: u32 = 2;
    let seq = Arc::new(AtomicU32::new(0));
    let flag = Arc::new(AtomicU32::new(0));
    let io = Arc::new(LayerIo::with_positions(1, 2, 2));

    // A palavra do passo anterior (m = 1) já está na área: com a gravação atrasada o host a leria velha, dentro da faixa.
    io.m[0].store(1, Ordering::Relaxed);
    let gpu = {
        let (seq, flag, io) = (seq.clone(), flag.clone(), io.clone());
        thread::spawn(move || {
            for (p, id) in [3u32, 4].into_iter().enumerate() {
                io.ids[p].store(id, Ordering::Relaxed);
                io.x[2 * p].store(10 * (p as u32 + 1), Ordering::Relaxed);
                io.x[2 * p + 1].store(10 * (p as u32 + 1) + 1, Ordering::Relaxed);
            }
            if !m_depois {
                io.m[0].store(M, Ordering::Relaxed);
            }
            seq.store(T, Ordering::Release);
            if m_depois {
                io.m[0].store(M, Ordering::Relaxed);
            }
            espera(|| super::backend::geq(flag.load(Ordering::Acquire), T));
            LayerIo::load(&io.y)
        })
    };

    let r = loop {
        if let Some(r) = read_step(&seq, &io, T, seq_relaxed, 1, 2, 2) {
            break r;
        }
        thread::yield_now();
    };
    let r = match r {
        Ok(r) => r,
        Err(m) => panic!("MOTOR_VIOLACAO: o host leu a palavra m velha ou fora da faixa: {m}"),
    };
    viola(r.m == 2 && r.ids == [3, 4] && r.x == [10, 11, 20, 21], &format!("o host leu dado velho: m={} {:?} {:?}", r.m, r.ids, r.x));
    let y: Vec<u32> = (0..r.m).flat_map(|p| (0..2).map(move |j| (p, j))).map(|(p, j)| r.x[p * 2 + j] + r.ids[p]).collect();
    write_answer(&io.y, &flag, T, &y, flag_first);

    let visto = gpu.join().expect("thread da GPU");
    viola(visto == [13, 14, 24, 25], &format!("a GPU leu y velho: {visto:?}"));
}

/// O modelo da cópia guardada: duas camadas, vítima na camada 1 (`after = 1`), lida pelo lançamento dela.
fn copia_guardada(early: bool) {
    const T: u32 = 5;
    let seq: Arc<[AtomicU32; 3]> = Arc::new([AtomicU32::new(0), AtomicU32::new(0), AtomicU32::new(0)]);
    let slot = Arc::new(UnsafeCell::new(1u32));

    let gpu = {
        let (seq, slot) = (seq.clone(), slot.clone());
        thread::spawn(move || {
            for l in 0..2 {
                seq[l].store(T, Ordering::Release); // o roteador da camada l
                if l == 1 {
                    // o lançamento da camada 1 lê o slot da vítima
                    let v = slot.with(|p| unsafe { *p });
                    viola(v == 1, "o lançamento leu o slot depois da cópia");
                }
            }
            seq[2].store(T, Ordering::Release); // depois do último lançamento
        })
    };

    let mut guardada = Some(1usize); // `after` da cópia guardada
    for l in 0..2 {
        espera(|| super::line::seq_reached(&seq[l], T, false));
        if let (Some(k), Some(after)) = (release_layer(l, early), guardada) {
            if held_due(after, k) {
                slot.with_mut(|p| unsafe { *p = 9 });
                guardada = None;
            }
        }
    }
    espera(|| super::line::seq_reached(&seq[2], T, false)); // a virada solta o que sobrou
    if guardada.take().is_some() {
        slot.with_mut(|p| unsafe { *p = 9 });
    }
    gpu.join().expect("thread da GPU");
}

/// Espera limitada: no `loom` uma espera sem fim é um ramo sem fim. Falhar a espera é uma violação do protocolo (uma
/// espera que ninguém solta), não um estouro de ramos. Só vale com no máximo uma thread esperando em giro de cada vez:
/// com duas, elas cedem a vez uma à outra até o limite acabar e acusam falsamente (rodada 4, rev-2). Por isso o cenário
/// `sozinha` deixa a principal sem giro.
fn espera_ate(f: impl Fn() -> bool) -> bool {
    for _ in 0..8 {
        if f() {
            return true;
        }
        thread::yield_now();
    }
    f()
}

/// Os mutantes e cenários do fechamento.
#[derive(Clone, Copy)]
struct Fecha {
    /// A ordem antiga do `Drop` (`close_order(true)`).
    old: bool,
    /// A linha de serviço drena a própria stream de cópia na saída.
    drain: bool,
    /// A linha de serviço ainda enfileira uma cópia antes de sair.
    enqueues: bool,
    /// A linha de serviço sai por falha (a guarda solta as flags, envenena e cancela).
    falha: bool,
    /// O fechamento da principal solta as flags.
    solta: bool,
    /// A linha de serviço sinaliza o término (`signal_done`).
    sinaliza: bool,
    /// A falha do serviço sem ajuda do fechamento: a principal não cancela, não solta flags e não espera em giro (só
    /// junta as threads); quem solta a GPU, envenena e cancela é `exit_service`.
    sozinha: bool,
    /// A ação do ramo de falha que `exit_service` omite (os mutantes de `sozinha`).
    skip: ExitSkip,
}

const LIMPO: Fecha = Fecha { old: false, drain: true, enqueues: true, falha: false, solta: true, sinaliza: true, sozinha: false,
                           skip: ExitSkip::None };

/// O fechamento das duas linhas do host contra uma GPU que espera a flag: a principal cancela, solta as flags
/// (`release_flags`) e percorre `close_order`; a linha de serviço enfileira uma cópia (ou sai cedo) e sai por
/// `exit_service` (a composição da guarda de produção) seguida de `signal_done`. A GPU só termina quando a flag chega ao
/// valor de soltura. Invariantes: nenhuma espera presa, o serviço terminou, e a memória só é liberada sem cópia pendente.
fn encerramento(c: Fecha) {
    const T: u32 = 5;
    let flag = Arc::new(AtomicU32::new(0));
    let cur_t = Arc::new(AtomicU32::new(T));
    let (cancel, poison) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
    let pending = Arc::new(AtomicU32::new(0)); // cópias na stream de cópia que ainda não foram sincronizadas
    let finished = Arc::new(AtomicBool::new(false));
    let gpu_done = Arc::new(AtomicBool::new(false));

    let gpu = {
        let (flag, gpu_done) = (flag.clone(), gpu_done.clone());
        thread::spawn(move || {
            let solta = espera_ate(|| super::backend::geq(flag.load(Ordering::Acquire), T));
            gpu_done.store(true, Ordering::Release);
            solta
        })
    };
    let svc = {
        let (flag, cur_t, cancel, poison, pending, finished) =
            (flag.clone(), cur_t.clone(), cancel.clone(), poison.clone(), pending.clone(), finished.clone());
        thread::spawn(move || {
            if c.enqueues {
                pending.fetch_add(1, Ordering::AcqRel); // `copy_after_reader` ainda em curso
            }
            exit_service(c.falha, cur_t.load(Ordering::Acquire), &[&*flag], &poison, &cancel, c.drain, c.skip, || {
                pending.swap(0, Ordering::AcqRel); // `sync(copy)`
            });
            if c.sinaliza {
                signal_done(&finished);
            }
        })
    };

    if c.sozinha {
        // A principal enxerga a falha assim que a GPU termina, antes de juntar o serviço (é o que `finish_token` faz).
        viola(gpu.join().expect("thread da GPU"), "a GPU ficou presa numa flag que ninguém soltou");
        viola(poison.load(Ordering::Acquire), "a GPU terminou o token antes de a falha ser publicada");
        viola(cancel.load(Ordering::Acquire), "a GPU terminou o token antes do cancelamento ser publicado");
        svc.join().expect("thread de serviço");
        viola(finished.load(Ordering::Acquire), "a linha de serviço não sinalizou o término");
        viola(poison.load(Ordering::Acquire), "a falha do serviço não envenenou");
        viola(cancel.load(Ordering::Acquire), "a falha do serviço não cancelou");
        viola(pending.load(Ordering::Acquire) == 0, "a memória seria liberada com cópia pendente na stream");
        return;
    }
    cancel.store(true, Ordering::Release);
    if c.solta {
        release_flags(T, &[&*flag]);
    }
    for step in close_order(c.old) {
        match step {
            CloseStep::SyncCompute => viola(espera_ate(|| gpu_done.load(Ordering::Acquire)), "a stream de cálculo ficou presa"),
            CloseStep::SyncCopy => {
                pending.swap(0, Ordering::AcqRel);
            }
            CloseStep::Join => viola(espera_ate(|| finished.load(Ordering::Acquire)), "a linha de serviço não terminou"),
        }
    }
    viola(pending.load(Ordering::Acquire) == 0, "a memória seria liberada com cópia pendente na stream");
    viola(gpu.join().expect("thread da GPU"), "a GPU ficou presa numa flag que ninguém soltou");
    svc.join().expect("thread de serviço");
}

fn limpa(nome: &str, f: fn()) {
    let t0 = std::time::Instant::now();
    let b = builder();
    eprintln!("loom {nome}: preemption_bound={:?}", b.preemption_bound);
    b.check(f);
    eprintln!("loom {nome}: limpo em {:?}", t0.elapsed());
}

/// O mutante precisa ser acusado por uma violação do protocolo, não por estouro de ramos nem por outro panic.
fn acusado(nome: &str, f: fn()) {
    let t0 = std::time::Instant::now();
    let b = builder();
    let r = catch_unwind(AssertUnwindSafe(move || b.check(f)));
    let payload = match &r {
        Ok(()) => "nenhum panic".to_string(),
        Err(e) => e
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "payload desconhecido".into()),
    };
    eprintln!("loom {nome}: {payload} ({:?})", t0.elapsed());
    assert!(
        payload.contains("MOTOR_VIOLACAO") || payload.contains("Causality violation"),
        "o mutante {nome} não foi acusado por uma violação do protocolo: {payload}"
    );
}

#[test]
fn loom_doorbell_limpo() {
    limpa("doorbell", || doorbell(false, false));
}

#[test]
fn loom_copia_guardada_limpa() {
    limpa("copia_guardada", || copia_guardada(false));
}

#[test]
fn loom_mutante_seq_relaxed() {
    acusado("SeqRelaxed", || doorbell(true, false));
}

#[test]
fn loom_mutante_flag_antes_do_y() {
    acusado("FlagBeforeY", || doorbell(false, true));
}

#[test]
fn loom_mutante_copia_uma_camada_cedo() {
    acusado("HeldOneLayerEarly", || copia_guardada(true));
}

#[test]
fn loom_encerramento_limpo() {
    limpa("encerramento", || encerramento(LIMPO));
}

#[test]
fn loom_encerramento_com_servico_que_sai_cedo() {
    limpa("encerramento_saida_cedo", || encerramento(Fecha { enqueues: false, ..LIMPO }));
}

#[test]
fn loom_encerramento_com_falha_do_servico() {
    limpa("encerramento_falha", || encerramento(Fecha { falha: true, ..LIMPO }));
}

/// A ordem antiga sozinha, com a drenagem da linha de serviço, sai limpa no modelo: quem acusa o mutante `OldDropOrder`
/// é a falta de drenagem que vem junto com ele em produção (`drain: !OldDropOrder`), não a posição do `Join`.
#[test]
fn loom_ordem_antiga_com_drenagem_sai_limpa() {
    limpa("ordem_antiga_com_drenagem", || encerramento(Fecha { old: true, ..LIMPO }));
}

#[test]
fn loom_mutante_ordem_antiga_do_drop() {
    acusado("OldDropOrder", || encerramento(Fecha { old: true, drain: false, ..LIMPO }));
}

#[test]
fn loom_mutante_servico_sem_drenagem() {
    acusado("ServicoSemDrenagem", || encerramento(Fecha { drain: false, ..LIMPO }));
}

#[test]
fn loom_mutante_fechamento_sem_soltar_as_flags() {
    acusado("SemSoltura", || encerramento(Fecha { solta: false, ..LIMPO }));
}

#[test]
fn loom_mutante_servico_sem_sinal_de_termino() {
    acusado("SemSinalDeTermino", || encerramento(Fecha { sinaliza: false, ..LIMPO }));
}

#[test]
fn loom_falha_do_servico_sozinha_solta_envenena_e_cancela() {
    limpa("falha_sozinha", || encerramento(Fecha { falha: true, sozinha: true, ..LIMPO }));
}

#[test]
fn loom_mutante_falha_sem_soltar_as_flags() {
    acusado("FalhaSemSoltura", || encerramento(Fecha { falha: true, sozinha: true, skip: ExitSkip::Release, ..LIMPO }));
}

#[test]
fn loom_mutante_falha_sem_envenenar() {
    acusado("FalhaSemPoison", || encerramento(Fecha { falha: true, sozinha: true, skip: ExitSkip::Poison, ..LIMPO }));
}

#[test]
fn loom_mutante_falha_sem_cancelar() {
    acusado("FalhaSemCancel", || encerramento(Fecha { falha: true, sozinha: true, skip: ExitSkip::Cancel, ..LIMPO }));
}

#[test]
fn loom_mutante_falha_publica_o_estado_depois_da_soltura() {
    acusado("FalhaEstadoDepois", || encerramento(Fecha { falha: true, sozinha: true, skip: ExitSkip::StateLast, ..LIMPO }));
}

#[test]
fn loom_doorbell_de_passo_limpo() {
    limpa("doorbell_passo", || doorbell_passo(false, false, false));
}

#[test]
fn loom_passo_mutante_seq_relaxed() {
    acusado("passo SeqRelaxed", || doorbell_passo(true, false, false));
}

#[test]
fn loom_passo_mutante_flag_antes_do_y() {
    acusado("passo FlagBeforeY", || doorbell_passo(false, true, false));
}

#[test]
fn loom_passo_mutante_m_depois_do_seq() {
    acusado("passo m depois do seq", || doorbell_passo(false, false, true));
}
