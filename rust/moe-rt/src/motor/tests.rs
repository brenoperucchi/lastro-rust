//! Testes do protocolo D1 no backend falso (plano, A0b, "Backend falso por trait, com teste de propriedade"; registro na
//! seção A1.1). Sem GPU.
//!
//! - As duas linhas do host e as duas filas andam pelo escalonador determinístico do backend falso, na ordem sorteada pelo
//!   `proptest` (semente e pesos por ator), com roteamentos, orçamentos, backlog, modo de admissão e falhas sorteados.
//! - Invariantes: nenhuma violação do conferidor (todo lançamento lê o expert que a tabela do token diz, a tabela só
//!   lista slot pronto, o `y` somado é o daquela camada e daquele token, nada em fila quando a memória cai), toda linha
//!   termina (o escalonador não trava) e as filas acabam vazias.
//! - No modo com barreira, acertos, cópias e experts na CPU batem com `tools/moe_sim.py` (fixtures de
//!   `tools/a1_fixtures_sim.py`).
//! - Os mutantes do backend falso (o idioma do B0): cada um roda nos mesmos casos que a versão certa, e o teste exige que
//!   a certa saia limpa em todos e o mutante seja acusado em pelo menos um, pelo tipo de violação esperado. As contagens
//!   saem no `eprintln!` (rodar com `--nocapture`).
use super::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg, Violation};
use super::line::{SlotState, TableBuilder};
use super::*;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config as PtConfig, RngSeed, TestRunner};
use std::time::{Duration, Instant};

const WIDTH: usize = 2;
const BUDGET: u64 = 300_000;

#[derive(Clone, Debug)]
enum Fault {
    None,
    /// A CPU de mentira falha na `n`-ésima camada calculada (a linha de serviço morre).
    CpuErr(usize),
    /// A CPU de mentira entra em `panic` na `n`-ésima camada calculada (a linha de serviço cai por unwind).
    CpuPanic(usize),
    /// A `n`-ésima operação da stream de cálculo falha (a principal morre no meio da enfileiração).
    MainErr(u64),
    /// O motor cai depois de enfileirar o token `k`, sem esperar por ele (o `Drop` no meio do token).
    DropMidToken(usize),
}

#[derive(Clone, Debug)]
struct Caso {
    layers: usize,
    experts: usize,
    topk: usize,
    slots: usize,
    backlog: usize,
    admit: AdmitMode,
    first_token: u32,
    prefill: Vec<u32>,
    routing: Vec<Vec<Vec<u32>>>,
    sched: SchedCfg,
    fault: Fault,
}

#[derive(Debug)]
struct Saida {
    stats: Option<Stats>,
    violations: Vec<Violation>,
    stalled: bool,
    err: Option<String>,
    queued: (usize, usize),
}

impl Saida {
    fn limpa(&self) -> bool {
        self.violations.is_empty() && !self.stalled && self.queued == (0, 0)
    }

    fn tem(&self, kind: &str) -> bool {
        self.violations.iter().any(|v| v.kind() == kind)
    }
}

fn config(c: &Caso, mutant: Option<Mutant>) -> Config {
    Config {
        layers: c.layers,
        experts: c.experts,
        topk: c.topk,
        width: WIDTH,
        slots: c.slots,
        backlog: c.backlog,
        admit: c.admit,
        first_token: c.first_token,
        prefill: c.prefill.clone(),
        mutant,
    }
}

fn fake_cfg(c: &Caso) -> FakeCfg {
    FakeCfg {
        layers: c.layers,
        experts: c.experts,
        topk: c.topk,
        width: WIDTH,
        routing: c.routing.clone(),
        fail_main_at: match c.fault {
            Fault::MainErr(n) => Some(n),
            _ => None,
        },
        copy_immediate: false,
    }
}

fn fake_cpu(c: &Caso) -> FakeCpu {
    FakeCpu {
        fail_at: if let Fault::CpuErr(n) = c.fault { Some(n) } else { None },
        panic_at: if let Fault::CpuPanic(n) = c.fault { Some(n) } else { None },
        ..Default::default()
    }
}

/// Roda os tokens de `c.routing` e lê os contadores no `Published` do token seguinte.
fn dirige<B: Backend>(eng: &mut Engine<B>, c: &Caso) -> (Option<Stats>, Option<String>) {
    let tokens = c.routing.len();
    for k in 0..=tokens {
        let p = match eng.next_published() {
            Ok(p) => p,
            Err(e) => return (None, Some(format!("{e:?}"))),
        };
        if k == tokens {
            return (Some(p.stats().clone()), None);
        }
        if let Fault::DropMidToken(d) = c.fault {
            if d == k {
                return (None, eng.enqueue_token(p).err().map(|e| format!("{e:?}")));
            }
        }
        if let Err(e) = eng.run_token(p) {
            return (None, Some(format!("{e:?}")));
        }
    }
    unreachable!()
}

fn roda(c: &Caso, mutant: Option<Mutant>) -> Saida {
    let gpu = FakeGpu::new_sched(fake_cfg(c), c.sched);
    let cfg = config(c, mutant);
    let parts = gpu.parts(&cfg);
    let mut eng = Engine::new(gpu.clone(), parts, cfg, fake_cpu(c)).expect("motor");
    let (stats, mut err) = dirige(&mut eng, c);
    let closed = eng.close();
    if err.is_none() {
        err = closed.err().map(|e| format!("{e:?}"));
    }
    Saida { stats, violations: gpu.violations(), stalled: gpu.stalled(), err, queued: gpu.queued() }
}

fn sched() -> impl Strategy<Value = SchedCfg> {
    (any::<u64>(), proptest::array::uniform4(1u32..=8)).prop_map(|(seed, weights)| SchedCfg { seed, weights, budget: BUDGET })
}

fn routing(layers: usize, experts: usize, topk: usize, tokens: usize) -> impl Strategy<Value = Vec<Vec<Vec<u32>>>> {
    let ids: Vec<u32> = (0..experts as u32).collect();
    let camada = proptest::sample::subsequence(ids, topk).prop_shuffle();
    proptest::collection::vec(proptest::collection::vec(camada, layers), tokens)
}

/// `admit`: o modo, ou `None` para sortear; `faults`: sorteia uma das saídas de erro; `drop`: sempre cai no meio de um
/// token.
fn caso(admit: Option<AdmitMode>, faults: bool, drop: bool) -> impl Strategy<Value = Caso> {
    (1usize..=3, 2usize..=5, 1usize..=3, 1usize..=6).prop_flat_map(move |(layers, experts, topk, tokens)| {
        let topk = topk.min(experts);
        let keys = layers * experts;
        let admit_s = match admit {
            Some(a) => Just(a).boxed(),
            None => prop_oneof![Just(AdmitMode::Event), Just(AdmitMode::TokenBarrier)].boxed(),
        };
        let first = prop_oneof![Just(1u32), Just((1u32 << 31) - 2), Just(u32::MAX - 1), any::<u32>()];
        let calls = (tokens * (5 * layers + 2 + keys)) as u64;
        let fault = if drop {
            (0..tokens).prop_map(Fault::DropMidToken).boxed()
        } else if faults {
            prop_oneof![
                1 => Just(Fault::None),
                1 => (0..tokens * layers).prop_map(Fault::CpuErr),
                1 => (0..tokens * layers).prop_map(Fault::CpuPanic),
                1 => (1..=calls).prop_map(Fault::MainErr),
                1 => (0..tokens).prop_map(Fault::DropMidToken),
            ]
            .boxed()
        } else {
            Just(Fault::None).boxed()
        };
        let perm = Just((0..keys as u32).collect::<Vec<_>>()).prop_shuffle();
        (0..=keys, 0usize..=5, admit_s, first, routing(layers, experts, topk, tokens), sched(), fault, perm, 0..=keys).prop_map(
            move |(slots, backlog, admit, first_token, routing, sched, fault, perm, npre)| Caso {
                layers,
                experts,
                topk,
                slots,
                backlog,
                admit,
                first_token,
                prefill: perm[..npre.min(slots)].to_vec(),
                routing,
                sched,
                fault,
            },
        )
    })
}

fn pt(cases: u32, seed: u64) -> PtConfig {
    PtConfig { cases, failure_persistence: None, rng_seed: RngSeed::Fixed(seed), ..PtConfig::default() }
}

proptest! {
    #![proptest_config(pt(256, 0xA11_0001))]

    /// Os dois modos, sem falha: limpo, sem erro, e com os contadores do último token.
    #[test]
    fn invariantes_em_qualquer_intercalacao(c in caso(None, false, false)) {
        let s = roda(&c, None);
        prop_assert!(s.limpa(), "{s:?}");
        prop_assert!(s.err.is_none(), "{s:?}");
        let st = s.stats.expect("contadores");
        prop_assert_eq!(st.needed, (c.routing.len() * c.layers * c.topk) as u64);
        prop_assert_eq!(st.hits + st.cpu, st.needed);
    }
}

proptest! {
    #![proptest_config(pt(256, 0xA11_0002))]

    /// As saídas de erro do passo 6: a linha de serviço morre, a principal morre, o `Drop` no meio do token. Toda espera
    /// é solta, toda linha termina, nada fica em fila quando a memória cai; a morte da linha de serviço chega à principal
    /// como erro.
    #[test]
    fn saidas_de_erro_soltam_tudo(c in caso(None, true, false)) {
        let s = roda(&c, None);
        prop_assert!(s.limpa(), "{s:?}");
        if let Fault::CpuErr(_) | Fault::CpuPanic(_) = c.fault {
            prop_assert!(s.err.as_deref().is_some_and(|e| e.contains("ServiceFailed") || e.contains("Cpu")), "{s:?}");
        }
        if let Fault::None = c.fault {
            prop_assert!(s.err.is_none(), "{s:?}");
        }
    }
}

// --- O modo com barreira contra o simulador -------------------------------------------------------------------------

struct Fixture {
    caso: Caso,
    esperado: Stats,
}

fn fixtures() -> Vec<Fixture> {
    let txt = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/a1_sim_token_barreira.json"));
    let v: serde_json::Value = serde_json::from_str(txt).expect("fixture");
    assert_eq!(v["versao"], 1);
    let n = |x: &serde_json::Value| x.as_u64().expect("número");
    v["casos"]
        .as_array()
        .expect("casos")
        .iter()
        .map(|k| {
            let routing = k["routing"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_array().unwrap().iter().map(|l| l.as_array().unwrap().iter().map(|e| n(e) as u32).collect()).collect())
                .collect();
            let e = &k["esperado"];
            Fixture {
                caso: Caso {
                    layers: n(&k["layers"]) as usize,
                    experts: n(&k["experts"]) as usize,
                    topk: n(&k["topk"]) as usize,
                    slots: n(&k["slots"]) as usize,
                    backlog: n(&k["backlog"]) as usize,
                    admit: AdmitMode::TokenBarrier,
                    first_token: 1,
                    prefill: Vec::new(),
                    routing,
                    sched: SchedCfg { seed: 1, weights: [1; 4], budget: BUDGET },
                    fault: Fault::None,
                },
                esperado: Stats {
                    needed: n(&e["needed"]),
                    hits: n(&e["hits"]),
                    cpu: n(&e["cpu"]),
                    copies: n(&e["copies_bg"]),
                    dropped: n(&e["dropped"]),
                    deferred: n(&e["copies_deferred"]),
                },
            }
        })
        .collect()
}

/// Os índices das fixtures em que o mutante `m` provoca uma violação do tipo `kind` no backend falso (em ao menos uma das
/// quatro intercalações). A versão certa sai limpa em todas. Os controles da GPU (A1.3d) usam as fixtures em que o falso já mostrou que o
/// mutante lê dado errado, para pedir que a GPU de verdade também o mostre.
pub(crate) fn fixtures_em_que_o_mutante_viola(m: Mutant, kind: &str) -> Vec<usize> {
    fixtures()
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            // As quatro intercalações da comparação com o simulador: basta o mutante violar numa delas.
            [[1, 1, 1, 1], [8, 8, 8, 1], [8, 8, 1, 8], [8, 1, 8, 8]].iter().any(|&weights| {
                let mut c = f.caso.clone();
                c.sched = SchedCfg { seed: 0x5EED, weights, budget: BUDGET };
                assert!(roda(&c, None).limpa());
                roda(&c, Some(m)).tem(kind)
            })
        })
        .map(|(i, _)| i)
        .collect()
}

/// Cada fixture em quatro intercalações (uniforme, fila de cópia lenta, fila de cálculo lenta, linha de serviço lenta),
/// mais quatro sorteadas: os contadores têm de ser os do simulador em todas.
#[test]
fn token_barrier_bate_com_o_simulador() {
    let fx = fixtures();
    assert_eq!(fx.len(), 64);
    let pesos = [[1, 1, 1, 1], [8, 8, 8, 1], [8, 8, 1, 8], [8, 1, 8, 8]];
    let mut runner = TestRunner::new(pt(1, 0xA11_0003));
    let mut runs = 0;
    let t0 = Instant::now();
    for (i, f) in fx.iter().enumerate() {
        for k in 0..8 {
            let mut c = f.caso.clone();
            c.sched = if k < 4 {
                SchedCfg { seed: 0x5EED + i as u64, weights: pesos[k], budget: BUDGET }
            } else {
                sched().new_tree(&mut runner).unwrap().current()
            };
            let s = roda(&c, None);
            assert!(s.limpa() && s.err.is_none(), "fixture {i}, intercalação {k}: {s:?}");
            assert_eq!(s.stats.as_ref(), Some(&f.esperado), "fixture {i}, intercalação {k}");
            runs += 1;
        }
    }
    let tot = fx.iter().fold(Stats::default(), |mut a, f| {
        a.hits += f.esperado.hits;
        a.copies += f.esperado.copies;
        a.deferred += f.esperado.deferred;
        a.cpu += f.esperado.cpu;
        a
    });
    eprintln!("token_barrier: {runs} execuções em {:.1?}, iguais ao simulador (fixtures: {tot:?})", t0.elapsed());
}

/// O mesmo com threads de verdade (o executor do backend falso na fila de cálculo e a cópia na hora).
#[test]
fn token_barrier_com_threads_de_verdade() {
    for (i, f) in fixtures().iter().enumerate().take(24) {
        let c = &f.caso;
        let gpu = FakeGpu::new_executor(fake_cfg(c));
        let cfg = config(c, None);
        let parts = gpu.parts(&cfg);
        let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
        let (stats, err) = dirige(&mut eng, c);
        eng.close().unwrap();
        gpu.shutdown();
        assert!(err.is_none(), "fixture {i}: {err:?}");
        assert_eq!(stats.as_ref(), Some(&f.esperado), "fixture {i}");
        assert_eq!(gpu.violations(), vec![], "fixture {i}");
    }
}

// --- Testes dirigidos ------------------------------------------------------------------------------------------------

/// Teste de estado (a) do A0b: a montagem da tabela recusa slot que não está residente.
#[test]
fn estado_a_tabela_recusa_slot_loading() {
    let mut tb = TableBuilder::new(4);
    assert_eq!(tb.insert(0, SlotState::Loading { key: 1 }), Err(0));
    assert_eq!(tb.insert(1, SlotState::Held { key: 2, after: 0 }), Err(1));
    assert_eq!(tb.insert(2, SlotState::Free), Err(2));
    assert_eq!(tb.insert(3, SlotState::Resident { key: 3 }), Ok(()));
    assert_eq!(tb.finish(), vec![NO_SLOT, NO_SLOT, NO_SLOT, 3]);
}

fn caso_fixo(first_token: u32) -> Caso {
    Caso {
        layers: 2,
        experts: 3,
        topk: 2,
        slots: 3,
        backlog: 4,
        admit: AdmitMode::Event,
        first_token,
        prefill: vec![5],
        routing: [vec![vec![0, 1], vec![2, 0]], vec![vec![1, 2], vec![0, 1]], vec![vec![0, 2], vec![2, 1]]].iter().cycle().take(6).cloned().collect(),
        sched: SchedCfg { seed: 99, weights: [2, 2, 2, 2], budget: BUDGET },
        fault: Fault::None,
    }
}

#[test]
fn published_de_outra_geracao_recusado() {
    let c = caso_fixo(1);
    let gpu = FakeGpu::new_sched(fake_cfg(&c), c.sched);
    let cfg = config(&c, None);
    let parts = gpu.parts(&cfg);
    let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
    let mut p = eng.next_published().unwrap();
    p.gen = p.gen.wrapping_add(1);
    assert!(matches!(eng.enqueue_token(p), Err(Error::WrongGeneration { expected: 1, got: 2 })));
    drop(eng);
    assert!(!gpu.stalled());
    assert_eq!(gpu.violations(), vec![]);
}

/// Um `Published` de outro motor com a mesma geração é recusado antes de qualquer operação de backend (achado P1 da
/// rodada `lastro-rust-2`): sem a identidade de origem, a GPU receberia a tabela do outro motor.
#[test]
fn published_de_outro_motor_recusado() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut c = caso_fixo(1);
        let mut mk = |prefill: Vec<u32>| {
            c.prefill = prefill;
            let gpu = FakeGpu::new_executor(fake_cfg(&c));
            let cfg = config(&c, None);
            let parts = gpu.parts(&cfg);
            let eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
            (gpu, eng)
        };
        let (ga, mut a) = mk(vec![0]);
        let (gb, mut b) = mk(vec![1]);
        let pa = a.next_published().unwrap();
        let pb = b.next_published().unwrap();
        let mesma_geracao = pa.generation() == pb.generation();
        let estranha = a.enqueue_token(pb).err();
        let depois = a.next_published().err();
        let propria = b.enqueue_token(pa).err(); // o A também não vale no B
        drop((a, b));
        ga.shutdown();
        gb.shutdown();
        let _ = tx.send((mesma_geracao, format!("{estranha:?}"), format!("{depois:?}"), format!("{propria:?}"),
                         ga.violations(), gb.violations(), ga.queued(), gb.queued()));
    });
    let (mesma, e1, e2, e3, va, vb, qa, qb) =
        rx.recv_timeout(Duration::from_secs(5)).expect("o motor travou com o Published de outro motor");
    assert!(mesma, "o teste precisa de duas publicações com a mesma geração");
    assert!(e1.contains("outro motor"), "{e1}");
    assert!(e2.contains("Cancelled"), "{e2}");
    assert!(e3.contains("outro motor"), "{e3}");
    assert_eq!((va, vb, qa, qb), (vec![], vec![], (0, 0), (0, 0)));
}

/// A mesma semente e os mesmos pesos dão a mesma execução (achado P2 da rodada `lastro-rust-2`: a linha de serviço
/// entrava no sorteio só quando a thread começava, e o resultado dependia do tempo real).
#[test]
fn mesma_semente_mesma_execucao() {
    let mut c = caso_fixo(1);
    c.sched = SchedCfg { seed: 12345, weights: [8, 1, 8, 1], budget: BUDGET };
    let ref_ = roda(&c, Some(Mutant::PublishedBeforeRecord));
    let chave = |s: &Saida| format!("{:?}|{:?}|{:?}", s.violations, s.stats, s.queued);
    for i in 0..10 {
        let s = roda(&c, Some(Mutant::PublishedBeforeRecord));
        assert_eq!(chave(&s), chave(&ref_), "a execução {i} diferiu da primeira");
    }
}

/// O contador atravessa 2^31 e o zero (o `GEQ` cíclico e o valor de soltura).
#[test]
fn contador_atravessa_2_31_e_o_zero() {
    for first in [(1u32 << 31) - 3, u32::MAX - 2] {
        let mut c = caso_fixo(first);
        let s = roda(&c, None);
        assert!(s.limpa() && s.err.is_none(), "{first}: {s:?}");
        c.fault = Fault::DropMidToken(4);
        let s = roda(&c, None);
        assert!(s.limpa(), "{first}, queda no meio do token: {s:?}");
    }
}

/// A linha de serviço em `panic` no meio do token: a chamada da principal volta com erro em menos de 1 s, com threads de
/// verdade, e nada fica em fila.
#[test]
fn panic_da_linha_de_servico_solta_a_principal() {
    // Sem a guarda `DoneOnExit` o `close` gira para sempre: o corpo roda numa thread e a espera tem teto, para o teste
    // ficar vermelho em vez de travar.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let c = caso_fixo(1);
        let gpu = FakeGpu::new_executor(fake_cfg(&c));
        let cfg = config(&c, None);
        let parts = gpu.parts(&cfg);
        let cpu = FakeCpu { panic_at: Some(3), ..Default::default() };
        let mut eng = Engine::new(gpu.clone(), parts, cfg, cpu).unwrap();
        let t0 = Instant::now();
        let mut err = None;
        for _ in 0..c.routing.len() {
            match eng.next_published().and_then(|p| eng.run_token(p)) {
                Ok(()) => {}
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }
        let dt = t0.elapsed();
        let fechou = eng.close();
        gpu.shutdown();
        let _ = tx.send((format!("{err:?}"), dt, matches!(fechou, Err(Error::ServiceFailed)), gpu.violations(), gpu.queued()));
    });
    let (err, dt, fechou, violations, queued) =
        rx.recv_timeout(Duration::from_secs(5)).expect("o motor travou depois do panic da linha de serviço");
    assert!(err.contains("ServiceFailed"), "{err}");
    assert!(dt < Duration::from_secs(1), "{dt:?}");
    assert!(fechou);
    assert_eq!(violations, vec![]);
    assert_eq!(queued, (0, 0));
    eprintln!("panic na linha de serviço: a principal voltou com ServiceFailed em {dt:?}");
}

/// Depois de um `WrongGeneration` o `Published` foi consumido: o motor cai e as chamadas seguintes voltam com `Cancelled`
/// em vez de travar (achado P2-1 da rodada lastro-rust-1).
#[test]
fn wrong_generation_derruba_o_motor_sem_travar() {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let c = caso_fixo(1);
        let gpu = FakeGpu::new_executor(fake_cfg(&c));
        let cfg = config(&c, None);
        let parts = gpu.parts(&cfg);
        let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
        let p1 = eng.next_published().unwrap();
        eng.enqueue_token(p1).unwrap();
        let p2 = eng.next_published().unwrap();
        let e1 = eng.enqueue_token(p2).err();
        let e2 = eng.next_published().err();
        let e3 = eng.finish_token().err();
        drop(eng);
        gpu.shutdown();
        let _ = tx.send((format!("{e1:?}"), format!("{e2:?}"), format!("{e3:?}"), gpu.violations(), gpu.queued()));
    });
    let (e1, e2, e3, violations, queued) =
        rx.recv_timeout(Duration::from_secs(5)).expect("o motor travou depois do WrongGeneration");
    assert!(e1.contains("WrongGeneration"), "{e1}");
    assert!(e2.contains("Cancelled"), "{e2}");
    assert!(e3.contains("Cancelled") || e3.contains("Config"), "{e3}");
    assert_eq!(violations, vec![]);
    assert_eq!(queued, (0, 0));
}

#[test]
fn finish_token_sem_token_enfileirado_e_recusado() {
    let c = caso_fixo(1);
    let gpu = FakeGpu::new_sched(fake_cfg(&c), c.sched);
    let cfg = config(&c, None);
    let parts = gpu.parts(&cfg);
    let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
    assert!(matches!(eng.finish_token(), Err(Error::Config(_))));
}

/// A validação do `prefill` vem na construção, na thread do chamador (achado P3-3).
#[test]
fn prefill_invalido_e_recusado_na_construcao() {
    for prefill in [vec![0, 0], vec![0, 1, 2, 3]] {
        let mut c = caso_fixo(1);
        c.slots = 3;
        c.prefill = prefill;
        let gpu = FakeGpu::new_sched(fake_cfg(&c), c.sched);
        let cfg = config(&c, None);
        let parts = gpu.parts(&cfg);
        assert!(matches!(Engine::new(gpu, parts, cfg, FakeCpu::default()), Err(Error::Config(_))));
    }
}

// --- Mutantes do backend falso ---------------------------------------------------------------------------------------

/// Roda `casos` casos de `estrategia` (semente fixa) na versão certa e no mutante. A certa sai limpa em todos; devolve em
/// quantos o mutante teve cada tipo de violação de `kinds`.
fn acusa(m: Mutant, estrategia: impl Strategy<Value = Caso>, casos: usize, seed: u64, kinds: &[&str]) -> Vec<usize> {
    let mut runner = TestRunner::new(pt(1, seed));
    let mut n = vec![0; kinds.len()];
    let mut menor: Option<Caso> = None;
    let t0 = Instant::now();
    for _ in 0..casos {
        let c = estrategia.new_tree(&mut runner).unwrap().current();
        let certo = roda(&c, None);
        assert!(certo.limpa(), "versão certa com violação: {certo:?}\n{c:?}");
        let mutante = roda(&c, Some(m));
        let mut pego = false;
        for (i, k) in kinds.iter().enumerate() {
            if mutante.tem(k) {
                n[i] += 1;
                pego = true;
            }
        }
        if pego && menor.as_ref().is_none_or(|x| x.routing.len() * x.layers > c.routing.len() * c.layers) {
            menor = Some(c);
        }
    }
    let desc: Vec<String> = kinds.iter().zip(&n).map(|(k, v)| format!("{k} em {v}/{casos}")).collect();
    eprintln!("mutante {m:?}: {} ({:.1?}); menor caso acusado: {menor:?}", desc.join(", "), t0.elapsed());
    n
}

const CASOS_MUTANTE: usize = 200;

#[test]
fn mutante_copia_na_escolha_da_vitima() {
    let n = acusa(Mutant::CopyAtVictim, caso(None, false, false), CASOS_MUTANTE, 0xA11_0101, &["lançamento"]);
    assert!(n[0] > 0, "o mutante não foi acusado");
}

#[test]
fn mutante_tabela_sem_as_duas_protecoes_do_ready() {
    let n = acusa(Mutant::NoReadyProtection, caso(Some(AdmitMode::Event), false, false), CASOS_MUTANTE, 0xA11_0102,
                  &["lançamento", "tabela"]);
    assert!(n[0] > 0, "nenhum lançamento leu slot em cópia");
}

#[test]
fn mutante_published_antes_de_gravar_o_ready() {
    // Não é mascarado: a espera da principal captura a última gravação feita antes da chamada, e aqui a gravação vem
    // depois do `Published`. Com a principal à frente da linha de serviço (pesos altos nela e na fila de cálculo), a
    // espera não segura nada e o lançamento lê o slot em cópia. Gravar o `ready` antes do `Published` é necessário.
    let favorece_a_principal = caso(Some(AdmitMode::Event), false, false).prop_map(|mut c| {
        c.sched.weights = [8, 1, 8, 1];
        c
    });
    let n = acusa(Mutant::PublishedBeforeRecord, favorece_a_principal, 400, 0xA11_0103, &["lançamento", "tabela"]);
    assert!(n[0] > 0, "o lançamento nunca leu um slot em cópia");
}

/// A vítima é de uma camada à frente da que está sendo servida, e essa camada lê o slot da vítima: o token 0 pede a chave
/// 0 na camada 0 (falta, admite) e a chave 2 (o prefill, camada 1) na camada 1. Soltar a cópia guardada ao ver `seq[l']`,
/// e não `seq[l'+1]`, faz a cópia correr com o lançamento da camada 1: o conferidor acusa o lançamento (e não um
/// `assert!` estrutural, achado P1-2).
#[test]
fn mutante_copia_uma_camada_cedo_e_acusado_no_lancamento() {
    let base = Caso {
        layers: 2,
        experts: 2,
        topk: 1,
        slots: 1,
        backlog: 2,
        admit: AdmitMode::Event,
        first_token: 1,
        prefill: vec![2],
        routing: vec![vec![vec![0], vec![0]], vec![vec![1], vec![1]]],
        sched: SchedCfg { seed: 0, weights: [1, 1, 1, 1], budget: BUDGET },
        fault: Fault::None,
    };
    let (mut lancamento, mut estrutural) = (0, 0);
    for seed in 0..200u64 {
        let c = Caso { sched: SchedCfg { seed, weights: [1 + (seed % 8) as u32, 1 + (seed / 8 % 8) as u32, 4, 3], budget: BUDGET }, ..base.clone() };
        let certo = roda(&c, None);
        assert!(certo.limpa() && certo.err.is_none(), "versão certa: {certo:?}");
        let m = roda(&c, Some(Mutant::HeldOneLayerEarly));
        if m.tem("lançamento") {
            lancamento += 1;
        }
        if m.err.is_some() {
            estrutural += 1;
        }
    }
    eprintln!("mutante HeldOneLayerEarly: lançamento em {lancamento}/200, saída com erro em {estrutural}/200");
    assert!(lancamento > 0, "o conferidor nunca acusou o lançamento");
}

/// A consulta antes da gravação: a tabela lista o slot em cópia, mas a espera da principal pelo `ready` (gravado antes do
/// `Published`) segura a GPU. O teste registra esse mascaramento: o lançamento nunca lê o slot errado.
#[test]
fn mutante_consulta_do_ready_antes_da_gravacao() {
    let n = acusa(Mutant::QueryBeforeRecord, caso(Some(AdmitMode::Event), false, false), CASOS_MUTANTE, 0xA11_0104,
                  &["tabela", "lançamento"]);
    assert!(n[0] > 0, "a tabela nunca listou um slot em cópia");
    assert_eq!(n[1], 0, "a espera da principal pelo ready devia mascarar o lançamento");
}

#[test]
fn mutante_ordem_antiga_do_drop() {
    let n = acusa(Mutant::OldDropOrder, caso(None, false, true), CASOS_MUTANTE, 0xA11_0105, &["liberada com fila"]);
    assert!(n[0] > 0, "nenhuma memória caiu com cópia na fila");
}
