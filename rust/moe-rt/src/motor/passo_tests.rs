//! A2-PASSO E4a, U1 (contrato-e4a.md v2): as formas do passo de M posições, sem a linha de serviço (que só aprende M > 1 na U2).
//! Layout da área da camada, `begin_step` em ordem FIFO no backend falso, `Route` por posição global, validação de `m` e das peças.
//! Com `max_positions == 1` o motor é o de sempre: a suíte de `tests.rs` roda sem alteração de asserção.
use super::backend::Backend;
use super::fake::{expert_out, peso, x_value, FakeCfg, FakeCpu, FakeGpu, Violation};
use super::sync::Ordering;
use std::sync::Arc;
use super::*;

const K: usize = 2;
const W: usize = 3;
const MP: usize = 4;

/// `routing[p][l]`: posições distintas para o teste distinguir a posição global.
fn routing() -> Vec<Vec<Vec<u32>>> {
    (0..8u32).map(|p| (0..2u32).map(|l| vec![(p + l) % 4, (p * 2 + l + 1) % 4]).collect()).collect()
}

fn fake(max_positions: usize) -> Arc<FakeGpu> {
    FakeGpu::new_executor(FakeCfg { layers: 2, experts: 4, topk: K, width: W, routing: routing(), fail_main_at: None,
                                    copy_immediate: false, max_positions })
}

fn config(max_positions: usize) -> Config {
    Config { layers: 2, experts: 4, topk: K, width: W, slots: 3, backlog: 4, admit: AdmitMode::TokenBarrier, first_token: 1,
             prefill: vec![], max_positions, mutant: None }
}

#[test]
fn offsets_com_uma_posicao_sao_os_de_antes() {
    let o = LayerIo::offsets(8, 1536, 1);
    assert_eq!(o, LayerOffsets { ids: 0, w: 8, x: 16, y: 16 + 1536, m: None, total: 16 + 2 * 1536 });
    // O layout legado que o backend de CUDA usava à mão: ids em 0, w em topk, x em 2·topk, y em 2·topk + hidden.
    assert_eq!((o.ids, o.w, o.x, o.y), (0, 8, 2 * 8, 2 * 8 + 1536));
    let io = LayerIo::new(8, 1536);
    assert_eq!((io.ids.len(), io.w.len(), io.x.len(), io.y.len(), io.m.len()), (8, 8, 1536, 1536, 0));
    assert_eq!(io.step_size(), 1);
}

#[test]
fn offsets_com_varias_posicoes_e_a_palavra_m_no_fim() {
    let (k, w, mp) = (8usize, 1536usize, 8usize);
    let o = LayerIo::offsets(k, w, mp);
    assert_eq!(o.ids, 0);
    assert_eq!(o.w, mp * k);
    assert_eq!(o.x, 2 * mp * k);
    assert_eq!(o.y, 2 * mp * k + mp * w);
    assert_eq!(o.m, Some(2 * mp * k + 2 * mp * w));
    assert_eq!(o.total, 2 * mp * k + 2 * mp * w + 1);
    let io = LayerIo::with_positions(k, w, mp);
    assert_eq!((io.ids.len(), io.w.len(), io.x.len(), io.y.len(), io.m.len()), (mp * k, mp * k, mp * w, mp * w, 1));
    assert_eq!(io.step_size(), 0, "a palavra começa em zero: o backend a escreve a cada passo");
    io.m[0].store(5, Ordering::Relaxed);
    assert_eq!(io.step_size(), 5);
}

/// `begin_step` entra na fila de cálculo: a `Route` enfileirada depois dele vê o descritor dele, mesmo com outro `begin_step`
/// enfileirado depois e antes de a fila andar (FIFO, nada de campo global mudado pelo host).
#[test]
fn begin_step_e_route_andam_em_ordem_fifo_por_posicao_global() {
    let gpu = fake(MP);
    let parts = gpu.parts(&config(MP));
    let s = &parts.compute;
    for io in &parts.io {
        for a in io.ids.iter().chain(io.x.iter()) {
            a.store(0xDEAD, Ordering::Relaxed);
        }
    }
    gpu.begin_step(s, 1, 0, 3).unwrap();
    gpu.route(s, 0, &parts.io[0]).unwrap();
    gpu.begin_step(s, 2, 3, 2).unwrap();
    gpu.route(s, 1, &parts.io[1]).unwrap();
    gpu.sync(s).unwrap();
    let r = routing();
    // Passo 1: posições globais 0, 1, 2 na camada 0, m = 3.
    let io0 = &parts.io[0];
    assert_eq!(io0.step_size(), 3);
    for p in 0..3 {
        assert_eq!(LayerIo::load(&io0.ids)[p * K..(p + 1) * K], r[p][0][..], "ids da posição {p}");
        for j in 0..W {
            assert_eq!(io0.x[p * W + j].load(Ordering::Relaxed), x_value(p, 0, j), "x da posição {p}");
        }
    }
    // A cauda (posição 3 do bloco) não foi escrita.
    assert!(LayerIo::load(&io0.ids)[3 * K..].iter().all(|&v| v == 0xDEAD));
    // Passo 2: posições globais 3 e 4 na camada 1, m = 2.
    let io1 = &parts.io[1];
    assert_eq!(io1.step_size(), 2);
    for p in 0..2 {
        assert_eq!(LayerIo::load(&io1.ids)[p * K..(p + 1) * K], r[3 + p][1][..], "ids da posição global {}", 3 + p);
        assert_eq!(io1.x[p * W].load(Ordering::Relaxed), x_value(3 + p, 1, 0));
    }
    assert!(LayerIo::load(&io1.ids)[2 * K..].iter().all(|&v| v == 0xDEAD));
    gpu.shutdown();
    assert_eq!(gpu.violations(), vec![]);
}

/// O conferidor do `AddY` olha cada posição do `y` com o `x` e as faltas dela: um `y` certo passa, um errado numa só posição é acusado
/// com a posição.
#[test]
fn add_y_confere_cada_posicao() {
    for erra in [None, Some(1usize)] {
        let gpu = fake(MP);
        let parts = gpu.parts(&config(MP));
        let s = &parts.compute;
        let io = &parts.io[0];
        gpu.begin_step(s, 1, 0, 3).unwrap();
        gpu.route(s, 0, io).unwrap();
        gpu.sync(s).unwrap();
        // Nenhum lançamento: todas as chaves são faltas da CPU. O `y` esperado de cada posição soma os experts dela.
        let r = routing();
        for p in 0..3 {
            for j in 0..W {
                let x = x_value(p, 0, j);
                let mut y = r[p][0].iter().enumerate().fold(0u32, |a, (j2, &e)| a.wrapping_add(peso(p, 0, j2).wrapping_mul(expert_out(e, x))));
                if erra == Some(p) {
                    y ^= 1;
                }
                io.y[p * W + j].store(y, Ordering::Relaxed);
            }
        }
        gpu.add_y(s, 0, io).unwrap();
        gpu.sync(s).unwrap();
        gpu.shutdown();
        match erra {
            None => assert_eq!(gpu.violations(), vec![]),
            Some(p) => {
                let v = gpu.violations();
                assert_eq!(v.len(), 1, "{v:?}");
                assert!(matches!(&v[0], Violation::Y { pos, layer: 0, .. } if *pos == p), "{v:?}");
            }
        }
    }
}

/// Um `m` fora da faixa (0 e acima do máximo) derruba o motor sem travar, com áreas de várias posições. Os `m` válidos são servidos (U2).
#[test]
fn m_fora_da_faixa_derruba_o_motor_sem_travar() {
    for m in [0usize, MP + 1] {
        let gpu = fake(MP);
        let cfg = config(MP);
        let parts = gpu.parts(&cfg);
        let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
        let p = eng.next_published().unwrap();
        assert!(matches!(eng.enqueue_step(p, m), Err(Error::Config(_))), "m = {m}");
        assert!(matches!(eng.finish_token(), Err(Error::Cancelled)), "o motor ficou derrubado");
        drop(eng);
        gpu.shutdown();
        assert!(!gpu.stalled());
        assert_eq!(gpu.violations(), vec![]);
    }
}

/// Um backend de capacidade 4 com áreas e configuração de uma posição (combinação que `Engine::new` aceita) roda no caminho legado:
/// o falso escolhe o layout pela área recebida, não pelo teto. Três tokens com rotas diferentes, sem pânico nem violação.
#[test]
fn capacidade_maior_que_a_configuracao_roda_no_caminho_legado() {
    let gpu = fake(MP);
    let cfg = config(1);
    let parts = gpu.parts(&cfg);
    let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
    let mut ultimo = None;
    for _ in 0..3 {
        let p = eng.next_published().unwrap();
        ultimo = Some(p.stats().clone());
        eng.run_token(p).unwrap();
    }
    let p = eng.next_published().unwrap();
    let st = p.stats().clone();
    assert_eq!(st.needed, 3 * 2 * K as u64, "três tokens, duas camadas, topk pedidos cada: {ultimo:?} {st:?}");
    assert_eq!(st.hits + st.cpu, st.needed);
    drop(p);
    assert!(eng.close().is_ok());
    gpu.shutdown();
    assert!(!gpu.stalled());
    assert_eq!(gpu.violations(), vec![]);
}

#[test]
fn pecas_que_nao_batem_com_max_positions_sao_recusadas() {
    // O backend não aceita mais posições do que declara (o padrão do trait é 1).
    let gpu = fake(1);
    let cfg = config(2);
    let parts = gpu.parts(&config(1));
    assert!(matches!(Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()), Err(Error::Config(_))));
    gpu.shutdown();
    // As áreas das camadas têm de ter o tamanho do `max_positions` da configuração.
    let gpu = fake(MP);
    let parts = gpu.parts(&config(1));
    assert!(matches!(Engine::new(gpu.clone(), parts, config(MP), FakeCpu::default()), Err(Error::Config(_))));
    gpu.shutdown();
    let gpu = fake(MP);
    let parts = gpu.parts(&config(MP));
    assert!(matches!(Engine::new(gpu.clone(), parts, config(1), FakeCpu::default()), Err(Error::Config(_))));
    gpu.shutdown();
    // max_positions = 0 nunca vale.
    let gpu = fake(MP);
    let parts = gpu.parts(&config(MP));
    assert!(matches!(Engine::new(gpu.clone(), parts, config(0), FakeCpu::default()), Err(Error::Config(_))));
    gpu.shutdown();
}

#[test]
fn motor_com_varias_posicoes_constroi_e_fecha() {
    let gpu = fake(MP);
    let cfg = config(MP);
    let parts = gpu.parts(&cfg);
    let eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).unwrap();
    assert!(eng.close().is_ok());
    gpu.shutdown();
    assert!(!gpu.stalled());
    assert_eq!(gpu.violations(), vec![]);
}

// --- U2: a linha de serviço com passos de M posições, contra o oráculo B (SimM com barreira) ---------------------------------------

use super::fake::SchedCfg;

struct CasoB {
    layers: usize,
    experts: usize,
    topk: usize,
    slots: usize,
    backlog: usize,
    passos: Vec<usize>,
    routing: Vec<Vec<Vec<u32>>>,
    esperado: Stats,
    tipo: String,
}

fn casos_b() -> Vec<CasoB> {
    let txt = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/a2_sim_passo_barreira.json"));
    let v: serde_json::Value = serde_json::from_str(txt).expect("fixture");
    assert_eq!(v["versao"], 2);
    let n = |x: &serde_json::Value| x.as_u64().expect("número");
    v["casos"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| {
            let e = &k["esperado"];
            CasoB {
                layers: n(&k["layers"]) as usize,
                experts: n(&k["experts"]) as usize,
                topk: n(&k["topk"]) as usize,
                slots: n(&k["slots"]) as usize,
                backlog: n(&k["backlog"]) as usize,
                passos: k["passos"].as_array().unwrap().iter().map(|x| n(x) as usize).collect(),
                routing: k["routing"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t.as_array().unwrap().iter().map(|l| l.as_array().unwrap().iter().map(|e| n(e) as u32).collect()).collect())
                    .collect(),
                esperado: Stats {
                    needed: n(&e["needed"]),
                    hits: n(&e["hits"]),
                    cpu: n(&e["cpu"]),
                    copies: n(&e["copies_bg"]),
                    dropped: n(&e["dropped"]),
                    deferred: n(&e["copies_deferred"]),
                },
                tipo: k["tipo"].as_str().unwrap().to_string(),
            }
        })
        .collect()
}

/// Roda os passos do caso; devolve os contadores lidos no `Published` seguinte ao último passo, as violações e o estado do falso.
fn roda_b(c: &CasoB, sched: SchedCfg, max_positions: usize) -> (Stats, Vec<Violation>, bool) {
    let gpu = FakeGpu::new_sched(
        FakeCfg { layers: c.layers, experts: c.experts, topk: c.topk, width: 2, routing: c.routing.clone(), fail_main_at: None,
                  copy_immediate: false, max_positions },
        sched,
    );
    let cfg = Config { layers: c.layers, experts: c.experts, topk: c.topk, width: 2, slots: c.slots, backlog: c.backlog,
                       admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions, mutant: None };
    let parts = gpu.parts(&cfg);
    let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu { weighted: true, ..Default::default() }).expect("motor");
    for &m in &c.passos {
        let p = eng.next_published().expect("published");
        eng.enqueue_step(p, m).expect("enfileira");
        eng.finish_token().expect("termina");
    }
    let p = eng.next_published().expect("published final");
    let st = p.stats().clone();
    drop(p);
    eng.close().expect("fecha");
    (st, gpu.violations(), gpu.stalled())
}

/// As 123 fixtures do SimM com barreira, em quatro intercalações: contadores iguais, sem violação e sem travar.
#[test]
fn passo_de_m_posicoes_bate_com_o_simulador_com_barreira() {
    let pesos = [[1, 1, 1, 1], [8, 8, 8, 1], [8, 8, 1, 8], [8, 1, 8, 8]];
    let casos = casos_b();
    assert_eq!(casos.len(), 123);
    assert!(casos.iter().any(|c| c.passos.iter().any(|&m| m > 1)));
    for (i, c) in casos.iter().enumerate() {
        for (k, w) in pesos.iter().enumerate() {
            let sched = SchedCfg { seed: 0x5EED + i as u64, weights: *w, budget: 300_000 };
            let (st, viol, stalled) = roda_b(c, sched, 8);
            assert!(viol.is_empty() && !stalled, "caso {i} ({}), intercalação {k}: {viol:?} stalled={stalled}", c.tipo);
            assert_eq!(st, c.esperado, "caso {i} ({}), intercalação {k}", c.tipo);
        }
    }
}

/// Um passo de `m` posições com o `LayerIo` de várias posições e `m = 1` em todos os passos dá o mesmo que o motor de uma posição.
#[test]
fn m_igual_a_um_com_areas_largas_bate_com_as_fixtures_a1() {
    let txt = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/a1_sim_token_barreira.json"));
    let v: serde_json::Value = serde_json::from_str(txt).expect("fixture");
    let n = |x: &serde_json::Value| x.as_u64().expect("número");
    for (i, k) in v["casos"].as_array().unwrap().iter().enumerate() {
        let routing: Vec<Vec<Vec<u32>>> = k["routing"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t.as_array().unwrap().iter().map(|l| l.as_array().unwrap().iter().map(|e| n(e) as u32).collect()).collect())
            .collect();
        let e = &k["esperado"];
        let c = CasoB {
            layers: n(&k["layers"]) as usize,
            experts: n(&k["experts"]) as usize,
            topk: n(&k["topk"]) as usize,
            slots: n(&k["slots"]) as usize,
            backlog: n(&k["backlog"]) as usize,
            passos: vec![1; routing.len()],
            routing,
            esperado: Stats { needed: n(&e["needed"]), hits: n(&e["hits"]), cpu: n(&e["cpu"]), copies: n(&e["copies_bg"]),
                              dropped: n(&e["dropped"]), deferred: n(&e["copies_deferred"]) },
            tipo: "a1".into(),
        };
        let sched = SchedCfg { seed: 0x5EED + i as u64, weights: [1, 1, 1, 1], budget: 300_000 };
        let (st, viol, stalled) = roda_b(&c, sched, 8);
        assert!(viol.is_empty() && !stalled, "caso A1 {i}: {viol:?}");
        assert_eq!(st, c.esperado, "caso A1 {i}");
    }
}

// --- U2: a leitura do passo (`read_step`) ------------------------------------------------------------------------------------------

use super::line::read_step;
use super::sync::AtomicU32;

fn area_cheia(k: usize, w: usize, mp: usize, m: u32) -> (AtomicU32, LayerIo) {
    let io = LayerIo::with_positions(k, w, mp);
    for (i, a) in io.ids.iter().enumerate() {
        a.store(100 + i as u32, Ordering::Relaxed);
    }
    for (i, a) in io.w.iter().enumerate() {
        a.store(200 + i as u32, Ordering::Relaxed);
    }
    for (i, a) in io.x.iter().enumerate() {
        a.store(300 + i as u32, Ordering::Relaxed);
    }
    io.m[0].store(m, Ordering::Relaxed);
    (AtomicU32::new(5), io)
}

/// Só as `m` primeiras posições são lidas; a cauda (sentinela) não é interpretada nem copiada.
#[test]
fn read_step_le_so_as_m_primeiras_posicoes() {
    let (k, w, mp) = (2usize, 3usize, 4usize);
    for m in 1..=mp as u32 {
        let (seq, io) = area_cheia(k, w, mp, m);
        let r = read_step(&seq, &io, 5, false, k, w, mp).expect("alcançado").expect("m válido");
        assert_eq!(r.m, m as usize);
        assert_eq!(r.ids, (0..m as usize * k).map(|i| 100 + i as u32).collect::<Vec<_>>());
        assert_eq!(r.w, (0..m as usize * k).map(|i| 200 + i as u32).collect::<Vec<_>>());
        assert_eq!(r.x, (0..m as usize * w).map(|i| 300 + i as u32).collect::<Vec<_>>());
    }
}

/// `m` fora da faixa é erro com o valor lido; antes de a `seq` chegar, nada é lido.
#[test]
fn read_step_recusa_m_fora_da_faixa_e_espera_a_seq() {
    let (k, w, mp) = (2usize, 3usize, 4usize);
    for m in [0u32, mp as u32 + 1, u32::MAX] {
        let (seq, io) = area_cheia(k, w, mp, m);
        assert!(matches!(read_step(&seq, &io, 5, false, k, w, mp), Some(Err(v)) if v == m as usize), "m = {m}");
    }
    let (seq, io) = area_cheia(k, w, mp, 2);
    assert!(read_step(&seq, &io, 6, false, k, w, mp).is_none(), "seq ainda em 5: não alcançou 6");
}

/// `m` divergente entre camadas do mesmo passo (o backend grava 2 na camada 0 e 3 na camada 1) derruba o motor sem travar e sem
/// violação; com a camada de `m` igual o mesmo passo é servido.
#[test]
fn m_divergente_entre_camadas_derruba_o_motor() {
    for (diverge, ok) in [(false, true), (true, false)] {
        let gpu = fake(MP);
        if diverge {
            gpu.set_m_word_override(1, 3);
        }
        let cfg = config(MP);
        let parts = gpu.parts(&cfg);
        let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu { weighted: true, ..Default::default() }).unwrap();
        let p = eng.next_published().unwrap();
        let r = eng.enqueue_step(p, 2).and_then(|_| eng.finish_token());
        assert_eq!(r.is_ok(), ok, "diverge = {diverge}: {r:?}");
        let fim = eng.close();
        assert_eq!(fim.is_ok(), ok, "close: {fim:?}");
        gpu.shutdown();
        assert!(!gpu.stalled());
        assert_eq!(gpu.violations(), vec![], "diverge = {diverge}");
        assert_eq!(gpu.queued(), (0, 0), "filas drenadas, diverge = {diverge}");
    }
}

// --- U3: o oráculo A (tabela congelada), o `compute_step` em lote e o padrão, o mutante de peso ------------------------------------

use super::backend::{CpuStep, HostExperts};

/// A CPU sem o override: usa o `compute_step` padrão (laço por posição sobre `compute`).
struct PorPosicao(FakeCpu);

impl HostExperts for PorPosicao {
    type Error = String;
    fn compute(&mut self, layer: usize, keys: &[u32], weights: &[u32], x: &[u32], y: &mut [u32]) -> Result<(), String> {
        self.0.compute(layer, keys, weights, x, y)
    }
}

const TOTAL_A: usize = 39;

fn rota_a(p: usize, l: usize) -> Vec<u32> {
    vec![((p + l) % 4) as u32, ((p + l + 1 + (p / 4) % 3) % 4) as u32]
}

fn passos_de(m: usize) -> Vec<usize> {
    let mut v = vec![m; TOTAL_A / m];
    if TOTAL_A % m != 0 {
        v.push(TOTAL_A % m);
    }
    v
}

/// Roda `passos` com a tabela congelada (`backlog = 0`, só o `prefill` residente) e devolve os contadores, as violações, o travamento e o
/// resultado da execução.
fn roda_a<C: HostExperts>(passos: &[usize], prefill: Vec<u32>, slots: usize, cpu: C, sched: SchedCfg) -> (Stats, Vec<Violation>, bool, Result<(), String>) {
    let (st, v, stalled, r, _) = roda_a_t(passos, prefill, slots, cpu, sched, 1);
    (st, v, stalled, r)
}

/// Como [`roda_a`], com o primeiro valor do contador de gerações e a contagem de `sync` lida antes de fechar.
fn roda_a_t<C: HostExperts>(passos: &[usize], prefill: Vec<u32>, slots: usize, cpu: C, sched: SchedCfg, first_token: u32)
    -> (Stats, Vec<Violation>, bool, Result<(), String>, [[u64; 2]; 3]) {
    let total: usize = passos.iter().sum();
    let routing: Vec<Vec<Vec<u32>>> = (0..total).map(|p| (0..2).map(|l| rota_a(p, l)).collect()).collect();
    let gpu = FakeGpu::new_sched(
        FakeCfg { layers: 2, experts: 4, topk: K, width: W, routing, fail_main_at: None, copy_immediate: false, max_positions: 8 },
        sched,
    );
    let cfg = Config { layers: 2, experts: 4, topk: K, width: W, slots, backlog: 0, admit: AdmitMode::TokenBarrier, first_token,
                       prefill, max_positions: 8, mutant: None };
    let parts = gpu.parts(&cfg);
    let mut eng = Engine::new(gpu.clone(), parts, cfg, cpu).expect("motor");
    let mut res = Ok(());
    for &m in passos {
        let r = eng.next_published().map_err(|e| format!("{e:?}")).and_then(|p| eng.enqueue_step(p, m).map_err(|e| format!("{e:?}")))
            .and_then(|_| eng.finish_token().map_err(|e| format!("{e:?}")));
        if r.is_err() {
            res = r;
            break;
        }
    }
    let st = if res.is_ok() { eng.next_published().map(|p| p.stats().clone()).unwrap_or_default() } else { Stats::default() };
    let syncs = gpu.sync_counts();
    let _ = eng.close();
    (st, gpu.violations(), gpu.stalled(), res, syncs)
}

/// A referência independente do agrupador: com a tabela congelada, cada pedido de chave no `prefill` é acerto e os outros são CPU.
fn esperado_a(prefill: &[u32]) -> Stats {
    let (mut hits, mut cpu) = (0u64, 0u64);
    for p in 0..TOTAL_A {
        for l in 0..2usize {
            for e in rota_a(p, l) {
                if prefill.contains(&((l * 4) as u32 + e)) { hits += 1 } else { cpu += 1 }
            }
        }
    }
    Stats { needed: hits + cpu, hits, cpu, ..Default::default() }
}

fn sched_a() -> SchedCfg {
    SchedCfg { seed: 0xA0A, weights: [1, 1, 1, 1], budget: 300_000 }
}

/// Oráculo A: passo contra posição a posição com a tabela congelada, M em {1, 2, 4, 8} (com bloco parcial no fim) e a sequência
/// 8, 1, 7, 2, 8, 8, 3, 1, 1: contadores exatos, sem violação (o conferidor do falso soma `peso · expert` por posição, por conta própria).
/// Com o override em lote e com o padrão por posição.
#[test]
fn oraculo_a_tabela_congelada_em_lote_e_por_posicao() {
    let prefill_variantes: [(Vec<u32>, usize); 3] = [(vec![], 0), (vec![0, 5], 2), (vec![0, 1, 2, 3, 4, 5, 6, 7], 8)];
    let mut seqs: Vec<Vec<usize>> = [1usize, 2, 4, 8].iter().map(|&m| passos_de(m)).collect();
    seqs.push(vec![8, 1, 7, 2, 8, 8, 3, 1, 1]);
    for (pre, slots) in &prefill_variantes {
        let want = esperado_a(pre);
        for passos in &seqs {
            assert_eq!(passos.iter().sum::<usize>(), TOTAL_A);
            let lote = roda_a(passos, pre.clone(), *slots, FakeCpu { weighted: true, ..Default::default() }, sched_a());
            let padrao = roda_a(passos, pre.clone(), *slots, PorPosicao(FakeCpu { weighted: true, ..Default::default() }), sched_a());
            for (nome, (st, viol, stalled, res)) in [("lote", lote), ("padrão", padrao)] {
                assert!(res.is_ok() && viol.is_empty() && !stalled, "{nome}, {passos:?}, prefill {pre:?}: {res:?} {viol:?}");
                // `dropped` conta uma admissão recusada por chave da união em cada passo: depende de M e é do oráculo B.
                assert_eq!(Stats { dropped: 0, ..st }, want, "{nome}, {passos:?}, prefill {pre:?}");
            }
        }
    }
}

/// O mutante de peso (o lote usa o peso da primeira posição do expert em todas) é acusado pelo conferidor com a posição e a camada; a
/// mesma execução com a CPU certa sai limpa. Sem acertos, todo expert repetido em posições com pesos diferentes expõe o erro.
#[test]
fn mutante_de_peso_na_cpu_em_lote_e_acusado() {
    let passos = [8usize, 8, 8, 8, 7];
    let certa = roda_a(&passos, vec![], 0, FakeCpu { weighted: true, ..Default::default() }, sched_a());
    assert!(certa.1.is_empty() && certa.3.is_ok());
    let errada = roda_a(&passos, vec![], 0, FakeCpu { weighted: true, peso_errado: true, ..Default::default() }, sched_a());
    assert!(errada.3.is_ok() && !errada.2);
    assert!(!errada.1.is_empty() && errada.1.iter().all(|v| matches!(v, Violation::Y { .. })), "{:?}", errada.1);
}

/// `fail_at` no `FakeCpu`: o lote conta UMA chamada por camada e o padrão por posição (que chama `compute` `m` vezes) conta uma por
/// posição. Passo de 8 posições e 2 camadas: por camada são as chamadas 0 e 1, então `fail_at = 2` não dispara no lote e dispara no
/// padrão (posição 2 da camada 0). Um lote que contasse por posição seria acusado aqui.
#[test]
fn falha_da_cpu_conta_uma_chamada_por_camada_no_lote_e_uma_por_posicao_no_padrao() {
    let f = || FakeCpu { weighted: true, fail_at: Some(2), ..Default::default() };
    let lote = roda_a(&[8], vec![], 0, f(), sched_a());
    assert!(lote.3.is_ok() && lote.1.is_empty() && !lote.2, "{:?}", lote);
    let padrao = roda_a(&[8], vec![], 0, PorPosicao(f()), sched_a());
    assert!(padrao.3.is_err() && !padrao.2, "{:?}", padrao);
    // E com `fail_at = 1` o lote cai na segunda camada.
    let g = || FakeCpu { weighted: true, fail_at: Some(1), ..Default::default() };
    assert!(roda_a(&[8], vec![], 0, g(), sched_a()).3.is_err());
}

/// Gerações cruzando 2^31 e 2^32 com passos de até 8 posições e tamanhos variáveis: `t` conta passos e a comparação cíclica (`geq`) segue
/// certa na volta. Contadores iguais aos de sem volta e sem violação.
#[test]
fn contador_de_geracoes_cruza_as_voltas_com_passos_de_tamanho_variavel() {
    let seqs: [Vec<usize>; 3] = [vec![8, 1, 7, 2, 8, 8, 3, 1, 1], passos_de(8), passos_de(3)];
    let pre = vec![0u32, 5];
    let base_want = esperado_a(&pre);
    for first in [(1u32 << 31) - 3, u32::MAX - 3, u32::MAX, 0x8000_0000] {
        for passos in &seqs {
            for w in [[1u32, 1, 1, 1], [8, 8, 1, 8]] {
                let sched = SchedCfg { seed: 0xC0DE ^ first as u64, weights: w, budget: 300_000 };
                let (st, viol, stalled, res, _) =
                    roda_a_t(passos, pre.clone(), 2, FakeCpu { weighted: true, ..Default::default() }, sched, first);
                assert!(res.is_ok() && viol.is_empty() && !stalled, "primeiro {first:#x}, {passos:?}: {res:?} {viol:?}");
                assert_eq!(Stats { dropped: 0, ..st }, base_want, "primeiro {first:#x}, {passos:?}");
            }
        }
    }
}

/// O contador de `sync` por papel e stream: quem dirige o motor (a thread do chamador, sem papel registrado) sincroniza a stream de
/// cálculo uma vez por passo (o `finish_token`), qualquer que seja o `m`, e nunca a de cópia antes de fechar; o serviço só sincroniza a
/// cópia, uma vez por passo mais a da partida. Passo de 8 não custa mais sincronizações que o de 1 por posição servida.
#[test]
fn sync_por_papel_e_stream_conta_uma_vez_por_passo() {
    let f = || FakeCpu { weighted: true, ..Default::default() };
    for passos in [vec![1usize; 8], vec![8], vec![8, 1, 7, 2, 8, 8, 3, 1, 1]] {
        let (_, viol, stalled, res, sy) = roda_a_t(&passos, vec![0], 2, f(), sched_a(), 1);
        assert!(res.is_ok() && viol.is_empty() && !stalled);
        let n = passos.len() as u64;
        assert_eq!(sy[2], [n, 0], "chamador: cálculo uma vez por passo, cópia nenhuma: {sy:?}");
        assert_eq!(sy[0], [0, 0], "a linha principal não sincroniza nada aqui: {sy:?}");
        assert_eq!(sy[1], [0, n + 1], "serviço: só a cópia, uma por passo e a da partida: {sy:?}");
    }
}

/// Semeaduras do escalonador com passos de `m` variável (M até 8): em 96 intercalações sorteadas, com a tabela congelada, os contadores
/// são os do oráculo A, sem violação nem travamento.
#[test]
fn semeaduras_do_escalonador_com_m_variavel() {
    let pre = vec![0u32, 3, 6];
    let want = esperado_a(&pre);
    let mut x = 0x9E37_79B9_7F4A_7C15u64;
    let mut rnd = move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    for i in 0..96 {
        // Tamanhos sorteados de 1 a 8 até somar 39 posições.
        let mut passos = Vec::new();
        let mut falta = TOTAL_A;
        while falta > 0 {
            let m = (1 + (rnd() % 8) as usize).min(falta);
            passos.push(m);
            falta -= m;
        }
        let w = [1 + (rnd() % 8) as u32, 1 + (rnd() % 8) as u32, 1 + (rnd() % 8) as u32, 1 + (rnd() % 8) as u32];
        let sched = SchedCfg { seed: rnd(), weights: w, budget: 300_000 };
        let (st, viol, stalled, res, _) = roda_a_t(&passos, pre.clone(), 3, FakeCpu { weighted: true, ..Default::default() }, sched, 1);
        assert!(res.is_ok() && viol.is_empty() && !stalled, "sorteio {i}, {passos:?}, {w:?}: {res:?} {viol:?}");
        assert_eq!(Stats { dropped: 0, ..st }, want, "sorteio {i}, {passos:?}");
    }
}
