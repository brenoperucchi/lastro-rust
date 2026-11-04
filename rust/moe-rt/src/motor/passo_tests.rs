//! A2-PASSO E4a, U1 (contrato-e4a.md v2): as formas do passo de M posições, sem a linha de serviço (que só aprende M > 1 na U2).
//! Layout da área da camada, `begin_step` em ordem FIFO no backend falso, `Route` por posição global, validação de `m` e das peças.
//! Com `max_positions == 1` o motor é o de sempre: a suíte de `tests.rs` roda sem alteração de asserção.
use super::backend::Backend;
use super::fake::{expert_out, x_value, FakeCfg, FakeCpu, FakeGpu, Violation};
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
                let mut y = r[p][0].iter().fold(0u32, |a, &e| a.wrapping_add(expert_out(e, x)));
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
    let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu::default()).expect("motor");
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
