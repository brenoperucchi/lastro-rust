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

/// Com áreas de várias posições a linha de serviço (U1) ainda lê a capacidade inteira: qualquer passo, até m = 1, é recusado e derruba o
/// motor sem travar. Fora da faixa (0 e acima do máximo) também. Hoje isto vale para todo m; a U2 troca a recusa pelo fatiamento.
#[test]
fn qualquer_m_com_areas_de_varias_posicoes_derruba_o_motor_sem_travar() {
    for m in [0usize, 1, 2, MP + 1] {
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
