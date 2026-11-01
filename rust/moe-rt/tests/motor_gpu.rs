//! A1.3a: as fixtures do simulador (`tools/a1_fixtures_sim.py`, protocolo `token` com barreira) rodam no `Engine` com o
//! `CudaBackend` e as duas linhas do host em threads de verdade. O roteamento do simulador é forçado por roteadores
//! construídos: a entrada do token `t` é o vetor com 1 na posição `t`, e o roteador da camada `l` dá ao expert que ocupa a
//! posição `p` da lista de `routing[t][l]` o logit `K - p` e zero aos demais. Os experts têm pesos zero, então o `h` não muda
//! e só os contadores contam (acertos, cópias, experts na CPU, recusas e cópias guardadas).
//!
//! Limite: o top-k do backend exige `K` potência de dois e `E` par (`GEMV_BM = 2`), então cabem as fixtures com `K` em
//! {1, 2} e `E` par (31 das 64).
use cutile::half::bf16;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream, Dims};
use moe_rt::motor::{AdmitMode, Config, Engine, Stats};
use std::sync::Arc;

const H: usize = 512;
const I: usize = 256;

struct Fixture {
    layers: usize,
    experts: usize,
    topk: usize,
    slots: usize,
    backlog: usize,
    routing: Vec<Vec<Vec<u32>>>,
    esperado: Stats,
}

fn fixtures() -> Vec<Fixture> {
    let txt = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/a1_sim_token_barreira.json"));
    let v: serde_json::Value = serde_json::from_str(txt).expect("fixture");
    let n = |x: &serde_json::Value| x.as_u64().expect("número");
    v["casos"]
        .as_array()
        .unwrap()
        .iter()
        .map(|k| {
            let e = &k["esperado"];
            Fixture {
                layers: n(&k["layers"]) as usize,
                experts: n(&k["experts"]) as usize,
                topk: n(&k["topk"]) as usize,
                slots: n(&k["slots"]) as usize,
                backlog: n(&k["backlog"]) as usize,
                routing: k["routing"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|t| t.as_array().unwrap().iter().map(|l| l.as_array().unwrap().iter().map(|x| n(x) as u32).collect()).collect())
                    .collect(),
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

/// Roda uma fixture no motor de CUDA e devolve os contadores lidos no `Published` do token seguinte ao último.
fn roda(device: &Arc<cuda_core::Device>, f: &Fixture) -> Stats {
    let tokens = f.routing.len();
    assert!(tokens <= H, "mais tokens que a largura");
    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    // Roteadores: [E, H], coluna t = logits do token t.
    let routers: Vec<Vec<bf16>> = (0..f.layers)
        .map(|l| {
            let mut r = vec![bf16::from_f32(0.0); f.experts * H];
            for (t, camadas) in f.routing.iter().enumerate() {
                for (p, &e) in camadas[l].iter().enumerate() {
                    r[e as usize * H + t] = bf16::from_f32((f.topk - p) as f32);
                }
            }
            r
        })
        .collect();
    let zero = (vec![bf16::from_f32(0.0); 2 * I * H], vec![bf16::from_f32(0.0); H * I]);
    let ex: Arc<Vec<(Vec<bf16>, Vec<bf16>)>> = Arc::new((0..f.layers * f.experts).map(|_| zero.clone()).collect());
    let dims = Dims { layers: f.layers, experts: f.experts, hidden: H, inter: I, topk: f.topk, slots: f.slots };
    let b = CudaBackend::new(device.clone(), dims, &routers, &compute).unwrap();
    let cfg = Config {
        layers: f.layers,
        experts: f.experts,
        topk: f.topk,
        width: H,
        slots: f.slots,
        backlog: f.backlog,
        admit: AdmitMode::TokenBarrier,
        first_token: 1,
        prefill: vec![],
    };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
    let cpu = CpuServe { cpu: Cpu::new(2), hidden: H, inter: I, experts: f.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
    for t in 0..=tokens {
        if t < tokens {
            let mut x = vec![0f32; H];
            x[t] = 1.0;
            b.set_input(&compute, &x).unwrap();
        }
        let p = eng.next_published().unwrap();
        if t == tokens {
            let s = p.stats().clone();
            eng.close().unwrap();
            return s;
        }
        eng.run_token(p).unwrap();
        let got = b.read_output(&compute).unwrap();
        assert!(got.iter().enumerate().all(|(j, v)| *v == if j == t { 1.0 } else { 0.0 }), "token {t}: h mudou com experts de peso zero");
    }
    unreachable!()
}

/// Os testes deste arquivo dividem a GPU e o contexto: um por vez (o mesmo cuidado de `crate::gpu_test_lock`).
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn gpu() -> Option<(std::sync::MutexGuard<'static, ()>, Arc<cuda_core::Device>)> {
    let trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    gpu_sem_trava().map(|d| (trava, d))
}

fn gpu_sem_trava() -> Option<Arc<cuda_core::Device>> {
    match cuda_core::Device::new(0) {
        Ok(d) => {
            d.bind_to_thread().unwrap();
            Some(d)
        }
        Err(e) => {
            assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
            None
        }
    }
}

fn uteis() -> Vec<(usize, Fixture)> {
    fixtures().into_iter().enumerate().filter(|(_, f)| f.topk.is_power_of_two() && f.experts % 2 == 0).collect()
}

/// Controle negativo: o roteamento que chega ao motor é mesmo o dos roteadores construídos. Com o expert deslocado pelo
/// índice do token (módulo `E`; um deslocamento igual para todos só renomearia os experts e deixaria os contadores iguais),
/// os contadores deixam de bater com os da fixture original na maioria; se o teste principal passasse por outro motivo (por
/// exemplo, um roteamento que não depende do roteador), este controle não acusaria nada.
#[test]
fn controle_roteamento_deslocado_e_acusado() {
    let Some((_trava, device)) = gpu() else { return };
    let mut acusadas = 0;
    let mut total = 0;
    for (_, f) in uteis() {
        let e = f.experts as u32;
        let rodada = Fixture { routing: f.routing.iter().enumerate().map(|(i, t)| t.iter().map(|l| l.iter().map(|x| (x + i as u32) % e).collect()).collect()).collect(), ..f };
        let s = roda(&device, &rodada);
        total += 1;
        if s != rodada.esperado {
            acusadas += 1;
        }
    }
    eprintln!("deslocado: {acusadas} de {total} fixtures acusadas");
    assert!(acusadas * 2 >= total, "o roteamento deslocado devia mudar os contadores: {acusadas} de {total}");
}

#[test]
fn token_barrier_no_cuda_bate_com_o_simulador() {
    let Some((_trava, device)) = gpu() else { return };
    let uteis = uteis();
    assert_eq!(uteis.len(), 31);
    let mut tot = Stats::default();
    for (i, f) in &uteis {
        let s = roda(&device, f);
        assert_eq!(s, f.esperado, "fixture {i} (L{} E{} K{} S{} backlog {})", f.layers, f.experts, f.topk, f.slots, f.backlog);
        tot.hits += s.hits;
        tot.copies += s.copies;
        tot.cpu += s.cpu;
        tot.deferred += s.deferred;
        tot.dropped += s.dropped;
    }
    eprintln!("31 fixtures no CUDA iguais ao simulador: {tot:?}");
    assert!(tot.hits > 0 && tot.copies > 0 && tot.cpu > 0 && tot.deferred > 0, "as fixtures precisam exercitar acertos, cópias, CPU e cópias guardadas: {tot:?}");
}
