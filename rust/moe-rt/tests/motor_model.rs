//! A1.3c: o modelo inteiro (embedding, atenção, blocos MoE pelo D1 e logits do vocabulário) no motor de CUDA, contra a passada
//! de referência na CPU (`CpuSession`), com tolerância de 1e-4 nos logits. O Granite MoE sintético de `tests/common`
//! (H = 256, 2 camadas, 4 cabeças de Q e 2 de KV, 4 experts com 2 ativos) tem 8 chaves de expert; o cache tem menos slots, então
//! os experts se dividem entre a GPU (acertos) e a CPU (faltas) durante a sequência.
mod common;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream};
use moe_rt::model::CpuSession;
use moe_rt::motor::{AdmitMode, Config, Engine, Stats};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn erro_rel(got: &[f32], want: &[f32]) -> f32 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f32::INFINITY;
    }
    let m = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / m
}

/// Roda `tokens` pelo motor e pela CPU; devolve o pior erro relativo dos logits e os contadores até o fim.
fn roda(tokens: &[usize], slots: usize, admit: AdmitMode) -> Option<(f32, Stats)> {
    roda_com_falha(tokens, slots, admit, None)
}

/// `falha_em`: no token com esse índice a cópia dos logits falha (injetada); os logits desse token se perdem, mas os seguintes
/// têm de voltar a bater com a CPU.
fn roda_com_falha(tokens: &[usize], slots: usize, admit: AdmitMode, falha_em: Option<usize>) -> Option<(f32, Stats)> {
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = match cuda_core::Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
            return None;
        }
    };
    device.bind_to_thread().unwrap();
    let (_scratch, m) = common::tiny_model("motor_model");
    let cpu_ref = Cpu::new(2);
    let mut sess = CpuSession::new(&m, &cpu_ref);
    let want: Vec<Vec<f32>> = tokens.iter().map(|&t| sess.step(t).0).collect();

    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    let (b, ex) = CudaBackend::new_model(device.clone(), &m, slots, 64, &compute).unwrap();
    let cfg = Config {
        layers: m.cfg.layers,
        experts: m.cfg.experts,
        topk: m.cfg.k,
        width: m.cfg.hidden,
        slots,
        backlog: 4,
        admit,
        first_token: 1,
        prefill: vec![],
        max_positions: 1,
    };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
    let cpu = CpuServe { cpu: Cpu::new(2), hidden: m.cfg.hidden, inter: m.cfg.inter, experts: m.cfg.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
    let mut pior = 0f32;
    let mut stats = Stats::default();
    for (i, &t) in tokens.iter().enumerate() {
        b.begin_token(&compute, t).unwrap();
        let p = eng.next_published().unwrap();
        stats = p.stats().clone();
        eng.run_token(p).unwrap();
        if falha_em == Some(i) {
            CudaBackend::injeta_falha_na_copia_dos_logits(true);
            let r = b.read_logits(&compute);
            CudaBackend::injeta_falha_na_copia_dos_logits(false);
            assert!(r.is_err(), "a falha injetada devia voltar como erro");
            continue;
        }
        let got = b.read_logits(&compute).unwrap();
        let err = erro_rel(&got, &want[i]);
        pior = pior.max(err);
        assert!(err <= 1e-4, "token {i} ({t}): erro relativo dos logits {err:e}");
    }
    eng.close().unwrap();
    Some((pior, stats))
}

const TOKENS: [usize; 14] = [3, 17, 5, 17, 3, 29, 5, 8, 17, 3, 5, 29, 8, 17];

/// Três orçamentos de slots (2, 5 e 8 de 8 chaves) nos dois modos de admissão: o resultado é o mesmo da CPU em todos. Com 2
/// slots a CPU atende a maior parte; com 8 todos os experts acabam residentes.
#[test]
fn modelo_inteiro_pelo_motor_bate_com_a_cpu_a_1e4() {
    for slots in [2usize, 5, 8] {
        for admit in [AdmitMode::Event, AdmitMode::TokenBarrier] {
            let Some((pior, stats)) = roda(&TOKENS, slots, admit) else { return };
            eprintln!("slots {slots} {admit:?}: pior erro relativo dos logits {pior:e}; contadores: {stats:?}");
            assert!(stats.cpu > 0, "o primeiro token tem de usar a CPU: {stats:?}");
            assert!(stats.hits > 0, "o teste precisa exercitar a GPU: {stats:?}");
        }
    }
}

/// Regressão: um erro na cópia dos logits não pode deixar o estado com um tensor de logits de forma errada (antes era de 1
/// elemento); os tokens seguintes voltam a bater com a CPU a 1e-4.
#[test]
fn erro_na_copia_dos_logits_nao_estraga_os_tokens_seguintes() {
    let Some((pior, _)) = roda_com_falha(&TOKENS[..6], 5, AdmitMode::TokenBarrier, Some(2)) else { return };
    eprintln!("depois da falha injetada: pior erro relativo {pior:e}");
}

/// A1.4b: `reset_sequence` começa uma conversa nova (posição 0) sem mexer no cache de experts: a mesma sequência passada duas vezes
/// dá os mesmos logits a 1e-4 da CPU nas duas.
#[test]
fn reset_da_sequencia_repete_os_logits_da_cpu() {
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = match cuda_core::Device::new(0) {
        Ok(d) => d,
        Err(e) => {
            assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
            return;
        }
    };
    device.bind_to_thread().unwrap();
    let (_scratch, m) = common::tiny_model("motor_reset");
    let cpu_ref = Cpu::new(2);
    let seq = [3usize, 17, 5, 29, 8];
    let mut sess = CpuSession::new(&m, &cpu_ref);
    let want: Vec<Vec<f32>> = seq.iter().map(|&t| sess.step(t).0).collect();
    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    let (b, ex) = CudaBackend::new_model(device.clone(), &m, 5, 64, &compute).unwrap();
    let cfg = Config { layers: m.cfg.layers, experts: m.cfg.experts, topk: m.cfg.k, width: m.cfg.hidden, slots: 5, backlog: 4, admit: AdmitMode::Event, first_token: 1, prefill: vec![], max_positions: 1 };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
    let cpu = CpuServe { cpu: Cpu::new(2), hidden: m.cfg.hidden, inter: m.cfg.inter, experts: m.cfg.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
    for volta in 0..2 {
        b.reset_sequence().unwrap();
        // O modelo sintético tem logits quase insensíveis ao contexto (a CPU dá os mesmos bits com outro contexto antes), então a
        // posição é conferida direto: um `reset` que não faz nada deixaria `posicao` em 5 na segunda volta.
        assert_eq!(b.sequence_position().unwrap(), 0, "volta {volta}: o reset tem de zerar a posição");
        for (i, &t) in seq.iter().enumerate() {
            b.begin_token(&compute, t).unwrap();
            let p = eng.next_published().unwrap();
            eng.run_token(p).unwrap();
            let got = b.read_logits(&compute).unwrap();
            let err = erro_rel(&got, &want[i]);
            assert!(err <= 1e-4, "volta {volta}, token {i}: erro relativo {err:e}");
        }
        assert_eq!(b.sequence_position().unwrap(), seq.len(), "volta {volta}: cada token avança a posição");
    }
    eng.close().unwrap();
}
