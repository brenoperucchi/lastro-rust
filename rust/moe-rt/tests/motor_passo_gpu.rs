//! A2-PASSO E4b: o oráculo do passo de várias posições no backend de CUDA. NÃO RODA SEM AUTORIZAÇÃO.
//!
//! Este arquivo é o harness do E4b escrito antes da janela de GPU (critério em
//! `qual-runs/A2-LOTE-20261103/criterio-e4b.txt`, portão de correção, seção 3). Os testes de GPU são `#[ignore]` e, mesmo com
//! `--ignored`, recusam rodar sem `E4B_GPU_AUTORIZADO=1` na variável de ambiente: só o "autorizo" explícito do Breno, depois da
//! aprovação do critério e da revisão do harness, abre uma janela. Sem o "autorizo" a suíte só compila.
//!
//! O que cada teste confere (tolerância de 1e-5 relativo à norma máxima do vetor entre o passo de M posições e o token a token do
//! MESMO motor; 1e-4 contra a passada de referência da CPU, como nos testes do A1.3c):
//! - os logits da última posição de cada bloco (M = 8, 3 e 1, blocos parciais);
//! - o KV, por efeito: um bloco seguinte lê o KV gravado pelo bloco anterior, então logits errados depois do primeiro bloco acusam
//!   o KV do passo (a leitura direta do cache fica fora do harness para não mexer na parte densa além do passo).
mod common;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream};
use moe_rt::model::CpuSession;
use moe_rt::motor::{AdmitMode, Config, Engine};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn exige_autorizacao() {
    assert_eq!(
        std::env::var("E4B_GPU_AUTORIZADO").as_deref(),
        Ok("1"),
        "o E4b na GPU só roda com o 'autorizo' explícito do Breno (E4B_GPU_AUTORIZADO=1); veja o critério em criterio-e4b.txt"
    );
}

fn erro_rel(got: &[f32], want: &[f32]) -> f32 {
    if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
        return f32::INFINITY;
    }
    let m = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
    got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / m
}

/// Roda os blocos (cada um é um passo de `bloco.len()` posições) e devolve os logits da última posição de cada bloco.
fn roda_blocos(blocos: &[Vec<usize>], slots: usize, admit: AdmitMode) -> Vec<Vec<f32>> {
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = cuda_core::Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let (_scratch, m) = common::tiny_model("motor_passo");
    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    let max_positions = 8;
    let (b, ex) = CudaBackend::new_model_positions(device.clone(), &m, slots, 64, &compute, max_positions).unwrap();
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
        max_positions,
    };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
    let cpu = CpuServe { cpu: Cpu::new(2), hidden: m.cfg.hidden, inter: m.cfg.inter, experts: m.cfg.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
    let mut out = Vec::new();
    for bloco in blocos {
        b.begin_step_tokens(&compute, bloco).unwrap();
        let p = eng.next_published().unwrap();
        eng.enqueue_step(p, bloco.len()).unwrap();
        eng.finish_token().unwrap();
        out.push(b.read_logits(&compute).unwrap());
    }
    eng.close().unwrap();
    out
}

const TOKENS: [usize; 14] = [3, 17, 5, 17, 3, 29, 5, 8, 17, 3, 5, 29, 8, 17];

/// O passo de M posições dá os mesmos logits da última posição que o token a token do mesmo motor (1e-5) e que a CPU de referência
/// (1e-4), nos dois modos de admissão e com poucos e muitos slots.
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn passo_de_oito_bate_com_o_token_a_token_na_gpu() {
    exige_autorizacao();
    let (_s, m) = common::tiny_model("motor_passo_ref");
    let cpu_ref = Cpu::new(2);
    let mut sess = CpuSession::new(&m, &cpu_ref);
    let ref_logits: Vec<Vec<f32>> = TOKENS.iter().map(|&t| sess.step(t).0).collect();
    // Blocos: 8, 3, 1, 2 (parcial) = 14 posições.
    let cortes = [8usize, 3, 1, 2];
    assert_eq!(cortes.iter().sum::<usize>(), TOKENS.len());
    for slots in [2usize, 5, 8] {
        for admit in [AdmitMode::Event, AdmitMode::TokenBarrier] {
            let mut blocos = Vec::new();
            let mut ult = Vec::new();
            let mut i = 0;
            for &c in &cortes {
                blocos.push(TOKENS[i..i + c].to_vec());
                i += c;
                ult.push(i - 1);
            }
            let passo = roda_blocos(&blocos, slots, admit);
            let uns: Vec<Vec<usize>> = TOKENS.iter().map(|&t| vec![t]).collect();
            let tok = roda_blocos(&uns, slots, admit);
            for (k, &idx) in ult.iter().enumerate() {
                let e1 = erro_rel(&passo[k], &tok[idx]);
                let e2 = erro_rel(&passo[k], &ref_logits[idx]);
                assert!(e1 <= 1e-5, "slots {slots} {admit:?}, bloco {k} (posição {idx}): passo contra token a token {e1:e}");
                assert!(e2 <= 1e-4, "slots {slots} {admit:?}, bloco {k} (posição {idx}): passo contra a CPU {e2:e}");
            }
        }
    }
}

/// Um `m` que não bate com os tokens preparados é recusado, sem travar o motor (erro do `begin_step`).
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn m_diferente_dos_tokens_preparados_e_recusado() {
    exige_autorizacao();
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = cuda_core::Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let (_scratch, m) = common::tiny_model("motor_passo_m");
    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    let (b, ex) = CudaBackend::new_model_positions(device.clone(), &m, 5, 64, &compute, 8).unwrap();
    let cfg = Config { layers: m.cfg.layers, experts: m.cfg.experts, topk: m.cfg.k, width: m.cfg.hidden, slots: 5, backlog: 4, admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions: 8 };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
    let cpu = CpuServe { cpu: Cpu::new(2), hidden: m.cfg.hidden, inter: m.cfg.inter, experts: m.cfg.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
    b.begin_step_tokens(&compute, &TOKENS[..4]).unwrap();
    let p = eng.next_published().unwrap();
    assert!(eng.enqueue_step(p, 3).is_err(), "m = 3 com 4 tokens preparados tem de ser recusado");
    // O erro derrubou o motor; fechar não pode travar (o resultado do fechamento não importa aqui).
    let _ = eng.close();
}

/// O portão: sem `E4B_GPU_AUTORIZADO=1` os testes de GPU recusam antes de tocar o device. Este teste não usa GPU e roda sempre.
#[test]
fn sem_autorizacao_o_portao_recusa_antes_de_tocar_a_gpu() {
    if std::env::var("E4B_GPU_AUTORIZADO").as_deref() == Ok("1") {
        return;
    }
    let r = std::panic::catch_unwind(exige_autorizacao);
    assert!(r.is_err(), "o portão tinha de recusar sem a autorização");
}
