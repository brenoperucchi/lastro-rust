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
//! - o KV de todas as camadas e posições, lido direto do cache (`read_kv`), passo contra token a token (1e-5): a fixture é quase
//!   insensível ao contexto nos logits, então o KV não pode ser conferido por efeito;
//! - a transição prefill em lote e depois decode de um token por `begin_token` + `run_token` no mesmo motor.
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
/// Os logits da última posição de cada bloco e o KV (K, V) de cada camada, todas as posições.
type Saida = (Vec<Vec<f32>>, Vec<(Vec<f32>, Vec<f32>)>);

fn roda_blocos(blocos: &[Vec<usize>], slots: usize, admit: AdmitMode, plana: bool) -> Saida {
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    roda_blocos_sem_trava(blocos, slots, admit, plana)
}

/// O corpo de `roda_blocos` para quem já segura `GPU` (o teste do mutante segura a trava nas duas execuções).
fn roda_blocos_sem_trava(blocos: &[Vec<usize>], slots: usize, admit: AdmitMode, plana: bool) -> Saida {
    let device = cuda_core::Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let (_scratch, m) = modelo_com("motor_passo", plana);
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
    let total: usize = blocos.iter().map(|b| b.len()).sum();
    let kv = (0..m.cfg.layers).map(|l| b.read_kv(&compute, l, total).unwrap()).collect();
    eng.close().unwrap();
    (out, kv)
}

/// O modelo do harness: a fixture do A1.3c com a atenção quase plana (`attention_multiplier` 1e-5) e pesos de atenção 0,3. Com a
/// configuração de sempre (0,125 e 0,1) o contexto não muda os logits nem o KV da camada 1, e o oráculo não teria sensibilidade à
/// causalidade. Sondagem da rodada 71 (CpuSession, contextos que diferem nos 8 primeiros tokens): o KV da camada 0 não depende do
/// contexto (só do token e da posição), o da camada 1 muda de 1e-4 a 2e-4 relativo nas posições 9 a 13, contra a tolerância de 1e-5
/// do oráculo (margem de uns 10 vezes, pequena: veja `o_contexto_muda_o_kv_da_camada_1`).
fn modelo(name: &str) -> (common::Scratch, moe_rt::model::HostModel) {
    modelo_com(name, true)
}

/// `plana = false` é a fixture de sempre (atenção pontiaguda), que cobre o caminho do Q e do score q·k que a plana dilui.
fn modelo_com(name: &str, plana: bool) -> (common::Scratch, moe_rt::model::HostModel) {
    if !plana {
        return common::tiny_model(name);
    }
    let mut c = common::tiny_config(33);
    c["attention_multiplier"] = serde_json::json!(1e-5);
    common::tiny_model_attn(name, c, 0.3)
}

const CORTES: [usize; 4] = [8, 3, 1, 2];

const TOKENS: [usize; 14] = [3, 17, 5, 17, 3, 29, 5, 8, 17, 3, 5, 29, 8, 17];

/// O passo de M posições dá os mesmos logits da última posição que o token a token do mesmo motor (1e-5) e que a CPU de referência
/// (1e-4), nos dois modos de admissão e com poucos e muitos slots.
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn passo_de_oito_bate_com_o_token_a_token_na_gpu() {
    exige_autorizacao();
    for plana in [true, false] {
        let (_s, m) = modelo_com("motor_passo_ref", plana);
        let cpu_ref = Cpu::new(2);
        let mut sess = CpuSession::new(&m, &cpu_ref);
        let ref_logits: Vec<Vec<f32>> = TOKENS.iter().map(|&t| sess.step(t).0).collect();
        // Blocos: 8, 3, 1, 2 (parcial) = 14 posições.
        let cortes = CORTES;
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
                let (passo, kv_passo) = roda_blocos(&blocos, slots, admit, plana);
                let uns: Vec<Vec<usize>> = TOKENS.iter().map(|&t| vec![t]).collect();
                let (tok, kv_tok) = roda_blocos(&uns, slots, admit, plana);
                for (l, ((kp, vp), (kt, vt))) in kv_passo.iter().zip(&kv_tok).enumerate() {
                    let (ek, ev) = (erro_rel(kp, kt), erro_rel(vp, vt));
                    assert!(ek <= 1e-5 && ev <= 1e-5, "plana {plana}, slots {slots} {admit:?}, camada {l}: KV do passo contra token a token K {ek:e} V {ev:e}");
                }
                for (k, &idx) in ult.iter().enumerate() {
                    let e1 = erro_rel(&passo[k], &tok[idx]);
                    let e2 = erro_rel(&passo[k], &ref_logits[idx]);
                    assert!(e1 <= 1e-5, "plana {plana}, slots {slots} {admit:?}, bloco {k} (posição {idx}): passo contra token a token {e1:e}");
                    assert!(e2 <= 1e-4, "plana {plana}, slots {slots} {admit:?}, bloco {k} (posição {idx}): passo contra a CPU {e2:e}");
                }
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
    let (_scratch, m) = modelo("motor_passo_m");
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

/// Prefill em lote (8 tokens por `begin_step_tokens`) e depois decode de dois tokens por `begin_token` + `run_token` no mesmo motor
/// com teto de 8 posições: o `begin_token` tem de registrar o token preparado, senão o `begin_step` recusa (regressão da rodada 70).
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn prefill_em_lote_e_depois_decode_por_begin_token() {
    exige_autorizacao();
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = cuda_core::Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let (_scratch, m) = modelo("motor_passo_dec");
    let cpu_ref = Cpu::new(2);
    let mut sess = CpuSession::new(&m, &cpu_ref);
    let want: Vec<Vec<f32>> = TOKENS[..10].iter().map(|&t| sess.step(t).0).collect();
    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    let (b, ex) = CudaBackend::new_model_positions(device.clone(), &m, 5, 64, &compute, 8).unwrap();
    let cfg = Config { layers: m.cfg.layers, experts: m.cfg.experts, topk: m.cfg.k, width: m.cfg.hidden, slots: 5, backlog: 4, admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions: 8 };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
    let cpu = CpuServe { cpu: Cpu::new(2), hidden: m.cfg.hidden, inter: m.cfg.inter, experts: m.cfg.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
    b.begin_step_tokens(&compute, &TOKENS[..8]).unwrap();
    let p = eng.next_published().unwrap();
    eng.enqueue_step(p, 8).unwrap();
    eng.finish_token().unwrap();
    let got = b.read_logits(&compute).unwrap();
    assert!(erro_rel(&got, &want[7]) <= 1e-4, "prefill em lote contra a CPU");
    for i in 8..10 {
        b.begin_token(&compute, TOKENS[i]).unwrap();
        let p = eng.next_published().unwrap();
        eng.run_token(p).unwrap();
        let got = b.read_logits(&compute).unwrap();
        assert!(erro_rel(&got, &want[i]) <= 1e-4, "decode {i} depois do prefill em lote");
    }
    eng.close().unwrap();
}

/// Controle de sensibilidade (só CPU, roda sempre): contextos que diferem só nos 8 primeiros tokens (mesmo último token de cada
/// posição) mudam o K e o V da camada 1 acima de 5e-5 relativo nas posições 9 e 13, 5 vezes a tolerância de 1e-5 do oráculo do KV.
/// O KV da camada 0 não depende do contexto e fica igual. Sem isso o oráculo da GPU não acusaria um `lens` errado no passo.
#[test]
fn o_contexto_muda_o_kv_da_camada_1() {
    let (_s, m) = modelo("motor_passo_ctx");
    let cpu = Cpu::new(2);
    let mut outro = TOKENS.to_vec();
    for t in outro.iter_mut().take(8) {
        *t = (*t + 1) % 33;
    }
    let (mut s1, mut s2) = (CpuSession::new(&m, &cpu), CpuSession::new(&m, &cpu));
    TOKENS.iter().for_each(|&t| {
        s1.step(t);
    });
    outro.iter().for_each(|&t| {
        s2.step(t);
    });
    let kvd = m.cfg.kv_heads * m.cfg.head_dim;
    let rel = |a: &[f32], b: &[f32], p: usize| erro_rel(&a[p * kvd..(p + 1) * kvd], &b[p * kvd..(p + 1) * kvd]);
    for p in [9usize, 13] {
        let ((k1, v1), (k2, v2)) = (s1.kv(1), s2.kv(1));
        let (ek, ev) = (rel(k2, k1, p), rel(v2, v1, p));
        assert!(ek > 5e-5 && ev > 5e-5, "camada 1, posição {p}: o contexto quase não muda o KV (K {ek:e}, V {ev:e}); o oráculo não teria sensibilidade");
    }
    let ((k1, _), (k2, _)) = (s1.kv(0), s2.kv(0));
    assert_eq!(rel(k2, k1, 13), 0.0, "o KV da camada 0 de uma posição só depende do token e da posição");
}

/// O mutante de causalidade (cada posição lê o comprimento da última do passo): o oráculo tem de acusar no KV da camada 1.
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn mutante_de_lens_e_acusado_pelo_oraculo_do_kv() {
    exige_autorizacao();
    /// Liga o mutante e o desliga no `Drop`, também em unwind; só existe com a trava `GPU` seguida.
    struct Mutante;
    impl Mutante {
        fn liga() -> Self {
            CudaBackend::mutante_lens_da_ultima(true);
            Mutante
        }
    }
    impl Drop for Mutante {
        fn drop(&mut self) {
            CudaBackend::mutante_lens_da_ultima(false);
        }
    }
    // A trava vale da referência até o mutante desligar: nenhum outro teste de GPU roda com a flag ligada.
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let blocos = vec![TOKENS[..8].to_vec(), TOKENS[8..11].to_vec()];
    let uns: Vec<Vec<usize>> = TOKENS[..11].iter().map(|&t| vec![t]).collect();
    let (_, kv_tok) = roda_blocos_sem_trava(&uns, 5, AdmitMode::TokenBarrier, true);
    let kv_mut = {
        let _m = Mutante::liga();
        roda_blocos_sem_trava(&blocos, 5, AdmitMode::TokenBarrier, true).1
    };
    let (ek, ev) = (erro_rel(&kv_mut[1].0, &kv_tok[1].0), erro_rel(&kv_mut[1].1, &kv_tok[1].1));
    assert!(ek > 1e-5 || ev > 1e-5, "o mutante de lens passou pelo oráculo do KV (K {ek:e}, V {ev:e})");
}

/// Uma leitura do KV que falha não pode trocar o cache da sequência por zeros: depois do erro, uma nova leitura devolve o mesmo KV.
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn leitura_do_kv_que_falha_preserva_o_cache() {
    exige_autorizacao();
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = cuda_core::Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let (_scratch, m) = modelo("motor_passo_kv");
    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    let (b, ex) = CudaBackend::new_model_positions(device.clone(), &m, 5, 64, &compute, 8).unwrap();
    let cfg = Config { layers: m.cfg.layers, experts: m.cfg.experts, topk: m.cfg.k, width: m.cfg.hidden, slots: 5, backlog: 4, admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions: 8 };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
    let cpu = CpuServe { cpu: Cpu::new(2), hidden: m.cfg.hidden, inter: m.cfg.inter, experts: m.cfg.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
    b.begin_step_tokens(&compute, &TOKENS[..4]).unwrap();
    let p = eng.next_published().unwrap();
    eng.enqueue_step(p, 4).unwrap();
    eng.finish_token().unwrap();
    let antes = b.read_kv(&compute, 1, 4).unwrap();
    assert!(antes.0.iter().any(|v| *v != 0.0));
    CudaBackend::injeta_falha_na_leitura_do_kv(true);
    let r = b.read_kv(&compute, 1, 4);
    CudaBackend::injeta_falha_na_leitura_do_kv(false);
    assert!(r.is_err());
    assert_eq!(b.read_kv(&compute, 1, 4).unwrap(), antes, "o cache mudou depois de uma leitura que falhou");
    eng.close().unwrap();
}
