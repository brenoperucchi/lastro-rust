//! A2, plano de desempenho das cópias, O1 com política de erro A (decisão do Breno, 11/11 15:23): o teste de GPU do caminho de erro de
//! `attn_block_b`. NÃO RODA SEM AUTORIZAÇÃO e NÃO FOI EXECUTADO: está escrito e compila; só uma janela autorizada que o nomeie o roda.
//!
//! Os testes são `#[ignore]` e, mesmo com `--ignored`, recusam rodar sem `E4B_GPU_AUTORIZADO=1` (o padrão de `motor_passo_gpu.rs`).
//!
//! O que confere, com injetores no padrão de `injeta_falha_na_copia_dos_logits`:
//! - falha em cada um dos 4 lançamentos reformados (RoPE de Q, RoPE de K, atenção, gemv de `o`) e na IDA de cada uma das 4 reformas:
//!   o erro propaga e o `Dense` segue utilizável. Depois do erro, `reset_sequence` e o mesmo token dão logits IGUAIS BIT A BIT aos de uma
//!   execução limpa;
//! - O3 (kernels que gravam direto no cache): falha no lançamento do gemv de V (1), do RoPE de K depois de V ter gravado a linha `cur` (2) e
//!   `cur` forçado para fora do cache (3, recusa no host): mesmo critério, o mesmo token depois de `reset_sequence` dá logits iguais bit a bit;
//! - falha na VOLTA de cada reforma (restauração): o dono é mantido e a stream é esperada antes do descarte; o campo fica vazio e o uso
//!   seguinte recusa com erro, sem tocar memória liberada.
mod common;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream};
use moe_rt::motor::{AdmitMode, Backend, Config, Engine};

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn exige_autorizacao() {
    assert_eq!(
        std::env::var("E4B_GPU_AUTORIZADO").as_deref(),
        Ok("1"),
        "o teste de erro da O1 na GPU só roda com o 'autorizo' explícito do Breno (E4B_GPU_AUTORIZADO=1)"
    );
}

#[derive(Clone, Copy, Debug)]
enum Injeta {
    /// Falha do lançamento reformado de número 1 a 4.
    Lancamento(usize),
    /// Falha da n-ésima reforma (1 a 8; ímpares são a ida, pares a volta).
    Reforma(usize),
    /// O3: 1 falha o lançamento do gemv de V, 2 o do RoPE de K (depois de V ter gravado a linha `cur`), 3 força `cur` para fora do cache.
    O3(usize),
}

struct Saida {
    /// A primeira chamada de `pre_layer` com a injeção ligada devolveu erro.
    primeiro_erro: bool,
    /// Uma segunda chamada (injeção desligada) na camada 0 também devolveu erro (campo vazio depois de restauração que falhou).
    segundo_erro: bool,
    /// Logits do token depois de `reset_sequence`, só quando o motor seguiu utilizável.
    logits: Option<Vec<f32>>,
}

const TOKEN: usize = 5;

/// Um motor novo; com `injeta`, o token passa antes por uma tentativa de `pre_layer` na camada 0 com a falha ligada.
fn roda(injeta: Option<Injeta>) -> Saida {
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = cuda_core::Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let (_scratch, m) = common::tiny_model("motor_reforma");
    let slots = 5;
    let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
    let (b, ex) = CudaBackend::new_model(device.clone(), &m, slots, 64, &compute).unwrap();
    let (mut primeiro_erro, mut segundo_erro, mut motor_ok) = (false, false, true);
    if let Some(inj) = injeta {
        b.begin_token(&compute, TOKEN).unwrap();
        match inj {
            Injeta::Lancamento(n) => CudaBackend::injeta_falha_no_lancamento_reformado(n),
            Injeta::Reforma(n) => CudaBackend::injeta_falha_na_reforma(n),
            Injeta::O3(n) => CudaBackend::injeta_falha_no_lancamento_o3(n),
        }
        primeiro_erro = Backend::pre_layer(&*b, &compute, 0).is_err();
        CudaBackend::injeta_falha_no_lancamento_reformado(0);
        CudaBackend::injeta_falha_na_reforma(0);
        CudaBackend::injeta_falha_no_lancamento_o3(0);
        // espera a stream antes de qualquer outra coisa (o resultado da leitura não importa)
        let _ = b.read_logits(&compute);
        if let Injeta::Reforma(n) = inj {
            if n % 2 == 0 {
                // restauração falhou: o campo ficou vazio; o uso seguinte tem de recusar com erro
                segundo_erro = Backend::pre_layer(&*b, &compute, 0).is_err();
                let _ = b.read_logits(&compute);
                motor_ok = false;
            }
        }
        if motor_ok {
            b.reset_sequence().unwrap();
        }
    }
    let logits = if motor_ok {
        let cfg = Config { layers: m.cfg.layers, experts: m.cfg.experts, topk: m.cfg.k, width: m.cfg.hidden, slots, backlog: 4, admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions: 1 };
        let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
        let cpu = CpuServe { cpu: Cpu::new(2), hidden: m.cfg.hidden, inter: m.cfg.inter, experts: m.cfg.experts, weights: ex.clone() };
        let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
        b.begin_token(&compute, TOKEN).unwrap();
        let p = eng.next_published().unwrap();
        eng.run_token(p).unwrap();
        let l = b.read_logits(&compute).unwrap();
        eng.close().unwrap();
        Some(l)
    } else {
        None
    };
    Saida { primeiro_erro, segundo_erro, logits }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Falha em cada lançamento reformado e na ida de cada reforma: o motor segue utilizável e o token seguinte sai igual bit a bit.
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn erro_no_lancamento_ou_na_ida_da_reforma_deixa_o_motor_utilizavel_e_igual_bit_a_bit() {
    exige_autorizacao();
    let limpo = roda(None).logits.expect("execução limpa");
    let casos: Vec<Injeta> = (1..=4).map(Injeta::Lancamento).chain([1, 3, 5, 7].into_iter().map(Injeta::Reforma)).chain((1..=3).map(Injeta::O3)).collect();
    for inj in casos {
        let s = roda(Some(inj));
        assert!(s.primeiro_erro, "{inj:?}: a falha injetada devia voltar como erro");
        let l = s.logits.unwrap_or_else(|| panic!("{inj:?}: o motor devia seguir utilizável"));
        assert_eq!(bits(&l), bits(&limpo), "{inj:?}: logits depois do erro diferem da execução limpa");
    }
}

/// Falha na volta (restauração) de cada reforma: o erro propaga, a stream é esperada antes do descarte e o campo vazio recusa o uso seguinte.
#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn erro_na_restauracao_espera_a_stream_e_o_campo_vazio_recusa_o_uso_seguinte() {
    exige_autorizacao();
    for n in [2usize, 4, 6, 8] {
        let s = roda(Some(Injeta::Reforma(n)));
        assert!(s.primeiro_erro, "reforma {n}: a falha injetada devia voltar como erro");
        assert!(s.segundo_erro, "reforma {n}: com o campo vazio o uso seguinte devia recusar com erro");
        assert!(s.logits.is_none());
    }
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
