//! Etapa 5d: `lastro --modelo <dir> [--novos N] [--data "Month DD, YYYY"] [TEXTO]`: um texto (argumento ou entrada padrão) entra, o texto da
//! resposta sai na saída padrão (texto puro, um newline no fim). Diagnósticos na saída de erro. Um modelo por execução, motor criado uma vez.
//!
//! Códigos: 0 ok; 2 recusa (argumentos, entrada, texto grande, modelo, portão ou UUID da GPU) ANTES de abrir qualquer device; 1 falha de execução.
//! Ordem das recusas: argumentos, data, modelo, tokenizer, texto (entrada padrão), tamanho, config, portões.
//! O laço é o da 5c (`geracao_regra::gera`); a lógica pura está em `lastro_cli`.

use cuda_core::Device;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream, LeitorCuda};
use moe_rt::geracao_regra as gr;
use moe_rt::lastro_cli as cli;
use moe_rt::leitura_regra::{BLOCO, MAX_LEN};
use moe_rt::model::{Config as ModelConfig, HostModel};
use moe_rt::motor::{AdmitMode, Config, Engine};
use moe_rt::primeira_palavra_regra::{data_de_hoje_utc, Sequencia};
use moe_rt::tokenizador::{formata_data, Tokenizador};
use moe_rt::weights::Checkpoint;
use std::io::Write;
use std::path::Path;

const FRACAO_SLOTS: f64 = 0.10;
const THREADS: usize = 8;

enum Falha {
    /// Recusa antes da GPU (código 2).
    Recusa(String),
    /// Falha de execução (código 1).
    Execucao(String),
}

fn d<T, E: std::fmt::Debug>(r: Result<T, E>) -> Result<T, String> {
    r.map_err(|e| format!("{e:?}"))
}

/// Motor e backend NOVOS (a mesma montagem do `sequencia_real` da primeira-palavra: tabela congelada de 10%, backlog 0, admissão Event, max_len 2.048).
fn sequencia_real(device: &std::sync::Arc<Device>, host: &HostModel, m: &ModelConfig) -> Result<Sequencia<CudaBackend, LeitorCuda>, String> {
    let slots = (FRACAO_SLOTS * (m.layers * m.experts) as f64) as usize;
    let (compute, copy) = (CudaStream(d(device.new_stream())?), CudaStream(d(device.new_stream())?));
    let (b, ex) = d(CudaBackend::new_model_positions(device.clone(), host, slots, MAX_LEN, &compute, BLOCO))?;
    let slots_all = m.layers * m.experts;
    let prefill: Vec<u32> = (0..slots).map(|i| (i * slots_all / slots) as u32).collect();
    let mcfg = Config { layers: m.layers, experts: m.experts, topk: m.k, width: m.hidden, slots, backlog: 0, admit: AdmitMode::Event, first_token: 1, prefill, max_positions: BLOCO };
    let parts = d(b.parts(compute.clone(), copy, d(b.weights(&ex))?, mcfg.first_token))?;
    let cpu = CpuServe { cpu: Cpu::new(THREADS), hidden: m.hidden, inter: m.inter, experts: m.experts, weights: ex.clone() };
    let eng = d(Engine::new(b.clone(), parts, mcfg, cpu))?;
    let prep = d(LeitorCuda::novo(b.clone(), compute.clone()))?;
    let le_logits: Box<dyn FnMut() -> Result<Vec<f32>, String>> = Box::new(move || d(b.read_logits(&compute)));
    Ok(Sequencia { eng, prep, le_logits })
}

fn principal() -> Result<(), Falha> {
    let r = Falha::Recusa;
    let argv: Vec<String> = std::env::args_os().skip(1).map(|a| a.into_string().map_err(|o| format!("argumento que não é UTF-8: {o:?}"))).collect::<Result<_, _>>().map_err(r)?;
    let args = cli::parse_args(argv).map_err(|e| Falha::Recusa(format!("{e}\n{}", cli::USO)))?;
    let data = match &args.data {
        Some(s) => cli::parse_data(s),
        None => {
            let (a, m, dia) = data_de_hoje_utc();
            formata_data(a, m, dia).map_err(|e| e.to_string())
        }
    }
    .map_err(Falha::Recusa)?;
    cli::valida_modelo(&args.modelo).map_err(Falha::Recusa)?;
    let tok = Tokenizador::carrega(&args.modelo.join("tokenizer.json")).map_err(|e| Falha::Recusa(format!("tokenizer.json ilegível: {e}")))?;
    let texto = cli::resolve_texto(args.texto.clone(), std::io::stdin().lock()).map_err(Falha::Recusa)?;
    cli::confere_tamanho(&tok, &texto, &data, args.novos).map_err(Falha::Recusa)?;
    let cfg = carrega_config(&args.modelo).map_err(|e| Falha::Recusa(format!("config.json inválido: {e}")))?;
    cli::portoes().map_err(Falha::Recusa)?;
    // A partir daqui é execução: qualquer erro é código 1.
    let exe = Falha::Execucao;
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(THREADS));
    let device = d(Device::new(0)).map_err(exe)?;
    let host = d(Checkpoint::open(&args.modelo)).and_then(|c| d(HostModel::load(&c, cfg.clone()))).map_err(Falha::Execucao)?;
    let mut motor: gr::MotorGeracao<CudaBackend, LeitorCuda> = gr::MotorGeracao::novo();
    let g = cli::executa(&texto, &data, args.novos, &tok, &mut motor, || sequencia_real(&device, &host, &cfg), &mut std::io::stdout());
    let fim = motor.encerra();
    let g = g.map_err(Falha::Execucao)?;
    eprintln!("({} ids, {} novos, prefill {:.1} ms, {:.2} ms por token (mediana), total {:.1} ms)", g.ids.len(), g.gerados.len(), g.prefill_ms, g.ms_por_token_mediana(), g.total_ms);
    fim.map_err(|e| Falha::Execucao(format!("encerramento do motor: {e}")))
}

fn carrega_config(mdir: &Path) -> Result<ModelConfig, String> {
    let bytes = std::fs::read(mdir.join("config.json")).map_err(|e| e.to_string())?;
    d(ModelConfig::from_json(&d(serde_json::from_slice(&bytes))?))
}

fn main() {
    let codigo = match principal() {
        Ok(()) => 0,
        Err(Falha::Recusa(e)) => {
            eprintln!("lastro: recusado: {e}");
            2
        }
        Err(Falha::Execucao(e)) => {
            eprintln!("lastro: falhou: {e}");
            1
        }
    };
    std::io::stdout().flush().ok();
    std::process::exit(codigo);
}
