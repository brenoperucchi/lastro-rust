//! Etapa 5e: `lastro-servidor --modelo <dir> --porta <N>`: servidor HTTP local (SÓ 127.0.0.1, sem opção de endereço), um cliente por vez, motor criado
//! UMA vez na partida e reutilizado (`nova_sequencia`). `GET /saude`, `POST /gerar?novos=N&data=...` (texto cru no corpo), `POST /encerrar`.
//!
//! Códigos: 0 encerrado por `POST /encerrar`; 2 recusa de partida (argumentos, modelo, porta ocupada, portão ou UUID da GPU) ANTES de abrir qualquer
//! device; 1 falha de execução (inclusive falha do motor num pedido: responde 500 e sai). A linha `escutando em 127.0.0.1:<porta>` vai para a saída
//! de erro só depois do motor pronto. SIGINT/SIGTERM: o padrão do sistema. A lógica pura está em `lastro_servidor`; o laço de geração é o da 5c.

use cuda_core::Device;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream, LeitorCuda};
use moe_rt::geracao_regra as gr;
use moe_rt::lastro_cli as cli;
use moe_rt::lastro_servidor as srv;
use moe_rt::leitura_regra::{BLOCO, MAX_LEN};
use moe_rt::model::{Config as ModelConfig, HostModel};
use moe_rt::motor::{AdmitMode, Config, Engine};
use moe_rt::primeira_palavra_regra::{data_de_hoje_utc, Sequencia};
use moe_rt::tokenizador::{formata_data, Tokenizador};
use moe_rt::weights::Checkpoint;
use std::net::{Ipv4Addr, SocketAddrV4, TcpListener};
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

fn hoje() -> String {
    let (a, m, dia) = data_de_hoje_utc();
    formata_data(a, m, dia).unwrap_or_default()
}

fn principal() -> Result<(), Falha> {
    let argv: Vec<String> = std::env::args_os().skip(1).map(|a| a.into_string().map_err(|o| format!("argumento que não é UTF-8: {o:?}"))).collect::<Result<_, _>>().map_err(Falha::Recusa)?;
    let args = srv::parse_args(argv).map_err(|e| Falha::Recusa(format!("{e}\n{}", srv::USO)))?;
    cli::valida_modelo(&args.modelo).map_err(Falha::Recusa)?;
    let tok = Tokenizador::carrega(&args.modelo.join("tokenizer.json")).map_err(|e| Falha::Recusa(format!("tokenizer.json ilegível: {e}")))?;
    let cfg = carrega_config(&args.modelo).map_err(|e| Falha::Recusa(format!("config.json inválido: {e}")))?;
    // Só loopback, endereço fixo no código. A porta ocupada é recusa de partida (antes dos portões e do device).
    let listener = TcpListener::bind(SocketAddrV4::new(Ipv4Addr::LOCALHOST, args.porta)).map_err(|e| Falha::Recusa(format!("porta {} indisponível em 127.0.0.1: {e}", args.porta)))?;
    cli::portoes().map_err(Falha::Recusa)?;
    // A partir daqui é execução: qualquer erro é código 1.
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(THREADS));
    let device = d(Device::new(0)).map_err(Falha::Execucao)?;
    let host = d(Checkpoint::open(&args.modelo)).and_then(|c| d(HostModel::load(&c, cfg.clone()))).map_err(Falha::Execucao)?;
    // Motor pronto ANTES de escutar: a sequência criada aqui é entregue ao primeiro pedido (`cria`), que a reutiliza; os demais usam `nova_sequencia`.
    let mut pronta = Some(sequencia_real(&device, &host, &cfg).map_err(Falha::Execucao)?);
    let mut motor: gr::MotorGeracao<CudaBackend, LeitorCuda> = gr::MotorGeracao::novo();
    eprintln!("escutando em 127.0.0.1:{}", args.porta);
    let fluxo = srv::serve(&listener, &tok, &mut motor, || match pronta.take() {
        Some(s) => Ok(s),
        None => sequencia_real(&device, &host, &cfg),
    }, srv::PRAZO_LEITURA, &hoje);
    drop(listener);
    let fim = motor.encerra();
    match fluxo {
        srv::Fluxo::Falha(e) => Err(Falha::Execucao(e)),
        _ => fim.map_err(|e| Falha::Execucao(format!("encerramento do motor: {e}"))),
    }
}

fn carrega_config(mdir: &Path) -> Result<ModelConfig, String> {
    let bytes = std::fs::read(mdir.join("config.json")).map_err(|e| e.to_string())?;
    d(ModelConfig::from_json(&d(serde_json::from_slice(&bytes))?))
}

fn main() {
    let codigo = match principal() {
        Ok(()) => 0,
        Err(Falha::Recusa(e)) => {
            eprintln!("lastro-servidor: recusado: {e}");
            2
        }
        Err(Falha::Execucao(e)) => {
            eprintln!("lastro-servidor: falhou: {e}");
            1
        }
    };
    std::process::exit(codigo);
}
