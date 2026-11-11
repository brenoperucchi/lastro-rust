//! Runtime MoE em Rust sobre o CUDA Rust da NVIDIA (plano `docs/PLANO-runtime-rust-moe-20261029.md`).
//!
//! M1a: o bloco MoE de uma camada (roteador + experts SwiGLU), por dois caminhos que o runtime vai combinar:
//! - [`cpu`]: experts calculados na CPU a partir dos pesos BF16 na RAM (o caminho dos experts frios);
//! - [`gpu`]: experts residentes na VRAM, com kernels Tile do cutile-rs.
//!
//! [`weights`] lê os tensores do safetensors; [`block`] tem os pesos de uma camada e a decisão do roteador.
pub mod affinity;
pub mod block;
pub mod cache;
pub mod cpu;
pub mod diag;
pub mod e4b_regra;
pub mod leitura_regra;
#[cfg(not(motor_loom))]
pub mod cuda_backend;
#[cfg(not(motor_loom))]
pub(crate) mod cuda_dense;
pub mod gpu;
pub mod gancho_coleta;
pub mod geracao_regra;
pub mod gpu_model;
pub mod hits;
pub mod lastro_cli;
pub mod lastro_servidor;
pub mod model;
pub mod motor;
pub mod npy;
pub mod perfil_bloco;
pub mod pool;
pub(crate) mod reforma;
pub mod primeira_palavra_regra;
pub mod reset_regra;
pub mod route;
pub mod tokenizador;
pub mod velocidade_regra;
pub mod weights;

/// Os testes de unidade que usam a GPU rodam em threads do mesmo processo; os dois testes de portão do núcleo fecham uma
/// stream com um callback de host, e qualquer sincronização do contexto feita por outro teste nesse meio tempo espera o
/// portão (prende até o prazo ou deixa a cópia assentar antes da conferência). Todo teste de unidade com GPU segura esta
/// trava do começo ao fim.
#[cfg(test)]
pub(crate) fn gpu_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
    GPU.lock().unwrap_or_else(|e| e.into_inner())
}
