//! Runtime MoE em Rust sobre o CUDA Rust da NVIDIA (plano `docs/PLANO-runtime-rust-moe-20261029.md`).
//!
//! M1a: o bloco MoE de uma camada (roteador + experts SwiGLU), por dois caminhos que o runtime vai combinar:
//! - [`cpu`]: experts calculados na CPU a partir dos pesos BF16 na RAM (o caminho dos experts frios);
//! - [`gpu`]: experts residentes na VRAM, com kernels Tile do cutile-rs.
//!
//! [`weights`] lê os tensores do safetensors; [`block`] tem os pesos de uma camada e a decisão do roteador.
pub mod block;
pub mod cache;
pub mod cpu;
pub mod gpu;
pub mod gpu_model;
pub mod model;
pub mod npy;
pub mod pool;
pub mod weights;
