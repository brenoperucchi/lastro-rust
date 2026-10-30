//! M0.9 do runtime MoE em Rust (plano `docs/PLANO-runtime-rust-moe-20261029.md`): monitor de console.
//!
//! - [`snapshot`]: o que o runtime publica (um JSON gravado de forma atômica num ponto único) e o [`snapshot::Publisher`]
//!   que ele usa para isso. O monitor só lê.
//! - [`sys`]: telemetria da máquina lida pelo próprio monitor (GPU pela NVML; CPU, RAM e disco por `/proc`), para a tela
//!   funcionar mesmo sem runtime rodando.
//! - [`ui`]: a TUI, na disposição do monitor de console do Strata (visões 1–7 e as mesmas teclas).
pub mod fmt;
pub mod snapshot;
pub mod sys;
pub mod ui;
