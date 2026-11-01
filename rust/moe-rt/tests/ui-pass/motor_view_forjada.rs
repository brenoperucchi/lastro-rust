//! Gêmeo que compila de ui/motor_view_forjada.rs: a `LayerView` vem de `serve_layer`.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::Error;
use std::sync::atomic::AtomicU32;


fn camada<B: Backend>(svc: &mut Service<B>, epoch: &TokenEpoch) -> Result<(), Error<B::Error>> {
    let forjada = svc.serve_layer(epoch, 0)?;
    let _served = answer(forjada, &[0, 0]);
    Ok(())
}

fn main() {}
