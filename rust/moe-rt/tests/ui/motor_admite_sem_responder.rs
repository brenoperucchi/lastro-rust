//! A admissão pede a camada respondida: com a `LayerView` ainda em mãos (a flag não foi escrita), `victim` não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::Error;
use std::sync::atomic::AtomicU32;


fn camada<B: Backend>(svc: &mut Service<B>, epoch: &TokenEpoch, pins: &Pins) -> Result<(), Error<B::Error>> {
    let view = svc.serve_layer(epoch, 0)?;
    let _v = svc.victim(&view, pins, 3)?;
    Ok(())
}

fn main() {}
