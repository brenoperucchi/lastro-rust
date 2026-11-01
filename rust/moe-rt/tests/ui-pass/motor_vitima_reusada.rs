//! Gêmeo que compila de ui/motor_vitima_reusada.rs: uma vítima por cópia.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::Error;
use std::sync::atomic::AtomicU32;


fn camada<B: Backend>(svc: &mut Service<B>, epoch: &TokenEpoch) -> Result<(), Error<B::Error>> {
    let view = svc.serve_layer(epoch, 0)?;
    let served = answer(view, &[0, 0]);
    let pins = svc.pin_layer(&served);
    for key in [3, 4] {
        let Some(v) = svc.victim(&served, &pins, key)? else { return Ok(()) };
        svc.copy_after_reader(v, key, &served)?;
    }
    Ok(())
}

fn main() {}
