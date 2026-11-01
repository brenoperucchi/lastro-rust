//! Gêmeo que compila de ui/motor_virada_com_served_vivo.rs.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn camada<B: Backend>(svc: &mut Service<B>, epoch: TokenEpoch) -> Result<(), Error<B::Error>> {
    let view = svc.serve_layer(&epoch, 0)?;
    let served = answer(view, &[0, 0]);
    let _pins = svc.pin_layer(&served);
    let _proximo = svc.end_token(epoch)?;
    Ok(())
}

fn main() {}
