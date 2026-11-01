//! Gêmeo que compila de ui/motor_lookup_depois_da_virada.rs.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn camada<B: Backend>(svc: &mut Service<B>, epoch: TokenEpoch) -> Result<(), Error<B::Error>> {
    let _slot = epoch.lookup(0);
    let _proximo = svc.end_token(epoch)?;
    Ok(())
}

fn main() {}
