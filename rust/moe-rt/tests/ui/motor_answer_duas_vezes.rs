//! A flag de uma camada é escrita uma vez só: `answer` consome a `LayerView`, e uma segunda chamada não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn camada<B: Backend>(svc: &mut Service<B>, epoch: &TokenEpoch) -> Result<(), Error<B::Error>> {
    let view = svc.serve_layer(epoch, 0)?;
    let _a = answer(view, &[0, 0]);
    let _b = answer(view, &[0, 0]);
    Ok(())
}

fn main() {}
