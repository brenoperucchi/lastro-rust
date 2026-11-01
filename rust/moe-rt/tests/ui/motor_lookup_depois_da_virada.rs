//! A tabela só muda na virada: depois de `end_token` consumir o `TokenEpoch`, consultá-lo não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn camada<B: Backend>(svc: &mut Service<B>, epoch: TokenEpoch) -> Result<(), Error<B::Error>> {
    let _proximo = svc.end_token(epoch)?;
    let _slot = epoch.lookup(0);
    Ok(())
}

fn main() {}
