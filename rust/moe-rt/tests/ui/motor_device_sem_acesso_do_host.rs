//! Espaços de memória: o `Device` não dá acesso ao host; indexá-lo não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn le(d: &Device<[u32]>) -> u32 {
    d[0]
}

fn main() {}
