//! Espaços de memória: o doorbell fica em memória mapeada; um `Device<AtomicU32>` no `wait_value` não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn espera<B: Backend>(b: &B, s: &B::Stream, seq: &Device<AtomicU32>) -> Result<(), B::Error> {
    b.wait_value(s, seq, 1)
}

fn main() {}
