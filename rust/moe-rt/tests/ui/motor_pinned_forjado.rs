//! Só o crate cria os tipos de memória: `Pinned::from_parts` é privado para quem usa o motor.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn forja() {
    let _f = Pinned::<u32>::from_parts;
}

fn main() {}
