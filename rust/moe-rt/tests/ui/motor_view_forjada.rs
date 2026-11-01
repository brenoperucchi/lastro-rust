//! A `LayerView` prova que a GPU passou pelo roteador da camada: montá-la à mão não compila (campos privados).
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::Error;
use std::sync::atomic::AtomicU32;


use moe_rt::motor::line::LayerView;

fn camada<B: Backend>(epoch: &TokenEpoch) {
    let forjada = LayerView { epoch, layer: 0 };
    let _served = answer(forjada, &[0, 0]);
}

fn main() {}
