//! Um token só é enfileirado com o seu `Published`: `enqueue_token` sem o argumento não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn enfileira<B: Backend>(e: &mut Engine<B>) -> Result<(), Error<B::Error>> {
    e.enqueue_token()
}

fn main() {}
