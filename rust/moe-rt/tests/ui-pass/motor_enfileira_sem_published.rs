//! Gêmeo que compila de ui/motor_enfileira_sem_published.rs.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn enfileira<B: Backend>(e: &mut Engine<B>) -> Result<(), Error<B::Error>> {
    let p = e.next_published()?;
    e.enqueue_token(p)
}

fn main() {}
