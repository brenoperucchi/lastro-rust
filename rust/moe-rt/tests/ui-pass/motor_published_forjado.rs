//! Gêmeo que compila de ui/motor_published_forjado.rs: o `Published` vem do motor.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::{Engine, Error, Published};

fn recebe<B: Backend>(e: &mut Engine<B>) -> Result<u32, Error<B::Error>> {
    let p = e.next_published()?;
    Ok(p.generation())
}

fn main() {}
