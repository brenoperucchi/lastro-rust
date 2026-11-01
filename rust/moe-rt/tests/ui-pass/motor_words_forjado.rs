//! Gêmeo que compila de ui/motor_words_forjado.rs: a `LayerIo` segura vem de `LayerIo::new`.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{LayerIo, Words};
use std::sync::atomic::{AtomicU32, Ordering};

fn recebe(io: &LayerIo) -> usize {
    io.ids.len() + io.y[0].load(Ordering::Relaxed) as usize
}

fn main() {
    let io = LayerIo::new(2, 4);
    let _ = recebe(&io);
}
