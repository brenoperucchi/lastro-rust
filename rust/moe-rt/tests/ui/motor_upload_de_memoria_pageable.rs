//! Espaços de memória: o upload da tabela lê memória pinned; uma fatia comum do host (pageable) no lugar dela não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn sobe<B: Backend>(b: &B, s: &B::Stream, dst: &Device<[u32]>, src: &[AtomicU32]) -> Result<(), B::Error> {
    b.upload_table(s, dst, src)
}

fn main() {}
