//! Espaços de memória: o lançamento lê a tabela no device; uma tabela `Pinned` (do host) no lugar dela não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::Error;
use std::sync::atomic::AtomicU32;


fn lanca<B: Backend>(b: &B, s: &B::Stream, tabela: &Pinned<[u32]>, banco: &Device<B::Bank>, io: &Mapped<LayerIo>)
    -> Result<(), B::Error> {
    b.launch_hits(s, 0, tabela, banco, io)
}

fn main() {}
