//! Gêmeo que compila de ui/motor_pinned_no_kernel.rs: o lançamento recebe a tabela do device.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::Error;
use std::sync::atomic::AtomicU32;


fn lanca<B: Backend>(b: &B, s: &B::Stream, tabela: &Device<[u32]>, banco: &Device<B::Bank>, io: &Mapped<LayerIo>)
    -> Result<(), B::Error> {
    b.launch_hits(s, 0, tabela, banco, io)
}

fn main() {}
