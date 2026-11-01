//! Gêmeo que compila de ui/motor_doorbell_no_device.rs.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn espera<B: Backend>(b: &B, s: &B::Stream, seq: &Mapped<AtomicU32>) -> Result<(), B::Error> {
    b.wait_value(s, seq, 1)
}

fn main() {}
