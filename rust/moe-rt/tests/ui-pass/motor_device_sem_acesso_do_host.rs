//! Gêmeo que compila de ui/motor_device_sem_acesso_do_host.rs.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn le(d: &Mapped<AtomicU32>) -> u32 {
    d.load(Ordering::Relaxed)
}

fn main() {}
