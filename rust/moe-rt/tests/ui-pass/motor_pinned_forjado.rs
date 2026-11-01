//! Gêmeo que compila de ui/motor_pinned_forjado.rs.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::line::{answer, Pins, Service, TokenEpoch};
use moe_rt::motor::mem::{Device, Mapped, Pinned};
use moe_rt::motor::{Engine, Error, Published};
use std::sync::atomic::{AtomicU32, Ordering};

fn recebe(_p: &Pinned<u32>) {}

fn main() {}
