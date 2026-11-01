//! `Words` tem representação privada: código seguro não monta uma fatia de palavras a partir de um ponteiro qualquer.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{LayerIo, Words};
use std::sync::atomic::AtomicU32;

fn forja() {
    let _w = Words(std::ptr::null::<AtomicU32>());
}

fn main() {}
