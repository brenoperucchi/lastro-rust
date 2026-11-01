//! Um `Published` só nasce da linha de serviço: construí-lo fora do crate (campos privados) não compila.
#![allow(unused_imports, dead_code)]
use moe_rt::motor::backend::{Backend, LayerIo};
use moe_rt::motor::{Engine, Error, Published};

fn forja() -> Option<u32> {
    let p = Published { origin: 1, gen: 1 };
    Some(p.generation())
}

fn main() {}
