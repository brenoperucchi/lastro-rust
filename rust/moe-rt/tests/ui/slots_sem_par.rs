//! Slots e Epoch só nascem juntos (new_pair), na mesma stream de cálculo: o construtor de um só é privado (E0624).
#![allow(unused_imports)]
use cutile::half::bf16;
use cuda_core::Stream;
use moe_rt::cache::core::{new_pair, Epoch, Slots};
use std::sync::Arc;

fn cria(experts: &[(&[bf16], &[bf16])], s: Arc<Stream>) {
    let _slots = Slots::new(experts, 256, 128, 1, s, 7);
}

fn main() {}
