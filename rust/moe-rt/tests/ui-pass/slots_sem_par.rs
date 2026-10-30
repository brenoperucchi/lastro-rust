//! Gêmeo que compila de ui/slots_sem_par.rs: o par vem de new_pair.
#![allow(unused_imports)]
use cutile::half::bf16;
use cuda_core::Stream;
use moe_rt::cache::core::{new_pair, Epoch, Slots};
use std::sync::Arc;

#[allow(dead_code)]
fn cria(experts: &[(&[bf16], &[bf16])], s: Arc<Stream>) {
    let _par: Result<(Slots, Epoch), _> = new_pair(experts, 256, 128, 8, 1, s);
}

fn main() {}
