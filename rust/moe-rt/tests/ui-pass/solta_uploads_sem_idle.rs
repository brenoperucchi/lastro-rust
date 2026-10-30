//! Gêmeo que compila de ui/solta_uploads_sem_idle.rs: solta os uploads antes de enfileirar o novo.
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

#[allow(dead_code)]
fn camada(epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>, h: Partition<Tensor<f32>>, y: Vec<f32>) -> Result<(), Error> {
    let (_lv, mut idle) = epoch.router_logits(router, xd)?;
    idle.release_uploads();
    drop(idle);
    let _h = epoch.add_cpu_out(h, y, 1.0)?;
    Ok(())
}

fn main() {}
