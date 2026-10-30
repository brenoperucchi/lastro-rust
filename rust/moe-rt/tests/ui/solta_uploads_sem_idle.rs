//! Os uploads da CPU só são soltos com a stream de cálculo parada: depois de enfileirar um, o Idle já não vale (E0499).
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

fn camada(epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>, h: Partition<Tensor<f32>>, y: Vec<f32>) -> Result<(), Error> {
    let (_lv, mut idle) = epoch.router_logits(router, xd)?;
    let _h = epoch.add_cpu_out(h, y, 1.0)?;
    idle.release_uploads();
    Ok(())
}

fn main() {}
