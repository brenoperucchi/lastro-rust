//! Uma vítima recebe uma cópia só: a cópia a consome (E0382 na segunda).
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

fn camada(slots: &mut Slots, epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>) -> Result<(), Error> {
    let pins = slots.pin_layer(&[0, 1]);
    let (_lv, idle) = epoch.router_logits(router, xd)?;
    let Some(v) = slots.victim(&pins, &idle)? else { return Ok(()) };
    slots.copy_on_compute(v, 0)?;
    slots.copy_on_compute(v, 1)?;
    Ok(())
}

fn main() {}
