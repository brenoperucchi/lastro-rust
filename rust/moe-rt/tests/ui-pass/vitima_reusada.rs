//! Gêmeo que compila de ui/vitima_reusada.rs: uma vítima por cópia.
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

#[allow(dead_code)]
fn camada(slots: &mut Slots, epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>) -> Result<(), Error> {
    let pins = slots.pin_layer(&[0, 1]);
    let (_lv, idle) = epoch.router_logits(router, xd)?;
    for key in [0, 1] {
        let Some(v) = slots.victim(&pins, &idle)? else { return Ok(()) };
        slots.copy_on_compute(v, key)?;
    }
    Ok(())
}

fn main() {}
