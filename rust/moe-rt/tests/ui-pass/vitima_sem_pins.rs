//! Gêmeo que compila de ui/vitima_sem_pins.rs: Pins vindo de pin_layer.
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

#[allow(dead_code)]
fn camada(slots: &mut Slots, epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>) -> Result<(), Error> {
    let (_lv, idle) = epoch.router_logits(router, xd)?;
    let pins = slots.pin_layer(&[0]);
    let Some(v) = slots.victim(&pins, &idle)? else { return Ok(()) };
    slots.copy_on_compute(v, 0)?;
    Ok(())
}

fn main() {}
