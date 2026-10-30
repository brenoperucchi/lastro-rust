//! Depois de lançar um expert, a stream de cálculo não está parada: o Idle da camada já não pode existir (E0499).
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

fn camada(slots: &mut Slots, epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>) -> Result<(), Error> {
    let pins = slots.pin_layer(&[0, 1]);
    let (_lv, idle) = epoch.router_logits(router, xd)?;
    if let Lookup::Hit(id) = slots.lookup(0) {
        epoch.launch_expert(slots.resident(id)?, xd, 1.0)?;
    }
    let Some(v) = slots.victim(&pins, &idle)? else { return Ok(()) };
    slots.copy_on_compute(v, 1)?;
    Ok(())
}

fn main() {}
