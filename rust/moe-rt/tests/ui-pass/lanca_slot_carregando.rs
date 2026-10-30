//! Gêmeo que compila de ui/lanca_slot_carregando.rs: lança um slot Resident.
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

#[allow(dead_code)]
fn camada(slots: &mut Slots, epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>, key: usize) -> Result<(), Error> {
    let _pins = slots.pin_layer(&[key]);
    let (_lv, idle) = epoch.router_logits(router, xd)?;
    drop(idle);
    if let Lookup::Hit(id) = slots.lookup(key) {
        epoch.launch_expert(slots.resident(id)?, xd, 1.0)?;
    }
    Ok(())
}

fn main() {}
