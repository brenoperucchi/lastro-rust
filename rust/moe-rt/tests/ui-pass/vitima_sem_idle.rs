//! Gêmeo que compila de ui/vitima_sem_idle.rs: vítima escolhida antes do primeiro lançamento.
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

#[allow(dead_code)]
fn camada(slots: &mut Slots, epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>) -> Result<(), Error> {
    let pins = slots.pin_layer(&[0, 1]);
    let (_lv, idle) = epoch.router_logits(router, xd)?;
    let novo = match slots.victim(&pins, &idle)? {
        Some(v) => Some(slots.copy_on_compute(v, 1)?),
        None => None,
    };
    drop(idle);
    if let Lookup::Hit(id) = slots.lookup(0) {
        epoch.launch_expert(slots.resident(id)?, xd, 1.0)?;
    }
    if let Some(id) = novo {
        epoch.launch_expert(slots.resident(id)?, xd, 1.0)?;
    }
    Ok(())
}

fn main() {}
