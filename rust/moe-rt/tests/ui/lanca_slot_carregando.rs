//! Um slot com cópia em segundo plano a caminho não serve para lançar: só um ResidentSlot serve.
#![allow(unused_imports)]
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Partition;
use moe_rt::cache::core::{Epoch, Lookup, Pins, Slots};

fn camada(slots: &mut Slots, epoch: &mut Epoch, router: &Tensor<bf16>, xd: &Tensor<f32>, key: usize) -> Result<(), Error> {
    let pins = slots.pin_layer(&[key]);
    let (_lv, idle) = epoch.router_logits(router, xd)?;
    let Some(v) = slots.victim(&pins, &idle)? else { return Ok(()) };
    let carregando = slots.copy_in_background(v, key)?;
    drop(idle);
    epoch.launch_expert(carregando, xd, 1.0)?;
    Ok(())
}

fn main() {}
