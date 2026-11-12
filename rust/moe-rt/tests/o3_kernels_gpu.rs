//! A2, O3: teste de GPU, isolado, dos dois kernels que gravam direto no cache K/V (`gpu::kernels::{rope_kv, gemv_row}`). NÃO RODA SEM
//! AUTORIZAÇÃO e NÃO FOI EXECUTADO: está escrito e compila; só uma janela autorizada que o nomeie o roda (`E4B_GPU_AUTORIZADO=1`).
//!
//! Critério, sem tolerância: a linha `cur` do cache sai IGUAL BIT A BIT à saída do kernel antigo (`rope`, `gemv`) com as mesmas entradas, e
//! todas as outras linhas ficam com o padrão sentinela com que o cache começou (nenhuma escrita fora da linha `cur`). Várias posições
//! `cur` (primeira, do meio, última) no mesmo cache.
use cuda_core::Device;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Reshape;
use moe_rt::gpu::{gemv_generics, kernels, GEMV_BM};
use std::sync::Arc;

static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn exige_autorizacao() {
    assert_eq!(std::env::var("E4B_GPU_AUTORIZADO").as_deref(), Ok("1"), "o teste de GPU da O3 só roda com o 'autorizo' do Breno (E4B_GPU_AUTORIZADO=1)");
}

fn valores(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Padrão que nenhum dos dois kernels produz (NaN com carga própria), para enxergar qualquer escrita fora da linha `cur`.
const SENTINELA: u32 = 0x7fc0_1234;

const MAX_LEN: usize = 8;
const KV_HEADS: usize = 2;
const D: usize = 64;
const HIDDEN: usize = 1024;

#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn rope_kv_e_gemv_row_gravam_so_a_linha_cur_e_iguais_bit_a_bit_aos_kernels_antigos() {
    exige_autorizacao();
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let s = device.new_stream().unwrap();
    let (half, kvd) = (D / 2, KV_HEADS * D);
    let up = |v: Vec<f32>, forma: &[usize]| -> Tensor<f32> { api::copy_host_vec_to_device(&Arc::new(v)).sync_on(&s).unwrap().reshape(forma).unwrap() };
    let sentinela = vec![f32::from_bits(SENTINELA); MAX_LEN * kvd];

    // RoPE de K: entrada reformada `[kv·2, half]` e `cos/sin` `[2, half]`.
    let x = up(valores(KV_HEADS * 2 * half, 11), &[KV_HEADS * 2, half]);
    let cs = up(valores(2 * half, 12), &[2, half]);
    let mut ref_rope = api::zeros::<f32>(&[KV_HEADS * 2 * half]).sync_on(&s).unwrap().reshape(&[KV_HEADS * 2, half]).unwrap();
    // SAFETY: formas de `cuda_dense`; só esta stream toca os tensores e `sync_on` espera.
    unsafe { kernels::rope((&mut ref_rope).partition([2, half]), &x, &cs).generics(vec![half.to_string()]).sync_on(&s).unwrap() };
    let esperado_k = ref_rope.to_host_vec().sync_on(&s).unwrap();

    // gemv de V: peso `[kvd, HIDDEN]` em bf16 e `xn` `[HIDDEN]`.
    let w_host: Vec<bf16> = valores(kvd * HIDDEN, 13).iter().map(|v| bf16::from_f32(*v)).collect();
    let w = api::copy_host_vec_to_device(&Arc::new(w_host)).sync_on(&s).unwrap().reshape(&[kvd, HIDDEN]).unwrap();
    let xn = api::copy_host_vec_to_device(&Arc::new(valores(HIDDEN, 14))).sync_on(&s).unwrap();
    let mut ref_v = api::zeros::<f32>(&[kvd]).sync_on(&s).unwrap();
    // SAFETY: como acima.
    unsafe { kernels::gemv((&mut ref_v).partition([GEMV_BM]), &w, &xn).generics(gemv_generics(HIDDEN)).sync_on(&s).unwrap() };
    let esperado_v = ref_v.to_host_vec().sync_on(&s).unwrap();

    for cur in [0usize, 3, MAX_LEN - 1] {
        let mut kc = up(sentinela.clone(), &[MAX_LEN, kvd]);
        let mut vc = up(sentinela.clone(), &[MAX_LEN, kvd]);
        let linha = cur as i32;
        // SAFETY: como acima; `cur < MAX_LEN`.
        unsafe {
            kernels::rope_kv((&mut kc).partition([1, D]).map([1, 1], KV_HEADS as u32), &x, &cs, linha)
                .generics(vec![half.to_string(), D.to_string(), "1".to_string(), "1".to_string()])
                .sync_on(&s)
                .unwrap();
            let mut g = gemv_generics(HIDDEN);
            g.extend(["1".to_string(), "1".to_string()]);
            kernels::gemv_row((&mut vc).partition([1, GEMV_BM]).map([1, 1], (kvd / GEMV_BM) as u32), &w, &xn, linha).generics(g).sync_on(&s).unwrap();
        }
        let (k, v) = (kc.to_host_vec().sync_on(&s).unwrap(), vc.to_host_vec().sync_on(&s).unwrap());
        for (nome, cache, esperado) in [("K", &k, &esperado_k), ("V", &v, &esperado_v)] {
            for linha in 0..MAX_LEN {
                let got = bits(&cache[linha * kvd..(linha + 1) * kvd]);
                if linha == cur {
                    assert_eq!(got, bits(esperado), "{nome} cur={cur}: a linha cur difere do kernel antigo");
                } else {
                    assert!(got.iter().all(|b| *b == SENTINELA), "{nome} cur={cur}: a linha {linha} (fora de cur) foi escrita");
                }
            }
        }
    }
}

/// O portão: sem `E4B_GPU_AUTORIZADO=1` o teste de GPU recusa antes de tocar o device. Este não usa GPU e roda sempre.
#[test]
fn sem_autorizacao_o_portao_recusa_antes_de_tocar_a_gpu() {
    if std::env::var("E4B_GPU_AUTORIZADO").as_deref() == Ok("1") {
        return;
    }
    assert!(std::panic::catch_unwind(exige_autorizacao).is_err(), "o portão tinha de recusar sem a autorização");
}
