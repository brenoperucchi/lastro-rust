//! A2, O3: teste de GPU, isolado, dos dois kernels que gravam direto no cache K/V (`gpu::kernels::{rope_kv, gemv_row}`). NÃO RODA SEM
//! AUTORIZAÇÃO e NÃO FOI EXECUTADO: está escrito e compila; só uma janela autorizada que o nomeie o roda (`E4B_GPU_AUTORIZADO=1`).
//!
//! Critério, sem tolerância: a linha `cur` do cache sai IGUAL BIT A BIT à saída do kernel antigo (`rope`, `gemv`) com as mesmas entradas, e
//! todas as outras linhas ficam com o padrão sentinela com que o cache começou (nenhuma escrita fora da linha `cur`). Duas formas reais (1B e 3B: hidden 1024 e 1536, 8 cabeças de KV, head_dim 64) e as posições `cur` 0, 1, 2, 4, 8 e 15 de um cache de 16
//! linhas (as cinco classes de divisibilidade do JIT, o meio e a última linha); o teste imprime quais formas e posições passaram.
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

const MAX_LEN: usize = 16;
/// Posições `cur` com as classes de divisibilidade que o JIT especializa (1, 2, 4, 8 e 16), mais o meio e a última linha.
const POSICOES: [usize; 6] = [0, 1, 2, 4, 8, 15];
/// As duas formas reais, `(hidden, kv_heads, head_dim)`: 1B (K = 1024) e 3B (K = 1536, `tile_k` 512, 3 iterações do laço de K).
const FORMAS: [(usize, usize, usize); 2] = [(1024, 8, 64), (1536, 8, 64)];

/// Uma forma: `rope_kv` e `gemv_row` contra `rope` e `gemv` com as mesmas entradas, em cada posição de `POSICOES`.
fn confere_forma(s: &std::sync::Arc<cuda_core::Stream>, hidden: usize, kv_heads: usize, d: usize) {
    let (half, kvd) = (d / 2, kv_heads * d);
    let up = |v: Vec<f32>, forma: &[usize]| -> Tensor<f32> { api::copy_host_vec_to_device(&Arc::new(v)).sync_on(s).unwrap().reshape(forma).unwrap() };
    let sentinela = vec![f32::from_bits(SENTINELA); MAX_LEN * kvd];

    // RoPE de K: entrada reformada `[kv·2, half]` e `cos/sin` `[2, half]`.
    let x = up(valores(kv_heads * 2 * half, 11), &[kv_heads * 2, half]);
    let cs = up(valores(2 * half, 12), &[2, half]);
    let mut ref_rope = api::zeros::<f32>(&[kv_heads * 2 * half]).sync_on(s).unwrap().reshape(&[kv_heads * 2, half]).unwrap();
    // SAFETY: formas de `cuda_dense`; só esta stream toca os tensores e `sync_on` espera.
    unsafe { kernels::rope((&mut ref_rope).partition([2, half]), &x, &cs).generics(vec![half.to_string()]).sync_on(s).unwrap() };
    let esperado_k = ref_rope.to_host_vec().sync_on(s).unwrap();

    // gemv de V: peso `[kvd, hidden]` em bf16 e `xn` `[hidden]`.
    let w_host: Vec<bf16> = valores(kvd * hidden, 13).iter().map(|v| bf16::from_f32(*v)).collect();
    let w = api::copy_host_vec_to_device(&Arc::new(w_host)).sync_on(s).unwrap().reshape(&[kvd, hidden]).unwrap();
    let xn = api::copy_host_vec_to_device(&Arc::new(valores(hidden, 14))).sync_on(s).unwrap();
    let mut ref_v = api::zeros::<f32>(&[kvd]).sync_on(s).unwrap();
    // SAFETY: como acima.
    unsafe { kernels::gemv((&mut ref_v).partition([GEMV_BM]), &w, &xn).generics(gemv_generics(hidden)).sync_on(s).unwrap() };
    let esperado_v = ref_v.to_host_vec().sync_on(s).unwrap();

    for cur in POSICOES {
        let mut kc = up(sentinela.clone(), &[MAX_LEN, kvd]);
        let mut vc = up(sentinela.clone(), &[MAX_LEN, kvd]);
        let linha = cur as i32;
        // SAFETY: como acima; `cur < MAX_LEN`.
        unsafe {
            kernels::rope_kv((&mut kc).partition([1, d]).map([1, 1], kv_heads as u32), &x, &cs, linha)
                .generics(vec![half.to_string(), d.to_string(), "1".to_string(), "1".to_string()])
                .sync_on(s)
                .unwrap();
            let mut g = gemv_generics(hidden);
            g.extend(["1".to_string(), "1".to_string()]);
            kernels::gemv_row((&mut vc).partition([1, GEMV_BM]).map([1, 1], (kvd / GEMV_BM) as u32), &w, &xn, linha).generics(g).sync_on(s).unwrap();
        }
        let (k, v) = (kc.to_host_vec().sync_on(s).unwrap(), vc.to_host_vec().sync_on(s).unwrap());
        for (nome, cache, esperado) in [("K", &k, &esperado_k), ("V", &v, &esperado_v)] {
            for l in 0..MAX_LEN {
                let got = bits(&cache[l * kvd..(l + 1) * kvd]);
                if l == cur {
                    assert_eq!(got, bits(esperado), "{nome} forma (hidden {hidden}, kv_heads {kv_heads}, d {d}) cur={cur}: a linha cur difere do kernel antigo");
                } else {
                    assert!(got.iter().all(|b| *b == SENTINELA), "{nome} forma (hidden {hidden}, kv_heads {kv_heads}, d {d}) cur={cur}: a linha {l} (fora de cur) foi escrita");
                }
            }
        }
        println!("ok: forma (hidden {hidden}, kv_heads {kv_heads}, head_dim {d}, kvd {kvd}) cur={cur}: K e V iguais bit a bit, demais linhas intactas");
    }
}

#[test]
#[ignore = "GPU: só com o autorizo do Breno (E4B_GPU_AUTORIZADO=1)"]
fn rope_kv_e_gemv_row_gravam_so_a_linha_cur_e_iguais_bit_a_bit_aos_kernels_antigos() {
    exige_autorizacao();
    let _trava = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let device = Device::new(0).expect("GPU");
    device.bind_to_thread().unwrap();
    let s = device.new_stream().unwrap();
    for (hidden, kv_heads, d) in FORMAS {
        confere_forma(&s, hidden, kv_heads, d);
        println!("forma (hidden {hidden}, kv_heads {kv_heads}, head_dim {d}) passou nas posições {POSICOES:?}");
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
