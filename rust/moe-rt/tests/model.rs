//! Testes do M1b sem o checkpoint real: configuração, RoPE, RMSNorm, leitor de `.npy` e o modelo inteiro num Granite MoE
//! sintético pequeno (GPU contra CPU, por mais de uma fatia de posições da atenção).
use moe_rt::gpu_model::{GpuModel, ATTN_BN};
use moe_rt::model::{rmsnorm, rope, Config, CpuSession};
use moe_rt::npy::Npy;
use std::fs;

mod common;
use common::{gpu_or_skip, scratch, tiny_config, tiny_model, tiny_model_with, values};

#[test]
fn configuracao_le_o_granite_e_recusa_variantes_nao_suportadas() {
    let c = Config::from_json(&tiny_config(33)).unwrap();
    assert_eq!((c.hidden, c.heads, c.kv_heads, c.head_dim, c.vocab), (256, 4, 2, 64, 33));
    let mut v = tiny_config(33);
    v["model_type"] = "llama".into();
    assert!(Config::from_json(&v).is_err());
    let mut v = tiny_config(33);
    v["rope_scaling"] = serde_json::json!({"type": "linear", "factor": 2.0});
    assert!(Config::from_json(&v).is_err());
    let mut v = tiny_config(33);
    v["attention_bias"] = true.into();
    assert!(Config::from_json(&v).is_err());
}

#[test]
fn rope_gira_as_metades_de_cada_cabeca_e_posicao_zero_e_identidade() {
    let c = Config::from_json(&tiny_config(33)).unwrap();
    let x = values(2 * c.head_dim, 5);
    let mut y = x.clone();
    rope(&mut y, c.head_dim, &c.rope_cos_sin(0));
    assert_eq!(x, y);

    let pos = 7;
    let cs = c.rope_cos_sin(pos);
    let mut y = x.clone();
    rope(&mut y, c.head_dim, &cs);
    let half = c.head_dim / 2;
    for head in 0..2 {
        for j in 0..half {
            let a = pos as f64 * (1.0 / 10000f64.powf((2 * j) as f64 / c.head_dim as f64));
            let (x1, x2) = (x[head * c.head_dim + j] as f64, x[head * c.head_dim + j + half] as f64);
            assert!((y[head * c.head_dim + j] as f64 - (x1 * a.cos() - x2 * a.sin())).abs() < 1e-5);
            assert!((y[head * c.head_dim + j + half] as f64 - (x2 * a.cos() + x1 * a.sin())).abs() < 1e-5);
        }
    }
}

#[test]
fn rmsnorm_normaliza_pela_raiz_da_media_dos_quadrados() {
    let y = rmsnorm(&[3.0, 4.0], &[1.0, 2.0], 0.0);
    let r = (12.5f32).sqrt();
    assert!((y[0] - 3.0 / r).abs() < 1e-6 && (y[1] - 8.0 / r).abs() < 1e-6);
}

#[test]
fn leitor_npy_le_f32_e_i32_e_recusa_dtype_errado() {
    let tmp = scratch("npy");
    let write = |descr: &str, shape: &str, data: &[u8]| {
        let mut head = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape}, }}");
        while (10 + head.len() + 1) % 64 != 0 {
            head.push(' ');
        }
        head.push('\n');
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(head.len() as u16).to_le_bytes());
        out.extend_from_slice(head.as_bytes());
        out.extend_from_slice(data);
        fs::write(&tmp.0, out).unwrap();
    };
    let f: Vec<u8> = [1.5f32, -2.0, 3.25, 0.0, 7.0, 8.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    write("<f4", "(2, 3)", &f);
    let n = Npy::read(&tmp.0).unwrap();
    assert_eq!(n.shape, vec![2, 3]);
    assert_eq!(n.f32().unwrap(), vec![1.5, -2.0, 3.25, 0.0, 7.0, 8.0]);
    assert!(n.i32().is_err());
    let i: Vec<u8> = [4i32, -1, 9].iter().flat_map(|v| v.to_le_bytes()).collect();
    write("<i4", "(3,)", &i);
    assert_eq!(Npy::read(&tmp.0).unwrap().i32().unwrap(), vec![4, -1, 9]);
}

#[test]
fn leitor_npy_recusa_arquivo_truncado_ou_malformado_sem_entrar_em_panico() {
    let tmp = scratch("npy-malformado");
    let head = |h: &str| {
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        out.extend_from_slice(h.as_bytes());
        out
    };
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("v2 com 10 bytes", b"\x93NUMPY\x02\x00\x00\x00".to_vec()),
        ("v3 com 11 bytes", b"\x93NUMPY\x03\x00\x00\x00\x00".to_vec()),
        ("v1 com cabeçalho além do fim", {
            let mut v = b"\x93NUMPY\x01\x00".to_vec();
            v.extend_from_slice(&200u16.to_le_bytes());
            v.extend_from_slice(b"{'descr': '<f4'");
            v
        }),
        ("shape vazio no fim", head("{'descr': '<f4', 'fortran_order': False, 'shape': ")),
        ("forma que estoura", {
            let mut v = head(&format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}, 8), }}", usize::MAX));
            v.extend_from_slice(&[0; 8]);
            v
        }),
    ];
    for (what, bytes) in cases {
        fs::write(&tmp.0, bytes).unwrap();
        let r = std::panic::catch_unwind(|| Npy::read(&tmp.0).and_then(|n| n.f32()));
        let r = r.unwrap_or_else(|_| panic!("pânico em vez de erro: {what}"));
        assert!(r.is_err(), "aceitou: {what}");
    }
}

#[test]
fn modelo_na_gpu_recusa_no_upload_dimensao_de_tile_que_nao_e_potencia_de_dois() {
    let Some(device) = gpu_or_skip() else { return };
    // 6 cabeças com H = 256: head_dim = 42, que a atenção e o RoPE usariam como tile e o cuTile só recusaria no JIT.
    let mut json = tiny_config(33);
    json["num_attention_heads"] = 6.into();
    let (_tmp, m) = tiny_model_with("gpu-head-42", json);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| GpuModel::upload(&m, device.new_stream().unwrap(), ATTN_BN).is_ok()));
    let msg = r.err().map(|p| p.downcast_ref::<String>().cloned().unwrap_or_default()).expect("upload deveria recusar head_dim = 42");
    assert!(msg.contains("potência de dois") && msg.contains("head_dim=42"), "{msg}");
}

/// Granite sintético com H = 768 = 3·256 (12 cabeças de Q e 4 de KV de 64): a forma do 3B (H = 1536 = 3·512) em miniatura.
fn config_768() -> serde_json::Value {
    let mut json = tiny_config(33);
    json["hidden_size"] = 768.into();
    json["num_attention_heads"] = 12.into();
    json["num_key_value_heads"] = 4.into();
    json
}

#[test]
fn modelo_na_gpu_bate_com_a_cpu_com_hidden_que_nao_e_potencia_de_dois() {
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model_with("gpu-768", config_768());
    let cpu = moe_rt::cpu::Cpu::new(3);
    let residente = GpuModel::upload(&m, device.new_stream().unwrap(), ATTN_BN).unwrap();
    let sync = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, CacheConfig { slots: 3, policy: Policy::LruSync, backlog: 2, cpu_threads: 2, admit: AdmitMode::Event }).unwrap();
    let viacpu = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, CacheConfig { slots: 3, policy: Policy::LruCpu, backlog: 2, cpu_threads: 2, admit: AdmitMode::Event }).unwrap();
    for (name, mut g) in [("residente", residente), ("lru_sync", sync), ("lru_cpu", viacpu)] {
        let mut c = CpuSession::new(&m, &cpu);
        for t in 0..12usize {
            let tok = (t * 11 + 5) % m.cfg.vocab;
            let (lg, rg) = g.step(tok).unwrap();
            let (lc, rc) = c.step(tok);
            let num: f64 = lg.iter().zip(&lc).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
            let den: f64 = lc.iter().map(|b| (*b as f64).powi(2)).sum();
            assert!((num / den).sqrt() < 1e-5, "{name}, posição {t}: erro relativo {:.2e}", (num / den).sqrt());
            assert!(rg.iter().zip(&rc).all(|(a, b)| a.experts == b.experts), "{name}, posição {t}: roteamento");
        }
    }
}

#[test]
fn atencao_recusa_part_com_menos_blocos_que_a_grade_pede() {
    use cutile::prelude::*;
    use cutile::tensor::Reshape;
    use moe_rt::gpu_model::{attention, ATTN_BN};
    use std::sync::Arc;
    let Some(device) = gpu_or_skip() else { return };
    let s = device.new_stream().unwrap();
    let (heads, kv_heads, d, ns, max_len) = (4usize, 2usize, 64usize, 8usize, 2 * ATTN_BN);
    let z = |shape: &[usize]| api::zeros::<f32>(&[shape.iter().product()]).sync_on(&s).unwrap().reshape(shape).unwrap();
    let (q, kc, vc) = (z(&[heads, d]), z(&[max_len, kv_heads * d]), z(&[max_len, kv_heads * d]));
    let mut out = z(&[heads, d]);
    // `part` dimensionado para ns = 4: com ns = 8 a 1ª passada escreveria o dobro de linhas.
    let mut part = z(&[heads * 4 * 2, d]);
    let lens = api::copy_host_vec_to_device(&Arc::new(vec![1i32])).sync_on(&s).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = unsafe { attention(&s, &mut out, &mut part, &q, &kc, &vc, &lens, 0.125, (heads, kv_heads, d, ns)) };
    }));
    let msg = r.expect_err("attention aceitou part com forma errada");
    let msg = msg.downcast_ref::<String>().cloned().or_else(|| msg.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
    assert!(msg.contains("atenção: forma de part"), "pânico com outra mensagem: {msg}");
}

#[test]
fn atencao_em_duas_passadas_bate_com_a_referencia_em_toda_fronteira_de_fatia() {
    use cutile::prelude::*;
    use cutile::tensor::Reshape;
    use moe_rt::gpu_model::{attention, ATTN_BN, ATTN_NS};
    use std::sync::Arc;
    let Some(device) = gpu_or_skip() else { return };
    let s = device.new_stream().unwrap();
    let (heads, kv_heads, d) = (4usize, 2usize, 64usize);
    let (g, kvd, b) = (heads / kv_heads, kv_heads * d, ATTN_BN);
    let scale = 0.125f32;
    let dev = |v: &[f32], shape: &[usize]| api::copy_host_vec_to_device(&Arc::new(v.to_vec())).sync_on(&s).unwrap().reshape(shape).unwrap();
    // (blocos por cabeça, max_len, posições). Com ns = 8 e até 32 fatias de BN, a divisão por bloco é desigual (9 fatias: 2
    // por bloco, 5 blocos com posições, o último com uma); com ns = 64 (o do modelo), 4.097 posições dão 2 fatias por bloco.
    let configs: [(usize, usize, Vec<usize>); 2] = [
        (8, 32 * b, vec![1, 2, b - 1, b, b + 1, 8 * b, 8 * b + 1, 9 * b - 1, 3 * b + 7, 17 * b + 5, 32 * b - 1, 32 * b]),
        (ATTN_NS, 128 * b, vec![1, b + 1, 64 * b, 64 * b + 1, 67 * b + 9, 128 * b]),
    ];
    for (ns, max_len, lens_list) in configs {
        // Posições depois de `len` ficam com valores finitos quaisquer: a máscara tem de escondê-las. Na versão ampliada, `q` ×4
        // e as posições de uma fatia no meio ×4 dão scores de dezenas e uma fatia dominante depois da primeira.
        for amp in [false, true] {
            let mut qv = values(heads * d, 11);
            let mut kv = values(max_len * kvd, 12);
            let vv = values(max_len * kvd, 13);
            if amp {
                qv.iter_mut().for_each(|x| *x *= 4.0);
                let mid = (max_len / 3 / b) * b;
                kv[mid * kvd..(mid + b) * kvd].iter_mut().for_each(|x| *x *= 4.0);
            }
            let q = dev(&qv, &[heads, d]);
            let (kc, vc) = (dev(&kv, &[max_len, kvd]), dev(&vv, &[max_len, kvd]));
            let mut part = dev(&vec![0.0; heads * ns * 2 * d], &[heads * ns * 2, d]);
            let mut out = dev(&vec![0.0; heads * d], &[heads, d]);
            for &len in &lens_list {
                let lens = api::copy_host_vec_to_device(&Arc::new(vec![len as i32])).sync_on(&s).unwrap();
                unsafe { attention(&s, &mut out, &mut part, &q, &kc, &vc, &lens, scale, (heads, kv_heads, d, ns)).unwrap() };
                unsafe { s.synchronize().unwrap() };
                let got = out.dup().to_host_vec().sync_on(&s).unwrap();
                let ctx = format!("ns {ns}, max_len {max_len}, ampliado {amp}, len {len}");
                for h in 0..heads {
                    let kvh = h / g;
                    let sc: Vec<f64> = (0..len)
                        .map(|t| (0..d).map(|i| qv[h * d + i] as f64 * kv[t * kvd + kvh * d + i] as f64).sum::<f64>() * scale as f64)
                        .collect();
                    let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let p: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                    let z: f64 = p.iter().sum();
                    for i in 0..d {
                        let want = (0..len).map(|t| p[t] * vv[t * kvd + kvh * d + i] as f64).sum::<f64>() / z;
                        let err = (got[h * d + i] as f64 - want).abs();
                        assert!(err < 1e-5, "{ctx}, cabeça {h}, coluna {i}: {} contra {want} (|Δ| {err:.1e})", got[h * d + i]);
                    }
                }
                // Blocos sem posições: saída 0 e lse = −∞ (a segunda passada não os lê, mas não podem ter NaN).
                let nb = len.div_ceil(b);
                let used = nb.div_ceil(nb.div_ceil(ns));
                let pv = part.dup().to_host_vec().sync_on(&s).unwrap();
                for h in 0..heads {
                    for sp in used..ns {
                        let row = (h * ns + sp) * 2;
                        assert!(pv[row * d..(row + 1) * d].iter().all(|&x| x == 0.0), "{ctx}: bloco vazio {sp} com saída");
                        assert!(pv[(row + 1) * d..(row + 2) * d].iter().all(|&x| x == f32::NEG_INFINITY), "{ctx}: bloco vazio {sp} sem lse = −∞");
                    }
                }
            }
        }
    }
}

#[test]
fn modelo_na_gpu_bate_com_a_cpu_por_mais_de_uma_fatia_da_atencao() {
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model("gpu-modelo");
    let steps = ATTN_BN + 6;
    let mut g = GpuModel::upload(&m, device.new_stream().unwrap(), 2 * ATTN_BN).unwrap();
    let cpu = moe_rt::cpu::Cpu::new(3);
    let mut c = CpuSession::new(&m, &cpu);
    let toks: Vec<usize> = values(steps, 77).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).collect();
    for (t, &tok) in toks.iter().enumerate() {
        let (lg, rg) = g.step(tok).unwrap();
        let (lc, rc) = c.step(tok);
        assert_eq!(lg.len(), m.cfg.vocab);
        let num: f64 = lg.iter().zip(&lc).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = lc.iter().map(|b| (*b as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        assert!(rel < 1e-5, "posição {t}: erro relativo {rel:.2e}");
        let same: Vec<bool> = rg.iter().zip(&rc).map(|(a, b)| a.experts == b.experts).collect();
        assert!(same.iter().all(|&x| x), "posição {t}: roteamento difere {same:?}");
    }
    // Depois de reset, a mesma sequência dá os mesmos logits (o cache antigo não vaza pela máscara).
    g.reset();
    let mut c = CpuSession::new(&m, &cpu);
    for &tok in &toks[..3] {
        let (lg, _) = g.step(tok).unwrap();
        let (lc, _) = c.step(tok);
        let mx = lg.iter().zip(&lc).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(mx < 1e-4, "depois de reset: |Δ| {mx}");
    }
}

/// Cache com orçamento pequeno: os logits têm de bater com a CPU, e os contadores têm de somar os experts pedidos.
fn cached_matches_cpu(policy: moe_rt::cache::Policy, slots: usize, backlog: usize) -> moe_rt::cache::CacheStats {
    use moe_rt::cache::{AdmitMode, CacheConfig};
    use moe_rt::gpu_model::Experts;
    let device = cutile::prelude::Device::new(0).expect("GPU");
    let (_tmp, m) = tiny_model(&format!("cache-{policy:?}-{slots}-{backlog}"));
    let cfg = CacheConfig { slots, policy, backlog, cpu_threads: 3, admit: AdmitMode::Event };
    let mut g = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
    let cpu = moe_rt::cpu::Cpu::new(3);
    let mut c = CpuSession::new(&m, &cpu);
    let steps = 40;
    let toks: Vec<usize> = values(steps, 91).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).collect();
    for (t, &tok) in toks.iter().enumerate() {
        let (lg, rg) = g.step(tok).unwrap();
        let (lc, rc) = c.step(tok);
        let num: f64 = lg.iter().zip(&lc).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = lc.iter().map(|b| (*b as f64).powi(2)).sum();
        assert!((num / den).sqrt() < 1e-5, "{policy:?} {slots} slots, posição {t}");
        assert!(rg.iter().zip(&rc).all(|(a, b)| a.experts == b.experts), "{policy:?} {slots} slots, posição {t}: roteamento");
    }
    match &g.experts {
        Experts::Cached(cache) => {
            let s = cache.stats.clone();
            assert_eq!(s.hits + s.sync_copies + s.cpu, (steps * m.cfg.layers * m.cfg.k) as u64, "{s:?}");
            assert!(cache.resident() + cache.loading_now() <= slots);
            s
        }
        _ => unreachable!(),
    }
}

#[test]
fn cache_lru_sync_bate_com_a_cpu_e_so_copia_quando_falta() {
    if gpu_or_skip().is_none() {
        return;
    }
    // 2 camadas × 4 experts = 8; com 3 slots há despejo a todo token.
    let s = cached_matches_cpu(moe_rt::cache::Policy::LruSync, 3, 2);
    assert_eq!(s.cpu, 0, "{s:?}");
    assert!(s.sync_copies > 0 && s.hits > 0, "{s:?}");
    // Com todos os slots, só as primeiras ocorrências de cada expert copiam.
    let s = cached_matches_cpu(moe_rt::cache::Policy::LruSync, 8, 2);
    assert!(s.sync_copies <= 8, "{s:?}");
    // Orçamento menor que k: o que não cabe na camada vai para a CPU.
    let s = cached_matches_cpu(moe_rt::cache::Policy::LruSync, 1, 2);
    assert!(s.cpu > 0, "{s:?}");
}

#[test]
fn cache_lru_cpu_bate_com_a_cpu_e_admite_em_segundo_plano() {
    if gpu_or_skip().is_none() {
        return;
    }
    let s = cached_matches_cpu(moe_rt::cache::Policy::LruCpu, 4, 2);
    assert_eq!(s.sync_copies, 0, "{s:?}");
    assert!(s.cpu > 0 && s.bg_copies > 0 && s.hits > 0, "{s:?}");
    // Sem slots, tudo na CPU e nenhuma admissão.
    let s = cached_matches_cpu(moe_rt::cache::Policy::LruCpu, 0, 2);
    assert_eq!((s.hits, s.bg_copies), (0, 0), "{s:?}");
}

/// Com a admissão em barreira, o `lru_cpu` não depende do tempo: duas execuções dão os mesmos logits bit a bit e os mesmos
/// contadores, e os logits continuam batendo com a CPU.
#[test]
fn cache_lru_cpu_com_barreira_repete_logits_e_contadores() {
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    use moe_rt::gpu_model::Experts;
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model("cache-barreira");
    let toks: Vec<usize> = values(40, 93).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).collect();
    let cpu = moe_rt::cpu::Cpu::new(3);
    let mut c = CpuSession::new(&m, &cpu);
    let want: Vec<Vec<f32>> = toks.iter().map(|&t| c.step(t).0).collect();
    let run = || {
        let cfg = CacheConfig { slots: 4, policy: Policy::LruCpu, backlog: 2, cpu_threads: 3, admit: AdmitMode::Barrier };
        let mut g = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
        let bits: Vec<Vec<u32>> = toks.iter().map(|&t| g.step(t).unwrap().0.iter().map(|x| x.to_bits()).collect()).collect();
        let Experts::Cached(cache) = &g.experts else { unreachable!() };
        let s = &cache.stats;
        (bits, (s.hits, s.sync_copies, s.bg_copies, s.bg_dropped, s.cpu))
    };
    let (a, sa) = run();
    let (b, sb) = run();
    assert_eq!(sa, sb, "contadores (acertos, cópias, em segundo plano, descartadas, CPU)");
    assert!(sa.0 > 0 && sa.2 > 0 && sa.4 > 0, "precisa de acertos, admissões e CPU: {sa:?}");
    for (t, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(x == y, "posição {t}: logits diferem entre as execuções");
        let lg: Vec<f32> = x.iter().map(|&u| f32::from_bits(u)).collect();
        let num: f64 = lg.iter().zip(&want[t]).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let den: f64 = want[t].iter().map(|q| (*q as f64).powi(2)).sum();
        assert!((num / den).sqrt() < 1e-5, "posição {t}: longe da CPU");
    }
}

/// LRU de referência com a regra certa: todo expert pedido na camada fica protegido de despejo nessa camada, inclusive os
/// acertos que vêm depois de uma falta na ordem do roteador. Mesmo desempate do cache (primeiro slot livre; senão o de uso mais
/// antigo, o de menor índice em empate). Devolve (acertos, cópias, CPU).
fn reference_lru(routes: &[Vec<Vec<usize>>], experts: usize, cap: usize) -> (u64, u64, u64) {
    let mut slots: Vec<Option<(usize, u64)>> = vec![None; cap];
    let (mut clock, mut hits, mut copies, mut cpu) = (0u64, 0u64, 0u64, 0u64);
    for step in routes {
        for (l, chosen) in step.iter().enumerate() {
            clock += 1;
            let keys: Vec<usize> = chosen.iter().map(|&e| l * experts + e).collect();
            for s in slots.iter_mut().flatten() {
                if keys.contains(&s.0) {
                    s.1 = clock;
                }
            }
            for &key in &keys {
                if slots.iter().flatten().any(|s| s.0 == key) {
                    hits += 1;
                    continue;
                }
                let free = slots.iter().position(|s| s.is_none());
                let victim = free.or_else(|| {
                    slots.iter().enumerate().filter(|(_, s)| s.unwrap().1 < clock).min_by_key(|(_, s)| s.unwrap().1).map(|(i, _)| i)
                });
                match victim {
                    Some(i) => {
                        slots[i] = Some((key, clock));
                        copies += 1;
                    }
                    None => cpu += 1,
                }
            }
        }
    }
    (hits, copies, cpu)
}

#[test]
fn cache_lru_sync_nao_despeja_acerto_da_mesma_camada_e_segue_o_lru_de_referencia() {
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    use moe_rt::gpu_model::Experts;
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model("cache-lru-ref");
    for slots in [3usize, 5, 6] {
        let cfg = CacheConfig { slots, policy: Policy::LruSync, backlog: 2, cpu_threads: 2, admit: AdmitMode::Event };
        let mut g = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
        let toks: Vec<usize> = values(48, 123).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).collect();
        let mut routes = Vec::new();
        for &tok in &toks {
            let (_, r) = g.step(tok).unwrap();
            routes.push(r.into_iter().map(|x| x.experts).collect::<Vec<_>>());
        }
        let (hits, copies, cpu) = reference_lru(&routes, m.cfg.experts, slots);
        let Experts::Cached(c) = &g.experts else { unreachable!() };
        let s = &c.stats;
        assert_eq!((s.hits, s.sync_copies, s.cpu), (hits, copies, cpu), "{slots} slots: cache {s:?}");
    }
}

#[test]
fn cache_lru_cpu_descarta_admissao_quando_o_backlog_esta_cheio() {
    if gpu_or_skip().is_none() {
        return;
    }
    // Backlog zero: toda falta vai para a CPU e toda admissão é descartada, então nada chega a ficar residente.
    let s = cached_matches_cpu(moe_rt::cache::Policy::LruCpu, 4, 0);
    assert_eq!((s.hits, s.bg_copies, s.sync_copies), (0, 0, 0), "{s:?}");
    assert_eq!(s.bg_dropped, s.cpu, "{s:?}");
    // Backlog um: cada falta vira exatamente uma cópia ou um descarte.
    let s = cached_matches_cpu(moe_rt::cache::Policy::LruCpu, 4, 1);
    assert_eq!(s.bg_copies + s.bg_dropped, s.cpu, "{s:?}");
    assert!(s.bg_copies > 0 && s.hits > 0, "{s:?}");
}

#[test]
fn cache_cheio_preenchido_antes_so_tem_acertos_e_bate_com_a_cpu() {
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    use moe_rt::gpu_model::Experts;
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model("cache-cheio");
    let total = m.cfg.layers * m.cfg.experts;
    let cfg = CacheConfig { slots: total, policy: Policy::LruSync, backlog: 2, cpu_threads: 3, admit: AdmitMode::Event };
    let mut g = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
    let Experts::Cached(c) = &mut g.experts else { unreachable!() };
    assert_eq!(c.fill().unwrap(), total);
    assert_eq!(c.fill().unwrap(), 0, "segunda chamada não tem o que copiar");
    assert_eq!(c.resident(), total);
    let cpu = moe_rt::cpu::Cpu::new(3);
    let mut s = CpuSession::new(&m, &cpu);
    let steps = 20;
    for (t, tok) in values(steps, 17).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).enumerate() {
        let (lg, _) = g.step(tok).unwrap();
        let (lc, _) = s.step(tok);
        let num: f64 = lg.iter().zip(&lc).map(|(a, b)| ((a - b) as f64).powi(2)).sum();
        let den: f64 = lc.iter().map(|b| (*b as f64).powi(2)).sum();
        assert!((num / den).sqrt() < 1e-5, "posição {t}");
    }
    let Experts::Cached(c) = &g.experts else { unreachable!() };
    assert_eq!((c.stats.hits, c.stats.sync_copies, c.stats.cpu), ((steps * m.cfg.layers * m.cfg.k) as u64, 0, 0));
}

#[test]
fn passo_com_token_fora_do_vocabulario_ou_cache_kv_cheio_devolve_erro_e_o_modelo_segue_utilizavel() {
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model("gpu-erros");
    let mut g = GpuModel::upload(&m, device.new_stream().unwrap(), ATTN_BN).unwrap();
    let cpu = moe_rt::cpu::Cpu::new(3);
    let mut c = CpuSession::new(&m, &cpu);
    assert!(g.step(m.cfg.vocab).is_err(), "token fora do vocabulário");
    // O erro de pré-condição não envenena: os passos seguintes batem com a CPU.
    let toks: Vec<usize> = (0..ATTN_BN).map(|t| (t * 7 + 3) % m.cfg.vocab).collect();
    for &tok in &toks {
        let (lg, _) = g.step(tok).unwrap();
        let (lc, _) = c.step(tok);
        let mx = lg.iter().zip(&lc).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(mx < 1e-4, "|Δ| {mx}");
    }
    assert!(g.step(1).is_err(), "cache KV cheio");
    g.reset();
    assert!(g.step(1).is_ok(), "depois de reset");
}

#[test]
fn cache_largado_com_admissoes_em_voo_nao_quebra_o_processo() {
    // Sem compute-sanitizer isto só confere que soltar o modelo logo depois de admissões em segundo plano não trava nem dá
    // erro; rodado sob `compute-sanitizer --tool memcheck`, pega uma cópia escrevendo em memória já liberada.
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model("cache-largado");
    for round in 0..4 {
        let cfg = CacheConfig { slots: 4, policy: Policy::LruCpu, backlog: 8, cpu_threads: 2, admit: AdmitMode::Event };
        let mut g = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
        for t in 0..=round {
            g.step((t * 5 + round) % m.cfg.vocab).unwrap();
        }
        drop(g);
    }
}

#[test]
fn passo_com_roteamento_dado_de_fora_calcula_o_mesmo_e_as_variantes_parciais_nao_quebram_o_modelo() {
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    use moe_rt::gpu::Count;
    use moe_rt::gpu_model::Probe;
    let Some(device) = gpu_or_skip() else { return };
    let (_tmp, m) = tiny_model("gpu-probe");
    let (layers, k) = (m.cfg.layers, m.cfg.k);
    let mut g = GpuModel::upload(&m, device.new_stream().unwrap(), ATTN_BN).unwrap();
    let toks: Vec<usize> = (0..10).map(|t| (t * 13 + 1) % m.cfg.vocab).collect();
    let normal: Vec<_> = toks.iter().map(|&t| g.step(t).unwrap()).collect();
    assert!(!g.has_timing_events(), "o passo normal criou eventos de medição");

    // Contagem por camada: atenção 13 (RMSNorm, Q/K/V, 2 RoPE, 2 cópias para o KV, as 2 passadas da atenção, saída, AXPBY,
    // RMSNorm do MoE); MoE 4 por expert e o AXPBY residual, mais GEMV e leitura dos logits do roteador (2 sincronizações) sem
    // rotas dadas. Por token: envio do embedding, do cos/sin e do `len`, RMSNorm final, cabeça e leitura dos logits (4
    // sincronizações).
    let per_token = |attn: usize, moe: usize, syncs: usize| Count { ops: layers * (attn + moe) + 6, syncs: layers * syncs + 4 };
    g.reset();
    let (_, _, tm) = g.step_probe(toks[0], Probe::default()).unwrap();
    assert!(g.has_timing_events());
    assert_eq!(tm.count, Some(per_token(13, 3 + 2 * k, 2)), "passo normal");

    for (skip_attn, skip_moe) in [(false, false), (true, false), (false, true), (true, true)] {
        g.reset();
        for (i, &t) in toks.iter().enumerate() {
            let probe = Probe { routes: Some(&normal[i].1), skip_attn, skip_moe };
            let (lg, _, tm) = g.step_probe(t, probe).unwrap();
            assert!(tm.gpu_ms > 0.0 && tm.total_ms >= tm.enqueue_ms);
            let want = per_token(if skip_attn { 0 } else { 13 }, if skip_moe { 0 } else { 1 + 2 * k }, 0);
            assert_eq!(tm.count, Some(want), "skip_attn = {skip_attn}, skip_moe = {skip_moe}");
            if !skip_attn && !skip_moe {
                assert_eq!(lg, normal[i].0, "posição {i}: o roteamento dado de fora mudou os logits");
            }
        }
    }

    // Rotas com menos camadas que o modelo: erro antes de enfileirar, e o modelo continua utilizável.
    g.reset();
    let short = &normal[0].1[..layers - 1];
    assert!(g.step_probe(toks[0], Probe { routes: Some(short), ..Probe::default() }).is_err());

    // Depois das variantes e do erro, o passo normal volta a dar o mesmo.
    g.reset();
    for (i, &t) in toks.iter().enumerate() {
        assert_eq!(g.step(t).unwrap().0, normal[i].0, "posição {i} depois das variantes");
    }

    // Com o cache de experts, rotas dadas de fora são recusadas; sem elas o passo mede, mas não conta as operações do MoE.
    let cfg = CacheConfig { slots: 3, policy: Policy::LruSync, backlog: 2, cpu_threads: 2, admit: AdmitMode::Event };
    let mut c = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
    assert!(c.step_probe(toks[0], Probe { routes: Some(&normal[0].1), ..Probe::default() }).is_err());
    let (lg, _, tm) = c.step_probe(toks[0], Probe::default()).unwrap();
    assert_eq!(lg, normal[0].0);
    assert_eq!(tm.count, None);
}

/// LFU de referência no modo barreira (toda cópia de uma camada vale a partir da seguinte), com a regra do C: todos os experts
/// pedidos na camada são tocados na `Freq` antes das decisões; a falta vai à CPU e, com menos de `backlog` cópias na fila,
/// toma o primeiro slot livre, senão o residente de menor pontuação fora dos pedidos da camada (o de menor índice em empate),
/// e só se a pontuação dele for menor que a da falta. Devolve (acertos, cópias, descartes, CPU, recusas do filtro).
fn reference_lfu(routes: &[Vec<Vec<usize>>], experts: usize, layers: usize, cap: usize, backlog: usize) -> (u64, u64, u64, u64, u64) {
    let mut freq = moe_rt::cache::Freq::new(layers * experts, layers);
    // (chave, residente?) por slot; não residente = carregando nesta camada.
    let mut slots: Vec<Option<(usize, bool)>> = vec![None; cap];
    let (mut step, mut hits, mut copies, mut dropped, mut cpu, mut filtered) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    for tok in routes {
        for (l, chosen) in tok.iter().enumerate() {
            slots.iter_mut().flatten().for_each(|s| s.1 = true);
            let keys: Vec<usize> = chosen.iter().map(|&e| l * experts + e).collect();
            for &k in &keys {
                freq.touch(k, step);
            }
            step += 1;
            for &key in &keys {
                if slots.iter().flatten().any(|s| s.0 == key && s.1) {
                    hits += 1;
                    continue;
                }
                cpu += 1;
                let loading = slots.iter().flatten().filter(|s| !s.1).count();
                let pick = if loading >= backlog {
                    None
                } else if let Some(i) = slots.iter().position(|s| s.is_none()) {
                    Some(i)
                } else {
                    let mut best: Option<(usize, f64)> = None;
                    for (i, s) in slots.iter().enumerate() {
                        let (k, res) = s.unwrap();
                        if res && !keys.contains(&k) {
                            let p = freq.score(k);
                            if best.is_none_or(|(_, b)| p < b) {
                                best = Some((i, p));
                            }
                        }
                    }
                    let admit = best.filter(|&(_, p)| p < freq.score(key)).map(|(i, _)| i);
                    filtered += u64::from(best.is_some() && admit.is_none());
                    admit
                };
                match pick {
                    Some(i) => {
                        slots[i] = Some((key, false));
                        copies += 1;
                    }
                    None => dropped += 1,
                }
            }
        }
    }
    (hits, copies, dropped, cpu, filtered)
}

/// C: o `lfu_cpu` no modo barreira segue o LFU de referência contador a contador, bate com a CPU e filtra admissões (a
/// referência conta as recusas do filtro, e os contadores iguais mostram que o cache recusou as mesmas).
#[test]
fn cache_lfu_cpu_segue_o_lfu_de_referencia_e_bate_com_a_cpu() {
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    use moe_rt::gpu_model::Experts;
    let Some(device) = gpu_or_skip() else { return };
    // 16 experts por camada (2 por token): com experts raros, a falta nem sempre vence o residente mais fraco.
    let mut json = tiny_config(33);
    json["num_local_experts"] = 16.into();
    let (_tmp, m) = tiny_model_with("cache-lfu-ref", json);
    let cpu = moe_rt::cpu::Cpu::new(3);
    // Duas conversas de 48 tokens (o KV do modelo de teste tem 64 posições); o cache persiste entre elas, como no m2a.
    let toks: Vec<usize> = values(96, 131).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).collect();
    let want: Vec<Vec<f32>> = toks
        .chunks(48)
        .flat_map(|conv| {
            let mut c = CpuSession::new(&m, &cpu);
            conv.iter().map(|&t| c.step(t).0).collect::<Vec<_>>()
        })
        .collect();
    let mut filtrou = 0;
    for (slots, backlog) in [(3usize, 2usize), (8, 8), (12, 1), (6, 4)] {
        let cfg = CacheConfig { slots, policy: Policy::LfuCpu, backlog, cpu_threads: 3, admit: AdmitMode::Barrier };
        let mut g = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
        let mut routes = Vec::new();
        for (t, &tok) in toks.iter().enumerate() {
            if t % 48 == 0 {
                g.reset();
            }
            let (lg, r) = g.step(tok).unwrap();
            let num: f64 = lg.iter().zip(&want[t]).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
            let den: f64 = want[t].iter().map(|q| (*q as f64).powi(2)).sum();
            assert!((num / den).sqrt() < 1e-5, "{slots} slots, posição {t}: longe da CPU");
            routes.push(r.into_iter().map(|x| x.experts).collect::<Vec<_>>());
        }
        let (hits, copies, dropped, ncpu, filtered) = reference_lfu(&routes, m.cfg.experts, m.cfg.layers, slots, backlog);
        let Experts::Cached(cache) = &g.experts else { unreachable!() };
        let s = &cache.stats;
        assert_eq!((s.hits, s.bg_copies, s.bg_dropped, s.cpu, s.sync_copies), (hits, copies, dropped, ncpu, 0), "{slots} slots: cache {s:?}");
        assert!(s.hits > 0 && s.bg_copies > 0, "{slots} slots: {s:?}");
        eprintln!("lfu_cpu {slots} slots, backlog {backlog}: {s:?}; recusas do filtro {filtered}");
        filtrou += filtered;
    }
    assert!(filtrou > 0, "nenhuma recusa do filtro de frequência: o teste não exercitou a admissão do LFU");
}

/// C: o `lfu_cpu` com admissão por evento bate com a CPU e respeita o orçamento, e o `clear` zera as frequências: depois de
/// uma sequência A e do `clear`, nenhuma chave tem frequência, e a sequência B dá os mesmos contadores que num cache novo
/// (modo barreira). O roteador do modelo de teste usa só 4 ou 5 chaves, pouco para a frequência velha aparecer nos
/// contadores; por isso a checagem direta da `Freq`. O toque do expert que ainda está carregando fica em `tests/portao.rs`.
#[test]
fn cache_lfu_cpu_por_evento_bate_com_a_cpu_e_clear_zera_as_frequencias() {
    use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
    use moe_rt::gpu_model::Experts;
    let s = cached_matches_cpu(Policy::LfuCpu, 4, 2);
    assert_eq!(s.sync_copies, 0, "{s:?}");
    assert!(s.cpu > 0 && s.hits > 0, "{s:?}");
    let Some(device) = gpu_or_skip() else { return };
    let mut json = tiny_config(33);
    json["num_local_experts"] = 16.into();
    let (_tmp, m) = tiny_model_with("cache-lfu-clear", json);
    let seq = |seed: u64| -> Vec<usize> { values(48, seed).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).collect() };
    let (a, b) = (seq(141), seq(143));
    let novo = || {
        let cfg = CacheConfig { slots: 6, policy: Policy::LfuCpu, backlog: 4, cpu_threads: 2, admit: AdmitMode::Barrier };
        GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap()
    };
    let keys = m.cfg.layers * m.cfg.experts;
    let roda = |g: &mut GpuModel, toks: &[usize]| {
        g.reset();
        for &t in toks {
            g.step(t).unwrap();
        }
        let Experts::Cached(cache) = &mut g.experts else { unreachable!() };
        assert!((0..keys).any(|k| cache.freq().score(k).is_finite()), "nenhuma frequência depois da sequência");
        let s = &cache.stats;
        let r = (s.hits, s.bg_copies, s.bg_dropped, s.cpu);
        cache.clear().unwrap();
        assert!((0..keys).all(|k| cache.freq().score(k) == f64::NEG_INFINITY), "o clear deixou frequências");
        cache.stats = Default::default();
        r
    };
    let fresco = roda(&mut novo(), &b);
    let mut g = novo();
    roda(&mut g, &a);
    let depois = roda(&mut g, &b);
    assert_eq!(depois, fresco, "B depois de A e do clear diferiu de B num cache novo: frequências ou slots sobraram");
}
