//! Ajudantes compartilhados pelos testes: diretório temporário, valores determinísticos, escrita de safetensors e o Granite
//! MoE sintético pequeno.
#![allow(dead_code)]
use cutile::half::bf16;
use moe_rt::model::{Config, HostModel};
use moe_rt::weights::Checkpoint;
use std::fs;
use std::path::{Path, PathBuf};

/// Diretório temporário com um único arquivo; removido sem recursão no fim.
pub struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        if let Some(dir) = self.0.parent() {
            let _ = fs::remove_dir(dir);
        }
    }
}

pub fn scratch(name: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("moe-rt-test-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    Scratch(dir.join("model.safetensors"))
}

pub fn values(n: usize, seed: u64) -> Vec<f32> {
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

pub fn bf(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

/// Escreve um safetensors com os tensores dados (nome, dtype, forma, bytes).
pub fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut head = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        head.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let head = serde_json::to_vec(&serde_json::Value::Object(head)).unwrap();
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    out.extend_from_slice(&data);
    fs::write(path, out).unwrap();
}

pub fn bytes(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|b| b.to_bits().to_le_bytes()).collect()
}

/// Dispositivo 0, ou `None` (teste pulado, com aviso) se não houver GPU. Com `MOE_RT_REQUIRE_GPU=1`, a falta de GPU é erro.
pub fn gpu_or_skip() -> Option<std::sync::Arc<cutile::prelude::Device>> {
    match cutile::prelude::Device::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
            eprintln!("sem GPU, teste pulado: {e:?}");
            None
        }
    }
}

pub fn tiny_config(vocab: usize) -> serde_json::Value {
    serde_json::json!({
        "model_type": "granitemoe", "hidden_size": 256, "intermediate_size": 128, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2, "vocab_size": vocab, "num_local_experts": 4,
        "num_experts_per_tok": 2, "rope_theta": 10000.0, "rms_norm_eps": 1e-6, "embedding_multiplier": 2.0,
        "residual_multiplier": 0.5, "attention_multiplier": 0.125, "logits_scaling": 4.0, "rope_scaling": null,
        "attention_bias": false, "tie_word_embeddings": true
    })
}

/// Granite MoE sintético: H=256, I=128, 2 camadas, 4 cabeças de Q e 2 de KV de 64, 4 experts com 2 ativos, vocabulário ímpar.
pub fn tiny_model(name: &str) -> (Scratch, HostModel) {
    tiny_model_with(name, tiny_config(33))
}

pub fn tiny_model_with(name: &str, json: serde_json::Value) -> (Scratch, HostModel) {
    tiny_model_attn(name, json, 0.1)
}

/// Como `tiny_model_with`, com a escala dos pesos q/k/v/o da atenção (0,1 deixa o contexto quase sem efeito nos logits).
pub fn tiny_model_attn(name: &str, json: serde_json::Value, attn: f32) -> (Scratch, HostModel) {
    let cfg = Config::from_json(&json).unwrap();
    let tmp = scratch(name);
    let (h, i, e, d) = (cfg.hidden, cfg.inter, cfg.experts, cfg.head_dim);
    let (qd, kvd) = (cfg.heads * d, cfg.kv_heads * d);
    let mut seed = 10u64;
    let mut w = |n: usize, scale: f32| {
        seed += 1;
        bf(&values(n, seed).into_iter().map(|x| x * scale).collect::<Vec<_>>())
    };
    let norm = |w: Vec<bf16>| w.into_iter().map(|x| bf16::from_f32(1.0 + x.to_f32())).collect::<Vec<_>>();
    let mut t: Vec<(String, Vec<usize>, Vec<bf16>)> = vec![
        ("model.embed_tokens.weight".into(), vec![cfg.vocab, h], w(cfg.vocab * h, 1.0)),
        ("model.norm.weight".into(), vec![h], norm(w(h, 0.1))),
    ];
    for l in 0..cfg.layers {
        let p = format!("model.layers.{l}");
        t.push((format!("{p}.input_layernorm.weight"), vec![h], norm(w(h, 0.1))));
        t.push((format!("{p}.post_attention_layernorm.weight"), vec![h], norm(w(h, 0.1))));
        t.push((format!("{p}.self_attn.q_proj.weight"), vec![qd, h], w(qd * h, attn)));
        t.push((format!("{p}.self_attn.k_proj.weight"), vec![kvd, h], w(kvd * h, attn)));
        t.push((format!("{p}.self_attn.v_proj.weight"), vec![kvd, h], w(kvd * h, attn)));
        t.push((format!("{p}.self_attn.o_proj.weight"), vec![h, qd], w(h * qd, attn)));
        t.push((format!("{p}.block_sparse_moe.router.layer.weight"), vec![e, h], w(e * h, 0.5)));
        t.push((format!("{p}.block_sparse_moe.input_linear.weight"), vec![e, 2 * i, h], w(e * 2 * i * h, 0.1)));
        t.push((format!("{p}.block_sparse_moe.output_linear.weight"), vec![e, h, i], w(e * h * i, 0.1)));
    }
    let tensors: Vec<(&str, &str, Vec<usize>, Vec<u8>)> = t.iter().map(|(n, s, v)| (n.as_str(), "BF16", s.clone(), bytes(v))).collect();
    write_safetensors(&tmp.0, &tensors);
    let ck = Checkpoint::open(tmp.0.parent().unwrap()).unwrap();
    let m = HostModel::load(&ck, cfg).unwrap();
    (tmp, m)
}
