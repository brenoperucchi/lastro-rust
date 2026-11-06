//! Modelo Granite MoE completo (`GraniteMoeForCausalLM` do transformers), pesos na RAM e passo de decode na CPU.
//!
//! Por camada: `h += r·atenção(rmsnorm(h))`, `h += r·moe(rmsnorm(h))`, com `r = residual_multiplier`. A atenção tem cabeças de
//! KV agrupadas, RoPE no estilo `rotate_half` (metades, não pares intercalados) e escala `attention_multiplier` no lugar de
//! `1/√d`. A entrada é o embedding vezes `embedding_multiplier`; a saída usa o embedding transposto (pesos amarrados) e divide
//! os logits por `logits_scaling`.
//!
//! Este caminho é a referência interna do runtime: o da GPU é conferido contra ele e contra o transformers.
use crate::block::{HostLayer, Routing};
use crate::cpu::Cpu;
use crate::weights::{Checkpoint, WeightError};
use cutile::half::bf16;

#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub hidden: usize,
    pub inter: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub experts: usize,
    pub k: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub embedding_multiplier: f32,
    pub residual_multiplier: f32,
    pub attention_multiplier: f32,
    pub logits_scaling: f32,
}

impl Config {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let u = |k: &str| v[k].as_u64().map(|x| x as usize).ok_or(format!("config.json sem {k}"));
        let f = |k: &str| v[k].as_f64().map(|x| x as f32).ok_or(format!("config.json sem {k}"));
        if v["model_type"].as_str() != Some("granitemoe") {
            return Err(format!("model_type {:?} (esperava granitemoe)", v["model_type"]));
        }
        if !v["rope_scaling"].is_null() || v["attention_bias"].as_bool() == Some(true) || v["tie_word_embeddings"].as_bool() == Some(false) {
            return Err("variante não suportada (rope_scaling, attention_bias ou embeddings não amarrados)".into());
        }
        let (hidden, heads) = (u("hidden_size")?, u("num_attention_heads")?);
        Ok(Self {
            hidden,
            inter: u("intermediate_size")?,
            layers: u("num_hidden_layers")?,
            heads,
            kv_heads: u("num_key_value_heads")?,
            head_dim: v["head_dim"].as_u64().map(|x| x as usize).unwrap_or(hidden / heads),
            vocab: u("vocab_size")?,
            experts: u("num_local_experts")?,
            k: u("num_experts_per_tok")?,
            rope_theta: f("rope_theta")?,
            eps: f("rms_norm_eps")?,
            embedding_multiplier: f("embedding_multiplier")?,
            residual_multiplier: f("residual_multiplier")?,
            attention_multiplier: f("attention_multiplier")?,
            logits_scaling: f("logits_scaling")?,
        })
    }

    /// `inv_freq[j] = 1 / θ^(2j/d)`, em f32 como no transformers.
    pub fn inv_freq(&self) -> Vec<f32> {
        let d = self.head_dim as f32;
        (0..self.head_dim / 2).map(|j| 1.0 / self.rope_theta.powf((2 * j) as f32 / d)).collect()
    }

    /// `[cos; sin]` da posição, metade da dimensão da cabeça cada (a outra metade repete).
    pub fn rope_cos_sin(&self, pos: usize) -> Vec<f32> {
        let f: Vec<f32> = self.inv_freq().iter().map(|w| pos as f32 * w).collect();
        f.iter().map(|a| a.cos()).chain(f.iter().map(|a| a.sin())).collect()
    }
}

pub struct HostDecoder {
    pub ln_attn: Vec<f32>,
    pub ln_moe: Vec<f32>,
    /// `[heads·d, H]`, `[kv·d, H]`, `[kv·d, H]`, `[H, heads·d]`
    pub q: Vec<bf16>,
    pub k: Vec<bf16>,
    pub v: Vec<bf16>,
    pub o: Vec<bf16>,
    pub moe: HostLayer,
}

pub struct HostModel {
    pub cfg: Config,
    /// `[V, H]`, também a cabeça de saída.
    pub embed: Vec<bf16>,
    pub layers: Vec<HostDecoder>,
    pub norm: Vec<f32>,
}

fn f32s(v: Vec<bf16>) -> Vec<f32> {
    v.into_iter().map(|b| b.to_f32()).collect()
}

impl HostModel {
    pub fn load(ck: &Checkpoint, cfg: Config) -> Result<Self, WeightError> {
        let get = |name: &str, shape: &[usize]| -> Result<Vec<bf16>, WeightError> {
            let (v, s) = ck.bf16(name)?;
            if s != shape {
                return Err(WeightError::Format(format!("{name}: forma {s:?} (esperava {shape:?})")));
            }
            Ok(v)
        };
        let (h, qd, kvd) = (cfg.hidden, cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("model.layers.{l}");
            let moe = HostLayer::load(ck, l, cfg.k)?;
            if moe.experts() != cfg.experts || moe.inter != cfg.inter || moe.hidden != h {
                return Err(WeightError::Format(format!("camada {l}: bloco MoE difere do config.json")));
            }
            layers.push(HostDecoder {
                ln_attn: f32s(get(&format!("{p}.input_layernorm.weight"), &[h])?),
                ln_moe: f32s(get(&format!("{p}.post_attention_layernorm.weight"), &[h])?),
                q: get(&format!("{p}.self_attn.q_proj.weight"), &[qd, h])?,
                k: get(&format!("{p}.self_attn.k_proj.weight"), &[kvd, h])?,
                v: get(&format!("{p}.self_attn.v_proj.weight"), &[kvd, h])?,
                o: get(&format!("{p}.self_attn.o_proj.weight"), &[h, qd])?,
                moe,
            });
        }
        Ok(Self {
            embed: get("model.embed_tokens.weight", &[cfg.vocab, h])?,
            norm: f32s(get("model.norm.weight", &[h])?),
            layers,
            cfg,
        })
    }

    /// Linha do embedding já multiplicada por `embedding_multiplier`.
    pub fn embedding(&self, token: usize) -> Vec<f32> {
        let h = self.cfg.hidden;
        self.embed[token * h..(token + 1) * h].iter().map(|b| b.to_f32() * self.cfg.embedding_multiplier).collect()
    }
}

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ms = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let r = 1.0 / (ms + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| g * (v * r)).collect()
}

/// RoPE no estilo `rotate_half`, em cada cabeça de `d` elementos de `x`.
pub fn rope(x: &mut [f32], d: usize, cos_sin: &[f32]) {
    let half = d / 2;
    let (c, s) = cos_sin.split_at(half);
    for head in x.chunks_exact_mut(d) {
        for j in 0..half {
            let (x1, x2) = (head[j], head[j + half]);
            head[j] = x1 * c[j] - x2 * s[j];
            head[j + half] = x2 * c[j] + x1 * s[j];
        }
    }
}

/// Estado de uma sequência na CPU: cache KV por camada, `[posição, kv·d]`.
pub struct CpuSession<'m> {
    pub model: &'m HostModel,
    pub cpu: &'m Cpu,
    k_cache: Vec<Vec<f32>>,
    v_cache: Vec<Vec<f32>>,
    pub pos: usize,
}

impl<'m> CpuSession<'m> {
    pub fn new(model: &'m HostModel, cpu: &'m Cpu) -> Self {
        let n = model.cfg.layers;
        Self { model, cpu, k_cache: vec![Vec::new(); n], v_cache: vec![Vec::new(); n], pos: 0 }
    }

    /// Um token: devolve os logits da próxima posição e a decisão do roteador em cada camada.
    /// O cache K e V da camada `l` até a posição corrente (`pos·kv_dim` valores cada): a referência do oráculo do KV do E4b.
    pub fn kv(&self, l: usize) -> (&[f32], &[f32]) {
        (&self.k_cache[l], &self.v_cache[l])
    }

    pub fn step(&mut self, token: usize) -> (Vec<f32>, Vec<Routing>) {
        let m = self.model;
        let c = &m.cfg;
        let (h, d, cpu) = (c.hidden, c.head_dim, self.cpu);
        let (qd, kvd) = (c.heads * d, c.kv_heads * d);
        let group = c.heads / c.kv_heads;
        let cs = c.rope_cos_sin(self.pos);
        let mut x = m.embedding(token);
        let mut routes = Vec::with_capacity(c.layers);
        for (l, layer) in m.layers.iter().enumerate() {
            let xn = rmsnorm(&x, &layer.ln_attn, c.eps);
            let (mut q, mut k, mut v) = (vec![0f32; qd], vec![0f32; kvd], vec![0f32; kvd]);
            cpu.gemv(&layer.q, qd, h, &xn, &mut q);
            cpu.gemv(&layer.k, kvd, h, &xn, &mut k);
            cpu.gemv(&layer.v, kvd, h, &xn, &mut v);
            rope(&mut q, d, &cs);
            rope(&mut k, d, &cs);
            self.k_cache[l].extend_from_slice(&k);
            self.v_cache[l].extend_from_slice(&v);
            let len = self.pos + 1;
            let (kc, vc) = (&self.k_cache[l], &self.v_cache[l]);
            let mut att = vec![0f32; qd];
            for hq in 0..c.heads {
                let kvh = hq / group;
                let qh = &q[hq * d..(hq + 1) * d];
                let scores: Vec<f32> = (0..len)
                    .map(|p| {
                        let kr = &kc[p * kvd + kvh * d..p * kvd + (kvh + 1) * d];
                        qh.iter().zip(kr).map(|(a, b)| a * b).sum::<f32>() * c.attention_multiplier
                    })
                    .collect();
                let mx = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = scores.iter().map(|s| (s - mx).exp()).collect();
                let sum: f32 = e.iter().sum();
                let out = &mut att[hq * d..(hq + 1) * d];
                for (p, w) in e.iter().enumerate() {
                    let vr = &vc[p * kvd + kvh * d..p * kvd + (kvh + 1) * d];
                    for (o, vv) in out.iter_mut().zip(vr) {
                        *o += w / sum * vv;
                    }
                }
            }
            let mut o = vec![0f32; h];
            cpu.gemv(&layer.o, h, qd, &att, &mut o);
            for (xv, ov) in x.iter_mut().zip(&o) {
                *xv += c.residual_multiplier * ov;
            }
            let xn = rmsnorm(&x, &layer.ln_moe, c.eps);
            let (y, r) = layer.moe.forward_cpu(&xn, cpu);
            for (xv, yv) in x.iter_mut().zip(&y) {
                *xv += c.residual_multiplier * yv;
            }
            routes.push(r);
        }
        let xn = rmsnorm(&x, &m.norm, c.eps);
        let mut logits = vec![0f32; c.vocab];
        cpu.gemv(&m.embed, c.vocab, h, &xn, &mut logits);
        for v in logits.iter_mut() {
            *v /= c.logits_scaling;
        }
        self.pos += 1;
        (logits, routes)
    }
}
