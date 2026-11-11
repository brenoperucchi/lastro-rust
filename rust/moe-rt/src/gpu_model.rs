//! O modelo inteiro na GPU, um token por passo (decode em batch 1), com todos os pesos na VRAM.
//!
//! Por token: o embedding sai do host (uma linha), e o resto roda numa stream: RMSNorm, projeções Q/K/V, RoPE, escrita no
//! cache KV (`[posições, kv·d]` por camada, cópia device→device), atenção de decode, projeção de saída e residual; depois o
//! bloco MoE do M1a, que sincroniza uma vez por camada para ler os logits do roteador; no fim RMSNorm e a cabeça de saída.
use crate::block::Routing;
use crate::cache::{CacheConfig, ExpertCache};
use crate::gpu::{kernels, GpuLayer, AXPBY_B, GEMV_BM};
use crate::model::{Config, HostModel};
use cuda_core::Stream;
use crate::gpu::{sync_on_err, Count};
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Reshape;
use std::sync::Arc;

/// Posições do cache KV por fatia da atenção.
pub const ATTN_BN: usize = 64;
/// Blocos por cabeça na primeira passada da atenção. Fixo: a fatia de cada bloco cresce com a posição (`⌈⌈len/ATTN_BN⌉/ns⌉`
/// fatias de `ATTN_BN`), e a grade não depende de `len` nem de `max_len`.
pub const ATTN_NS: usize = 64;

/// Atenção de decode em duas passadas (`attn_split` e `attn_combine` em [`crate::gpu::kernels`]): `q` e `out` com
/// `[heads, d]`, caches `[max_len, kv_heads·d]`, `part` com `[heads·ns·2, d]` (`ns` blocos por cabeça; o modelo usa
/// [`ATTN_NS`]), `lens` com um `i32`, o número de posições válidas.
///
/// # Safety
/// Enfileira sem sincronizar: quem chama mantém os tensores vivos até a próxima sincronização da stream. `lens` tem de estar
/// entre 1 e `max_len`, e as formas têm de ser as de cima: os kernels não conferem limites.
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention(
    s: &Arc<Stream>,
    out: &mut Tensor<f32>,
    part: &mut Tensor<f32>,
    q: &Tensor<f32>,
    kc: &Tensor<f32>,
    vc: &Tensor<f32>,
    lens: &Tensor<i32>,
    scale: f32,
    dims: (usize, usize, usize, usize),
) -> Result<(), Error> {
    attention_b(s, out, part, q, kc, vc, lens, scale, dims, None)
}

/// [`attention`] com a F2 nível B: com `acc` presente carimba `B_ATTN_PREP` (asserts, `dim` e `gen_split`, até antes do primeiro lançamento) e
/// `B_LANCA` em cada um dos dois lançamentos; com `None` (o caminho de [`attention`]) não lê relógio. Ordem, argumentos, stream e erros iguais.
///
/// # Safety
/// As mesmas de [`attention`].
#[allow(clippy::too_many_arguments)]
pub unsafe fn attention_b(
    s: &Arc<Stream>,
    out: &mut Tensor<f32>,
    part: &mut Tensor<f32>,
    q: &Tensor<f32>,
    kc: &Tensor<f32>,
    vc: &Tensor<f32>,
    lens: &Tensor<i32>,
    scale: f32,
    (heads, kv_heads, d, ns): (usize, usize, usize, usize),
    mut acc: Option<&mut crate::perfil_bloco::AccB>,
) -> Result<(), Error> {
    use crate::perfil_bloco as pb;
    let t_prep = pb::b_ini(&acc);
    // Os kernels não conferem limites (`unchecked_accesses`) e a grade sai da forma de `part`: uma forma errada leria ou
    // escreveria fora dos tensores em silêncio.
    let dim = |t: &Tensor<f32>| t.shape().iter().map(|&v| v as usize).collect::<Vec<_>>();
    assert!(ns >= 1 && kv_heads >= 1 && heads % kv_heads == 0, "atenção: heads {heads}, kv_heads {kv_heads}, ns {ns}");
    assert_eq!(dim(part), [heads * ns * 2, d], "atenção: forma de part");
    assert_eq!(dim(q), [heads, d], "atenção: forma de q");
    assert_eq!(dim(out), [heads, d], "atenção: forma de out");
    let kv = dim(kc);
    assert!(kv.len() == 2 && kv[1] == kv_heads * d && kv[0] % ATTN_BN == 0, "atenção: forma do cache de K {kv:?}");
    assert_eq!(dim(vc), kv, "atenção: forma do cache de V");
    let gen_split = vec![ATTN_BN.to_string(), d.to_string(), (heads / kv_heads).to_string()];
    pb::b_fim(&mut acc, pb::B_ATTN_PREP, t_prep);
    let t0 = pb::b_ini(&acc);
    let r1 = kernels::attn_split(part.partition([2, d]), q, kc, vc, lens, scale, ns as i32).generics(gen_split).async_on(s).map_err(sync_on_err(s));
    pb::b_fim(&mut acc, pb::B_LANCA, t0);
    r1?;
    let t0 = pb::b_ini(&acc);
    let r2 = kernels::attn_combine(out.partition([1, d]), &*part, lens, ns as i32)
        .generics(vec![ATTN_BN.to_string(), d.to_string()])
        .async_on(s)
        .map_err(sync_on_err(s));
    pb::b_fim(&mut acc, pb::B_LANCA, t0);
    r2?;
    Ok(())
}

/// Variações do passo para medir de onde vem o custo fixo (`m2b-custo-fixo`). Com `skip_attn` ou `skip_moe` os logits não valem;
/// com `skip_attn`, o MoE lê um `xn` que não vem do passo (o que sobrou do passo anterior), o que só serve para medir tempo
/// enquanto nenhum kernel depender dos dados.
#[derive(Default, Clone, Copy)]
pub struct Probe<'a> {
    /// Roteamento de cada camada dado de fora (de uma passada normal no mesmo token): pula o GEMV do roteador e a leitura dos
    /// logits por camada, então o passo só sincroniza na leitura dos logits finais. Só com experts residentes.
    pub routes: Option<&'a [Routing]>,
    /// Pula RMSNorm, Q/K/V, RoPE, cache KV, atenção e projeção de saída.
    pub skip_attn: bool,
    /// Pula o bloco MoE inteiro (roteador e experts).
    pub skip_moe: bool,
}

/// Tempos de um passo medido com [`GpuModel::step_probe`].
#[derive(Default, Clone, Copy, Debug)]
pub struct StepTimes {
    /// Parede no host até o último lançamento voltar, antes da leitura dos logits.
    pub enqueue_ms: f64,
    /// Na GPU, entre um evento gravado no começo do passo e outro antes da leitura dos logits (inclui a GPU parada à espera do
    /// host).
    pub gpu_ms: f64,
    /// Parede do passo inteiro.
    pub total_ms: f64,
    /// Operações enfileiradas e sincronizações do passo; `None` com o cache de experts, cujo bloco MoE não conta as suas.
    pub count: Option<Count>,
}

struct Decoder {
    ln_attn: Tensor<f32>,
    ln_moe: Tensor<f32>,
    q: Tensor<bf16>,
    k: Tensor<bf16>,
    v: Tensor<bf16>,
    o: Tensor<bf16>,
    k_cache: Tensor<f32>,
    v_cache: Tensor<f32>,
}

/// Buffers de trabalho de um token, reaproveitados.
struct Bufs {
    xn: Tensor<f32>,
    q: Tensor<f32>,
    k: Tensor<f32>,
    v: Tensor<f32>,
    qr: Tensor<f32>,
    kr: Tensor<f32>,
    att: Tensor<f32>,
    /// Saída da primeira passada da atenção, `[heads·ns·2, d]`.
    part: Tensor<f32>,
    o: Tensor<f32>,
    logits: Tensor<f32>,
}

/// Onde ficam os experts.
pub enum Experts {
    /// Todos na VRAM, um `GpuLayer` por camada (M1b).
    Resident(Vec<GpuLayer>),
    /// Orçamento de slots com cache (M2a).
    Cached(ExpertCache),
}

pub struct GpuModel {
    pub cfg: Config,
    pub max_len: usize,
    pub pos: usize,
    embed: Vec<bf16>,
    /// Embedding com uma linha de zeros a mais quando V é ímpar (o GEMV trabalha em pares de linhas).
    lm_head: Tensor<bf16>,
    norm: Tensor<f32>,
    layers: Vec<Decoder>,
    pub experts: Experts,
    /// Eventos para cronometrar a GPU em `step_probe` (criados na primeira medição).
    events: Option<(cuda_core::Event, cuda_core::Event)>,
    stream: Arc<Stream>,
    bufs: Option<Bufs>,
}

fn f32_dev(v: &[f32], s: &Arc<Stream>) -> Result<Tensor<f32>, Error> {
    Ok(api::copy_host_vec_to_device(&Arc::new(v.to_vec())).sync_on(s)?)
}

fn bf16_dev(v: &[bf16], rows: usize, cols: usize, s: &Arc<Stream>) -> Result<Tensor<bf16>, Error> {
    Ok(api::copy_host_vec_to_device(&Arc::new(v.to_vec())).sync_on(s)?.reshape(&[rows, cols])?)
}

impl GpuModel {
    /// Bytes de VRAM dos pesos densos (atenção, normas, cabeça de saída), sem experts, cache KV nem buffers.
    pub fn dense_bytes(m: &HostModel) -> usize {
        let per_layer: usize = m.layers.iter().map(|l| (l.q.len() + l.k.len() + l.v.len() + l.o.len()) * 2).sum();
        per_layer + m.embed.len() * 2
    }

    /// Bytes de VRAM dos pesos com todos os experts residentes.
    pub fn weight_bytes(m: &HostModel) -> usize {
        Self::dense_bytes(m) + m.layers.iter().map(|l| crate::gpu::layer_weight_bytes(&l.moe)).sum::<usize>()
    }

    /// Todos os experts na VRAM.
    pub fn upload(m: &HostModel, stream: Arc<Stream>, max_len: usize) -> Result<Self, Error> {
        let experts = m.layers.iter().map(|l| GpuLayer::upload(&l.moe, stream.clone())).collect::<Result<_, _>>()?;
        Self::upload_with(m, stream, max_len, Experts::Resident(experts))
    }

    /// Experts servidos por um cache com orçamento de slots.
    pub fn upload_cached(m: &HostModel, stream: Arc<Stream>, max_len: usize, cfg: CacheConfig) -> Result<Self, Error> {
        let cache = ExpertCache::new(m, stream.clone(), cfg)?;
        Self::upload_with(m, stream, max_len, Experts::Cached(cache))
    }

    fn upload_with(m: &HostModel, stream: Arc<Stream>, max_len: usize, experts: Experts) -> Result<Self, Error> {
        let c = m.cfg.clone();
        let (h, d) = (c.hidden, c.head_dim);
        let (qd, kvd) = (c.heads * d, c.kv_heads * d);
        let ok = h % AXPBY_B == 0 && qd % GEMV_BM == 0 && kvd % GEMV_BM == 0 && d % 2 == 0 && c.heads % c.kv_heads == 0 && max_len % ATTN_BN == 0;
        assert!(ok, "formas fora dos tamanhos de bloco: {c:?}, max_len {max_len}");
        crate::gpu::assert_tile_dims(&[("head_dim", d), ("head_dim/2", d / 2)]);
        let s = &stream;
        let mut layers = Vec::with_capacity(c.layers);
        for l in &m.layers {
            layers.push(Decoder {
                ln_attn: f32_dev(&l.ln_attn, s)?,
                ln_moe: f32_dev(&l.ln_moe, s)?,
                q: bf16_dev(&l.q, qd, h, s)?,
                k: bf16_dev(&l.k, kvd, h, s)?,
                v: bf16_dev(&l.v, kvd, h, s)?,
                o: bf16_dev(&l.o, h, qd, s)?,
                k_cache: api::zeros::<f32>(&[max_len, kvd]).sync_on(s)?,
                v_cache: api::zeros::<f32>(&[max_len, kvd]).sync_on(s)?,
            });
        }
        let vocab_pad = c.vocab.next_multiple_of(GEMV_BM);
        let mut head = m.embed.clone();
        head.resize(vocab_pad * h, bf16::ZERO);
        let z = |n: usize| api::zeros::<f32>(&[n]).sync_on(s);
        let part = z(c.heads * ATTN_NS * 2 * d)?.reshape(&[c.heads * ATTN_NS * 2, d])?;
        let bufs = Bufs { xn: z(h)?, q: z(qd)?, k: z(kvd)?, v: z(kvd)?, qr: z(qd)?, kr: z(kvd)?, att: z(qd)?, part, o: z(h)?, logits: z(vocab_pad)? };
        Ok(Self {
            lm_head: bf16_dev(&head, vocab_pad, h, s)?,
            norm: f32_dev(&m.norm, s)?,
            embed: m.embed.clone(),
            layers,
            experts,
            max_len,
            pos: 0,
            stream,
            bufs: Some(bufs),
            cfg: c,
            events: None,
        })
    }

    /// Volta ao início da sequência (o cache antigo fica, mas a máscara da atenção não o lê).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Um token: logits da próxima posição e a decisão do roteador em cada camada. Depois de um erro a stream é sincronizada
    /// e o modelo fica inutilizável (as chamadas seguintes devolvem erro).
    pub fn step(&mut self, token: usize) -> Result<(Vec<f32>, Vec<Routing>), Error> {
        if self.pos >= self.max_len {
            return Err(tensor_error(&format!("cache KV cheio ({} posições)", self.max_len)));
        }
        if token >= self.cfg.vocab {
            return Err(tensor_error(&format!("token {token} fora do vocabulário ({})", self.cfg.vocab)));
        }
        let Some(b) = self.bufs.take() else {
            return Err(tensor_error("modelo inutilizável depois de um erro anterior"));
        };
        let out = self.step_with(b, token, Probe::default(), false).map(|(l, r, _)| (l, r));
        if out.is_err() {
            // SAFETY: só espera a stream; tensores soltos no erro podem ter kernels pendentes.
            let _ = unsafe { self.stream.synchronize() };
        }
        out
    }

    /// Se os eventos de medição já foram criados (só [`step_probe`](Self::step_probe) os cria; o passo normal não paga por eles).
    #[doc(hidden)]
    pub fn has_timing_events(&self) -> bool {
        self.events.is_some()
    }

    /// Um passo com as variações de [`Probe`] e os tempos medidos. Mesmo contrato de erro de [`step`](Self::step).
    pub fn step_probe(&mut self, token: usize, probe: Probe) -> Result<(Vec<f32>, Vec<Routing>, StepTimes), Error> {
        if let Some(r) = probe.routes {
            if !matches!(self.experts, Experts::Resident(_)) {
                return Err(tensor_error("Probe::routes só vale com experts residentes"));
            }
            if r.len() != self.cfg.layers {
                return Err(tensor_error(&format!("Probe::routes com {} camadas, o modelo tem {}", r.len(), self.cfg.layers)));
            }
        }
        if self.pos >= self.max_len {
            return Err(tensor_error(&format!("cache KV cheio ({} posições)", self.max_len)));
        }
        if token >= self.cfg.vocab {
            return Err(tensor_error(&format!("token {token} fora do vocabulário ({})", self.cfg.vocab)));
        }
        let Some(b) = self.bufs.take() else {
            return Err(tensor_error("modelo inutilizável depois de um erro anterior"));
        };
        let out = self.step_with(b, token, probe, true);
        if out.is_err() {
            // SAFETY: só espera a stream.
            let _ = unsafe { self.stream.synchronize() };
        }
        out
    }

    /// `timing`: grava eventos na stream para medir a GPU (só em `step_probe`; o passo normal não paga por isso).
    fn step_with(&mut self, mut b: Bufs, token: usize, probe: Probe, timing: bool) -> Result<(Vec<f32>, Vec<Routing>, StepTimes), Error> {
        let t0 = std::time::Instant::now();
        if timing && self.events.is_none() {
            let dev = self.stream.device();
            self.events = Some((dev.new_event().map_err(Error::from)?, dev.new_event().map_err(Error::from)?));
        }
        let events = if timing { self.events.as_ref() } else { None };
        if let Some((e0, _)) = events {
            e0.record(&self.stream).map_err(Error::from)?;
        }
        let c = self.cfg.clone();
        let (h, d) = (c.hidden, c.head_dim);
        let (qd, kvd, half) = (c.heads * d, c.kv_heads * d, d / 2);
        let s = self.stream.clone();

        let x: Vec<f32> = self.embed[token * h..(token + 1) * h].iter().map(|v| v.to_f32() * c.embedding_multiplier).collect();
        let mut n = Count::default();
        let mut counted = true;
        let mut hs: Tensor<f32> = f32_dev(&x, &s).map_err(sync_on_err(&s))?;
        let cs: Tensor<f32> = f32_dev(&c.rope_cos_sin(self.pos), &s).map_err(sync_on_err(&s))?.reshape(&[2, half]).map_err(sync_on_err(&s))?;
        n.ops += 2;
        n.syncs += 2;
        let len = self.pos + 1;
        let lens: Tensor<i32> = api::copy_host_vec_to_device(&Arc::new(vec![len as i32])).sync_on(&s).map_err(sync_on_err(&s))?;
        n.ops += 1;
        n.syncs += 1;

        let norm_c = crate::gpu::tile_k(h);
        let gen_norm = vec![h.to_string(), norm_c.to_string()];
        let gen_hh = crate::gpu::gemv_generics(h);
        let gen_qd = crate::gpu::gemv_generics(qd);
        let gen_rope = vec![half.to_string()];
        let gen_ax = vec![AXPBY_B.to_string()];
        let mut routes = Vec::with_capacity(c.layers);

        for (l, layer) in self.layers.iter_mut().enumerate() {
            if !probe.skip_attn {
                // SAFETY (todos os `async_on` do laço): `unchecked_accesses`: as formas são múltiplas dos tamanhos de bloco
                // (conferido em `upload`). Os tensores enfileirados são campos de `self`, `b`, `hs` ou `cs`, que vivem até a
                // leitura síncrona dos logits no fim do passo; só kernels e cópias desta stream os tocam, em ordem.
                unsafe {
                    let (xn, _, _, _) = kernels::rmsnorm(b.xn.partition([norm_c]), &hs, &layer.ln_attn, c.eps).generics(gen_norm.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    b.xn = xn.unpartition();
                    let (q, _, _) = kernels::gemv(b.q.partition([GEMV_BM]), &layer.q, &b.xn).generics(gen_hh.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    let (k, _, _) = kernels::gemv(b.k.partition([GEMV_BM]), &layer.k, &b.xn).generics(gen_hh.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    let (v, _, _) = kernels::gemv(b.v.partition([GEMV_BM]), &layer.v, &b.xn).generics(gen_hh.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    b.v = v.unpartition();

                    let q2 = q.unpartition().reshape(&[c.heads * 2, half]).map_err(sync_on_err(&s))?;
                    let qr = b.qr.reshape(&[c.heads * 2, half]).map_err(sync_on_err(&s))?.partition([2, half]);
                    let (qr, _, _) = kernels::rope(qr, &q2, &cs).generics(gen_rope.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    b.q = q2.reshape(&[qd]).map_err(sync_on_err(&s))?;
                    let k2 = k.unpartition().reshape(&[c.kv_heads * 2, half]).map_err(sync_on_err(&s))?;
                    let kr = b.kr.reshape(&[c.kv_heads * 2, half]).map_err(sync_on_err(&s))?.partition([2, half]);
                    let (kr, _, _) = kernels::rope(kr, &k2, &cs).generics(gen_rope.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    b.k = k2.reshape(&[kvd]).map_err(sync_on_err(&s))?;
                    b.kr = kr.unpartition().reshape(&[kvd]).map_err(sync_on_err(&s))?;

                    let off = (self.pos * kvd * 4) as u64;
                    cuda_core::memcpy_dtod_async::<f32>(layer.k_cache.device_pointer().cu_deviceptr() + off, b.kr.device_pointer().cu_deviceptr(), kvd, &s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    cuda_core::memcpy_dtod_async::<f32>(layer.v_cache.device_pointer().cu_deviceptr() + off, b.v.device_pointer().cu_deviceptr(), kvd, &s).map_err(sync_on_err(&s))?;
                    n.ops += 1;

                    let q_heads = qr.unpartition().reshape(&[c.heads, d]).map_err(sync_on_err(&s))?;
                    let mut att = b.att.reshape(&[c.heads, d]).map_err(sync_on_err(&s))?;
                    attention(&s, &mut att, &mut b.part, &q_heads, &layer.k_cache, &layer.v_cache, &lens, c.attention_multiplier, (c.heads, c.kv_heads, d, ATTN_NS))
                        .map_err(sync_on_err(&s))?;
                    n.ops += 2;
                    b.qr = q_heads.reshape(&[qd]).map_err(sync_on_err(&s))?;
                    b.att = att.reshape(&[qd]).map_err(sync_on_err(&s))?;

                    let (o, _, _) = kernels::gemv(b.o.partition([GEMV_BM]), &layer.o, &b.att).generics(gen_qd.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    b.o = o.unpartition();
                    let (h2, _, _, _) = kernels::axpby(hs.partition([AXPBY_B]), &b.o, c.residual_multiplier, 1.0f32).generics(gen_ax.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    hs = h2.unpartition();

                    let (xn, _, _, _) = kernels::rmsnorm(b.xn.partition([norm_c]), &hs, &layer.ln_moe, c.eps).generics(gen_norm.clone()).async_on(&s).map_err(sync_on_err(&s))?;
                    n.ops += 1;
                    b.xn = xn.unpartition();
                }
            }
            if probe.skip_moe {
                continue;
            }
            let hp = hs.partition([AXPBY_B]);
            let given = probe.routes.map(|r| &r[l]);
            let (h2, r) = match &mut self.experts {
                Experts::Resident(v) => {
                    let out = v[l].add_into_routed(&b.xn, hp, c.residual_multiplier, given).map_err(sync_on_err(&s))?;
                    n.ops += v[l].count.ops;
                    n.syncs += v[l].count.syncs;
                    out
                }
                Experts::Cached(cache) => {
                    counted = false;
                    cache.add_into(l, &b.xn, hp, c.residual_multiplier).map_err(sync_on_err(&s))?
                }
            };
            hs = h2.unpartition();
            routes.push(r);
        }

        // SAFETY: como no laço; a leitura síncrona logo abaixo encerra o passo.
        let logits = unsafe {
            let (xn, _, _, _) = kernels::rmsnorm(b.xn.partition([norm_c]), &hs, &self.norm, c.eps).generics(gen_norm.clone()).async_on(&s).map_err(sync_on_err(&s))?;
            n.ops += 1;
            b.xn = xn.unpartition();
            let (lg, _, _) = kernels::gemv(b.logits.partition([GEMV_BM]), &self.lm_head, &b.xn).generics(gen_hh.clone()).async_on(&s).map_err(sync_on_err(&s))?;
            n.ops += 1;
            Arc::new(lg.unpartition())
        };
        let enqueue_ms = t0.elapsed().as_secs_f64() * 1e3;
        let events = if timing { self.events.as_ref() } else { None };
        if let Some((_, e1)) = events {
            e1.record(&s).map_err(sync_on_err(&s))?;
        }
        let mut lv = api::copy_device_to_host_vec(&logits).sync_on(&s).map_err(sync_on_err(&s))?;
        n.ops += 1;
        n.syncs += 1;
        let gpu_ms = match events {
            Some((e0, e1)) => e0.elapsed_time(e1).map_err(Error::from)? as f64,
            None => 0.0,
        };
        b.logits = Arc::try_unwrap(logits).map_err(|_| "logits ainda referenciados").unwrap();
        lv.truncate(c.vocab);
        for v in lv.iter_mut() {
            *v /= c.logits_scaling;
        }
        drop((hs, cs, lens));
        self.bufs = Some(b);
        self.pos += 1;
        let count = counted.then_some(n);
        Ok((lv, routes, StepTimes { enqueue_ms, gpu_ms, total_ms: t0.elapsed().as_secs_f64() * 1e3, count }))
    }
}

impl Drop for GpuModel {
    fn drop(&mut self) {
        // Antes de os campos caírem: kernels e cópias do cache KV enfileirados ainda podem usar os pesos e os buffers, e o cutile
        // devolve a memória numa stream de desalocação que não espera esta.
        // SAFETY: só espera a stream.
        let _ = unsafe { self.stream.synchronize() };
    }
}
