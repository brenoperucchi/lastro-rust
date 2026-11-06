//! A1.3c: a parte densa do modelo (atenção, normas e cabeça de saída) no backend de CUDA, para o motor D1 rodar o Granite MoE
//! inteiro. Por camada o motor chama `pre_layer` (atenção e a norma de entrada do bloco MoE, que deixa em `xn` o que o roteador
//! e os experts leem) antes do `route`; depois da última, `post_layers` (norma final e a cabeça de saída, que deixa os logits).
//!
//! Os tensores de trabalho ficam sempre donos do estado e os lançadores recebem partições emprestadas (`&mut`), pela razão da
//! rodada 10 do A1.2: um lançamento que falha só derruba o empréstimo, e o tensor segue vivo até a espera da stream. Onde um
//! kernel pede outra forma do mesmo buffer (por exemplo `[cabeças·2, metade]` no RoPE), o dado passa por uma cópia
//! device-para-device (`memcpy_dtod_async`) entre dois buffers já formados, e não por `reshape`, que consome o tensor.
#![allow(unsafe_code)]
use crate::cuda_backend::{fail_k, kernel_err, CudaError};
use crate::gpu::{gemv_generics, kernels, tile_k, AXPBY_B, GEMV_BM};
use crate::gpu_model::{attention, ATTN_BN, ATTN_NS};
use crate::model::{Config, HostModel};
use cuda_core::{sys, Stream};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::PartitionMut;
use std::sync::Arc;

struct DenseLayer {
    ln_attn: Tensor<f32>,
    ln_moe: Tensor<f32>,
    q: Tensor<bf16>,
    k: Tensor<bf16>,
    v: Tensor<bf16>,
    o: Tensor<bf16>,
    k_cache: Tensor<f32>,
    v_cache: Tensor<f32>,
}

pub struct Dense {
    cfg: Config,
    max_len: usize,
    /// Posição do próximo token.
    pos: usize,
    /// Posição do token em curso (a do `begin_token` mais recente).
    cur: usize,
    embed: Vec<bf16>,
    lm_head: Tensor<bf16>,
    norm: Tensor<f32>,
    layers: Vec<DenseLayer>,
    // buffers de trabalho, cada um na forma de quem o lê
    q: Tensor<f32>,
    k: Tensor<f32>,
    v: Tensor<f32>,
    q2: Tensor<f32>,
    qr: Tensor<f32>,
    qh: Tensor<f32>,
    k2: Tensor<f32>,
    kr: Tensor<f32>,
    att2: Tensor<f32>,
    att: Tensor<f32>,
    part: Tensor<f32>,
    o: Tensor<f32>,
    logits: Tensor<f32>,
    cs: Tensor<f32>,
    lens: Tensor<i32>,
    /// E4b: o `cos/sin` e o comprimento do cache de cada posição do passo em curso, e a posição da primeira.
    step_cs: Vec<Tensor<f32>>,
    step_lens: Vec<Tensor<i32>>,
    step_base: usize,
}

fn f32_dev(v: &[f32], s: &Arc<Stream>) -> Result<Tensor<f32>, CudaError> {
    api::copy_host_vec_to_device(&Arc::new(v.to_vec())).sync_on(s).map_err(kernel_err)
}

fn bf16_dev(v: &[bf16], rows: usize, cols: usize, s: &Arc<Stream>) -> Result<Tensor<bf16>, CudaError> {
    use cutile::tensor::Reshape;
    api::copy_host_vec_to_device(&Arc::new(v.to_vec())).sync_on(s).map_err(kernel_err)?.reshape(&[rows, cols]).map_err(kernel_err)
}

fn zeros(shape: &[usize], s: &Arc<Stream>) -> Result<Tensor<f32>, CudaError> {
    api::zeros::<f32>(shape).sync_on(s).map_err(kernel_err)
}

/// Copia `n` floats de `src` (desde o começo) para `dst` a partir do float `off`, na stream.
fn dtod(dst: &Tensor<f32>, off: usize, src: &Tensor<f32>, n: usize, s: &Arc<Stream>, raw: sys::CUstream) -> Result<(), CudaError> {
    // SAFETY: os dois tensores têm pelo menos `off + n` e `n` floats (conferido por quem chama); só enfileira uma cópia.
    unsafe { cuda_core::memcpy_dtod_async::<f32>(dst.device_pointer().cu_deviceptr() + (off * 4) as u64, src.device_pointer().cu_deviceptr(), n, s) }
        .map_err(fail_k(raw))
}

/// Só para o teste de regressão: faz `read_logits` falhar depois de trocar o tensor de logits.
pub(crate) static FALHA_NA_COPIA_DOS_LOGITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Só para o teste de regressão do caminho de erro de `read_kv`.
pub static FALHA_NA_LEITURA_DO_KV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Só para o mutante do oráculo do E4b: `attn_block_at` usa o comprimento da última posição do passo em todas (atenção não causal).
pub static MUTANTE_LENS_DA_ULTIMA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Dense {
    pub fn new(m: &HostModel, s: &Arc<Stream>, max_len: usize) -> Result<Self, CudaError> {
        let c = m.cfg.clone();
        let (h, d) = (c.hidden, c.head_dim);
        let (qd, kvd, half) = (c.heads * d, c.kv_heads * d, d / 2);
        let ok = h % AXPBY_B == 0 && qd % GEMV_BM == 0 && kvd % GEMV_BM == 0 && d % 2 == 0 && c.heads % c.kv_heads == 0 && max_len % ATTN_BN == 0 && max_len > 0;
        if !ok {
            return Err(CudaError::Config("modelo fora dos tamanhos de bloco"));
        }
        crate::gpu::assert_tile_dims(&[("head_dim", d), ("head_dim/2", d / 2)]);
        let mut layers = Vec::with_capacity(c.layers);
        for l in &m.layers {
            layers.push(DenseLayer {
                ln_attn: f32_dev(&l.ln_attn, s)?,
                ln_moe: f32_dev(&l.ln_moe, s)?,
                q: bf16_dev(&l.q, qd, h, s)?,
                k: bf16_dev(&l.k, kvd, h, s)?,
                v: bf16_dev(&l.v, kvd, h, s)?,
                o: bf16_dev(&l.o, h, qd, s)?,
                k_cache: zeros(&[max_len, kvd], s)?,
                v_cache: zeros(&[max_len, kvd], s)?,
            });
        }
        let vocab_pad = c.vocab.next_multiple_of(GEMV_BM);
        let mut head = m.embed.clone();
        head.resize(vocab_pad * h, bf16::ZERO);
        Ok(Self {
            lm_head: bf16_dev(&head, vocab_pad, h, s)?,
            norm: f32_dev(&m.norm, s)?,
            embed: m.embed.clone(),
            layers,
            max_len,
            pos: 0,
            cur: 0,
            q: zeros(&[qd], s)?,
            k: zeros(&[kvd], s)?,
            v: zeros(&[kvd], s)?,
            q2: zeros(&[c.heads * 2, half], s)?,
            qr: zeros(&[c.heads * 2, half], s)?,
            qh: zeros(&[c.heads, d], s)?,
            k2: zeros(&[c.kv_heads * 2, half], s)?,
            kr: zeros(&[c.kv_heads * 2, half], s)?,
            att2: zeros(&[c.heads, d], s)?,
            att: zeros(&[qd], s)?,
            part: zeros(&[c.heads * ATTN_NS * 2, d], s)?,
            o: zeros(&[h], s)?,
            logits: zeros(&[vocab_pad], s)?,
            cs: zeros(&[2, half], s)?,
            lens: api::copy_host_vec_to_device(&Arc::new(vec![1i32])).sync_on(s).map_err(kernel_err)?,
            step_cs: Vec::new(),
            step_lens: Vec::new(),
            step_base: 0,
            cfg: c,
        })
    }

    pub fn cfg(&self) -> &Config {
        &self.cfg
    }

    /// Começa um token: devolve o embedding vezes o multiplicador, e carrega o `cos/sin` da posição e o comprimento do cache.
    /// Chamar com a stream ociosa (depois de `finish_token`).
    pub fn begin_token(&mut self, token: usize, s: &Arc<Stream>) -> Result<Vec<f32>, CudaError> {
        use cutile::tensor::Reshape;
        if self.pos >= self.max_len {
            return Err(CudaError::Config("cache KV cheio"));
        }
        if token >= self.cfg.vocab {
            return Err(CudaError::Config("token fora do vocabulário"));
        }
        let (h, half) = (self.cfg.hidden, self.cfg.head_dim / 2);
        let x: Vec<f32> = self.embed[token * h..(token + 1) * h].iter().map(|v| v.to_f32() * self.cfg.embedding_multiplier).collect();
        self.cs = f32_dev(&self.cfg.rope_cos_sin(self.pos), s)?.reshape(&[2, half]).map_err(kernel_err)?;
        self.lens = api::copy_host_vec_to_device(&Arc::new(vec![(self.pos + 1) as i32])).sync_on(s).map_err(kernel_err)?;
        self.cur = self.pos;
        self.pos += 1;
        self.step_cs.clear();
        self.step_lens.clear();
        Ok(x)
    }

    /// Começa um passo de `tokens.len()` posições consecutivas (E4b): devolve o embedding de cada uma e guarda o `cos/sin` e o
    /// comprimento do cache de cada posição (`pos + p + 1`). A atenção da posição `p` na camada `l` lê o cache até ela (as posições
    /// anteriores do bloco já gravaram a camada `l` antes, na ordem da stream), então o bloco é causal. Chamar com a stream ociosa.
    pub fn begin_step(&mut self, tokens: &[usize], s: &Arc<Stream>) -> Result<Vec<Vec<f32>>, CudaError> {
        use cutile::tensor::Reshape;
        if tokens.is_empty() || self.pos + tokens.len() > self.max_len {
            return Err(CudaError::Config("cache KV cheio"));
        }
        if tokens.iter().any(|&t| t >= self.cfg.vocab) {
            return Err(CudaError::Config("token fora do vocabulário"));
        }
        let (h, half) = (self.cfg.hidden, self.cfg.head_dim / 2);
        let mut xs = Vec::with_capacity(tokens.len());
        let (mut cs, mut lens) = (Vec::new(), Vec::new());
        for (p, &token) in tokens.iter().enumerate() {
            xs.push(self.embed[token * h..(token + 1) * h].iter().map(|v| v.to_f32() * self.cfg.embedding_multiplier).collect());
            cs.push(f32_dev(&self.cfg.rope_cos_sin(self.pos + p), s)?.reshape(&[2, half]).map_err(kernel_err)?);
            lens.push(api::copy_host_vec_to_device(&Arc::new(vec![(self.pos + p + 1) as i32])).sync_on(s).map_err(kernel_err)?);
        }
        self.step_cs = cs;
        self.step_lens = lens;
        self.step_base = self.pos;
        self.pos += tokens.len();
        Ok(xs)
    }

    /// A atenção da posição `p` do passo (E4b): põe o `cos/sin`, o comprimento e a posição de escrita dela no lugar dos de `begin_token`
    /// e roda [`attn_block`](Self::attn_block). Com um passo de uma posição vinda de `begin_token` (sem `begin_step`), `p = 0` é o
    /// caminho de sempre.
    pub fn attn_block_at(&mut self, s: &Arc<Stream>, l: usize, p: usize, h: &mut Tensor<f32>, xn: &mut Tensor<f32>) -> Result<(), CudaError> {
        if self.step_cs.is_empty() {
            return if p == 0 { self.attn_block(s, l, h, xn) } else { Err(CudaError::Config("posição do passo sem begin_step")) };
        }
        if p >= self.step_cs.len() {
            return Err(CudaError::Config("posição fora do passo"));
        }
        let pl = if MUTANTE_LENS_DA_ULTIMA.load(std::sync::atomic::Ordering::SeqCst) { self.step_lens.len() - 1 } else { p };
        std::mem::swap(&mut self.cs, &mut self.step_cs[p]);
        std::mem::swap(&mut self.lens, &mut self.step_lens[pl]);
        let old_cur = std::mem::replace(&mut self.cur, self.step_base + p);
        let r = self.attn_block(s, l, h, xn);
        std::mem::swap(&mut self.cs, &mut self.step_cs[p]);
        std::mem::swap(&mut self.lens, &mut self.step_lens[pl]);
        self.cur = old_cur;
        r
    }

    /// A posição do próximo token na sequência.
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Volta ao início da sequência (o cache antigo fica, mas a máscara da atenção não o lê).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// A atenção da camada `l` sobre `h` (residual) e a norma de entrada do bloco MoE em `xn`.
    pub fn attn_block(&mut self, s: &Arc<Stream>, l: usize, h: &mut Tensor<f32>, xn: &mut Tensor<f32>) -> Result<(), CudaError> {
        let raw = s.cu_stream();
        let c = self.cfg.clone();
        let (hd, d) = (c.hidden, c.head_dim);
        let (qd, kvd, half) = (c.heads * d, c.kv_heads * d, d / 2);
        let norm_c = tile_k(hd);
        let gen_norm = vec![hd.to_string(), norm_c.to_string()];
        let (gen_hh, gen_qd) = (gemv_generics(hd), gemv_generics(qd));
        let gen_rope = vec![half.to_string()];
        let lay = &self.layers[l];
        // SAFETY (todos os `async_on` daqui): `unchecked_accesses`; as formas são as de `new`, múltiplas dos blocos; os tensores
        // são do estado e vivem além da sincronização da stream; só esta stream os toca, em ordem.
        unsafe {
            kernels::rmsnorm((&mut *xn).partition([norm_c]), &*h, &lay.ln_attn, c.eps).generics(gen_norm.clone()).async_on(s).map_err(fail_k(raw))?;
            kernels::gemv((&mut self.q).partition([GEMV_BM]), &lay.q, &*xn).generics(gen_hh.clone()).async_on(s).map_err(fail_k(raw))?;
            kernels::gemv((&mut self.k).partition([GEMV_BM]), &lay.k, &*xn).generics(gen_hh.clone()).async_on(s).map_err(fail_k(raw))?;
            kernels::gemv((&mut self.v).partition([GEMV_BM]), &lay.v, &*xn).generics(gen_hh.clone()).async_on(s).map_err(fail_k(raw))?;
        }
        dtod(&self.q2, 0, &self.q, qd, s, raw)?;
        dtod(&self.k2, 0, &self.k, kvd, s, raw)?;
        // SAFETY: como acima.
        unsafe {
            kernels::rope((&mut self.qr).partition([2, half]), &self.q2, &self.cs).generics(gen_rope.clone()).async_on(s).map_err(fail_k(raw))?;
            kernels::rope((&mut self.kr).partition([2, half]), &self.k2, &self.cs).generics(gen_rope.clone()).async_on(s).map_err(fail_k(raw))?;
        }
        dtod(&lay.k_cache, self.cur * kvd, &self.kr, kvd, s, raw)?;
        dtod(&lay.v_cache, self.cur * kvd, &self.v, kvd, s, raw)?;
        dtod(&self.qh, 0, &self.qr, qd, s, raw)?;
        // SAFETY: `lens` está entre 1 e `max_len` (`begin_token`); as formas são as de `new`.
        unsafe { attention(s, &mut self.att2, &mut self.part, &self.qh, &lay.k_cache, &lay.v_cache, &self.lens, c.attention_multiplier, (c.heads, c.kv_heads, d, ATTN_NS)) }
            .map_err(fail_k(raw))?;
        dtod(&self.att, 0, &self.att2, qd, s, raw)?;
        // SAFETY: como acima.
        unsafe {
            kernels::gemv((&mut self.o).partition([GEMV_BM]), &lay.o, &self.att).generics(gen_qd).async_on(s).map_err(fail_k(raw))?;
            kernels::axpby((&mut *h).partition([AXPBY_B]), &self.o, c.residual_multiplier, 1.0f32).generics(vec![AXPBY_B.to_string()]).async_on(s).map_err(fail_k(raw))?;
            kernels::rmsnorm((&mut *xn).partition([norm_c]), &*h, &lay.ln_moe, c.eps).generics(gen_norm).async_on(s).map_err(fail_k(raw))?;
        }
        Ok(())
    }

    /// A norma final e a cabeça de saída: deixa os logits (sem a divisão por `logits_scaling`) em `self.logits`.
    pub fn head(&mut self, s: &Arc<Stream>, h: &Tensor<f32>, xn: &mut Tensor<f32>) -> Result<(), CudaError> {
        let raw = s.cu_stream();
        let hd = self.cfg.hidden;
        let norm_c = tile_k(hd);
        // SAFETY: como em `attn_block`.
        unsafe {
            kernels::rmsnorm((&mut *xn).partition([norm_c]), h, &self.norm, self.cfg.eps).generics(vec![hd.to_string(), norm_c.to_string()]).async_on(s).map_err(fail_k(raw))?;
            kernels::gemv((&mut self.logits).partition([GEMV_BM]), &self.lm_head, &*xn).generics(gemv_generics(hd)).async_on(s).map_err(fail_k(raw))?;
        }
        Ok(())
    }

    /// O cache K e V da camada `l` até `len` posições, copiado para o host (E4b, oráculo do KV). Com a stream ociosa (quem chama
    /// sincronizou). Devolve `len·kv_dim` valores de cada um.
    pub fn read_kv(&mut self, s: &Arc<Stream>, l: usize, len: usize) -> Result<(Vec<f32>, Vec<f32>), CudaError> {
        let kvd = self.cfg.kv_heads * self.cfg.head_dim;
        if len > self.max_len {
            return Err(CudaError::Config("len maior que o cache KV"));
        }
        let mut lado = |k: bool| -> Result<Vec<f32>, CudaError> {
            let lay = &mut self.layers[l];
            let slot = if k { &mut lay.k_cache } else { &mut lay.v_cache };
            let t = std::mem::replace(slot, zeros(&[self.max_len, kvd], s)?);
            let t = Arc::new(t);
            // O resultado da cópia só é propagado depois de devolver o tensor original: uma leitura que falha não pode deixar o cache
            // da sequência trocado por zeros.
            let copia = if FALHA_NA_LEITURA_DO_KV.load(std::sync::atomic::Ordering::SeqCst) {
                Err(CudaError::Config("falha injetada na leitura do KV"))
            } else {
                api::copy_device_to_host_vec(&t).sync_on(s).map_err(kernel_err)
            };
            *slot = Arc::try_unwrap(t).map_err(|_| CudaError::Config("cache KV ainda referenciado"))?;
            let mut v = copia?;
            v.truncate(len * kvd);
            Ok(v)
        };
        let kk = lado(true)?;
        let vv = lado(false)?;
        Ok((kk, vv))
    }

    /// Os logits do último token, depois de a stream terminar (quem chama sincronizou).
    pub fn read_logits(&mut self, s: &Arc<Stream>) -> Result<Vec<f32>, CudaError> {
        let falha = FALHA_NA_COPIA_DOS_LOGITS.load(std::sync::atomic::Ordering::SeqCst);
        let pad = self.cfg.vocab.next_multiple_of(GEMV_BM);
        let l = std::mem::replace(&mut self.logits, zeros(&[pad], s)?);
        let l = Arc::new(l);
        if falha {
            return Err(CudaError::Config("falha injetada na cópia dos logits"));
        }
        let mut v = api::copy_device_to_host_vec(&l).sync_on(s).map_err(kernel_err)?;
        self.logits = Arc::try_unwrap(l).map_err(|_| CudaError::Config("logits ainda referenciados"))?;
        v.truncate(self.cfg.vocab);
        v.iter_mut().for_each(|x| *x /= self.cfg.logits_scaling);
        Ok(v)
    }
}
