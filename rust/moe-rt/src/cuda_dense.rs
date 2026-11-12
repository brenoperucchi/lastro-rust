//! A1.3c: a parte densa do modelo (atenção, normas e cabeça de saída) no backend de CUDA, para o motor D1 rodar o Granite MoE
//! inteiro. Por camada o motor chama `pre_layer` (atenção e a norma de entrada do bloco MoE, que deixa em `xn` o que o roteador
//! e os experts leem) antes do `route`; depois da última, `post_layers` (norma final e a cabeça de saída, que deixa os logits).
//!
//! Os tensores de trabalho ficam sempre donos do estado e os lançadores recebem partições emprestadas (`&mut`), pela razão da
//! rodada 10 do A1.2: um lançamento que falha só derruba o empréstimo, e o tensor segue vivo até a espera da stream. Onde um
//! kernel pede outra forma do mesmo buffer (por exemplo `[cabeças·2, metade]` no RoPE), a O1 do plano de cópias (11/11, aprovada
//! pelo Breno às 15:23, política de erro A) usa `reshape`, que muda só os metadados: o tensor sai do campo, é reformado,
//! emprestado ao kernel e devolvido na forma original por [`crate::reforma::com_forma`]; em qualquer erro ele volta ao campo e, se a
//! restauração falhar, o dono espera a stream antes de cair. A O3 (12/11) tirou as duas cópias device-para-device do cache K/V: o gemv de V
//! (`gemv_row`) e o RoPE de K (`rope_kv`) gravam direto na linha `cur` do cache da camada, que continua sendo do campo (só um empréstimo
//! `&mut` por lançamento, a mesma política A); a posição é conferida no host (`cur < max_len`) antes do primeiro lançamento.
#![allow(unsafe_code)]
use crate::cuda_backend::{fail_k, kernel_err, sync_before_drop, CudaError};
use crate::reforma::{com_forma, Reformavel};
use crate::cache_direto::inicio_da_linha;
use crate::gpu::{gemv_generics, kernels, tile_k, AXPBY_B, GEMV_BM};
use crate::gpu_model::{attention_b, ATTN_BN, ATTN_NS};
use crate::perfil_bloco::{self as pb, AccB};
use crate::model::{Config, HostModel};
use cuda_core::Stream;
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
    // buffers de trabalho. `q`, `k`, `qr` e `att2` são `Option` porque a O1 os tira do campo para reformar (`com_forma`); fora de uma
    // chamada de `attn_block_b` estão sempre `Some`, na forma de `new` (só `None` se uma restauração falhou depois de a stream ser esperada).
    q: Option<Tensor<f32>>,
    k: Option<Tensor<f32>>,
    qr: Option<Tensor<f32>>,
    att2: Option<Tensor<f32>>,
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

/// Só para o teste de regressão: faz `read_logits` falhar depois de trocar o tensor de logits.
pub(crate) static FALHA_NA_COPIA_DOS_LOGITS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Só para o teste de regressão do caminho de erro de `read_kv`.
pub static FALHA_NA_LEITURA_DO_KV: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Só para o mutante do oráculo do E4b: `attn_block_at` usa o comprimento da última posição do passo em todas (atenção não causal).
pub static MUTANTE_LENS_DA_ULTIMA: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Só para o teste de GPU da O1: 1 a 4 faz falhar o lançamento reformado de mesmo número (RoPE de Q, RoPE de K, atenção, gemv de `o`),
/// depois de a stream esperar como em qualquer falha de lançamento; 0 desliga. Uma leitura relaxada por lançamento, só nesses quatro.
pub static FALHA_NO_LANCAMENTO_REFORMADO: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Só para o teste de GPU da O1: `n > 0` faz falhar a n-ésima chamada seguinte de reforma (em `attn_block_b`: 1 e 2 Q, 3 e 4 K, 5 e 6 `qr`,
/// 7 e 8 `att2`; ímpares são a ida, pares a volta), devolvendo o dono; 0 desliga.
pub static FALHA_NA_REFORMA_N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Só para o teste de GPU da O3: 1 faz falhar o lançamento do gemv de V (antes de ele sair), 2 o do RoPE de K (depois de V ter saído), 3 força a
/// posição `cur` para fora do cache (a conferência do host recusa antes do primeiro lançamento); 0 desliga. Uma leitura relaxada por chamada.
pub static FALHA_NO_LANCAMENTO_O3: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn injeta_o3(n: usize) -> bool {
    FALHA_NO_LANCAMENTO_O3.load(std::sync::atomic::Ordering::Relaxed) == n
}

fn injeta_lancamento(n: usize) -> bool {
    FALHA_NO_LANCAMENTO_REFORMADO.load(std::sync::atomic::Ordering::Relaxed) == n
}

fn injeta_reforma() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    let n = FALHA_NA_REFORMA_N.load(Relaxed);
    if n == 0 {
        return false;
    }
    FALHA_NA_REFORMA_N.store(n - 1, Relaxed);
    n == 1
}

/// Strides de um tensor contíguo de forma `shape` (a mesma regra que o cutile confere em `reshape`).
fn strides_contiguos(shape: &[i32]) -> Vec<i32> {
    let mut st = vec![1i32; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * shape[i + 1];
    }
    st
}

impl Reformavel for Tensor<f32> {
    type Erro = CudaError;
    fn forma(&self) -> Vec<usize> {
        self.shape().iter().map(|&x| x as usize).collect()
    }
    /// Confere antes de consumir o tensor exatamente o que `reshape` confere (contíguo e mesmo número de elementos), para que o erro devolva
    /// o dono; passada a conferência, `reshape` só repete as mesmas verificações e não falha.
    fn reformar(self, forma: &[usize]) -> Result<Self, (Option<Self>, CudaError)> {
        use cutile::tensor::Reshape;
        if injeta_reforma() {
            return Err((Some(self), CudaError::Config("falha injetada na reforma")));
        }
        let n: usize = forma.iter().product();
        if n != self.size() || self.strides() != strides_contiguos(self.shape()).as_slice() {
            return Err((Some(self), CudaError::Config("reforma incompatível com o tensor")));
        }
        self.reshape(forma).map_err(|e| (None, kernel_err(e)))
    }
}

/// O tensor de trabalho de um campo reformável, ou o erro de campo vazio (uma restauração anterior falhou).
fn campo(o: &mut Option<Tensor<f32>>) -> Result<&mut Tensor<f32>, CudaError> {
    o.as_mut().ok_or(CudaError::Config("buffer de trabalho ausente (uma restauração anterior falhou)"))
}

fn ausente() -> CudaError {
    CudaError::Config("buffer de trabalho ausente (uma restauração anterior falhou)")
}

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
            q: Some(zeros(&[qd], s)?),
            k: Some(zeros(&[kvd], s)?),
            qr: Some(zeros(&[c.heads * 2, half], s)?),
            att2: Some(zeros(&[c.heads, d], s)?),
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

    /// O comprimento máximo do cache KV desta construção (só leitura).
    pub fn max_len(&self) -> usize {
        self.max_len
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
        self.attn_block_at_b(s, l, p, h, xn, None)
    }

    /// [`attn_block_at`](Self::attn_block_at) com a F2 nível B: com `acc` presente carimba `B_SETUP` (os swaps antes e a restauração depois de
    /// `attn_block_b`); com `None` não lê relógio. Mesma ordem, mesmos swaps, mesma restauração e mesmos erros.
    pub fn attn_block_at_b(&mut self, s: &Arc<Stream>, l: usize, p: usize, h: &mut Tensor<f32>, xn: &mut Tensor<f32>, mut acc: Option<&mut AccB>) -> Result<(), CudaError> {
        if self.step_cs.is_empty() {
            return if p == 0 { self.attn_block_b(s, l, h, xn, acc) } else { Err(CudaError::Config("posição do passo sem begin_step")) };
        }
        if p >= self.step_cs.len() {
            return Err(CudaError::Config("posição fora do passo"));
        }
        let t_setup = pb::b_ini(&acc);
        let pl = if MUTANTE_LENS_DA_ULTIMA.load(std::sync::atomic::Ordering::SeqCst) { self.step_lens.len() - 1 } else { p };
        std::mem::swap(&mut self.cs, &mut self.step_cs[p]);
        std::mem::swap(&mut self.lens, &mut self.step_lens[pl]);
        let old_cur = std::mem::replace(&mut self.cur, self.step_base + p);
        pb::b_fim(&mut acc, pb::B_SETUP, t_setup);
        let r = self.attn_block_b(s, l, h, xn, acc.as_deref_mut());
        let t_setup = pb::b_ini(&acc);
        std::mem::swap(&mut self.cs, &mut self.step_cs[p]);
        std::mem::swap(&mut self.lens, &mut self.step_lens[pl]);
        self.cur = old_cur;
        pb::b_fim(&mut acc, pb::B_SETUP, t_setup);
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
    #[allow(dead_code)]
    pub fn attn_block(&mut self, s: &Arc<Stream>, l: usize, h: &mut Tensor<f32>, xn: &mut Tensor<f32>) -> Result<(), CudaError> {
        self.attn_block_b(s, l, h, xn, None)
    }

    /// [`attn_block`](Self::attn_block) com a F2 nível B: com `acc` presente carimba `B_GENERICS` (o prelúdio), `B_LANCA` (cada expressão
    /// `async_on`, as nove daqui; as duas de `attention_b` ficam lá) e `B_DTOD` (sem uso desde a O3: não há mais cópias); com `None` não lê relógio. Mesma ordem de
    /// lançamentos, argumentos, stream e erros.
    pub fn attn_block_b(&mut self, s: &Arc<Stream>, l: usize, h: &mut Tensor<f32>, xn: &mut Tensor<f32>, mut acc: Option<&mut AccB>) -> Result<(), CudaError> {
        // Mede a expressão `$e` (um `Result`) e a devolve; sem acumulador não lê relógio.
        macro_rules! med {
            ($op:expr, $e:expr) => {{
                let t0 = pb::b_ini(&acc);
                let r = $e;
                pb::b_fim(&mut acc, $op, t0);
                r
            }};
        }
        let t_gen = pb::b_ini(&acc);
        let raw = s.cu_stream();
        let c = self.cfg.clone();
        let (hd, d) = (c.hidden, c.head_dim);
        let (qd, kvd, half) = (c.heads * d, c.kv_heads * d, d / 2);
        let norm_c = tile_k(hd);
        let gen_norm = vec![hd.to_string(), norm_c.to_string()];
        let (gen_hh, gen_qd) = (gemv_generics(hd), gemv_generics(qd));
        let gen_rope = vec![half.to_string()];
        let max_len = self.max_len;
        // O3: a posição `cur` como início da linha do cache, conferida no host antes do primeiro lançamento.
        let cur_o3 = if injeta_o3(3) { max_len } else { self.cur };
        let linha = inicio_da_linha(cur_o3, max_len).map_err(CudaError::Config)?;
        let (gen_rope_kv, gen_gemv_row) = (vec![half.to_string(), d.to_string(), "1".to_string(), "1".to_string()], {
            let mut g = gemv_generics(hd);
            g.extend(["1".to_string(), "1".to_string()]);
            g
        });
        let lay = &mut self.layers[l];
        pb::b_fim(&mut acc, pb::B_GENERICS, t_gen);
        // SAFETY (todos os `async_on` daqui): `unchecked_accesses`; as formas são as de `new`, múltiplas dos blocos; os tensores
        // são do estado e vivem além da sincronização da stream; só esta stream os toca, em ordem.
        unsafe {
            med!(pb::B_LANCA, kernels::rmsnorm((&mut *xn).partition([norm_c]), &*h, &lay.ln_attn, c.eps).generics(gen_norm.clone()).async_on(s).map_err(fail_k(raw)))?;
            med!(pb::B_LANCA, kernels::gemv(campo(&mut self.q)?.partition([GEMV_BM]), &lay.q, &*xn).generics(gen_hh.clone()).async_on(s).map_err(fail_k(raw)))?;
            med!(pb::B_LANCA, kernels::gemv(campo(&mut self.k)?.partition([GEMV_BM]), &lay.k, &*xn).generics(gen_hh.clone()).async_on(s).map_err(fail_k(raw)))?;
            if injeta_o3(1) {
                sync_before_drop(raw);
                return Err(CudaError::Config("falha injetada no lançamento O3 1"));
            }
            // O3: o gemv de V grava direto na linha `cur` do cache V (sem a cópia device-para-device).
            med!(pb::B_LANCA, kernels::gemv_row((&mut lay.v_cache).partition([1, GEMV_BM]).map([1, 1], (kvd / GEMV_BM) as u32), &lay.v, &*xn, linha).generics(gen_gemv_row).async_on(s).map_err(fail_k(raw)))?;
        }
        // O1: `q` e `k` viram `[cabeças·2, metade]` por reforma (sem cópia) só enquanto o RoPE os lê; as reformas ficam fora de qualquer
        // operação medida (caem no resíduo do nível B). Mesmos kernels, mesmos argumentos, mesma ordem de lançamento.
        let seguro = |e: CudaError| -> CudaError {
            sync_before_drop(raw);
            e
        };
        // SAFETY: como acima.
        com_forma(&mut self.q, &[c.heads * 2, half], ausente, || sync_before_drop(raw), |q2| {
            if injeta_lancamento(1) {
                return Err(seguro(CudaError::Config("falha injetada no lançamento reformado 1")));
            }
            unsafe { med!(pb::B_LANCA, kernels::rope(campo(&mut self.qr)?.partition([2, half]), &*q2, &self.cs).generics(gen_rope.clone()).async_on(s).map_err(fail_k(raw))) }
            .map(|_| ())
        })?;
        com_forma(&mut self.k, &[c.kv_heads * 2, half], ausente, || sync_before_drop(raw), |k2| {
            if injeta_lancamento(2) {
                return Err(seguro(CudaError::Config("falha injetada no lançamento reformado 2")));
            }
            if injeta_o3(2) {
                return Err(seguro(CudaError::Config("falha injetada no lançamento O3 2")));
            }
            // SAFETY: como acima. O3: o RoPE de K grava direto na linha `cur` do cache K (sem a cópia device-para-device).
            unsafe { med!(pb::B_LANCA, kernels::rope_kv((&mut lay.k_cache).partition([1, d]).map([1, 1], c.kv_heads as u32), &*k2, &self.cs, linha).generics(gen_rope_kv.clone()).async_on(s).map_err(fail_k(raw))) }
            .map(|_| ())
        })?;
        // `qr` (`[cabeças·2, metade]`, saída do RoPE de Q) vira `[cabeças, d]` para a atenção.
        com_forma(&mut self.qr, &[c.heads, d], ausente, || sync_before_drop(raw), |qh| {
            if injeta_lancamento(3) {
                return Err(seguro(CudaError::Config("falha injetada no lançamento reformado 3")));
            }
            // SAFETY: `lens` está entre 1 e `max_len` (`begin_token`); as formas são as de `new`.
            unsafe { attention_b(s, campo(&mut self.att2)?, &mut self.part, &*qh, &lay.k_cache, &lay.v_cache, &self.lens, c.attention_multiplier, (c.heads, c.kv_heads, d, ATTN_NS), acc.as_deref_mut()) }
                .map_err(fail_k(raw))
        })?;
        // `att2` (`[cabeças, d]`, saída da atenção) vira `[qd]` para o gemv de `o`.
        com_forma(&mut self.att2, &[qd], ausente, || sync_before_drop(raw), |att| {
            if injeta_lancamento(4) {
                return Err(seguro(CudaError::Config("falha injetada no lançamento reformado 4")));
            }
            // SAFETY: como acima.
            unsafe { med!(pb::B_LANCA, kernels::gemv((&mut self.o).partition([GEMV_BM]), &lay.o, &*att).generics(gen_qd).async_on(s).map_err(fail_k(raw))) }
            .map(|_| ())
        })?;
        // SAFETY: como acima.
        unsafe {
            med!(pb::B_LANCA, kernels::axpby((&mut *h).partition([AXPBY_B]), &self.o, c.residual_multiplier, 1.0f32).generics(vec![AXPBY_B.to_string()]).async_on(s).map_err(fail_k(raw)))?;
            med!(pb::B_LANCA, kernels::rmsnorm((&mut *xn).partition([norm_c]), &*h, &lay.ln_moe, c.eps).generics(gen_norm).async_on(s).map_err(fail_k(raw)))?;
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
