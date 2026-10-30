//! Caminho de GPU: experts residentes na VRAM e kernels Tile do cutile-rs.
//!
//! Por token: `x` vai para a GPU, o roteador roda lá e os logits voltam ao host (a escolha dos experts fica no host
//! porque é o host que decide o que está no cache); depois os k experts são enfileirados na mesma stream sem
//! sincronizar (dois lançamentos por expert: GEMV gate_up com SwiGLU na saída, GEMV down com `y = β·y + g·o` na saída) e
//! só a leitura de `y` espera a GPU.
use crate::block::{HostLayer, Routing};
use cuda_core::Stream;
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::{Partition, Reshape};
use std::sync::Arc;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// `BM` linhas de `y = W·x`, W BF16 e x f32, acumulando em f32 (como o GEMV do M0, mas com a ativação em f32).
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemv<const BM: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        w: &Tensor<bf16, { [-1, K] }>,
        x: &Tensor<f32, { [K] }>,
    ) {
        let part_w = w.partition(const_shape![BM, BK]);
        let part_x = x.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
        for i in 0i32..(K / BK) {
            let tw: Tile<bf16, { [BM, BK] }> = part_w.load([pid.0, i]);
            let tx: Tile<f32, { [1, BK] }> = part_x.load([i]).reshape(const_shape![1, BK]);
            let tw: Tile<f32, { [BM, BK] }> = convert_tile(tw);
            acc = acc + tw * tx.broadcast(const_shape![BM, BK]);
        }
        let r: Tile<f32, { [BM] }> = reduce_sum(acc, 1i32);
        y.store(r);
    }

    /// GEMV do gate_up com o SwiGLU na saída: `a[r] = silu(g[r])·u[r]`, com `g = W[..I]·x` e `u = W[I..]·x`. Cada bloco
    /// calcula `BM` linhas do gate e as `BM` linhas do up que casam com elas (`I/BM` blocos), lendo os mesmos bytes do GEMV
    /// de `2I` linhas sem escrever o vetor `h` de `2I`. Mesma fórmula do `gemv` seguido do SwiGLU de antes do M2f, mas não os
    /// mesmos bits: o compilador contrai as contas de outro jeito (erro dos logits do 1B 7,0e-6 → 5,6e-6, ambos contra o
    /// transformers).
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemv_swiglu<const BM: i32, const BK: i32, const K: i32, const I: i32>(
        a: &mut Tensor<f32, { [BM] }>,
        w: &Tensor<bf16, { [-1, K] }>,
        x: &Tensor<f32, { [K] }>,
    ) {
        let part_w = w.partition(const_shape![BM, BK]);
        let part_x = x.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc_g: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
        let mut acc_u: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
        for i in 0i32..(K / BK) {
            let tx: Tile<f32, { [1, BK] }> = part_x.load([i]).reshape(const_shape![1, BK]);
            let txb: Tile<f32, { [BM, BK] }> = tx.broadcast(const_shape![BM, BK]);
            let tg: Tile<bf16, { [BM, BK] }> = part_w.load([pid.0, i]);
            let tu: Tile<bf16, { [BM, BK] }> = part_w.load([pid.0 + I / BM, i]);
            let tg: Tile<f32, { [BM, BK] }> = convert_tile(tg);
            let tu: Tile<f32, { [BM, BK] }> = convert_tile(tu);
            acc_g = acc_g + tg * txb;
            acc_u = acc_u + tu * txb;
        }
        // Como na atenção: a saída do `reduce_sum` só combina com outros tiles depois de um `reshape`.
        let g: Tile<f32, { [BM] }> = reduce_sum(acc_g, 1i32);
        let g: Tile<f32, { [BM] }> = g.reshape(const_shape![BM]);
        let u: Tile<f32, { [BM] }> = reduce_sum(acc_u, 1i32);
        let u: Tile<f32, { [BM] }> = u.reshape(const_shape![BM]);
        let one: Tile<f32, { [BM] }> = constant(1.0, const_shape![BM]);
        let silu: Tile<f32, { [BM] }> = g / (one + exp(negf(g)));
        a.store(silu * u);
    }

    /// GEMV do down com o acúmulo do expert na saída: `y = β·y + g·(W·x)` (β = 0 no primeiro expert do token). Mesma fórmula
    /// do `gemv` seguido do `axpby` de antes do M2f. β = 0 descarta um `y` anterior finito, mas não um NaN/Inf (0·NaN = NaN):
    /// o `y` do scratch só fica não finito se um token já saiu inválido, e depois de um erro a camada é descartada.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemv_axpby<const BM: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        w: &Tensor<bf16, { [-1, K] }>,
        x: &Tensor<f32, { [K] }>,
        g: f32,
        beta: f32,
    ) {
        let part_w = w.partition(const_shape![BM, BK]);
        let part_x = x.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
        for i in 0i32..(K / BK) {
            let tw: Tile<bf16, { [BM, BK] }> = part_w.load([pid.0, i]);
            let tx: Tile<f32, { [1, BK] }> = part_x.load([i]).reshape(const_shape![1, BK]);
            let tw: Tile<f32, { [BM, BK] }> = convert_tile(tw);
            acc = acc + tw * tx.broadcast(const_shape![BM, BK]);
        }
        let r: Tile<f32, { [BM] }> = reduce_sum(acc, 1i32);
        let r: Tile<f32, { [BM] }> = r.reshape(const_shape![BM]);
        let ty: Tile<f32, { [BM] }> = y.load();
        let tg: Tile<f32, { [BM] }> = g.broadcast(const_shape![BM]);
        let tb: Tile<f32, { [BM] }> = beta.broadcast(const_shape![BM]);
        y.store(ty * tb + r * tg);
    }

    /// `out = w ⊙ x / √(média(x²) + eps)` em fatias de `C` (`N` = tamanho, múltiplo de `C`): cada bloco soma os quadrados
    /// do vetor inteiro e escreve a própria fatia. Assim `N` não precisa ser potência de dois, só `C`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn rmsnorm<const N: i32, const C: i32>(out: &mut Tensor<f32, { [C] }>, x: &Tensor<f32, { [-1] }>, w: &Tensor<f32, { [-1] }>, eps: f32) {
        let part_x = x.partition(const_shape![C]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [C] }> = constant(0.0, const_shape![C]);
        for i in 0i32..(N / C) {
            let t: Tile<f32, { [C] }> = part_x.load([i]);
            acc = acc + t * t;
        }
        let sq: Tile<f32, { [1, C] }> = acc.reshape(const_shape![1, C]);
        let ss: Tile<f32, { [1] }> = reduce_sum(sq, 1i32);
        let ss: Tile<f32, { [] }> = ss.reshape(shape![]);
        let ss: f32 = tile_to_scalar(ss);
        let n: f32 = convert_scalar(N);
        let r: Tile<f32, { [] }> = sqrt(scalar_to_tile(ss / n + eps), rounding::NearestEven, ftz::Disabled);
        let r: f32 = tile_to_scalar(r);
        let r: f32 = 1.0f32 / r;
        let tx: Tile<f32, { [C] }> = part_x.load([pid.0]);
        let tw: Tile<f32, { [C] }> = w.partition(const_shape![C]).load([pid.0]);
        let tr: Tile<f32, { [C] }> = r.broadcast(const_shape![C]);
        out.store(tw * (tx * tr));
    }

    /// RoPE `rotate_half` numa cabeça por bloco: `x` visto como `[cabeças·2, HALF]` (as duas metades de cada cabeça em
    /// linhas seguidas), `cs = [cos; sin]` da posição.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn rope<const HALF: i32>(out: &mut Tensor<f32, { [2, HALF] }>, x: &Tensor<f32, { [-1, HALF] }>, cs: &Tensor<f32, { [-1, HALF] }>) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let part_x = x.partition(const_shape![1, HALF]);
        let x1: Tile<f32, { [1, HALF] }> = part_x.load([2i32 * pid.0, 0i32]);
        let x2: Tile<f32, { [1, HALF] }> = part_x.load([2i32 * pid.0 + 1i32, 0i32]);
        let part_cs = cs.partition(const_shape![1, HALF]);
        let c: Tile<f32, { [1, HALF] }> = part_cs.load([0i32, 0i32]);
        let s: Tile<f32, { [1, HALF] }> = part_cs.load([1i32, 0i32]);
        let mut part_out: PartitionMut<f32, { [1, HALF] }> = out.partition_mut(const_shape![1, HALF]);
        part_out.store(x1 * c - x2 * s, [0i32, 0i32]);
        part_out.store(x2 * c + x1 * s, [1i32, 0i32]);
    }

    /// Atenção de decode, primeira passada (divisão do KV, como no flash-decoding): o bloco `h·ns + sp` cuida da cabeça de Q
    /// `h` (cabeça de KV `h / G`) contra a fatia `sp` das posições do cache `[posições, kv·D]`. As `nb = ⌈len/BN⌉` fatias de
    /// `BN` posições se dividem entre os `ns` blocos da cabeça, `⌈nb/ns⌉` por bloco; `len` vem de um tensor da GPU, então a
    /// grade (`cabeças × ns`) não depende da posição nem de `max_len`. Escreve a saída normalizada na linha 0 e o log da soma
    /// das exponenciais (`m + ln l`) na linha 1; um bloco sem posições sai com 0 e −∞, e a segunda passada não o lê.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn attn_split<const BN: i32, const D: i32, const G: i32>(
        out: &mut Tensor<f32, { [2, D] }>,
        q: &Tensor<f32, { [-1, D] }>,
        kc: &Tensor<f32, { [-1, -1] }>,
        vc: &Tensor<f32, { [-1, -1] }>,
        lens: &Tensor<i32, { [-1] }>,
        scale: f32,
        ns: i32,
    ) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let h: i32 = pid.0 / ns;
        let sp: i32 = pid.0 - h * ns;
        let kvh: i32 = h / G;
        let len: i32 = tile_to_scalar(lens.partition(const_shape![1]).load([0i32]).reshape(const_shape![]));
        let tq: Tile<f32, { [1, D] }> = q.partition(const_shape![1, D]).load([h, 0i32]);
        let tq: Tile<f32, { [BN, D] }> = tq.broadcast(const_shape![BN, D]);
        let part_k = kc.partition(const_shape![BN, D]);
        let part_v = vc.partition(const_shape![BN, D]);
        // Tudo em [1, BN]: o resultado de `reduce_sum` só combina com os outros tiles depois de um `reshape`.
        let offs: Tile<i32, { [BN] }> = iota(const_shape![BN]);
        let offs: Tile<i32, { [1, BN] }> = offs.reshape(const_shape![1, BN]);
        let len_t: Tile<i32, { [1, BN] }> = broadcast_scalar(len, const_shape![1, BN]);
        let zero: Tile<f32, { [1, BN] }> = constant(0.0f32, const_shape![1, BN]);
        let ninf: Tile<f32, { [1, BN] }> = constant(f32::NEG_INFINITY, const_shape![1, BN]);
        let ts: Tile<f32, { [1, BN] }> = broadcast_scalar(scale, const_shape![1, BN]);
        let mut m_i: Tile<f32, { [1, 1] }> = constant(f32::NEG_INFINITY, const_shape![1, 1]);
        let mut l_i: Tile<f32, { [1, 1] }> = constant(0.0f32, const_shape![1, 1]);
        let mut acc: Tile<f32, { [1, D] }> = constant(0.0f32, const_shape![1, D]);
        // Só as fatias de BN com alguma posição antes de `len`: uma fatia toda mascarada daria exp(−∞ + ∞).
        let nb: i32 = ceil_div(len, BN);
        let per: i32 = ceil_div(nb, ns);
        let j0: i32 = sp * per;
        let j1: i32 = min(j0 + per, nb);
        for j in j0..j1 {
            let kt: Tile<f32, { [BN, D] }> = part_k.load([j, kvh]);
            let sc: Tile<f32, { [BN] }> = reduce_sum(kt * tq, 1i32);
            let sc: Tile<f32, { [1, BN] }> = sc.reshape(const_shape![1, BN]);
            let pos: Tile<i32, { [1, BN] }> = broadcast_scalar(j * BN, const_shape![1, BN]) + offs;
            let mask: Tile<bool, { [1, BN] }> = lt_tile(pos, len_t);
            let sc: Tile<f32, { [1, BN] }> = sc * ts + select(mask, zero, ninf);
            let mx: Tile<f32, { [1] }> = reduce_max(sc, 1i32);
            let m_new: Tile<f32, { [1, 1] }> = max_tile(m_i, mx.reshape(const_shape![1, 1]));
            let p: Tile<f32, { [1, BN] }> = exp(sc - m_new.broadcast(const_shape![1, BN]));
            let ps: Tile<f32, { [1] }> = reduce_sum(p, 1i32);
            let alpha: Tile<f32, { [1, 1] }> = exp(m_i - m_new);
            l_i = l_i * alpha + ps.reshape(const_shape![1, 1]);
            let vt: Tile<f32, { [BN, D] }> = part_v.load([j, kvh]);
            let pv: Tile<f32, { [BN, D] }> = p.reshape(const_shape![BN, 1]).broadcast(const_shape![BN, D]) * vt;
            let pv: Tile<f32, { [D] }> = reduce_sum(pv, 0i32);
            acc = acc * alpha.broadcast(const_shape![1, D]) + pv.reshape(const_shape![1, D]);
            m_i = m_new;
        }
        // Fatia vazia: l = 0, acc = 0; o piso em l evita 0/0 e dá ln l = −∞ em vez de NaN.
        let tiny: Tile<f32, { [1, 1] }> = constant(1e-30f32, const_shape![1, 1]);
        let l_g: Tile<f32, { [1, 1] }> = max_tile(l_i, tiny);
        let lse: Tile<f32, { [1, 1] }> = m_i + log(l_g);
        let mut part_out: PartitionMut<f32, { [1, D] }> = out.partition_mut(const_shape![1, D]);
        part_out.store(true_div(acc, l_g.broadcast(const_shape![1, D])), [0i32, 0i32]);
        part_out.store(lse.broadcast(const_shape![1, D]), [1i32, 0i32]);
    }

    /// Atenção de decode, segunda passada: uma cabeça por bloco junta, em ordem, os blocos de `attn_split` que têm alguma
    /// posição antes de `len` (`part` = `[cabeças·ns·2, D]`), com o softmax online sobre os `lse`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn attn_combine<const BN: i32, const D: i32>(out: &mut Tensor<f32, { [1, D] }>, part: &Tensor<f32, { [-1, D] }>, lens: &Tensor<i32, { [-1] }>, ns: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let len: i32 = tile_to_scalar(lens.partition(const_shape![1]).load([0i32]).reshape(const_shape![]));
        // A mesma divisão de `attn_split`: os blocos `0..used` têm posições.
        let nb: i32 = ceil_div(len, BN);
        let used: i32 = ceil_div(nb, ceil_div(nb, ns));
        let pp = part.partition(const_shape![1, D]);
        let base: i32 = pid.0 * ns * 2;
        // A fatia 0 sempre tem posições (len ≥ 1): o máximo começa finito.
        let mut acc: Tile<f32, { [1, D] }> = pp.load([base, 0i32]);
        let mut m: Tile<f32, { [1, D] }> = pp.load([base + 1i32, 0i32]);
        let mut den: Tile<f32, { [1, D] }> = constant(1.0f32, const_shape![1, D]);
        for sp in 1i32..used {
            let o: Tile<f32, { [1, D] }> = pp.load([base + 2i32 * sp, 0i32]);
            let lse: Tile<f32, { [1, D] }> = pp.load([base + 2i32 * sp + 1i32, 0i32]);
            let m_new: Tile<f32, { [1, D] }> = max_tile(m, lse);
            let a: Tile<f32, { [1, D] }> = exp(m - m_new);
            let w: Tile<f32, { [1, D] }> = exp(lse - m_new);
            acc = acc * a + o * w;
            den = den * a + w;
            m = m_new;
        }
        out.store(true_div(acc, den));
    }

    /// `y = β·y + g·o`, `B` elementos por bloco (β = 0 no primeiro expert do token).
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn axpby<const B: i32>(y: &mut Tensor<f32, { [B] }>, o: &Tensor<f32, { [-1] }>, g: f32, beta: f32) {
        let part_o = o.partition(const_shape![B]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let to: Tile<f32, { [B] }> = part_o.load([pid.0]);
        let ty: Tile<f32, { [B] }> = y.load();
        let tg: Tile<f32, { [B] }> = g.broadcast(const_shape![B]);
        let tb: Tile<f32, { [B] }> = beta.broadcast(const_shape![B]);
        y.store(ty * tb + to * tg);
    }
}

/// Tamanhos de bloco. GEMV: o M0 mediu BM=2 com BK=K como o melhor para matrizes de 1024 colunas nesta GPU.
pub const GEMV_BM: usize = 2;
/// Linhas por bloco do `gemv_swiglu`. Cada bloco já lê duas matrizes (gate e up), então BM = 1 devolve a grade de `2I/2`
/// blocos do GEMV de `2I` linhas: no M2f, 2,53 → 2,16 µs no 1B e 2,93 → 2,30 µs no 3B (`m2c-kernels`, grafo). O
/// `gemv_axpby` não ganhou com BM = 1 (1,64 → 1,60 e 1,83 → 1,85 µs) e fica com `GEMV_BM`.
pub const SWIGLU_BM: usize = 1;
pub const AXPBY_B: usize = 256;

/// Buffers de trabalho na VRAM, reaproveitados entre tokens. Cada kernel recebe o buffer de saída e o devolve.
struct Scratch {
    logits: Partition<Tensor<f32>>,
    a: Partition<Tensor<f32>>,
    y: Partition<Tensor<f32>>,
}

const POISONED: &str = "camada MoE inutilizável depois de um erro anterior";

/// Uma camada com todos os experts residentes na VRAM.
pub struct GpuLayer {
    pub hidden: usize,
    pub inter: usize,
    pub k: usize,
    router: Tensor<bf16>,
    gate_up: Vec<Tensor<bf16>>,
    down: Vec<Tensor<bf16>>,
    stream: Arc<Stream>,
    scratch: Option<Scratch>,
    /// Do último `add_into_routed`.
    pub(crate) count: Count,
}

/// Operações enfileiradas (kernels e cópias) e sincronizações com o host, contadas no código que as faz (`m2b-custo-fixo`).
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Count {
    pub ops: usize,
    pub syncs: usize,
}

/// Para `.map_err` num `?` depois de enfileirar trabalho: espera a stream antes de o erro sair da função, enquanto os tensores
/// locais ainda vivem. Sem isso eles cairiam no retorno com kernels pendentes: `async_on` não segura os recursos depois de
/// enfileirar (`cuda-async/src/device_operation.rs`, doc de `async_on`), e `DeviceBuffer::drop` libera com `cuMemFreeAsync`
/// numa stream de desalocação que não espera as nossas (`cuda-async/src/device_buffer.rs:128`).
pub(crate) fn sync_on_err<E: Into<Error>>(s: &Stream) -> impl FnOnce(E) -> Error + '_ {
    move |e| {
        // SAFETY: só espera a stream.
        let _ = unsafe { s.synchronize() };
        e.into()
    }
}

/// Maior tile ao longo de K no GEMV e no RMSNorm.
pub const TILE_K_MAX: usize = 1024;

/// Tile ao longo de um vetor de tamanho `k`: a maior potência de dois que divide `k`, até [`TILE_K_MAX`]. O cuTile só aceita
/// tiles em potência de dois, então H = 1536 (Granite 3B) vira 3 fatias de 512, e H = 1024 continua uma fatia só.
pub fn tile_k(k: usize) -> usize {
    assert!(k > 0);
    (1usize << k.trailing_zeros()).min(TILE_K_MAX)
}

/// Parâmetros do GEMV para um vetor de entrada de tamanho `k`: `[BM, BK, K]`.
pub fn gemv_generics(k: usize) -> Vec<String> {
    vec![GEMV_BM.to_string(), tile_k(k).to_string(), k.to_string()]
}

/// Generics dos dois lançamentos de um expert, montados uma vez por camada.
pub(crate) struct ExpertGen {
    h: usize,
    i: usize,
    gate_up: Vec<String>,
    down: Vec<String>,
}

impl ExpertGen {
    pub(crate) fn new(h: usize, i: usize) -> Self {
        let gate_up = vec![SWIGLU_BM.to_string(), tile_k(h).to_string(), h.to_string(), i.to_string()];
        Self { h, i, gate_up, down: gemv_generics(i) }
    }
}

/// Enfileira um expert: `a = SwiGLU(W_gate_up·x)` (`kernels::gemv_swiglu`) e `y = β·y + g·(W_down·a)` (`kernels::gemv_axpby`),
/// com `a` (I) particionado em `SWIGLU_BM` e `y` (H) em `GEMV_BM`. Cada um tem nome de kernel próprio: o cache de lançamento do cutile-rs
/// guarda uma especialização por função, e K = H e K = I num mesmo nome se derrubariam a cada expert (M2e).
///
/// # Safety
/// Enfileira sem sincronizar e sem conferir limites: H e I múltiplos de `GEMV_BM`, `gate_up` com `[2I, H]`, `down` com
/// `[H, I]`, e quem chama mantém pesos e `xd` vivos até a próxima sincronização da stream.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn enqueue_expert(
    s: &Arc<Stream>,
    gen: &ExpertGen,
    a: Partition<Tensor<f32>>,
    y: Partition<Tensor<f32>>,
    gate_up: &Tensor<bf16>,
    down: &Tensor<bf16>,
    xd: &Tensor<f32>,
    g: f32,
    beta: f32,
) -> Result<(Partition<Tensor<f32>>, Partition<Tensor<f32>>), Error> {
    // Os kernels não conferem limites e a grade sai da forma de `a` e `y`: uma forma errada leria ou escreveria fora dos
    // tensores em silêncio. Comparações de inteiros, uma vez por expert.
    let (h, i) = (gen.h as i32, gen.i as i32);
    assert!(gate_up.shape() == [2 * i, h] && down.shape() == [h, i] && xd.shape() == [h], "expert: formas dos pesos ou de x");
    assert!(a.num_bytes() == gen.i * 4 && y.num_bytes() == gen.h * 4, "expert: formas de a ou de y");
    let (aa, _, _) = kernels::gemv_swiglu(a, gate_up, xd).generics(gen.gate_up.clone()).async_on(s).map_err(sync_on_err(s))?;
    let aa = aa.unpartition();
    let (yy, _, _, _, _) = kernels::gemv_axpby(y, down, &aa, g, beta).generics(gen.down.clone()).async_on(s).map_err(sync_on_err(s))?;
    Ok((aa.partition([SWIGLU_BM]), yy))
}

/// Os kernels que ainda carregam a dimensão inteira num tile (RoPE e atenção, em `head_dim`) exigem potência de dois. Sem esta
/// checagem, a forma errada passaria pelo upload e só falharia na compilação JIT do primeiro token.
pub fn assert_tile_dims(dims: &[(&str, usize)]) {
    let bad: Vec<String> = dims.iter().filter(|(_, v)| !v.is_power_of_two()).map(|(n, v)| format!("{n}={v}")).collect();
    assert!(bad.is_empty(), "dimensões usadas como tile têm de ser potência de dois: {}", bad.join(", "));
}

/// Bytes de VRAM ocupados pelos pesos de uma camada (roteador + experts).
pub fn layer_weight_bytes(host: &HostLayer) -> usize {
    host.router.len() * 2 + host.experts() * host.expert_bytes()
}

impl GpuLayer {
    pub fn upload(host: &HostLayer, stream: Arc<Stream>) -> Result<Self, Error> {
        let (h, i, e) = (host.hidden, host.inter, host.experts());
        let check = h % GEMV_BM == 0 && i % GEMV_BM == 0 && e % GEMV_BM == 0 && h % AXPBY_B == 0;
        assert!(check, "formas fora dos tamanhos de bloco: H={h} I={i} E={e}");
        let put = |w: &[bf16], rows: usize, cols: usize| -> Result<Tensor<bf16>, Error> {
            Ok(api::copy_host_vec_to_device(&Arc::new(w.to_vec())).sync_on(&stream)?.reshape(&[rows, cols])?)
        };
        let router = put(&host.router, e, h)?;
        let gate_up = host.gate_up.iter().map(|w| put(w, 2 * i, h)).collect::<Result<_, _>>()?;
        let down = host.down.iter().map(|w| put(w, h, i)).collect::<Result<_, _>>()?;
        let zeros = |n: usize, b: usize| -> Result<Partition<Tensor<f32>>, Error> {
            Ok(api::zeros::<f32>(&[n]).sync_on(&stream)?.partition([b]))
        };
        let scratch = Scratch {
            logits: zeros(e, GEMV_BM)?,
            a: zeros(i, SWIGLU_BM)?,
            y: zeros(h, GEMV_BM)?,
        };
        Ok(Self { hidden: h, inter: i, k: host.k, router, gate_up, down, stream, scratch: Some(scratch), count: Count::default() })
    }

    /// Espera a stream depois de um erro. Os `?` das funções internas já sincronizam antes de soltar os locais (`sync_on_err`);
    /// esta segunda espera cobre um erro que saia por outro caminho.
    fn sync_after<T>(&self, out: Result<T, Error>) -> Result<T, Error> {
        if out.is_err() {
            // SAFETY: só espera a stream.
            let _ = unsafe { self.stream.synchronize() };
        }
        out
    }

    /// Bloco inteiro na GPU para um token, com `x` e `y` no host. Depois de um erro a camada fica inutilizável (as chamadas
    /// seguintes devolvem erro).
    pub fn forward(&mut self, x: &[f32]) -> Result<(Vec<f32>, Routing), Error> {
        let out = self.forward_with(x);
        self.sync_after(out)
    }

    fn forward_with(&mut self, x: &[f32]) -> Result<(Vec<f32>, Routing), Error> {
        assert_eq!(x.len(), self.hidden);
        let xd: Tensor<f32> = api::copy_host_vec_to_device(&Arc::new(x.to_vec())).sync_on(&self.stream).map_err(sync_on_err(&self.stream))?;
        let r = self.run(&xd, None).map_err(sync_on_err(&self.stream))?;
        let mut sc = self.scratch.take().ok_or_else(|| tensor_error(POISONED)).map_err(sync_on_err(&self.stream))?;
        let y = Arc::new(sc.y.unpartition());
        let yv = api::copy_device_to_host_vec(&y).sync_on(&self.stream).map_err(sync_on_err(&self.stream))?;
        sc.y = Arc::try_unwrap(y).map_err(|_| "y ainda referenciado").unwrap().partition([GEMV_BM]);
        self.scratch = Some(sc);
        Ok((yv, r))
    }

    /// `h ← h + resid·moe(xd)`, com `xd` e `h` na VRAM (`h` particionado em `AXPBY_B`). Só a leitura dos logits do
    /// roteador sincroniza; o resto fica enfileirado na stream.
    pub fn add_into(&mut self, xd: &Tensor<f32>, h: Partition<Tensor<f32>>, resid: f32) -> Result<(Partition<Tensor<f32>>, Routing), Error> {
        self.add_into_routed(xd, h, resid, None)
    }

    /// Como [`add_into`](Self::add_into), mas com o roteamento dado de fora quando `given` é `Some`: pula o GEMV do roteador e a
    /// leitura dos logits, e então nada sincroniza. Serve para medir quanto custam essas sincronizações (`m2b-custo-fixo`).
    pub fn add_into_routed(&mut self, xd: &Tensor<f32>, h: Partition<Tensor<f32>>, resid: f32, given: Option<&Routing>) -> Result<(Partition<Tensor<f32>>, Routing), Error> {
        let out = self.add_into_with(xd, h, resid, given);
        self.sync_after(out)
    }

    fn add_into_with(&mut self, xd: &Tensor<f32>, h: Partition<Tensor<f32>>, resid: f32, given: Option<&Routing>) -> Result<(Partition<Tensor<f32>>, Routing), Error> {
        self.count = Count::default();
        let r = self.run(xd, given).map_err(sync_on_err(&self.stream))?;
        let mut sc = self.scratch.take().ok_or_else(|| tensor_error(POISONED)).map_err(sync_on_err(&self.stream))?;
        let y = sc.y.unpartition();
        // SAFETY: `unchecked_accesses`: H é múltiplo de AXPBY_B (conferido em `upload`). `async_on`: `y` volta para o
        // scratch e só é reescrito por kernels desta stream; quem chama mantém `xd` vivo até sincronizar.
        let (h, _, _, _) = unsafe { kernels::axpby(h, &y, resid, 1.0f32).generics(vec![AXPBY_B.to_string()]).async_on(&self.stream).map_err(sync_on_err(&self.stream))? };
        self.count.ops += 1;
        sc.y = y.partition([GEMV_BM]);
        self.scratch = Some(sc);
        Ok((h, r))
    }

    /// Roteador (com leitura dos logits, a menos que `given` traga o roteamento) e os k experts; a saída fica em `scratch.y`.
    fn run(&mut self, xd: &Tensor<f32>, given: Option<&Routing>) -> Result<Routing, Error> {
        let (h, i) = (self.hidden, self.inter);
        let s = &self.stream;
        let mut sc = self.scratch.take().ok_or_else(|| tensor_error(POISONED)).map_err(sync_on_err(s))?;

        // Roteador e leitura dos logits.
        let gen_h = gemv_generics(h);
        let r = match given {
            Some(r) => r.clone(),
            None => {
                // SAFETY: `unchecked_accesses`: E e H são múltiplos de BM e BK (conferido em `upload`).
                let (logits, _, _) = unsafe { kernels::gemv(sc.logits, &self.router, xd) }.generics(gen_h.clone()).sync_on(s).map_err(sync_on_err(s))?;
                let logits = Arc::new(logits.unpartition());
                let lv = api::copy_device_to_host_vec(&logits).sync_on(s).map_err(sync_on_err(s))?;
                self.count.ops += 2;
                self.count.syncs += 2;
                sc.logits = Arc::try_unwrap(logits).map_err(|_| "logits ainda referenciados").unwrap().partition([GEMV_BM]);
                Routing::from_logits(&lv, self.k)
            }
        };

        // Experts: tudo enfileirado na stream.
        let gen = ExpertGen::new(h, i);
        for (n, (&e, &g)) in r.experts.iter().zip(&r.weights).enumerate() {
            let beta = if n == 0 { 0.0f32 } else { 1.0 };
            // SAFETY: as formas são múltiplas dos tamanhos de bloco (conferido em `upload`); pesos e `xd` vivem até a
            // próxima sincronização de quem chama.
            let (a, y) = unsafe { enqueue_expert(s, &gen, sc.a, sc.y, &self.gate_up[e], &self.down[e], xd, g, beta)? };
            (sc.a, sc.y) = (a, y);
            self.count.ops += 2;
        }
        self.scratch = Some(sc);
        Ok(r)
    }
}

impl Drop for GpuLayer {
    fn drop(&mut self) {
        // Kernels enfileirados com `async_on` ainda podem ler os pesos e o scratch.
        // SAFETY: só espera a stream.
        let _ = unsafe { self.stream.synchronize() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enqueue_expert_recusa_a_com_tamanho_errado() {
        let device = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
                return;
            }
        };
        let s = device.new_stream().unwrap();
        let (h, i) = (256usize, 128usize);
        let w = |r: usize, k: usize| api::zeros::<bf16>(&[r * k]).sync_on(&s).unwrap().reshape(&[r, k]).unwrap();
        let (gate_up, down) = (w(2 * i, h), w(h, i));
        let xd = api::zeros::<f32>(&[h]).sync_on(&s).unwrap();
        // `a` com 2I em vez de I: a grade do `gemv_swiglu` teria o dobro de blocos e leria o up como se fosse o gate.
        let a = api::zeros::<f32>(&[2 * i]).sync_on(&s).unwrap().partition([SWIGLU_BM]);
        let y = api::zeros::<f32>(&[h]).sync_on(&s).unwrap().partition([GEMV_BM]);
        let gen = ExpertGen::new(h, i);
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: a asserção recusa antes de qualquer lançamento.
            let _ = unsafe { enqueue_expert(&s, &gen, a, y, &gate_up, &down, &xd, 1.0, 0.0) };
        }));
        let msg = r.expect_err("enqueue_expert aceitou a com tamanho errado");
        let msg = msg.downcast_ref::<String>().cloned().or_else(|| msg.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
        assert!(msg.contains("expert: formas de a ou de y"), "pânico com outra mensagem: {msg}");
    }
}
