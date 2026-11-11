//! O `trait Backend` do protocolo D1: as operações de stream que o motor pede (plano, A0b, "Backend falso por trait").
//!
//! No backend de CUDA (A1.2/A1.3; no A1.1 ficam só a interface e a correspondência abaixo):
//! - `copy_expert`: `cuMemcpyHtoDAsync` do expert pinned para o slot do banco;
//! - `record` / `wait_event` / `query`: `cuEventRecord` / `cuStreamWaitEvent` / `cuEventQuery` (a espera captura a última
//!   gravação feita antes da chamada; um evento nunca gravado responde completo);
//! - `upload_table`: `cuMemcpyHtoDAsync` do lado pinned da tabela para a tabela do device;
//! - `route`: o roteador e o top-k na GPU, que escrevem ids, pesos do softmax e `x` na área mapeada da camada;
//! - `launch_hits`: a seleção dos acertos pela tabela do token e o lançamento agrupado (passo 5);
//! - `add_y`: a soma da saída da CPU, lida da área mapeada;
//! - `write_value` / `wait_value`: `cuStreamWriteValue32_v2` / `cuStreamWaitValue32_v2` com `GEQ` cíclico;
//! - `sync`: `cuStreamSynchronize`.
use super::mem::{Device, Mapped, Pinned};
use super::sync::{spin_hint, AtomicU32, Ordering};
use std::fmt::Debug;

/// O `GEQ` cíclico do `cuStreamWaitValue32` (`(int32_t)(a − b) >= 0`, `/opt/cuda/include/cuda.h:535`).
pub fn geq(a: u32, b: u32) -> bool {
    (a.wrapping_sub(b) as i32) >= 0
}

/// Valor de soltura para o token corrente `t` (passo 6): `t + 2^30`, na meia faixa à frente de toda espera em curso.
pub fn release_value(t: u32) -> u32 {
    t.wrapping_add(1 << 30)
}

/// A área pinned mapeada de uma camada: a GPU escreve ids (na ordem do top-k), os pesos do softmax (bits de `f32`, na mesma ordem) e `x`; o host escreve `y`. Os atômicos são
/// lidos e escritos com `Relaxed`: a ordem vem da sequência (`Acquire`) e da flag (`Release`) do doorbell.
pub struct LayerIo {
    pub ids: Words,
    pub w: Words,
    pub x: Words,
    pub y: Words,
    /// O tamanho do passo (`m`), uma palavra no fim da área. Só existe com `max_positions > 1`; com 1 posição o campo é vazio e o
    /// layout é o de sempre (a linha de serviço usa `m = 1` sem ler nada).
    pub m: Words,
}

/// Os deslocamentos (em palavras) e o tamanho da área de uma camada para `max_positions` posições. Fonte única do layout: a
/// `LayerIo`, o backend falso e o backend de CUDA usam esta função. Com `max_positions == 1` dá `0, topk, 2·topk, 2·topk + width` e
/// `2·topk + 2·width`, o layout de antes do passo de várias posições (sem a palavra `m`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerOffsets {
    pub ids: usize,
    pub w: usize,
    pub x: usize,
    pub y: usize,
    /// A palavra do tamanho do passo, no fim; `None` com uma posição.
    pub m: Option<usize>,
    /// O tamanho total da área, em palavras.
    pub total: usize,
}

/// Palavras de 32 bits compartilhadas com a GPU: um bloco do heap (backend falso) ou uma fatia de memória host mapeada que o
/// backend de CUDA aloca. A representação é privada: só `LayerIo::new` (heap, seguro) e `LayerIo::from_raw` (`unsafe`,
/// só do crate) criam um `Words`, então código seguro não forja uma fatia.
pub struct Words(Repr);

enum Repr {
    Heap(Box<[AtomicU32]>),
    /// Fora do `loom`, que não tem `AtomicU32::from_ptr`.
    #[cfg(not(motor_loom))]
    Raw { ptr: *const AtomicU32, len: usize },
}

// SAFETY: `Raw` só nasce em `LayerIo::from_raw`, que exige memória viva e acessada por atômicos.
#[cfg(not(motor_loom))]
#[allow(unsafe_code)]
unsafe impl Send for Words {}
#[cfg(not(motor_loom))]
#[allow(unsafe_code)]
unsafe impl Sync for Words {}

impl std::ops::Deref for Words {
    type Target = [AtomicU32];
    #[allow(unsafe_code)]
    fn deref(&self) -> &[AtomicU32] {
        match &self.0 {
            Repr::Heap(b) => b,
            // SAFETY: o construtor `LayerIo::from_raw` exige `len` palavras vivas e alinhadas a partir de `ptr`.
            #[cfg(not(motor_loom))]
            Repr::Raw { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
        }
    }
}

impl LayerIo {
    /// Os deslocamentos da área para `max_positions` posições (ver [`LayerOffsets`]).
    pub fn offsets(topk: usize, width: usize, max_positions: usize) -> LayerOffsets {
        assert!(max_positions >= 1);
        let (ids, w) = (0, max_positions * topk);
        let x = 2 * max_positions * topk;
        let y = x + max_positions * width;
        let end = y + max_positions * width;
        let m = (max_positions > 1).then_some(end);
        LayerOffsets { ids, w, x, y, m, total: end + usize::from(m.is_some()) }
    }

    pub fn new(topk: usize, width: usize) -> Self {
        Self::with_positions(topk, width, 1)
    }

    /// Uma `LayerIo` no heap para `max_positions` posições.
    pub fn with_positions(topk: usize, width: usize, max_positions: usize) -> Self {
        let z = |n: usize| Words(Repr::Heap((0..n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>().into_boxed_slice()));
        let mp = max_positions;
        assert!(mp >= 1);
        Self { ids: z(mp * topk), w: z(mp * topk), x: z(mp * width), y: z(mp * width), m: z(usize::from(mp > 1)) }
    }

    /// O `m` publicado pela GPU (1 sem a palavra). Lido depois do `Acquire` de `seq[l]`.
    pub fn step_size(&self) -> usize {
        if self.m.is_empty() {
            1
        } else {
            self.m[0].load(Ordering::Relaxed) as usize
        }
    }

    /// Uma `LayerIo` sobre `2·topk + 2·width` palavras contíguas (ids, pesos, `x`, `y`), na ordem dos campos.
    ///
    /// # Safety
    /// `base` aponta essas palavras, alinhadas a 4 bytes, vivas enquanto a `LayerIo` e o `Mapped` que a guarda existirem, e
    /// só acessadas por atômicos (do host) ou pelos kernels do backend.
    #[cfg(not(motor_loom))]
    #[allow(unsafe_code)]
    pub(crate) unsafe fn from_raw(base: *const AtomicU32, topk: usize, width: usize) -> Self {
        // SAFETY: mesmas exigências, com uma posição.
        unsafe { Self::from_raw_positions(base, topk, width, 1) }
    }

    /// Como [`from_raw`](Self::from_raw), para `max_positions` posições: `LayerIo::offsets(topk, width, max_positions).total` palavras.
    ///
    /// # Safety
    /// As mesmas de `from_raw`, com `offsets(..).total` palavras vivas.
    #[cfg(not(motor_loom))]
    #[allow(unsafe_code)]
    pub(crate) unsafe fn from_raw_positions(base: *const AtomicU32, topk: usize, width: usize, max_positions: usize) -> Self {
        let o = Self::offsets(topk, width, max_positions);
        let mp = max_positions;
        // SAFETY: os deslocamentos ficam dentro das `o.total` palavras.
        unsafe {
            let at = |off: usize, len: usize| Words(Repr::Raw { ptr: base.add(off), len });
            Self { ids: at(o.ids, mp * topk), w: at(o.w, mp * topk), x: at(o.x, mp * width), y: at(o.y, mp * width),
                   m: at(o.m.unwrap_or(o.total), usize::from(o.m.is_some())) }
        }
    }

    pub fn load(v: &[AtomicU32]) -> Vec<u32> {
        v.iter().map(|a| a.load(Ordering::Relaxed)).collect()
    }
}

/// Qual linha do host chama (para os ganchos do escalonador do backend falso).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Main,
    Service,
}

pub trait Backend: Send + Sync + 'static {
    type Error: Debug + Send + 'static;
    type Stream: Send + Sync + 'static;
    type Event: Send + Sync + 'static;
    /// Os pesos de todos os experts na RAM (origem das cópias).
    type Weights: ?Sized + Send + Sync + 'static;
    /// O banco de slots no device.
    type Bank: ?Sized + 'static;

    /// Copia o expert `key` de `src` para o slot `slot` do banco, na stream `s`.
    fn copy_expert(&self, s: &Self::Stream, dst: &Device<Self::Bank>, slot: usize, src: &Pinned<Self::Weights>, key: u32)
        -> Result<(), Self::Error>;
    fn record(&self, ev: &Self::Event, s: &Self::Stream) -> Result<(), Self::Error>;
    /// A stream `s` espera a última gravação de `ev` feita antes desta chamada.
    fn wait_event(&self, s: &Self::Stream, ev: &Self::Event) -> Result<(), Self::Error>;
    /// Se a última gravação de `ev` feita antes desta chamada já terminou (nunca gravado: completo).
    fn query(&self, ev: &Self::Event) -> Result<bool, Self::Error>;
    /// Publica a tabela do token: o lado pinned vai para a tabela do device, na stream `s`.
    fn upload_table(&self, s: &Self::Stream, dst: &Device<[u32]>, src: &Pinned<[AtomicU32]>) -> Result<(), Self::Error>;
    /// O maior número de posições por passo que o backend aceita (as áreas e os kernels dele). O padrão é 1: um backend que só
    /// calcula uma posição não aceita `max_positions > 1` em silêncio.
    fn max_positions(&self) -> usize {
        1
    }
    /// Começo de um passo de `m` posições, a partir da posição `pos` da sequência, na stream `s` e na geração `t`. Enfileira (ou
    /// captura) o descritor do passo em ordem FIFO: operações anteriores do passo anterior não podem ver o valor novo. O padrão
    /// não faz nada (o backend falso com uma posição e o motor por token).
    fn begin_step(&self, _s: &Self::Stream, _t: u32, _pos: u64, _m: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Começa uma sequência nova no backend (posição do modelo volta a 0), com a stream de cálculo ociosa e entre dois tokens. Só o
    /// `Engine::nova_sequencia` chama. O padrão não faz nada (o backend falso sem estado de sequência).
    fn reset_sequence(&self) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Antes do `route` da camada: o que o modelo roda antes do bloco MoE (no A1.3, a atenção e a norma de entrada do bloco).
    /// O padrão não faz nada (o backend falso e a pilha de blocos MoE do A1.2).
    fn pre_layer(&self, _s: &Self::Stream, _layer: usize) -> Result<(), Self::Error> {
        Ok(())
    }
    /// Variante de [`pre_layer`](Self::pre_layer) com o acumulador da F2 nível B (`LASTRO_PERFIL_F2B`). O motor só a chama com a F2B ligada. O
    /// padrão é idêntico ao `pre_layer` (o backend falso e o desligado ficam iguais): deixa o acumulador vazio, e a IB3 invalida o pedido.
    fn pre_layer_f2b(&self, s: &Self::Stream, layer: usize, _acc: &mut crate::perfil_bloco::AccB) -> Result<(), Self::Error> {
        self.pre_layer(s, layer)
    }
    /// Depois do `add_y` da última camada: a norma final e a cabeça de saída (A1.3). O padrão não faz nada.
    fn post_layers(&self, _s: &Self::Stream) -> Result<(), Self::Error> {
        Ok(())
    }
    fn route(&self, s: &Self::Stream, layer: usize, io: &Mapped<LayerIo>) -> Result<(), Self::Error>;
    fn launch_hits(&self, s: &Self::Stream, layer: usize, table: &Device<[u32]>, bank: &Device<Self::Bank>, io: &Mapped<LayerIo>)
        -> Result<(), Self::Error>;
    fn add_y(&self, s: &Self::Stream, layer: usize, io: &Mapped<LayerIo>) -> Result<(), Self::Error>;
    fn write_value(&self, s: &Self::Stream, at: &Mapped<AtomicU32>, v: u32) -> Result<(), Self::Error>;
    /// A stream espera `*at >= v` no `GEQ` cíclico.
    fn wait_value(&self, s: &Self::Stream, at: &Mapped<AtomicU32>, v: u32) -> Result<(), Self::Error>;
    fn sync(&self, s: &Self::Stream) -> Result<(), Self::Error>;

    /// Uma volta de espera ativa do host (sequência, canal, `join`).
    fn relax(&self) -> Result<(), Self::Error> {
        spin_hint();
        Ok(())
    }
    fn on_thread_start(&self, _role: Role) {}
    fn on_thread_end(&self, _role: Role) {}
}

/// Um expert calculado na CPU num passo de várias posições: as posições que o pedem e o peso de cada uma (nas mesmas posições).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuGroup {
    pub key: u32,
    pub positions: Vec<usize>,
    pub weights: Vec<u32>,
}

/// O trabalho da CPU numa camada de um passo de `m` posições. `keys[p]` e `weights[p]` são as faltas da posição `p` na ordem do top-k
/// dela (o caminho de uma posição, que é a referência do passo); `groups` são as mesmas faltas por expert (a união em ordem da primeira
/// aparição), com as posições e os pesos de cada um.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuStep {
    pub m: usize,
    pub width: usize,
    pub keys: Vec<Vec<u32>>,
    pub weights: Vec<Vec<u32>>,
    pub groups: Vec<CpuGroup>,
}

/// Os experts calculados na CPU pela linha de serviço.
pub trait HostExperts: Send + 'static {
    type Error: Debug + Send + 'static;
    /// `y = Σ peso · expert(key, x)` sobre `keys` (as faltas da camada `layer`, na ordem do top-k); `weights` são os bits `f32`
    /// dos pesos do softmax das mesmas chaves, que a GPU publicou.
    fn compute(&mut self, layer: usize, keys: &[u32], weights: &[u32], x: &[u32], y: &mut [u32]) -> Result<(), Self::Error>;

    /// A camada de um passo de `step.m` posições: `x` e `y` têm `m·width` palavras e `y` é escrito inteiro (posições sem faltas ficam
    /// em zero). O padrão chama [`compute`](Self::compute) posição por posição, com as faltas dela na ordem do top-k: é o caminho de
    /// uma posição repetido, a referência do passo. Uma implementação em lote (U3 e E4b) sobrescreve este método.
    fn compute_step(&mut self, layer: usize, step: &CpuStep, x: &[u32], y: &mut [u32]) -> Result<(), Self::Error> {
        let w = step.width;
        for p in 0..step.m {
            self.compute(layer, &step.keys[p], &step.weights[p], &x[p * w..(p + 1) * w], &mut y[p * w..(p + 1) * w])?;
        }
        Ok(())
    }
}

#[cfg(all(test, not(motor_loom)))]
mod testes {
    use super::*;

    #[test]
    fn layer_io_tem_um_peso_por_id() {
        let io = LayerIo::new(8, 1536);
        assert_eq!((io.ids.len(), io.w.len(), io.x.len(), io.y.len()), (8, 8, 1536, 1536));
    }
}
