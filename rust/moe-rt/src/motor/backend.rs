//! O `trait Backend` do protocolo D1: as operações de stream que o motor pede (plano, A0b, "Backend falso por trait").
//!
//! No backend de CUDA (A1.2/A1.3; no A1.1 ficam só a interface e a correspondência abaixo):
//! - `copy_expert`: `cuMemcpyHtoDAsync` do expert pinned para o slot do banco;
//! - `record` / `wait_event` / `query`: `cuEventRecord` / `cuStreamWaitEvent` / `cuEventQuery` (a espera captura a última
//!   gravação feita antes da chamada; um evento nunca gravado responde completo);
//! - `upload_table`: `cuMemcpyHtoDAsync` do lado pinned da tabela para a tabela do device;
//! - `route`: o roteador e o top-k na GPU, que escrevem ids e `x` na área mapeada da camada;
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

/// A área pinned mapeada de uma camada: a GPU escreve ids (na ordem do top-k) e `x`; o host escreve `y`. Os atômicos são
/// lidos e escritos com `Relaxed`: a ordem vem da sequência (`Acquire`) e da flag (`Release`) do doorbell.
pub struct LayerIo {
    pub ids: Box<[AtomicU32]>,
    pub x: Box<[AtomicU32]>,
    pub y: Box<[AtomicU32]>,
}

impl LayerIo {
    pub fn new(topk: usize, width: usize) -> Self {
        let z = |n: usize| (0..n).map(|_| AtomicU32::new(0)).collect::<Vec<_>>().into_boxed_slice();
        Self { ids: z(topk), x: z(width), y: z(width) }
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

/// Os experts calculados na CPU pela linha de serviço.
pub trait HostExperts: Send + 'static {
    type Error: Debug + Send + 'static;
    /// `y = Σ expert(key, x)` sobre `keys` (as faltas da camada `layer`, na ordem do top-k).
    fn compute(&mut self, layer: usize, keys: &[u32], x: &[u32], y: &mut [u32]) -> Result<(), Self::Error>;
}
