//! A1.1: o protocolo D1 do motor sem parada por camada, genérico sobre um [`Backend`] (plano
//! `docs/PLANO-runtime-rust-moe-20261029.md`, seção A0b; registro em "A1.1 — o protocolo sem GPU").
//!
//! - Residência fixa por token: a tabela do token ([`TokenEpoch`]) só muda na virada.
//! - Duas linhas no host, com posse separada. A principal ([`Engine`]) é dona da stream de cálculo e só vê um
//!   [`Published`] por token. A de serviço ([`Service`], numa thread própria) é dona do cache, da stream de cópia e do
//!   `TokenEpoch`.
//! - O doorbell em memória mapeada: a GPU grava `seq[l] = t` e espera `flag[l] >= t` (`GEQ` cíclico); `seq[L]` marca o
//!   fim do token. O host responde a camada ([`answer`]) e só depois decide as admissões (`lfu_cpu`, a mesma
//!   [`Freq`](crate::cache::Freq) do `ExpertCache`).
//! - A cópia de uma vítima espera o último leitor pelo doorbell ([`Service::copy_after_reader`]).
//!
//! O `ExpertCache` de hoje (`crate::cache`) não muda. No A1.1 não há GPU: o backend é o falso ([`fake`]), com `proptest`
//! (`tests.rs`) e `loom` (`loom_tests.rs`, com `--cfg motor_loom`).
#![deny(unsafe_code)]

pub mod backend;
pub mod engine;
pub mod fake;
pub mod leitura;
pub mod line;
pub mod mem;
pub mod sync;

#[cfg(all(test, motor_loom))]
mod loom_tests;
#[cfg(all(test, not(motor_loom)))]
pub(crate) mod tests;
#[cfg(all(test, not(motor_loom)))]
mod passo_tests;
#[cfg(all(test, not(motor_loom)))]
mod leitura_tests;
#[cfg(all(test, not(motor_loom)))]
mod sequencia_tests;

pub use backend::{geq, release_value, Backend, CpuGroup, CpuStep, HostExperts, LayerIo, LayerOffsets, Role};
pub use engine::{Engine, Parts, Published};
pub use line::{answer, LayerView, Pins, Served, Service, TokenEpoch, Victim};
pub use mem::{Device, Mapped, Pinned};

/// Chave sem slot na tabela do token.
pub const NO_SLOT: u32 = u32::MAX;

/// Quando uma admissão vira residente.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitMode {
    /// Na virada do token, o slot cujo `ready` já terminou na consulta; o backlog conta as cópias ainda não terminadas.
    Event,
    /// Determinístico (passo 7 do A0b): a linha de serviço sincroniza a stream de cópia antes do `settle` da virada, e o
    /// backlog conta todas as cópias do token. Acertos, cópias e experts na CPU batem com `tools/moe_sim.py` no protocolo
    /// `token` com barreira.
    TokenBarrier,
}

#[derive(Clone, Debug)]
pub struct Config {
    pub layers: usize,
    pub experts: usize,
    pub topk: usize,
    /// Largura de `x` e `y` na área mapeada da camada.
    pub width: usize,
    pub slots: usize,
    /// Cópias de admissão a caminho, no máximo (o simulador usa 4).
    pub backlog: usize,
    pub admit: AdmitMode,
    /// O primeiro token (o contador começa em 1; os testes começam perto de 2^31 e de 2^32 para o `GEQ` cíclico).
    pub first_token: u32,
    /// Chaves copiadas antes do primeiro token (o simulador começa vazio).
    pub prefill: Vec<u32>,
    /// O número máximo de posições de um passo (1 = o motor por token de sempre). Define o tamanho das áreas das camadas.
    pub max_positions: usize,
    #[cfg(test)]
    pub(crate) mutant: Option<Mutant>,
}

impl Config {
    #[allow(unused_variables)]
    pub(crate) fn is(&self, m: Mutant) -> bool {
        #[cfg(test)]
        return self.mutant == Some(m);
        #[cfg(not(test))]
        false
    }

    pub(crate) fn key(&self, layer: usize, expert: u32) -> u32 {
        (layer * self.experts) as u32 + expert
    }

    pub(crate) fn layer_of(&self, key: u32) -> usize {
        key as usize / self.experts
    }
}

/// Contadores da linha de serviço, nos nomes do simulador entre parênteses.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Stats {
    /// Experts pedidos (`needed`).
    pub needed: u64,
    /// Na tabela do token (`hits`).
    pub hits: u64,
    /// Calculados na CPU (`cpu`).
    pub cpu: u64,
    /// Admissões com cópia (`copies_bg`).
    pub copies: u64,
    /// Admissões recusadas: backlog cheio, orçamento zero ou vítima com pontuação maior ou igual (`dropped`).
    pub dropped: u64,
    /// Cópias guardadas até o doorbell da camada seguinte à da vítima (`copies_deferred`).
    pub deferred: u64,
}

#[derive(Debug)]
pub enum Error<E> {
    Backend(E),
    Cpu(String),
    /// O cancelamento foi marcado (o `Drop`, uma falha da outra linha).
    Cancelled,
    /// A linha de serviço falhou (erro ou `panic`); a guarda dela soltou as esperas.
    ServiceFailed,
    /// Um `Published` de outra geração.
    WrongGeneration { expected: u32, got: u32 },
    /// A tabela recusou um slot que não está residente (teste de estado (a) do A0b).
    LoadingInTable { slot: usize },
    Config(&'static str),
}

/// Mutantes dos testes (o idioma do B0: cada um tem um teste que o acusa; tabela no plano, seção A1.1).
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mutant {
    /// `loom`: a flag escrita antes da saída da CPU.
    FlagBeforeY,
    /// `loom`: `seq` lida com `Relaxed`.
    SeqRelaxed,
    /// `loom`: a cópia guardada sai ao ver `seq[l']`, e não `seq[l'+1]` (antes do `seq` da camada seguinte à da vítima).
    HeldOneLayerEarly,
    /// Backend falso: a cópia enfileirada na escolha da vítima.
    CopyAtVictim,
    /// Backend falso: a tabela sem as duas proteções do `ready` (a consulta no `settle` e a espera da stream de cálculo).
    NoReadyProtection,
    /// Backend falso: o `Published` mandado antes de gravar o `ready` das cópias soltas na virada.
    PublishedBeforeRecord,
    /// Backend falso: a consulta do `ready` (o `settle`) antes da gravação dele.
    QueryBeforeRecord,
    /// Backend falso: a ordem antiga do `Drop` (sincronizar as streams antes do `join`, sem a drenagem pela linha de
    /// serviço).
    OldDropOrder,
}
