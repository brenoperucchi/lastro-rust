//! M2a: cache de experts na VRAM com orçamento de slots, origem pinned na RAM e despejo LRU.
//!
//! Os slots são globais, endereçados pelo par (camada, expert), como no simulador (`tools/moe_sim.py`). Duas políticas:
//! - [`Policy::LruSync`]: toda falta é copiada antes de calcular, na própria stream de cálculo. A ordem da stream garante que
//!   a cópia só sobrescreve o slot despejado depois das leituras pendentes dele.
//! - [`Policy::LruCpu`]: a falta é calculada na CPU enquanto os experts residentes rodam na GPU; o expert é admitido em
//!   segundo plano por outra stream, que antes espera tudo o que já foi enfileirado no cálculo (então também as leituras do
//!   slot despejado). O slot só vale depois que o evento da cópia termina; até lá o expert continua indo para a CPU. Se já
//!   houver `backlog` cópias na fila, a admissão é descartada.
//!
//! O protocolo de segurança mora em [`core`] (B0), o único módulo com `unsafe`; este arquivo é a política e os contadores.
//!
//! O cache guarda um buffer pinned por expert (gate_up seguido de down), que serve às cópias e ao caminho de CPU. Não é a
//! única cópia do processo: o `HostModel` de onde os buffers saem continua com os seus pesos enquanto quem chamou o mantiver
//! vivo (os binários do M2a o mantêm, então os experts ocupam a RAM duas vezes).
#![deny(unsafe_code)]

#[allow(unsafe_code)]
pub mod core;

use self::core::{Epoch, Lookup, SlotId, Slots};
use crate::block::Routing;
use crate::cpu::{Cpu, ExpertRef};
use crate::gpu::{AXPBY_B, GEMV_BM};
use crate::model::HostModel;
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::{Partition, Reshape};
use cuda_core::Stream;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Policy {
    LruSync,
    LruCpu,
}

impl Policy {
    pub fn name(self) -> &'static str {
        match self {
            Policy::LruSync => "lru_sync",
            Policy::LruCpu => "lru_cpu",
        }
    }
}

/// Quando uma admissão em segundo plano (`LruCpu`) vira slot válido.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdmitMode {
    /// Quando o evento da cópia já terminou no começo de uma camada (`ready.query()`): o número de admissões e de acertos
    /// depende do tempo e varia entre execuções (M2f, terceira medida).
    Event,
    /// Só para teste e reprodução: a stream de cópia é sincronizada no começo de cada camada, então toda cópia enfileirada
    /// numa camada vale a partir da seguinte. Acertos, cópias e logits deixam de depender do tempo; o host espera as cópias.
    Barrier,
}

#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub slots: usize,
    pub policy: Policy,
    /// Cópias de admissão em segundo plano na fila, no máximo (o simulador usa 4).
    pub backlog: usize,
    pub cpu_threads: usize,
    pub admit: AdmitMode,
}

/// Contadores acumulados desde a criação (ou desde `reset_stats`).
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub hits: u64,
    pub sync_copies: u64,
    pub cpu: u64,
    pub bg_copies: u64,
    pub bg_dropped: u64,
    /// Tempo de parede do host esperando a leitura dos logits do roteador (inclui o cálculo da GPU até ali).
    pub router_wait_ms: f64,
    /// Tempo de parede dos experts na CPU.
    pub cpu_ms: f64,
}

pub struct ExpertCache {
    pub hidden: usize,
    pub inter: usize,
    pub k: usize,
    pub experts: usize,
    pub layers: usize,
    pub cfg: CacheConfig,
    pub stats: CacheStats,
    routers: Vec<Tensor<bf16>>,
    epoch: Epoch,
    slots: Slots,
    cpu: Cpu,
    poisoned: bool,
}

impl ExpertCache {
    pub fn new(m: &HostModel, compute: Arc<Stream>, cfg: CacheConfig) -> Result<Self, Error> {
        let c = &m.cfg;
        let (h, i, e) = (c.hidden, c.inter, c.experts);
        let total = c.layers * e;
        assert!(cfg.slots <= total, "mais slots ({}) que experts ({total})", cfg.slots);
        let ok = h % GEMV_BM == 0 && i % GEMV_BM == 0 && e % GEMV_BM == 0 && h % AXPBY_B == 0;
        assert!(ok, "formas fora dos tamanhos de bloco: H={h} I={i} E={e}");
        let s = &compute;
        let mut routers = Vec::with_capacity(c.layers);
        let mut experts: Vec<(&[bf16], &[bf16])> = Vec::with_capacity(total);
        for l in &m.layers {
            routers.push(api::copy_host_vec_to_device(&Arc::new(l.moe.router.clone())).sync_on(s)?.reshape(&[e, h])?);
            for x in 0..e {
                experts.push((&l.moe.gate_up[x], &l.moe.down[x]));
            }
        }
        let (slots, epoch) = self::core::new_pair(&experts, h, i, e, cfg.slots, compute)?;
        Ok(Self {
            hidden: h,
            inter: i,
            k: c.k,
            experts: e,
            layers: c.layers,
            stats: CacheStats::default(),
            routers,
            epoch,
            slots,
            cpu: Cpu::new(cfg.cpu_threads),
            cfg,
            poisoned: false,
        })
    }

    /// Bytes de um expert.
    pub fn expert_bytes(&self) -> usize {
        3 * self.inter * self.hidden * 2
    }

    /// Slots com expert válido (sem contar cópias em andamento).
    pub fn resident(&self) -> usize {
        self.slots.resident_count()
    }

    pub fn loading_now(&self) -> usize {
        self.slots.loading_count()
    }

    /// Esvazia o cache (para começar uma medição do zero).
    pub fn clear(&mut self) -> Result<(), Error> {
        self.slots.clear()
    }

    /// Põe nos slots livres os experts que ainda não estão no cache, em ordem de chave, sem despejar ninguém; volta com a
    /// stream de cálculo sincronizada e o número de experts copiados. Com orçamento para todos, o cache fica completo.
    pub fn fill(&mut self) -> Result<usize, Error> {
        self.slots.settle(true)?;
        let mut copied = 0;
        for key in 0..self.layers * self.experts {
            if self.slots.lookup(key) != Lookup::Miss {
                continue;
            }
            let Some(v) = self.slots.free_slot() else { break };
            self.slots.copy_on_compute(v, key)?;
            copied += 1;
        }
        self.slots.sync_all();
        Ok(copied)
    }

    /// `h ← h + resid·moe_l(xd)`. Sincroniza uma vez para ler os logits do roteador (e `xd`, se algum expert for para a
    /// CPU); o resto fica enfileirado na stream de cálculo.
    ///
    /// A ausência de corrida entre cópias e cálculo vem do protocolo de [`core`], agora em tipos: a vítima só é escolhida
    /// com a stream de cálculo parada (`Idle`) e fora dos experts pedidos na camada (`Pins`), a cópia em segundo plano grava
    /// o próprio evento de ordem, e um slot só é lido como `Resident`.
    ///
    /// Depois de um erro, as duas streams são sincronizadas e o cache fica inutilizável (as chamadas seguintes devolvem erro).
    pub fn add_into(&mut self, l: usize, xd: &Tensor<f32>, h: Partition<Tensor<f32>>, resid: f32) -> Result<(Partition<Tensor<f32>>, Routing), Error> {
        if self.poisoned {
            return Err(tensor_error("cache de experts inutilizável depois de um erro anterior"));
        }
        let out = self.add_into_with(l, xd, h, resid);
        if out.is_err() {
            self.poisoned = true;
            self.slots.sync_all();
        }
        out
    }

    fn add_into_with(&mut self, l: usize, xd: &Tensor<f32>, mut h: Partition<Tensor<f32>>, resid: f32) -> Result<(Partition<Tensor<f32>>, Routing), Error> {
        let (hd, i, e) = (self.hidden, self.inter, self.experts);
        let t0 = Instant::now();
        let (lv, mut idle) = self.epoch.router_logits(&self.routers[l], xd)?;
        self.stats.router_wait_ms += t0.elapsed().as_secs_f64() * 1e3;
        // A stream de cálculo está parada aqui: o que foi enviado da CPU antes já foi consumido.
        idle.release_uploads();
        let r = Routing::from_logits(&lv, self.k);

        self.slots.settle(self.cfg.admit == AdmitMode::Barrier)?;
        // Todo expert pedido nesta camada que já está num slot fica protegido do despejo, inclusive os acertos que vêm depois
        // de uma falta na ordem do roteador.
        let keys: Vec<usize> = r.experts.iter().map(|&x| l * e + x).collect();
        let pins = self.slots.pin_layer(&keys);
        let mut gpu: Vec<(SlotId, f32)> = Vec::new();
        let mut cpu: Vec<(usize, f32)> = Vec::new();
        for (&key, &g) in keys.iter().zip(&r.weights) {
            match self.slots.lookup(key) {
                Lookup::Hit(id) => {
                    self.stats.hits += 1;
                    gpu.push((id, g));
                }
                Lookup::Loading => {
                    // Admissão ainda em andamento: só acontece em LruCpu.
                    self.stats.cpu += 1;
                    cpu.push((key, g));
                }
                Lookup::Miss => match self.cfg.policy {
                    Policy::LruSync => match self.slots.victim(&pins, &idle)? {
                        Some(v) => {
                            let id = self.slots.copy_on_compute(v, key)?;
                            self.stats.sync_copies += 1;
                            gpu.push((id, g));
                        }
                        None => {
                            // Orçamento menor que k (ou zero): sem slot livre nesta camada, calcula na CPU.
                            self.stats.cpu += 1;
                            cpu.push((key, g));
                        }
                    },
                    Policy::LruCpu => {
                        self.stats.cpu += 1;
                        cpu.push((key, g));
                        let admitted = if self.slots.loading_count() < self.cfg.backlog {
                            match self.slots.victim(&pins, &idle)? {
                                Some(v) => {
                                    self.slots.copy_in_background(v, key)?;
                                    true
                                }
                                None => false,
                            }
                        } else {
                            false
                        };
                        if admitted {
                            self.stats.bg_copies += 1;
                        } else {
                            self.stats.bg_dropped += 1;
                        }
                    }
                },
            }
        }

        // `x` no host para os experts da CPU, lido agora, com a stream de cálculo ainda sem os experts da GPU (em LruCpu ela
        // está parada; em LruSync só há CPU quando faltou slot, e aí a espera inclui as cópias desta camada).
        let x = if cpu.is_empty() { Vec::new() } else { idle.read_x(xd)? };
        drop(idle);

        // Experts na GPU: enfileirados sem sincronizar.
        for &(id, g) in &gpu {
            let slot = self.slots.resident(id)?;
            self.epoch.launch_expert(slot, xd, g)?;
        }
        h = self.epoch.finish_gpu(h, resid)?;

        // Experts na CPU, enquanto a GPU calcula os dela.
        if !cpu.is_empty() {
            let t0 = Instant::now();
            let refs: Vec<ExpertRef> = cpu
                .iter()
                .map(|&(key, g)| {
                    let (gate_up, down) = self.slots.weights(key);
                    ExpertRef { gate_up, down, weight: g }
                })
                .collect();
            let y = self.cpu.experts(&x, &refs, hd, i);
            self.stats.cpu_ms += t0.elapsed().as_secs_f64() * 1e3;
            h = self.epoch.add_cpu_out(h, y, resid)?;
        }
        Ok((h, r))
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        // Antes de os campos caírem (os roteadores vêm antes de `epoch` e `slots` na declaração).
        self.slots.sync_all();
    }
}
