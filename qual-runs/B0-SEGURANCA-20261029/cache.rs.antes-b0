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
//! O cache guarda um buffer pinned por expert (gate_up seguido de down), que serve às cópias e ao caminho de CPU. Não é a
//! única cópia do processo: o `HostModel` de onde os buffers saem continua com os seus pesos enquanto quem chamou o mantiver
//! vivo (os binários do M2a o mantêm, então os experts ocupam a RAM duas vezes).
use crate::block::Routing;
use crate::cpu::{Cpu, ExpertRef};
use crate::gpu::{enqueue_expert, kernels, ExpertGen, AXPBY_B, GEMV_BM, SWIGLU_BM};
use crate::model::HostModel;
use cuda_core::{CudaContext, Event, PinnedHostBuffer, Stream};
use crate::gpu::sync_on_err;
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::{Partition, Reshape};
use std::collections::VecDeque;
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

struct Slot {
    gate_up: Tensor<bf16>,
    down: Tensor<bf16>,
    key: Option<usize>,
    last_use: u64,
    /// Cópia de admissão ainda não confirmada como terminada.
    loading: bool,
    ready: Event,
}

struct Scratch {
    logits: Partition<Tensor<f32>>,
    a: Partition<Tensor<f32>>,
    y: Partition<Tensor<f32>>,
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
    store: Vec<PinnedHostBuffer<u16>>,
    slots: Vec<Slot>,
    /// chave (camada·E + expert) → slot
    slot_of: Vec<Option<usize>>,
    clock: u64,
    loading: VecDeque<usize>,
    compute: Arc<Stream>,
    copy: Arc<Stream>,
    order: Event,
    scratch: Option<Scratch>,
    /// Tensores com a saída da CPU enviada à GPU; vivem até a próxima sincronização da stream de cálculo.
    keep: Vec<Tensor<f32>>,
    cpu: Cpu,
    _ctx: Arc<CudaContext>,
}

impl ExpertCache {
    pub fn new(m: &HostModel, compute: Arc<Stream>, cfg: CacheConfig) -> Result<Self, Error> {
        let c = &m.cfg;
        let (h, i, e) = (c.hidden, c.inter, c.experts);
        let total = c.layers * e;
        assert!(cfg.slots <= total, "mais slots ({}) que experts ({total})", cfg.slots);
        let ok = h % GEMV_BM == 0 && i % GEMV_BM == 0 && e % GEMV_BM == 0 && h % AXPBY_B == 0;
        assert!(ok, "formas fora dos tamanhos de bloco: H={h} I={i} E={e}");
        let ctx = CudaContext::new(compute.device().ordinal())?;
        let device = compute.device().clone();
        let s = &compute;

        let mut routers = Vec::with_capacity(c.layers);
        let mut store = Vec::with_capacity(total);
        for l in &m.layers {
            routers.push(api::copy_host_vec_to_device(&Arc::new(l.moe.router.clone())).sync_on(s)?.reshape(&[e, h])?);
            for x in 0..e {
                let bits: Vec<u16> = l.moe.gate_up[x].iter().chain(&l.moe.down[x]).map(|b| b.to_bits()).collect();
                store.push(PinnedHostBuffer::from_slice(&ctx, &bits)?);
            }
        }
        let mut slots = Vec::with_capacity(cfg.slots);
        for _ in 0..cfg.slots {
            slots.push(Slot {
                gate_up: api::zeros::<bf16>(&[2 * i * h]).sync_on(s)?.reshape(&[2 * i, h])?,
                down: api::zeros::<bf16>(&[h * i]).sync_on(s)?.reshape(&[h, i])?,
                key: None,
                last_use: 0,
                loading: false,
                ready: device.new_event()?,
            });
        }
        let z = |n: usize, b: usize| -> Result<Partition<Tensor<f32>>, Error> { Ok(api::zeros::<f32>(&[n]).sync_on(s)?.partition([b])) };
        let scratch = Scratch { logits: z(e, GEMV_BM)?, a: z(i, SWIGLU_BM)?, y: z(h, GEMV_BM)? };
        Ok(Self {
            hidden: h,
            inter: i,
            k: c.k,
            experts: e,
            layers: c.layers,
            stats: CacheStats::default(),
            routers,
            store,
            slots,
            slot_of: vec![None; total],
            clock: 0,
            loading: VecDeque::new(),
            copy: device.new_stream()?,
            order: device.new_event()?,
            compute,
            scratch: Some(scratch),
            keep: Vec::new(),
            cpu: Cpu::new(cfg.cpu_threads),
            cfg,
            _ctx: ctx,
        })
    }

    /// Bytes de um expert.
    pub fn expert_bytes(&self) -> usize {
        3 * self.inter * self.hidden * 2
    }

    /// Slots com expert válido (sem contar cópias em andamento).
    pub fn resident(&self) -> usize {
        self.slots.iter().filter(|s| s.key.is_some() && !s.loading).count()
    }

    pub fn loading_now(&self) -> usize {
        self.loading.len()
    }

    /// Esvazia o cache (para começar uma medição do zero). Exige a stream de cópia parada.
    pub fn clear(&mut self) -> Result<(), Error> {
        unsafe { self.copy.synchronize() }?;
        self.loading.clear();
        for s in &mut self.slots {
            s.key = None;
            s.loading = false;
            s.last_use = 0;
        }
        self.slot_of.iter_mut().for_each(|x| *x = None);
        Ok(())
    }

    /// Põe nos slots livres os experts que ainda não estão no cache, em ordem de chave, sem despejar ninguém; volta com a
    /// stream de cálculo sincronizada e o número de experts copiados. Com orçamento para todos, o cache fica completo.
    pub fn fill(&mut self) -> Result<usize, Error> {
        unsafe { self.copy.synchronize() }?;
        self.settle()?;
        let mut copied = 0;
        let cs = self.compute.clone();
        for key in 0..self.slot_of.len() {
            if self.slot_of[key].is_some() {
                continue;
            }
            let Some(sl) = self.slots.iter().position(|s| s.key.is_none()) else { break };
            // SAFETY: o slot está livre, então nenhuma leitura pendente dele; a cópia vai na stream de cálculo.
            unsafe { self.copy_into(sl, key, &cs)? };
            copied += 1;
        }
        unsafe { cs.synchronize() }?;
        Ok(copied)
    }

    fn weights(&self, key: usize) -> (&[bf16], &[bf16]) {
        let buf = self.store[key].as_slice();
        // SAFETY: `bf16` é `repr(transparent)` sobre `u16`; mesma largura e alinhamento.
        let all: &[bf16] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const bf16, buf.len()) };
        all.split_at(2 * self.inter * self.hidden)
    }

    /// Cópias de admissão já terminadas viram slots válidos.
    fn settle(&mut self) -> Result<(), Error> {
        if self.cfg.admit == AdmitMode::Barrier && !self.loading.is_empty() {
            // SAFETY: só espera a stream de cópia.
            unsafe { self.copy.synchronize() }?;
        }
        while let Some(&s) = self.loading.front() {
            if !self.slots[s].ready.query()? {
                break;
            }
            self.slots[s].loading = false;
            self.loading.pop_front();
        }
        Ok(())
    }

    /// Slot a sobrescrever: um livre, senão o de uso mais antigo que não está carregando nem guarda expert pedido nesta camada
    /// (`add_into` marca todos eles com `last_use = clock` antes de classificar acertos e faltas).
    fn victim(&self) -> Option<usize> {
        if let Some(i) = self.slots.iter().position(|s| s.key.is_none()) {
            return Some(i);
        }
        self.slots.iter().enumerate().filter(|(_, s)| !s.loading && s.last_use < self.clock).min_by_key(|(_, s)| s.last_use).map(|(i, _)| i)
    }

    /// Copia o expert `key` para o slot `s` na stream dada.
    ///
    /// # Safety
    /// Quem chama garante que nenhuma leitura pendente do slot roda depois da cópia (mesma stream, ou espera por evento).
    unsafe fn copy_into(&mut self, s: usize, key: usize, stream: &Arc<Stream>) -> Result<(), Error> {
        // O expert antigo sai do mapa antes das cópias: se uma delas falhar no meio, o slot fica vazio, nunca meio sobrescrito
        // com o nome do expert antigo ou do novo.
        if let Some(old) = self.slots[s].key.take() {
            self.slot_of[old] = None;
        }
        let gu = 2 * self.inter * self.hidden;
        let src = self.store[key].as_ptr();
        let slot = &mut self.slots[s];
        // SAFETY: origem com 3·I·H u16, destinos com 2·I·H e H·I bf16; o buffer pinned vive enquanto o cache existir (o `Drop`
        // sincroniza a stream de cópia antes de soltá-lo).
        unsafe {
            cuda_core::memcpy_htod_async(slot.gate_up.device_pointer().cu_deviceptr(), src, gu, stream)?;
            cuda_core::memcpy_htod_async(slot.down.device_pointer().cu_deviceptr(), src.add(gu), gu / 2, stream)?;
        }
        slot.key = Some(key);
        slot.last_use = self.clock;
        self.slot_of[key] = Some(s);
        Ok(())
    }

    /// `h ← h + resid·moe_l(xd)`. Sincroniza uma vez para ler os logits do roteador (e `xd`, se algum expert for para a
    /// CPU); o resto fica enfileirado na stream de cálculo.
    ///
    /// A ausência de corrida entre cópias e cálculo vem deste protocolo, não dos tipos: a stream de cálculo está parada quando
    /// as vítimas são escolhidas (leitura dos logits), os experts pedidos na camada não podem ser despejados, a admissão em
    /// segundo plano espera um evento gravado no cálculo, e um slot carregando só é lido depois que o evento da cópia termina.
    ///
    /// Depois de um erro, as duas streams são sincronizadas e o cache fica inutilizável (as chamadas seguintes devolvem erro).
    pub fn add_into(&mut self, l: usize, xd: &Tensor<f32>, h: Partition<Tensor<f32>>, resid: f32) -> Result<(Partition<Tensor<f32>>, Routing), Error> {
        let Some(sc) = self.scratch.take() else {
            return Err(tensor_error("cache de experts inutilizável depois de um erro anterior"));
        };
        let out = self.add_into_with(sc, l, xd, h, resid);
        if out.is_err() {
            self.sync_all();
        }
        out
    }

    fn sync_all(&self) {
        // SAFETY: só espera as streams; erros aqui não têm a quem ser devolvidos.
        unsafe {
            let _ = self.compute.synchronize();
            let _ = self.copy.synchronize();
        }
    }

    fn add_into_with(&mut self, mut sc: Scratch, l: usize, xd: &Tensor<f32>, mut h: Partition<Tensor<f32>>, resid: f32) -> Result<(Partition<Tensor<f32>>, Routing), Error> {
        let (hd, i, e) = (self.hidden, self.inter, self.experts);
        let s = self.compute.clone();
        let gen_h = crate::gpu::gemv_generics(hd);

        let t0 = Instant::now();
        // SAFETY: `unchecked_accesses`: E e H são múltiplos de BM e BK (conferido em `new`).
        let (logits, _, _) = unsafe { kernels::gemv(sc.logits, &self.routers[l], xd) }.generics(gen_h.clone()).sync_on(&s).map_err(sync_on_err(&s))?;
        let logits = Arc::new(logits.unpartition());
        let lv = api::copy_device_to_host_vec(&logits).sync_on(&s).map_err(sync_on_err(&s))?;
        sc.logits = Arc::try_unwrap(logits).map_err(|_| "logits ainda referenciados").unwrap().partition([GEMV_BM]);
        self.stats.router_wait_ms += t0.elapsed().as_secs_f64() * 1e3;
        // A stream de cálculo está parada aqui: o que foi enviado da CPU antes já foi consumido.
        self.keep.clear();
        let r = Routing::from_logits(&lv, self.k);

        self.settle().map_err(sync_on_err(&s))?;
        self.clock += 1;
        // Primeira passada: todo expert pedido nesta camada que já está num slot fica protegido do despejo, inclusive os acertos
        // que vêm depois de uma falta na ordem do roteador.
        for &x in &r.experts {
            if let Some(sl) = self.slot_of[l * e + x] {
                self.slots[sl].last_use = self.clock;
            }
        }
        let mut gpu: Vec<(usize, f32)> = Vec::new();
        let mut cpu: Vec<(usize, f32)> = Vec::new();
        for (&x, &g) in r.experts.iter().zip(&r.weights) {
            let key = l * e + x;
            match self.slot_of[key] {
                Some(sl) if !self.slots[sl].loading => {
                    self.slots[sl].last_use = self.clock;
                    self.stats.hits += 1;
                    gpu.push((sl, g));
                }
                Some(_) => {
                    // Admissão ainda em andamento: só acontece em LruCpu.
                    self.stats.cpu += 1;
                    cpu.push((key, g));
                }
                None => match self.cfg.policy {
                    Policy::LruSync => match self.victim() {
                        Some(sl) => {
                            // SAFETY: cópia na stream de cálculo, depois de todas as leituras já enfileiradas do slot.
                            unsafe { self.copy_into(sl, key, &s).map_err(sync_on_err(&s))? };
                            self.stats.sync_copies += 1;
                            gpu.push((sl, g));
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
                        if self.loading.len() < self.cfg.backlog {
                            if let Some(sl) = self.victim() {
                                self.order.record(&s).map_err(sync_on_err(&s))?;
                                self.copy.wait_event(&self.order).map_err(sync_on_err(&s))?;
                                // SAFETY: a stream de cópia espera o evento gravado depois de tudo o que já foi
                                // enfileirado no cálculo, incluindo as leituras do slot despejado; o slot só volta a ser
                                // lido depois que `ready` termina.
                                let cs = self.copy.clone();
                                unsafe { self.copy_into(sl, key, &cs).map_err(sync_on_err(&s))? };
                                let slot = &mut self.slots[sl];
                                slot.loading = true;
                                slot.ready.record(&self.copy).map_err(sync_on_err(&s))?;
                                self.loading.push_back(sl);
                                self.stats.bg_copies += 1;
                            } else {
                                self.stats.bg_dropped += 1;
                            }
                        } else {
                            self.stats.bg_dropped += 1;
                        }
                    }
                },
            }
        }

        // `x` no host para os experts da CPU, lido agora, com a stream de cálculo ainda sem os experts da GPU (em LruCpu ela
        // está parada; em LruSync só há CPU quando faltou slot, e aí a espera inclui as cópias desta camada).
        let mut x = Vec::new();
        if !cpu.is_empty() {
            x = vec![0f32; hd];
            // SAFETY: `xd` tem H f32 e foi escrito antes da sincronização do roteador; `x` vive até o `synchronize`.
            unsafe {
                cuda_core::memcpy_dtoh_async(x.as_mut_ptr(), xd.device_pointer().cu_deviceptr(), hd, &s).map_err(sync_on_err(&s))?;
                s.synchronize().map_err(sync_on_err(&s))?;
            }
        }

        // Experts na GPU: enfileirados sem sincronizar.
        let gen = ExpertGen::new(hd, i);
        let gen_ax = vec![AXPBY_B.to_string()];
        for (n, &(sl, g)) in gpu.iter().enumerate() {
            let beta = if n == 0 { 0.0f32 } else { 1.0 };
            let slot = &self.slots[sl];
            // SAFETY: formas conferidas em `new`; slots, `xd` e o scratch vivem até a próxima sincronização, e o slot não é
            // sobrescrito antes disso (ver `copy_into` e `victim`).
            let (a, y) = unsafe { enqueue_expert(&s, &gen, sc.a, sc.y, &slot.gate_up, &slot.down, xd, g, beta)? };
            (sc.a, sc.y) = (a, y);
        }
        if !gpu.is_empty() {
            let y = sc.y.unpartition();
            // SAFETY: como acima.
            let (h2, _, _, _) = unsafe { kernels::axpby(h, &y, resid, 1.0f32).generics(gen_ax.clone()).async_on(&s).map_err(sync_on_err(&s))? };
            h = h2;
            sc.y = y.partition([GEMV_BM]);
        }

        // Experts na CPU, enquanto a GPU calcula os dela.
        if !cpu.is_empty() {
            let t0 = Instant::now();
            let refs: Vec<ExpertRef> = cpu
                .iter()
                .map(|&(key, g)| {
                    let (gate_up, down) = self.weights(key);
                    ExpertRef { gate_up, down, weight: g }
                })
                .collect();
            let y = self.cpu.experts(&x, &refs, hd, i);
            self.stats.cpu_ms += t0.elapsed().as_secs_f64() * 1e3;
            // SAFETY: a cópia retém o vetor de origem; `yd` fica em `keep` até a próxima sincronização da stream.
            let yd = unsafe { api::copy_host_vec_to_device(&Arc::new(y)).async_on(&s).map_err(sync_on_err(&s))? };
            let (h2, _, _, _) = unsafe { kernels::axpby(h, &yd, resid, 1.0f32).generics(gen_ax.clone()).async_on(&s).map_err(sync_on_err(&s))? };
            h = h2;
            self.keep.push(yd);
        }
        self.scratch = Some(sc);
        Ok((h, r))
    }
}

impl Drop for ExpertCache {
    fn drop(&mut self) {
        // Antes de os campos caírem: cópias em segundo plano leem `store` e escrevem `slots`, e kernels pendentes leem os slots.
        // O cutile solta a memória do device numa stream de desalocação própria, que não espera estas.
        self.sync_all();
    }
}
