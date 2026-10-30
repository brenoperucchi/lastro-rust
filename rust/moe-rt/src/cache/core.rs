//! B0: o núcleo do cache, onde fica todo o `unsafe` (o resto de `cache` tem `#![deny(unsafe_code)]`).
//!
//! O protocolo de segurança do cache (seção "B0 — desenho revisado" do plano) passa a ter tipos:
//! - um slot está `Free`, `Reserved` (escolhido como vítima, à espera da cópia), `Loading` (cópia em segundo plano ainda não
//!   confirmada) ou `Resident`; o mapa chave → slot só lista slots `Resident` (como o `host_res` do Strata, que só recebe o
//!   slot novo depois do evento da cópia);
//! - um kernel só lê um slot por um [`ResidentSlot`], que só sai de [`Slots::resident`] para um slot `Resident` da mesma
//!   geração do [`SlotId`] (conferido em tempo de execução);
//! - escolher vítima pede [`Pins`] e [`Idle`]. O `Pins` só sai de [`Slots::pin_layer`] (prova que a camada foi fixada, não
//!   que a lista de chaves passada estava completa; quem garante que um acerto não foi despejado é a geração conferida em
//!   [`Slots::resident`]). O `Idle` prova que nada foi lançado desde a leitura dos logits (cópias na própria stream de
//!   cálculo podem estar na fila). A vítima é um [`Victim`], que a cópia consome;
//! - a cópia grava o evento de ordem ela mesma ([`Slots::copy_in_background`]), então nenhum lançamento fica entre o evento e
//!   a cópia;
//! - os uploads da saída da CPU só são soltos por [`Idle::release_uploads`], e o [`Idle`] só existe entre a sincronização
//!   do roteador e o primeiro lançamento da camada: ele toma a [`Epoch`] emprestada, e lançar pede `&mut Epoch`;
//! - `Slots` e `Epoch` só nascem juntos ([`new_pair`]), na mesma stream de cálculo, com um id de instância. `Pins`, `Idle`,
//!   `SlotId`, `Victim` e `ResidentSlot` levam esse id, conferido em tempo de execução: os de outra instância dão erro.
use crate::gpu::{enqueue_expert, kernels, sync_on_err, ExpertGen, AXPBY_B, GEMV_BM, SWIGLU_BM};
use cuda_core::{CudaContext, Event, PinnedHostBuffer, Stream};
use cutile::error::tensor_error;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::{Partition, Reshape};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Um slot, com a geração em que foi visto `Resident`. Não serve para lançar: [`Slots::resident`] confere estado e geração.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotId {
    index: usize,
    gen: u64,
    owner: u64,
}

/// Os experts pedidos numa camada, marcados para não serem despejados nela. Só [`Slots::pin_layer`] cria um.
#[derive(Debug)]
pub struct Pins {
    clock: u64,
    owner: u64,
}

/// Um slot escolhido para receber uma cópia. Não é `Clone`: a cópia o consome. Se for largado sem cópia, o slot fica
/// reservado até o próximo [`Slots::clear`].
#[must_use = "uma vítima largada sem cópia deixa o slot reservado até o próximo clear"]
#[derive(Debug)]
pub struct Victim {
    index: usize,
    gen: u64,
    owner: u64,
}

/// Uma cópia em segundo plano já enfileirada. O slot só vira `Resident` em [`Slots::settle`], depois do evento da cópia.
#[derive(Debug)]
pub struct LoadingSlot {
    _index: usize,
}

/// Empréstimo dos pesos de um slot `Resident` para um lançamento.
pub struct ResidentSlot<'a> {
    gate_up: &'a Tensor<bf16>,
    down: &'a Tensor<bf16>,
    owner: u64,
}

/// O que o mapa diz de uma chave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lookup {
    Hit(SlotId),
    /// Cópia em segundo plano a caminho: vai para a CPU, sem nova admissão.
    Loading,
    Miss,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Free,
    Reserved,
    /// `last_use` do momento da cópia, como antes do B0 (o LRU não muda com o B0).
    Loading { key: usize, last_use: u64 },
    Resident { key: usize, last_use: u64 },
}

struct Slot {
    gate_up: Tensor<bf16>,
    down: Tensor<bf16>,
    state: State,
    gen: u64,
    ready: Event,
}

static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

/// Cria os slots e a `Epoch` na mesma stream de cálculo, ligados por um id de instância. É o único jeito de criar os dois:
/// a cópia na stream de cálculo só fica ordenada depois das leituras se os lançamentos forem na mesma stream.
/// `experts[key] = (gate_up, down)` de cada expert, na ordem das chaves; `router_rows` é o número de linhas do roteador.
pub fn new_pair(
    experts: &[(&[bf16], &[bf16])],
    hidden: usize,
    inter: usize,
    router_rows: usize,
    n_slots: usize,
    compute: Arc<Stream>,
) -> Result<(Slots, Epoch), Error> {
    let owner = NEXT_OWNER.fetch_add(1, Ordering::Relaxed);
    let slots = Slots::new(experts, hidden, inter, n_slots, compute.clone(), owner)?;
    let epoch = Epoch::new(hidden, inter, router_rows, compute, owner)?;
    Ok((slots, epoch))
}

fn foreign() -> Error {
    tensor_error("objeto de outra instância do cache")
}

/// Os slots, a origem pinned dos experts e as duas streams por onde as cópias passam.
pub struct Slots {
    owner: u64,
    hidden: usize,
    inter: usize,
    store: Vec<PinnedHostBuffer<u16>>,
    slots: Vec<Slot>,
    /// chave (camada·E + expert) → slot `Resident`
    slot_of: Vec<Option<usize>>,
    /// chave com cópia em segundo plano a caminho
    pending_key: Vec<bool>,
    /// slots `Loading`, na ordem das cópias
    pending: VecDeque<usize>,
    clock: u64,
    compute: Arc<Stream>,
    copy: Arc<Stream>,
    order: Event,
    _ctx: Arc<CudaContext>,
}

impl Slots {
    fn new(experts: &[(&[bf16], &[bf16])], hidden: usize, inter: usize, n_slots: usize, compute: Arc<Stream>, owner: u64) -> Result<Self, Error> {
        let (h, i) = (hidden, inter);
        let ctx = CudaContext::new(compute.device().ordinal())?;
        let device = compute.device().clone();
        let mut store = Vec::with_capacity(experts.len());
        for (gu, dn) in experts {
            assert!(gu.len() == 2 * i * h && dn.len() == h * i, "expert com forma errada");
            let bits: Vec<u16> = gu.iter().chain(dn.iter()).map(|b| b.to_bits()).collect();
            store.push(PinnedHostBuffer::from_slice(&ctx, &bits)?);
        }
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(Slot {
                gate_up: api::zeros::<bf16>(&[2 * i * h]).sync_on(&compute)?.reshape(&[2 * i, h])?,
                down: api::zeros::<bf16>(&[h * i]).sync_on(&compute)?.reshape(&[h, i])?,
                state: State::Free,
                gen: 0,
                ready: device.new_event()?,
            });
        }
        let total = experts.len();
        Ok(Self {
            owner,
            hidden: h,
            inter: i,
            store,
            slots,
            slot_of: vec![None; total],
            pending_key: vec![false; total],
            pending: VecDeque::new(),
            clock: 0,
            copy: device.new_stream()?,
            order: device.new_event()?,
            compute,
            _ctx: ctx,
        })
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn resident_count(&self) -> usize {
        self.slots.iter().filter(|s| matches!(s.state, State::Resident { .. })).count()
    }

    pub fn loading_count(&self) -> usize {
        self.pending.len()
    }

    /// Só para testes: fecha a stream de cópia com um callback de host até o `send` no transmissor devolvido (ou 10 s). As
    /// admissões enfileiradas depois ficam `Loading`. Nada pode sincronizar a stream de cópia antes do `send`.
    pub fn hold_copies(&self) -> Result<std::sync::mpsc::Sender<()>, Error> {
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        self._ctx.bind_to_thread()?;
        // SAFETY: contexto corrente (acima); o callback não chama CUDA.
        unsafe {
            self.copy.launch_host_function(move || {
                let _ = rx.recv_timeout(std::time::Duration::from_secs(10));
            })
        }?;
        Ok(tx)
    }

    /// Pesos de um expert na origem pinned (para o caminho de CPU).
    pub fn weights(&self, key: usize) -> (&[bf16], &[bf16]) {
        let buf = self.store[key].as_slice();
        // SAFETY: `bf16` é `repr(transparent)` sobre `u16`; mesma largura e alinhamento.
        let all: &[bf16] = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const bf16, buf.len()) };
        all.split_at(2 * self.inter * self.hidden)
    }

    /// Esvazia o cache. Espera a stream de cópia; as leituras ainda enfileiradas no cálculo não importam, porque toda cópia
    /// seguinte vai depois delas (na mesma stream, ou atrás do evento de ordem).
    pub fn clear(&mut self) -> Result<(), Error> {
        // SAFETY: só espera a stream de cópia.
        unsafe { self.copy.synchronize() }?;
        self.pending.clear();
        for s in &mut self.slots {
            s.state = State::Free;
            s.gen += 1;
        }
        self.slot_of.iter_mut().for_each(|x| *x = None);
        self.pending_key.iter_mut().for_each(|x| *x = false);
        Ok(())
    }

    /// Cópias em segundo plano já terminadas viram `Resident`. Com `barrier`, espera todas antes (admissão sem tempo).
    pub fn settle(&mut self, barrier: bool) -> Result<(), Error> {
        if barrier && !self.pending.is_empty() {
            // SAFETY: só espera a stream de cópia.
            unsafe { self.copy.synchronize() }?;
        }
        while let Some(&s) = self.pending.front() {
            if !self.slots[s].ready.query()? {
                break;
            }
            self.pending.pop_front();
            let slot = &mut self.slots[s];
            let State::Loading { key, last_use } = slot.state else { unreachable!("slot pendente fora de Loading") };
            slot.state = State::Resident { key, last_use };
            slot.gen += 1;
            self.slot_of[key] = Some(s);
            self.pending_key[key] = false;
        }
        Ok(())
    }

    /// Começa uma camada: os experts pedidos (`keys`) que estão `Resident` ficam protegidos do despejo nela. Os que ainda estão
    /// `Loading` também ganham o uso desta camada (como antes do B0, quando o slot carregando já estava no mapa), o que conta
    /// para o LRU depois do `settle`.
    pub fn pin_layer(&mut self, keys: &[usize]) -> Pins {
        self.clock += 1;
        for &k in keys {
            if let Some(s) = self.slot_of[k] {
                if let State::Resident { last_use, .. } = &mut self.slots[s].state {
                    *last_use = self.clock;
                }
            }
        }
        for &s in &self.pending {
            if let State::Loading { key, last_use } = &mut self.slots[s].state {
                if keys.contains(key) {
                    *last_use = self.clock;
                }
            }
        }
        Pins { clock: self.clock, owner: self.owner }
    }

    pub fn lookup(&self, key: usize) -> Lookup {
        match self.slot_of[key] {
            Some(s) => Lookup::Hit(SlotId { index: s, gen: self.slots[s].gen, owner: self.owner }),
            None if self.pending_key[key] => Lookup::Loading,
            None => Lookup::Miss,
        }
    }

    /// Slot a sobrescrever: um livre, senão o `Resident` de uso mais antigo que não guarda expert pedido nesta camada. O slot
    /// sai do mapa na hora (o expert antigo passa a ser falta) e fica `Reserved` até a cópia.
    pub fn victim(&mut self, pins: &Pins, idle: &Idle<'_>) -> Result<Option<Victim>, Error> {
        self.check_pins(pins, idle)?;
        let free = self.slots.iter().position(|s| s.state == State::Free);
        let pick = free.or_else(|| {
            self.slots
                .iter()
                .enumerate()
                .filter_map(|(i, s)| match s.state {
                    State::Resident { last_use, .. } if last_use < self.clock => Some((i, last_use)),
                    _ => None,
                })
                .min_by_key(|&(_, u)| u)
                .map(|(i, _)| i)
        });
        Ok(pick.map(|i| self.reserve(i)))
    }

    /// Como [`victim`](Self::victim), para as políticas que pontuam os experts: um slot livre, senão o `Resident` de menor
    /// `prio(chave)` fora dos pinos (empate pelo índice do slot), e esse só se a prioridade for menor que `ceiling`. Recusar
    /// não muda estado nenhum.
    pub fn victim_min_by(&mut self, pins: &Pins, idle: &Idle<'_>, prio: impl Fn(usize) -> f64, ceiling: f64) -> Result<Option<Victim>, Error> {
        self.check_pins(pins, idle)?;
        if let Some(i) = self.slots.iter().position(|s| s.state == State::Free) {
            return Ok(Some(self.reserve(i)));
        }
        let mut best: Option<(usize, f64)> = None;
        for (i, s) in self.slots.iter().enumerate() {
            if let State::Resident { key, last_use } = s.state {
                if last_use < self.clock {
                    let p = prio(key);
                    if best.is_none_or(|(_, b)| p < b) {
                        best = Some((i, p));
                    }
                }
            }
        }
        Ok(match best {
            Some((i, p)) if p < ceiling => Some(self.reserve(i)),
            _ => None,
        })
    }

    fn check_pins(&self, pins: &Pins, idle: &Idle<'_>) -> Result<(), Error> {
        if pins.owner != self.owner || idle.e.owner != self.owner {
            return Err(foreign());
        }
        if pins.clock != self.clock {
            return Err(tensor_error("Pins de outra camada"));
        }
        Ok(())
    }

    /// Um slot `Free`, sem despejar ninguém (para `fill`).
    pub fn free_slot(&mut self) -> Option<Victim> {
        let i = self.slots.iter().position(|s| s.state == State::Free)?;
        Some(self.reserve(i))
    }

    fn reserve(&mut self, i: usize) -> Victim {
        if let State::Resident { key, .. } = self.slots[i].state {
            if self.slot_of[key] == Some(i) {
                self.slot_of[key] = None;
            }
        }
        self.slots[i].state = State::Reserved;
        self.slots[i].gen += 1;
        Victim { index: i, gen: self.slots[i].gen, owner: self.owner }
    }

    /// A vítima ainda está reservada (nada de `clear` ou outra cópia desde a reserva) e a chave é falta.
    fn check_copy(&self, v: &Victim, key: usize) -> Result<(), Error> {
        if v.owner != self.owner {
            return Err(foreign());
        }
        let s = &self.slots[v.index];
        if s.state != State::Reserved || s.gen != v.gen {
            return Err(tensor_error("vítima velha: o slot mudou desde a reserva"));
        }
        if self.slot_of[key].is_some() || self.pending_key[key] {
            return Err(tensor_error("cópia de uma chave que já está no cache ou a caminho"));
        }
        Ok(())
    }

    /// Copia o expert `key` para a vítima na stream de cálculo: o slot vale para os lançamentos seguintes da mesma stream.
    pub fn copy_on_compute(&mut self, v: Victim, key: usize) -> Result<SlotId, Error> {
        self.check_copy(&v, key)?;
        let cs = self.compute.clone();
        // SAFETY: a cópia vai na stream de cálculo, depois de todas as leituras já enfileiradas do slot.
        unsafe { self.copy_raw(v.index, key, &cs)? };
        let slot = &mut self.slots[v.index];
        slot.state = State::Resident { key, last_use: self.clock };
        slot.gen += 1;
        self.slot_of[key] = Some(v.index);
        Ok(SlotId { index: v.index, gen: slot.gen, owner: self.owner })
    }

    /// Copia o expert `key` para a vítima na stream de cópia, atrás de um evento gravado agora no cálculo (então depois de
    /// todas as leituras já enfileiradas do slot). O slot só vira `Resident` em [`Slots::settle`].
    pub fn copy_in_background(&mut self, v: Victim, key: usize) -> Result<LoadingSlot, Error> {
        self.check_copy(&v, key)?;
        self.order.record(&self.compute)?;
        self.copy.wait_event(&self.order)?;
        let cs = self.copy.clone();
        // SAFETY: a stream de cópia espera o evento gravado acima, depois de tudo o que já foi enfileirado no cálculo; o slot
        // está `Reserved` e só é lido de novo depois de `ready` (em `settle`).
        unsafe { self.copy_raw(v.index, key, &cs)? };
        let slot = &mut self.slots[v.index];
        slot.ready.record(&self.copy)?;
        slot.state = State::Loading { key, last_use: self.clock };
        self.pending_key[key] = true;
        self.pending.push_back(v.index);
        Ok(LoadingSlot { _index: v.index })
    }

    /// # Safety
    /// Nenhuma leitura pendente do slot roda depois da cópia (mesma stream, ou espera por evento).
    unsafe fn copy_raw(&mut self, s: usize, key: usize, stream: &Arc<Stream>) -> Result<(), Error> {
        let gu = 2 * self.inter * self.hidden;
        let src = self.store[key].as_ptr();
        let slot = &self.slots[s];
        // SAFETY: origem com 3·I·H u16, destinos com 2·I·H e H·I bf16; o buffer pinned vive enquanto `Slots` existir (o `Drop`
        // sincroniza as streams antes de soltá-lo).
        unsafe {
            cuda_core::memcpy_htod_async(slot.gate_up.device_pointer().cu_deviceptr(), src, gu, stream)?;
            cuda_core::memcpy_htod_async(slot.down.device_pointer().cu_deviceptr(), src.add(gu), gu / 2, stream)?;
        }
        Ok(())
    }

    /// Os pesos de um slot para um lançamento, se ele ainda está `Resident` na geração do `id`.
    pub fn resident(&self, id: SlotId) -> Result<ResidentSlot<'_>, Error> {
        if id.owner != self.owner {
            return Err(foreign());
        }
        let s = &self.slots[id.index];
        match s.state {
            State::Resident { .. } if s.gen == id.gen => Ok(ResidentSlot { gate_up: &s.gate_up, down: &s.down, owner: self.owner }),
            _ => Err(tensor_error("slot mudou desde a consulta")),
        }
    }

    /// Espera as duas streams (depois de um erro, e no `Drop`).
    pub fn sync_all(&self) {
        // SAFETY: só espera as streams; erros aqui não têm a quem ser devolvidos.
        unsafe {
            let _ = self.compute.synchronize();
            let _ = self.copy.synchronize();
        }
    }
}

impl Drop for Slots {
    fn drop(&mut self) {
        // Cópias em segundo plano leem `store` e escrevem os slots, e kernels pendentes leem os slots; o cutile solta a memória
        // do device numa stream de desalocação própria, que não espera estas.
        self.sync_all();
    }
}

struct Scratch {
    logits: Partition<Tensor<f32>>,
    a: Partition<Tensor<f32>>,
    y: Partition<Tensor<f32>>,
}

/// A stream de cálculo do cache, com o scratch e os uploads da CPU. Lançar pede `&mut Epoch`; o [`Idle`] a toma emprestada.
pub struct Epoch {
    owner: u64,
    hidden: usize,
    compute: Arc<Stream>,
    scratch: Option<Scratch>,
    /// Tensores com a saída da CPU enviada à GPU; vivem até a próxima sincronização da stream de cálculo.
    keep: Vec<Tensor<f32>>,
    gen: ExpertGen,
    gen_h: Vec<String>,
    gen_ax: Vec<String>,
    /// Experts da GPU enfileirados nesta camada.
    launched: usize,
}

const POISONED: &str = "cache de experts inutilizável depois de um erro anterior";

/// A stream de cálculo parada (logo depois da leitura dos logits do roteador). Enquanto ele vive, nada é lançado.
pub struct Idle<'a> {
    e: &'a mut Epoch,
}

impl Epoch {
    fn new(hidden: usize, inter: usize, experts: usize, compute: Arc<Stream>, owner: u64) -> Result<Self, Error> {
        let z = |n: usize, b: usize| -> Result<Partition<Tensor<f32>>, Error> { Ok(api::zeros::<f32>(&[n]).sync_on(&compute)?.partition([b])) };
        let scratch = Scratch { logits: z(experts, GEMV_BM)?, a: z(inter, SWIGLU_BM)?, y: z(hidden, GEMV_BM)? };
        Ok(Self {
            owner,
            hidden,
            compute: compute.clone(),
            scratch: Some(scratch),
            keep: Vec::new(),
            gen: ExpertGen::new(hidden, inter),
            gen_h: crate::gpu::gemv_generics(hidden),
            gen_ax: vec![AXPBY_B.to_string()],
            launched: 0,
        })
    }

    /// Calcula os logits do roteador e espera a leitura deles: a stream de cálculo fica parada, e o [`Idle`] devolvido diz isso.
    pub fn router_logits(&mut self, router: &Tensor<bf16>, xd: &Tensor<f32>) -> Result<(Vec<f32>, Idle<'_>), Error> {
        let mut sc = self.scratch.take().ok_or_else(|| tensor_error(POISONED))?;
        let s = self.compute.clone();
        // SAFETY: `unchecked_accesses`: E e H são múltiplos de BM e BK (conferido em `ExpertCache::new`).
        let (logits, _, _) = unsafe { kernels::gemv(sc.logits, router, xd) }.generics(self.gen_h.clone()).sync_on(&s).map_err(sync_on_err(&s))?;
        let logits = Arc::new(logits.unpartition());
        let lv = api::copy_device_to_host_vec(&logits).sync_on(&s).map_err(sync_on_err(&s))?;
        sc.logits = Arc::try_unwrap(logits).map_err(|_| "logits ainda referenciados").unwrap().partition([GEMV_BM]);
        self.scratch = Some(sc);
        self.launched = 0;
        Ok((lv, Idle { e: self }))
    }

    /// Enfileira um expert residente, acumulando em `y` (o primeiro da camada sobrescreve).
    pub fn launch_expert(&mut self, slot: ResidentSlot<'_>, xd: &Tensor<f32>, g: f32) -> Result<(), Error> {
        if slot.owner != self.owner {
            return Err(foreign());
        }
        let sc = self.scratch.take().ok_or_else(|| tensor_error(POISONED))?;
        let beta = if self.launched == 0 { 0.0f32 } else { 1.0 };
        let s = self.compute.clone();
        // SAFETY: formas conferidas em `ExpertCache::new` e em `enqueue_expert`; o slot é `Resident` (o `ResidentSlot` o
        // prova) e só é sobrescrito depois desta leitura (cópia na mesma stream, ou atrás do evento de ordem); `xd` e o
        // scratch vivem até a próxima sincronização.
        let (a, y) = unsafe { enqueue_expert(&s, &self.gen, sc.a, sc.y, slot.gate_up, slot.down, xd, g, beta)? };
        self.scratch = Some(Scratch { logits: sc.logits, a, y });
        self.launched += 1;
        Ok(())
    }

    /// `h ← h + resid·y`, se algum expert foi lançado nesta camada.
    pub fn finish_gpu(&mut self, h: Partition<Tensor<f32>>, resid: f32) -> Result<Partition<Tensor<f32>>, Error> {
        if self.launched == 0 {
            return Ok(h);
        }
        let mut sc = self.scratch.take().ok_or_else(|| tensor_error(POISONED))?;
        let s = self.compute.clone();
        let y = sc.y.unpartition();
        // SAFETY: `y` e `h` com H elementos, H múltiplo de `AXPBY_B`; vivem até a próxima sincronização.
        let (h2, _, _, _) = unsafe { kernels::axpby(h, &y, resid, 1.0f32).generics(self.gen_ax.clone()).async_on(&s).map_err(sync_on_err(&s))? };
        sc.y = y.partition([GEMV_BM]);
        self.scratch = Some(sc);
        Ok(h2)
    }

    /// `h ← h + resid·y_cpu`: envia a saída dos experts da CPU e a soma; o tensor enviado fica até [`Idle::release_uploads`].
    pub fn add_cpu_out(&mut self, h: Partition<Tensor<f32>>, y: Vec<f32>, resid: f32) -> Result<Partition<Tensor<f32>>, Error> {
        assert_eq!(y.len(), self.hidden, "saída da CPU com tamanho errado");
        let s = self.compute.clone();
        // SAFETY: a cópia retém o vetor de origem; `yd` fica em `keep` até a próxima sincronização da stream.
        let yd = unsafe { api::copy_host_vec_to_device(&Arc::new(y)).async_on(&s).map_err(sync_on_err(&s))? };
        // SAFETY: como em `finish_gpu`.
        let (h2, _, _, _) = unsafe { kernels::axpby(h, &yd, resid, 1.0f32).generics(self.gen_ax.clone()).async_on(&s).map_err(sync_on_err(&s))? };
        self.keep.push(yd);
        Ok(h2)
    }

    /// Espera a stream de cálculo (depois de um erro).
    pub fn sync(&self) {
        // SAFETY: só espera a stream.
        let _ = unsafe { self.compute.synchronize() };
    }
}

impl Drop for Epoch {
    fn drop(&mut self) {
        // O scratch e os uploads em `keep` são lidos por kernels pendentes.
        self.sync();
    }
}

impl Idle<'_> {
    /// Solta os uploads da saída da CPU das camadas anteriores: a stream de cálculo já os consumiu.
    pub fn release_uploads(&mut self) {
        self.e.keep.clear();
    }

    /// `x` no host para os experts da CPU. Espera a stream de cálculo, que ainda não tem os experts da GPU desta camada (só as
    /// cópias da própria stream, se houver).
    pub fn read_x(&mut self, xd: &Tensor<f32>) -> Result<Vec<f32>, Error> {
        let hd = self.e.hidden;
        let s = self.e.compute.clone();
        let mut x = vec![0f32; hd];
        // SAFETY: `xd` tem H f32 e foi escrito antes da sincronização do roteador; `x` vive até o `synchronize`.
        unsafe {
            cuda_core::memcpy_dtoh_async(x.as_mut_ptr(), xd.device_pointer().cu_deviceptr(), hd, &s).map_err(sync_on_err(&s))?;
            s.synchronize().map_err(sync_on_err(&s))?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    //! Testes do núcleo na GPU. Os dois que forçam estados que o protocolo nunca deixa acontecer usam um portão: um callback
    //! de host (`cuLaunchHostFunc`) que bloqueia uma stream até o teste mandar abrir.
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    const H: usize = 256;
    const I: usize = 128;
    const E: usize = 8;
    const RODADAS: usize = 20;

    /// A GPU com a trava dos testes de unidade (`crate::gpu_test_lock`), segura enquanto o teste viver.
    fn device() -> Option<(std::sync::MutexGuard<'static, ()>, Arc<Device>)> {
        let trava = crate::gpu_test_lock();
        match Device::new(0) {
            Ok(d) => Some((trava, d)),
            Err(e) => {
                assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
                None
            }
        }
    }

    fn pesos(n: usize, key: usize) -> Vec<bf16> {
        (0..n).map(|k| bf16::from_f32(((k * 37 + key * 11) % 17) as f32 * 0.01 - 0.08)).collect()
    }

    /// Par com `n_keys` experts distintos e `n_slots` slots, mais um roteador zerado e um `x` qualquer.
    fn par(s: &Arc<Stream>, n_keys: usize, n_slots: usize) -> (Slots, Epoch, Tensor<bf16>, Tensor<f32>) {
        let w: Vec<(Vec<bf16>, Vec<bf16>)> = (0..n_keys).map(|k| (pesos(2 * I * H, k), pesos(H * I, k))).collect();
        let experts: Vec<(&[bf16], &[bf16])> = w.iter().map(|(g, d)| (g.as_slice(), d.as_slice())).collect();
        let (slots, epoch) = new_pair(&experts, H, I, E, n_slots, s.clone()).unwrap();
        let router = api::zeros::<bf16>(&[E * H]).sync_on(s).unwrap().reshape(&[E, H]).unwrap();
        let x: Vec<f32> = (0..H).map(|k| ((k % 13) as f32 - 6.0) * 0.05).collect();
        let xd = api::copy_host_vec_to_device(&Arc::new(x)).sync_on(s).unwrap();
        (slots, epoch, router, xd)
    }

    /// Fecha `stream` até o `send` no transmissor devolvido. O prazo abre o portão mesmo se o teste travar antes do `send`, em
    /// vez de prender a stream para sempre.
    fn portao(slots: &Slots, stream: &Arc<Stream>) -> mpsc::Sender<()> {
        let (tx, rx) = mpsc::channel::<()>();
        slots._ctx.bind_to_thread().unwrap();
        // SAFETY: contexto corrente (acima); o callback não chama CUDA.
        unsafe {
            stream.launch_host_function(move || {
                let _ = rx.recv_timeout(Duration::from_secs(10));
            })
        }
        .unwrap();
        tx
    }

    fn abre(tx: mpsc::Sender<()>) {
        tx.send(()).expect("o portão abriu pelo prazo: algo sincronizou a stream fechada antes do send");
    }

    /// A `copy_in_background` sem gravar nem esperar o evento de ordem.
    fn copia_sem_ordem(slots: &mut Slots, v: Victim, key: usize) -> Result<(), Error> {
        slots.check_copy(&v, key)?;
        let cs = slots.copy.clone();
        // SAFETY: nenhuma; é o mutante, e a corrida é o que o teste quer ver.
        unsafe { slots.copy_raw(v.index, key, &cs)? };
        let slot = &mut slots.slots[v.index];
        slot.ready.record(&slots.copy)?;
        slot.state = State::Loading { key, last_use: slots.clock };
        slots.pending_key[key] = true;
        slots.pending.push_back(v.index);
        Ok(())
    }

    fn saida(h: Partition<Tensor<f32>>, s: &Arc<Stream>) -> Vec<f32> {
        api::copy_device_to_host_vec(&Arc::new(h.unpartition())).sync_on(s).unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// Controle negativo do B0 (seção "B0 — desenho revisado", teste 2). No protocolo do cache a stream de cálculo está parada
    /// quando a vítima é escolhida, então a espera do evento de ordem nunca é exercitada por ele. Aqui a janela é forçada:
    /// expert 0 no slot, um lançamento dele atrás de um portão na stream de cálculo, e só então a cópia do expert 1 para o
    /// mesmo slot (`copia`: nenhuma, pela `copy_in_background`, ou pelo mutante). Devolve `h = y` do lançamento.
    fn rodada(slots: &mut Slots, epoch: &mut Epoch, xd: &Tensor<f32>, s: &Arc<Stream>, copia: Option<bool>) -> Vec<f32> {
        slots.clear().unwrap();
        let v = slots.free_slot().unwrap();
        let id = slots.copy_on_compute(v, 0).unwrap();
        // Antes do portão: `sync_on` na stream fechada só voltaria pelo prazo do callback.
        let h0 = api::zeros::<f32>(&[H]).sync_on(s).unwrap().partition([AXPBY_B]);
        slots.sync_all();
        let tx = copia.map(|_| portao(slots, s));
        epoch.launched = 0;
        epoch.launch_expert(slots.resident(id).unwrap(), xd, 1.0).unwrap();
        let h = epoch.finish_gpu(h0, 1.0).unwrap();
        if let Some(sem_ordem) = copia {
            let v = slots.reserve(id.index);
            if sem_ordem {
                copia_sem_ordem(slots, v, 1).unwrap();
                // A cópia do mutante termina antes de o portão abrir: a corrida deixa de depender de tempo. (Na tipada isso
                // travaria até o prazo, porque a cópia espera o portão pelo evento de ordem.)
                // SAFETY: só espera a stream de cópia.
                unsafe { slots.copy.synchronize() }.unwrap();
            } else {
                slots.copy_in_background(v, 1).unwrap();
            }
        }
        if let Some(tx) = tx {
            abre(tx);
        }
        slots.sync_all();
        saida(h, s)
    }

    #[test]
    fn copia_em_segundo_plano_espera_a_leitura_do_slot() {
        let Some((_trava, device)) = device() else { return };
        let s = device.new_stream().unwrap();
        let (mut slots, mut epoch, _router, xd) = par(&s, 2, 1);

        // Referências: o expert 0 e o expert 1 lidos sem cópia nenhuma depois.
        let y0 = rodada(&mut slots, &mut epoch, &xd, &s, None);
        slots.clear().unwrap();
        let v = slots.free_slot().unwrap();
        let id1 = slots.copy_on_compute(v, 1).unwrap();
        epoch.launched = 0;
        epoch.launch_expert(slots.resident(id1).unwrap(), &xd, 1.0).unwrap();
        let h0 = api::zeros::<f32>(&[H]).sync_on(&s).unwrap().partition([AXPBY_B]);
        let y1 = saida(epoch.finish_gpu(h0, 1.0).unwrap(), &s);
        assert_ne!(bits(&y0), bits(&y1), "os dois experts dão a mesma saída: o teste não distinguiria a corrida");

        let tipada = (0..RODADAS).filter(|_| bits(&rodada(&mut slots, &mut epoch, &xd, &s, Some(false))) == bits(&y0)).count();
        let mutante = (0..RODADAS).filter(|_| bits(&rodada(&mut slots, &mut epoch, &xd, &s, Some(true))) == bits(&y1)).count();
        eprintln!("janela forçada: tipada igual à referência (expert antigo) em {tipada}/{RODADAS}; mutante leu o expert novo em {mutante}/{RODADAS}");
        assert_eq!(tipada, RODADAS, "a cópia pela copy_in_background sobrescreveu o slot antes da leitura");
        assert_eq!(mutante, RODADAS, "o mutante sem a espera do evento não leu o expert novo: a janela não foi forçada");
    }

    /// Estados que a barreira nunca alcança (ela drena `pending` antes do `pin_layer`): um slot `Loading` fica fora do mapa,
    /// não é promovido antes do evento, e ganha o uso das camadas que o pedem, o que decide a vítima depois do `settle`.
    /// `pin` falso é o controle negativo (o mesmo que um `pin_layer` que esquecesse os slots carregando).
    fn vitima_depois_de_carregar(s: &Arc<Stream>, pin: bool) -> usize {
        let (mut slots, mut epoch, router, xd) = par(s, 4, 2);
        for key in [0, 1] {
            let v = slots.free_slot().unwrap();
            slots.copy_on_compute(v, key).unwrap();
        }
        slots.sync_all();
        // Camada 1: pede o expert 2, que vai para o slot 0 (último uso 0 nos dois, desempate pelo índice).
        // Só operações assíncronas entre o portão e o `abre`: o `router_logits` (que aloca e sincroniza) vem antes de fechar.
        let pins = slots.pin_layer(&[2]);
        let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
        let tx = portao(&slots, &slots.copy.clone());
        let v = slots.victim(&pins, &idle).unwrap().unwrap();
        assert_eq!(v.index, 0);
        slots.copy_in_background(v, 2).unwrap();
        drop(idle);
        assert_eq!(slots.lookup(2), Lookup::Loading);
        assert_eq!(slots.resident_count(), 1);
        assert_eq!(slots.loading_count(), 1);
        slots.settle(false).unwrap();
        assert_eq!(slots.lookup(2), Lookup::Loading, "settle promoveu um slot com a cópia ainda presa no portão");
        // Camada 2 usa o expert 1 (slot 1, último uso 2); camada 3 pede o 2 de novo, ainda carregando (ou nada, no controle).
        let _ = slots.pin_layer(&[1]);
        let _ = slots.pin_layer(if pin { &[2][..] } else { &[][..] });
        abre(tx);
        slots.settle(true).unwrap();
        assert!(matches!(slots.lookup(2), Lookup::Hit(_)));
        // Camada 4 pede o expert 3: a vítima é o slot de uso mais antigo.
        let pins = slots.pin_layer(&[3]);
        let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
        let v = slots.victim(&pins, &idle).unwrap().unwrap();
        v.index
    }

    #[test]
    fn slot_carregando_fica_fora_do_mapa_e_ganha_o_uso_da_camada() {
        let Some((_trava, device)) = device() else { return };
        let s = device.new_stream().unwrap();
        assert_eq!(vitima_depois_de_carregar(&s, true), 1, "o slot que carregava o expert pedido na camada 3 foi despejado");
        assert_eq!(vitima_depois_de_carregar(&s, false), 0, "controle negativo: sem o pin da camada 3 a vítima devia ser o slot 0");
    }

    /// As conferências em tempo de execução: geração ao lançar, `Pins` de outra camada, vítima velha, chave que não é falta, e
    /// objetos de outra instância.
    #[test]
    fn vitima_por_prioridade_recusa_sem_mudar_estado() {
        // C: o lfu_cpu só despeja o residente de menor pontuação, e só se o candidato o vence; recusar não pode reservar.
        let Some((_trava, device)) = device() else { return };
        let s = device.new_stream().unwrap();
        let (mut slots, mut epoch, router, xd) = par(&s, 5, 3);
        let prio = |key: usize| [3.0, 1.0, 1.0, 9.0, 0.0][key];
        // Com slot livre, admite sem olhar a prioridade.
        let pins = slots.pin_layer(&[]);
        {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            let v = slots.victim_min_by(&pins, &idle, prio, f64::NEG_INFINITY).unwrap().expect("slot livre recusado");
            slots.copy_on_compute(v, 0).unwrap();
        }
        for key in [1, 2] {
            let v = slots.free_slot().unwrap();
            slots.copy_on_compute(v, key).unwrap();
        }
        slots.sync_all();
        let antes: Vec<(State, u64)> = slots.slots.iter().map(|x| (x.state, x.gen)).collect();

        // Cheio. O menor fora dos pinos é o 1 ou o 2 (empate em 1,0): sai o de menor índice de slot. Teto igual: recusa.
        let pins = slots.pin_layer(&[]);
        {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            assert!(slots.victim_min_by(&pins, &idle, prio, 1.0).unwrap().is_none(), "empate com o teto admitido");
        }
        let depois: Vec<(State, u64)> = slots.slots.iter().map(|x| (x.state, x.gen)).collect();
        assert_eq!(antes, depois, "recusar mudou estado ou geração");
        assert!(matches!(slots.lookup(1), Lookup::Hit(_)) && matches!(slots.lookup(2), Lookup::Hit(_)));

        // Pino: com 1 fixado, o candidato que vence sai pelo 2; o 0 (3,0) fica.
        let slot_de = |slots: &Slots, key: usize| slots.slot_of[key].unwrap();
        let s2 = slot_de(&slots, 2);
        let pins = slots.pin_layer(&[1]);
        {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            let v = slots.victim_min_by(&pins, &idle, prio, 9.0).unwrap().expect("candidato que vence recusado");
            assert_eq!(v.index, s2);
            slots.copy_on_compute(v, 3).unwrap();
        }
        slots.sync_all();
        assert!(matches!(slots.lookup(2), Lookup::Miss) && matches!(slots.lookup(1), Lookup::Hit(_)));

        // Sem pinos e empate entre slots: o de menor índice. Aqui 1 (1,0) é o único mínimo; depois dele, 0 (3,0) < 3 (9,0).
        let s1 = slot_de(&slots, 1);
        let s0 = slot_de(&slots, 0);
        let pins = slots.pin_layer(&[]);
        {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            let v = slots.victim_min_by(&pins, &idle, prio, 2.0).unwrap().unwrap();
            assert_eq!(v.index, s1);
            slots.copy_on_compute(v, 4).unwrap();
        }
        let pins = slots.pin_layer(&[4]);
        {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            let v = slots.victim_min_by(&pins, &idle, |k| if k == 4 { 0.0 } else { 5.0 }, 6.0).unwrap().unwrap();
            assert_eq!(v.index, s0.min(slot_de(&slots, 3)), "empate não saiu pelo menor índice");
            slots.copy_on_compute(v, 2).unwrap();
        }
        slots.sync_all();
    }

    #[test]
    fn conferencias_em_tempo_de_execucao() {
        let Some((_trava, device)) = device() else { return };
        let s = device.new_stream().unwrap();
        let (mut slots, mut epoch, router, xd) = par(&s, 3, 2);
        let mut ids = Vec::new();
        for key in [0, 1] {
            let v = slots.free_slot().unwrap();
            ids.push(slots.copy_on_compute(v, key).unwrap());
        }
        slots.sync_all();

        // Pins de outra camada.
        let velho = slots.pin_layer(&[]);
        let pins = slots.pin_layer(&[]);
        {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            assert!(slots.victim(&velho, &idle).is_err(), "Pins de outra camada aceito");
            // Sem chave fixada, o acerto do expert 0 pode ser despejado; o SlotId velho deixa de servir.
            let v = slots.victim(&pins, &idle).unwrap().unwrap();
            assert_eq!(v.index, ids[0].index);
            // Chave que já está no cache.
            assert!(slots.check_copy(&v, 1).is_err(), "cópia de chave residente aceita");
            slots.copy_on_compute(v, 2).unwrap();
        }
        assert!(slots.resident(ids[0]).is_err(), "SlotId de um slot sobrescrito ainda serve para lançar");
        assert!(slots.resident(ids[1]).is_ok());

        // Vítima velha: reservada antes de um clear.
        let pins = slots.pin_layer(&[]);
        let v = {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            slots.victim(&pins, &idle).unwrap().unwrap()
        };
        slots.clear().unwrap();
        assert!(slots.copy_on_compute(v, 0).is_err(), "vítima de antes do clear aceita");

        // Outra instância: Pins, Idle, SlotId e ResidentSlot do outro par dão erro.
        let (mut slots2, mut epoch2, router2, xd2) = par(&s, 3, 2);
        let v2 = slots2.free_slot().unwrap();
        let id2 = slots2.copy_on_compute(v2, 0).unwrap();
        slots2.sync_all();
        let pins2 = slots2.pin_layer(&[]);
        let pins = slots.pin_layer(&[]);
        {
            let (_lv, idle) = epoch.router_logits(&router, &xd).unwrap();
            assert!(slots.victim(&pins2, &idle).is_err(), "Pins de outra instância aceito");
        }
        {
            let (_lv, idle2) = epoch2.router_logits(&router2, &xd2).unwrap();
            assert!(slots.victim(&pins, &idle2).is_err(), "Idle de outra instância aceito");
        }
        assert!(slots.resident(id2).is_err(), "SlotId de outra instância aceito");
        assert!(epoch.launch_expert(slots2.resident(id2).unwrap(), &xd, 1.0).is_err(), "ResidentSlot de outra instância lançado");
    }

    /// Custo de host do protocolo por camada (critério do B0: ≤ 3 µs) na parte que não depende do tamanho dos experts:
    /// `settle` com 4 cópias em segundo plano já prontas na fila (as `query()` dos eventos), `pin_layer`, a consulta de k
    /// chaves e a conferência de estado, geração e instância de cada acerto, com 64 slots e k = 8. Escolha de vítima e cópias
    /// ficam de fora (custam o mesmo que antes do B0; o tempo de parede do `m2a` as cobre).
    #[test]
    fn protocolo_custa_pouco_no_host() {
        let Some((_trava, device)) = device() else { return };
        let s = device.new_stream().unwrap();
        let (n, k, fila) = (64usize, 8usize, 4usize);
        let gu = vec![bf16::from_f32(0.0); 2 * I * H];
        let dn = vec![bf16::from_f32(0.0); H * I];
        let experts: Vec<(&[bf16], &[bf16])> = (0..2 * n).map(|_| (gu.as_slice(), dn.as_slice())).collect();
        let (mut slots, _epoch) = new_pair(&experts, H, I, E, n, s.clone()).unwrap();
        while let Some(v) = slots.free_slot() {
            let key = slots.resident_count();
            slots.copy_on_compute(v, key).unwrap();
        }
        slots.sync_all();
        // `fila` slots ficam como cópias já terminadas na fila, para o `settle` consultar os eventos a cada camada; o
        // estado é refeito a cada volta (fora da medida não dá: o `settle` os promove).
        for j in 0..fila {
            slots.slots[j].ready.record(&slots.copy).unwrap();
        }
        slots.sync_all();
        let carregar = |slots: &mut Slots| {
            for j in 0..fila {
                if let State::Resident { key, last_use } = slots.slots[j].state {
                    slots.slot_of[key] = None;
                    slots.pending_key[key] = true;
                    slots.slots[j].state = State::Loading { key, last_use };
                    slots.pending.push_back(j);
                }
            }
        };
        let camadas = 100_000usize;
        let mut medido = Duration::ZERO;
        let mut acertos = 0usize;
        for c in 0..camadas {
            carregar(&mut slots);
            let t0 = std::time::Instant::now();
            slots.settle(false).unwrap();
            let keys: Vec<usize> = (0..k).map(|j| (c * 7 + j * 13) % (2 * n)).collect();
            let _pins = slots.pin_layer(&keys);
            for &key in &keys {
                if let Lookup::Hit(id) = slots.lookup(key) {
                    acertos += slots.resident(id).is_ok() as usize;
                }
            }
            medido += t0.elapsed();
        }
        assert_eq!(slots.loading_count(), 0, "o settle não promoveu as cópias prontas");
        let us = medido.as_secs_f64() * 1e6 / camadas as f64;
        eprintln!("protocolo no host: {us:.3} µs por camada ({fila} eventos consultados, {acertos} acertos em {camadas} camadas, k = {k}, {n} slots)");
        assert!(acertos > 0);
        assert!(us <= 3.0, "protocolo custa {us:.3} µs por camada no host (critério: ≤ 3)");
    }
}
