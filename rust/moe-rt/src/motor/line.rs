//! A linha de serviço: dona do cache (slots, tabela, frequências, admissões), da stream de cópia e do [`TokenEpoch`]
//! (plano, A0b, passos 2 a 4 e "Tipos").
//!
//! Por token: [`Service::serve_layer`] espera `seq[l]` e devolve a [`LayerView`]; [`answer`] escreve a saída da CPU e a
//! flag, uma vez só, e devolve o [`Served`]; com ele vêm os [`Pins`], a vítima ([`Service::victim`]) e a cópia
//! ([`Service::copy_after_reader`]). [`Service::end_token`] consome o `TokenEpoch`, faz a virada e manda o
//! [`Published`] do token seguinte.
//!
//! A escolha das admissões é a do simulador no protocolo `token` (`tools/moe_sim.py`, `Sim.layer`, `admit_bg`,
//! `make_room` e `victim`), para o modo com barreira bater com ele: a vítima é a residente fora dos pins com o menor
//! `(pontuação, versão, chave)`, e a versão de uma chave sobe a cada vez que o simulador a empurra no heap (no `settle` e
//! num acerto de chave residente).
use super::backend::{geq, release_value, Backend, HostExperts, LayerIo, Role};
use super::engine::{Published, Shared};
use super::mem::{Mapped, Pinned};
use super::sync::{AtomicBool, AtomicU32, Ordering};
use super::{AdmitMode, Config, Error, Mutant, Stats, NO_SLOT};
use crate::cache::Freq;
use std::sync::Arc;

type R<T, B> = Result<T, Error<<B as Backend>::Error>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SlotState {
    Free,
    Resident { key: u32 },
    /// Cópia enfileirada e `ready` gravado; ainda fora da tabela.
    Loading { key: u32 },
    /// A cópia para uma vítima da camada `after` espera `seq[after + 1]` (ou `seq[L]`).
    Held { key: u32, after: usize },
}

/// A tabela de um token: chave → slot ou [`NO_SLOT`]. Sai da virada e é consumida pela seguinte.
pub struct TokenEpoch {
    t: u32,
    table: Vec<u32>,
}

impl TokenEpoch {
    pub fn token(&self) -> u32 {
        self.t
    }

    pub fn lookup(&self, key: u32) -> Option<usize> {
        let s = self.table[key as usize];
        (s != NO_SLOT).then_some(s as usize)
    }
}

/// O que a GPU publicou na camada `layer` deste token. Prova que a GPU passou pelo roteador de `layer` (o host viu
/// `seq[layer] >= t` com `Acquire` antes de ler ids e `x`).
pub struct LayerView<'t> {
    epoch: &'t TokenEpoch,
    layer: usize,
    keys: Vec<u32>,
    misses: Vec<u32>,
    /// Os bits `f32` dos pesos do softmax, na ordem de `keys`.
    w: Vec<u32>,
    x: Vec<u32>,
    io: Arc<Mapped<LayerIo>>,
    flag: Arc<Mapped<AtomicU32>>,
    flag_first: bool,
}

impl<'t> LayerView<'t> {
    pub fn layer(&self) -> usize {
        self.layer
    }

    /// As chaves pedidas, na ordem do top-k.
    pub fn keys(&self) -> &[u32] {
        &self.keys
    }

    pub fn x(&self) -> &[u32] {
        &self.x
    }

    /// As chaves fora da tabela do token, que vão à CPU, na ordem do top-k.
    pub fn cpu_keys(&self) -> Vec<u32> {
        self.keys.iter().copied().filter(|&k| self.epoch.lookup(k).is_none()).collect()
    }

    /// Os pesos (bits `f32`) das chaves de [`cpu_keys`](Self::cpu_keys), na mesma ordem.
    pub fn cpu_weights(&self) -> Vec<u32> {
        self.keys.iter().zip(&self.w).filter(|(&k, _)| self.epoch.lookup(k).is_none()).map(|(_, &w)| w).collect()
    }
}

/// A camada respondida: a saída da CPU e a flag já foram escritas. Continua provando que a GPU passou pelo roteador da
/// camada; as admissões pedem um `&Served`.
pub struct Served<'t> {
    epoch: &'t TokenEpoch,
    layer: usize,
    keys: Vec<u32>,
    misses: Vec<u32>,
}

impl<'t> Served<'t> {
    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn token(&self) -> u32 {
        self.epoch.t
    }

    /// As faltas a admitir (fora da tabela e sem cópia em andamento), na ordem do top-k.
    pub fn misses(&self) -> &[u32] {
        &self.misses
    }
}

/// `seq[i] >= t` visto pelo host: `Acquire`, que ordena a leitura de ids e `x` depois dela (`Relaxed` só no mutante).
pub(crate) fn seq_reached(seq: &AtomicU32, t: u32, relaxed: bool) -> bool {
    geq(seq.load(if relaxed { Ordering::Relaxed } else { Ordering::Acquire }), t)
}

/// A saída da CPU (`Relaxed`) e depois a flag com `Release`; o mutante `FlagBeforeY` inverte a ordem.
pub(crate) fn write_answer(y_io: &[AtomicU32], flag: &AtomicU32, t: u32, y: &[u32], flag_first: bool) {
    if flag_first {
        flag.store(t, Ordering::Release);
    }
    for (a, &v) in y_io.iter().zip(y) {
        a.store(v, Ordering::Relaxed);
    }
    if !flag_first {
        flag.store(t, Ordering::Release);
    }
}

/// A camada de vítimas cujas cópias guardadas saem quando o host vê `seq[l]`: as da camada `l - 1`, cujo lançamento a GPU
/// já terminou (ela grava `seq[l]` depois dele). O mutante `HeldOneLayerEarly` solta as da própria camada `l`.
pub(crate) fn release_layer(l: usize, early: bool) -> Option<usize> {
    if early { Some(l) } else { l.checked_sub(1) }
}

/// A leitura de ids e `x` da camada: só depois de ver `seq >= t` (a composição de produção, que o modelo do `loom` usa).
pub(crate) fn read_layer(seq: &AtomicU32, io: &LayerIo, t: u32, relaxed: bool) -> Option<(Vec<u32>, Vec<u32>)> {
    seq_reached(seq, t, relaxed).then(|| (LayerIo::load(&io.ids), LayerIo::load(&io.x)))
}

/// A cópia guardada com `after` sai quando a soltura é da camada `k`.
pub(crate) fn held_due(after: usize, k: usize) -> bool {
    after == k
}

/// Escreve a saída da CPU e depois a flag com `Release`, uma vez só por camada (consome a `LayerView`).
pub fn answer<'t>(view: LayerView<'t>, y: &[u32]) -> Served<'t> {
    let t = view.epoch.t;
    write_answer(&view.io.y, &view.flag, t, y, view.flag_first);
    Served { epoch: view.epoch, layer: view.layer, keys: view.keys, misses: view.misses }
}

/// As chaves pedidas na camada, protegidas da escolha de vítima. Só saem de [`Service::pin_layer`].
pub struct Pins {
    layer: usize,
    keys: Vec<u32>,
}

/// Um slot para a admissão: livre, ou o de uma residente despejada. Não é `Clone`; [`Service::copy_after_reader`] o
/// consome.
#[derive(Debug)]
#[must_use = "a vítima já saiu da tabela: sem `copy_after_reader` o slot fica livre enquanto a GPU ainda pode lê-lo"]
pub struct Victim {
    slot: usize,
    evicted: Option<u32>,
}

/// A montagem da tabela do token: só aceita slot residente, cujo `ready` terminou na consulta (teste de estado (a)).
pub(crate) struct TableBuilder {
    table: Vec<u32>,
}

impl TableBuilder {
    pub(crate) fn new(keys: usize) -> Self {
        Self { table: vec![NO_SLOT; keys] }
    }

    pub(crate) fn insert(&mut self, slot: usize, st: SlotState) -> Result<(), usize> {
        match st {
            SlotState::Resident { key } => {
                self.table[key as usize] = slot as u32;
                Ok(())
            }
            _ => Err(slot),
        }
    }

    pub(crate) fn finish(self) -> Vec<u32> {
        self.table
    }
}

pub struct Service<B: Backend> {
    b: Arc<B>,
    sh: Arc<Shared<B>>,
    copy: Arc<B::Stream>,
    weights: Pinned<B::Weights>,
    cfg: Config,
    freq: Freq,
    step: u64,
    slots: Vec<SlotState>,
    key_slot: Vec<u32>,
    version: Vec<u64>,
    /// Slots com cópia guardada, na ordem das admissões.
    held: Vec<usize>,
    /// Cópias soltas na virada cujo `ready` ainda não foi gravado (só nos mutantes da gravação tardia).
    late_record: Vec<usize>,
    sides: [Option<Pinned<[AtomicU32]>>; 2],
    stats: Stats,
}

impl<B: Backend> Service<B> {
    pub(crate) fn new(b: Arc<B>, sh: Arc<Shared<B>>, copy: Arc<B::Stream>, weights: Pinned<B::Weights>, cfg: Config,
                      sides: [Pinned<[AtomicU32]>; 2]) -> Self {
        let keys = cfg.layers * cfg.experts;
        let [s0, s1] = sides;
        Self {
            b,
            sh,
            copy,
            weights,
            freq: Freq::new(keys, cfg.layers),
            step: 0,
            slots: vec![SlotState::Free; cfg.slots],
            key_slot: vec![NO_SLOT; keys],
            version: vec![0; keys],
            held: Vec::new(),
            late_record: Vec::new(),
            sides: [Some(s0), Some(s1)],
            stats: Stats::default(),
            cfg,
        }
    }

    fn be<T>(r: Result<T, B::Error>) -> R<T, B> {
        r.map_err(Error::Backend)
    }

    /// Espera `seq[i] >= t` com `Acquire` (ou `Relaxed`, no mutante), consultando o cancelamento.
    fn wait_seq(&self, i: usize, t: u32) -> R<(), B> {
        let relaxed = self.cfg.is(Mutant::SeqRelaxed);
        loop {
            if seq_reached(&self.sh.seq[i], t, relaxed) {
                return Ok(());
            }
            if self.sh.cancel.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            Self::be(self.b.relax())?;
        }
    }

    /// Espera `seq[l] >= t` e lê ids e `x` (por [`read_layer`]), consultando o cancelamento.
    fn wait_layer(&self, l: usize, t: u32) -> R<(Vec<u32>, Vec<u32>, Vec<u32>), B> {
        let relaxed = self.cfg.is(Mutant::SeqRelaxed);
        let since = crate::diag::now();
        let mut first = true;
        loop {
            if let Some((ids, x)) = read_layer(&self.sh.seq[l], &self.sh.io[l], t, relaxed) {
                crate::diag::wait_done(since, first);
                // Os pesos vêm da mesma publicação: lidos depois do `Acquire` de `seq`, como ids e `x`.
                return Ok((ids, LayerIo::load(&self.sh.io[l].w), x));
            }
            first = false;
            if self.sh.cancel.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            Self::be(self.b.relax())?;
        }
    }

    fn enqueue_copy(&mut self, slot: usize, key: u32, record: bool) -> R<(), B> {
        Self::be(self.b.copy_expert(&self.copy, &self.sh.bank, slot, &self.weights, key))?;
        if record {
            Self::be(self.b.record(&self.sh.ready[slot], &self.copy))?;
        } else {
            self.late_record.push(slot);
        }
        self.slots[slot] = SlotState::Loading { key };
        Ok(())
    }

    fn record_late(&mut self) -> R<(), B> {
        for slot in std::mem::take(&mut self.late_record) {
            Self::be(self.b.record(&self.sh.ready[slot], &self.copy))?;
        }
        Ok(())
    }

    /// Enfileira as cópias guardadas para vítimas da camada `layer` (o host viu o `seq` que a GPU grava depois dos
    /// experts dela).
    fn release_held(&mut self, layer: usize, record: bool) -> R<(), B> {
        let mut rest = Vec::new();
        for slot in std::mem::take(&mut self.held) {
            match self.slots[slot] {
                SlotState::Held { key, after } if held_due(after, layer) => self.enqueue_copy(slot, key, record)?,
                _ => rest.push(slot),
            }
        }
        self.held = rest;
        Ok(())
    }

    fn table(&self) -> R<Vec<u32>, B> {
        let mut tb = TableBuilder::new(self.key_slot.len());
        for (i, &st) in self.slots.iter().enumerate() {
            if let SlotState::Resident { .. } = st {
                tb.insert(i, st).map_err(|slot| Error::LoadingInTable { slot })?;
            }
        }
        Ok(tb.finish())
    }

    /// Escreve a tabela no lado do buffer pinned do token `t` (esperando a principal devolver o lado, se preciso) e
    /// monta o `Published`.
    fn publish(&mut self, t: u32, table: &[u32], ready: Vec<usize>) -> R<Published, B> {
        let idx = (t % 2) as usize;
        while self.sides[idx].is_none() {
            if let Some((i, side)) = self.sh.returned.try_recv() {
                self.sides[i] = Some(side);
                continue;
            }
            if self.sh.cancel.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            Self::be(self.b.relax())?;
        }
        let side = self.sides[idx].take().expect("lado presente");
        for (a, &v) in side.iter().zip(table) {
            a.store(v, Ordering::Relaxed);
        }
        Ok(Published { origin: self.sh.id, gen: t, side_idx: idx, side, ready, stats: self.stats.clone() })
    }

    /// Antes do primeiro token: copia o `prefill`, espera a stream de cópia e publica a tabela do primeiro token.
    pub(crate) fn start(&mut self) -> R<TokenEpoch, B> {
        let t = self.cfg.first_token;
        self.sh.cur_t.store(t, Ordering::Release);
        let prefill = self.cfg.prefill.clone();
        if prefill.len() > self.slots.len() {
            return Err(Error::Config("prefill maior que os slots"));
        }
        for (slot, &key) in prefill.iter().enumerate() {
            self.enqueue_copy(slot, key, true)?;
            self.key_slot[key as usize] = slot as u32;
        }
        Self::be(self.b.sync(&self.copy))?;
        let mut ready = Vec::new();
        for (slot, &key) in prefill.iter().enumerate() {
            self.slots[slot] = SlotState::Resident { key };
            self.version[key as usize] += 1;
            ready.push(slot);
        }
        let table = self.table()?;
        let p = self.publish(t, &table, ready)?;
        self.sh.published.send(p);
        Ok(TokenEpoch { t, table })
    }

    /// Espera `seq[l] >= t`, solta as cópias guardadas da camada anterior e lê ids e `x` publicados pela GPU.
    pub fn serve_layer<'t>(&mut self, epoch: &'t TokenEpoch, l: usize) -> R<LayerView<'t>, B> {
        let t = epoch.t;
        let (ids, w, x) = self.wait_layer(l, t)?;
        if let Some(k) = release_layer(l, self.cfg.is(Mutant::HeldOneLayerEarly)) {
            self.release_held(k, true)?;
        }
        let io = self.sh.io[l].clone();
        let keys: Vec<u32> = ids.iter().map(|&e| self.cfg.key(l, e)).collect();
        for &k in &keys {
            self.freq.touch(k as usize, self.step);
        }
        self.step += 1;
        let mut misses = Vec::new();
        for &k in &keys {
            self.stats.needed += 1;
            if epoch.lookup(k).is_some() {
                self.stats.hits += 1;
                let s = self.key_slot[k as usize];
                if s != NO_SLOT && self.slots[s as usize] == (SlotState::Resident { key: k }) {
                    self.version[k as usize] += 1;
                }
            } else {
                self.stats.cpu += 1;
                if self.key_slot[k as usize] == NO_SLOT {
                    misses.push(k);
                }
            }
        }
        let flag = self.sh.flag[l].clone();
        Ok(LayerView { epoch, layer: l, keys, misses, w, x, io, flag, flag_first: self.cfg.is(Mutant::FlagBeforeY) })
    }

    pub fn pin_layer(&self, served: &Served<'_>) -> Pins {
        Pins { layer: served.layer, keys: served.keys.clone() }
    }

    fn pending(&self) -> R<usize, B> {
        let mut n = 0;
        for (i, st) in self.slots.iter().enumerate() {
            n += match (self.cfg.admit, st) {
                (_, SlotState::Held { .. }) => 1,
                (AdmitMode::TokenBarrier, SlotState::Loading { .. }) => 1,
                (AdmitMode::Event, SlotState::Loading { .. }) => !Self::be(self.b.query(&self.sh.ready[i]))? as usize,
                _ => 0,
            };
        }
        Ok(n)
    }

    /// O slot para admitir `candidate` depois da resposta da camada, ou `None` (recusa contada em `dropped`): backlog
    /// cheio, orçamento zero, ou a vítima de menor `(pontuação, versão, chave)` fora dos pins não tem pontuação menor que
    /// a do candidato.
    pub fn victim(&mut self, served: &Served<'_>, pins: &Pins, candidate: u32) -> R<Option<Victim>, B> {
        if pins.layer != served.layer {
            return Err(Error::Config("Pins de outra camada"));
        }
        if !served.misses().contains(&candidate) {
            return Err(Error::Config("candidato fora das faltas da camada"));
        }
        if self.key_slot[candidate as usize] != NO_SLOT {
            return Err(Error::Config("candidato que já tem slot"));
        }
        if self.slots.is_empty() || self.pending()? >= self.cfg.backlog {
            self.stats.dropped += 1;
            return Ok(None);
        }
        if let Some(slot) = self.slots.iter().position(|s| *s == SlotState::Free) {
            return Ok(Some(Victim { slot, evicted: None }));
        }
        let mut best: Option<(f64, u64, u32, usize)> = None;
        for (i, st) in self.slots.iter().enumerate() {
            let SlotState::Resident { key } = *st else { continue };
            if pins.keys.contains(&key) {
                continue;
            }
            let c = (self.freq.score(key as usize), self.version[key as usize], key, i);
            let better = match best {
                None => true,
                Some(b) => c.0 < b.0 || (c.0 == b.0 && (c.1, c.2) < (b.1, b.2)),
            };
            if better {
                best = Some(c);
            }
        }
        match best {
            Some((score, _, key, slot)) if score < self.freq.score(candidate as usize) => {
                self.key_slot[key as usize] = NO_SLOT;
                self.slots[slot] = SlotState::Free;
                Ok(Some(Victim { slot, evicted: Some(key) }))
            }
            _ => {
                self.stats.dropped += 1;
                Ok(None)
            }
        }
    }

    /// A cópia de `key` para o slot da vítima. Se a vítima é de uma camada já passada no token (ou o slot estava livre),
    /// enfileira na hora: o host viu `seq[l]`, gravado depois dos experts dela. Senão guarda a cópia, que
    /// `serve_layer(l' + 1)` (ou `end_token`, para a última camada) enfileira.
    pub fn copy_after_reader(&mut self, v: Victim, key: u32, served: &Served<'_>) -> R<(), B> {
        if !served.misses().contains(&key) {
            return Err(Error::Config("chave fora das faltas da camada"));
        }
        if self.key_slot[key as usize] != NO_SLOT {
            return Err(Error::Config("chave que já tem slot"));
        }
        if self.slots[v.slot] != SlotState::Free {
            return Err(Error::Config("slot da vítima já usado"));
        }
        self.stats.copies += 1;
        self.key_slot[key as usize] = v.slot as u32;
        match v.evicted.map(|e| self.cfg.layer_of(e)) {
            Some(after) if after >= served.layer && !self.cfg.is(Mutant::CopyAtVictim) => {
                self.slots[v.slot] = SlotState::Held { key, after };
                self.held.push(v.slot);
                self.stats.deferred += 1;
                Ok(())
            }
            _ => self.enqueue_copy(v.slot, key, true),
        }
    }

    /// A virada: espera `seq[L]`, solta as cópias da última camada, faz o `settle` por consulta (no modo com barreira,
    /// depois de sincronizar a stream de cópia), monta a tabela do token seguinte e manda o `Published`.
    pub fn end_token(&mut self, epoch: TokenEpoch) -> R<TokenEpoch, B> {
        let l_end = self.cfg.layers;
        self.wait_seq(l_end, epoch.t)?;
        let late = self.cfg.is(Mutant::PublishedBeforeRecord) || self.cfg.is(Mutant::QueryBeforeRecord);
        if !self.cfg.is(Mutant::HeldOneLayerEarly) {
            self.release_held(l_end - 1, !late)?;
        }
        if !self.held.is_empty() {
            return Err(Error::Config("cópia guardada que não saiu no token"));
        }
        if self.cfg.admit == AdmitMode::TokenBarrier {
            Self::be(self.b.sync(&self.copy))?;
        }
        let mut ready = Vec::new();
        for i in 0..self.slots.len() {
            let SlotState::Loading { key } = self.slots[i] else { continue };
            let done = self.cfg.is(Mutant::NoReadyProtection) || Self::be(self.b.query(&self.sh.ready[i]))?;
            if done {
                self.slots[i] = SlotState::Resident { key };
                self.version[key as usize] += 1;
                ready.push(i);
            }
        }
        if self.cfg.is(Mutant::QueryBeforeRecord) {
            self.record_late()?;
        }
        let t = epoch.t.wrapping_add(1);
        let table = self.table()?;
        let p = self.publish(t, &table, ready)?;
        self.sh.cur_t.store(t, Ordering::Release);
        self.sh.published.send(p);
        if self.cfg.is(Mutant::PublishedBeforeRecord) {
            self.record_late()?;
        }
        Ok(TokenEpoch { t, table })
    }
}

/// Na saída da linha de serviço, por qualquer caminho: se ela falhou (erro que não é o cancelamento, ou `panic`),
/// solta todas as flags, marca o cancelamento e envenena; sempre drena a própria stream de cópia, depois da última cópia
/// que enfileirou (passo 6 do A0b).
struct ExitGuard<B: Backend> {
    b: Arc<B>,
    sh: Arc<Shared<B>>,
    copy: Arc<B::Stream>,
    failed: bool,
    drain: bool,
}

impl<B: Backend> Drop for ExitGuard<B> {
    fn drop(&mut self) {
        let flags: Vec<&AtomicU32> = self.sh.flag.iter().map(|f| &***f).collect();
        exit_service(self.failed || std::thread::panicking(), self.sh.cur_t.load(Ordering::Acquire), &flags, &self.sh.poison,
                     &self.sh.cancel, self.drain, ExitSkip::None, || {
            let _ = self.b.sync(&self.copy);
        });
    }
}

/// Solta todas as flags com o valor de soltura do token `t`: nenhuma espera da GPU por uma resposta fica presa.
pub(crate) fn release_flags(t: u32, flags: &[&AtomicU32]) {
    let rv = release_value(t);
    for f in flags {
        f.store(rv, Ordering::Release);
    }
}

fn publish_failure(skip: ExitSkip, poison: &AtomicBool, cancel: &AtomicBool) {
    if skip != ExitSkip::Poison {
        poison.store(true, Ordering::Release);
    }
    if skip != ExitSkip::Cancel {
        cancel.store(true, Ordering::Release);
    }
}

/// Uma ação do ramo de falha que o modelo do `loom` omite para mostrar que ela pesa (produção sempre passa `None`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum ExitSkip {
    None,
    #[cfg_attr(not(motor_loom), allow(dead_code))]
    Release,
    #[cfg_attr(not(motor_loom), allow(dead_code))]
    Poison,
    #[cfg_attr(not(motor_loom), allow(dead_code))]
    Cancel,
    /// A ordem inversa: soltar as flags antes de publicar `poison` e `cancel`.
    #[cfg_attr(not(motor_loom), allow(dead_code))]
    StateLast,
}

/// A saída da linha de serviço (a composição que o `ExitGuard` executa e o modelo do `loom` chama): se falhou, solta as
/// flags, envenena e cancela; sempre que `drain`, drena a própria stream de cópia.
pub(crate) fn exit_service(failed: bool, t: u32, flags: &[&AtomicU32], poison: &AtomicBool, cancel: &AtomicBool, drain: bool,
                           skip: ExitSkip, sync_copy: impl FnOnce()) {
    if failed {
        // O estado de falha vem antes da soltura: a GPU só termina o token depois de ver a flag, e a principal, ao ver a
        // GPU terminar, tem de já enxergar `poison` e `cancel` (rodada 5, rev-1). `StateLast` é o mutante da ordem inversa.
        if skip != ExitSkip::StateLast {
            publish_failure(skip, poison, cancel);
        }
        if skip != ExitSkip::Release {
            release_flags(t, flags);
        }
        if skip == ExitSkip::StateLast {
            publish_failure(skip, poison, cancel);
        }
    }
    if drain {
        sync_copy();
    }
}

struct EndGuard<B: Backend>(Arc<B>);

impl<B: Backend> Drop for EndGuard<B> {
    fn drop(&mut self) {
        self.0.on_thread_end(Role::Service);
    }
}

fn run<B: Backend, C: HostExperts>(svc: &mut Service<B>, cpu: &mut C) -> R<(), B> {
    let mut epoch = svc.start()?;
    let (layers, width) = (svc.cfg.layers, svc.cfg.width);
    loop {
        for l in 0..layers {
            let view = svc.serve_layer(&epoch, l)?;
            let mut y = vec![0u32; width];
            let ck = view.cpu_keys();
            let has_cpu = !ck.is_empty();
            cpu.compute(l, &ck, &view.cpu_weights(), view.x(), &mut y).map_err(|e| Error::Cpu(format!("{e:?}")))?;
            let served = answer(view, &y);
            crate::diag::answered();
            if has_cpu {
                crate::diag::inject(crate::diag::Site::Livro);
            }
            let pins = svc.pin_layer(&served);
            for &key in served.misses() {
                if let Some(v) = svc.victim(&served, &pins, key)? {
                    svc.copy_after_reader(v, key, &served)?;
                }
            }
            crate::diag::layer_end(has_cpu);
            if has_cpu {
                crate::diag::inject(crate::diag::Site::AntesDoWait);
            }
        }
        epoch = svc.end_token(epoch)?;
        crate::diag::token_end();
    }
}

/// O corpo da thread de serviço. Volta com `Err(Cancelled)` no encerramento normal.
pub(crate) fn service_main<B: Backend, C: HostExperts>(mut svc: Service<B>, mut cpu: C) -> R<(), B> {
    let b = svc.b.clone();
    b.on_thread_start(Role::Service);
    let _end = EndGuard(b.clone());
    let mut guard = ExitGuard { b, sh: svc.sh.clone(), copy: svc.copy.clone(), failed: true,
                                drain: !svc.cfg.is(Mutant::OldDropOrder) };
    let r = run(&mut svc, &mut cpu);
    guard.failed = !matches!(r, Err(Error::Cancelled));
    drop(guard);
    r
}

#[cfg(all(test, not(motor_loom)))]
mod candidato_tests {
    use super::*;
    use crate::motor::fake::{FakeCfg, FakeGpu};
    use crate::motor::sync::{AtomicBool, Chan};

    /// `victim` e `copy_after_reader` recusam uma chave que não está nas faltas da camada (achado P3-2b da rodada
    /// `lastro-rust-2`): uma chave residente ou de outra camada não pode ganhar um segundo slot.
    #[test]
    fn candidato_fora_das_faltas_e_recusado() {
        let cfg = Config { layers: 2, experts: 3, topk: 2, width: 2, slots: 3, backlog: 4, admit: AdmitMode::Event,
                           first_token: 1, prefill: vec![], mutant: None };
        let gpu = FakeGpu::new_executor(FakeCfg { layers: 2, experts: 3, topk: 2, width: 2, routing: vec![], fail_main_at: None,
                                                  copy_immediate: false });
        let parts = gpu.parts(&cfg);
        let sh = Arc::new(Shared {
            id: 1,
            seq: parts.seq.into_iter().map(Arc::new).collect(),
            flag: parts.flag.into_iter().map(Arc::new).collect(),
            io: parts.io.into_iter().map(Arc::new).collect(),
            bank: parts.bank,
            ready: parts.ready,
            cancel: AtomicBool::new(false),
            poison: AtomicBool::new(false),
            cur_t: AtomicU32::new(1),
            published: Chan::new(),
            returned: Chan::new(),
        });
        let mut svc = Service::new(gpu.clone(), sh, Arc::new(parts.copy), parts.weights, cfg, parts.sides);
        let epoch = TokenEpoch { t: 1, table: vec![NO_SLOT; 6] };
        let served = Served { epoch: &epoch, layer: 0, keys: vec![0, 1], misses: vec![0, 1, 2] };
        let pins = svc.pin_layer(&served);
        assert!(matches!(svc.victim(&served, &pins, 4), Err(Error::Config(_))), "chave de outra camada");
        assert!(matches!(svc.victim(&served, &pins, 3), Err(Error::Config(_))), "chave que não foi pedida");
        let v = svc.victim(&served, &pins, 0).unwrap().expect("há slot livre");
        assert!(matches!(svc.copy_after_reader(v, 5, &served), Err(Error::Config(_))));
        // a mesma falta admitida duas vezes no mesmo `Served` não ganha um segundo slot (rodada 3, rev-2)
        let v = svc.victim(&served, &pins, 0).unwrap().expect("há slot livre");
        let antes = svc.victim(&served, &pins, 0).unwrap().expect("há slot livre"); // obtido antes da primeira cópia
        svc.copy_after_reader(v, 0, &served).unwrap();
        assert!(matches!(svc.victim(&served, &pins, 0), Err(Error::Config(_))), "a chave 0 já tem slot");
        assert!(matches!(svc.copy_after_reader(antes, 0, &served), Err(Error::Config(_))), "a segunda cópia da chave 0");
        // dois `Victim` do mesmo slot livre, com chaves diferentes: a segunda cópia é recusada (rodada 5, rev-2)
        let a = svc.victim(&served, &pins, 1).unwrap().expect("há slot livre");
        let b = svc.victim(&served, &pins, 2).unwrap().expect("o mesmo slot ainda está livre");
        svc.copy_after_reader(a, 1, &served).unwrap();
        assert!(matches!(svc.copy_after_reader(b, 2, &served), Err(Error::Config(_))), "o slot da vítima já foi usado");
    }
}
