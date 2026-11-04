//! O backend falso (plano, A0b, "Backend falso por trait, com teste de propriedade"): uma GPU de mentira.
//!
//! - Cada slot guarda o id do expert e fica inválido enquanto a cópia roda (a cópia tem começo e fim, em dois passos da
//!   fila de cópia).
//! - Cada stream é uma fila FIFO.
//! - A espera por evento e a consulta capturam a última gravação feita antes da chamada; um evento nunca gravado responde
//!   completo, como o `cudaEventQuery`.
//! - O doorbell e as áreas das camadas são atômicos de verdade (`Mapped`), escritos e lidos pelas operações da fila de
//!   cálculo com a ordem do `cuStreamWriteValue32` (`Release`) e do `cuStreamWaitValue32` (`Acquire`).
//!
//! O conferidor anota violações em vez de entrar em `panic`:
//! - [`Violation::Launch`]: um lançamento leu um slot que não tem o expert que a tabela do token diz (reescrito, em cópia
//!   ou com cópia na fila);
//! - [`Violation::Table`]: a tabela publicada lista um slot que não está pronto (o teste de estado (a), visto da GPU);
//! - [`Violation::Y`]: a saída da CPU somada não é a das faltas daquela camada e daquele token (dado velho);
//! - [`Violation::FreedWithQueued`] e [`Violation::Unfinished`]: memória liberada, ou o executor parado, com operação
//!   na fila.
//!
//! Dois jeitos de andar as filas:
//! - escalonador determinístico ([`FakeGpu::new_sched`], só na build normal): as duas linhas do host e as duas filas
//!   avançam uma de cada vez, na ordem sorteada por uma semente ([`SchedCfg`]); toda chamada do backend e toda volta de
//!   espera é um ponto de troca. É o que o `proptest` usa;
//! - executor ([`FakeGpu::new_executor`]): uma thread que anda a fila de cálculo, e a cópia roda na hora em que é
//!   enfileirada. É o que o `loom` usa (a GPU como terceira thread), e um teste com threads de verdade.
use super::backend::{geq, Backend, CpuStep, HostExperts, LayerIo, Role};
use super::engine::Parts;
use super::mem::{AllocId, Device, Mapped, Pinned, Release};
use super::sync::{lock, thread, AtomicU32, Mutex, Ordering};
use super::{Config, NO_SLOT};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct FakeCfg {
    pub layers: usize,
    pub experts: usize,
    pub topk: usize,
    pub width: usize,
    /// `routing[i][l]`: os ids do top-k da camada `l` no `i`-ésimo token (circular).
    pub routing: Vec<Vec<Vec<u32>>>,
    /// A `n`-ésima operação enfileirada na stream de cálculo falha (a principal falha no meio da enfileiração).
    pub fail_main_at: Option<u64>,
    /// A cópia roda inteira na hora em que é enfileirada (o executor).
    pub copy_immediate: bool,
    /// O M máximo do passo (1 = por token, sem a palavra `m` nem a operação `BeginStep`: as filas são as de sempre). Com mais de 1,
    /// `routing[p]` é o roteamento da posição global `p` da sequência (circular) e `x` também depende da posição global.
    pub max_positions: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FakeError {
    /// O escalonador passou do orçamento de passos: algo travou.
    Stalled,
    /// Falha injetada (`fail_main_at`).
    Injected,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Violation {
    Launch { token: usize, pos: usize, layer: usize, key: u32, slot: u32, found: Option<u32>, copying: bool },
    Table { key: u32, slot: u32, found: Option<u32>, copying: bool, queued: bool },
    Y { token: usize, pos: usize, layer: usize, expected: Vec<u32>, got: Vec<u32> },
    FreedWithQueued { id: u64, ops: usize },
    Unfinished { ops: usize },
}

impl Violation {
    pub fn kind(&self) -> &'static str {
        match self {
            Violation::Launch { .. } => "lançamento",
            Violation::Table { .. } => "tabela",
            Violation::Y { .. } => "y",
            Violation::FreedWithQueued { .. } => "liberada com fila",
            Violation::Unfinished { .. } => "executor com fila",
        }
    }
}

pub struct FakeStream(usize);
pub struct FakeEvent(usize);
/// O banco de slots do backend falso (o conteúdo fica no estado da GPU de mentira).
pub struct FakeBank;

const COMPUTE: usize = 0;
const COPY: usize = 1;

enum Op {
    Copy { bank: AllocId, weights: AllocId, slot: usize, key: u32, started: bool },
    Record { ev: usize, n: u64 },
    WaitEvent { ev: usize, n: u64 },
    Upload { dst: AllocId, src_id: AllocId, src: Arc<[AtomicU32]> },
    BeginStep { pos: u64, m: usize },
    Route { layer: usize, io_id: AllocId, io: Arc<LayerIo> },
    Launch { layer: usize, table: AllocId, bank: AllocId, io_id: AllocId, io: Arc<LayerIo> },
    AddY { layer: usize, io_id: AllocId, io: Arc<LayerIo> },
    WriteValue { at_id: AllocId, at: Arc<AtomicU32>, v: u32 },
    WaitValue { at_id: AllocId, at: Arc<AtomicU32>, v: u32 },
}

impl Op {
    fn refs(&self, id: AllocId) -> bool {
        match self {
            Op::Copy { bank, weights, .. } => *bank == id || *weights == id,
            Op::Record { .. } | Op::WaitEvent { .. } | Op::BeginStep { .. } => false,
            Op::Upload { dst, src_id, .. } => *dst == id || *src_id == id,
            Op::Route { io_id, .. } | Op::AddY { io_id, .. } => *io_id == id,
            Op::Launch { table, bank, io_id, .. } => *table == id || *bank == id || *io_id == id,
            Op::WriteValue { at_id, .. } | Op::WaitValue { at_id, .. } => *at_id == id,
        }
    }
}

#[derive(Clone, Copy)]
struct Slot {
    key: Option<u32>,
    copying: bool,
}

struct State {
    q: [VecDeque<Op>; 2],
    /// Por evento: (gravações feitas, maior gravação concluída).
    ev: Vec<(u64, u64)>,
    bank: Vec<Slot>,
    tables: HashMap<AllocId, Vec<u32>>,
    next_id: u64,
    /// Envios de tabela já executados (o token corrente da GPU é `token - 1`).
    token: usize,
    /// A posição global da sequência e o tamanho do passo corrente da GPU (só com `max_positions > 1`; `BeginStep` os instala em ordem FIFO).
    pos: usize,
    m: usize,
    /// Só para teste: a palavra `m` escrita na camada indicada, no lugar do `m` do passo (um backend que erra entre camadas).
    m_word_override: Option<(usize, u32)>,
    /// `hits[camada][posição]`: as chaves lançadas na GPU naquela posição.
    hits: Vec<Vec<Vec<u32>>>,
    /// A fila de cálculo passou de uma flag com o valor de soltura: dali em diante o `y` não é conferido.
    released: bool,
    violations: Vec<Violation>,
    main_calls: u64,
    shutdown: bool,
}

/// A saída de mentira de um expert, igual na CPU de mentira e no conferidor.
pub fn expert_out(key: u32, x: u32) -> u32 {
    x.wrapping_mul(0x9E37_79B1).wrapping_add(key.wrapping_mul(40503)).rotate_left(key % 31 + 1)
}

/// O peso (inteiro pequeno, distinto por posição global, camada e posto no top-k) que o falso publica nas áreas de várias posições e que
/// entra na soma do conferidor e do `FakeCpu` ponderado: um peso trocado de posição ou de posto muda o `y`.
pub(crate) fn peso(gpos: usize, layer: usize, rank: usize) -> u32 {
    1 + ((gpos * 3 + layer * 5 + rank * 7) % 11) as u32
}

pub(crate) fn x_value(token: usize, layer: usize, j: usize) -> u32 {
    (((token as u32).wrapping_add(1)) << 16) ^ ((layer as u32) << 8) ^ (j as u32 + 1)
}

fn step(st: &mut State, cfg: &FakeCfg, q: usize) -> bool {
    let Some(mut op) = st.q[q].pop_front() else { return false };
    match &mut op {
        Op::Copy { slot, key, started, .. } => {
            if !*started && !cfg.copy_immediate {
                *started = true;
                st.bank[*slot] = Slot { key: None, copying: true };
                st.q[q].push_front(op);
                return true;
            }
            st.bank[*slot] = Slot { key: Some(*key), copying: false };
        }
        Op::Record { ev, n } => {
            let e = &mut st.ev[*ev];
            e.1 = e.1.max(*n);
        }
        Op::WaitEvent { ev, n } => {
            if st.ev[*ev].1 < *n {
                st.q[q].push_front(op);
                return false;
            }
        }
        Op::Upload { dst, src, .. } => {
            st.tables.insert(*dst, src.iter().map(|a| a.load(Ordering::Relaxed)).collect());
            st.token += 1;
        }
        Op::BeginStep { pos, m } => {
            st.pos = *pos as usize;
            st.m = *m;
        }
        Op::Route { layer, io, .. } => {
            // O caminho vem do formato da área recebida (a palavra `m`), não do teto do backend: um backend de capacidade 4 serve áreas de 1.
            if io.m.is_empty() {
                let tok = st.token.saturating_sub(1);
                let ids = &cfg.routing[tok % cfg.routing.len()][*layer];
                for (a, &e) in io.ids.iter().zip(ids) {
                    a.store(e, Ordering::Relaxed);
                }
                // Pesos do softmax: o backend falso publica pesos iguais (o motor ainda não os usa; entram com o A1.3).
                for a in io.w.iter() {
                    a.store((1.0f32 / io.w.len() as f32).to_bits(), Ordering::Relaxed);
                }
                for (j, a) in io.x.iter().enumerate() {
                    a.store(x_value(tok, *layer, j), Ordering::Relaxed);
                }
            } else {
                // Passo de `m` posições: ids, pesos e `x` por posição global, depois a palavra `m`. Só o prefixo ativo é escrito: a cauda
                // `m..max_positions` fica com o dado do passo anterior (a linha de serviço tem de fatiar por `m`).
                let (m, k, w) = (st.m, cfg.topk, cfg.width);
                for p in 0..m {
                    let gpos = st.pos + p;
                    let ids = &cfg.routing[gpos % cfg.routing.len()][*layer];
                    for (j, &e) in ids.iter().enumerate() {
                        io.ids[p * k + j].store(e, Ordering::Relaxed);
                        io.w[p * k + j].store(peso(gpos, *layer, j), Ordering::Relaxed);
                    }
                    for j in 0..w {
                        io.x[p * w + j].store(x_value(gpos, *layer, j), Ordering::Relaxed);
                    }
                }
                let word = match st.m_word_override {
                    Some((l, v)) if l == *layer => v,
                    _ => m as u32,
                };
                io.m[0].store(word, Ordering::Relaxed);
            }
        }
        Op::Launch { layer, table, io, .. } => {
            let (m, k) = (if io.m.is_empty() { 1 } else { st.m }, cfg.topk);
            let ids = LayerIo::load(&io.ids);
            let mut por_pos = vec![Vec::new(); cfg.max_positions];
            for p in 0..m {
                for &e in &ids[p * k..(p + 1) * k] {
                    let key = (*layer * cfg.experts) as u32 + e;
                    let s = st.tables.get(table).map_or(NO_SLOT, |t| t[key as usize]);
                    if s == NO_SLOT {
                        continue;
                    }
                    let sl = st.bank[s as usize];
                    if sl.key != Some(key) || sl.copying {
                        let token = st.token;
                        st.violations.push(Violation::Launch { token, pos: p, layer: *layer, key, slot: s, found: sl.key, copying: sl.copying });
                    }
                    por_pos[p].push(key);
                }
            }
            st.hits[*layer] = por_pos;
        }
        Op::AddY { layer, io, .. } => {
            if !st.released {
                let (m, k, w) = (if io.m.is_empty() { 1 } else { st.m }, cfg.topk, cfg.width);
                let (x, ids, got) = (LayerIo::load(&io.x), LayerIo::load(&io.ids), LayerIo::load(&io.y));
                let ponderado = !io.m.is_empty();
                for p in 0..m {
                    // As faltas da posição com o posto de cada uma; nas áreas de várias posições o peso entra na soma.
                    let cpu: Vec<(usize, u32)> = ids[p * k..(p + 1) * k].iter().enumerate().map(|(j, &e)| (j, (*layer * cfg.experts) as u32 + e))
                        .filter(|(_, key)| !st.hits[*layer][p].contains(key)).collect();
                    let gpos = st.pos + p;
                    let expected: Vec<u32> = x[p * w..(p + 1) * w].iter().map(|&xj| {
                        cpu.iter().fold(0u32, |a, &(j, key)| {
                            let o = expert_out(key, xj);
                            a.wrapping_add(if ponderado { peso(gpos, *layer, j).wrapping_mul(o) } else { o })
                        })
                    }).collect();
                    if expected != got[p * w..(p + 1) * w] {
                        let token = st.token;
                        st.violations.push(Violation::Y { token, pos: p, layer: *layer, expected, got: got[p * w..(p + 1) * w].to_vec() });
                    }
                }
            }
        }
        Op::WriteValue { at, v, .. } => at.store(*v, Ordering::Release),
        Op::WaitValue { at, v, .. } => {
            let cur = at.load(Ordering::Acquire);
            if !geq(cur, *v) {
                st.q[q].push_front(op);
                return false;
            }
            if cur != *v {
                st.released = true;
            }
        }
    }
    true
}

pub struct FakeGpu {
    st: Mutex<State>,
    cfg: FakeCfg,
    #[cfg(not(motor_loom))]
    sched: Option<sched::Sched>,
    exec: std::sync::Mutex<Option<thread::JoinHandle<()>>>,
}

impl FakeGpu {
    fn build(cfg: FakeCfg, #[cfg(not(motor_loom))] sched: Option<sched::Sched>) -> Self {
        let st = State {
            q: [VecDeque::new(), VecDeque::new()],
            ev: Vec::new(),
            bank: Vec::new(),
            tables: HashMap::new(),
            next_id: 0,
            token: 0,
            pos: 0,
            m: 1,
            m_word_override: None,
            hits: vec![vec![Vec::new(); cfg.max_positions]; cfg.layers],
            released: false,
            violations: Vec::new(),
            main_calls: 0,
            shutdown: false,
        };
        Self {
            st: Mutex::new(st),
            cfg,
            #[cfg(not(motor_loom))]
            sched,
            exec: std::sync::Mutex::new(None),
        }
    }

    /// GPU andada pelo escalonador determinístico; a thread que chama vira a linha principal.
    #[cfg(not(motor_loom))]
    pub fn new_sched(cfg: FakeCfg, s: SchedCfg) -> Arc<Self> {
        Arc::new(Self::build(cfg, Some(sched::Sched::new(s))))
    }

    /// GPU andada por uma thread executora (a fila de cálculo); a cópia roda na hora.
    pub fn new_executor(mut cfg: FakeCfg) -> Arc<Self> {
        cfg.copy_immediate = true;
        let g = Arc::new(Self::build(
            cfg,
            #[cfg(not(motor_loom))]
            None,
        ));
        let e = g.clone();
        let h = thread::spawn(move || loop {
            let mut st = lock(&e.st);
            if step(&mut st, &e.cfg, COMPUTE) {
                continue;
            }
            if st.shutdown {
                let ops = st.q[COMPUTE].len();
                if ops > 0 {
                    st.violations.push(Violation::Unfinished { ops });
                }
                break;
            }
            drop(st);
            yield_now();
        });
        *g.exec.lock().unwrap_or_else(|e| e.into_inner()) = Some(h);
        g
    }

    /// Só para teste: a camada `layer` passa a gravar `word` na palavra `m` (diferente da dos outros passos).
    pub fn set_m_word_override(&self, layer: usize, word: u32) {
        lock(&self.st).m_word_override = Some((layer, word));
    }

    /// Para o executor (depois de o motor cair).
    pub fn shutdown(&self) {
        lock(&self.st).shutdown = true;
        if let Some(h) = self.exec.lock().unwrap_or_else(|e| e.into_inner()).take() {
            h.join().expect("executor da GPU falsa");
        }
    }

    pub fn violations(&self) -> Vec<Violation> {
        lock(&self.st).violations.clone()
    }

    /// Se o escalonador passou do orçamento (algo travou).
    pub fn stalled(&self) -> bool {
        #[cfg(not(motor_loom))]
        if let Some(s) = &self.sched {
            return s.stalled();
        }
        false
    }

    /// Operações ainda nas filas (cálculo, cópia).
    pub fn queued(&self) -> (usize, usize) {
        let st = lock(&self.st);
        (st.q[COMPUTE].len(), st.q[COPY].len())
    }

    #[cfg(not(motor_loom))]
    fn step_queue(&self, q: usize) {
        let mut st = lock(&self.st);
        step(&mut st, &self.cfg, q);
    }

    fn point(&self) -> Result<(), FakeError> {
        #[cfg(not(motor_loom))]
        if let Some(s) = &self.sched {
            return s.yield_now(self);
        }
        Ok(())
    }

    fn new_id(&self) -> AllocId {
        let mut st = lock(&self.st);
        st.next_id += 1;
        AllocId(st.next_id)
    }

    fn owner(self: &Arc<Self>) -> Arc<dyn Release> {
        self.clone()
    }

    pub fn pinned_table(self: &Arc<Self>, len: usize) -> Pinned<[AtomicU32]> {
        let v: Arc<[AtomicU32]> = (0..len).map(|_| AtomicU32::new(NO_SLOT)).collect();
        Pinned::from_parts(v, self.new_id(), self.owner())
    }

    /// Os pesos de mentira: o expert `k` é o número `k`.
    pub fn weights(self: &Arc<Self>, keys: usize) -> Pinned<[u32]> {
        let v: Arc<[u32]> = (0..keys as u32).collect();
        Pinned::from_parts(v, self.new_id(), self.owner())
    }

    pub fn mapped_u32(self: &Arc<Self>, v: u32) -> Mapped<AtomicU32> {
        let id = self.new_id();
        Mapped::from_parts(Arc::new(AtomicU32::new(v)), id.0, id, self.owner())
    }

    pub fn mapped_io(self: &Arc<Self>, topk: usize, width: usize) -> Mapped<LayerIo> {
        self.mapped_io_positions(topk, width, 1)
    }

    pub fn mapped_io_positions(self: &Arc<Self>, topk: usize, width: usize, max_positions: usize) -> Mapped<LayerIo> {
        let id = self.new_id();
        Mapped::from_parts(Arc::new(LayerIo::with_positions(topk, width, max_positions)), id.0, id, self.owner())
    }

    pub fn device_table(self: &Arc<Self>, len: usize) -> Device<[u32]> {
        let id = self.new_id();
        lock(&self.st).tables.insert(id, vec![NO_SLOT; len]);
        Device::from_parts(id.0, id, self.owner())
    }

    pub fn bank(self: &Arc<Self>, slots: usize) -> Device<FakeBank> {
        let id = self.new_id();
        lock(&self.st).bank = vec![Slot { key: None, copying: false }; slots];
        Device::from_parts(id.0, id, self.owner())
    }

    pub fn event(&self) -> FakeEvent {
        let mut st = lock(&self.st);
        st.ev.push((0, 0));
        FakeEvent(st.ev.len() - 1)
    }

    /// As peças de um motor com a configuração `cfg` (o doorbell começa em `first_token − 1`).
    pub fn parts(self: &Arc<Self>, cfg: &Config) -> Parts<FakeGpu> {
        let keys = cfg.layers * cfg.experts;
        let init = cfg.first_token.wrapping_sub(1);
        Parts {
            compute: FakeStream(COMPUTE),
            copy: FakeStream(COPY),
            bank: self.bank(cfg.slots),
            table: self.device_table(keys),
            sides: [self.pinned_table(keys), self.pinned_table(keys)],
            weights: self.weights(keys),
            ready: (0..cfg.slots).map(|_| self.event()).collect(),
            seq: (0..=cfg.layers).map(|_| self.mapped_u32(init)).collect(),
            flag: (0..cfg.layers).map(|_| self.mapped_u32(init)).collect(),
            io: (0..cfg.layers).map(|_| self.mapped_io_positions(cfg.topk, cfg.width, cfg.max_positions)).collect(),
        }
    }

    fn enqueue(&self, q: usize, op: Op) -> Result<(), FakeError> {
        let mut st = lock(&self.st);
        if q == COMPUTE {
            st.main_calls += 1;
            if Some(st.main_calls) == self.cfg.fail_main_at {
                return Err(FakeError::Injected);
            }
        }
        st.q[q].push_back(op);
        if q == COPY && self.cfg.copy_immediate {
            while step(&mut st, &self.cfg, COPY) {}
        }
        Ok(())
    }
}

fn yield_now() {
    #[cfg(motor_loom)]
    loom::thread::yield_now();
    #[cfg(not(motor_loom))]
    std::thread::yield_now();
}

impl Release for FakeGpu {
    fn release(&self, id: AllocId) {
        let mut st = lock(&self.st);
        let ops = st.q.iter().flatten().filter(|op| op.refs(id)).count();
        if ops > 0 {
            st.violations.push(Violation::FreedWithQueued { id: id.0, ops });
        }
    }
}

impl Backend for FakeGpu {
    type Error = FakeError;
    type Stream = FakeStream;
    type Event = FakeEvent;
    type Weights = [u32];
    type Bank = FakeBank;

    fn copy_expert(&self, s: &FakeStream, dst: &Device<FakeBank>, slot: usize, src: &Pinned<[u32]>, key: u32) -> Result<(), FakeError> {
        self.point()?;
        self.enqueue(s.0, Op::Copy { bank: dst.id(), weights: src.id(), slot, key: src[key as usize], started: false })
    }

    fn record(&self, ev: &FakeEvent, s: &FakeStream) -> Result<(), FakeError> {
        self.point()?;
        let n = {
            let mut st = lock(&self.st);
            st.ev[ev.0].0 += 1;
            st.ev[ev.0].0
        };
        self.enqueue(s.0, Op::Record { ev: ev.0, n })
    }

    fn wait_event(&self, s: &FakeStream, ev: &FakeEvent) -> Result<(), FakeError> {
        self.point()?;
        let n = lock(&self.st).ev[ev.0].0;
        self.enqueue(s.0, Op::WaitEvent { ev: ev.0, n })
    }

    fn query(&self, ev: &FakeEvent) -> Result<bool, FakeError> {
        self.point()?;
        let (rec, done) = lock(&self.st).ev[ev.0];
        Ok(done >= rec)
    }

    fn upload_table(&self, s: &FakeStream, dst: &Device<[u32]>, src: &Pinned<[AtomicU32]>) -> Result<(), FakeError> {
        self.point()?;
        {
            let mut st = lock(&self.st);
            for (key, a) in src.iter().enumerate() {
                let slot = a.load(Ordering::Relaxed);
                if slot == NO_SLOT {
                    continue;
                }
                let sl = st.bank[slot as usize];
                let queued = st.q[COPY].iter().any(|op| matches!(op, Op::Copy { slot: s, .. } if *s == slot as usize));
                if sl.key != Some(key as u32) || sl.copying || queued {
                    st.violations.push(Violation::Table { key: key as u32, slot, found: sl.key, copying: sl.copying, queued });
                }
            }
        }
        self.enqueue(s.0, Op::Upload { dst: dst.id(), src_id: src.id(), src: src.shared() })
    }

    fn max_positions(&self) -> usize {
        self.cfg.max_positions
    }

    fn begin_step(&self, s: &FakeStream, _t: u32, pos: u64, m: usize) -> Result<(), FakeError> {
        self.point()?;
        self.enqueue(s.0, Op::BeginStep { pos, m })
    }

    fn route(&self, s: &FakeStream, layer: usize, io: &Mapped<LayerIo>) -> Result<(), FakeError> {
        self.point()?;
        self.enqueue(s.0, Op::Route { layer, io_id: io.id(), io: io.shared() })
    }

    fn launch_hits(&self, s: &FakeStream, layer: usize, table: &Device<[u32]>, bank: &Device<FakeBank>, io: &Mapped<LayerIo>)
        -> Result<(), FakeError> {
        self.point()?;
        self.enqueue(s.0, Op::Launch { layer, table: table.id(), bank: bank.id(), io_id: io.id(), io: io.shared() })
    }

    fn add_y(&self, s: &FakeStream, layer: usize, io: &Mapped<LayerIo>) -> Result<(), FakeError> {
        self.point()?;
        self.enqueue(s.0, Op::AddY { layer, io_id: io.id(), io: io.shared() })
    }

    fn write_value(&self, s: &FakeStream, at: &Mapped<AtomicU32>, v: u32) -> Result<(), FakeError> {
        self.point()?;
        self.enqueue(s.0, Op::WriteValue { at_id: at.id(), at: at.shared(), v })
    }

    fn wait_value(&self, s: &FakeStream, at: &Mapped<AtomicU32>, v: u32) -> Result<(), FakeError> {
        self.point()?;
        self.enqueue(s.0, Op::WaitValue { at_id: at.id(), at: at.shared(), v })
    }

    fn sync(&self, s: &FakeStream) -> Result<(), FakeError> {
        loop {
            if lock(&self.st).q[s.0].is_empty() {
                return Ok(());
            }
            self.relax()?;
        }
    }

    fn relax(&self) -> Result<(), FakeError> {
        #[cfg(not(motor_loom))]
        if let Some(s) = &self.sched {
            return s.yield_now(self);
        }
        yield_now();
        Ok(())
    }

    fn on_thread_start(&self, role: Role) {
        #[cfg(not(motor_loom))]
        if let Some(s) = &self.sched {
            s.start(role);
        }
        let _ = role;
    }

    fn on_thread_end(&self, role: Role) {
        #[cfg(not(motor_loom))]
        if let Some(s) = &self.sched {
            s.end(role, self);
        }
        let _ = role;
    }
}

/// A CPU de mentira: `y[j] = Σ expert_out(chave, x[j])` sobre as faltas, ou, com `weighted`, `Σ peso · expert_out(chave, x[j])` (os pesos
/// inteiros que o falso publica nas áreas de várias posições). Falha ou entra em `panic` na `n`-ésima camada calculada, se pedido: uma
/// chamada por camada, qualquer que seja o caminho (`compute` ou o `compute_step` em lote, que ela sobrescreve).
#[derive(Default)]
pub struct FakeCpu {
    pub(crate) calls: usize,
    pub fail_at: Option<usize>,
    pub panic_at: Option<usize>,
    pub weighted: bool,
    /// Mutante de teste: o lote usa o peso da primeira posição do expert em todas as posições dele.
    pub peso_errado: bool,
}

impl FakeCpu {
    fn tick(&mut self, layer: usize) -> Result<(), String> {
        let c = self.calls;
        self.calls += 1;
        if self.panic_at == Some(c) {
            panic!("panic injetado na CPU de mentira (camada {layer}, chamada {c})");
        }
        if self.fail_at == Some(c) {
            return Err(format!("falha injetada na CPU de mentira (camada {layer}, chamada {c})"));
        }
        Ok(())
    }

    fn termo(&self, key: u32, w: u32, xj: u32) -> u32 {
        let o = expert_out(key, xj);
        if self.weighted { w.wrapping_mul(o) } else { o }
    }
}

impl HostExperts for FakeCpu {
    type Error = String;

    fn compute(&mut self, layer: usize, keys: &[u32], weights: &[u32], x: &[u32], y: &mut [u32]) -> Result<(), String> {
        self.tick(layer)?;
        for (yj, &xj) in y.iter_mut().zip(x) {
            *yj = keys.iter().zip(weights).fold(0u32, |a, (&k, &w)| a.wrapping_add(self.termo(k, w, xj)));
        }
        Ok(())
    }

    /// Em lote, por expert: cada grupo soma `peso · expert(chave, x)` na posição de cada pedido. Outra ordem de soma que a do caminho por
    /// posição (a soma em `u32` com volta é comutativa, então o resultado é o mesmo).
    fn compute_step(&mut self, layer: usize, step: &CpuStep, x: &[u32], y: &mut [u32]) -> Result<(), String> {
        self.tick(layer)?;
        y.fill(0);
        let w = step.width;
        for g in &step.groups {
            for (i, &p) in g.positions.iter().enumerate() {
                let peso = if self.peso_errado { g.weights[0] } else { g.weights[i] };
                for j in 0..w {
                    let t = self.termo(g.key, peso, x[p * w + j]);
                    y[p * w + j] = y[p * w + j].wrapping_add(t);
                }
            }
        }
        Ok(())
    }
}

#[cfg(not(motor_loom))]
pub use sched::SchedCfg;

#[cfg(not(motor_loom))]
mod sched {
    //! O escalonador determinístico: só um ator anda por vez (a linha principal, a de serviço, a fila de cálculo, a de
    //! cópia), na ordem de um gerador com semente e pesos por ator. As linhas do host trocam de vez em cada chamada do
    //! backend e em cada volta de espera; as filas andam uma operação por escolha.
    use super::{FakeError, FakeGpu, Role};
    use std::cell::Cell;
    use std::sync::{Condvar, Mutex, MutexGuard};

    #[derive(Clone, Copy, Debug)]
    pub struct SchedCfg {
        pub seed: u64,
        /// Pesos de escolha: principal, serviço, fila de cálculo, fila de cópia (todos >= 1).
        pub weights: [u32; 4],
        /// Escolhas até declarar travamento.
        pub budget: u64,
    }

    struct St {
        turn: usize,
        alive: [bool; 2],
        rng: u64,
        w: [u32; 4],
        steps: u64,
        budget: u64,
        stalled: bool,
    }

    impl St {
        fn pick(&mut self) -> usize {
            self.rng ^= self.rng << 13;
            self.rng ^= self.rng >> 7;
            self.rng ^= self.rng << 17;
            let total: u32 = self.w.iter().sum();
            let mut r = (self.rng % total as u64) as u32;
            for (i, &w) in self.w.iter().enumerate() {
                if r < w {
                    return i;
                }
                r -= w;
            }
            3
        }
    }

    thread_local! {
        static ACTOR: Cell<usize> = const { Cell::new(usize::MAX) };
    }

    pub(crate) struct Sched {
        m: Mutex<St>,
        cv: Condvar,
    }

    fn lk(m: &Mutex<St>) -> MutexGuard<'_, St> {
        m.lock().unwrap_or_else(|e| e.into_inner())
    }

    impl Sched {
        pub(crate) fn new(c: SchedCfg) -> Self {
            assert!(c.weights.iter().all(|&w| w >= 1), "todo ator precisa de peso");
            ACTOR.with(|a| a.set(0));
            let st = St { turn: 0, alive: [true, true], rng: c.seed | 1, w: c.weights, steps: 0, budget: c.budget, stalled: false };
            Self { m: Mutex::new(st), cv: Condvar::new() }
        }

        pub(crate) fn stalled(&self) -> bool {
            lk(&self.m).stalled
        }

        /// Escolhe o próximo ator do host vivo, andando as filas quando elas são escolhidas.
        fn next_host(&self, st: &mut St, gpu: &FakeGpu) -> Option<usize> {
            loop {
                st.steps += 1;
                if st.steps > st.budget {
                    st.stalled = true;
                    self.cv.notify_all();
                    return None;
                }
                let a = st.pick();
                if a >= 2 {
                    gpu.step_queue(a - 2);
                } else if st.alive[a] {
                    return Some(a);
                }
            }
        }

        pub(crate) fn yield_now(&self, gpu: &FakeGpu) -> Result<(), FakeError> {
            let me = ACTOR.with(|a| a.get());
            if me > 1 {
                return Ok(());
            }
            let mut st = lk(&self.m);
            if st.stalled {
                return Err(FakeError::Stalled);
            }
            let Some(a) = self.next_host(&mut st, gpu) else { return Err(FakeError::Stalled) };
            if a == me {
                return Ok(());
            }
            st.turn = a;
            self.cv.notify_all();
            while st.turn != me && !st.stalled {
                st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
            if st.stalled {
                Err(FakeError::Stalled)
            } else {
                Ok(())
            }
        }

        pub(crate) fn start(&self, role: Role) {
            assert_eq!(role, Role::Service, "só a linha de serviço tem thread própria");
            ACTOR.with(|a| a.set(1));
            let mut st = lk(&self.m);
            st.alive[1] = true;
            while st.turn != 1 && !st.stalled {
                st = self.cv.wait(st).unwrap_or_else(|e| e.into_inner());
            }
        }

        pub(crate) fn end(&self, _role: Role, gpu: &FakeGpu) {
            ACTOR.with(|a| a.set(usize::MAX));
            let mut st = lk(&self.m);
            st.alive[1] = false;
            if st.turn == 1 && !st.stalled {
                if let Some(a) = self.next_host(&mut st, gpu) {
                    st.turn = a;
                }
                self.cv.notify_all();
            }
        }
    }
}
