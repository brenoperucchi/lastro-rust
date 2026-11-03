//! A linha principal: dona da stream de cálculo, a única que enfileira nela (plano, A0b, passos 2, 3 e 6). Só vê um
//! [`Published`] por token; enfileirar um token pede o `Published` dele, e um de outra geração é recusado.
//!
//! O `Drop` (e a falha da principal no meio da enfileiração) segue a ordem do passo 6: cancelamento, flags com o valor de
//! soltura, `join` da linha de serviço (que já drenou a stream de cópia), sincronização da stream de cálculo, e só então a
//! liberação da memória (os campos caem depois do corpo do `Drop`).
use super::backend::{Backend, HostExperts, LayerIo};
use super::line::{release_flags, service_main, Service};
use super::mem::{Device, Mapped, Pinned};
use super::sync::{thread, AtomicBool, AtomicU32, Chan, Ordering};
use super::{Config, Error, Mutant, Stats};
use std::sync::Arc;

type R<T, B> = Result<T, Error<<B as Backend>::Error>>;

/// A tabela do token seguinte, mandada pela linha de serviço na virada: o lado do buffer pinned com a tabela, a geração e
/// os slots recém-admitidos (cujo `ready` a stream de cálculo espera).
pub struct Published {
    /// O motor de origem: um `Published` de outro motor é recusado mesmo com a geração igual.
    pub(crate) origin: u64,
    pub(crate) gen: u32,
    pub(crate) side_idx: usize,
    pub(crate) side: Pinned<[AtomicU32]>,
    pub(crate) ready: Vec<usize>,
    pub(crate) stats: Stats,
}

impl Published {
    pub fn generation(&self) -> u32 {
        self.gen
    }

    pub fn ready_slots(&self) -> &[usize] {
        &self.ready
    }

    /// Os contadores da linha de serviço até o fim do token anterior.
    pub fn stats(&self) -> &Stats {
        &self.stats
    }
}

/// O que as duas linhas compartilham: o doorbell, as áreas das camadas, o banco de slots, os eventos `ready`, o
/// cancelamento e os dois canais.
pub(crate) struct Shared<B: Backend> {
    /// A identidade deste motor (copiada para cada `Published` que a linha de serviço cria).
    pub id: u64,
    pub seq: Vec<Arc<Mapped<AtomicU32>>>,
    pub flag: Vec<Arc<Mapped<AtomicU32>>>,
    pub io: Vec<Arc<Mapped<LayerIo>>>,
    pub bank: Device<B::Bank>,
    pub ready: Vec<B::Event>,
    pub cancel: AtomicBool,
    pub poison: AtomicBool,
    /// O token corrente da linha de serviço (para o valor de soltura da guarda dela).
    pub cur_t: AtomicU32,
    pub published: Chan<Published>,
    /// Os lados do buffer da tabela devolvidos pela principal, com o índice.
    pub returned: Chan<(usize, Pinned<[AtomicU32]>)>,
}

/// As peças que o backend cria para um motor.
pub struct Parts<B: Backend> {
    pub compute: B::Stream,
    pub copy: B::Stream,
    pub bank: Device<B::Bank>,
    pub table: Device<[u32]>,
    pub sides: [Pinned<[AtomicU32]>; 2],
    pub weights: Pinned<B::Weights>,
    /// Um evento `ready` por slot, criado uma vez.
    pub ready: Vec<B::Event>,
    /// `L + 1` sequências (a última marca o fim do token).
    pub seq: Vec<Mapped<AtomicU32>>,
    pub flag: Vec<Mapped<AtomicU32>>,
    pub io: Vec<Mapped<LayerIo>>,
}

pub struct Engine<B: Backend> {
    b: Arc<B>,
    sh: Arc<Shared<B>>,
    compute: B::Stream,
    /// Só o mutante da ordem antiga do `Drop` a usa; a linha de serviço é a única que enfileira nela.
    copy: Arc<B::Stream>,
    table: Device<[u32]>,
    cfg: Config,
    service: Option<thread::JoinHandle<R<(), B>>>,
    #[cfg_attr(motor_loom, allow(dead_code))]
    done: Arc<AtomicBool>,
    service_result: Option<R<(), B>>,
    /// O próximo token a enfileirar (ou o enfileirado, até o `finish_token`).
    t: u32,
    current: Option<(usize, Pinned<[AtomicU32]>)>,
    closed: bool,
}

impl<B: Backend> Engine<B> {
    pub fn new<C: HostExperts>(b: Arc<B>, parts: Parts<B>, cfg: Config, cpu: C) -> R<Self, B> {
        let keys = cfg.layers * cfg.experts;
        let ok = cfg.layers > 0
            && cfg.topk > 0
            && cfg.topk <= cfg.experts
            && parts.seq.len() == cfg.layers + 1
            && parts.flag.len() == cfg.layers
            && parts.io.len() == cfg.layers
            && parts.ready.len() == cfg.slots
            && parts.sides.iter().all(|s| s.len() == keys)
            && cfg.prefill.iter().all(|&k| (k as usize) < keys);
        if !ok {
            return Err(Error::Config("peças e configuração não batem"));
        }
        if cfg.prefill.len() > cfg.slots {
            return Err(Error::Config("prefill maior que os slots"));
        }
        let mut vistas = vec![false; keys];
        for &k in &cfg.prefill {
            if std::mem::replace(&mut vistas[k as usize], true) {
                return Err(Error::Config("prefill com chave repetida"));
            }
        }
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        let sh = Arc::new(Shared {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            seq: parts.seq.into_iter().map(Arc::new).collect(),
            flag: parts.flag.into_iter().map(Arc::new).collect(),
            io: parts.io.into_iter().map(Arc::new).collect(),
            bank: parts.bank,
            ready: parts.ready,
            cancel: AtomicBool::new(false),
            poison: AtomicBool::new(false),
            cur_t: AtomicU32::new(cfg.first_token),
            published: Chan::new(),
            returned: Chan::new(),
        });
        let copy = Arc::new(parts.copy);
        let svc = Service::new(b.clone(), sh.clone(), copy.clone(), parts.weights, cfg.clone(), parts.sides);
        let done = Arc::new(AtomicBool::new(false));
        let d = done.clone();
        // Nome da thread só para a afinidade efetiva por papel ser legível (A1-DIAG10); o comportamento não muda.
        let service = thread::Builder::new()
            .name("moe-servico".into())
            .spawn(move || {
                // `done` sobe também quando a linha de serviço sai por `panic`; senão o `join` girava para sempre.
                let _done = DoneOnExit(d);
                // A1w: quem chama o Pool é fixado onde nasce, e não na primeira `run` (sem experts na CPU, como no motor_cheio, ela nunca roda).
                if let Some(p) = crate::affinity::compute_plan() {
                    crate::affinity::set_current(&[p[0]]).expect("afinidade da linha de serviço");
                }
                service_main(svc, cpu)
            })
            .expect("thread da linha de serviço");
        Ok(Self { b, sh, compute: parts.compute, copy, table: parts.table, t: cfg.first_token, cfg, service: Some(service),
                  done, service_result: None, current: None, closed: false })
    }

    /// Espera o `Published` do próximo token.
    pub fn next_published(&mut self) -> R<Published, B> {
        loop {
            if let Some(p) = self.sh.published.try_recv() {
                return Ok(p);
            }
            if self.sh.poison.load(Ordering::Acquire) {
                return Err(Error::ServiceFailed);
            }
            if self.closed || self.sh.cancel.load(Ordering::Acquire) {
                return Err(Error::Cancelled);
            }
            self.b.relax().map_err(Error::Backend)?;
        }
    }

    fn enqueue_inner(&self, p: &Published) -> Result<(), B::Error> {
        let (b, c, t) = (&*self.b, &self.compute, self.t);
        b.upload_table(c, &self.table, &p.side)?;
        if !self.cfg.is(Mutant::NoReadyProtection) {
            for &s in &p.ready {
                b.wait_event(c, &self.sh.ready[s])?;
            }
        }
        for l in 0..self.cfg.layers {
            let io = &self.sh.io[l];
            b.pre_layer(c, l)?;
            b.route(c, l, io)?;
            b.write_value(c, &self.sh.seq[l], t)?;
            b.launch_hits(c, l, &self.table, &self.sh.bank, io)?;
            b.wait_value(c, &self.sh.flag[l], t)?;
            b.add_y(c, l, io)?;
        }
        b.post_layers(c)?;
        b.write_value(c, &self.sh.seq[self.cfg.layers], t)
    }

    /// Enfileira o token do `Published`: a tabela, as esperas pelos `ready` e as camadas.
    pub fn enqueue_token(&mut self, p: Published) -> R<(), B> {
        if self.closed {
            return Err(Error::Cancelled);
        }
        if p.origin != self.sh.id {
            self.shutdown();
            return Err(Error::Config("Published de outro motor"));
        }
        if p.gen != self.t {
            // O `Published` é consumido e a linha de serviço ficaria esperando por ele: derruba o motor, e as chamadas
            // seguintes voltam com `Cancelled` em vez de travar.
            let e = Error::WrongGeneration { expected: self.t, got: p.gen };
            self.shutdown();
            return Err(e);
        }
        let r = self.enqueue_inner(&p);
        self.current = Some((p.side_idx, p.side));
        if let Err(e) = r {
            self.shutdown();
            return Err(Error::Backend(e));
        }
        Ok(())
    }

    /// Lê os logits do token (sincroniza a stream de cálculo) e devolve o lado do buffer da tabela.
    pub fn finish_token(&mut self) -> R<(), B> {
        if self.closed {
            return Err(Error::Cancelled);
        }
        if self.current.is_none() {
            return Err(Error::Config("finish_token sem token enfileirado"));
        }
        if let Err(e) = self.b.sync(&self.compute) {
            self.shutdown();
            return Err(Error::Backend(e));
        }
        if self.sh.poison.load(Ordering::Acquire) {
            self.shutdown();
            return Err(Error::ServiceFailed);
        }
        if let Some(side) = self.current.take() {
            self.sh.returned.send(side);
        }
        self.t = self.t.wrapping_add(1);
        Ok(())
    }

    pub fn run_token(&mut self, p: Published) -> R<(), B> {
        self.enqueue_token(p)?;
        self.finish_token()
    }

    fn join(&mut self) {
        if let Some(h) = self.service.take() {
            #[cfg(not(motor_loom))]
            while !self.done.load(Ordering::Acquire) {
                if self.b.relax().is_err() {
                    std::thread::yield_now();
                }
            }
            self.service_result = Some(h.join().unwrap_or(Err(Error::ServiceFailed)));
        }
    }

    fn shutdown(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        self.sh.cancel.store(true, Ordering::Release);
        let flags: Vec<&AtomicU32> = self.sh.flag.iter().map(|f| &***f).collect();
        release_flags(self.t, &flags);
        for step in close_order(self.cfg.is(Mutant::OldDropOrder)) {
            match step {
                CloseStep::SyncCompute => {
                    let _ = self.b.sync(&self.compute);
                }
                CloseStep::SyncCopy => {
                    let _ = self.b.sync(&self.copy);
                }
                CloseStep::Join => self.join(),
            }
        }
    }

    /// Os ids dos experts que o roteador da GPU escolheu na camada `l` do último token terminado (`run_token` ou `finish_token`
    /// já voltou). Só vale com o motor parado entre dois tokens: durante um token a área está sendo escrita.
    pub fn layer_ids(&self, l: usize) -> Vec<u32> {
        LayerIo::load(&self.sh.io[l].ids)
    }

    /// Encerra na ordem do `Drop` e devolve o resultado da linha de serviço (o cancelamento conta como sucesso).
    pub fn close(mut self) -> R<(), B> {
        self.shutdown();
        match self.service_result.take() {
            Some(Err(Error::Cancelled)) | None => Ok(()),
            Some(r) => r,
        }
    }
}

/// Um passo do fechamento, depois do cancelamento e das flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CloseStep {
    SyncCompute,
    SyncCopy,
    Join,
}

/// A ordem do fechamento (passo 6 do A0b): o `join` da linha de serviço (que drena a própria stream de cópia) vem antes
/// da sincronização da stream de cálculo; a ordem antiga (mutante) sincronizava antes do `join`. Com a drenagem da linha de
/// serviço no `ExitGuard`, as duas ordens liberam a memória sem cópia pendente: o modelo do `loom` percorre esta lista e
/// mostra isso (a ordem antiga só é acusada junto com a falta de drenagem que o mutante `OldDropOrder` desliga).
pub(crate) fn close_order(old: bool) -> &'static [CloseStep] {
    if old {
        &[CloseStep::SyncCompute, CloseStep::SyncCopy, CloseStep::Join]
    } else {
        &[CloseStep::Join, CloseStep::SyncCompute]
    }
}

struct DoneOnExit(Arc<AtomicBool>);

impl Drop for DoneOnExit {
    fn drop(&mut self) {
        signal_done(&self.0);
    }
}

/// O sinal de término da linha de serviço, também no unwind (sem ele o `join` girava para sempre).
pub(crate) fn signal_done(done: &AtomicBool) {
    done.store(true, Ordering::Release);
}

impl<B: Backend> Drop for Engine<B> {
    fn drop(&mut self) {
        self.shutdown();
    }
}
