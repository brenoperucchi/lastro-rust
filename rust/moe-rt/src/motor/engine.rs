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
    /// O instrumento de prefill (`crate::perfil_bloco`): `None` com a variável desligada (nada é alocado nem cronometrado).
    pub pf: Option<Arc<crate::perfil_bloco::Compartilhado>>,
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
    /// A posição da sequência do próximo passo (avança `m` a cada `finish_token`); não entra no protocolo (`t` é por passo).
    pos: u64,
    /// O tamanho do passo enfileirado, até o `finish_token`.
    cur_m: usize,
    current: Option<(usize, Pinned<[AtomicU32]>)>,
    closed: bool,
    /// Acumulador da F2 do host por operação: `None` com a variável desligada (nada é alocado nem cronometrado).
    f2_acc: Option<crate::perfil_bloco::AccHost>,
}

impl<B: Backend> Engine<B> {
    pub fn new<C: HostExperts>(b: Arc<B>, parts: Parts<B>, cfg: Config, cpu: C) -> R<Self, B> {
        Self::new_pf2(b, parts, cfg, cpu, crate::perfil_bloco::ligado_do_ambiente(), crate::perfil_bloco::f2_do_ambiente())
    }

    /// Como [`Engine::new`], com o instrumento de prefill (`perfil_bloco`) escolhido por quem chama em vez de lido do ambiente: `false` não
    /// aloca nada e deixa só um `Option` vazio no caminho quente.
    pub fn new_pf<C: HostExperts>(b: Arc<B>, parts: Parts<B>, cfg: Config, cpu: C, perfil: bool) -> R<Self, B> {
        Self::new_pf2(b, parts, cfg, cpu, perfil, false)
    }

    /// Como [`Engine::new_pf`], com a F2 do host por operação (`perfil_bloco`, plano da F2) escolhida por quem chama: só vale com `perfil`
    /// ligado; `false` não aloca acumulador nem lê relógio em `enqueue_step`.
    pub fn new_pf2<C: HostExperts>(b: Arc<B>, parts: Parts<B>, cfg: Config, cpu: C, perfil: bool, f2: bool) -> R<Self, B> {
        let f2 = crate::perfil_bloco::f2_efetivo(perfil, f2);
        let keys = cfg.layers * cfg.experts;
        let ok = cfg.layers > 0
            && cfg.topk > 0
            && cfg.topk <= cfg.experts
            && parts.seq.len() == cfg.layers + 1
            && parts.flag.len() == cfg.layers
            && parts.io.len() == cfg.layers
            && parts.ready.len() == cfg.slots
            && parts.sides.iter().all(|s| s.len() == keys)
            && cfg.prefill.iter().all(|&k| (k as usize) < keys)
            && cfg.max_positions >= 1
            && cfg.max_positions <= b.max_positions()
            && parts.io.iter().all(|io| {
                let o = LayerIo::offsets(cfg.topk, cfg.width, cfg.max_positions);
                io.ids.len() == cfg.max_positions * cfg.topk
                    && io.w.len() == cfg.max_positions * cfg.topk
                    && io.x.len() == cfg.max_positions * cfg.width
                    && io.y.len() == cfg.max_positions * cfg.width
                    && io.m.len() == usize::from(o.m.is_some())
            });
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
            pf: perfil.then(|| Arc::new(crate::perfil_bloco::Compartilhado::novo_f2(f2, cfg.layers))),
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
        let f2_acc = f2.then(|| crate::perfil_bloco::AccHost::com_capacidade(cfg.layers));
        Ok(Self { b, sh, compute: parts.compute, copy, table: parts.table, t: cfg.first_token, cfg, service: Some(service),
                  done, service_result: None, current: None, closed: false, pos: 0, cur_m: 1, f2_acc })
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

    /// Com `acc` ausente (o padrão) é idêntica à de antes: cada `marca` testa um `Option` vazio e não lê relógio.
    fn enqueue_inner(&self, p: &Published, m: usize, mut acc: Option<&mut crate::perfil_bloco::AccHost>) -> Result<(), B::Error> {
        use crate::perfil_bloco as pb;
        let (b, c, t) = (&*self.b, &self.compute, self.t);
        if let Some(a) = acc.as_deref_mut() {
            a.abre(p.ready.len());
        }
        let mut marca = |op: u8, camada: u32| {
            if let Some(a) = acc.as_deref_mut() {
                a.marca(op, camada);
            }
        };
        if self.cfg.max_positions > 1 {
            b.begin_step(c, t, self.pos, m)?;
        }
        marca(pb::H_BEGIN_STEP, pb::SEM_CAMADA);
        b.upload_table(c, &self.table, &p.side)?;
        marca(pb::H_UPLOAD_TABELA, pb::SEM_CAMADA);
        if !self.cfg.is(Mutant::NoReadyProtection) {
            for &s in &p.ready {
                b.wait_event(c, &self.sh.ready[s])?;
            }
        }
        marca(pb::H_WAIT_EVENT, pb::SEM_CAMADA);
        for l in 0..self.cfg.layers {
            let io = &self.sh.io[l];
            let cam = l as u32;
            b.pre_layer(c, l)?;
            marca(pb::H_PRE_LAYER, cam);
            b.route(c, l, io)?;
            marca(pb::H_ROUTE, cam);
            b.write_value(c, &self.sh.seq[l], t)?;
            marca(pb::H_SEQ, cam);
            b.launch_hits(c, l, &self.table, &self.sh.bank, io)?;
            marca(pb::H_LAUNCH_HITS, cam);
            b.wait_value(c, &self.sh.flag[l], t)?;
            marca(pb::H_WAIT_FLAG, cam);
            b.add_y(c, l, io)?;
            marca(pb::H_ADD_Y, cam);
        }
        b.post_layers(c)?;
        marca(pb::H_POST_LAYERS, pb::SEM_CAMADA);
        b.write_value(c, &self.sh.seq[self.cfg.layers], t)?;
        marca(pb::H_SEQ_FIM, pb::SEM_CAMADA);
        Ok(())
    }

    /// Entrega o acumulador da F2 do último `enqueue_step` ao registro compartilhado (a rotina de leitura chama depois de t4, fora de F2).
    pub(crate) fn perfil_f2_entrega(&self, gen: u32, m: usize, pos_ini: u64, t2: u64, t3: u64) {
        if let (Some(p), Some(a)) = (&self.sh.pf, &self.f2_acc) {
            if p.f2() {
                p.registra_f2_bloco(a.para_bloco(gen, m, pos_ini, t2, t3));
            }
        }
    }

    /// Se a coleta da F2 do host existe neste motor.
    pub fn perfil_f2_ligado(&self) -> bool {
        self.f2_acc.is_some()
    }

    /// A capacidade do acumulador da F2 (para conferir que o caminho quente não realoca).
    pub fn perfil_f2_capacidade(&self) -> Option<usize> {
        self.f2_acc.as_ref().map(|a| a.capacidade())
    }

    /// Se o instrumento de prefill existe neste motor (a variável foi lida como ligada na criação).
    pub fn perfil_ligado(&self) -> bool {
        self.sh.pf.is_some()
    }

    /// Se o instrumento está armado agora (um branch previsível: `None` ou flag relaxada).
    pub(crate) fn perfil_ativo(&self) -> bool {
        self.sh.pf.as_ref().is_some_and(|p| p.ativo())
    }

    /// Arma o instrumento para um prefill de `blocos_previstos` blocos; sem instrumento devolve uma guarda inerte, sem alocar.
    pub fn perfil_armar(&self, blocos_previstos: usize) -> crate::perfil_bloco::Armado {
        crate::perfil_bloco::Armado::arma(self.sh.pf.clone(), blocos_previstos)
    }

    pub(crate) fn perfil_registra_bloco(&self, r: crate::perfil_bloco::RecBloco) {
        if let Some(p) = &self.sh.pf {
            p.registra_bloco(r);
        }
    }

    /// A posição da sequência do próximo passo (só leitura; a rotina de leitura de N ids a confere com a do backend).
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// O maior número de posições por passo desta configuração (só leitura).
    pub fn max_positions(&self) -> usize {
        self.cfg.max_positions
    }

    /// Se o motor já foi encerrado (falha ou `close`): nenhum passo novo é aceito.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Encerra o motor na ordem do `Drop`, sem consumi-lo (a rotina de leitura o usa quando o preparo de um bloco falha no meio).
    pub(crate) fn encerra(&mut self) {
        self.shutdown();
    }

    /// Começa uma sequência nova no mesmo motor: a posição volta a 0 e o backend zera a dele (`Backend::reset_sequence`, com a stream
    /// de cálculo ociosa: não há token em andamento). A geração `t` NÃO é zerada (o protocolo com a linha de serviço continua
    /// contíguo), e a linha de serviço, a tabela e a política de cache seguem como estão. Recusa motor encerrado e token enfileirado
    /// (sem efeito); falha do backend encerra o motor.
    pub fn nova_sequencia(&mut self) -> R<(), B> {
        if self.closed {
            return Err(Error::Cancelled);
        }
        if self.current.is_some() {
            return Err(Error::Config("nova_sequencia com token em andamento"));
        }
        if let Err(e) = self.b.reset_sequence() {
            self.shutdown();
            return Err(Error::Backend(e));
        }
        self.pos = 0;
        Ok(())
    }

    /// Enfileira o token do `Published`: a tabela, as esperas pelos `ready` e as camadas (um passo de uma posição).
    pub fn enqueue_token(&mut self, p: Published) -> R<(), B> {
        self.enqueue_step(p, 1)
    }

    /// Enfileira um passo de `m` posições (`1 <= m <= max_positions`). Um `m` fora da faixa derruba o motor (o `Published` já foi
    /// consumido e a linha de serviço ficaria esperando por ele), como uma geração errada.
    pub fn enqueue_step(&mut self, p: Published, m: usize) -> R<(), B> {
        // Carimbo próprio de entrada da F2 (antes das validações); sem instrumento armado nada é lido.
        let entrada_f2 = if self.f2_acc.is_some() && self.perfil_ativo() { crate::perfil_bloco::agora_ns() } else { 0 };
        if self.closed {
            return Err(Error::Cancelled);
        }
        if m < 1 || m > self.cfg.max_positions {
            self.shutdown();
            return Err(Error::Config("m fora de 1..=max_positions"));
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
        // F2 do host: o acumulador só sai do lugar com o instrumento armado (prefill); decode e desligado não leem relógio.
        let mut acc = if self.f2_acc.is_some() && self.perfil_ativo() { self.f2_acc.take() } else { None };
        if let Some(a) = acc.as_mut() {
            a.reinicia(entrada_f2);
        }
        let r = self.enqueue_inner(&p, m, acc.as_mut());
        self.cur_m = m;
        self.current = Some((p.side_idx, p.side));
        if let Err(e) = r {
            self.f2_acc = acc;
            self.shutdown();
            return Err(Error::Backend(e));
        }
        if let Some(a) = acc.as_mut() {
            a.fecha_saida(crate::perfil_bloco::agora_ns());
            self.f2_acc = acc;
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
        self.pos += self.cur_m as u64;
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
        let ids = LayerIo::load(&self.sh.io[l].ids);
        ids[..self.cur_m * self.cfg.topk].to_vec()
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
