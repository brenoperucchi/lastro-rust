//! Conjunto fixo de threads para o caminho de CPU: `run(n, f)` executa `f(0..n)` repartido entre as threads (e quem chamou)
//! e só volta quando todas terminaram.
//!
//! Um passo de decode faz dezenas de despachos curtos (dezenas de µs cada), então criar threads por despacho, como no M1a,
//! custa mais que o trabalho. Aqui as threads ficam girando um tempo curto depois de cada despacho, à espera do próximo, e só
//! então dormem numa `Condvar`. As tarefas são pegas por um contador atômico, o que também equilibra núcleos P e E.
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;

/// Voltas de espera ativa antes de dormir (~100 µs nesta máquina).
const SPIN: u32 = 1 << 16;

type Job = *const (dyn Fn(usize) + Sync);

struct Shared {
    /// Geração do despacho atual; muda a cada `run`.
    generation: AtomicU64,
    /// Escrito só por `run`, quando nenhuma thread está no despacho anterior (todas confirmaram em `finished`).
    job: UnsafeCell<Option<Job>>,
    tasks: AtomicUsize,
    next: AtomicUsize,
    /// Threads que terminaram o despacho atual.
    finished: AtomicUsize,
    panicked: AtomicBool,
    stop: AtomicBool,
    lock: Mutex<()>,
    wake: Condvar,
}

// SAFETY: `job` só é escrito por `run` entre despachos (com todas as threads fora dele) e só é lido pelas threads depois de
// verem a geração nova com `Acquire`, que ordena a leitura depois da escrita feita antes do `Release` em `run`.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct Pool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Serializa chamadas concorrentes a `run`.
    serial: Mutex<()>,
    /// A1-DIAG10, A1w: CPU de quem chama (`[0]`) e de cada trabalhador (`[i]`); `None` não fixa nada.
    plan: Option<Vec<usize>>,
    caller_pinned: AtomicBool,
}

impl Pool {
    /// `threads` contando quem chama `run` (então `threads - 1` threads novas).
    pub fn new(threads: usize) -> Self {
        Self::with_plan(threads, crate::affinity::compute_plan())
    }

    /// Como `new`, com a colocação do A1w: o trabalhador `i` fixo em `plan[i]` e quem chama em `plan[0]`, na primeira `run`.
    pub fn with_plan(threads: usize, plan: Option<Vec<usize>>) -> Self {
        if let Some(p) = &plan {
            assert!(p.len() >= threads.max(1), "plano de afinidade com {} CPUs para {} participantes", p.len(), threads);
        }
        let shared = Arc::new(Shared {
            generation: AtomicU64::new(0),
            job: UnsafeCell::new(None),
            tasks: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
            finished: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        });
        let workers = (1..threads.max(1))
            .map(|i| {
                let s = shared.clone();
                let cpu = plan.as_ref().map(|p| p[i]);
                std::thread::Builder::new()
                    .name(format!("moe-cpu-{i}"))
                    .spawn(move || {
                        if let Some(c) = cpu {
                            crate::affinity::set_current(&[c]).expect("afinidade do trabalhador");
                        }
                        worker(&s)
                    })
                    .expect("thread do caminho de CPU")
            })
            .collect();
        Self { shared, workers, serial: Mutex::new(()), plan, caller_pinned: AtomicBool::new(false) }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Executa `f(i)` para `i` em `0..tasks`; volta quando todas as tarefas terminaram. Um pânico numa tarefa descarta as que
    /// sobraram e é repassado a quem chamou, e o pool continua utilizável.
    ///
    /// Não é reentrante: `f` não pode chamar `run` do mesmo pool (a trava `serial` já está com quem chamou, e o despacho
    /// interno ficaria esperando para sempre).
    pub fn run(&self, tasks: usize, f: &(dyn Fn(usize) + Sync)) {
        let _serial = self.serial.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(p) = &self.plan {
            if !self.caller_pinned.swap(true, Ordering::Relaxed) {
                crate::affinity::set_current(&[p[0]]).expect("afinidade de quem chama o Pool");
            }
        }
        let s = &*self.shared;
        let n = self.workers.len();
        if n == 0 || tasks <= 1 {
            (0..tasks).for_each(f);
            return;
        }
        // SAFETY: todas as threads confirmaram o despacho anterior (espera no fim deste método), então ninguém lê `job`.
        // O ponteiro apaga o tempo de vida de `f`; ele só é usado até a espera abaixo, enquanto `f` ainda vive.
        unsafe {
            let f: &'static (dyn Fn(usize) + Sync) = std::mem::transmute(f);
            *s.job.get() = Some(f as Job);
        }
        s.tasks.store(tasks, Ordering::Relaxed);
        s.next.store(0, Ordering::Relaxed);
        s.finished.store(0, Ordering::Relaxed);
        {
            let _g = s.lock.lock().unwrap_or_else(|e| e.into_inner());
            s.generation.fetch_add(1, Ordering::Release);
            s.wake.notify_all();
        }
        // Um pânico aqui também tem de esperar as outras threads saírem antes de `f` deixar de existir.
        let mine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| claim(s, f)));
        if mine.is_err() {
            s.next.store(usize::MAX / 2, Ordering::Relaxed);
        }
        let mut spins = 0u32;
        while s.finished.load(Ordering::Acquire) < n {
            spins += 1;
            if spins < SPIN {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        // SAFETY: todas as threads saíram do despacho.
        unsafe { *s.job.get() = None };
        // Limpo antes de repassar o pânico de quem chamou: senão um pânico de thread no mesmo despacho sobra para o próximo.
        let worker_panicked = s.panicked.swap(false, Ordering::Relaxed);
        if let Err(p) = mine {
            std::panic::resume_unwind(p);
        }
        if worker_panicked {
            panic!("tarefa do caminho de CPU entrou em pânico");
        }
    }
}

fn claim(s: &Shared, f: &(dyn Fn(usize) + Sync)) {
    let tasks = s.tasks.load(Ordering::Relaxed);
    loop {
        let i = s.next.fetch_add(1, Ordering::Relaxed);
        if i >= tasks {
            break;
        }
        f(i);
    }
}

fn worker(s: &Shared) {
    let mut seen = 0u64;
    loop {
        let mut spins = 0u32;
        let generation = loop {
            let g = s.generation.load(Ordering::Acquire);
            if g != seen || s.stop.load(Ordering::Relaxed) {
                break g;
            }
            spins += 1;
            if spins < SPIN {
                std::hint::spin_loop();
                continue;
            }
            let guard = s.lock.lock().unwrap_or_else(|e| e.into_inner());
            // Conferido com a trava: `run` incrementa a geração com ela, então não há aviso perdido.
            if s.generation.load(Ordering::Acquire) == seen && !s.stop.load(Ordering::Relaxed) {
                drop(s.wake.wait(guard).unwrap_or_else(|e| e.into_inner()));
            }
            spins = 0;
        };
        if s.stop.load(Ordering::Relaxed) {
            return;
        }
        seen = generation;
        // SAFETY: a geração nova foi publicada depois da escrita de `job` (Release/Acquire), e `run` não a reescreve antes de
        // esta thread contar em `finished`.
        let job = unsafe { *s.job.get() };
        // Sem tarefa não deveria acontecer; se acontecer, conta como pânico de tarefa em vez de matar a thread (o que deixaria
        // `run` esperando por ela para sempre).
        let ok = job.is_some_and(|job| std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| claim(s, unsafe { &*job }))).is_ok());
        if !ok {
            s.panicked.store(true, Ordering::Relaxed);
            // As tarefas que sobraram são descartadas para `run` não esperar uma tarefa que nunca termina.
            s.next.store(usize::MAX / 2, Ordering::Relaxed);
        }
        s.finished.fetch_add(1, Ordering::Release);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        {
            let _g = self.shared.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.shared.stop.store(true, Ordering::Relaxed);
            self.shared.wake.notify_all();
        }
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

/// Ponteiro de saída compartilhado entre tarefas que escrevem em faixas disjuntas.
#[derive(Clone, Copy)]
pub struct Out(pub *mut f32);
// SAFETY: quem cria um `Out` garante que cada tarefa escreve numa faixa própria e que o buffer vive até `run` voltar.
unsafe impl Send for Out {}
unsafe impl Sync for Out {}

impl Out {
    /// Escreve `v` na posição `i`. Método (e não `out.0` direto) para a closure capturar o `Out` inteiro, que é `Sync`.
    ///
    /// # Safety
    /// `i` está dentro do buffer, e nenhuma outra tarefa do mesmo despacho escreve ou lê a posição `i`.
    #[inline]
    pub unsafe fn write(self, i: usize, v: f32) {
        unsafe { *self.0.add(i) = v };
    }
}

#[cfg(test)]
mod testes_afinidade {
    use super::*;

    /// Com plano, o chamador (na primeira `run`) e cada `moe-cpu-i` ficam fixos na CPU do plano, e as máscaras efetivas lidas
    /// de dentro de cada thread são as pedidas.
    #[test]
    fn plano_fixa_chamador_e_trabalhadores() {
        let livres = crate::affinity::get_current().unwrap();
        if livres.len() < 3 {
            return;
        }
        let plan = vec![livres[0], livres[1], livres[2]];
        let seen = Arc::new(Mutex::new(Vec::<(String, Vec<usize>)>::new()));
        let caller = {
            let seen = seen.clone();
            let plan = plan.clone();
            std::thread::spawn(move || {
                let pool = Pool::with_plan(3, Some(plan));
                // tarefas longas o bastante para todas as threads pegarem alguma
                pool.run(64, &|_| {
                    let me = std::thread::current();
                    let nome = me.name().unwrap_or("chamador").to_string();
                    let m = crate::affinity::get_current().unwrap();
                    seen.lock().unwrap().push((nome, m));
                    std::thread::sleep(std::time::Duration::from_micros(200));
                });
            })
        };
        caller.join().unwrap();
        for (nome, m) in seen.lock().unwrap().iter() {
            let want = match nome.as_str() {
                "moe-cpu-1" => plan[1],
                "moe-cpu-2" => plan[2],
                _ => plan[0],
            };
            assert_eq!(m, &vec![want], "{nome}");
        }
    }

    /// Sem plano nada é fixado: as threads herdam a máscara de fora (A0).
    #[test]
    fn sem_plano_herda_a_mascara() {
        let antes = crate::affinity::get_current().unwrap();
        let pool = Pool::with_plan(2, None);
        let visto = Arc::new(Mutex::new(Vec::new()));
        pool.run(8, &|_| {
            visto.lock().unwrap().push(crate::affinity::get_current().unwrap());
            std::thread::sleep(std::time::Duration::from_micros(100));
        });
        assert!(visto.lock().unwrap().iter().all(|m| *m == antes));
    }
}
