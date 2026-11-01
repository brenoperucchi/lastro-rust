//! A interface pequena sobre atômicos, threads e canal que as duas linhas do host e o backend falso usam: `std` na build
//! normal, `loom` com `--cfg motor_loom` (o modelo das intercalações em `loom_tests.rs`).
//!
//! O canal é uma fila atrás de um `Mutex`, sem bloqueio: quem recebe gira com [`crate::motor::Backend::relax`], que também
//! consulta o cancelamento. Assim o escalonador determinístico do backend falso (`fake::Sched`) controla toda espera.

#[cfg(not(motor_loom))]
pub use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
#[cfg(not(motor_loom))]
pub use std::sync::{Arc, Mutex, MutexGuard};
#[cfg(not(motor_loom))]
pub use std::thread;

#[cfg(motor_loom)]
pub use loom::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
#[cfg(motor_loom)]
pub use loom::sync::{Arc, Mutex, MutexGuard};
#[cfg(motor_loom)]
pub use loom::thread;

use std::collections::VecDeque;

/// Uma volta de espera ativa: `spin_loop` na build normal; no `loom`, cede a vez (sem isso o modelo não termina).
pub fn spin_hint() {
    #[cfg(not(motor_loom))]
    std::hint::spin_loop();
    #[cfg(motor_loom)]
    loom::thread::yield_now();
}

/// Trava ignorando envenenamento: um `panic` de uma linha não pode impedir a outra de fechar.
pub fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Canal de mão única entre as duas linhas, sem bloqueio.
pub struct Chan<T>(Arc<Mutex<VecDeque<T>>>);

impl<T> Clone for Chan<T> {
    fn clone(&self) -> Self {
        Chan(self.0.clone())
    }
}

impl<T> Chan<T> {
    pub fn new() -> Self {
        Chan(Arc::new(Mutex::new(VecDeque::new())))
    }

    pub fn send(&self, v: T) {
        lock(&self.0).push_back(v);
    }

    pub fn try_recv(&self) -> Option<T> {
        lock(&self.0).pop_front()
    }
}

impl<T> Default for Chan<T> {
    fn default() -> Self {
        Self::new()
    }
}
