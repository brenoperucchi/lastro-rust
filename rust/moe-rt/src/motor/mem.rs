//! Tipos por espaço de memória (plano, A0b, "O que o Rust dá a mais"): cada um tem o seu construtor, e as operações do
//! [`Backend`](super::Backend) só aceitam o tipo certo.
//! - [`Pinned<T>`]: memória do host travada na RAM, origem das cópias H2D (os experts, os dois lados da tabela). Dá acesso
//!   do host; memória pageable comum (`Vec`, `Box`) não passa por ela.
//! - [`Mapped<T>`]: pinned e mapeada, com um endereço do host e outro do device. É onde ficam o doorbell
//!   (`Mapped<AtomicU32>`) e a área de cada camada ([`LayerIo`](super::LayerIo)). O lado do host usa a ordem de memória
//!   explícita dos atômicos: a saída da CPU e depois a flag com `Release`; a sequência com `Acquire` antes de ler ids e `x`.
//! - [`Device<T>`]: memória do device (o banco de slots, a tabela do token). Não dá acesso do host.
//!
//! Os construtores são do crate: só um backend (o falso hoje; o de CUDA no A1.2/A1.3) cria esses valores. No A1.1 a
//! memória é do host (`Arc<T>`), com o endereço do device igual a um identificador; o backend de CUDA troca o interior por
//! `cuMemHostAlloc(PORTABLE)`, `cuMemHostAlloc(PORTABLE | DEVICEMAP)` com `cuMemHostGetDevicePointer`, e `cuMemAlloc`,
//! sem mudar a interface. A liberação passa pelo dono ([`Release`]), que no backend falso confere que nenhuma operação em
//! fila ainda usa a região (passo 6 do A0b: nada em fila quando a memória cai).
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;

/// Identificador de uma alocação, para o dono saber o que está sendo liberado.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AllocId(pub u64);

/// Quem libera a memória de uma alocação (o backend que a criou).
pub trait Release: Send + Sync {
    fn release(&self, id: AllocId);
}

/// Memória do host pinned (não mapeada): origem das cópias para o device.
pub struct Pinned<T: ?Sized> {
    host: Arc<T>,
    id: AllocId,
    owner: Arc<dyn Release>,
}

impl<T: ?Sized> Pinned<T> {
    pub(crate) fn from_parts(host: Arc<T>, id: AllocId, owner: Arc<dyn Release>) -> Self {
        Self { host, id, owner }
    }

    pub fn id(&self) -> AllocId {
        self.id
    }

    /// O conteúdo, para o backend ler na hora em que a operação roda (a memória fica viva até lá).
    pub(crate) fn shared(&self) -> Arc<T> {
        self.host.clone()
    }
}

impl<T: ?Sized> Deref for Pinned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.host
    }
}

impl<T: ?Sized> Drop for Pinned<T> {
    fn drop(&mut self) {
        self.owner.release(self.id);
    }
}

/// Memória do host pinned e mapeada no device.
pub struct Mapped<T: ?Sized> {
    host: Arc<T>,
    dev: u64,
    id: AllocId,
    owner: Arc<dyn Release>,
}

impl<T: ?Sized> Mapped<T> {
    pub(crate) fn from_parts(host: Arc<T>, dev: u64, id: AllocId, owner: Arc<dyn Release>) -> Self {
        Self { host, dev, id, owner }
    }

    pub fn id(&self) -> AllocId {
        self.id
    }

    /// O endereço do lado do device (o que vai para `cuStreamWaitValue32` e para os kernels).
    pub fn dev_addr(&self) -> u64 {
        self.dev
    }

    pub(crate) fn shared(&self) -> Arc<T> {
        self.host.clone()
    }
}

impl<T: ?Sized> Deref for Mapped<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.host
    }
}

impl<T: ?Sized> Drop for Mapped<T> {
    fn drop(&mut self) {
        self.owner.release(self.id);
    }
}

/// Memória do device. Sem acesso do host: só operações de stream a leem ou escrevem.
pub struct Device<T: ?Sized> {
    dev: u64,
    id: AllocId,
    owner: Arc<dyn Release>,
    _t: PhantomData<fn() -> Box<T>>,
}

impl<T: ?Sized> Device<T> {
    pub(crate) fn from_parts(dev: u64, id: AllocId, owner: Arc<dyn Release>) -> Self {
        Self { dev, id, owner, _t: PhantomData }
    }

    pub fn id(&self) -> AllocId {
        self.id
    }

    pub fn dev_addr(&self) -> u64 {
        self.dev
    }
}

impl<T: ?Sized> Drop for Device<T> {
    fn drop(&mut self) {
        self.owner.release(self.id);
    }
}
