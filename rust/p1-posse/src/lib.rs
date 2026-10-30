//! Spike M0/P1 do runtime MoE em Rust (plano `docs/PLANO-runtime-rust-moe-20261029.md`).
//!
//! Pergunta: com o CUDA Rust, o compilador consegue recusar evictar ou sobrescrever um slot de expert na VRAM enquanto
//! um kernel ainda o lê? O desenho testado é o da consulta ask-5: o cache entrega o slot como empréstimo (`&Tensor`) ao
//! kernel do cutile, e a operação de kernel carrega esse empréstimo até ser enviada; evictar ou fazer upload exigem
//! `&mut SlotCache`. O upload a partir da RAM é uma operação própria (`Upload`), porque o cuda-core só oferece a cópia
//! assíncrona host→device como `unsafe fn`: o `unsafe` fica todo aqui, atrás de um empréstimo `&'a mut` do slot e da
//! retenção do buffer de origem até o fim do envio.
use cuda_async::device_context::with_default_device_policy;
use cuda_async::device_future::DeviceFuture;
use cuda_async::device_operation::{DeviceOp, ExecutionContext};
use cuda_async::error::DeviceError;
use cuda_core::sys::CUdeviceptr;
use cuda_core::Stream;
use cutile::prelude::*;
use std::future::IntoFuture;
use std::marker::PhantomData;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Copia o expert do slot para a saída; faz o papel do kernel que lê o expert.
    #[cutile::entry()]
    fn read_expert<const S: [i32; 1]>(y: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        y.store(x.load_like(y));
    }
}

/// Pesos de um expert na RAM. O `Arc` deixa o upload reter a origem até o fim da cópia.
#[derive(Clone)]
pub struct HostExpert(pub Arc<Vec<f32>>);

/// Slots de tamanho fixo na VRAM. `None` = livre.
pub struct SlotCache {
    slots: Vec<Option<Tensor<f32>>>,
    len: usize,
}

impl SlotCache {
    pub fn new(n_slots: usize, len: usize, stream: &Arc<Stream>) -> Result<Self, Error> {
        let mut slots = Vec::with_capacity(n_slots);
        for _ in 0..n_slots {
            slots.push(Some(api::zeros::<f32>(&[len]).sync_on(stream)?));
        }
        Ok(Self { slots, len })
    }

    /// Empréstimo do slot residente para um kernel. Enquanto a operação que o recebeu existir, o cache não pode ser
    /// alterado.
    pub fn lease(&self, i: usize) -> &Tensor<f32> {
        self.slots[i].as_ref().expect("slot livre")
    }

    /// Libera o slot (o tensor é devolvido ao pool de forma assíncrona).
    pub fn evict(&mut self, i: usize) {
        self.slots[i] = None;
    }

    /// Operação de upload RAM→VRAM para o slot `i`, que precisa estar alocado.
    pub fn upload<'a>(&'a mut self, i: usize, src: &HostExpert) -> Upload<'a> {
        assert_eq!(src.0.len(), self.len, "expert com tamanho diferente do slot");
        let dst = self.slots[i].as_mut().expect("slot livre").device_pointer().cu_deviceptr();
        Upload { dst, n: self.len, src: src.0.clone(), _slot: PhantomData }
    }
}

/// Cópia host→device para um slot emprestado de forma exclusiva por `'a`.
pub struct Upload<'a> {
    dst: CUdeviceptr,
    n: usize,
    src: Arc<Vec<f32>>,
    _slot: PhantomData<&'a mut Tensor<f32>>,
}

impl DeviceOp for Upload<'_> {
    type Output = ();
    unsafe fn execute(self, ctx: &ExecutionContext) -> Result<(), DeviceError> {
        // A origem fica viva até o envio terminar, mesmo que o future seja esquecido.
        ctx.retain(self.src.clone())?;
        // SAFETY: `dst` aponta para um slot de `n` f32 emprestado com exclusividade por `'a`; `src` tem `n` elementos e
        // está retido no contexto até a conclusão.
        unsafe { cuda_core::memcpy_htod_async(self.dst, self.src.as_ptr(), self.n, ctx.get_cuda_stream())? };
        Ok(())
    }
}

impl<'a> IntoFuture for Upload<'a> {
    type Output = Result<(), DeviceError>;
    type IntoFuture = DeviceFuture<(), Upload<'a>>;
    fn into_future(self) -> Self::IntoFuture {
        match with_default_device_policy(|policy| {
            let stream = policy.next_stream()?;
            Ok(DeviceFuture::scheduled(self, ExecutionContext::new(stream)))
        }) {
            Ok(Ok(future)) => future,
            Ok(Err(e)) => DeviceFuture::failed(e),
            Err(e) => DeviceFuture::failed(e),
        }
    }
}
