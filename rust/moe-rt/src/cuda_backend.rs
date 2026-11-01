//! A1.2d: o backend de CUDA do protocolo D1 (`crate::motor::Backend`), sobre o driver (`cuda_core::sys`) e os kernels cutile
//! de `route.rs` e `hits.rs`. Os tipos de memória do motor (`Pinned`, `Mapped`, `Device`) ganham aqui o seu interior real:
//! `cuMemHostAlloc` (pinned e mapeada) e `cuMemAlloc`.
//!
//! Por camada, na stream de cálculo: o roteador (GEMV dos logits) e o top-k, que publicam ids, pesos e `x` na área mapeada
//! (`route`); `cuStreamWriteValue32(seq)`; a seleção dos acertos e os dois kernels agrupados sobre o banco único (`launch_hits`);
//! `cuStreamWaitValue32(flag)`; a soma da saída da GPU e da CPU no estado `h` (`add_y`).
//!
//! O estado `h` (a ativação corrente, `[hidden]`) vive no backend: `set_input` o carrega antes do token e `read_output` o lê
//! depois. O modelo deste passo é uma pilha de blocos MoE (`h ← h + moe(h)`); a atenção entra no A1.3.
#![allow(unsafe_code)]
use crate::gpu::{gemv_generics, kernels as gk, tile_k, AXPBY_B, GEMV_BM, SWIGLU_BM};
use crate::hits::hits_kernels;
use crate::motor::mem::{AllocId, Release};
use crate::motor::{Backend, Device, HostExperts, LayerIo, Mapped, Pinned, Role, NO_SLOT};
use crate::route::route_kernels;
use cuda_core::{sys, Device as CuDevice, Stream};
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::{PartitionMut, Reshape};
use cutile::tile_kernel::DevicePointer;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cutile::module]
pub mod add_kernels {
    use cutile::core::*;

    /// `h += y_gpu + y_host`, `B` elementos por bloco; `yh_p` é a saída da CPU na área mapeada (bits de `f32`).
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn add_y<const B: i32>(h: &mut Tensor<f32, { [B] }>, yg: &Tensor<f32, { [-1] }>, yh_p: *mut f32, n: i32) {
        let pid: (i32, i32, i32) = get_tile_block_id();
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[n] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut f32, { [] }> = pointer_to_tile(yh_p);
        let yh: Tensor<f32, { [-1] }> = make_tensor_view(ptr_tile, shape, strides, new_token_unordered());
        let tg: Tile<f32, { [B] }> = yg.partition(const_shape![B]).load([pid.0]);
        let th: Tile<f32, { [B] }> = yh.partition(const_shape![B]).load([pid.0]);
        let t: Tile<f32, { [B] }> = h.load();
        h.store(t + tg + th);
    }
}

#[derive(Debug)]
pub enum CudaError {
    /// Código do driver e a chamada.
    Driver(u32, &'static str),
    Kernel(String),
    Config(&'static str),
}

macro_rules! ck {
    ($e:expr) => {{
        let code = unsafe { $e };
        if code != sys::cudaError_enum_CUDA_SUCCESS {
            return Err(CudaError::Driver(code as u32, stringify!($e)));
        }
    }};
}

fn kernel_err<E: std::fmt::Debug>(e: E) -> CudaError {
    CudaError::Kernel(format!("{e:?}"))
}

/// Quantas vezes um caminho de erro esperou a stream antes de soltar os tensores locais (para o teste de regressão).
#[cfg(test)]
pub(crate) static ERR_SYNCS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
/// Só nos testes: faz o lançamento de `moe_down` falhar na compilação (parâmetro genérico inválido), depois de enfileirar o resto.
#[cfg(test)]
pub(crate) static FAULT_DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Espera a stream de cálculo. Os kernels já enfileirados leem os tensores que as variáveis locais seguram; o cutile libera
/// um tensor com `cuMemFreeAsync` numa stream de desalocação que não espera a nossa (`cuda-async/src/device_buffer.rs:128`),
/// então o erro tem de esperar a stream ANTES de a função retornar e soltar os locais (o idioma de `gpu::sync_on_err`).
fn sync_before_drop(raw: sys::CUstream) {
    #[cfg(test)]
    ERR_SYNCS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: só espera a stream.
    unsafe { sys::cuStreamSynchronize(raw) };
}

/// Para `.map_err` num `?` depois de enfileirar trabalho: espera a stream e converte o erro do cutile.
fn fail_k<E: std::fmt::Debug>(raw: sys::CUstream) -> impl FnOnce(E) -> CudaError {
    move |e| {
        sync_before_drop(raw);
        kernel_err(e)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub layers: usize,
    pub experts: usize,
    pub hidden: usize,
    pub inter: usize,
    pub topk: usize,
    pub slots: usize,
}

impl Dims {
    fn gu_elems(&self) -> usize {
        2 * self.inter * self.hidden
    }
    fn dn_elems(&self) -> usize {
        self.hidden * self.inter
    }
    /// O banco tem um slot a mais, o de preenchimento (zerado, fora do cache): os grupos além da contagem de acertos o leem.
    fn bank_slots(&self) -> usize {
        self.slots + 1
    }
    fn expert_bytes(&self) -> usize {
        (self.gu_elems() + self.dn_elems()) * 2
    }
}

/// Memória do host do driver, liberada ao cair.
struct HostMem(*mut c_void);
// SAFETY: um ponteiro de alocação do driver; só é liberado no `Drop`.
unsafe impl Send for HostMem {}
unsafe impl Sync for HostMem {}

impl Drop for HostMem {
    fn drop(&mut self) {
        // SAFETY: alocado por `cuMemHostAlloc`.
        unsafe { sys::cuMemFreeHost(self.0) };
    }
}

fn host_alloc(bytes: usize, mapped: bool) -> Result<(HostMem, u64), CudaError> {
    let mut p = std::ptr::null_mut();
    let flags = sys::CU_MEMHOSTALLOC_PORTABLE | if mapped { sys::CU_MEMHOSTALLOC_DEVICEMAP } else { 0 };
    ck!(sys::cuMemHostAlloc(&mut p, bytes.max(4), flags));
    let mem = HostMem(p);
    let mut dev = 0;
    if mapped {
        ck!(sys::cuMemHostGetDevicePointer_v2(&mut dev, p, 0));
    }
    // SAFETY: `bytes.max(4)` bytes recém-alocados.
    unsafe { std::ptr::write_bytes(p as *mut u8, 0, bytes.max(4)) };
    Ok((mem, dev))
}

/// A `LayerIo` sobre a memória mapeada, e a memória.
struct LayerGuard {
    io: LayerIo,
    _mem: HostMem,
}
unsafe impl Send for LayerGuard {}
unsafe impl Sync for LayerGuard {}

/// Pesos dos experts em memória pinned, `[chave][gate_up, down]` em BF16.
pub struct CudaWeights {
    base: *mut u8,
    per_key: usize,
    _mem: HostMem,
}
unsafe impl Send for CudaWeights {}
unsafe impl Sync for CudaWeights {}

/// Marca do banco de slots: uma alocação `[(slots + 1) · gate_up][(slots + 1) · down]`; o último slot é o de preenchimento.
pub struct CudaBank;

#[derive(Clone)]
pub struct CudaStream(pub Arc<Stream>);

impl CudaStream {
    fn raw(&self) -> sys::CUstream {
        self.0.cu_stream() as sys::CUstream
    }
}

pub struct CudaEvent(sys::CUevent);
// SAFETY: um evento do driver; o driver aceita o uso de qualquer thread com o contexto corrente.
unsafe impl Send for CudaEvent {}
unsafe impl Sync for CudaEvent {}

impl Drop for CudaEvent {
    fn drop(&mut self) {
        // SAFETY: criado por `cuEventCreate`.
        unsafe { sys::cuEventDestroy_v2(self.0) };
    }
}

/// Quem libera as alocações do device.
struct Owner {
    dev: Mutex<HashMap<u64, sys::CUdeviceptr>>,
    next: AtomicU64,
}

impl Release for Owner {
    fn release(&self, id: AllocId) {
        if let Some(p) = self.dev.lock().unwrap_or_else(|e| e.into_inner()).remove(&id.0) {
            // SAFETY: alocado por `cuMemAlloc`; o motor só solta depois de sincronizar as streams (fechamento).
            unsafe { sys::cuMemFree_v2(p) };
        }
    }
}

/// Buffers de trabalho (reaproveitados entre camadas) e a ativação corrente.
/// Tensores de trabalho, sempre donos aqui: os lançadores recebem partições emprestadas (`&mut`), então um lançamento que
/// falha derruba só o empréstimo e o tensor continua vivo até a espera da stream (o cutile libera com `cuMemFreeAsync`).
struct State {
    h: Tensor<f32>,
    logits: Tensor<f32>,
    hs: Tensor<i32>,
    hw: Tensor<f32>,
    hc: Tensor<i32>,
    a: Tensor<f32>,
    yg: Tensor<f32>,
}

pub struct CudaBackend {
    dims: Dims,
    device: Arc<CuDevice>,
    owner: Arc<Owner>,
    router: Vec<Tensor<bf16>>,
    st: Mutex<State>,
}

fn ptr<T>(addr: u64) -> DevicePointer<T> {
    // SAFETY: o endereço vem de uma alocação do driver (mapeada ou do device) viva enquanto o kernel roda.
    unsafe { DevicePointer::<T>::from_cu_deviceptr(addr as sys::CUdeviceptr) }
}

impl CudaBackend {
    /// `routers[l]` é o roteador da camada `l`, `[experts, hidden]` em BF16. Deixa o contexto do device corrente na thread.
    pub fn new(device: Arc<CuDevice>, dims: Dims, routers: &[Vec<bf16>], stream: &CudaStream) -> Result<Arc<Self>, CudaError> {
        let ok = dims.topk.is_power_of_two()
            && dims.topk <= dims.experts
            && routers.len() == dims.layers
            && routers.iter().all(|r| r.len() == dims.experts * dims.hidden)
            && dims.hidden % AXPBY_B == 0
            && dims.hidden % GEMV_BM == 0
            && dims.inter % SWIGLU_BM == 0
            && dims.experts % GEMV_BM == 0;
        if !ok {
            return Err(CudaError::Config("dimensões fora dos tamanhos de bloco ou topk que não é potência de dois"));
        }
        device.bind_to_thread().map_err(kernel_err)?;
        let s = &stream.0;
        let router = routers
            .iter()
            .map(|r| {
                api::copy_host_vec_to_device(&Arc::new(r.clone()))
                    .sync_on(s)
                    .map_err(kernel_err)?
                    .reshape(&[dims.experts, dims.hidden])
                    .map_err(kernel_err)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let zeros_f = |n: usize| -> Result<Tensor<f32>, CudaError> { api::zeros::<f32>(&[n]).sync_on(s).map_err(kernel_err) };
        let zeros_i = |n: usize| -> Result<Tensor<i32>, CudaError> { api::zeros::<i32>(&[n]).sync_on(s).map_err(kernel_err) };
        let k = dims.topk;
        let st = State {
            h: zeros_f(dims.hidden)?,
            logits: zeros_f(dims.experts)?,
            hs: zeros_i(k)?,
            hw: zeros_f(k)?,
            hc: zeros_i(1)?,
            a: zeros_f(k * dims.inter)?,
            yg: zeros_f(dims.hidden)?,
        };
        Ok(Arc::new(Self { dims, device, owner: Arc::new(Owner { dev: Mutex::new(HashMap::new()), next: AtomicU64::new(1) }), router, st: Mutex::new(st) }))
    }

    fn new_id(&self) -> AllocId {
        AllocId(self.owner.next.fetch_add(1, Ordering::Relaxed))
    }

    fn owner(&self) -> Arc<dyn Release> {
        self.owner.clone()
    }

    // ---- as peças de um motor ----

    /// Memória do device.
    fn dev_alloc(&self, bytes: usize) -> Result<(u64, AllocId), CudaError> {
        let mut p = 0;
        ck!(sys::cuMemAlloc_v2(&mut p, bytes.max(4)));
        let id = self.new_id();
        self.owner.dev.lock().unwrap_or_else(|e| e.into_inner()).insert(id.0, p);
        Ok((p as u64, id))
    }

    pub fn bank(&self) -> Result<Device<CudaBank>, CudaError> {
        let bytes = self.dims.bank_slots() * self.dims.expert_bytes();
        let (p, id) = self.dev_alloc(bytes)?;
        ck!(sys::cuMemsetD8_v2(p as sys::CUdeviceptr, 0, bytes));
        // O memset síncrono roda na stream nula, que não ordena as streams não bloqueantes do motor: espera o contexto.
        ck!(sys::cuCtxSynchronize());
        Ok(Device::from_parts(p, id, self.owner()))
    }

    pub fn device_table(&self) -> Result<Device<[u32]>, CudaError> {
        let (p, id) = self.dev_alloc(self.dims.layers * self.dims.experts * 4)?;
        Ok(Device::from_parts(p, id, self.owner()))
    }

    pub fn pinned_table(&self) -> Result<Pinned<[AtomicU32]>, CudaError> {
        let n = self.dims.layers * self.dims.experts;
        let (mem, _) = host_alloc(n * 4, false)?;
        let p = mem.0 as *const AtomicU32;
        for j in 0..n {
            // SAFETY: `n` palavras alocadas acima.
            unsafe { (*p.add(j)).store(NO_SLOT, Ordering::Relaxed) };
        }
        let id = self.new_id();
        // SAFETY: a memória vive enquanto `mem` (guardado em `keep`); só é tocada por atômicos e pelo driver.
        Ok(unsafe { Pinned::from_raw(std::ptr::slice_from_raw_parts(p, n), Arc::new(mem), id, self.owner()) })
    }

    pub fn mapped_u32(&self, v: u32) -> Result<Mapped<AtomicU32>, CudaError> {
        let (mem, dev) = host_alloc(4, true)?;
        let p = mem.0 as *const AtomicU32;
        // SAFETY: uma palavra alocada acima.
        unsafe { (*p).store(v, Ordering::Relaxed) };
        let id = self.new_id();
        // SAFETY: como em `pinned_table`; `dev` é o endereço do device dessa palavra.
        Ok(unsafe { Mapped::from_raw(p, Arc::new(mem), dev, id, self.owner()) })
    }

    pub fn mapped_io(&self) -> Result<Mapped<LayerIo>, CudaError> {
        let (k, w) = (self.dims.topk, self.dims.hidden);
        let (mem, dev) = host_alloc((2 * k + 2 * w) * 4, true)?;
        // SAFETY: `2k + 2w` palavras zeradas e alinhadas, vivas enquanto o `LayerGuard` viver.
        let io = unsafe { LayerIo::from_raw(mem.0 as *const AtomicU32, k, w) };
        let guard = Arc::new(LayerGuard { io, _mem: mem });
        let p: *const LayerIo = &guard.io;
        let id = self.new_id();
        // SAFETY: `p` aponta dentro de `guard`, que `keep` segura.
        Ok(unsafe { Mapped::from_raw(p, guard, dev, id, self.owner()) })
    }

    /// Os pesos de todos os experts em memória pinned: `experts_bf16[chave] = (gate_up, down)`, chave = camada·E + expert.
    pub fn weights(&self, experts: &[(Vec<bf16>, Vec<bf16>)]) -> Result<Pinned<CudaWeights>, CudaError> {
        if experts.len() != self.dims.layers * self.dims.experts {
            return Err(CudaError::Config("número de experts"));
        }
        let per_key = self.dims.expert_bytes();
        let (mem, _) = host_alloc(experts.len() * per_key, false)?;
        let base = mem.0 as *mut u8;
        for (k, (gu, dn)) in experts.iter().enumerate() {
            if gu.len() != self.dims.gu_elems() || dn.len() != self.dims.dn_elems() {
                return Err(CudaError::Config("formas do expert"));
            }
            // SAFETY: `per_key` bytes por chave dentro da alocação.
            unsafe {
                let dst = base.add(k * per_key);
                std::ptr::copy_nonoverlapping(gu.as_ptr() as *const u8, dst, gu.len() * 2);
                std::ptr::copy_nonoverlapping(dn.as_ptr() as *const u8, dst.add(gu.len() * 2), dn.len() * 2);
            }
        }
        let id = self.new_id();
        Ok(Pinned::from_parts(Arc::new(CudaWeights { base, per_key, _mem: mem }), id, self.owner()))
    }

    pub fn event(&self) -> Result<CudaEvent, CudaError> {
        let mut e = std::ptr::null_mut();
        ck!(sys::cuEventCreate(&mut e, sys::CUevent_flags_enum_CU_EVENT_DISABLE_TIMING as _));
        Ok(CudaEvent(e))
    }

    /// As peças de um motor com a configuração `cfg`. As flags e sequências começam em `first_token − 1`.
    pub fn parts(
        &self,
        compute: CudaStream,
        copy: CudaStream,
        weights: Pinned<CudaWeights>,
        first_token: u32,
    ) -> Result<crate::motor::Parts<CudaBackend>, CudaError> {
        let d = &self.dims;
        let init = first_token.wrapping_sub(1);
        Ok(crate::motor::Parts {
            compute,
            copy,
            bank: self.bank()?,
            table: self.device_table()?,
            sides: [self.pinned_table()?, self.pinned_table()?],
            weights,
            ready: (0..d.slots).map(|_| self.event()).collect::<Result<_, _>>()?,
            seq: (0..=d.layers).map(|_| self.mapped_u32(init)).collect::<Result<_, _>>()?,
            flag: (0..d.layers).map(|_| self.mapped_u32(init)).collect::<Result<_, _>>()?,
            io: (0..d.layers).map(|_| self.mapped_io()).collect::<Result<_, _>>()?,
        })
    }

    // ---- a ativação corrente ----

    /// Carrega a ativação de entrada do token (na stream de cálculo; espera o que já estava nela).
    pub fn set_input(&self, s: &CudaStream, x: &[f32]) -> Result<(), CudaError> {
        if x.len() != self.dims.hidden {
            return Err(CudaError::Config("tamanho da entrada"));
        }
        let h = api::copy_host_vec_to_device(&Arc::new(x.to_vec())).sync_on(&s.0).map_err(kernel_err)?;
        self.st.lock().unwrap_or_else(|e| e.into_inner()).h = h;
        Ok(())
    }

    /// Lê a ativação depois de sincronizar a stream de cálculo.
    pub fn read_output(&self, s: &CudaStream) -> Result<Vec<f32>, CudaError> {
        ck!(sys::cuStreamSynchronize(s.raw()));
        let mut st = self.st.lock().unwrap_or_else(|e| e.into_inner());
        let h = std::mem::replace(&mut st.h, api::zeros::<f32>(&[self.dims.hidden]).sync_on(&s.0).map_err(kernel_err)?);
        let h = Arc::new(h);
        let v = api::copy_device_to_host_vec(&h).sync_on(&s.0).map_err(kernel_err)?;
        st.h = Arc::try_unwrap(h).map_err(|_| CudaError::Config("h ainda referenciado"))?;
        Ok(v)
    }
}

impl Backend for CudaBackend {
    type Error = CudaError;
    type Stream = CudaStream;
    type Event = CudaEvent;
    type Weights = CudaWeights;
    type Bank = CudaBank;

    fn copy_expert(&self, s: &CudaStream, dst: &Device<CudaBank>, slot: usize, src: &Pinned<CudaWeights>, key: u32) -> Result<(), CudaError> {
        let d = &self.dims;
        let (gu, dn) = (d.gu_elems() * 2, d.dn_elems() * 2);
        if slot >= d.slots || key as usize >= d.layers * d.experts {
            return Err(CudaError::Config("slot ou chave fora do banco"));
        }
        let from = unsafe { src.base.add(key as usize * src.per_key) };
        let to_gu = dst.dev_addr() + (slot * gu) as u64;
        let to_dn = dst.dev_addr() + (d.bank_slots() * gu + slot * dn) as u64;
        ck!(sys::cuMemcpyHtoDAsync_v2(to_gu as sys::CUdeviceptr, from as *const c_void, gu, s.raw()));
        ck!(sys::cuMemcpyHtoDAsync_v2(to_dn as sys::CUdeviceptr, from.add(gu) as *const c_void, dn, s.raw()));
        Ok(())
    }

    fn record(&self, ev: &CudaEvent, s: &CudaStream) -> Result<(), CudaError> {
        ck!(sys::cuEventRecord(ev.0, s.raw()));
        Ok(())
    }

    fn wait_event(&self, s: &CudaStream, ev: &CudaEvent) -> Result<(), CudaError> {
        ck!(sys::cuStreamWaitEvent(s.raw(), ev.0, 0));
        Ok(())
    }

    fn query(&self, ev: &CudaEvent) -> Result<bool, CudaError> {
        let code = unsafe { sys::cuEventQuery(ev.0) };
        match code {
            c if c == sys::cudaError_enum_CUDA_SUCCESS => Ok(true),
            c if c == sys::cudaError_enum_CUDA_ERROR_NOT_READY => Ok(false),
            c => Err(CudaError::Driver(c as u32, "cuEventQuery")),
        }
    }

    fn upload_table(&self, s: &CudaStream, dst: &Device<[u32]>, src: &Pinned<[AtomicU32]>) -> Result<(), CudaError> {
        ck!(sys::cuMemcpyHtoDAsync_v2(dst.dev_addr() as sys::CUdeviceptr, src.as_ptr() as *const c_void, src.len() * 4, s.raw()));
        Ok(())
    }

    fn route(&self, s: &CudaStream, layer: usize, io: &Mapped<LayerIo>) -> Result<(), CudaError> {
        let raw = s.raw();
        let d = self.dims;
        let mut guard = self.st.lock().unwrap_or_else(|e| e.into_inner());
        let st = &mut *guard;
        // SAFETY: E e H são múltiplos de BM e BK (conferidos em `new`). Os tensores do estado ficam donos aqui; o lançador só
        // recebe empréstimos, então uma falha não os solta antes da espera da stream (`fail_k`).
        unsafe { gk::gemv((&mut st.logits).partition([GEMV_BM]), &self.router[layer], &st.h).generics(gemv_generics(d.hidden)).async_on(&s.0).map_err(fail_k(raw))? };
        let base = io.dev_addr();
        let (ids, w, x) = (base, base + (d.topk * 4) as u64, base + (2 * d.topk * 4) as u64);
        // SAFETY: um bloco; as três áreas mapeadas têm K, K e H palavras; `logits` tem E elementos.
        unsafe {
            route_kernels::route_topk(ptr::<i32>(ids), ptr::<f32>(w), &st.logits, d.experts as i32, &st.h, ptr::<f32>(x))
                .generics(vec![d.experts.next_power_of_two().to_string(), d.topk.to_string(), d.hidden.to_string(), tile_k(d.hidden).to_string(), "1".into()])
                .grid((1, 1, 1))
                .async_on(&s.0)
                .map_err(fail_k(raw))?;
        }
        Ok(())
    }

    fn launch_hits(&self, s: &CudaStream, layer: usize, table: &Device<[u32]>, bank: &Device<CudaBank>, io: &Mapped<LayerIo>) -> Result<(), CudaError> {
        let raw = s.raw();
        let d = self.dims;
        let mut guard = self.st.lock().unwrap_or_else(|e| e.into_inner());
        let st = &mut *guard;
        let base = io.dev_addr();
        let (ids, w) = (base, base + (d.topk * 4) as u64);
        // SAFETY: um bloco; K ids e K pesos mapeados; a linha da camada cabe na tabela de L·E entradas.
        unsafe {
            hits_kernels::hit_select((&mut st.hs).partition([d.topk]), (&mut st.hw).partition([d.topk]), (&mut st.hc).partition([1]), ptr::<i32>(ids), ptr::<f32>(w), ptr::<i32>(table.dev_addr()), (d.layers * d.experts) as i32, (layer * d.experts) as i32, d.slots as i32)
                .generics(vec![d.topk.to_string()])
                .async_on(&s.0)
                .map_err(fail_k(raw))?;
        }
        let gu_bytes = d.gu_elems() * 2;
        let (gu_p, dn_p) = (bank.dev_addr(), bank.dev_addr() + (d.bank_slots() * gu_bytes) as u64);
        // SAFETY: H e I múltiplos de BM e BK; os slots da lista estão em 0..S.
        unsafe {
            hits_kernels::moe_gate_up((&mut st.a).partition([SWIGLU_BM]), ptr::<bf16>(gu_p), (d.bank_slots() * 2 * d.inter) as i32, &st.h, &st.hs)
                .generics(vec![SWIGLU_BM.to_string(), tile_k(d.hidden).to_string(), d.hidden.to_string(), d.inter.to_string()])
                .async_on(&s.0)
                .map_err(fail_k(raw))?;
        }
        #[cfg(test)]
        let down_bm = if FAULT_DOWN.load(Ordering::Relaxed) { "não-é-número".to_string() } else { GEMV_BM.to_string() };
        #[cfg(not(test))]
        let down_bm = GEMV_BM.to_string();
        // SAFETY: como acima; `a` tem K·I elementos.
        unsafe {
            hits_kernels::moe_down((&mut st.yg).partition([GEMV_BM]), ptr::<bf16>(dn_p), (d.bank_slots() * d.hidden) as i32, &st.a, &st.hs, &st.hw, &st.hc)
                .generics(vec![down_bm, tile_k(d.inter).to_string(), d.hidden.to_string(), d.inter.to_string(), d.topk.to_string()])
                .async_on(&s.0)
                .map_err(fail_k(raw))?;
        }
        Ok(())
    }

    fn add_y(&self, s: &CudaStream, _layer: usize, io: &Mapped<LayerIo>) -> Result<(), CudaError> {
        let raw = s.raw();
        let d = self.dims;
        let mut guard = self.st.lock().unwrap_or_else(|e| e.into_inner());
        let st = &mut *guard;
        let y_host = io.dev_addr() + ((2 * d.topk + d.hidden) * 4) as u64;
        // SAFETY: H múltiplo de AXPBY_B; a área `y` mapeada tem H palavras.
        unsafe {
            add_kernels::add_y((&mut st.h).partition([AXPBY_B]), &st.yg, ptr::<f32>(y_host), d.hidden as i32).generics(vec![AXPBY_B.to_string()]).async_on(&s.0).map_err(fail_k(raw))?
        };
        Ok(())
    }

    fn write_value(&self, s: &CudaStream, at: &Mapped<AtomicU32>, v: u32) -> Result<(), CudaError> {
        ck!(sys::cuStreamWriteValue32_v2(s.raw(), at.dev_addr() as sys::CUdeviceptr, v, sys::CUstreamWriteValue_flags_enum_CU_STREAM_WRITE_VALUE_DEFAULT as _));
        Ok(())
    }

    fn wait_value(&self, s: &CudaStream, at: &Mapped<AtomicU32>, v: u32) -> Result<(), CudaError> {
        ck!(sys::cuStreamWaitValue32_v2(s.raw(), at.dev_addr() as sys::CUdeviceptr, v, sys::CUstreamWaitValue_flags_enum_CU_STREAM_WAIT_VALUE_GEQ as _));
        Ok(())
    }

    fn sync(&self, s: &CudaStream) -> Result<(), CudaError> {
        ck!(sys::cuStreamSynchronize(s.raw()));
        Ok(())
    }

    fn on_thread_start(&self, _role: Role) {
        let _ = self.device.bind_to_thread();
    }
}

/// A CPU do teste e do A1.3: os experts de uma camada calculados por `Cpu` a partir dos pesos BF16 na RAM.
pub struct CpuServe {
    pub cpu: crate::cpu::Cpu,
    pub hidden: usize,
    pub inter: usize,
    pub experts: usize,
    /// `(gate_up, down)` por chave (camada · E + expert).
    pub weights: Arc<Vec<(Vec<bf16>, Vec<bf16>)>>,
}

impl HostExperts for CpuServe {
    type Error = String;

    fn compute(&mut self, layer: usize, keys: &[u32], weights: &[u32], x: &[u32], y: &mut [u32]) -> Result<(), String> {
        let xf: Vec<f32> = x.iter().map(|&b| f32::from_bits(b)).collect();
        let refs: Vec<crate::cpu::ExpertRef> = keys
            .iter()
            .zip(weights)
            .map(|(&k, &w)| {
                debug_assert_eq!(k as usize / self.experts, layer);
                let (gu, dn) = &self.weights[k as usize];
                crate::cpu::ExpertRef { gate_up: gu, down: dn, weight: f32::from_bits(w) }
            })
            .collect();
        if refs.is_empty() {
            y.iter_mut().for_each(|v| *v = 0);
            return Ok(());
        }
        let out = self.cpu.experts(&xf, &refs, self.hidden, self.inter);
        for (d, v) in y.iter_mut().zip(out) {
            *d = v.to_bits();
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Routing;
    use crate::cpu::{Cpu, ExpertRef};
    use crate::motor::{AdmitMode, Config, Engine};

    const L: usize = 3;
    const E: usize = 16;
    const H: usize = 512;
    const I: usize = 256;
    const K: usize = 4;
    const SLOTS: usize = 14;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((*s >> 32) as u32) as i32) as f32 / (1u64 << 31) as f32
    }

    fn rnd(n: usize, scale: f32, seed: &mut u64) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(lcg(seed) * scale)).collect()
    }

    /// Referência: a pilha de blocos MoE inteira na CPU, `h ← h + Σ peso · expert(h)`.
    fn ref_forward(cpu: &Cpu, routers: &[Vec<bf16>], ex: &[(Vec<bf16>, Vec<bf16>)], x: &[f32]) -> (Vec<f32>, Vec<Vec<usize>>) {
        let mut h = x.to_vec();
        let mut routes = Vec::new();
        for l in 0..L {
            let mut logits = vec![0f32; E];
            cpu.gemv(&routers[l], E, H, &h, &mut logits);
            let r = Routing::from_logits(&logits, K);
            let refs: Vec<ExpertRef> = r.experts.iter().zip(&r.weights).map(|(&e, &g)| ExpertRef { gate_up: &ex[l * E + e].0, down: &ex[l * E + e].1, weight: g }).collect();
            let y = cpu.experts(&h, &refs, H, I);
            h.iter_mut().zip(&y).for_each(|(a, b)| *a += b);
            routes.push(r.experts);
        }
        (h, routes)
    }

    /// O oráculo: erro relativo máximo; tamanho diferente ou qualquer valor não finito dá infinito (um NaN não pode sumir
    /// da métrica, como acontecia com `f32::max` iniciado em zero).
    fn erro_rel(got: &[f32], want: &[f32]) -> f32 {
        if got.len() != want.len() || got.iter().chain(want).any(|v| !v.is_finite()) {
            return f32::INFINITY;
        }
        let m = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        got.iter().zip(want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / m
    }

    #[test]
    fn oraculo_recusa_nan_e_tamanho_errado() {
        let w = [1.0f32, -2.0, 0.5];
        assert_eq!(erro_rel(&w, &w), 0.0);
        assert!(erro_rel(&[f32::NAN; 3], &w).is_infinite(), "saída toda NaN passou");
        assert!(erro_rel(&[1.0, f32::NAN, 0.5], &w).is_infinite(), "um NaN sumiu da métrica");
        assert!(erro_rel(&[1.0, -2.0, f32::INFINITY], &w).is_infinite());
        assert!(erro_rel(&[1.0, -2.0], &w).is_infinite(), "tamanho diferente passou");
        assert!(erro_rel(&[1.0, -2.0, 0.6], &w) > 1e-4);
    }

    /// Regressão (rodadas 9 e 10): um lançamento que falha depois de enfileirar trabalho tem de esperar a stream antes de
    /// soltar os tensores. A stream fica presa numa espera de valor (como a flag do doorbell); o erro real vem da compilação de
    /// `moe_down` (`FAULT_DOWN`). Sem a espera em `fail_k`, `launch_hits` volta na hora, com os kernels ainda na stream.
    #[test]
    fn erro_apos_enfileirar_espera_a_stream_antes_de_soltar_os_tensores() {
        let _trava = crate::gpu_test_lock();
        let Ok(device) = CuDevice::new(0) else { return };
        device.bind_to_thread().unwrap();
        let compute = CudaStream(device.new_stream().unwrap());
        let mut seed = 3u64;
        let routers: Vec<Vec<bf16>> = (0..L).map(|_| rnd(E * H, 0.2, &mut seed)).collect();
        let dims = Dims { layers: L, experts: E, hidden: H, inter: I, topk: K, slots: SLOTS };
        let b = CudaBackend::new(device, dims, &routers, &compute).unwrap();
        let (io, table, bank) = (b.mapped_io().unwrap(), b.device_table().unwrap(), b.bank().unwrap());
        let trava = b.mapped_u32(0).unwrap();
        // Tabela publicada (todas as entradas sem slot) antes de qualquer kernel; o lado pinned fica vivo até a cópia acabar.
        let lado = b.pinned_table().unwrap();
        b.upload_table(&compute, &table, &lado).unwrap();
        b.set_input(&compute, &(0..H).map(|_| lcg(&mut seed)).collect::<Vec<_>>()).unwrap();
        b.route(&compute, 0, &io).unwrap();
        // Aquece a compilação dos kernels (o JIT demora mais que a espera abaixo).
        b.launch_hits(&compute, 0, &table, &bank, &io).unwrap();
        b.sync(&compute).unwrap();
        // Prende a stream: tudo que `launch_hits` enfileirar fica esperando a flag.
        b.wait_value(&compute, &trava, 1).unwrap();
        FAULT_DOWN.store(true, Ordering::Relaxed);
        let antes = ERR_SYNCS.load(Ordering::Relaxed);
        let voltou = std::sync::atomic::AtomicBool::new(false);
        let r = std::thread::scope(|sc| {
            let h = sc.spawn(|| {
                let r = b.launch_hits(&compute, 0, &table, &bank, &io);
                voltou.store(true, Ordering::SeqCst);
                r
            });
            let fim = std::time::Instant::now() + std::time::Duration::from_secs(3);
            while !voltou.load(Ordering::SeqCst) && std::time::Instant::now() < fim {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            let preso = !voltou.load(Ordering::SeqCst);
            trava.store(1, Ordering::SeqCst);
            let r = h.join().unwrap();
            (preso, r)
        });
        FAULT_DOWN.store(false, Ordering::Relaxed);
        b.sync(&compute).unwrap();
        assert!(r.0, "launch_hits voltou com a stream ainda presa: o erro não esperou a stream antes de soltar os tensores");
        assert!(matches!(r.1, Err(CudaError::Kernel(_))), "esperava Kernel, veio {:?}", r.1);
        assert!(ERR_SYNCS.load(Ordering::Relaxed) > antes);
    }

    #[test]
    fn pilha_de_blocos_moe_pelo_d1_bate_com_a_cpu_a_1e4() {
        let _trava = crate::gpu_test_lock();
        let device = match CuDevice::new(0) {
            Ok(d) => d,
            Err(e) => {
                assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
                return;
            }
        };
        device.bind_to_thread().unwrap();
        let (compute, copy) = (CudaStream(device.new_stream().unwrap()), CudaStream(device.new_stream().unwrap()));
        let mut seed = 5u64;
        let routers: Vec<Vec<bf16>> = (0..L).map(|_| rnd(E * H, 0.2, &mut seed)).collect();
        let ex: Arc<Vec<(Vec<bf16>, Vec<bf16>)>> = Arc::new((0..L * E).map(|_| (rnd(2 * I * H, 0.06, &mut seed), rnd(H * I, 0.06, &mut seed))).collect());
        let dims = Dims { layers: L, experts: E, hidden: H, inter: I, topk: K, slots: SLOTS };
        let b = CudaBackend::new(device.clone(), dims, &routers, &compute).unwrap();
        let cfg = Config {
            layers: L,
            experts: E,
            topk: K,
            width: H,
            slots: SLOTS,
            backlog: 4,
            admit: AdmitMode::Event,
            first_token: 1,
            prefill: vec![],
            #[cfg(test)]
            mutant: None,
        };
        let parts = b.parts(compute.clone(), copy, b.weights(&ex).unwrap(), cfg.first_token).unwrap();
        let cpu = CpuServe { cpu: Cpu::new(4), hidden: H, inter: I, experts: E, weights: ex.clone() };
        let mut eng = Engine::new(b.clone(), parts, cfg, cpu).unwrap();
        let reference_cpu = Cpu::new(4);
        let mut pior = 0f32;
        let mut stats = crate::motor::Stats::default();
        // Duas entradas alternadas: o roteamento se repete, o cache enche e os acertos da GPU passam a dominar.
        let entradas: Vec<Vec<f32>> = (0..2).map(|_| (0..H).map(|_| lcg(&mut seed)).collect()).collect();
        for tok in 0..16 {
            let x = entradas[tok % 2].clone();
            let (want, _routes) = ref_forward(&reference_cpu, &routers, &ex, &x);
            b.set_input(&compute, &x).unwrap();
            let p = eng.next_published().unwrap();
            stats = p.stats().clone();
            eng.run_token(p).unwrap();
            let got = b.read_output(&compute).unwrap();
            let err = erro_rel(&got, &want);
            pior = pior.max(err);
            assert!(err <= 1e-4, "token {tok}: erro relativo {err:e}");
        }
        eprintln!("pior erro relativo {pior:e}; contadores até o último token: {stats:?}");
        assert!(stats.hits > 0 && stats.cpu > 0, "o teste precisa exercitar a GPU e a CPU: {stats:?}");
        assert!(stats.hits * 4 >= stats.needed, "poucos acertos da GPU para o teste valer: {stats:?}");
        eng.close().unwrap();
    }
}
