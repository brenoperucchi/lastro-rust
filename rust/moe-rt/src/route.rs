//! A1.2a: o top-k do roteador na GPU (plano, A1.2). Um bloco só: dos logits `[E]` tira os `K` maiores (empate: o de menor
//! índice primeiro, como `Routing::from_logits`), calcula o softmax sobre eles e publica ids, pesos e o `x` da camada por
//! ponteiro em memória host mapeada (A1.2b). A visibilidade para o host vem do `cuStreamWriteValue32` que a stream enfileira
//! depois (A0b, passo 1); o teste daqui lê depois de `sync_on`.
//!
//! Pré-condições: logits finitos (NaN não é coberto; `-0` e `+0` empatam aqui e não em `total_cmp`), `K <= E <= EP`, `EP`
//! e `K` potências de dois, `H` múltiplo de `BX`.
#[cutile::module]
pub mod route_kernels {
    use cutile::core::*;

    /// `TS` é o sinal do desempate: 1 escolhe o menor índice entre os empatados (produção); -1 escolhe o maior e só existe
    /// para o teste provar que ele enxerga o desempate (mutante).
    /// `H` e `BX`: largura do `x` e a fatia (potência de dois que divide `H`; 512 para 1536).
    /// `EP` é `E` arredondado para potência de dois (tile); `K` também é potência de dois (8 no Granite).
    /// `ids[j]` e `w[j]` são o j-ésimo maior logit e o seu peso no softmax sobre os `K`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn route_topk<const EP: i32, const K: i32, const H: i32, const BX: i32, const TS: i32>(
        ids_p: *mut i32,
        w_p: *mut f32,
        logits: &Tensor<f32, { [-1] }>,
        e: i32,
        x: &Tensor<f32, { [-1] }>,
        x_p: *mut f32,
    ) {
        // Publica o x da camada (H palavras) na área mapeada, em fatias de BX.
        let px = x.partition(const_shape![BX]);
        let mut xo: Tensor<f32, { [-1] }> = tensor_from_ptr(x_p, H);
        let mut pxo = xo.partition_mut(const_shape![BX]);
        for b in 0i32..(H / BX) {
            let t: Tile<f32, { [BX] }> = px.load([b]);
            pxo.store(t, [b]);
        }
        let part = logits.partition(const_shape![EP]);
        let raw: Tile<f32, { [EP] }> = part.load([0i32]);
        let idx: Tile<i32, { [EP] }> = iota(const_shape![EP]);
        let e_b: Tile<i32, { [EP] }> = e.broadcast(const_shape![EP]);
        let valid: Tile<bool, { [EP] }> = lt_tile(idx, e_b);
        // Menos infinito: fica abaixo de qualquer logit finito, então nem o padding nem um expert já escolhido vencem.
        let neg: Tile<f32, { [EP] }> = constant(f32::NEG_INFINITY, const_shape![EP]);
        let tsv: i32 = TS;
        let ts: Tile<i32, { [EP] }> = tsv.broadcast(const_shape![EP]);
        let sidx: Tile<i32, { [EP] }> = idx * ts;
        let mut cur: Tile<f32, { [EP] }> = select(valid, raw, neg);

        let slot: Tile<i32, { [K] }> = iota(const_shape![K]);
        let mut out_i: Tile<i32, { [K] }> = constant(0i32, const_shape![K]);
        let mut out_v: Tile<f32, { [K] }> = constant(0.0f32, const_shape![K]);
        let none: Tile<i32, { [EP] }> = constant(2147483647i32, const_shape![EP]);
        for j in 0i32..K {
            let m_t: Tile<f32, { [1] }> = reduce_max(cur, 0i32);
            let m: f32 = tile_to_scalar(m_t.reshape(const_shape![]));
            let m_b: Tile<f32, { [EP] }> = m.broadcast(const_shape![EP]);
            let hit: Tile<bool, { [EP] }> = eq_tile(cur, m_b);
            let cand: Tile<i32, { [EP] }> = select(hit, sidx, none);
            let i_t: Tile<i32, { [1] }> = reduce_min(cand, 0i32);
            let i: i32 = tile_to_scalar::<i32, i32>(i_t.reshape(const_shape![])) * TS;
            let i_b: Tile<i32, { [EP] }> = i.broadcast(const_shape![EP]);
            let pick: Tile<bool, { [K] }> = eq_tile(slot, j.broadcast(const_shape![K]));
            out_i = select(pick, i.broadcast(const_shape![K]), out_i);
            out_v = select(pick, m.broadcast(const_shape![K]), out_v);
            let gone: Tile<bool, { [EP] }> = eq_tile(idx, i_b);
            cur = select(gone, neg, cur);
        }
        // O primeiro é o maior: soma `exp(v - v0)` e divide.
        let top_t: Tile<f32, { [1] }> = reduce_max(out_v, 0i32);
        let top: f32 = tile_to_scalar(top_t.reshape(const_shape![]));
        let ex: Tile<f32, { [K] }> = exp(out_v - top.broadcast(const_shape![K]));
        let sum_t: Tile<f32, { [1] }> = reduce_sum(ex, 0i32);
        let sum: f32 = tile_to_scalar(sum_t.reshape(const_shape![]));
        let mut ids: Tensor<i32, { [-1] }> = tensor_from_ptr(ids_p, K);
        let mut w: Tensor<f32, { [-1] }> = tensor_from_ptr(w_p, K);
        ids.partition_mut(const_shape![K]).store(out_i, [0i32]);
        w.partition_mut(const_shape![K]).store(ex / sum.broadcast(const_shape![K]), [0i32]);
    }

    /// Uma vista de `len` elementos sobre um ponteiro do device (aqui, o lado do device de memória host mapeada).
    unsafe fn tensor_from_ptr<T: ElementType>(ptr: *mut T, len: i32) -> Tensor<T, { [-1] }> {
        let shape: Shape<{ [-1] }> = Shape::<{ [-1] }> { dims: &[len] };
        let strides: Array<{ [-1] }> = Array::<{ [-1] }> { dims: &[1i32] };
        let ptr_tile: PointerTile<*mut T, { [] }> = pointer_to_tile(ptr);
        make_tensor_view(ptr_tile, shape, strides, new_token_unordered())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Routing;
    use cuda_core::Device;
    use cutile::prelude::*;
    use cutile::tile_kernel::DevicePointer;
    use std::sync::Arc;

    /// Memória host pinned e mapeada (como o `Mapped` do `a1-doorbell`).
    struct Mapped {
        host: *mut u32,
        dev: cuda_core::sys::CUdeviceptr,
    }
    impl Mapped {
        fn new(words: usize) -> Self {
            use cuda_core::sys;
            let mut p = std::ptr::null_mut();
            // SAFETY: alocação do driver; vive até o fim do teste (vaza de propósito, alguns KiB por teste).
            unsafe {
                assert_eq!(sys::cuMemHostAlloc(&mut p, words * 4, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP), 0);
                let mut dev = 0;
                assert_eq!(sys::cuMemHostGetDevicePointer_v2(&mut dev, p, 0), 0);
                for j in 0..words {
                    (p as *mut u32).add(j).write_volatile(0xdead_beef);
                }
                Self { host: p as *mut u32, dev }
            }
        }
        fn get(&self, j: usize) -> u32 {
            // SAFETY: `j` dentro da alocação.
            unsafe { self.host.add(j).read_volatile() }
        }
    }

    /// Logits com sinal, em [-8, 8).
    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((*s >> 32) as u32) as i32) as f32 / (1u64 << 31) as f32 * 8.0
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Tipo {
        Distintos,
        Empates,
        Negativos,
        Espalhados,
        TodosIguais(f32),
        DoisMaximos,
        EmpateNoCorte,
    }

    fn logits(tipo: Tipo, e: usize, seed: &mut u64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..e).map(|_| lcg(seed)).collect();
        match tipo {
            Tipo::Distintos => {}
            Tipo::Empates => v.iter_mut().for_each(|x| *x = (*x * 2.0).floor() / 2.0),
            Tipo::Negativos => v.iter_mut().for_each(|x| *x -= 10.0),
            Tipo::Espalhados => v.iter_mut().for_each(|x| *x *= 12.0),
            Tipo::TodosIguais(c) => v.iter_mut().for_each(|x| *x = c),
            Tipo::DoisMaximos => {
                v.iter_mut().for_each(|x| *x -= 20.0);
                v[e - 1] = 5.0;
                v[3] = 5.0;
            }
            // Os índices 5, 9 e 20 empatam na última vaga do corte; só o primeiro (5) cabe.
            Tipo::EmpateNoCorte => {
                v.iter_mut().for_each(|x| *x -= 20.0);
                for (r, i) in [0usize, 1, 2, 3, 4, 6, 7].iter().enumerate() {
                    v[*i] = 10.0 - r as f32;
                }
                for i in [5usize, 9, 20] {
                    v[i] = 1.0;
                }
            }
        }
        v
    }

    const K: usize = 8;
    const H: usize = 1536;

    /// Roda o kernel num caso. `e_kernel` é o `E` que o kernel recebe (`= e` na produção; `= ep` tira a máscara do padding,
    /// o mutante). Devolve (ids iguais, pesos iguais, x igual).
    fn roda(s: &Arc<cuda_core::Stream>, mi: &Mapped, mw: &Mapped, mx: &Mapped, lv: &[f32], ep: usize, e_kernel: usize, ts: i32, seed: &mut u64) -> (bool, bool, bool) {
        let host = Routing::from_logits(lv, K);
        let xv: Vec<f32> = (0..H).map(|_| lcg(seed)).collect();
        let xd = api::copy_host_vec_to_device(&Arc::new(xv.clone())).sync_on(s).unwrap();
        let ld = api::copy_host_vec_to_device(&Arc::new(lv.to_vec())).sync_on(s).unwrap();
        // SAFETY: um bloco; as áreas mapeadas têm K, K e H palavras, `logits` tem `lv.len()` elementos e a leitura além deles
        // é mascarada (ou, no mutante, devolve o zero do preenchimento).
        unsafe {
            route_kernels::route_topk(
                DevicePointer::<i32>::from_cu_deviceptr(mi.dev),
                DevicePointer::<f32>::from_cu_deviceptr(mw.dev),
                &ld,
                e_kernel as i32,
                &xd,
                DevicePointer::<f32>::from_cu_deviceptr(mx.dev),
            )
            .generics(vec![ep.to_string(), K.to_string(), H.to_string(), 512.to_string(), ts.to_string()])
            .grid((1, 1, 1))
            .sync_on(s)
            .unwrap();
        }
        let gi: Vec<usize> = (0..K).map(|j| mi.get(j) as i32 as usize).collect();
        let gw: Vec<f32> = (0..K).map(|j| f32::from_bits(mw.get(j))).collect();
        let ids_ok = gi == host.experts;
        let w_ok = gw.iter().zip(&host.weights).all(|(a, b)| (a - b).abs() <= 1e-6 * b.abs().max(1e-3));
        let x_ok = (0..H).all(|j| mx.get(j) == xv[j].to_bits());
        (ids_ok, w_ok, x_ok)
    }

    fn gpu() -> Option<Arc<cuda_core::Stream>> {
        match Device::new(0) {
            Ok(d) => Some(d.new_stream().unwrap()),
            Err(e) => {
                assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
                None
            }
        }
    }

    const TIPOS: [Tipo; 8] = [
        Tipo::Distintos,
        Tipo::Empates,
        Tipo::Negativos,
        Tipo::Espalhados,
        Tipo::TodosIguais(-1.0e30),
        Tipo::TodosIguais(-2.0e30),
        Tipo::DoisMaximos,
        Tipo::EmpateNoCorte,
    ];

    /// Para cada tipo, `por_tipo` casos em E = 32 (EP = 32) e E = 40 (EP = 64). Devolve as falhas por tipo.
    fn varre(s: &Arc<cuda_core::Stream>, por_tipo: usize, mascara: bool, ts: i32) -> Vec<(Tipo, usize)> {
        let (mi, mw, mx) = (Mapped::new(K), Mapped::new(K), Mapped::new(H));
        let mut seed = 7u64;
        let mut out = Vec::new();
        for tipo in TIPOS {
            let mut ruim = 0;
            for (e, ep) in [(32usize, 32usize), (40, 64)] {
                for _ in 0..por_tipo {
                    let lv = logits(tipo, e, &mut seed);
                    let ek = if mascara { e } else { ep };
                    let (a, b, c) = roda(s, &mi, &mw, &mx, &lv, ep, ek, ts, &mut seed);
                    ruim += !(a && b && c) as usize;
                }
            }
            out.push((tipo, ruim));
        }
        out
    }

    #[test]
    fn topk_na_gpu_bate_com_o_do_host() {
        let _trava = crate::gpu_test_lock();
        let Some(s) = gpu() else { return };
        let r = varre(&s, 50, true, 1);
        assert!(r.iter().all(|(_, n)| *n == 0), "casos diferentes do host: {r:?}");
    }

    /// Controle: o desempate pelo maior índice é acusado nos tipos com empate e só neles.
    #[test]
    fn controle_desempate_pelo_maior_indice_e_acusado_so_nos_empates() {
        let _trava = crate::gpu_test_lock();
        let Some(s) = gpu() else { return };
        let r = varre(&s, 50, true, -1);
        for (tipo, n) in &r {
            let tem_empate = matches!(tipo, Tipo::Empates | Tipo::TodosIguais(_) | Tipo::DoisMaximos | Tipo::EmpateNoCorte);
            assert_eq!(*n > 0, tem_empate, "{tipo:?}: {n} falhas; resultado completo {r:?}");
        }
    }

    /// Controle: sem a máscara do padding (o kernel recebe `E = EP`), os casos de E = 40 com logits menores que zero são
    /// acusados, e os de E = 32 (sem padding) não.
    #[test]
    fn controle_sem_a_mascara_do_padding_e_acusado() {
        let _trava = crate::gpu_test_lock();
        let Some(s) = gpu() else { return };
        let r = varre(&s, 50, false, 1);
        let por = |t: Tipo| r.iter().find(|(x, _)| *x == t).unwrap().1;
        assert!(por(Tipo::Negativos) > 0 && por(Tipo::TodosIguais(-1.0e30)) > 0, "mutante sem máscara passou: {r:?}");
    }
}
