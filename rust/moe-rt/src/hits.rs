//! A1.2c: um lançamento para todos os acertos de uma camada (plano, A0b passo 5). Três kernels cutile sobre o banco único de
//! slots (`[S·2I, H]` para gate_up e `[S·H, I]` para down, BF16):
//! - `hit_select`: um bloco; a partir dos ids e pesos do top-k e da linha da camada na tabela do token (slot ou -1) monta a
//!   lista compacta de acertos (slot, peso, contagem), na ordem do top-k;
//! - `moe_gate_up`: grade `(K·I/BM)`, o bloco `(j, r)` calcula `BM` linhas do SwiGLU do acerto `j`;
//! - `moe_down`: grade `H/BM`, cada bloco percorre os acertos em ordem e soma `peso · (W_down · a)`.
//! Posições além da contagem leem o slot de preenchimento (`pad`, fora do cache) e são descartadas no `moe_down` por máscara
//! (um slot com NaN não vaza). Nenhuma delas lê um slot que a linha de serviço possa estar copiando.
//! A ordem da soma dos experts muda em relação ao caminho de hoje: os logits passam a ser comparados pela tolerância de 1e-4.
#[cutile::module]
pub mod hits_kernels {
    use cutile::core::*;

    /// Lista compacta de acertos. `table` é a tabela do token (slot ou -1, `NO_SLOT` visto como `i32`); `base` é o começo da
    /// linha da camada. `ids_p` e `w_p` apontam os `K` ids e pesos do top-k (memória mapeada ou do device).
    /// `hs[p]` e `hw[p]` são o slot e o peso do p-ésimo acerto (`pad` e 0.0 além da contagem); `hc[0]` é a contagem.
    /// `pad` é um slot que o cache nunca usa nem reescreve (o banco tem um slot a mais, zerado): os grupos além da contagem
    /// leem esse, e não um slot administrado, que a cópia pode estar escrevendo.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn hit_select<const K: i32>(
        hs: &mut Tensor<i32, { [K] }>,
        hw: &mut Tensor<f32, { [K] }>,
        hc: &mut Tensor<i32, { [1] }>,
        ids_p: *mut i32,
        w_p: *mut f32,
        table_p: *mut i32,
        tlen: i32,
        base: i32,
        pad: i32,
    ) {
        let table: Tensor<i32, { [-1] }> = tensor_from_ptr(table_p, tlen);
        let idv: Tensor<i32, { [-1] }> = tensor_from_ptr(ids_p, K);
        let wv: Tensor<f32, { [-1] }> = tensor_from_ptr(w_p, K);
        let pid = idv.partition(const_shape![1]);
        let pw = wv.partition(const_shape![1]);
        let pt = table.partition(const_shape![1]);
        let pos: Tile<i32, { [K] }> = iota(const_shape![K]);
        let zero_i: Tile<i32, { [K] }> = constant(0i32, const_shape![K]);
        let one_i: Tile<i32, { [K] }> = constant(1i32, const_shape![K]);
        let mut cnt: Tile<i32, { [K] }> = constant(0i32, const_shape![K]);
        let mut out_s: Tile<i32, { [K] }> = pad.broadcast(const_shape![K]);
        let mut out_w: Tile<f32, { [K] }> = constant(0.0f32, const_shape![K]);
        for j in 0i32..K {
            let id: i32 = tile_to_scalar(pid.load([j]).reshape(const_shape![]));
            let s: i32 = tile_to_scalar(pt.load([base + id]).reshape(const_shape![]));
            let wj: f32 = tile_to_scalar(pw.load([j]).reshape(const_shape![]));
            let s_t: Tile<i32, { [K] }> = s.broadcast(const_shape![K]);
            let w_t: Tile<f32, { [K] }> = wj.broadcast(const_shape![K]);
            let miss: Tile<bool, { [K] }> = lt_tile(s_t, zero_i);
            let hit_i: Tile<i32, { [K] }> = select(miss, zero_i, one_i);
            let at: Tile<bool, { [K] }> = eq_tile(pos, cnt);
            let at_i: Tile<i32, { [K] }> = select(at, one_i, zero_i) * hit_i;
            let wr: Tile<bool, { [K] }> = eq_tile(at_i, one_i);
            out_s = select(wr, s_t, out_s);
            out_w = select(wr, w_t, out_w);
            cnt = cnt + hit_i;
        }
        let c: Tile<i32, { [1] }> = reduce_max(cnt, 0i32);
        let c: Tile<i32, { [1] }> = c.reshape(const_shape![1]);
        hs.store(out_s);
        hw.store(out_w);
        hc.store(c);
    }

    /// `a[j·I + r] = silu(gate) · up` do acerto `j`, `BM` linhas por bloco. `w` = `[S·2I, H]`, `x` = `[H]`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn moe_gate_up<const BM: i32, const BK: i32, const H: i32, const I: i32>(
        a: &mut Tensor<f32, { [BM] }>,
        w_p: *mut bf16,
        rows: i32,
        x: &Tensor<f32, { [H] }>,
        hs: &Tensor<i32, { [-1] }>,
    ) {
        let wg_shape: Shape<{ [-1, H] }> = Shape::<{ [-1, H] }> { dims: &[rows] };
        let wg_strides: Array<{ [H, 1] }> = Array::<{ [H, 1] }> { dims: &[] };
        let wg_ptr: PointerTile<*mut bf16, { [] }> = pointer_to_tile(w_p);
        let w: Tensor<bf16, { [-1, H] }> = make_tensor_view(wg_ptr, wg_shape, wg_strides, new_token_unordered());
        let part_w = w.partition(const_shape![BM, BK]);
        let part_x = x.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let nr: i32 = I / BM;
        let j: i32 = pid.0 / nr;
        let r: i32 = pid.0 - j * nr;
        let slot: i32 = tile_to_scalar(hs.partition(const_shape![1]).load([j]).reshape(const_shape![]));
        let row: i32 = slot * 2i32 * nr + r;
        let mut acc_g: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
        let mut acc_u: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
        for i in 0i32..(H / BK) {
            let tx: Tile<f32, { [1, BK] }> = part_x.load([i]).reshape(const_shape![1, BK]);
            let txb: Tile<f32, { [BM, BK] }> = tx.broadcast(const_shape![BM, BK]);
            let tg: Tile<bf16, { [BM, BK] }> = part_w.load([row, i]);
            let tu: Tile<bf16, { [BM, BK] }> = part_w.load([row + nr, i]);
            let tg: Tile<f32, { [BM, BK] }> = convert_tile(tg);
            let tu: Tile<f32, { [BM, BK] }> = convert_tile(tu);
            acc_g = acc_g + tg * txb;
            acc_u = acc_u + tu * txb;
        }
        let g: Tile<f32, { [BM] }> = reduce_sum(acc_g, 1i32);
        let g: Tile<f32, { [BM] }> = g.reshape(const_shape![BM]);
        let u: Tile<f32, { [BM] }> = reduce_sum(acc_u, 1i32);
        let u: Tile<f32, { [BM] }> = u.reshape(const_shape![BM]);
        let one: Tile<f32, { [BM] }> = constant(1.0, const_shape![BM]);
        let silu: Tile<f32, { [BM] }> = g / (one + exp(negf(g)));
        a.store(silu * u);
    }

    /// `y = Σ_j peso_j · (W_down[slot_j] · a_j)` sobre os `hc[0]` primeiros acertos, em ordem. `w` = `[S·H, I]`, `a` = `[K·I]`.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn moe_down<const BM: i32, const BK: i32, const H: i32, const I: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        w_p: *mut bf16,
        rows: i32,
        a: &Tensor<f32, { [-1] }>,
        hs: &Tensor<i32, { [-1] }>,
        hw: &Tensor<f32, { [-1] }>,
        hc: &Tensor<i32, { [-1] }>,
    ) {
        let wd_shape: Shape<{ [-1, I] }> = Shape::<{ [-1, I] }> { dims: &[rows] };
        let wd_strides: Array<{ [I, 1] }> = Array::<{ [I, 1] }> { dims: &[] };
        let wd_ptr: PointerTile<*mut bf16, { [] }> = pointer_to_tile(w_p);
        let w: Tensor<bf16, { [-1, I] }> = make_tensor_view(wd_ptr, wd_shape, wd_strides, new_token_unordered());
        let part_w = w.partition(const_shape![BM, BK]);
        let part_a = a.partition(const_shape![BK]);
        let ps = hs.partition(const_shape![1]);
        let pw = hw.partition(const_shape![1]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let nb: i32 = H / BM;
        let cnt: i32 = tile_to_scalar(hc.partition(const_shape![1]).load([0i32]).reshape(const_shape![]));
        let cnt_t: Tile<i32, { [BM] }> = cnt.broadcast(const_shape![BM]);
        let zero: Tile<f32, { [BM] }> = constant(0.0f32, const_shape![BM]);
        let mut tot: Tile<f32, { [BM] }> = constant(0.0f32, const_shape![BM]);
        for j in 0i32..K {
            let slot: i32 = tile_to_scalar(ps.load([j]).reshape(const_shape![]));
            let gw: f32 = tile_to_scalar(pw.load([j]).reshape(const_shape![]));
            let mut acc: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
            for i in 0i32..(I / BK) {
                let tw: Tile<bf16, { [BM, BK] }> = part_w.load([slot * nb + pid.0, i]);
                let ta: Tile<f32, { [1, BK] }> = part_a.load([j * (I / BK) + i]).reshape(const_shape![1, BK]);
                let tw: Tile<f32, { [BM, BK] }> = convert_tile(tw);
                acc = acc + tw * ta.broadcast(const_shape![BM, BK]);
            }
            let r: Tile<f32, { [BM] }> = reduce_sum(acc, 1i32);
            let r: Tile<f32, { [BM] }> = r.reshape(const_shape![BM]);
            let live: Tile<bool, { [BM] }> = lt_tile(j.broadcast(const_shape![BM]), cnt_t);
            tot = tot + select(live, r * gw.broadcast(const_shape![BM]), zero);
        }
        y.store(tot);
    }

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
    use crate::cpu::{Cpu, ExpertRef};
    use crate::gpu::{tile_k, GEMV_BM, SWIGLU_BM};
    use cuda_core::Device;
    use cutile::half::bf16;
    use cutile::prelude::*;
    use cutile::tensor::Reshape;
    use std::sync::Arc;

    const H: usize = 1536;
    const I: usize = 512;
    const E: usize = 40;
    const K: usize = 8;
    /// Slot 0 é o de preenchimento (`pad`): fica com NaN e nunca entra na tabela. As posições além da contagem o leem, e a
    /// máscara tem de impedir que o NaN chegue ao `y`.
    const S: usize = 9;
    const RESIDENTES: [usize; 8] = [3, 8, 11, 17, 25, 33, 5, 29];
    const CAMADA: usize = 1;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((*s >> 32) as u32) as i32) as f32 / (1u64 << 31) as f32
    }

    fn pesos(n: usize, escala: f32, seed: &mut u64) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(lcg(seed) * escala)).collect()
    }

    struct Banco {
        gu: Vec<bf16>,
        dn: Vec<bf16>,
    }

    fn banco(seed: &mut u64) -> Banco {
        let (pg, pd) = (2 * I * H, H * I);
        let mut gu = vec![bf16::NAN; S * pg];
        let mut dn = vec![bf16::NAN; S * pd];
        for s in 1..S {
            gu[s * pg..(s + 1) * pg].copy_from_slice(&pesos(pg, 0.05, seed));
            dn[s * pd..(s + 1) * pd].copy_from_slice(&pesos(pd, 0.05, seed));
        }
        Banco { gu, dn }
    }

    /// Roda os três kernels e devolve (y, contagem, slots, pesos).
    fn gpu(s: &Arc<cuda_core::Stream>, b: &Banco, tabela: &[i32], ids: &[usize], w: &[f32], x: &[f32]) -> (Vec<f32>, i32, Vec<i32>, Vec<f32>) {
        let up = |v: Vec<f32>| api::copy_host_vec_to_device(&Arc::new(v)).sync_on(s).unwrap();
        let bank_gu = api::copy_host_vec_to_device(&Arc::new(b.gu.clone())).sync_on(s).unwrap().reshape(&[S * 2 * I, H]).unwrap();
        let bank_dn = api::copy_host_vec_to_device(&Arc::new(b.dn.clone())).sync_on(s).unwrap().reshape(&[S * H, I]).unwrap();
        let tab = api::copy_host_vec_to_device(&Arc::new(tabela.to_vec())).sync_on(s).unwrap();
        let ids_d = api::copy_host_vec_to_device(&Arc::new(ids.iter().map(|&i| i as i32).collect::<Vec<_>>())).sync_on(s).unwrap();
        let w_d = up(w.to_vec());
        let xd = up(x.to_vec());
        let hs = api::zeros::<i32>(&[K]).sync_on(s).unwrap().partition([K]);
        let hw = api::zeros::<f32>(&[K]).sync_on(s).unwrap().partition([K]);
        let hc = api::zeros::<i32>(&[1]).sync_on(s).unwrap().partition([1]);
        // SAFETY: um bloco; `ids_d` e `w_d` têm K elementos; a tabela tem L·E e a linha da camada cabe nela.
        let (hs, hw, hc, ..) = unsafe {
            hits_kernels::hit_select(hs, hw, hc, ids_d.device_pointer(), w_d.device_pointer(), tab.device_pointer(), (2 * E) as i32, (CAMADA * E) as i32, 0)
                .generics(vec![K.to_string()])
                .sync_on(s)
                .unwrap()
        };
        let (hs, hw, hc) = (hs.unpartition(), hw.unpartition(), hc.unpartition());
        let a = api::zeros::<f32>(&[K * I]).sync_on(s).unwrap().partition([SWIGLU_BM]);
        // SAFETY: H e I são múltiplos de BM e BK; os slots lidos estão em 0..S (a lista só tem slots da tabela ou 0).
        let (a, ..) = unsafe {
            hits_kernels::moe_gate_up(a, bank_gu.device_pointer(), (S * 2 * I) as i32, &xd, &hs)
                .generics(vec![SWIGLU_BM.to_string(), tile_k(H).to_string(), H.to_string(), I.to_string()])
                .sync_on(s)
                .unwrap()
        };
        let a = a.unpartition();
        let y = api::zeros::<f32>(&[H]).sync_on(s).unwrap().partition([GEMV_BM]);
        // SAFETY: como acima; `a` tem K·I elementos.
        let (y, ..) = unsafe {
            hits_kernels::moe_down(y, bank_dn.device_pointer(), (S * H) as i32, &a, &hs, &hw, &hc)
                .generics(vec![GEMV_BM.to_string(), tile_k(I).to_string(), H.to_string(), I.to_string(), K.to_string()])
                .sync_on(s)
                .unwrap()
        };
        let host = |t: &Arc<Tensor<f32>>| api::copy_device_to_host_vec(t).sync_on(s).unwrap();
        let yv = host(&Arc::new(y.unpartition()));
        let cnt = api::copy_device_to_host_vec(&Arc::new(hc)).sync_on(s).unwrap()[0];
        let hsv = api::copy_device_to_host_vec(&Arc::new(hs)).sync_on(s).unwrap();
        let hwv = host(&Arc::new(hw));
        (yv, cnt, hsv, hwv)
    }

    /// A referência: os experts da lista que estão na tabela, na ordem do top-k, pelo caminho de CPU.
    fn ref_cpu(cpu: &Cpu, b: &Banco, tabela: &[i32], ids: &[usize], w: &[f32], x: &[f32], so_ate: usize) -> (Vec<f32>, Vec<i32>) {
        let (pg, pd) = (2 * I * H, H * I);
        let mut refs = Vec::new();
        let mut slots = Vec::new();
        for (&e, &g) in ids.iter().zip(w) {
            let sl = tabela[CAMADA * E + e];
            if sl >= 0 && refs.len() < so_ate {
                let s = sl as usize;
                refs.push(ExpertRef { gate_up: &b.gu[s * pg..(s + 1) * pg], down: &b.dn[s * pd..(s + 1) * pd], weight: g });
                slots.push(sl);
            }
        }
        (if refs.is_empty() { vec![0.0; H] } else { cpu.experts(x, &refs, H, I) }, slots)
    }

    fn erro_rel(a: &[f32], r: &[f32]) -> f32 {
        if a.len() != r.len() || a.iter().chain(r).any(|v| !v.is_finite()) {
            return f32::INFINITY;
        }
        let m = r.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        a.iter().zip(r).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / m
    }

    #[test]
    fn acertos_na_gpu_batem_com_a_cpu_a_1e4() {
        let _trava = crate::gpu_test_lock();
        let device = match Device::new(0) {
            Ok(d) => d,
            Err(e) => {
                assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
                return;
            }
        };
        let s = device.new_stream().unwrap();
        let cpu = Cpu::new(4);
        let mut seed = 11u64;
        let b = banco(&mut seed);
        let mut tabela = vec![-1i32; 2 * E];
        for (k, &e) in RESIDENTES.iter().enumerate() {
            tabela[CAMADA * E + e] = 1 + k as i32;
        }
        let mut pior = 0f32;
        let mut contagens = [0usize; K + 1];
        // Viés nos logits: 0 mistura acertos e faltas; +20 põe os residentes no topo (8 acertos); -20 tira todos (0 acertos).
        for (caso, vies) in [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 20.0, -20.0].iter().cycle().take(40).enumerate() {
            let lv: Vec<f32> = (0..E).map(|e| lcg(&mut seed) * 4.0 + if RESIDENTES.contains(&e) { *vies } else { 0.0 }).collect();
            let r = Routing::from_logits(&lv, K);
            let x: Vec<f32> = (0..H).map(|_| lcg(&mut seed)).collect();
            let (y, cnt, hs, hw) = gpu(&s, &b, &tabela, &r.experts, &r.weights, &x);
            let (yr, slots) = ref_cpu(&cpu, &b, &tabela, &r.experts, &r.weights, &x, K);
            assert_eq!(cnt as usize, slots.len(), "caso {caso}: contagem");
            assert_eq!(&hs[..slots.len()], &slots[..], "caso {caso}: slots na ordem do top-k");
            assert!(hw[slots.len()..].iter().all(|&v| v == 0.0), "caso {caso}: pesos além da contagem");
            contagens[slots.len()] += 1;
            let e = erro_rel(&y, &yr);
            pior = pior.max(e);
            assert!(e <= 1e-4, "caso {caso} ({} acertos): erro relativo {e:e}", slots.len());
            assert!(y.iter().all(|v| v.is_finite()), "caso {caso}: y não finito (o NaN do slot 0 vazou)");
            // Controle embutido: sem o último acerto o resultado tem de ficar fora da tolerância.
            if slots.len() >= 2 {
                let (menos, _) = ref_cpu(&cpu, &b, &tabela, &r.experts, &r.weights, &x, slots.len() - 1);
                assert!(erro_rel(&y, &menos) > 1e-4, "caso {caso}: a tolerância não distingue um acerto a menos");
            }
        }
        eprintln!("pior erro relativo {pior:e}; casos por número de acertos {contagens:?}");
        assert!(contagens[0] > 0 && contagens[K] > 0 && contagens[1..K].iter().any(|&n| n > 0), "faltou cobrir 0, 8 e acertos parciais: {contagens:?}");
    }
}
