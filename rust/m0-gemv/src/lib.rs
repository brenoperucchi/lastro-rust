//! Kernels e utilitários compartilhados pelos benchmarks do M0 (plano `docs/PLANO-runtime-rust-moe-20261029.md`).
use cutile::half::bf16;

#[cutile::module]
pub mod kernels {
    use cutile::core::*;

    /// Cada bloco calcula `BM` linhas de `y = W·x`: percorre K em fatias de `BK`, acumula produtos em f32 e reduz.
    #[cutile::entry(unchecked_accesses = true)]
    unsafe fn gemv<const BM: i32, const BK: i32, const K: i32>(
        y: &mut Tensor<f32, { [BM] }>,
        w: &Tensor<bf16, { [-1, K] }>,
        x: &Tensor<bf16, { [K] }>,
    ) {
        let part_w = w.partition(const_shape![BM, BK]);
        let part_x = x.partition(const_shape![BK]);
        let pid: (i32, i32, i32) = get_tile_block_id();
        let mut acc: Tile<f32, { [BM, BK] }> = constant(0.0, const_shape![BM, BK]);
        for i in 0i32..(K / BK) {
            let tw: Tile<bf16, { [BM, BK] }> = part_w.load([pid.0, i]);
            let tx: Tile<bf16, { [1, BK] }> = part_x.load([i]).reshape(const_shape![1, BK]);
            let tw: Tile<f32, { [BM, BK] }> = convert_tile(tw);
            let tx: Tile<f32, { [1, BK] }> = convert_tile(tx);
            acc = acc + tw * tx.broadcast(const_shape![BM, BK]);
        }
        let r: Tile<f32, { [BM] }> = reduce_sum(acc, 1i32);
        y.store(r);
    }
}

/// Gerador determinístico simples (xorshift) para valores em [-1, 1).
pub fn host_values(n: usize, seed: u64) -> Vec<bf16> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            bf16::from_f32(((s >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0)
        })
        .collect()
}
