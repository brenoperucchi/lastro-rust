//! Caminho de CPU: GEMV com pesos BF16 e ativações f32 sobre um conjunto fixo de threads ([`Pool`]). É como o runtime calcula
//! um expert que não está na VRAM (no M0, calcular na CPU saiu mais barato que copiar um expert pequeno em batch 1).
//!
//! O produto escalar usa AVX2 + FMA quando a CPU tem (BF16 → f32 é deslocar 16 bits), senão um laço escalar. Os k experts de
//! uma camada vão em dois despachos: todas as linhas de gate_up dos k, depois todas as linhas de down já somando no `y`.
use crate::pool::{Out, Pool};
use cutile::half::bf16;

/// Linhas por tarefa: pequeno o bastante para equilibrar entre threads, grande o bastante para o contador atômico não pesar.
const ROWS_PER_TASK: usize = 16;

#[inline]
fn f32_of(b: bf16) -> f32 {
    f32::from_bits((b.to_bits() as u32) << 16)
}

fn dot_scalar(w: &[bf16], x: &[f32]) -> f32 {
    let mut acc = [0f32; 4];
    for (wc, xc) in w.chunks_exact(4).zip(x.chunks_exact(4)) {
        for j in 0..4 {
            acc[j] += f32_of(wc[j]) * xc[j];
        }
    }
    let tail: f32 = w.chunks_exact(4).remainder().iter().zip(x.chunks_exact(4).remainder()).map(|(a, b)| f32_of(*a) * b).sum();
    acc[0] + acc[1] + acc[2] + acc[3] + tail
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(w: &[bf16], x: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let n = w.len().min(x.len());
    let (wp, xp) = (w.as_ptr() as *const u16, x.as_ptr());
    let mut acc = [_mm256_setzero_ps(); 4];
    let mut i = 0;
    // SAFETY: todas as leituras estão em `i..i + 32 <= n`, dentro das duas fatias.
    unsafe {
        while i + 32 <= n {
            for (j, a) in acc.iter_mut().enumerate() {
                let raw = _mm_loadu_si128(wp.add(i + 8 * j) as *const __m128i);
                let wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
                *a = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xp.add(i + 8 * j)), *a);
            }
            i += 32;
        }
        let s = _mm256_add_ps(_mm256_add_ps(acc[0], acc[1]), _mm256_add_ps(acc[2], acc[3]));
        let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s) + dot_scalar(&w[i..n], &x[i..n])
    }
}

/// Produto escalar de uma linha BF16 com `x`.
pub fn dot(w: &[bf16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2 e FMA conferidos acima.
        return unsafe { dot_avx2(w, x) };
    }
    dot_scalar(w, x)
}

/// `silu(gate) · up`, com `h = [gate; up]` (as duas metades de `gate_up`).
pub fn swiglu(h: &[f32]) -> Vec<f32> {
    let i = h.len() / 2;
    (0..i).map(|j| h[j] / (1.0 + (-h[j]).exp()) * h[i + j]).collect()
}

/// Um expert escolhido pelo roteador: pesos na RAM e peso do softmax.
pub struct ExpertRef<'a> {
    pub gate_up: &'a [bf16],
    pub down: &'a [bf16],
    pub weight: f32,
}

pub struct Cpu {
    pool: Pool,
}

impl Cpu {
    pub fn new(threads: usize) -> Self {
        Self { pool: Pool::new(threads) }
    }

    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    /// `y = W·x`, W em linhas (`rows × cols`).
    pub fn gemv(&self, w: &[bf16], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
        assert_eq!(w.len(), rows * cols);
        assert_eq!(x.len(), cols);
        assert_eq!(y.len(), rows);
        let out = Out(y.as_mut_ptr());
        self.pool.run(rows.div_ceil(ROWS_PER_TASK), &|t| {
            let (r0, r1) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(rows));
            for r in r0..r1 {
                // SAFETY: a tarefa `t` escreve só as linhas `r0..r1`, disjuntas das outras; `y` vive até `run` voltar.
                unsafe { out.write(r, dot(&w[r * cols..(r + 1) * cols], x)) };
            }
        });
    }

    /// `Σ g_e · down_e(silu(gate_e) ⊙ up_e)` para os experts dados, com `[gate; up] = gate_up_e·x`.
    pub fn experts(&self, x: &[f32], list: &[ExpertRef], hidden: usize, inter: usize) -> Vec<f32> {
        assert_eq!(x.len(), hidden);
        let ne = list.len();
        let mut y = vec![0f32; hidden];
        if ne == 0 {
            return y;
        }
        for e in list {
            assert_eq!(e.gate_up.len(), 2 * inter * hidden);
            assert_eq!(e.down.len(), hidden * inter);
        }
        // Despacho 1: as 2I linhas de gate_up de cada expert.
        let mut h = vec![0f32; ne * 2 * inter];
        let per = (2 * inter).div_ceil(ROWS_PER_TASK);
        let out = Out(h.as_mut_ptr());
        self.pool.run(ne * per, &|t| {
            let (e, c) = (t / per, t % per);
            let (r0, r1) = (c * ROWS_PER_TASK, ((c + 1) * ROWS_PER_TASK).min(2 * inter));
            let w = list[e].gate_up;
            for r in r0..r1 {
                // SAFETY: (expert, faixa de linhas) é único por tarefa; `h` vive até `run` voltar.
                unsafe { out.write(e * 2 * inter + r, dot(&w[r * hidden..(r + 1) * hidden], x)) };
            }
        });
        let a: Vec<Vec<f32>> = h.chunks_exact(2 * inter).map(swiglu).collect();
        // Despacho 2: cada tarefa faz uma faixa de linhas de `y` somando os experts.
        let out = Out(y.as_mut_ptr());
        self.pool.run(hidden.div_ceil(ROWS_PER_TASK), &|t| {
            let (r0, r1) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(hidden));
            for r in r0..r1 {
                let mut acc = 0f32;
                for (e, ae) in list.iter().zip(&a) {
                    acc += e.weight * dot(&e.down[r * inter..(r + 1) * inter], ae);
                }
                // SAFETY: a faixa `r0..r1` é só desta tarefa; `y` vive até `run` voltar.
                unsafe { out.write(r, acc) };
            }
        });
        y
    }
}
