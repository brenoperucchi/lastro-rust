//! A2-LOTE (CPU): tempo de UM expert do Granite para n posições juntas, 8 threads, pesos BF16 e ativações f32.
//! Cada linha de peso é lida uma vez da memória e usada pelas n posições (o dot de n=1 é o `cpu::dot` do motor).
//! Critério: qual-runs/A2-LOTE-20261103/criterio.txt. Uso: `lote-cpu [--rapido]` (taskset nos P-cores 0-15 por fora).
//! Saída: linhas `CPU <modelo> n=<n> us_por_expert=<x> max_err_rel=<e>`; qualquer erro > 1e-5 sai com código 3.
use cutile::half::bf16;
use moe_rt::cpu::{dot, swiglu};
use moe_rt::pool::{Out, Pool};
use std::time::Instant;

const ROWS_PER_TASK: usize = 16;
const NEXP: usize = 64;

fn f32_of(b: bf16) -> f32 {
    f32::from_bits((b.to_bits() as u32) << 16)
}

/// Produto de uma linha de peso por T posições (`xs[j*cols..]`), T acumuladores independentes por conjunto, U conjuntos.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn tile<const T: usize, const U: usize>(w: &[bf16], xs: &[f32], cols: usize, out: &mut [f32; 8]) {
    use std::arch::x86_64::*;
    let wp = w.as_ptr() as *const u16;
    let mut acc = [[_mm256_setzero_ps(); T]; U];
    let mut i = 0;
    let step = 8 * U;
    // SAFETY: leituras em i..i+8U <= cols nas duas fatias; xs tem T posições de `cols`.
    unsafe {
        while i + step <= cols {
            for u in 0..U {
                let raw = _mm_loadu_si128(wp.add(i + 8 * u) as *const __m128i);
                let wf = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(raw), 16));
                for j in 0..T {
                    acc[u][j] = _mm256_fmadd_ps(wf, _mm256_loadu_ps(xs.as_ptr().add(j * cols + i + 8 * u)), acc[u][j]);
                }
            }
            i += step;
        }
        for j in 0..T {
            let mut s = acc[0][j];
            for u in 1..U {
                s = _mm256_add_ps(s, acc[u][j]);
            }
            let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
            let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
            let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
            let mut r = _mm_cvtss_f32(s);
            for k in i..cols {
                r += f32_of(w[k]) * xs[j * cols + k];
            }
            out[j] = r;
        }
    }
}

/// `out[j] = w · x_j` para as n posições; n=1 usa o `dot` do motor.
fn dot_n(w: &[bf16], xs: &[f32], cols: usize, n: usize, out: &mut [f32]) {
    if n == 1 {
        out[0] = dot(w, &xs[..cols]);
        return;
    }
    let mut p = 0;
    let mut t = [0f32; 8];
    while p < n {
        let rest = n - p;
        let x = &xs[p * cols..];
        let c = if rest >= 8 { 8 } else if rest >= 4 { 4 } else if rest >= 2 { 2 } else { 1 };
        // SAFETY: AVX2 e FMA conferidos em `main`.
        unsafe {
            match c {
                8 => tile::<8, 1>(w, x, cols, &mut t),
                4 => tile::<4, 2>(w, x, cols, &mut t),
                2 => tile::<2, 2>(w, x, cols, &mut t),
                _ => tile::<1, 4>(w, x, cols, &mut t),
            }
        }
        out[p..p + c].copy_from_slice(&t[..c]);
        p += c;
    }
}

/// Um expert para n posições: saída `y[j*hidden..]` (sem peso do roteador).
fn expert(pool: &Pool, gu: &[bf16], dn: &[bf16], xs: &[f32], hidden: usize, inter: usize, n: usize, y: &mut [f32]) {
    let rows1 = 2 * inter;
    let mut h = vec![0f32; n * rows1];
    let out = Out(h.as_mut_ptr());
    pool.run(rows1.div_ceil(ROWS_PER_TASK), &|t| {
        let (r0, r1) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(rows1));
        let mut o = vec![0f32; n];
        for r in r0..r1 {
            dot_n(&gu[r * hidden..(r + 1) * hidden], xs, hidden, n, &mut o);
            for j in 0..n {
                // SAFETY: (posição, linha) é único por tarefa; `h` vive até `run` voltar.
                unsafe { out.write(j * rows1 + r, o[j]) };
            }
        }
    });
    let mut a = vec![0f32; n * inter];
    for j in 0..n {
        a[j * inter..(j + 1) * inter].copy_from_slice(&swiglu(&h[j * rows1..(j + 1) * rows1]));
    }
    let out = Out(y.as_mut_ptr());
    pool.run(hidden.div_ceil(ROWS_PER_TASK), &|t| {
        let (r0, r1) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(hidden));
        let mut o = vec![0f32; n];
        for r in r0..r1 {
            dot_n(&dn[r * inter..(r + 1) * inter], &a, inter, n, &mut o);
            for j in 0..n {
                // SAFETY: idem.
                unsafe { out.write(j * hidden + r, o[j]) };
            }
        }
    });
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1
    }
}

fn main() {
    assert!(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma"));
    let rapido = std::env::args().any(|a| a == "--rapido");
    let (reps, por_rep) = if rapido { (3, NEXP) } else { (15, 10 * NEXP) };
    let pool = Pool::new(8);
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut falhou = false;
    for (nome, hidden, inter) in [("1b", 1024usize, 512usize), ("3b", 1536, 512)] {
        let gus: Vec<Vec<bf16>> = (0..NEXP).map(|_| (0..2 * inter * hidden).map(|_| bf16::from_f32(rng.f())).collect()).collect();
        let dns: Vec<Vec<bf16>> = (0..NEXP).map(|_| (0..hidden * inter).map(|_| bf16::from_f32(rng.f())).collect()).collect();
        for n in [1usize, 2, 4, 8, 16, 32] {
            let xs: Vec<f32> = (0..n * hidden).map(|_| rng.f() * 10.0).collect();
            // Correção: o lote contra n execuções de n=1, no expert 0.
            let mut yb = vec![0f32; n * hidden];
            expert(&pool, &gus[0], &dns[0], &xs, hidden, inter, n, &mut yb);
            let mut ref_ = vec![0f32; n * hidden];
            for j in 0..n {
                expert(&pool, &gus[0], &dns[0], &xs[j * hidden..(j + 1) * hidden], hidden, inter, 1, &mut ref_[j * hidden..(j + 1) * hidden]);
            }
            let mx = ref_.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-30);
            let err = yb.iter().zip(&ref_).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / mx;
            if !(err <= 1e-5) {
                falhou = true;
            }
            let mut y = vec![0f32; n * hidden];
            for e in 0..NEXP {
                expert(&pool, &gus[e], &dns[e], &xs, hidden, inter, n, &mut y);
            }
            let mut ts = Vec::new();
            for _ in 0..reps {
                let t0 = Instant::now();
                for k in 0..por_rep {
                    let e = k % NEXP;
                    expert(&pool, &gus[e], &dns[e], &xs, hidden, inter, n, &mut y);
                }
                ts.push(t0.elapsed().as_secs_f64() * 1e6 / por_rep as f64);
            }
            ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("CPU {nome} n={n} us_por_expert={:.2} max_err_rel={:.2e}", ts[ts.len() / 2], err);
        }
    }
    if falhou {
        eprintln!("CORRECAO FALHOU");
        std::process::exit(3);
    }
}
