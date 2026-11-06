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

    /// `y[r] = f(r)` para todas as linhas, em faixas de [`ROWS_PER_TASK`] repartidas pelas threads do `Pool`. É o caminho de
    /// escrita das saídas da CPU (`gemv` e a soma dos experts): cada tarefa escreve só a sua faixa, com stores comuns (nenhum
    /// store não temporal em `rust/moe-rt/src`, o que `tests/sem_store_nao_temporal.rs` confere), e `run` só volta quando
    /// todas terminaram. Quem grava uma flag com `Release` depois deste retorno publica as escritas das threads do `Pool`
    /// (A1.3b; o conferidor `a1-pool-y` exercita isso com `y` em memória host mapeada lida pela GPU).
    pub fn fill_rows(&self, y: &mut [f32], f: &(dyn Fn(usize) -> f32 + Sync)) {
        let rows = y.len();
        let out = Out(y.as_mut_ptr());
        self.pool.run(rows.div_ceil(ROWS_PER_TASK), &|t| {
            let (r0, r1) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(rows));
            for r in r0..r1 {
                // SAFETY: a tarefa `t` escreve só as linhas `r0..r1`, disjuntas das outras; `y` vive até `run` voltar.
                unsafe { out.write(r, f(r)) };
            }
        });
    }

    /// `y = W·x`, W em linhas (`rows × cols`).
    pub fn gemv(&self, w: &[bf16], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) {
        assert_eq!(w.len(), rows * cols);
        assert_eq!(x.len(), cols);
        assert_eq!(y.len(), rows);
        self.fill_rows(y, &|r| dot(&w[r * cols..(r + 1) * cols], x));
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
        self.fill_rows(&mut y, &|r| {
            let mut acc = 0f32;
            for (e, ae) in list.iter().zip(&a) {
                acc += e.weight * dot(&e.down[r * inter..(r + 1) * inter], ae);
            }
            acc
        });
        y
    }
}


/// Um expert de um passo de várias posições: os pesos na RAM e, para cada posição que o pede, o `x` dela e o peso do softmax.
pub struct StepExpert<'a> {
    pub gate_up: &'a [bf16],
    pub down: &'a [bf16],
    /// `(posição do passo, peso do softmax)` das posições que pedem o expert, na ordem do passo.
    pub members: Vec<(usize, f32)>,
}

/// `out[j] = w · x_j` para as `n` posições de `xs` (E4b: cada linha de peso é lida uma vez e usada pelas `n` posições; o kernel
/// de `lote-cpu`, A2-LOTE). `n = 1` usa o `dot` do motor.
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

/// Produto de uma linha de peso por `n` posições (`xs[j*cols..]`), em blocos de 8, 4, 2 ou 1 posições.
fn dot_n(w: &[bf16], xs: &[f32], cols: usize, n: usize, out: &mut [f32]) {
    if n == 1 {
        out[0] = dot(w, &xs[..cols]);
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        let mut p = 0;
        let mut t = [0f32; 8];
        while p < n {
            let rest = n - p;
            let x = &xs[p * cols..];
            let c = if rest >= 8 { 8 } else if rest >= 4 { 4 } else if rest >= 2 { 2 } else { 1 };
            // SAFETY: AVX2 e FMA conferidos acima; `x` tem pelo menos `c` posições de `cols`.
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
        return;
    }
    for j in 0..n {
        out[j] = dot(w, &xs[j * cols..(j + 1) * cols]);
    }
}

impl Cpu {
    /// O passo de `m` posições na CPU (E4b): `y[p] = Σ_e peso(e, p) · expert_e(x[p])` sobre os experts da CPU da camada, com cada linha
    /// de peso lida uma vez e usada por todas as posições que pedem o expert. `x` e `y` têm `m·hidden` palavras; `y` é escrito inteiro
    /// (posições sem faltas ficam em zero). Dois despachos do `Pool`, como o `experts`. A ordem de soma entre experts é a da lista (o
    /// de primeira aparição no passo), então o resultado difere do caminho por posição só na ordem de soma de ponto flutuante.
    pub fn experts_step(&self, x: &[f32], list: &[StepExpert], m: usize, hidden: usize, inter: usize) -> Vec<f32> {
        assert_eq!(x.len(), m * hidden);
        let mut y = vec![0f32; m * hidden];
        if list.is_empty() {
            return y;
        }
        // Uma linha de `a` por (expert, membro): `a[off[e] + j]` com `j` o índice do membro.
        let mut off = Vec::with_capacity(list.len());
        let mut tot = 0;
        for e in list {
            assert_eq!(e.gate_up.len(), 2 * inter * hidden);
            assert_eq!(e.down.len(), hidden * inter);
            assert!(!e.members.is_empty() && e.members.len() <= 32 && e.members.iter().all(|&(p, _)| p < m));
            off.push(tot);
            tot += e.members.len();
        }
        // x dos membros de cada expert, contíguo.
        let xs: Vec<Vec<f32>> = list.iter().map(|e| e.members.iter().flat_map(|&(p, _)| x[p * hidden..(p + 1) * hidden].iter().copied()).collect()).collect();
        let mut a = vec![0f32; tot * inter];
        let per = inter.div_ceil(ROWS_PER_TASK);
        let out = Out(a.as_mut_ptr());
        self.pool.run(list.len() * per, &|t| {
            let (e, c) = (t / per, t % per);
            let n = list[e].members.len();
            let (r0, r1) = (c * ROWS_PER_TASK, ((c + 1) * ROWS_PER_TASK).min(inter));
            let (mut og, mut ou) = ([0f32; 32], [0f32; 32]);
            for r in r0..r1 {
                dot_n(&list[e].gate_up[r * hidden..(r + 1) * hidden], &xs[e], hidden, n, &mut og);
                dot_n(&list[e].gate_up[(inter + r) * hidden..(inter + r + 1) * hidden], &xs[e], hidden, n, &mut ou);
                for j in 0..n {
                    // SAFETY: (expert, membro, linha) é único por tarefa; `a` vive até `run` voltar.
                    unsafe { out.write((off[e] + j) * inter + r, og[j] / (1.0 + (-og[j]).exp()) * ou[j]) };
                }
            }
        });
        let out = Out(y.as_mut_ptr());
        let rows_per = hidden.div_ceil(ROWS_PER_TASK);
        self.pool.run(rows_per, &|t| {
            let (r0, r1) = (t * ROWS_PER_TASK, ((t + 1) * ROWS_PER_TASK).min(hidden));
            let mut o = [0f32; 32];
            let mut acc = vec![0f32; m];
            for r in r0..r1 {
                acc.iter_mut().for_each(|v| *v = 0.0);
                for (e, ex) in list.iter().enumerate() {
                    let n = ex.members.len();
                    dot_n(&ex.down[r * inter..(r + 1) * inter], &a[off[e] * inter..(off[e] + n) * inter], inter, n, &mut o);
                    for (j, &(p, w)) in ex.members.iter().enumerate() {
                        acc[p] += w * o[j];
                    }
                }
                for p in 0..m {
                    // SAFETY: a tarefa escreve só as linhas `r0..r1` de cada posição; `y` vive até `run` voltar.
                    unsafe { out.write(p * hidden + r, acc[p]) };
                }
            }
        });
        y
    }
}

#[cfg(test)]
mod testes_e4b {
    use super::*;

    fn lcg(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    fn rnd(n: usize, scale: f32, s: &mut u64) -> Vec<bf16> {
        (0..n).map(|_| bf16::from_f32(lcg(s) * scale)).collect()
    }

    /// O passo em lote contra o caminho por posição (`Cpu::experts` por posição, só com os experts que ela pede), para M em 1, 2, 4, 8
    /// e membros por expert de 1 a M. A tolerância é a do critério do E4b (1e-5 relativo à norma máxima do vetor).
    #[test]
    fn experts_step_bate_com_o_caminho_por_posicao() {
        let (h, i, ne) = (96usize, 40usize, 6usize);
        let mut s = 7u64;
        let ws: Vec<(Vec<bf16>, Vec<bf16>)> = (0..ne).map(|_| (rnd(2 * i * h, 0.3, &mut s), rnd(h * i, 0.3, &mut s))).collect();
        let cpu = Cpu::new(4);
        for m in [1usize, 2, 4, 8] {
            for seed in 0..6u64 {
                let mut r = 1000 * m as u64 + seed;
                let x: Vec<f32> = (0..m * h).map(|_| lcg(&mut r)).collect();
                // Cada posição pede um subconjunto não vazio de experts, com pesos próprios.
                let mut pedidos: Vec<Vec<(usize, f32)>> = Vec::new();
                for _ in 0..m {
                    let mut v = Vec::new();
                    for e in 0..ne {
                        let sorteio = lcg(&mut r) + 0.5;
                        if sorteio < 0.55 || e == 0 {
                            v.push((e, lcg(&mut r) + 0.6));
                        }
                    }
                    pedidos.push(v);
                }
                let mut lista: Vec<StepExpert> = Vec::new();
                let mut ids = Vec::new();
                for p in 0..m {
                    for &(e, w) in &pedidos[p] {
                        match ids.iter().position(|&k| k == e) {
                            Some(g) => lista[g].members.push((p, w)),
                            None => {
                                ids.push(e);
                                lista.push(StepExpert { gate_up: &ws[e].0, down: &ws[e].1, members: vec![(p, w)] });
                            }
                        }
                    }
                }
                let got = cpu.experts_step(&x, &lista, m, h, i);
                let mut want = Vec::new();
                for p in 0..m {
                    let refs: Vec<ExpertRef> = pedidos[p].iter().map(|&(e, w)| ExpertRef { gate_up: &ws[e].0, down: &ws[e].1, weight: w }).collect();
                    want.extend(cpu.experts(&x[p * h..(p + 1) * h], &refs, h, i));
                }
                for p in 0..m {
                    let (g, w) = (&got[p * h..(p + 1) * h], &want[p * h..(p + 1) * h]);
                    let norma = w.iter().fold(0f32, |a, v| a.max(v.abs())).max(1e-12);
                    let err = g.iter().zip(w).fold(0f32, |a, (u, v)| a.max((u - v).abs())) / norma;
                    assert!(err <= 1e-5, "m = {m}, semente {seed}, posição {p}: erro relativo {err}");
                }
            }
        }
    }

    /// Uma posição sem pedidos fica em zero, e uma lista vazia dá tudo zero (o `y` ativo é escrito inteiro).
    #[test]
    fn experts_step_posicao_sem_pedido_fica_em_zero() {
        let (h, i) = (64usize, 32usize);
        let mut s = 3u64;
        let (gu, dn) = (rnd(2 * i * h, 0.3, &mut s), rnd(h * i, 0.3, &mut s));
        let cpu = Cpu::new(2);
        let x: Vec<f32> = (0..3 * h).map(|k| (k % 7) as f32 * 0.1).collect();
        let y = cpu.experts_step(&x, &[StepExpert { gate_up: &gu, down: &dn, members: vec![(1, 0.5)] }], 3, h, i);
        assert!(y[..h].iter().all(|v| *v == 0.0) && y[2 * h..].iter().all(|v| *v == 0.0));
        assert!(y[h..2 * h].iter().any(|v| *v != 0.0));
        assert!(cpu.experts_step(&x, &[], 3, h, i).iter().all(|v| *v == 0.0));
    }
}

#[cfg(test)]
mod testes_a13b {
    use super::*;

    #[test]
    fn fill_rows_escreve_cada_linha_uma_vez() {
        let cpu = Cpu::new(4);
        for n in [0usize, 1, 7, 64, 1536, 1537] {
            let mut y = vec![-1f32; n];
            cpu.fill_rows(&mut y, &|r| r as f32 * 2.0);
            assert!(y.iter().enumerate().all(|(r, v)| *v == r as f32 * 2.0), "n = {n}");
        }
    }
}
