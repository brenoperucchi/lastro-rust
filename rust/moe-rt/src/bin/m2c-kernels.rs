//! Passo 1.5 do corte do custo fixo: o que é cada ~3 µs de GPU por operação do passo (`m2b-custo-fixo`). Para cada kernel de
//! uma camada, com as dimensões do modelo (pesos sintéticos: o tempo não depende dos valores), N lançamentos seguidos na
//! mesma stream:
//! - `fila`: lançados um a um (`async_on`), como no passo de hoje; eventos nas pontas dão o tempo de GPU por lançamento e o
//!   relógio do host dá o custo de enfileirar;
//! - `grafo`: os mesmos N lançamentos capturados num `CudaGraph` e reproduzidos; sobra o tempo do kernel com o menor intervalo
//!   entre kernels que a GPU consegue. A atenção são duas passadas (`attn_split` sobre a grade cabeças × `ATTN_NS`,
//!   `attn_combine` por cabeça). Cada medida se repete `REPS` vezes: mediana e, para a fila, mínimo e máximo.
//!
//! A diferença `fila − grafo` é o intervalo entre lançamentos que um grafo tiraria; o `grafo` contra o piso de banda mostra o
//! quanto o kernel aproveita a placa. Os GEMV do expert aparecem nas duas formas: separados (`gemv` gate_up, `gemv` down e o
//! `axpby`, como até o M2e, sem o SwiGLU, que saiu da biblioteca; o M2e tem a linha dele) e fundidos (`gemv_swiglu`,
//! `gemv_axpby`, desde o M2f). No fim, o custo no host: escalar fixo contra escalar que muda a cada lançamento, e os 2
//! lançamentos de um expert como `gpu::enqueue_expert` os faz (partições próprias, `partition`/`unpartition` a cada
//! lançamento). Os controles do cache de lançamento (M2e) ficam nos binários de `qual-runs/M2D-ATENCAO-20261029/r3/bin`.
//!
//! O i7-14700KF é híbrido: uma medida em que a thread cai num E-core sai com host ~2× (um kernel diferente a cada execução).
//! Rodar com `taskset -c <P-core>`.
//!
//! Os pesos repetidos de um GEMV pequeno e o KV da atenção cabem na L2 (72 MB) até ~16 mil posições no 1B: o tempo aqui é um
//! limite inferior do passo, que lê da DRAM.
//!
//!   m2c-kernels --model <dir> [--n N] [--max-len N]
use cuda_async::cuda_graph::CudaGraph;
use cutile::half::bf16;
use cutile::prelude::*;
use cutile::tensor::Reshape;
use moe_rt::gpu::{gemv_generics, kernels, tile_k, AXPBY_B, GEMV_BM, SWIGLU_BM};
use moe_rt::gpu_model::{ATTN_BN, ATTN_NS};
use moe_rt::model::Config;
use std::path::PathBuf;
use std::time::Instant;

/// Custo de enfileirar no host, em µs por lançamento: `$body` roda `$experts` vezes (com `$e` o índice) e enfileira `$per`
/// lançamentos por volta; uma volta de aquecimento e a mediana de `REPS`.
macro_rules! host_loop {
    ($s:expr, $experts:expr, $per:expr, |$e:ident| $body:block) => {{
        let (s, experts) = (&$s, $experts);
        let mut v = Vec::new();
        for rep in 0..REPS + 1 {
            unsafe { s.synchronize()? };
            let t = Instant::now();
            for $e in 0..experts {
                #[allow(unused_unsafe)]
                unsafe {
                    $body
                }
            }
            if rep > 0 {
                v.push(t.elapsed().as_secs_f64() * 1e6 / ($per * experts) as f64);
            }
        }
        unsafe { s.synchronize()? };
        median(v)
    }};
}

/// Reproduções do grafo medidas (depois de uma de aquecimento).
const REPLAYS: usize = 5;
/// Repetições de cada medida.
const REPS: usize = 3;
/// Banda de referência da RTX 4090 para o piso (a mesma do plano).
const GBPS: f64 = 900.0;

type Res<T> = Result<T, Box<dyn std::error::Error>>;

struct Row {
    name: String,
    blocks: usize,
    bytes: usize,
    fila_us: f64,
    fila_min: f64,
    fila_max: f64,
    host_us: f64,
    grafo_us: f64,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Mede `n` lançamentos de `$op` (uma expressão que monta o lançamento, com `$i` o índice do lançamento) na fila e num grafo.
macro_rules! medir {
    ($rows:expr, $s:expr, $ev:expr, $n:expr, $name:expr, $blocks:expr, $bytes:expr, |$i:ident| $op:expr) => {{
        let (s, (e0, e1), n) = (&$s, &$ev, $n);
        // Aquecimento: compilação JIT e caches.
        for $i in 0..8usize {
            #[allow(unused_unsafe)]
            let _ = unsafe { $op.async_on(s)? };
        }
        unsafe { s.synchronize()? };
        let (mut fila, mut host) = (Vec::new(), Vec::new());
        for _ in 0..REPS {
            e0.record(s)?;
            let t = Instant::now();
            for $i in 0..n {
                #[allow(unused_unsafe)]
                let _ = unsafe { $op.async_on(s)? };
            }
            host.push(t.elapsed().as_secs_f64() * 1e6 / n as f64);
            e1.record(s)?;
            unsafe { s.synchronize()? };
            fila.push(e0.elapsed_time(e1)? as f64 * 1e3 / n as f64);
        }
        let (fila_min, fila_max) = (fila.iter().cloned().fold(f64::INFINITY, f64::min), fila.iter().cloned().fold(0.0, f64::max));
        let (fila_us, host_us) = (median(fila), median(host));
        let g = CudaGraph::scope(s, |sc| {
            for $i in 0..n {
                #[allow(unused_unsafe)]
                let _ = sc.record(unsafe { $op })?;
            }
            Ok(())
        })?;
        unsafe { g.launch().async_on(s)? };
        unsafe { s.synchronize()? };
        let mut grafo = Vec::new();
        for _ in 0..REPS {
            e0.record(s)?;
            for _ in 0..REPLAYS {
                unsafe { g.launch().async_on(s)? };
            }
            e1.record(s)?;
            unsafe { s.synchronize()? };
            grafo.push(e0.elapsed_time(e1)? as f64 * 1e3 / (n * REPLAYS) as f64);
        }
        let grafo_us = median(grafo);
        $rows.push(Row { name: $name.to_string(), blocks: $blocks, bytes: $bytes, fila_us, fila_min, fila_max, host_us, grafo_us });
    }};
}

/// Só o custo de enfileirar no host (sem grafo), para comparar escalar fixo e escalar que muda.
macro_rules! host_only {
    ($s:expr, $n:expr, |$i:ident| $op:expr) => {{
        let (s, n) = (&$s, $n);
        for $i in 0..8usize {
            #[allow(unused_unsafe)]
            let _ = unsafe { $op.async_on(s)? };
        }
        unsafe { s.synchronize()? };
        let t = Instant::now();
        for $i in 0..n {
            #[allow(unused_unsafe)]
            let _ = unsafe { $op.async_on(s)? };
        }
        let us = t.elapsed().as_secs_f64() * 1e6 / n as f64;
        unsafe { s.synchronize()? };
        us
    }};
}

fn main() -> Res<()> {
    let mut a = std::env::args().skip(1);
    let (mut model, mut n, mut max_len) = (PathBuf::new(), 500usize, 16384usize);
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| panic!("{k} sem valor"));
        match k.as_str() {
            "--model" => model = PathBuf::from(v()),
            "--n" => n = v().parse().expect("--n"),
            "--max-len" => max_len = v().parse().expect("--max-len"),
            _ => panic!("opção desconhecida: {k}"),
        }
    }
    let c = Config::from_json(&serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?)?;
    let name = model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let (h, i, d) = (c.hidden, c.inter, c.head_dim);
    let (qd, kvd, half, g) = (c.heads * d, c.kv_heads * d, d / 2, c.heads / c.kv_heads);
    let vocab_p = c.vocab.next_multiple_of(GEMV_BM);

    let device = Device::new(0)?;
    let s = device.new_stream()?;
    let ev = (s.device().new_event()?, s.device().new_event()?);
    let f32z = |len: usize| api::zeros::<f32>(&[len]).sync_on(&s);
    let w = |r: usize, k: usize| -> Res<Tensor<bf16>> { Ok(api::zeros::<bf16>(&[r * k]).sync_on(&s)?.reshape(&[r, k])?) };

    // Pesos de uma camada e da cabeça, buffers como no passo.
    let (wq, wk, wo, router, gate_up, down, lm_head) = (w(qd, h)?, w(kvd, h)?, w(h, qd)?, w(c.experts, h)?, w(2 * i, h)?, w(h, i)?, w(vocab_p, h)?);
    let lnw = f32z(h)?;
    let (mut hs, mut xn, mut q, mut kk, mut o) = (f32z(h)?, f32z(h)?, f32z(qd)?, f32z(kvd)?, f32z(h)?);
    let mut logits_r = f32z(c.experts)?;
    let (mut hh, mut aa, mut oo, mut yy) = (f32z(2 * i)?, f32z(i)?, f32z(h)?, f32z(h)?);
    let mut logits = f32z(vocab_p)?;
    let q2 = f32z(qd)?.reshape(&[c.heads * 2, half])?;
    let mut qr = f32z(qd)?.reshape(&[c.heads * 2, half])?;
    let cs = f32z(2 * half)?.reshape(&[2, half])?;
    let qh = f32z(qd)?.reshape(&[c.heads, d])?;
    let mut att = f32z(qd)?.reshape(&[c.heads, d])?;
    let kc = f32z(max_len * kvd)?.reshape(&[max_len, kvd])?;
    let vc = f32z(max_len * kvd)?.reshape(&[max_len, kvd])?;
    let ns = ATTN_NS;
    let mut part = f32z(c.heads * ns * 2 * d)?.reshape(&[c.heads * ns * 2, d])?;
    let mut kv_dst = f32z(kvd)?;
    let kv_src = f32z(kvd)?;
    // Um bloco só: o menor kernel possível, para o custo de lançamento em si.
    let mut tiny = f32z(AXPBY_B)?;
    let tiny_src = f32z(AXPBY_B)?;

    let norm_c = tile_k(h);
    let gen_norm = vec![h.to_string(), norm_c.to_string()];
    let gen_split = vec![ATTN_BN.to_string(), d.to_string(), g.to_string()];
    let gen_comb = vec![ATTN_BN.to_string(), d.to_string()];
    let i32_dev = |v: i32| api::copy_host_vec_to_device(&std::sync::Arc::new(vec![v])).sync_on(&s);
    let gen_ax = vec![AXPBY_B.to_string()];
    let gen_gu = vec![SWIGLU_BM.to_string(), tile_k(h).to_string(), h.to_string(), i.to_string()];
    let gen_rope = vec![half.to_string()];
    let gemv_bytes = |r: usize, k: usize| r * k * 2 + k * 4 + r * 4;
    let scale = c.attention_multiplier;

    let mut rows: Vec<Row> = Vec::new();
    medir!(rows, s, ev, n, "axpby 1 bloco (mínimo)", 1, AXPBY_B * 12, |_j| kernels::axpby((&mut tiny).partition([AXPBY_B]), &tiny_src, 1.0f32, 1.0f32).generics(gen_ax.clone()));
    medir!(rows, s, ev, n, "rmsnorm", h / norm_c, h * 12, |_j| kernels::rmsnorm((&mut xn).partition([norm_c]), &hs, &lnw, c.eps).generics(gen_norm.clone()));
    medir!(rows, s, ev, n, "gemv q", qd / GEMV_BM, gemv_bytes(qd, h), |_j| kernels::gemv((&mut q).partition([GEMV_BM]), &wq, &xn).generics(gemv_generics(h)));
    medir!(rows, s, ev, n, "gemv k (ou v)", kvd / GEMV_BM, gemv_bytes(kvd, h), |_j| kernels::gemv((&mut kk).partition([GEMV_BM]), &wk, &xn).generics(gemv_generics(h)));
    medir!(rows, s, ev, n, "rope q", c.heads, qd * 12, |_j| kernels::rope((&mut qr).partition([2, half]), &q2, &cs).generics(gen_rope.clone()));
    medir!(rows, s, ev, n, "cópia para o KV", 0, kvd * 8, |_j| api::memcpy(&mut kv_dst, &kv_src));
    for len in [1usize, 64, 256, 1024, 4096, 16384].into_iter().filter(|&l| l <= max_len) {
        let lens = i32_dev(len as i32)?;
        let used = len.div_ceil(ATTN_BN).div_ceil(len.div_ceil(ATTN_BN).div_ceil(ns));
        medir!(rows, s, ev, n, format!("atenção 1ª len={len}"), c.heads * ns, len * kvd * 8, |_j| kernels::attn_split((&mut part).partition([2, d]), &qh, &kc, &vc, &lens, scale, ns as i32).generics(gen_split.clone()));
        medir!(rows, s, ev, n, format!("atenção 2ª len={len}"), c.heads, used * 2 * d * 4 * c.heads, |_j| kernels::attn_combine((&mut att).partition([1, d]), &part, &lens, ns as i32).generics(gen_comb.clone()));
    }
    // O piso de ~13 µs da 1ª passada: custo da grade (cabeças × ns blocos) ou latência de um bloco? Com len pequeno quase todo
    // bloco sai sem trabalho; variar ns separa os dois.
    for ns2 in [8usize, 16, 32, 64] {
        let mut part2 = f32z(c.heads * ns2 * 2 * d)?.reshape(&[c.heads * ns2 * 2, d])?;
        for len in [1usize, 256].into_iter().filter(|&l| l <= max_len) {
            let lens = i32_dev(len as i32)?;
            medir!(rows, s, ev, n, format!("atenção 1ª ns={ns2} len={len}"), c.heads * ns2, len * kvd * 8, |_j| kernels::attn_split((&mut part2).partition([2, d]), &qh, &kc, &vc, &lens, scale, ns2 as i32).generics(gen_split.clone()));
        }
    }
    medir!(rows, s, ev, n, "gemv o", h / GEMV_BM, gemv_bytes(h, qd), |_j| kernels::gemv((&mut o).partition([GEMV_BM]), &wo, &att.view(&[qd])?).generics(gemv_generics(qd)));
    medir!(rows, s, ev, n, "axpby residual", h / AXPBY_B, h * 12, |_j| kernels::axpby((&mut hs).partition([AXPBY_B]), &o, 1.0f32, 1.0f32).generics(gen_ax.clone()));
    medir!(rows, s, ev, n, "gemv roteador", c.experts / GEMV_BM, gemv_bytes(c.experts, h), |_j| kernels::gemv((&mut logits_r).partition([GEMV_BM]), &router, &xn).generics(gemv_generics(h)));
    medir!(rows, s, ev, n, "gemv gate_up", 2 * i / GEMV_BM, gemv_bytes(2 * i, h), |_j| kernels::gemv((&mut hh).partition([GEMV_BM]), &gate_up, &xn).generics(gemv_generics(h)));
    medir!(rows, s, ev, n, "gemv down", h / GEMV_BM, gemv_bytes(h, i), |_j| kernels::gemv((&mut oo).partition([GEMV_BM]), &down, &aa).generics(gemv_generics(i)));
    medir!(rows, s, ev, n, "axpby expert", h / AXPBY_B, h * 12, |_j| kernels::axpby((&mut yy).partition([AXPBY_B]), &oo, 0.5f32, 1.0f32).generics(gen_ax.clone()));
    medir!(rows, s, ev, n, "gemv_swiglu (gate_up)", i / SWIGLU_BM, gemv_bytes(2 * i, h) - i * 4, |_j| kernels::gemv_swiglu((&mut aa).partition([SWIGLU_BM]), &gate_up, &xn).generics(gen_gu.clone()));
    // Comparação com o BM = 2 da primeira versão do M2f (I/2 blocos, metade da grade do GEMV de 2I linhas); produção: `SWIGLU_BM`.
    let gen_gu2 = vec!["2".to_string(), tile_k(h).to_string(), h.to_string(), i.to_string()];
    medir!(rows, s, ev, n, "gemv_swiglu BM=2", i / 2, gemv_bytes(2 * i, h) - i * 4, |_j| kernels::gemv_swiglu((&mut aa).partition([2]), &gate_up, &xn).generics(gen_gu2.clone()));
    let gen_ax1 = vec!["1".to_string(), tile_k(i).to_string(), i.to_string()];
    medir!(rows, s, ev, n, "gemv_axpby BM=1", h, gemv_bytes(h, i) + h * 4, |_j| kernels::gemv_axpby((&mut yy).partition([1]), &down, &aa, 0.5f32, 1.0f32).generics(gen_ax1.clone()));
    medir!(rows, s, ev, n, "gemv_axpby (down)", h / GEMV_BM, gemv_bytes(h, i) + h * 4, |_j| kernels::gemv_axpby((&mut yy).partition([GEMV_BM]), &down, &aa, 0.5f32, 1.0f32).generics(gemv_generics(i)));
    medir!(rows, s, ev, n / 4, "gemv cabeça", vocab_p / GEMV_BM, gemv_bytes(vocab_p, h), |_j| kernels::gemv((&mut logits).partition([GEMV_BM]), &lm_head, &xn).generics(gemv_generics(h)));

    println!(
        "m2c-kernels: {name}; H = {h}, I = {i}, {} cabeças de Q e {} de KV (d = {d}), E = {}; max_len {max_len}; {n} lançamentos por kernel, {REPS} repetições, grafo reproduzido {REPLAYS}×",
        c.heads,
        c.kv_heads,
        c.experts
    );
    println!(
        "\n{:<24} | {:>6} | {:>9} {:>13} {:>9} {:>9} | {:>9} | {:>8} {:>8}",
        "kernel", "blocos", "fila µs", "fila mín–máx", "grafo µs", "diferença", "host µs", "piso µs", "GB/s"
    );
    for r in &rows {
        let floor = r.bytes as f64 / (GBPS * 1e3);
        println!(
            "{:<24} | {:>6} | {:>9.2} {:>13} {:>9.2} {:>9.2} | {:>9.2} | {:>8.2} {:>8.0}",
            r.name,
            r.blocks,
            r.fila_us,
            format!("{:.2}–{:.2}", r.fila_min, r.fila_max),
            r.grafo_us,
            r.fila_us - r.grafo_us,
            r.host_us,
            floor,
            r.bytes as f64 / (r.grafo_us * 1e3)
        );
    }
    println!("\nfila: GPU por lançamento com lançamentos um a um (mediana e faixa das repetições); grafo: o mesmo capturado num CudaGraph;");
    println!("diferença: o que o grafo tira (só limpa onde a GPU limita: nos kernels pequenos a fila fica perto do host);");
    println!("host: custo de enfileirar um lançamento; piso: bytes lidos e escritos a {GBPS} GB/s; GB/s: banda efetiva no grafo.");
    println!("(atenção: K e V das len posições, sem contar a releitura das cabeças de Q que dividem a mesma cabeça de KV.)");

    // Escalar fixo contra escalar que muda a cada lançamento.
    let ax_fixo = host_only!(s, n, |_j| kernels::axpby((&mut yy).partition([AXPBY_B]), &oo, 0.5f32, 1.0f32).generics(gen_ax.clone()));
    let ax_muda = host_only!(s, n, |j| kernels::axpby((&mut yy).partition([AXPBY_B]), &oo, 0.5f32 + j as f32 * 1e-3, 1.0f32).generics(gen_ax.clone()));
    println!("\ncusto de enfileirar no host (µs por lançamento): escalar fixo contra escalar que muda a cada lançamento");
    println!("  (a atenção lê len de um tensor da GPU desde o M2d; até o M2c, o len escalar custava 4,75 µs contra 1,31 fixo)");
    println!("  axpby, g (f32):     {ax_fixo:>7.2}  {ax_muda:>7.2}");

    // Os 2 lançamentos de um expert como `gpu::enqueue_expert` (buffers movidos para dentro do lançamento e de volta), contra
    // o mesmo lançamento repetido (o cache de lançamento sempre acerta).
    let gen_i = gemv_generics(i);
    let experts = (n / 4).max(1);
    let (mut pa, mut py) = (f32z(i)?.partition([SWIGLU_BM]), f32z(h)?.partition([GEMV_BM]));
    let expert = host_loop!(s, experts, 2, |e| {
        let beta = if e % 8 == 0 { 0.0f32 } else { 1.0 };
        let (aa2, _, _) = kernels::gemv_swiglu(pa, &gate_up, &xn).generics(gen_gu.clone()).async_on(&s)?;
        let aa2 = aa2.unpartition();
        let (yy2, _, _, _, _) = kernels::gemv_axpby(py, &down, &aa2, 0.5f32, beta).generics(gen_i.clone()).async_on(&s)?;
        py = yy2;
        pa = aa2.partition([SWIGLU_BM]);
    });
    let repetido = host_loop!(s, experts, 1, |_e| {
        kernels::gemv_swiglu((&mut aa).partition([SWIGLU_BM]), &gate_up, &xn).generics(gen_gu.clone()).async_on(&s)?;
    });
    println!("\ncusto de enfileirar no host, por lançamento, nos 2 lançamentos de um expert ({experts} experts, mediana de {REPS}):");
    println!("  como em gpu::enqueue_expert (partições próprias):  {expert:>7.2}");
    println!("  referência: só o gemv_swiglu, repetido:           {repetido:>7.2}");
    Ok(())
}
