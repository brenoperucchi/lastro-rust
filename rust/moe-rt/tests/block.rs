//! Testes do M1a sem o checkpoint real: roteador, GEMV e SwiGLU da CPU, leitor de safetensors e o bloco inteiro numa
//! camada sintética pequena (CPU contra uma implementação ingênua em f64; GPU contra a CPU, se houver GPU).
use cutile::half::bf16;
use moe_rt::block::{HostLayer, Routing};
use moe_rt::cpu;
use moe_rt::gpu::GpuLayer;
use moe_rt::weights::Checkpoint;
use std::fs;

mod common;
use common::{bf, bytes, gpu_or_skip, scratch, values, write_safetensors, Scratch};

#[test]
fn roteador_escolhe_top_k_em_ordem_e_desempata_pelo_menor_indice() {
    let r = Routing::from_logits(&[0.5, 2.0, -1.0, 2.0, 1.0], 3);
    assert_eq!(r.experts, vec![1, 3, 4]);
    let ex: Vec<f32> = [2.0f32, 2.0, 1.0].iter().map(|v| v.exp()).collect();
    let sum: f32 = ex.iter().sum();
    for (w, e) in r.weights.iter().zip(&ex) {
        assert!((w - e / sum).abs() < 1e-6);
    }
    assert!((r.weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn gemv_da_cpu_bate_com_f64_com_sobra_de_colunas_e_mais_threads_que_linhas() {
    for &(rows, cols, threads) in &[(7usize, 13usize, 3usize), (3, 64, 8), (64, 1024, 4), (1, 5, 1)] {
        let w = bf(&values(rows * cols, 11 + rows as u64));
        let x = values(cols, 7);
        let mut y = vec![0f32; rows];
        cpu::Cpu::new(threads).gemv(&w, rows, cols, &x, &mut y);
        for r in 0..rows {
            let want: f64 = (0..cols).map(|c| w[r * cols + c].to_f32() as f64 * x[c] as f64).sum();
            assert!((y[r] as f64 - want).abs() <= 1e-5 * (1.0 + want.abs()), "{rows}x{cols} linha {r}");
        }
    }
}

#[test]
fn produto_escalar_bate_com_f64_em_tamanhos_com_e_sem_sobra_de_32() {
    for &n in &[0usize, 1, 7, 31, 32, 33, 64, 100, 1024, 1029] {
        let w = bf(&values(n, 3 + n as u64));
        let x = values(n, 9);
        let want: f64 = w.iter().zip(&x).map(|(a, b)| a.to_f32() as f64 * *b as f64).sum();
        let got = cpu::dot(&w, &x) as f64;
        assert!((got - want).abs() <= 1e-5 * (1.0 + want.abs()), "n={n}: {got} vs {want}");
    }
}

#[test]
fn pool_cobre_cada_tarefa_uma_vez_em_muitos_despachos_seguidos() {
    use moe_rt::pool::Pool;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let pool = Pool::new(6);
    for round in 0..2000usize {
        let n = round % 37;
        let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
        pool.run(n, &|i| {
            hits[i].fetch_add(1, Ordering::Relaxed);
        });
        assert!(hits.iter().all(|h| h.load(Ordering::Relaxed) == 1), "despacho {round}");
    }
}

#[test]
fn pool_propaga_panico_de_tarefa_e_continua_utilizavel() {
    use moe_rt::pool::Pool;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let pool = Pool::new(4);
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.run(64, &|i| {
            if i == 17 {
                panic!("tarefa 17");
            }
        })
    }));
    assert!(r.is_err());
    let count = AtomicUsize::new(0);
    pool.run(100, &|_| {
        count.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(count.load(Ordering::Relaxed), 100);
}

#[test]
fn pool_com_panico_de_quem_chama_e_de_uma_thread_no_mesmo_despacho_nao_contamina_o_seguinte() {
    use moe_rt::pool::Pool;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    let pool = Pool::new(4);
    let caller = std::thread::current().id();
    let (caller_in, worker_in) = (AtomicBool::new(false), AtomicBool::new(false));
    // Cada lado espera o outro entrar numa tarefa antes de entrar em pânico, para os dois pânicos caírem no mesmo despacho.
    let wait_for = |flag: &AtomicBool| {
        let t0 = Instant::now();
        while !flag.load(Ordering::Acquire) && t0.elapsed() < Duration::from_secs(20) {
            std::hint::spin_loop();
        }
    };
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pool.run(64, &|_| {
            if std::thread::current().id() == caller {
                caller_in.store(true, Ordering::Release);
                wait_for(&worker_in);
                panic!("tarefa de quem chama");
            }
            worker_in.store(true, Ordering::Release);
            wait_for(&caller_in);
            panic!("tarefa de uma thread");
        })
    }));
    assert!(r.is_err());
    assert!(caller_in.load(Ordering::Relaxed) && worker_in.load(Ordering::Relaxed), "os dois pânicos não caíram no mesmo despacho");
    let count = AtomicUsize::new(0);
    pool.run(100, &|_| {
        count.fetch_add(1, Ordering::Relaxed);
    });
    assert_eq!(count.load(Ordering::Relaxed), 100);
}

#[test]
fn swiglu_e_silu_da_primeira_metade_vezes_a_segunda() {
    let a = cpu::swiglu(&[0.0, 1.0, -2.0, 3.0, 4.0, 5.0]);
    let silu = |g: f32| g / (1.0 + (-g).exp());
    let want = [silu(0.0) * 3.0, silu(1.0) * 4.0, silu(-2.0) * 5.0];
    for (g, w) in a.iter().zip(want) {
        assert!((g - w).abs() < 1e-6);
    }
}

#[test]
fn leitor_de_safetensors_le_bf16_e_recusa_dtype_ausente_e_offsets_ruins() {
    let tmp = scratch("leitor");
    let v = bf(&values(6, 3));
    write_safetensors(&tmp.0, &[("a", "BF16", vec![2, 3], bytes(&v)), ("b", "F32", vec![1], vec![0; 4])]);
    let ck = Checkpoint::open(tmp.0.parent().unwrap()).unwrap();
    let (got, shape) = ck.bf16("a").unwrap();
    assert_eq!(shape, vec![2, 3]);
    assert_eq!(got, v);
    assert!(ck.bf16("b").unwrap_err().to_string().contains("F32"));
    assert!(ck.bf16("c").unwrap_err().to_string().contains("ausente"));

    // Offsets além do fim do arquivo.
    let mut raw = fs::read(&tmp.0).unwrap();
    raw.truncate(raw.len() - 6);
    fs::write(&tmp.0, raw).unwrap();
    let err = Checkpoint::open(tmp.0.parent().unwrap()).err().expect("deveria recusar");
    assert!(err.to_string().contains("fora do arquivo"), "{err}");
}

#[test]
fn leitor_de_safetensors_recusa_cabecalho_malformado_sem_entrar_em_panico() {
    let tmp = scratch("leitor-malformado");
    let write_head = |entry: serde_json::Value| {
        let head = serde_json::to_vec(&serde_json::json!({ "a": entry })).unwrap();
        let mut out = (head.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&head);
        out.extend_from_slice(&[0u8; 16]);
        fs::write(&tmp.0, out).unwrap();
    };
    let bad = [
        serde_json::json!({"dtype": "BF16", "shape": [2], "data_offsets": [0]}),
        serde_json::json!({"dtype": "BF16", "shape": [2], "data_offsets": []}),
        serde_json::json!({"dtype": "BF16", "shape": [2], "data_offsets": [0, 4, 8]}),
        serde_json::json!({"dtype": "BF16", "shape": [2], "data_offsets": ["0", 4]}),
        serde_json::json!({"dtype": "BF16", "shape": [2], "data_offsets": [u64::MAX, u64::MAX]}),
        serde_json::json!({"dtype": "BF16", "shape": [2], "data_offsets": [u64::MAX - 4, u64::MAX]}),
    ];
    for entry in bad {
        write_head(entry.clone());
        let r = std::panic::catch_unwind(|| Checkpoint::open(tmp.0.parent().unwrap()).err());
        let err = r.unwrap_or_else(|_| panic!("pânico em vez de erro para {entry}"));
        assert!(err.is_some(), "aceitou {entry}");
    }
    // Forma cujo número de bytes estoura usize: erro na leitura, não pânico.
    write_head(serde_json::json!({"dtype": "BF16", "shape": [u64::MAX, 4], "data_offsets": [0, 4]}));
    let ck = Checkpoint::open(tmp.0.parent().unwrap()).unwrap();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ck.bf16("a").is_err()));
    assert!(r.unwrap_or(false), "forma enorme deveria dar erro, não pânico nem sucesso");
}

/// Camada sintética no layout do Granite MoE e a saída esperada calculada em f64.
fn synthetic_layer(name: &str, e: usize, h: usize, i: usize, k: usize) -> (Scratch, HostLayer) {
    let tmp = scratch(name);
    let p = "model.layers.3.block_sparse_moe";
    // Escalas pequenas para as ativações ficarem na faixa de um modelo real.
    let scale = |v: Vec<f32>, s: f32| v.into_iter().map(|x| x * s).collect::<Vec<_>>();
    let router = bf(&values(e * h, 1));
    let gu = bf(&scale(values(e * 2 * i * h, 2), 0.1));
    let dn = bf(&scale(values(e * h * i, 3), 0.1));
    write_safetensors(
        &tmp.0,
        &[
            (&format!("{p}.router.layer.weight"), "BF16", vec![e, h], bytes(&router)),
            (&format!("{p}.input_linear.weight"), "BF16", vec![e, 2 * i, h], bytes(&gu)),
            (&format!("{p}.output_linear.weight"), "BF16", vec![e, h, i], bytes(&dn)),
        ],
    );
    let ck = Checkpoint::open(tmp.0.parent().unwrap()).unwrap();
    let layer = HostLayer::load(&ck, 3, k).unwrap();
    (tmp, layer)
}

fn naive(l: &HostLayer, x: &[f32]) -> (Vec<f64>, Vec<usize>) {
    let (h, i) = (l.hidden, l.inter);
    let f = |b: bf16| b.to_f32() as f64;
    let logits: Vec<f32> = (0..l.experts()).map(|e| (0..h).map(|c| f(l.router[e * h + c]) * x[c] as f64).sum::<f64>() as f32).collect();
    let r = Routing::from_logits(&logits, l.k);
    let mut y = vec![0f64; h];
    for (&e, &g) in r.experts.iter().zip(&r.weights) {
        let hh: Vec<f64> = (0..2 * i).map(|row| (0..h).map(|c| f(l.gate_up[e][row * h + c]) * x[c] as f64).sum()).collect();
        let a: Vec<f64> = (0..i).map(|j| hh[j] / (1.0 + (-hh[j]).exp()) * hh[i + j]).collect();
        for row in 0..h {
            let o: f64 = (0..i).map(|c| f(l.down[e][row * i + c]) * a[c]).sum();
            y[row] += g as f64 * o;
        }
    }
    (y, r.experts)
}

fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    let num: f64 = got.iter().zip(want).map(|(g, w)| (*g as f64 - w).powi(2)).sum();
    let den: f64 = want.iter().map(|w| w * w).sum();
    (num / den).sqrt()
}

#[test]
fn bloco_na_cpu_bate_com_implementacao_ingenua_em_f64() {
    let (_tmp, l) = synthetic_layer("cpu", 6, 24, 10, 2);
    assert_eq!((l.experts(), l.hidden, l.inter), (6, 24, 10));
    let cpu = cpu::Cpu::new(3);
    for t in 0..4 {
        let x = values(l.hidden, 100 + t);
        let (y, r) = l.forward_cpu(&x, &cpu);
        let (want, experts) = naive(&l, &x);
        assert_eq!(r.experts, experts);
        assert!(rel_err(&y, &want) < 1e-5, "token {t}: {}", rel_err(&y, &want));
    }
}

#[test]
fn camada_com_formas_trocadas_e_recusada() {
    let tmp = scratch("formas");
    let p = "model.layers.0.block_sparse_moe";
    let z = |n: usize| vec![0u8; n * 2];
    write_safetensors(
        &tmp.0,
        &[
            (&format!("{p}.router.layer.weight"), "BF16", vec![4, 8], z(32)),
            (&format!("{p}.input_linear.weight"), "BF16", vec![4, 8, 8], z(256)),
            (&format!("{p}.output_linear.weight"), "BF16", vec![4, 4, 8], z(128)),
        ],
    );
    let ck = Checkpoint::open(tmp.0.parent().unwrap()).unwrap();
    let err = HostLayer::load(&ck, 0, 2).err().expect("deveria recusar");
    assert!(err.to_string().contains("formas inesperadas"), "{err}");
}

#[test]
fn tile_ao_longo_de_k_e_a_maior_potencia_de_dois_que_divide_ate_1024() {
    use moe_rt::gpu::tile_k;
    for (k, want) in [(1024, 1024), (512, 512), (1536, 512), (768, 256), (2048, 1024), (3072, 1024), (96, 32)] {
        assert_eq!(tile_k(k), want, "k = {k}");
        assert_eq!(k % tile_k(k), 0);
    }
}

#[test]
fn bloco_na_gpu_bate_com_a_cpu_com_hidden_que_nao_e_potencia_de_dois() {
    let Some(device) = gpu_or_skip() else { return };
    // H = 768 = 3·256 tem a forma do Granite 3B (H = 1536 = 3·512): múltiplo dos blocos, mas não potência de dois, então o
    // GEMV não pode usar H inteiro como tile.
    let (_tmp, l) = synthetic_layer("gpu-768", 4, 768, 128, 2);
    let mut g = GpuLayer::upload(&l, device.new_stream().unwrap()).unwrap();
    for t in 0..4 {
        let x = values(l.hidden, 300 + t);
        let (yg, rg) = g.forward(&x).unwrap();
        let (want, experts) = naive(&l, &x);
        assert_eq!(rg.experts, experts, "token {t}");
        assert!(rel_err(&yg, &want) < 1e-5, "token {t}: {}", rel_err(&yg, &want));
    }
}

#[test]
fn bloco_na_gpu_bate_com_a_cpu() {
    let Some(device) = gpu_or_skip() else { return };
    let stream = device.new_stream().unwrap();
    // Menores formas que respeitam os tamanhos de bloco dos kernels (H múltiplo de 256, I de 128).
    let (_tmp, l) = synthetic_layer("gpu", 4, 256, 128, 2);
    let mut g = GpuLayer::upload(&l, stream).unwrap();
    for t in 0..6 {
        let x = values(l.hidden, 200 + t);
        let (yg, rg) = g.forward(&x).unwrap();
        let (want, experts) = naive(&l, &x);
        assert_eq!(rg.experts, experts, "token {t}");
        assert!(rel_err(&yg, &want) < 1e-5, "token {t}: {}", rel_err(&yg, &want));
    }
}

#[test]
fn down_dos_experts_nao_divide_o_nome_do_gemv() {
    // O cache de lançamento do cutile-rs guarda uma especialização por função de kernel: um GEMV com K = I (o down) no mesmo
    // nome dos GEMV com K = H compila e dá o mesmo resultado, mas refaz a resolução no host a cada expert (M2e). O down é o
    // `gemv_axpby`, num lugar só (`gpu::enqueue_expert`): a contagem 1 é de propósito, e um segundo lançamento legítimo (um
    // prefill, por exemplo) muda este teste junto. A medida de verdade é o host por lançamento no `m2c-kernels` (~1,25 µs).
    for f in ["gpu.rs", "cache.rs", "gpu_model.rs"] {
        let src = fs::read_to_string(format!("{}/src/{f}", env!("CARGO_MANIFEST_DIR"))).unwrap();
        let bad: Vec<&str> = src.lines().filter(|l| l.contains("kernels::gemv(") && l.contains("down")).collect();
        assert!(bad.is_empty(), "{f}: down lançado pelo gemv: {bad:?}");
        let n = src.matches("kernels::gemv_axpby(").count();
        assert_eq!(n, usize::from(f == "gpu.rs"), "{f}: lançamentos do gemv_axpby");
    }
}

#[test]
fn gemv_swiglu_e_gemv_axpby_batem_com_f64() {
    use cutile::prelude::*;
    use cutile::tensor::Reshape;
    use moe_rt::gpu::{gemv_generics, kernels, tile_k, GEMV_BM, SWIGLU_BM};
    use std::sync::Arc;
    let Some(device) = gpu_or_skip() else { return };
    let s = device.new_stream().unwrap();
    // (H, I): H = 768 e 1.536 (o 3B) têm três fatias ao longo de K no gate_up; I = 768 dá três fatias no down; I = 128 é o menor
    // I dos testes de bloco.
    for &(h, i) in &[(768usize, 128usize), (256, 128), (1024, 512), (1536, 512), (256, 768)] {
        let gu = bf(&values(2 * i * h, 50 + h as u64));
        let dn = bf(&values(h * i, 51 + i as u64));
        let x = values(h, 52);
        let y0 = values(h, 53);
        let up = |v: &[f32], shape: &[usize]| api::copy_host_vec_to_device(&Arc::new(v.to_vec())).sync_on(&s).unwrap().reshape(shape).unwrap();
        let gud = api::copy_host_vec_to_device(&Arc::new(gu.clone())).sync_on(&s).unwrap().reshape(&[2 * i, h]).unwrap();
        let dnd = api::copy_host_vec_to_device(&Arc::new(dn.clone())).sync_on(&s).unwrap().reshape(&[h, i]).unwrap();
        let xd = up(&x, &[h]);
        let dot = |w: &[bf16], r: usize, k: usize, v: &[f32]| (0..k).map(|c| w[r * k + c].to_f32() as f64 * v[c] as f64).sum::<f64>();
        let want_a: Vec<f64> = (0..i)
            .map(|r| {
                let (g, u) = (dot(&gu, r, h, &x), dot(&gu, i + r, h, &x));
                g / (1.0 + (-g).exp()) * u
            })
            .collect();
        // O de produção (`SWIGLU_BM`) e o BM = 2: com BM = 2 o up do bloco `pid` começa no bloco `pid + I/2`.
        let mut av = Vec::new();
        for bm in [SWIGLU_BM, GEMV_BM] {
            let mut a = api::zeros::<f32>(&[i]).sync_on(&s).unwrap();
            // SAFETY: `unchecked_accesses`: H e I múltiplos de BM e das fatias; tudo vive até o `sync_on`.
            unsafe {
                let gen = vec![bm.to_string(), tile_k(h).to_string(), h.to_string(), i.to_string()];
                kernels::gemv_swiglu((&mut a).partition([bm]), &gud, &xd).generics(gen).sync_on(&s).unwrap();
            }
            av = api::copy_device_to_host_vec(&Arc::new(a)).sync_on(&s).unwrap();
            assert!(rel_err(&av, &want_a) < 1e-5, "gemv_swiglu BM={bm} H={h} I={i}: {}", rel_err(&av, &want_a));
        }
        // β = 0 descarta o `y` anterior (finito: um NaN passaria, ver `gemv_axpby`); β = 1 acumula.
        for (g, beta) in [(0.375f32, 0.0f32), (0.625, 1.0)] {
            let ad = up(&av, &[i]);
            let mut y = up(&y0, &[h]);
            // SAFETY: como acima.
            unsafe {
                kernels::gemv_axpby((&mut y).partition([GEMV_BM]), &dnd, &ad, g, beta).generics(gemv_generics(i)).sync_on(&s).unwrap();
            }
            let yv = api::copy_device_to_host_vec(&Arc::new(y)).sync_on(&s).unwrap();
            let want: Vec<f64> = (0..h).map(|r| beta as f64 * y0[r] as f64 + g as f64 * dot(&dn, r, i, &av)).collect();
            assert!(rel_err(&yv, &want) < 1e-5, "gemv_axpby H={h} I={i} β={beta}: {}", rel_err(&yv, &want));
        }
    }
}
