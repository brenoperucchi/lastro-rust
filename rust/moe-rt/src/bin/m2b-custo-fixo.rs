//! Passo 1 do corte do custo fixo: de onde vêm os ~4,4 ms/token (1B) e ~7 ms/token (3B) com todos os experts residentes,
//! contra 1,76 e 3,45 ms do llama.cpp.
//!
//! Passa os tokens do `texto-a` pelo mesmo modelo em variantes do passo (`GpuModel::step_probe`), cada uma do começo:
//! - `normal`: o passo de sempre (2 sincronizações por camada para ler os logits do roteador, mais 3 por token); grava o
//!   roteamento de cada token;
//! - `sem_sync`: o roteamento gravado entra pronto, sem roteador nem leitura por camada; os logits têm de sair iguais aos do
//!   `normal` (confere que a variante calcula a mesma coisa);
//! - `so_atencao`, `so_moe`: metade da camada (com o roteamento gravado, sem sincronização por camada);
//! - `base`: nem atenção nem MoE (envio do embedding e do cos/sin, RMSNorm final, cabeça de saída e leitura dos logits).
//!
//! Por variante: mediana por token do tempo total, do tempo no host até o último lançamento e do tempo na GPU entre eventos, e
//! as operações e sincronizações por token, contadas no código do passo. Com `--repeat N`, os tokens do `texto-a` entram N
//! vezes seguidas (contexto longo) e sai também a mediana por faixa de posição.
//!
//!   m2b-custo-fixo --model <dir> --ref <dir do texto-a> [--rounds N] [--repeat N]
use cutile::prelude::Device;
use moe_rt::block::Routing;
use moe_rt::gpu::Count;
use moe_rt::gpu_model::{GpuModel, Probe, StepTimes, ATTN_BN};
use moe_rt::model::{Config, HostModel};
use moe_rt::npy::Npy;
use moe_rt::weights::Checkpoint;
use std::path::PathBuf;

/// Os logits do `sem_sync` têm de ser iguais aos do `normal` bit a bit (mesmos kernels, mesma ordem; só muda quem escolhe os
/// experts).
const MAX_REL: f64 = 0.0;
/// Faixas de posição da tabela de contexto longo, como no `m2a-cache`.
const POS_BUCKETS: [(usize, usize); 4] = [(0, 256), (256, 512), (512, 1024), (1024, usize::MAX)];

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    if v.is_empty() { f64::NAN } else { v[v.len() / 2] }
}

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    let r = (num / den).sqrt();
    if r.is_nan() { f64::INFINITY } else { r }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let (mut model, mut refdir, mut rounds, mut repeat) = (PathBuf::new(), PathBuf::new(), 3usize, 1usize);
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| panic!("{k} sem valor"));
        match k.as_str() {
            "--model" => model = PathBuf::from(v()),
            "--ref" => refdir = PathBuf::from(v()),
            "--rounds" => rounds = v().parse().expect("--rounds"),
            "--repeat" => repeat = v().parse().expect("--repeat"),
            _ => panic!("opção desconhecida: {k}"),
        }
    }
    let cfg = Config::from_json(&serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?)?;
    let host = HostModel::load(&Checkpoint::open(&model)?, cfg.clone())?;
    let once: Vec<usize> = Npy::read(&refdir.join("tokens.npy"))?.i32()?.into_iter().map(|t| t as usize).collect();
    let tokens: Vec<usize> = once.iter().copied().cycle().take(once.len() * repeat.max(1)).collect();
    let device = Device::new(0)?;
    let mut g = GpuModel::upload(&host, device.new_stream()?, (tokens.len() + 1).next_multiple_of(ATTN_BN))?;
    let name = model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    println!(
        "m2b-custo-fixo: {name}; {} camadas, k = {}; {} tokens ({} do texto-a × {repeat}); {rounds} rodadas por variante (mediana de todas)",
        cfg.layers,
        cfg.k,
        tokens.len(),
        once.len()
    );

    // Aquecimento (compilação JIT dos kernels) e o roteamento de referência de cada token.
    g.reset();
    let mut routes: Vec<Vec<Routing>> = Vec::with_capacity(tokens.len());
    let mut want: Vec<Vec<f32>> = Vec::with_capacity(tokens.len());
    for &t in &tokens {
        let (lg, r, _) = g.step_probe(t, Probe::default())?;
        routes.push(r);
        want.push(lg);
    }

    let variants: [(&str, bool, bool, bool); 5] = [
        ("normal", false, false, false),
        ("sem_sync", true, false, false),
        ("so_atencao", true, false, true),
        ("so_moe", true, true, false),
        ("base", true, true, true),
    ];
    let mut all_ok = true;
    let mut rows = Vec::new();
    for (vname, use_routes, skip_attn, skip_moe) in variants {
        let mut times: Vec<StepTimes> = Vec::with_capacity(rounds * tokens.len());
        let mut max_rel = 0f64;
        for _ in 0..rounds {
            g.reset();
            for (i, &t) in tokens.iter().enumerate() {
                let probe = Probe { routes: if use_routes { Some(&routes[i]) } else { None }, skip_attn, skip_moe };
                let (lg, _, tm) = g.step_probe(t, probe)?;
                if !skip_attn && !skip_moe {
                    max_rel = max_rel.max(rel_err(&lg, &want[i]));
                }
                times.push(tm);
            }
        }
        let check = if skip_attn || skip_moe {
            "–".to_string()
        } else {
            let ok = max_rel <= MAX_REL;
            all_ok &= ok;
            format!("{} ({max_rel:.1e})", if ok { "ok" } else { "FALHOU" })
        };
        let total = median(times.iter().map(|t| t.total_ms).collect());
        let enq = median(times.iter().map(|t| t.enqueue_ms).collect());
        let gpu = median(times.iter().map(|t| t.gpu_ms).collect());
        // O passo tem as mesmas operações em todo token.
        let count = times[0].count.expect("experts residentes contam as operações");
        assert!(times.iter().all(|t| t.count == Some(count)), "{vname}: contagem de operações variou entre tokens");
        // Mediana da GPU por faixa de posição (a posição do i-ésimo tempo de cada rodada é i).
        let by_pos: Vec<f64> = POS_BUCKETS
            .iter()
            .map(|&(a, b)| median(times.iter().enumerate().filter(|(i, _)| (a..b).contains(&(i % tokens.len()))).map(|(_, t)| t.gpu_ms).collect()))
            .collect();
        rows.push((vname, total, enq, gpu, count, check, by_pos));
    }
    println!(
        "\n{:<11} | {:>9} {:>9} {:>9} | {:>8} | {:>5} {:>5} | logits contra o normal",
        "variante", "total", "host", "GPU", "tok/s", "ops", "syncs"
    );
    for (vname, total, enq, gpu, Count { ops, syncs }, check, _) in &rows {
        println!("{vname:<11} | {total:>9.3} {enq:>9.3} {gpu:>9.3} | {:>8.1} | {ops:>5} {syncs:>5} | {check}", 1e3 / total);
    }
    println!("\ntotal: parede do passo; host: parede até o último lançamento voltar (antes da leitura dos logits);");
    println!("GPU: entre eventos no começo do passo e antes da leitura dos logits (inclui a GPU parada à espera do host). ms, medianas.");
    println!("ops: operações enfileiradas (kernels e cópias) por token; syncs: esperas do host pela GPU por token.");
    if tokens.len() > POS_BUCKETS[0].1 {
        println!("\nGPU por faixa de posição (ms, mediana):");
        let head: Vec<String> = POS_BUCKETS
            .iter()
            .map(|&(a, b)| if b == usize::MAX { format!("{a}+") } else { format!("{a}-{b}") })
            .collect();
        println!("{:<11} | {}", "variante", head.iter().map(|h| format!("{h:>9}")).collect::<Vec<_>>().join(" "));
        for (vname, .., by_pos) in &rows {
            println!("{vname:<11} | {}", by_pos.iter().map(|v| format!("{v:>9.3}")).collect::<Vec<_>>().join(" "));
        }
    }
    println!("resultado: {}", if all_ok { "sem_sync calcula o mesmo que o normal" } else { "FALHOU" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
