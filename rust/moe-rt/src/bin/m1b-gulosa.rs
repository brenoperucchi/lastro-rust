//! M1b: geração gulosa livre do Granite MoE em Rust, comparada com a do transformers (`tools/ref_greedy.py`).
//!
//! Para cada caso: o prompt entra token a token (o mesmo passo do decode), depois o modelo gera N tokens escolhendo sempre o
//! maior logit e realimentando a própria escolha. A sequência gerada tem de ser igual à do transformers; uma divergência só é
//! aceita num passo em que, na referência, os dois primeiros logits estavam mais perto que `TIE_GAP`.
//!
//! O caminho `cache` é o da GPU com os experts no cache do M2a (`--cache-policy lru_sync|lru_cpu`, `--cache-budget` em fração
//! dos experts), que persiste entre os casos.
//!
//!   m1b-gulosa --model <dir> --ref <greedy.json> [--paths gpu,cpu,cache] [--threads N] [--cache-policy lru_cpu]
//!              [--cache-budget 0.1] [--backlog N]
use cutile::prelude::Device;
use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
use moe_rt::gpu_model::{GpuModel, ATTN_BN};
use moe_rt::model::{Config, CpuSession, HostModel};
use moe_rt::weights::Checkpoint;
use std::path::PathBuf;
use std::time::Instant;

const TIE_GAP: f64 = 1e-3;

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

fn ids(v: &serde_json::Value) -> Vec<usize> {
    v.as_array().expect("lista de ids").iter().map(|x| x.as_u64().expect("id") as usize).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut a = std::env::args().skip(1);
    let (mut model, mut refp, mut paths, mut threads) = (PathBuf::new(), PathBuf::new(), vec!["gpu".to_string(), "cpu".to_string()], 8usize);
    let (mut policy, mut budget, mut backlog) = (Policy::LruCpu, 0.1f64, 4usize);
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| panic!("{k} sem valor"));
        match k.as_str() {
            "--model" => model = PathBuf::from(v()),
            "--ref" => refp = PathBuf::from(v()),
            "--paths" => paths = v().split(',').map(|s| s.trim().to_string()).collect(),
            "--threads" => threads = v().parse().expect("--threads"),
            "--cache-policy" => {
                policy = match v().as_str() {
                    "lru_sync" => Policy::LruSync,
                    "lru_cpu" => Policy::LruCpu,
                    p => panic!("política desconhecida: {p}"),
                }
            }
            "--cache-budget" => budget = v().parse().expect("--cache-budget"),
            "--backlog" => backlog = v().parse().expect("--backlog"),
            _ => panic!("opção desconhecida: {k}"),
        }
    }
    let refj: serde_json::Value = serde_json::from_slice(&std::fs::read(&refp)?)?;
    let cases = refj["cases"].as_array().ok_or("greedy.json sem cases")?;
    let cfg = Config::from_json(&serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?)?;
    let host = HostModel::load(&Checkpoint::open(&model)?, cfg.clone())?;
    let longest = cases.iter().map(|c| ids(&c["prompt_ids"]).len() + ids(&c["generated_ids"]).len()).max().unwrap_or(0);
    let max_len = (longest + 1).next_multiple_of(ATTN_BN);

    // Modelos na GPU por caminho ("gpu" com experts residentes, "cache" com o cache do M2a).
    let mut gpus: Vec<(String, GpuModel)> = Vec::new();
    let device = if paths.iter().any(|p| p != "cpu") { Some(Device::new(0)?) } else { None };
    for p in &paths {
        let g = match p.as_str() {
            "cpu" => continue,
            "gpu" => GpuModel::upload(&host, device.as_ref().unwrap().new_stream()?, max_len)?,
            "cache" => {
                let slots = (budget * (cfg.layers * cfg.experts) as f64) as usize;
                println!("cache: {policy:?}, {slots} slots ({:.0}% dos experts), backlog {backlog}", budget * 100.0);
                let cc = CacheConfig { slots, policy, backlog, cpu_threads: threads, admit: AdmitMode::Event };
                GpuModel::upload_cached(&host, device.as_ref().unwrap().new_stream()?, max_len, cc)?
            }
            _ => panic!("caminho desconhecido: {p}"),
        };
        gpus.push((p.clone(), g));
    }

    let cpu_engine = moe_rt::cpu::Cpu::new(threads);
    let mut all_ok = true;
    for path in &paths {
        for (ci, case) in cases.iter().enumerate() {
            let prompt = ids(&case["prompt_ids"]);
            let want = ids(&case["generated_ids"]);
            let gaps: Vec<f64> = case["top_gap"].as_array().ok_or("sem top_gap")?.iter().map(|g| g.as_f64().unwrap_or(f64::INFINITY)).collect();
            let mut cpu = CpuSession::new(&host, &cpu_engine);
            let mut gpu = gpus.iter_mut().find(|(p, _)| p == path).map(|(_, g)| g);
            if let Some(g) = gpu.as_mut() {
                g.reset();
            }
            let mut step = |t: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
                Ok(match gpu.as_mut() {
                    Some(g) => g.step(t)?.0,
                    None => cpu.step(t).0,
                })
            };
            let t0 = Instant::now();
            let mut logits = Vec::new();
            for &t in &prompt {
                logits = step(t)?;
            }
            let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
            let t0 = Instant::now();
            let mut got = Vec::with_capacity(want.len());
            for _ in 0..want.len() {
                let next = argmax(&logits);
                got.push(next);
                logits = step(next)?;
            }
            let gen_ms = t0.elapsed().as_secs_f64() * 1e3;
            let first_diff = got.iter().zip(&want).position(|(a, b)| a != b);
            let (status, ok) = match first_diff {
                None => (format!("{}/{} iguais", got.len(), want.len()), true),
                Some(i) if gaps[i] < TIE_GAP => (format!("igual até o passo {i}, onde a referência tinha empate (gap {:.1e})", gaps[i]), true),
                Some(i) => (format!("DIVERGE no passo {i} (gap da referência {:.3}): {} vs {}", gaps[i], got[i], want[i]), false),
            };
            all_ok &= ok;
            println!(
                "{path:<5} caso {ci}: prompt {} tokens em {prefill_ms:.0} ms; {} gerados em {gen_ms:.0} ms ({:.1} tok/s) | {status}",
                prompt.len(),
                got.len(),
                got.len() as f64 / (gen_ms / 1e3)
            );
        }
    }
    println!("\nresultado: {}", if all_ok { "geração igual à do transformers" } else { "FALHOU" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
