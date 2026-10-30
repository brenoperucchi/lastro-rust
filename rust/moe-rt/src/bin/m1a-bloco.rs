//! M1a: o bloco MoE de uma camada do Granite MoE na RTX 4090, conferido contra o transformers.
//!
//! Lê os pesos da camada no safetensors, passa token a token a entrada real do bloco (capturada por
//! `tools/ref_moe_block.py`) pelos dois caminhos (GPU com todos os experts na VRAM; CPU) e compara com a saída do
//! transformers: experts escolhidos, pesos do softmax e erro relativo da saída. Publica o andamento para o
//! `moe-monitor`.
//!
//!   m1a-bloco --model <dir> --ref <dir> [--layers 0,11,23] [--threads N] [--metrics <arquivo>]
use moe_monitor::snapshot::{self, Event, Publisher, Request, Snapshot, State};
use moe_rt::block::{HostLayer, Routing};
use moe_rt::gpu::{self, GpuLayer};
use moe_rt::weights::Checkpoint;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Erro relativo máximo aceito por token (norma 2) nos tokens em que o top-k bate.
const MAX_REL: f64 = 1e-4;
/// Fração máxima de tokens com top-k diferente (empates de logit podem trocar o k-ésimo expert).
const MAX_TOPK_MISS: f64 = 0.01;

struct Args {
    model: PathBuf,
    refdir: PathBuf,
    layers: Option<Vec<usize>>,
    threads: usize,
    metrics: PathBuf,
}

fn parse_args() -> Args {
    let mut a = std::env::args().skip(1);
    let (mut model, mut refdir, mut layers, mut threads, mut metrics) = (None, None, None, 8, snapshot::default_path());
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| panic!("{k} sem valor"));
        match k.as_str() {
            "--model" => model = Some(PathBuf::from(v())),
            "--ref" => refdir = Some(PathBuf::from(v())),
            "--layers" => layers = Some(v().split(',').map(|s| s.trim().parse().expect("camada")).collect()),
            "--threads" => threads = v().parse().expect("--threads"),
            "--metrics" => metrics = PathBuf::from(v()),
            _ => panic!("opção desconhecida: {k}"),
        }
    }
    Args { model: model.expect("--model"), refdir: refdir.expect("--ref"), layers, threads, metrics }
}

fn read_f32(p: &Path) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn read_i32(p: &Path) -> Vec<i32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Referência de uma camada: entrada, saída e decisão do roteador por token.
struct Reference {
    tokens: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    idx: Vec<i32>,
    w: Vec<f32>,
}

/// Comparação acumulada de um caminho contra a referência.
#[derive(Default)]
struct Score {
    tokens: usize,
    topk_miss: usize,
    max_rel: f64,
    sum_rel: f64,
    max_w_diff: f64,
    ms: Vec<f64>,
}

impl Score {
    fn add(&mut self, got: &[f32], r: &Routing, want: &[f32], idx: &[i32], w: &[f32], ms: f64) {
        self.tokens += 1;
        self.ms.push(ms);
        let same = r.experts.iter().zip(idx).all(|(&a, &b)| a as i32 == b);
        if !same {
            self.topk_miss += 1;
            return;
        }
        let (mut num, mut den) = (0f64, 0f64);
        for (g, t) in got.iter().zip(want) {
            num += ((g - t) as f64).powi(2);
            den += (*t as f64).powi(2);
        }
        let rel = (num / den.max(1e-30)).sqrt();
        // `f64::max` descartaria um NaN; ele tem de reprovar.
        self.max_rel = if rel.is_nan() { f64::INFINITY } else { self.max_rel.max(rel) };
        self.sum_rel += rel;
        for (a, b) in r.weights.iter().zip(w) {
            self.max_w_diff = self.max_w_diff.max((a - b).abs() as f64);
        }
    }

    fn median_ms(&self) -> f64 {
        let mut v = self.ms.clone();
        v.sort_by(f64::total_cmp);
        if v.is_empty() { 0.0 } else { v[v.len() / 2] }
    }

    fn ok(&self) -> bool {
        self.max_rel <= MAX_REL && (self.topk_miss as f64) <= MAX_TOPK_MISS * self.tokens as f64
    }

    fn line(&self, name: &str) -> String {
        let matched = (self.tokens - self.topk_miss).max(1);
        format!(
            "{name:<4} top-k igual {}/{} | erro rel. máx {:.2e} médio {:.2e} | Δpeso máx {:.1e} | {:.3} ms/token (mediana) | {}",
            self.tokens - self.topk_miss,
            self.tokens,
            self.max_rel,
            self.sum_rel / matched as f64,
            self.max_w_diff,
            self.median_ms(),
            if self.ok() { "OK" } else { "FALHOU" }
        )
    }
}

/// Estado publicado para o monitor.
struct Monitor {
    publisher: Option<Publisher>,
    snap: Snapshot,
    last: Instant,
}

impl Monitor {
    fn new(path: &Path, args: &Args) -> Self {
        let publisher = match Publisher::new(path) {
            Ok(p) => Some(p),
            Err(e) => {
                eprintln!("aviso: métricas desligadas ({}: {e})", path.display());
                None
            }
        };
        let mut snap = Snapshot { schema: snapshot::SCHEMA, pid: std::process::id(), started_ms: snapshot::now_ms(), ..Default::default() };
        snap.settings.insert("marco".into(), "M1a: bloco MoE de uma camada".into());
        snap.settings.insert("pesos".into(), args.model.display().to_string());
        snap.settings.insert("referência".into(), args.refdir.display().to_string());
        snap.settings.insert("threads da CPU".into(), args.threads.to_string());
        Self { publisher, snap, last: Instant::now() }
    }

    fn event(&mut self, level: &str, message: String) {
        self.snap.events.push(Event { ts_ms: snapshot::now_ms(), level: level.into(), message });
        let n = self.snap.events.len();
        if n > 64 {
            self.snap.events.drain(..n - 64);
        }
        self.publish(true);
    }

    fn publish(&mut self, force: bool) {
        if !force && self.last.elapsed().as_millis() < 200 {
            return;
        }
        self.last = Instant::now();
        self.snap.ts_ms = snapshot::now_ms();
        if let Some(p) = &self.publisher {
            if let Err(e) = p.publish(&self.snap) {
                eprintln!("aviso: falha ao publicar métricas: {e}");
                self.publisher = None;
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(args.refdir.join("meta.json"))?)?;
    let tokens = meta["tokens"].as_u64().ok_or("meta.json sem tokens")? as usize;
    let hidden = meta["hidden"].as_u64().ok_or("meta.json sem hidden")? as usize;
    let k = meta["k"].as_u64().ok_or("meta.json sem k")? as usize;
    let layers: Vec<usize> = args.layers.clone().unwrap_or_else(|| {
        meta["layers"].as_array().expect("meta.json sem layers").iter().map(|v| v.as_u64().unwrap() as usize).collect()
    });
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(args.model.join("config.json"))?)?;
    let model_name = args.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();

    let mut mon = Monitor::new(&args.metrics, &args);
    mon.snap.model.name = model_name.clone();
    mon.snap.model.layers = config["num_hidden_layers"].as_u64().unwrap_or(0) as u32;
    mon.snap.model.experts = config["num_local_experts"].as_u64().unwrap_or(0) as u32;
    mon.snap.model.experts_per_tok = k as u32;
    mon.snap.context.max = tokens as u64;
    println!("M1a: {model_name}, {tokens} tokens, camadas {layers:?}, métricas em {}", args.metrics.display());

    let ck = Checkpoint::open(&args.model)?;
    let device = cutile::prelude::Device::new(0)?;
    let stream = device.new_stream()?;
    let cpu = moe_rt::cpu::Cpu::new(args.threads);
    let mut all_ok = true;

    for &layer in &layers {
        let t0 = Instant::now();
        let host = HostLayer::load(&ck, layer, k)?;
        assert_eq!(host.hidden, hidden, "hidden do safetensors difere do meta.json");
        let load_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t0 = Instant::now();
        let mut gpu_layer = GpuLayer::upload(&host, stream.clone())?;
        let upload_ms = t0.elapsed().as_secs_f64() * 1e3;
        let mb = gpu::layer_weight_bytes(&host) as f64 / 1e6;
        println!(
            "\n== camada {layer}: {} experts de {:.2} MB; leitura {load_ms:.0} ms, envio à VRAM {mb:.0} MB em {upload_ms:.0} ms ==",
            host.experts(),
            host.expert_bytes() as f64 / 1e6
        );
        mon.snap.settings.insert("camada".into(), layer.to_string());
        mon.snap.settings.insert("tamanho do expert (MB)".into(), format!("{:.2}", host.expert_bytes() as f64 / 1e6));
        mon.event("info", format!("camada {layer}: {mb:.0} MB na VRAM em {upload_ms:.0} ms"));

        let p = |s: &str| args.refdir.join(format!("layer-{layer}.{s}"));
        let rf = Reference { tokens, x: read_f32(&p("x.f32")), y: read_f32(&p("y.f32")), idx: read_i32(&p("idx.i32")), w: read_f32(&p("w.f32")) };
        assert_eq!(rf.x.len(), tokens * hidden);
        assert_eq!(rf.idx.len(), tokens * k);

        // Aquecimento: a primeira chamada compila os kernels (JIT) e não entra no tempo.
        let t0 = Instant::now();
        gpu_layer.forward(&rf.x[..hidden])?;
        println!("aquecimento da GPU (compilação dos kernels): {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);

        for path in ["gpu", "cpu"] {
            let mut score = Score::default();
            let started = snapshot::now_ms();
            let e = host.experts() as u32;
            mon.snap.cache = Default::default();
            mon.snap.cache.slots_total = e;
            mon.snap.cache.slots_resident = if path == "gpu" { e } else { 0 };
            mon.snap.settings.insert("caminho dos experts".into(), path.into());
            for t in 0..rf.tokens {
                let x = &rf.x[t * hidden..(t + 1) * hidden];
                let t0 = Instant::now();
                let (y, r) = if path == "gpu" { gpu_layer.forward(x)? } else { host.forward_cpu(x, &cpu) };
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                score.add(&y, &r, &rf.y[t * hidden..(t + 1) * hidden], &rf.idx[t * k..(t + 1) * k], &rf.w[t * k..(t + 1) * k], ms);

                let c = &mut mon.snap.cache;
                if path == "gpu" {
                    c.total_hits += k as u64;
                    c.hit_frac = 1.0;
                } else {
                    c.total_cpu += k as u64;
                    c.cpu_frac = 1.0;
                }
                // Um token por vez (batch 1), como no decode.
                mon.snap.state = State::Generating { tokens: t as u64 + 1, max_tokens: rf.tokens as u64 };
                mon.snap.context.used = t as u64 + 1;
                mon.snap.speed.decode_now = 1e3 / ms;
                mon.snap.window.ms_per_token = ms;
                mon.publish(false);
            }
            let line = score.line(path);
            println!("{line}");
            all_ok &= score.ok();
            let tok_s = 1e3 / score.median_ms().max(1e-9);
            mon.snap.speed.decode_last = tok_s;
            mon.snap.requests.insert(0, Request {
                started_ms: started,
                prompt_tokens: 0,
                reused_tokens: 0,
                output_tokens: rf.tokens as u64,
                duration_ms: snapshot::now_ms() - started,
                prefill_tok_s: 0.0,
                decode_tok_s: tok_s,
                status: format!("{} c{layer} {}", if score.ok() { "ok" } else { "falhou" }, path),
            });
            mon.event(if score.ok() { "info" } else { "erro" }, format!("camada {layer} {line}"));
        }
    }

    mon.snap.state = if all_ok { State::Idle } else { State::Error { message: "M1a: divergência da referência".into() } };
    mon.publish(true);
    println!("\nresultado: {}", if all_ok { "todas as camadas dentro da tolerância" } else { "FALHOU" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
