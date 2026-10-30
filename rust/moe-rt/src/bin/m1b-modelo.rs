//! M1b: o Granite MoE inteiro em Rust, conferido token a token contra os logits do transformers (teacher forcing).
//!
//! Passa os tokens de `tools/ref_logits.py` um por vez (decode com cache KV) pelo caminho pedido e compara, em cada
//! posição, os logits com os do transformers e os experts escolhidos em cada camada com os dele. Publica o andamento no
//! `moe-monitor`.
//!
//!   m1b-modelo --model <dir> --ref <dir> [--paths gpu,cpu] [--tokens N] [--threads N] [--metrics <arquivo>]
use moe_monitor::snapshot::{self, Event, Publisher, Request, Snapshot, State};
use moe_rt::block::Routing;
use moe_rt::gpu_model::GpuModel;
use moe_rt::model::{Config, CpuSession, HostModel};
use moe_rt::npy::Npy;
use moe_rt::weights::Checkpoint;
use cutile::prelude::Device;
use std::path::PathBuf;
use std::time::Instant;

/// Erro relativo máximo aceito nos logits de uma posição (norma 2).
const MAX_REL: f64 = 1e-4;
/// Um argmax diferente só é aceito se, na referência, os dois primeiros logits estiverem mais perto que isto.
const TIE_GAP: f32 = 1e-2;

struct Args {
    model: PathBuf,
    refdir: PathBuf,
    paths: Vec<String>,
    tokens: Option<usize>,
    threads: usize,
    metrics: PathBuf,
}

fn parse_args() -> Args {
    let mut a = std::env::args().skip(1);
    let mut r = Args { model: PathBuf::new(), refdir: PathBuf::new(), paths: vec!["gpu".into(), "cpu".into()], tokens: None, threads: 8, metrics: snapshot::default_path() };
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| panic!("{k} sem valor"));
        match k.as_str() {
            "--model" => r.model = PathBuf::from(v()),
            "--ref" => r.refdir = PathBuf::from(v()),
            "--paths" => r.paths = v().split(',').map(|s| s.trim().to_string()).collect(),
            "--tokens" => r.tokens = Some(v().parse().expect("--tokens")),
            "--threads" => r.threads = v().parse().expect("--threads"),
            "--metrics" => r.metrics = PathBuf::from(v()),
            _ => panic!("opção desconhecida: {k}"),
        }
    }
    assert!(!r.model.as_os_str().is_empty() && !r.refdir.as_os_str().is_empty(), "uso: --model <dir> --ref <dir>");
    for p in &r.paths {
        assert!(p == "gpu" || p == "cpu", "caminho desconhecido: {p}");
    }
    r
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

/// Diferença entre o maior e o segundo maior valor.
fn top_gap(v: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in v {
        if x > a {
            b = a;
            a = x;
        } else if x > b {
            b = x;
        }
    }
    a - b
}

#[derive(Default)]
struct Score {
    tokens: usize,
    argmax_same: usize,
    argmax_tie: usize,
    argmax_bad: usize,
    max_rel: f64,
    sum_rel: f64,
    max_abs: f64,
    route_total: usize,
    route_same: usize,
    ms: Vec<f64>,
}

impl Score {
    fn add(&mut self, got: &[f32], want: &[f32], routes: &[Routing], ref_experts: &[Vec<i32>], ms: f64) {
        self.tokens += 1;
        self.ms.push(ms);
        let (mut num, mut den, mut mx) = (0f64, 0f64, 0f64);
        for (g, w) in got.iter().zip(want) {
            let e = (*g - *w) as f64;
            num += e * e;
            den += (*w as f64).powi(2);
            mx = mx.max(e.abs());
        }
        let rel = (num / den.max(1e-30)).sqrt();
        // `f64::max` descartaria um NaN; ele tem de reprovar.
        self.max_rel = if rel.is_nan() { f64::INFINITY } else { self.max_rel.max(rel) };
        self.sum_rel += rel;
        self.max_abs = self.max_abs.max(mx);
        if argmax(got) == argmax(want) {
            self.argmax_same += 1;
        } else if top_gap(want) < TIE_GAP {
            self.argmax_tie += 1;
        } else {
            self.argmax_bad += 1;
        }
        for (r, want) in routes.iter().zip(ref_experts) {
            self.route_total += 1;
            if r.experts.iter().zip(want).all(|(&a, &b)| a as i32 == b) {
                self.route_same += 1;
            }
        }
    }

    fn median_ms(&self) -> f64 {
        let mut v = self.ms.clone();
        v.sort_by(f64::total_cmp);
        if v.is_empty() { 0.0 } else { v[v.len() / 2] }
    }

    fn ok(&self) -> bool {
        self.max_rel <= MAX_REL && self.argmax_bad == 0
    }

    fn line(&self, name: &str) -> String {
        format!(
            "{name:<4} argmax igual {}/{} (empates {}, divergências {}) | erro rel. logits máx {:.2e} médio {:.2e} | |Δ| máx {:.2e} | roteamento igual {}/{} | {:.2} ms/token (mediana) | {}",
            self.argmax_same,
            self.tokens,
            self.argmax_tie,
            self.argmax_bad,
            self.max_rel,
            self.sum_rel / self.tokens.max(1) as f64,
            self.max_abs,
            self.route_same,
            self.route_total,
            self.median_ms(),
            if self.ok() { "OK" } else { "FALHOU" }
        )
    }
}

struct Monitor {
    publisher: Option<Publisher>,
    snap: Snapshot,
    last: Instant,
}

impl Monitor {
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

    fn event(&mut self, level: &str, message: String) {
        self.snap.events.push(Event { ts_ms: snapshot::now_ms(), level: level.into(), message });
        let n = self.snap.events.len();
        if n > 64 {
            self.snap.events.drain(..n - 64);
        }
        self.publish(true);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let p = |f: &str| args.refdir.join(f);
    let tokens = Npy::read(&p("tokens.npy"))?.i32()?;
    let logits_npy = Npy::read(&p("logits.npy"))?;
    let ref_logits = logits_npy.f32()?;
    let experts_npy = Npy::read(&p("experts.npy"))?;
    let (n_layers, k) = (experts_npy.shape[0], experts_npy.shape[2]);
    let ref_experts = experts_npy.i32()?;
    let total = args.tokens.unwrap_or(tokens.len()).min(tokens.len());
    let vocab = logits_npy.shape[1];

    let cfg = Config::from_json(&serde_json::from_slice(&std::fs::read(args.model.join("config.json"))?)?)?;
    assert_eq!((cfg.vocab, cfg.layers, cfg.k), (vocab, n_layers, k), "referência e config.json não batem");
    let name = args.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    println!("M1b: {name}, {total} tokens, caminhos {:?}, métricas em {}", args.paths, args.metrics.display());

    let t0 = Instant::now();
    let ck = Checkpoint::open(&args.model)?;
    let host = HostModel::load(&ck, cfg.clone())?;
    println!("pesos lidos em {:.1} s", t0.elapsed().as_secs_f64());

    let publisher = Publisher::new(&args.metrics).map_err(|e| eprintln!("aviso: métricas desligadas: {e}")).ok();
    let mut snap = Snapshot { schema: snapshot::SCHEMA, pid: std::process::id(), started_ms: snapshot::now_ms(), ..Default::default() };
    snap.model.name = name.clone();
    snap.model.layers = cfg.layers as u32;
    snap.model.experts = cfg.experts as u32;
    snap.model.experts_per_tok = cfg.k as u32;
    snap.context.max = total as u64;
    snap.settings.insert("marco".into(), "M1b: modelo inteiro, teacher forcing".into());
    snap.settings.insert("pesos".into(), args.model.display().to_string());
    snap.settings.insert("referência".into(), args.refdir.display().to_string());
    snap.settings.insert("threads da CPU".into(), args.threads.to_string());
    let expert_mb = host.layers[0].moe.expert_bytes() as f64 / 1e6;
    snap.settings.insert("tamanho do expert (MB)".into(), format!("{expert_mb:.2}"));
    let mut mon = Monitor { publisher, snap, last: Instant::now() };

    let mut gpu = None;
    if args.paths.iter().any(|p| p == "gpu") {
        let device = Device::new(0)?;
        let stream = device.new_stream()?;
        let t0 = Instant::now();
        let mut g = GpuModel::upload(&host, stream, (total + 1).next_multiple_of(moe_rt::gpu_model::ATTN_BN))?;
        let mb = GpuModel::weight_bytes(&host) as f64 / 1e6;
        println!("pesos na VRAM: {mb:.0} MB em {:.1} s", t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        g.step(tokens[0] as usize)?;
        g.reset();
        println!("aquecimento da GPU (compilação dos kernels): {:.1} s", t0.elapsed().as_secs_f64());
        mon.event("info", format!("modelo na VRAM: {mb:.0} MB"));
        gpu = Some((device, g));
    }

    let cpu_engine = moe_rt::cpu::Cpu::new(args.threads);
    let mut all_ok = true;
    for path in &args.paths {
        let mut score = Score::default();
        let started = snapshot::now_ms();
        let slots = (cfg.layers * cfg.experts) as u32;
        mon.snap.cache = Default::default();
        mon.snap.cache.slots_total = slots;
        mon.snap.cache.slots_resident = if path == "gpu" { slots } else { 0 };
        mon.snap.settings.insert("caminho".into(), path.clone());
        let mut cpu = CpuSession::new(&host, &cpu_engine);
        for t in 0..total {
            let tok = tokens[t] as usize;
            let t1 = Instant::now();
            let (lg, routes) = if path == "gpu" { gpu.as_mut().unwrap().1.step(tok)? } else { cpu.step(tok) };
            let ms = t1.elapsed().as_secs_f64() * 1e3;
            let want = &ref_logits[t * vocab..(t + 1) * vocab];
            let re: Vec<Vec<i32>> = (0..n_layers).map(|l| ref_experts[(l * tokens.len() + t) * k..(l * tokens.len() + t + 1) * k].to_vec()).collect();
            score.add(&lg, want, &routes, &re, ms);

            let c = &mut mon.snap.cache;
            if path == "gpu" {
                c.total_hits += (cfg.layers * cfg.k) as u64;
                c.hit_frac = 1.0;
            } else {
                c.total_cpu += (cfg.layers * cfg.k) as u64;
                c.cpu_frac = 1.0;
            }
            mon.snap.state = State::Generating { tokens: t as u64 + 1, max_tokens: total as u64 };
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
            output_tokens: total as u64,
            duration_ms: snapshot::now_ms() - started,
            decode_tok_s: tok_s,
            status: format!("{} {path}", if score.ok() { "ok" } else { "falhou" }),
            ..Default::default()
        });
        mon.event(if score.ok() { "info" } else { "erro" }, line);
        if path == "gpu" {
            gpu.as_mut().unwrap().1.reset();
        }
    }
    mon.snap.state = if all_ok { State::Idle } else { State::Error { message: "M1b: divergência da referência".into() } };
    mon.publish(true);
    println!("\nresultado: {}", if all_ok { "dentro da tolerância" } else { "FALHOU" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Score;

    #[test]
    fn nan_fora_do_argmax_reprova() {
        let want = [1.0f32, 5.0, 2.0];
        let mut s = Score::default();
        s.add(&want, &want, &[], &[], 1.0);
        s.add(&[1.0, 5.0, -f32::NAN], &want, &[], &[], 1.0);
        assert!(!s.ok());
    }
}
