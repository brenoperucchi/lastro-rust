//! M2a: o Granite MoE inteiro com os experts num cache de VRAM com orçamento, medido nas mesmas conversas do simulador.
//!
//! Para cada política e orçamento:
//! 1. confere os logits do `texto-a` contra o transformers (o cache não pode mudar a conta);
//! 2. esvazia o cache e passa as conversas pedidas token a token (decode em batch 1, o cache persiste entre conversas como no
//!    simulador), conferindo o roteamento contra o traço, os logits de cada token contra os da passada com experts residentes
//!    (que roda sempre primeiro), e medindo o tempo por token e os contadores do cache;
//! 3. imprime ao lado o que o simulador previu para o mesmo orçamento (`sim-1b.json`).
//!
//! `cache_full` é o caminho do cache com slots para todos os experts, preenchido antes das conversas: só acertos, então a
//! diferença para `resident` é o custo fixo do caminho do cache (leitura do roteador no host, lançamentos por expert).
//!
//! No fim, as contagens exatas do cache e um hash dos logits de cada configuração (FNV-1a sobre os bits, texto-a e conversas),
//! para comparar dois binários bit a bit. Com `--admit barreira` a admissão do `lru_cpu` não depende do tempo
//! ([`AdmitMode::Barrier`]), e as contagens e o hash se repetem entre execuções.
//!
//!   m2a-cache --model <dir> --traces <dir> --ref <dir> --sim <sim.json> [--convs 4,5,6,7] [--budgets 0.1,0.25,0.5,0.75]
//!             [--policies resident,cache_full,lru_sync,lru_cpu] [--threads N] [--backlog N] [--admit evento|barreira]
//!             [--metrics <arquivo>]
use cutile::prelude::Device;
use moe_monitor::snapshot::{self, Event, Publisher, Request, Snapshot, State};
use moe_rt::cache::{AdmitMode, CacheConfig, Policy};
use moe_rt::gpu_model::{Experts, GpuModel, ATTN_BN};
use moe_rt::model::{Config, HostModel};
use moe_rt::npy::Npy;
use moe_rt::weights::Checkpoint;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Erro relativo máximo (norma L2 do vetor de logits) contra a referência; o maior medido no M1b foi 9e-6.
const MAX_REL: f64 = 1e-4;

struct Args {
    model: PathBuf,
    traces: PathBuf,
    refdir: PathBuf,
    sim: Option<PathBuf>,
    convs: Vec<usize>,
    budgets: Vec<f64>,
    policies: Vec<String>,
    threads: usize,
    backlog: usize,
    admit: AdmitMode,
    metrics: PathBuf,
}

fn parse_args() -> Args {
    let mut a = std::env::args().skip(1);
    let mut r = Args {
        model: PathBuf::new(),
        traces: PathBuf::new(),
        refdir: PathBuf::new(),
        sim: None,
        convs: vec![4, 5, 6, 7],
        budgets: vec![0.1, 0.25, 0.5, 0.75],
        policies: vec!["resident".into(), "cache_full".into(), "lru_sync".into(), "lru_cpu".into()],
        threads: 8,
        backlog: 4,
        admit: AdmitMode::Event,
        metrics: snapshot::default_path(),
    };
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| panic!("{k} sem valor"));
        match k.as_str() {
            "--model" => r.model = PathBuf::from(v()),
            "--traces" => r.traces = PathBuf::from(v()),
            "--ref" => r.refdir = PathBuf::from(v()),
            "--sim" => r.sim = Some(PathBuf::from(v())),
            "--convs" => r.convs = v().split(',').map(|s| s.trim().parse().expect("--convs")).collect(),
            "--budgets" => r.budgets = v().split(',').map(|s| s.trim().parse().expect("--budgets")).collect(),
            "--policies" => r.policies = v().split(',').map(|s| s.trim().to_string()).collect(),
            "--threads" => r.threads = v().parse().expect("--threads"),
            "--backlog" => r.backlog = v().parse().expect("--backlog"),
            "--admit" => {
                r.admit = match v().as_str() {
                    "evento" => AdmitMode::Event,
                    "barreira" => AdmitMode::Barrier,
                    o => panic!("--admit: evento ou barreira, não {o}"),
                }
            }
            "--metrics" => r.metrics = PathBuf::from(v()),
            _ => panic!("opção desconhecida: {k}"),
        }
    }
    r
}

struct Conv {
    tokens: Vec<i32>,
    experts: Vec<i32>,
}

fn load_conv(dir: &Path) -> Result<Conv, String> {
    Ok(Conv { tokens: Npy::read(&dir.join("tokens.npy"))?.i32()?, experts: Npy::read(&dir.join("experts.npy"))?.i32()? })
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

fn rel_err(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (a, b) in got.iter().zip(want) {
        num += ((a - b) as f64).powi(2);
        den += (*b as f64).powi(2);
    }
    (num / den).sqrt()
}

/// Maior erro relativo e quantos argmax diferem.
#[derive(Default, Clone, Copy)]
struct Check {
    max_rel: f64,
    argmax_bad: usize,
}

impl Check {
    fn add(&mut self, got: &[f32], want: &[f32]) {
        let rel = rel_err(got, want);
        // `f64::max` descartaria um NaN, e um -NaN fora do argmax não mudaria o argmax: tem de reprovar.
        self.max_rel = if rel.is_nan() { f64::INFINITY } else { self.max_rel.max(rel) };
        self.argmax_bad += (argmax(got) != argmax(want)) as usize;
    }

    fn ok(&self) -> bool {
        self.max_rel <= MAX_REL && self.argmax_bad == 0
    }

    fn show(&self) -> String {
        if self.ok() { format!("{:.1e}", self.max_rel) } else { format!("FALHOU {:.1e} {}am", self.max_rel, self.argmax_bad) }
    }
}

/// Faixas de posição no contexto para a mediana por posição (a atenção de decode cresce com o contexto).
const POS_BUCKETS: [(usize, usize); 4] = [(0, 256), (256, 512), (512, 1024), (1024, usize::MAX)];

/// FNV-1a de 64 bits, acumulado sobre os bits de cada logit.
fn fnv(h: &mut u64, v: &[f32]) {
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            *h = (*h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    if s.is_empty() { f64::NAN } else { s[s.len() / 2] }
}

/// Previsão do simulador para (orçamento, política): ms por token e frações.
fn sim_row(sim: &serde_json::Value, frac: f64, policy: &str) -> Option<(f64, f64, f64)> {
    let r = sim["results"].as_array()?.iter().find(|r| (r["cap_frac"].as_f64().unwrap_or(-1.0) - frac).abs() < 1e-9 && r["policy"] == policy)?;
    let (t, n) = (r["tokens"].as_f64()?, r["needed"].as_f64()?);
    Some((r["ms"].as_f64()? / t, r["hits"].as_f64()? / n, r["cpu"].as_f64()? / n))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cfg = Config::from_json(&serde_json::from_slice(&std::fs::read(args.model.join("config.json"))?)?)?;
    let host = HostModel::load(&Checkpoint::open(&args.model)?, cfg.clone())?;
    let convs: Vec<Conv> = args.convs.iter().map(|c| load_conv(&args.traces.join(format!("conversa-{c}")))).collect::<Result<_, _>>()?;
    let ref_tokens = Npy::read(&args.refdir.join("tokens.npy"))?.i32()?;
    let ref_logits = Npy::read(&args.refdir.join("logits.npy"))?.f32()?;
    let sim: Option<serde_json::Value> = match &args.sim {
        Some(p) => Some(serde_json::from_slice(&std::fs::read(p)?)?),
        None => None,
    };
    let total_tokens: usize = convs.iter().map(|c| c.tokens.len()).sum();
    let longest = convs.iter().map(|c| c.tokens.len()).max().unwrap_or(0).max(ref_tokens.len());
    let max_len = (longest + 1).next_multiple_of(ATTN_BN);
    let slots_all = cfg.layers * cfg.experts;
    let expert_mb = host.layers[0].moe.expert_bytes() as f64 / 1e6;
    let name = args.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    println!(
        "M2a: {name}; conversas {:?} ({total_tokens} tokens); {slots_all} experts de {expert_mb:.2} MB; CPU com {} threads; backlog {}; admissão {}",
        args.convs,
        args.threads,
        args.backlog,
        if args.admit == AdmitMode::Barrier { "barreira" } else { "evento" }
    );

    let publisher = Publisher::new(&args.metrics).map_err(|e| eprintln!("aviso: métricas desligadas: {e}")).ok();
    let mut snap = Snapshot { schema: snapshot::SCHEMA, pid: std::process::id(), started_ms: snapshot::now_ms(), ..Default::default() };
    snap.model.name = name.clone();
    snap.model.layers = cfg.layers as u32;
    snap.model.experts = cfg.experts as u32;
    snap.model.experts_per_tok = cfg.k as u32;
    snap.settings.insert("marco".into(), "M2a: cache de experts".into());
    snap.settings.insert("tamanho do expert (MB)".into(), format!("{expert_mb:.2}"));
    snap.settings.insert("threads da CPU".into(), args.threads.to_string());
    let mut last_pub = Instant::now();
    let publish = |snap: &mut Snapshot, last: &mut Instant, force: bool| {
        if force || last.elapsed().as_millis() >= 200 {
            *last = Instant::now();
            snap.ts_ms = snapshot::now_ms();
            if let Some(p) = &publisher {
                let _ = p.publish(snap);
            }
        }
    };

    let device = Device::new(0)?;
    // A passada residente vai primeiro: os logits dela por token são a referência das conversas para as outras.
    let mut configs: Vec<(String, Option<(f64, Policy)>)> = vec![("resident".into(), None)];
    for p in &args.policies {
        match p.as_str() {
            "resident" => {}
            "cache_full" => configs.push(("cache_full".into(), Some((1.0, Policy::LruSync)))),
            "lru_sync" | "lru_cpu" => {
                let pol = if p == "lru_sync" { Policy::LruSync } else { Policy::LruCpu };
                for &b in &args.budgets {
                    configs.push((p.clone(), Some((b, pol))));
                }
            }
            _ => panic!("política desconhecida: {p}"),
        }
    }

    let mut all_ok = true;
    let vocab = cfg.vocab;
    // Logits de cada token das conversas na passada residente (tokens × vocabulário).
    let mut conv_ref: Vec<f32> = Vec::new();
    let mut pos_lines: Vec<String> = Vec::new();
    let mut exact_lines: Vec<String> = Vec::new();
    println!(
        "\n{:<10} {:>6} {:>5} | {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} {:>6} | {:>8} {:>8} | {:>11} | {:>14} | {}",
        "política", "orçam.", "slots", "ms/tok", "mediana", "tok/s", "acerto", "CPU", "cóp.s", "cóp.bg", "roteador", "CPU ms", "rot.igual", "sim ms/acerto", "logits texto-a / conversas"
    );
    for (pname, cc) in &configs {
        let stream = device.new_stream()?;
        let (mut g, slots) = match cc {
            None => (GpuModel::upload(&host, stream, max_len)?, slots_all),
            Some((frac, pol)) => {
                let slots = (frac * slots_all as f64) as usize;
                let cfgc = CacheConfig { slots, policy: *pol, backlog: args.backlog, cpu_threads: args.threads, admit: args.admit };
                (GpuModel::upload_cached(&host, stream, max_len, cfgc)?, slots)
            }
        };
        snap.settings.insert("política".into(), pname.clone());
        snap.settings.insert("orçamento de slots".into(), format!("{slots} de {slots_all}"));

        // 1. Logits do texto-a (também aquece os kernels).
        let mut text_check = Check::default();
        let mut hash = 0xcbf29ce484222325u64;
        for (t, &tok) in ref_tokens.iter().enumerate() {
            let (lg, _) = g.step(tok as usize)?;
            fnv(&mut hash, &lg);
            text_check.add(&lg, &ref_logits[t * vocab..(t + 1) * vocab]);
        }

        // 2. Conversas, com o cache vazio no começo (cheio, em `cache_full`).
        if let Experts::Cached(c) = &mut g.experts {
            c.clear()?;
            if pname == "cache_full" {
                c.fill()?;
            }
            c.stats = Default::default();
        }
        let reference = pname == "resident";
        let mut conv_check = Check::default();
        let mut idx = 0usize;
        let started = snapshot::now_ms();
        let mut ms = Vec::with_capacity(total_tokens);
        let mut by_pos: Vec<Vec<f64>> = vec![Vec::new(); POS_BUCKETS.len()];
        let (mut route_same, mut route_total) = (0usize, 0usize);
        let mut done = 0u64;
        for conv in &convs {
            g.reset();
            let n = conv.tokens.len();
            for (t, &tok) in conv.tokens.iter().enumerate() {
                let t0 = Instant::now();
                let (lg, routes) = g.step(tok as usize)?;
                let dt = t0.elapsed().as_secs_f64() * 1e3;
                ms.push(dt);
                fnv(&mut hash, &lg);
                if let Some(b) = POS_BUCKETS.iter().position(|&(a, z)| t >= a && t < z) {
                    by_pos[b].push(dt);
                }
                if reference {
                    conv_ref.extend_from_slice(&lg);
                } else {
                    conv_check.add(&lg, &conv_ref[idx * vocab..(idx + 1) * vocab]);
                }
                idx += 1;
                for (l, r) in routes.iter().enumerate() {
                    let want = &conv.experts[(l * n + t) * cfg.k..(l * n + t + 1) * cfg.k];
                    route_total += 1;
                    if r.experts.iter().zip(want).all(|(&a, &b)| a as i32 == b) {
                        route_same += 1;
                    }
                }
                done += 1;
                snap.state = State::Generating { tokens: done, max_tokens: total_tokens as u64 };
                snap.speed.decode_now = 1e3 / dt;
                snap.window.ms_per_token = dt;
                snap.cache.slots_total = slots as u32;
                match &g.experts {
                    Experts::Cached(c) => {
                        let s = &c.stats;
                        let need = (s.hits + s.sync_copies + s.cpu).max(1) as f64;
                        snap.cache.slots_resident = c.resident() as u32;
                        snap.cache.slots_loading = c.loading_now() as u32;
                        snap.cache.copies_inflight = c.loading_now() as u32;
                        snap.cache.hit_frac = s.hits as f64 / need;
                        snap.cache.copy_frac = s.sync_copies as f64 / need;
                        snap.cache.cpu_frac = s.cpu as f64 / need;
                        snap.cache.total_hits = s.hits;
                        snap.cache.total_copies = s.sync_copies + s.bg_copies;
                        snap.cache.total_cpu = s.cpu;
                    }
                    Experts::Resident(_) => {
                        snap.cache.slots_resident = slots as u32;
                        snap.cache.hit_frac = 1.0;
                    }
                }
                publish(&mut snap, &mut last_pub, false);
            }
        }
        let logits_ok = text_check.ok() && conv_check.ok();
        all_ok &= logits_ok;
        let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
        let med = median(&ms);
        pos_lines.push(format!(
            "{pname:<10} {:>6} | {}",
            cc.map(|(f, _)| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "100%".into()),
            by_pos.iter().map(|v| format!("{:>7.2} ({:>4})", median(v), v.len())).collect::<Vec<_>>().join(" ")
        ));
        let (hit, cpu, sync, bg, rw, cms) = match &g.experts {
            Experts::Cached(c) => {
                let s = &c.stats;
                let need = (s.hits + s.sync_copies + s.cpu).max(1) as f64;
                let tk = total_tokens as f64;
                (s.hits as f64 / need, s.cpu as f64 / need, s.sync_copies as f64 / tk, s.bg_copies as f64 / tk, s.router_wait_ms / tk, s.cpu_ms / tk)
            }
            Experts::Resident(_) => (1.0, 0.0, 0.0, 0.0, f64::NAN, 0.0),
        };
        let simtxt = match (&sim, cc) {
            (Some(sj), Some((frac, pol))) => sim_row(sj, *frac, pol.name()).map(|(m, h, _)| format!("{m:>6.2} {:>5.0}%", h * 100.0)).unwrap_or_default(),
            (Some(sj), None) => sj["params"]["tg_gpu"].as_f64().map(|t| format!("{:>6.2}   100%", 1e3 / t)).unwrap_or_default(),
            _ => String::new(),
        };
        let budget = cc.map(|(f, _)| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "100%".into());
        let counts = match &g.experts {
            Experts::Cached(c) => {
                let s = &c.stats;
                format!("{:>9} {:>9} {:>9} {:>9} {:>9}", s.hits, s.sync_copies, s.bg_copies, s.bg_dropped, s.cpu)
            }
            Experts::Resident(_) => format!("{:>9} {:>9} {:>9} {:>9} {:>9}", "–", "–", "–", "–", "–"),
        };
        exact_lines.push(format!("{pname:<10} {budget:>6} | {counts} | {hash:016x}"));
        let line = format!(
            "{pname:<10} {budget:>6} {slots:>5} | {mean:>7.2} {med:>7.2} {:>7.1} | {:>5.1}% {:>5.1}% {sync:>6.2} {bg:>6.2} | {rw:>8.2} {cms:>8.2} | {:>11} | {simtxt:>14} | {} / {}",
            1e3 / mean,
            hit * 100.0,
            cpu * 100.0,
            format!("{route_same}/{route_total}"),
            text_check.show(),
            if reference { "referência".to_string() } else { conv_check.show() }
        );
        println!("{line}");
        snap.speed.decode_last = 1e3 / mean;
        snap.requests.insert(0, Request {
            started_ms: started,
            output_tokens: total_tokens as u64,
            duration_ms: snapshot::now_ms() - started,
            decode_tok_s: 1e3 / mean,
            status: format!("{} {pname} {budget}", if logits_ok { "ok" } else { "falhou" }),
            ..Default::default()
        });
        snap.events.push(Event { ts_ms: snapshot::now_ms(), level: if logits_ok { "info" } else { "erro" }.into(), message: line });
        if snap.events.len() > 64 {
            snap.events.remove(0);
        }
        publish(&mut snap, &mut last_pub, true);
    }
    snap.state = if all_ok { State::Idle } else { State::Error { message: "M2a: logits divergem da referência".into() } };
    publish(&mut snap, &mut last_pub, true);
    println!("\nmediana de ms/token por posição no contexto (tokens na faixa):");
    println!(
        "{:<10} {:>6} | {}",
        "política",
        "orçam.",
        POS_BUCKETS.iter().map(|&(a, z)| if z == usize::MAX { format!("{:>14}", format!("{a}+")) } else { format!("{:>14}", format!("{a}-{z}")) }).collect::<Vec<_>>().join(" ")
    );
    for l in &pos_lines {
        println!("{l}");
    }
    println!("\ncontagens exatas nas conversas e hash dos logits (texto-a e conversas):");
    println!("{:<10} {:>6} | {:>9} {:>9} {:>9} {:>9} {:>9} | {}", "política", "orçam.", "acertos", "cóp.s", "cóp.bg", "descart.", "CPU", "hash");
    for l in &exact_lines {
        println!("{l}");
    }
    println!("\nms/tok e tok/s: média nas conversas, mediana ao lado; acerto e CPU: fração dos experts pedidos; cóp.s e cóp.bg: cópias por token;");
    println!("roteador: espera do host pela leitura dos logits, por token (inclui o cálculo da GPU até ali); CPU ms: experts na CPU por token;");
    println!("sim: ms/token e acerto previstos pelo simulador para o mesmo orçamento e política (residente: 1000/tg_gpu do llama.cpp).");
    println!("logits: maior erro relativo (L2) contra o transformers no texto-a e, nas conversas, contra a passada residente; limite {MAX_REL:.0e}, argmax igual;");
    println!("resultado: {}", if all_ok { "logits dentro da tolerância em todas as configurações" } else { "FALHOU" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::Check;

    #[test]
    fn nan_fora_do_argmax_reprova() {
        // `f64::max` descarta NaN, e um -NaN fica abaixo de tudo no `total_cmp`, então nem o erro nem o argmax o pegariam.
        let want = [1.0f32, 5.0, 2.0];
        let mut c = Check::default();
        c.add(&[1.0, 5.0, -f32::NAN], &want);
        assert!(!c.ok());
        let mut c = Check::default();
        c.add(&want, &want);
        c.add(&[1.0, 5.0, f32::NAN], &want);
        c.add(&want, &want);
        assert!(!c.ok(), "um NaN no meio também reprova");
    }
}
