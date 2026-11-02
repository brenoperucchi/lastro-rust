//! A1.4b: o Granite MoE inteiro pelo motor D1 (`Engine` + `CudaBackend::new_model`), nas mesmas conversas e com a mesma
//! conferência de logits do `m2a-cache`. É a linha L1/L4/L6 do critério `qual-runs/A1-MEDIDA-20261101/criterio.txt`; as linhas
//! L2/L3 (o `lfu_cpu` de hoje e o residente) são do `m2a-cache`.
//!
//! Configurações (`--policies`, na ordem dada; a `resident` roda sempre primeiro e dá os logits de referência das conversas):
//!   resident        o `GpuModel` com todos os experts residentes (referência de logits e de ms/token da sessão);
//!   motor_cheio     o motor com slots para todos os experts, preenchido antes das conversas (só acertos): a base de espera;
//!   motor_evento    o motor, orçamento `--budgets`, admissão por evento (a linha principal);
//!   motor_barreira  o motor, orçamento `--budgets`, admissão com barreira por token (`TokenBarrier`; só relato).
//!
//! Para cada configuração do motor: (1) confere os logits do `texto-a` contra o transformers num motor à parte (o mesmo caminho,
//! descartado depois, para o cache da medida começar vazio); (2) sobe o motor de novo, passa o aquecimento (sem cronometrar nem
//! conferir) e as conversas token a token, medindo o tempo de `begin_token` a `read_logits` e conferindo os logits contra os da
//! passada residente (limite 1e-4, argmax igual). Os contadores são a diferença entre o início da medida e o fim.
//!
//! Roteamento: depois de cada token (fora do cronômetro) os ids que a GPU escolheu em cada camada (`Engine::layer_ids`) são
//! comparados, como conjunto, com os do traço; no máximo 0,01% dos pares (token, camada) podem diferir (quase empates). O total
//! de experts pedidos (`needed` = tokens × camadas × top-k) confere a janela dos contadores.
//!
//!   a1-motor --model <dir> --traces <dir> --ref <dir> [--convs 4,5,6,7] [--budgets 0.1,0.25,0.5] [--warmup 0,1,2,3]
//!            [--policies resident,motor_cheio,motor_evento,motor_barreira] [--threads N] [--backlog N] [--ms-dir <dir>]
use cuda_core::Device;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream};
use moe_rt::gpu_model::{ATTN_BN, GpuModel};
use moe_rt::model::{Config as ModelConfig, HostModel};
use moe_rt::motor::{AdmitMode, Config, Engine, Stats};
use moe_rt::npy::Npy;
use moe_rt::weights::Checkpoint;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Erro relativo máximo (norma L2 do vetor de logits) contra a referência, como no `m2a-cache`.
const MAX_REL: f64 = 1e-4;

struct Args {
    model: PathBuf,
    traces: PathBuf,
    refdir: PathBuf,
    convs: Vec<usize>,
    warmup: Vec<usize>,
    budgets: Vec<f64>,
    policies: Vec<String>,
    threads: usize,
    backlog: usize,
    ms_dir: Option<PathBuf>,
    sem_aquecimento: bool,
}

fn parse_args() -> Args {
    let mut a = std::env::args().skip(1);
    let mut r = Args {
        model: PathBuf::new(),
        traces: PathBuf::new(),
        refdir: PathBuf::new(),
        convs: vec![4, 5, 6, 7],
        warmup: vec![],
        budgets: vec![0.1, 0.25, 0.5],
        policies: vec!["resident".into(), "motor_cheio".into(), "motor_evento".into()],
        threads: 8,
        backlog: 4,
        ms_dir: None,
        sem_aquecimento: false,
    };
    while let Some(k) = a.next() {
        let mut v = || a.next().unwrap_or_else(|| panic!("{k} pede um valor"));
        match k.as_str() {
            "--model" => r.model = PathBuf::from(v()),
            "--traces" => r.traces = PathBuf::from(v()),
            "--ref" => r.refdir = PathBuf::from(v()),
            "--convs" => r.convs = v().split(',').map(|s| s.trim().parse().expect("--convs")).collect(),
            "--warmup" => r.warmup = v().split(',').filter(|s| !s.trim().is_empty()).map(|s| s.trim().parse().expect("--warmup")).collect(),
            "--budgets" => r.budgets = v().split(',').map(|s| s.trim().parse().expect("--budgets")).collect(),
            "--policies" => r.policies = v().split(',').map(|s| s.trim().to_string()).collect(),
            "--threads" => r.threads = v().parse().expect("--threads"),
            "--backlog" => r.backlog = v().parse().expect("--backlog"),
            "--ms-dir" => r.ms_dir = Some(PathBuf::from(v())),
            "--sem-aquecimento" => r.sem_aquecimento = true,
            _ => panic!("opção desconhecida: {k}"),
        }
    }
    r
}

struct Conv {
    tokens: Vec<i32>,
    /// Os ids do traço, `(camada · n + token) · k + j`.
    experts: Vec<i32>,
}

fn load_conv(dir: &Path) -> Result<Conv, String> {
    Ok(Conv { tokens: Npy::read(&dir.join("tokens.npy"))?.i32()?, experts: Npy::read(&dir.join("experts.npy"))?.i32()? })
}

/// Os erros do motor e do backend não implementam `std::error::Error`: viram texto de depuração.
trait D<T> {
    fn d(self) -> Result<T, String>;
}

impl<T, E: std::fmt::Debug> D<T> for Result<T, E> {
    fn d(self) -> Result<T, String> {
        self.map_err(|e| format!("{e:?}"))
    }
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

#[derive(Default, Clone, Copy)]
struct Check {
    max_rel: f64,
    argmax_bad: usize,
}

impl Check {
    fn add(&mut self, got: &[f32], want: &[f32]) {
        // Um vetor de outro tamanho, vazio ou com elemento não finito reprova (o `zip` aceitaria um prefixo).
        if got.is_empty() || got.len() != want.len() || got.iter().any(|x| !x.is_finite()) {
            self.max_rel = f64::INFINITY;
            self.argmax_bad += 1;
            return;
        }
        let rel = rel_err(got, want);
        // `f64::max` descartaria um NaN: ele tem de reprovar.
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

/// FNV-1a de 64 bits sobre os bits de cada logit (como no `m2a-cache`).
fn fnv(h: &mut u64, v: &[f32]) {
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            *h = (*h ^ b as u64).wrapping_mul(0x100000001b3);
        }
    }
}

/// Os ids da GPU e os do traço são o mesmo conjunto (a ordem dentro da camada não conta).
fn mesmos_ids(got: &[u32], want: &[i32]) -> bool {
    let mut g = got.to_vec();
    let mut w: Vec<u32> = want.iter().map(|&e| e as u32).collect();
    g.sort_unstable();
    w.sort_unstable();
    g == w
}

fn pct(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    sorted[(((sorted.len() - 1) as f64) * p).round() as usize]
}

/// O motor de uma configuração: o backend, a stream de cálculo e o `Engine`.
struct Motor {
    b: std::sync::Arc<CudaBackend>,
    compute: CudaStream,
    eng: Engine<CudaBackend>,
}

fn sobe(device: &std::sync::Arc<Device>, host: &HostModel, cfg: &ModelConfig, max_len: usize, slots: usize, admit: AdmitMode, prefill: Vec<u32>, a: &Args) -> Result<Motor, Box<dyn std::error::Error>> {
    let (compute, copy) = (CudaStream(device.new_stream().d()?), CudaStream(device.new_stream().d()?));
    let (b, ex) = CudaBackend::new_model(device.clone(), host, slots, max_len, &compute).d()?;
    let mcfg = Config { layers: cfg.layers, experts: cfg.experts, topk: cfg.k, width: cfg.hidden, slots, backlog: a.backlog, admit, first_token: 1, prefill };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).d()?, mcfg.first_token).d()?;
    let cpu = CpuServe { cpu: Cpu::new(a.threads), hidden: cfg.hidden, inter: cfg.inter, experts: cfg.experts, weights: ex.clone() };
    let eng = Engine::new(b.clone(), parts, mcfg, cpu).d()?;
    Ok(Motor { b, compute, eng })
}

impl Motor {
    /// Um token: do embedding aos logits, já em CPU.
    fn passo(&mut self, tok: usize) -> Result<Vec<f32>, String> {
        self.b.begin_token(&self.compute, tok).d()?;
        let p = self.eng.next_published().d()?;
        self.eng.run_token(p).d()?;
        self.b.read_logits(&self.compute).d()
    }
}

fn sub(a: &Stats, b: &Stats) -> Stats {
    Stats { needed: a.needed - b.needed, hits: a.hits - b.hits, cpu: a.cpu - b.cpu, copies: a.copies - b.copies, dropped: a.dropped - b.dropped, deferred: a.deferred - b.deferred }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(args.model.join("config.json"))?)?)?;
    let host = HostModel::load(&Checkpoint::open(&args.model)?, cfg.clone())?;
    let convs: Vec<Conv> = args.convs.iter().map(|c| load_conv(&args.traces.join(format!("conversa-{c}")))).collect::<Result<_, _>>()?;
    let warm: Vec<Conv> = args.warmup.iter().map(|c| load_conv(&args.traces.join(format!("conversa-{c}")))).collect::<Result<_, _>>()?;
    let ref_tokens = Npy::read(&args.refdir.join("tokens.npy"))?.i32()?;
    let ref_logits = Npy::read(&args.refdir.join("logits.npy"))?.f32()?;
    let total_tokens: usize = convs.iter().map(|c| c.tokens.len()).sum();
    let longest = convs.iter().chain(&warm).map(|c| c.tokens.len()).max().unwrap_or(0).max(ref_tokens.len());
    let max_len = (longest + 1).next_multiple_of(ATTN_BN);
    let slots_all = cfg.layers * cfg.experts;
    let name = args.model.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    println!(
        "A1.4b: {name}; conversas {:?} ({total_tokens} tokens); {slots_all} experts; CPU com {} threads; backlog {}{}",
        args.convs,
        args.threads,
        args.backlog,
        if args.warmup.is_empty() { String::new() } else { format!("; aquecimento {:?}", args.warmup) }
    );
    if let Some(d) = &args.ms_dir {
        std::fs::create_dir_all(d)?;
    }
    // Sem aquecimento o primeiro token cronometrado paga a partida do motor (o `prefill` do motor_cheio levou 485 ms): recusa-se.
    if args.warmup.is_empty() && !args.sem_aquecimento && args.policies.iter().any(|p| p.starts_with("motor_")) {
        return Err("as linhas do motor pedem --warmup (ou --sem-aquecimento, só para testes curtos)".into());
    }

    let device = Device::new(0)?;
    let mut configs: Vec<(String, Option<(f64, AdmitMode)>)> = vec![("resident".into(), None)];
    for p in &args.policies {
        match p.as_str() {
            "resident" => {}
            "motor_cheio" => configs.push(("motor_cheio".into(), Some((1.0, AdmitMode::Event)))),
            "motor_evento" | "motor_barreira" => {
                let admit = if p == "motor_evento" { AdmitMode::Event } else { AdmitMode::TokenBarrier };
                for &b in &args.budgets {
                    configs.push((p.clone(), Some((b, admit))));
                }
            }
            _ => panic!("política desconhecida: {p}"),
        }
    }

    let vocab = cfg.vocab;
    let mut conv_ref: Vec<f32> = Vec::new();
    let mut all_ok = true;
    let mut cauda: Vec<String> = Vec::new();
    let mut exatas: Vec<String> = Vec::new();
    println!(
        "\n{:<14} {:>6} {:>5} | {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} {:>6} | {:>11} | logits texto-a / conversas",
        "política", "orçam.", "slots", "ms/tok", "mediana", "tok/s", "acerto", "CPU", "cóp.", "descart.", "needed e roteamento"
    );
    for (pname, cc) in &configs {
        let reference = cc.is_none();
        let budget = cc.map(|(f, _)| format!("{:.0}%", f * 100.0)).unwrap_or_else(|| "100%".into());
        let slots = cc.map(|(f, _)| (f * slots_all as f64) as usize).unwrap_or(slots_all);
        let prefill: Vec<u32> = if pname == "motor_cheio" { (0..slots_all as u32).collect() } else { vec![] };
        let mut hash = 0xcbf29ce484222325u64;
        let mut text_check = Check::default();
        let mut conv_check = Check::default();
        let mut ms = Vec::with_capacity(total_tokens);
        let mut stats = Stats::default();
        let mut route: Option<(usize, usize)> = None;

        if let Some((_, admit)) = cc {
            // 1. texto-a num motor à parte.
            let mut m = sobe(&device, &host, &cfg, max_len, slots, *admit, prefill.clone(), &args)?;
            m.b.reset_sequence().d()?;
            for (t, &tok) in ref_tokens.iter().enumerate() {
                let lg = m.passo(tok as usize)?;
                fnv(&mut hash, &lg);
                text_check.add(&lg, &ref_logits[t * vocab..(t + 1) * vocab]);
            }
            m.eng.close().d()?;
            // 2. o motor da medida, com o cache como veio (vazio ou cheio), aquecido nas conversas de aquecimento.
            let mut m = sobe(&device, &host, &cfg, max_len, slots, *admit, prefill, &args)?;
            for conv in &warm {
                m.b.reset_sequence().d()?;
                for &tok in &conv.tokens {
                    m.passo(tok as usize)?;
                }
            }
            // Os contadores do `Published` valem até o fim do token anterior: o do primeiro token medido é o ponto de partida.
            let mut base: Option<Stats> = None;
            let mut idx = 0usize;
            let (mut route_same, mut route_total) = (0usize, 0usize);
            for conv in &convs {
                m.b.reset_sequence().d()?;
                let n = conv.tokens.len();
                for (t, &tok) in conv.tokens.iter().enumerate() {
                    let t0 = Instant::now();
                    m.b.begin_token(&m.compute, tok as usize).d()?;
                    let p = m.eng.next_published().d()?;
                    if base.is_none() {
                        base = Some(p.stats().clone());
                    }
                    m.eng.run_token(p).d()?;
                    let lg = m.b.read_logits(&m.compute).d()?;
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                    fnv(&mut hash, &lg);
                    conv_check.add(&lg, &conv_ref[idx * vocab..(idx + 1) * vocab]);
                    idx += 1;
                    // Os ids que a GPU escolheu em cada camada contra o traço (fora do cronômetro, motor parado entre tokens).
                    for l in 0..cfg.layers {
                        route_total += 1;
                        route_same += mesmos_ids(&m.eng.layer_ids(l), &conv.experts[(l * n + t) * cfg.k..(l * n + t + 1) * cfg.k]) as usize;
                    }
                }
            }
            route = Some((route_same, route_total));
            // O `Published` seguinte traz os contadores até o fim do último token; solta-se antes de fechar, como no teste do A1.3.
            let fim = m.eng.next_published().d().map(|p| p.stats().clone())?;
            stats = sub(&fim, &base.unwrap_or_default());
            m.eng.close().d()?;
        } else {
            let stream = device.new_stream()?;
            let mut g = GpuModel::upload(&host, stream, max_len)?;
            for (t, &tok) in ref_tokens.iter().enumerate() {
                let (lg, _) = g.step(tok as usize)?;
                fnv(&mut hash, &lg);
                text_check.add(&lg, &ref_logits[t * vocab..(t + 1) * vocab]);
            }
            for conv in &convs {
                g.reset();
                for &tok in &conv.tokens {
                    let t0 = Instant::now();
                    let (lg, _) = g.step(tok as usize)?;
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                    fnv(&mut hash, &lg);
                    conv_ref.extend_from_slice(&lg);
                }
            }
        }

        let logits_ok = text_check.ok() && conv_check.ok();
        let needed_ok = reference || stats.needed == (total_tokens * cfg.layers * cfg.k) as u64;
        // Roteamento: no máximo 0,01% dos pares (token, camada) com outro conjunto de ids (os quase empates do critério).
        let (rs, rt) = route.unwrap_or((0, 0));
        let route_ok = reference || (rt - rs) * 10_000 <= rt;
        all_ok &= logits_ok && needed_ok && route_ok;
        let mean = ms.iter().sum::<f64>() / ms.len().max(1) as f64;
        let mut sorted = ms.clone();
        sorted.sort_by(f64::total_cmp);
        let med = pct(&sorted, 0.5);
        let need = stats.needed.max(1) as f64;
        let tk = total_tokens as f64;
        println!(
            "{pname:<14} {budget:>6} {slots:>5} | {mean:>7.2} {med:>7.2} {:>7.1} | {:>5.1}% {:>5.1}% {:>6.2} {:>6.2} | {:>11} | {} / {}",
            1e3 / mean,
            stats.hits as f64 / need * 100.0,
            stats.cpu as f64 / need * 100.0,
            stats.copies as f64 / tk,
            stats.dropped as f64 / tk,
            if reference { "–".to_string() } else { format!("{needed_ok} rot {rs}/{rt}") },
            text_check.show(),
            if reference { "referência".to_string() } else { conv_check.show() }
        );
        let limite = 3.0 * med;
        let acima = ms.iter().filter(|&&x| x > limite).count();
        let mut top: Vec<(usize, f64)> = ms.iter().cloned().enumerate().collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        cauda.push(format!(
            "{pname:<14} {budget:>6} | p50 {:>7.2} p95 {:>7.2} p99 {:>7.2} max {:>7.2} | acima de 3×mediana: {acima} | 10 maiores (índice:ms) {}",
            med,
            pct(&sorted, 0.95),
            pct(&sorted, 0.99),
            sorted.last().copied().unwrap_or(f64::NAN),
            top.iter().take(10).map(|(i, v)| format!("{i}:{v:.1}")).collect::<Vec<_>>().join(" ")
        ));
        exatas.push(format!(
            "{pname:<14} {budget:>6} | needed {:>9} hits {:>9} cpu {:>9} cópias {:>9} descart. {:>9} adiadas {:>9} | {hash:016x}",
            stats.needed, stats.hits, stats.cpu, stats.copies, stats.dropped, stats.deferred
        ));
        if let Some(d) = &args.ms_dir {
            let txt: String = ms.iter().map(|x| format!("{x:.4}\n")).collect();
            std::fs::write(d.join(format!("a1-motor-ms-{pname}-{}.txt", budget.trim_end_matches('%'))), txt)?;
        }
    }
    println!("\ncauda de latência (ms por token):");
    for l in &cauda {
        println!("{l}");
    }
    println!("\ncontagens exatas nas conversas e hash dos logits (texto-a e conversas):");
    for l in &exatas {
        println!("{l}");
    }
    println!("\nresultado: {}", if all_ok { "logits dentro da tolerância e experts pedidos certos em todas as configurações" } else { "FALHOU" });
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{mesmos_ids, Check};

    #[test]
    fn troca_de_um_id_ou_tamanho_diferente_reprova_o_roteamento() {
        assert!(mesmos_ids(&[3, 1, 7], &[7, 3, 1]), "a ordem não conta");
        assert!(!mesmos_ids(&[3, 1, 7], &[7, 3, 2]), "um id trocado reprova");
        assert!(!mesmos_ids(&[3, 1], &[3, 1, 7]), "tamanho diferente reprova");
    }

    #[test]
    fn prefixo_vazio_extra_e_nao_finito_reprovam() {
        let want = [5.0f32, 1.0];
        let mut c = Check::default();
        c.add(&[5.0], &want);
        assert!(!c.ok(), "um prefixo não pode passar");
        let mut c = Check::default();
        c.add(&[5.0, 1.0, 0.0], &want);
        assert!(!c.ok(), "elementos extras reprovam");
        let mut c = Check::default();
        c.add(&[], &want);
        assert!(!c.ok(), "vetor vazio reprova");
        let mut c = Check::default();
        c.add(&[5.0, f32::INFINITY], &want);
        assert!(!c.ok(), "Inf reprova");
        let mut c = Check::default();
        c.add(&[0.0, 0.0], &[0.0, 0.0]);
        assert!(!c.ok(), "referência toda zero não tem erro relativo definido e reprova");
        let mut c = Check::default();
        c.add(&want, &want);
        assert!(c.ok());
    }
}
