//! A2-PASSO E4b: o programa de medida do passo de várias posições (M = 8 contra M = 1 do próprio motor), pelo critério
//! `qual-runs/A2-LOTE-20261103/criterio-e4b.txt` (aprovado pelo Breno em 06/11/2026). NÃO RODA SEM AUTORIZAÇÃO.
//!
//! Só roda na GPU com `E4B_MEDIDA_AUTORIZADA=1` no ambiente: o "autorizo" explícito do Breno, que nomeia a janela, abre isso. Sem a
//! variável o programa recusa antes de tocar o device. `--ensaio` imprime o plano sem abrir o device (e sem a variável).
//!
//! Por célula (um modelo), nesta ordem:
//!  1. Referência X: a passada residente (`GpuModel`) sobre os 64 primeiros tokens do `texto-a`; os logits da posição 63.
//!  2. Passada de portão (a): motor de 10% de slots com a TABELA CONGELADA (prefill de chaves espaçadas por igual e `backlog = 0`, que
//!     recusa toda admissão) e divisão mista fixa; M = 8 contra token a token, logits dos fins de bloco e o KV inteiro a 1e-5 (norma
//!     máxima relativa), contadores `hits > 0` e `cpu > 0` e iguais entre os dois. Falhou: a célula é INVALIDA e nenhum tempo é lido.
//!  3. Controle C antes: instância à parte do motor por token (`max_positions = 1`), 5 repetições do bloco de 64 posições.
//!  4. 15 pares (2 de aquecimento), ABBA a partir do primeiro par usado: uma repetição de M = 1 e uma de M = 8 sobre o mesmo bloco,
//!     as duas pelo caminho do passo (`begin_step_tokens`, `enqueue_step`, `finish_token`; M = 1 com `m = 1`). Região cronometrada:
//!     do início do primeiro `begin_step_tokens` ao fim do último `finish_token`; os logits são lidos depois dela.
//!  5. Controle C depois: nova instância, 5 repetições.
//!  6. Portão (b) em toda repetição: logits da última posição contra X, L2 relativa 1e-4 e argmax igual. Falhou: INVALIDA.
//!  7. Rótulo pela regra de `moe_rt::e4b_regra` (a mesma testada em CPU); no fim o resultado do A2 se as duas células rodaram.
//!
//! Não implementado de propósito: os carimbos de g_M (5 pares extras com evento de GPU no início e no fim de cada camada). Os eventos
//! do backend são criados sem cronometragem e instrumentar as camadas é mudança de backend. g_M é só descritivo e não entra em regra
//! de decisão; o relatório diz "g_M: não medido".
//!
//! Pontos que o critério não fixa e que este programa decide (registrados no relatório e na revisão):
//!  - cada instância do controle C faz um bloco de partida SEM cronômetro antes das 5 repetições (a instância nasce fria e o critério
//!    pede as 10 observações válidas, sem aquecimento próprio);
//!  - um erro do motor numa repetição encerra a célula daquele ponto em diante (o motor não continua depois de um erro): os pares
//!    seguintes ficam perdidos, e a regra decide com o n que sobrou;
//!  - a integridade (hash do binário e do critério antes e depois) falhando torna todas as células INVALIDA.
//!
//!   e4b-medida --celula <nome>:<dir do modelo>:<dir da referência texto-a> [--celula ...] --saida <dir> [--criterio <arquivo>]
//!              [--threads N] [--afin a0|a1w|a1m] [--ensaio]
use cuda_core::Device;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream};
use moe_rt::e4b_regra::*;
use moe_rt::gpu_model::{GpuModel, ATTN_BN};
use moe_rt::model::{Config as ModelConfig, HostModel};
use moe_rt::motor::{AdmitMode, Config, Engine, Stats};
use moe_rt::npy::Npy;
use moe_rt::weights::Checkpoint;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const VARIAVEL: &str = "E4B_MEDIDA_AUTORIZADA";
/// Fração de slots (seção 2: 10% como no A1-DIAG10).
const FRACAO_SLOTS: f64 = 0.10;
const POSICOES_MAX: usize = 8;
const BACKLOG: usize = 4;

struct Args {
    celulas: Vec<(String, PathBuf, PathBuf)>,
    saida: PathBuf,
    criterio: PathBuf,
    threads: usize,
    afin: moe_rt::affinity::Afin,
    ensaio: bool,
}

fn parse_args(v: Vec<String>) -> Result<Args, String> {
    let mut a = v.into_iter();
    let mut r = Args {
        celulas: vec![],
        saida: PathBuf::new(),
        criterio: PathBuf::from("/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-e4b.txt"),
        threads: 8,
        afin: moe_rt::affinity::Afin::A0,
        ensaio: false,
    };
    while let Some(k) = a.next() {
        let mut val = || a.next().ok_or_else(|| format!("{k} pede um valor"));
        match k.as_str() {
            "--celula" => {
                let x = val()?;
                let p: Vec<&str> = x.splitn(3, ':').collect();
                if p.len() != 3 || p.iter().any(|s| s.is_empty()) {
                    return Err(format!("--celula nome:modelo:referencia, não {x}"));
                }
                r.celulas.push((p[0].to_string(), PathBuf::from(p[1]), PathBuf::from(p[2])));
            }
            "--saida" => r.saida = PathBuf::from(val()?),
            "--criterio" => r.criterio = PathBuf::from(val()?),
            "--threads" => r.threads = val()?.parse().map_err(|_| "--threads".to_string())?,
            "--afin" => r.afin = moe_rt::affinity::Afin::parse(&val()?)?,
            "--ensaio" => r.ensaio = true,
            _ => return Err(format!("opção desconhecida: {k}")),
        }
    }
    if r.celulas.is_empty() {
        return Err("pelo menos uma --celula".into());
    }
    if r.celulas.len() > 2 {
        return Err("no máximo duas células (1B e 3B)".into());
    }
    if r.saida.as_os_str().is_empty() {
        return Err("--saida é obrigatória".into());
    }
    Ok(r)
}

/// O portão de autorização: só `1` abre. Recusa antes de qualquer acesso ao device.
fn exige_autorizacao(valor: Option<&str>) -> Result<(), String> {
    match valor {
        Some("1") => Ok(()),
        _ => Err(format!("a medida do E4b na GPU só roda com o 'autorizo' explícito do Breno ({VARIAVEL}=1); veja o critério em criterio-e4b.txt")),
    }
}

/// As chaves da tabela congelada: `slots` chaves espaçadas por igual no espaço camada·E + expert (divisão mista fixa).
fn chaves_congeladas(slots_all: usize, slots: usize) -> Vec<u32> {
    (0..slots).map(|i| (i * slots_all / slots) as u32).collect()
}

trait D<T> {
    fn d(self) -> Result<T, String>;
}

impl<T, E: std::fmt::Debug> D<T> for Result<T, E> {
    fn d(self) -> Result<T, String> {
        self.map_err(|e| format!("{e:?}"))
    }
}

struct Motor {
    b: Arc<CudaBackend>,
    compute: CudaStream,
    eng: Engine<CudaBackend>,
}

struct Cenario<'a> {
    device: &'a Arc<Device>,
    host: &'a HostModel,
    cfg: &'a ModelConfig,
    max_len: usize,
    threads: usize,
}

impl Cenario<'_> {
    fn sobe(&self, slots: usize, backlog: usize, prefill: Vec<u32>, max_positions: usize) -> Result<Motor, String> {
        let (compute, copy) = (CudaStream(self.device.new_stream().d()?), CudaStream(self.device.new_stream().d()?));
        let (b, ex) = CudaBackend::new_model_positions(self.device.clone(), self.host, slots, self.max_len, &compute, max_positions).d()?;
        let c = self.cfg;
        let mcfg = Config { layers: c.layers, experts: c.experts, topk: c.k, width: c.hidden, slots, backlog, admit: AdmitMode::Event, first_token: 1, prefill, max_positions };
        let parts = b.parts(compute.clone(), copy, b.weights(&ex).d()?, mcfg.first_token).d()?;
        let cpu = CpuServe { cpu: Cpu::new(self.threads), hidden: c.hidden, inter: c.inter, experts: c.experts, weights: ex.clone() };
        let eng = Engine::new(b.clone(), parts, mcfg, cpu).d()?;
        Ok(Motor { b, compute, eng })
    }

    fn slots_10(&self) -> usize {
        (FRACAO_SLOTS * (self.cfg.layers * self.cfg.experts) as f64) as usize
    }
}

impl Motor {
    /// Um passo de `toks.len()` posições pelo caminho do passo. Devolve os contadores do `Published` do passo.
    fn passo(&mut self, toks: &[usize]) -> Result<Stats, String> {
        self.b.begin_step_tokens(&self.compute, toks).d()?;
        let p = self.eng.next_published().d()?;
        let s = p.stats().clone();
        self.eng.enqueue_step(p, toks.len()).d()?;
        self.eng.finish_token().d()?;
        Ok(s)
    }

    /// Um token pelo motor por token de sempre (o controle C).
    fn token(&mut self, tok: usize) -> Result<(), String> {
        self.b.begin_token(&self.compute, tok).d()?;
        let p = self.eng.next_published().d()?;
        self.eng.run_token(p).d()
    }

    /// Os contadores até o fim do último passo (o `Published` seguinte).
    fn stats_agora(&mut self) -> Result<Stats, String> {
        self.eng.next_published().d().map(|p| p.stats().clone())
    }

    /// Uma repetição do bloco: reinicia a sequência (fora do cronômetro), cronometra e lê os logits da última posição depois.
    /// `m` = 0 é o controle C (motor por token); `m >= 1` o caminho do passo com blocos de `m` posições.
    fn repeticao(&mut self, tokens: &[usize], m: usize) -> Result<(f64, Vec<f32>), String> {
        self.b.reset_sequence().d()?;
        let t0 = Instant::now();
        if m == 0 {
            for &t in tokens {
                self.token(t)?;
            }
        } else {
            for c in tokens.chunks(m) {
                self.passo(c)?;
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let logits = self.b.read_logits(&self.compute).d()?;
        Ok((secs, logits))
    }
}

#[derive(Default)]
struct PortaoA {
    ok: bool,
    logits_max: f64,
    kv_max: f64,
    hits: (u64, u64),
    cpu: (u64, u64),
    needed: (u64, u64),
    motivo: String,
}

fn sub(a: &Stats, b: &Stats) -> Stats {
    Stats { needed: a.needed - b.needed, hits: a.hits - b.hits, cpu: a.cpu - b.cpu, copies: a.copies - b.copies, dropped: a.dropped - b.dropped, deferred: a.deferred - b.deferred }
}

/// Uma passada do portão (a): blocos de `m` posições com a tabela congelada; devolve os logits dos fins de bloco de 8, o KV e os contadores.
fn passada_portao(c: &Cenario, tokens: &[usize], m: usize) -> Result<(Vec<Vec<f32>>, Vec<(Vec<f32>, Vec<f32>)>, Stats), String> {
    let slots = c.slots_10();
    let mut mt = c.sobe(slots, 0, chaves_congeladas(c.cfg.layers * c.cfg.experts, slots), POSICOES_MAX)?;
    mt.b.reset_sequence().d()?;
    let (mut base, mut fins) = (None, Vec::new());
    for (i, ch) in tokens.chunks(m).enumerate() {
        let s = mt.passo(ch)?;
        base.get_or_insert(s);
        let fim_do_bloco_de_8 = ((i + 1) * m) % POSICOES_MAX == 0;
        if fim_do_bloco_de_8 {
            fins.push(mt.b.read_logits(&mt.compute).d()?);
        }
    }
    let fim = mt.stats_agora()?;
    let kv = (0..c.cfg.layers).map(|l| mt.b.read_kv(&mt.compute, l, tokens.len()).d()).collect::<Result<Vec<_>, _>>()?;
    mt.eng.close().d()?;
    Ok((fins, kv, sub(&fim, &base.unwrap_or_default())))
}

fn portao_a(c: &Cenario, tokens: &[usize]) -> PortaoA {
    let mut r = PortaoA::default();
    let go = || -> Result<PortaoA, String> {
        let (l8, kv8, s8) = passada_portao(c, tokens, POSICOES_MAX)?;
        let (l1, kv1, s1) = passada_portao(c, tokens, 1)?;
        let mut r = PortaoA { hits: (s8.hits, s1.hits), cpu: (s8.cpu, s1.cpu), needed: (s8.needed, s1.needed), ..Default::default() };
        if l8.len() != l1.len() {
            r.motivo = "número de fins de bloco diferente".into();
            return Ok(r);
        }
        r.logits_max = l8.iter().zip(&l1).map(|(a, b)| erro_max_rel(a, b)).fold(0.0, f64::max);
        r.kv_max = kv8.iter().zip(&kv1).map(|((k8, v8), (k1, v1))| erro_max_rel(k8, k1).max(erro_max_rel(v8, v1))).fold(0.0, f64::max);
        let contadores_ok = s8.hits > 0 && s8.cpu > 0 && s8.hits == s1.hits && s8.cpu == s1.cpu && s8.needed == s1.needed;
        r.ok = r.logits_max <= TOL_PORTAO_MAX && r.kv_max <= TOL_PORTAO_MAX && contadores_ok;
        if !r.ok {
            r.motivo = format!("logits {:.2e}, KV {:.2e}, contadores_ok {contadores_ok}", r.logits_max, r.kv_max);
        }
        Ok(r)
    };
    match go() {
        Ok(x) => r = x,
        Err(e) => r.motivo = format!("erro: {e}"),
    }
    r
}

#[derive(Clone, Debug)]
struct LinhaPar {
    par: usize,
    aquecimento: bool,
    m1_primeiro: bool,
    tp1: Option<f64>,
    tp8: Option<f64>,
    erro1: f64,
    erro8: f64,
}

/// As repetições do controle C: 5 num motor por token novo, depois de um bloco de partida sem cronômetro.
fn controle(c: &Cenario, tokens: &[usize], x: &[f32], portoes_ok: &mut bool) -> Vec<Option<f64>> {
    let slots = c.slots_10();
    let mut out: Vec<Option<f64>> = vec![None; C_REPETICOES_LADO];
    let mut go = || -> Result<(), String> {
        let mut mt = c.sobe(slots, BACKLOG, vec![], 1)?;
        mt.repeticao(tokens, 0)?; // partida, sem cronômetro e sem contar
        for slot in out.iter_mut() {
            let (secs, lg) = mt.repeticao(tokens, 0)?;
            if !portao_repeticao(&lg, x) {
                *portoes_ok = false;
            }
            *slot = Some(secs / tokens.len() as f64);
        }
        mt.eng.close().d()
    };
    if let Err(e) = go() {
        eprintln!("controle C: {e}");
    }
    out
}

struct SaidaCelula {
    rotulo: Rotulo,
    relatorio: String,
}

fn celula(c: &Cenario, nome: &str, ref_dir: &Path, saida: &Path, afin: moe_rt::affinity::Afin) -> Result<SaidaCelula, Box<dyn std::error::Error>> {
    let ref_tokens = Npy::read(&ref_dir.join("tokens.npy"))?.i32()?;
    if ref_tokens.len() < BLOCO {
        return Err(format!("a referência tem {} tokens, menos que o bloco de {BLOCO}", ref_tokens.len()).into());
    }
    let tokens: Vec<usize> = ref_tokens[..BLOCO].iter().map(|&t| t as usize).collect();
    let mut rel = format!("== célula {nome}: {} camadas, {} experts, {} slots ({}%)\n", c.cfg.layers, c.cfg.experts, c.slots_10(), FRACAO_SLOTS * 100.0);

    // 1. A referência X: a passada residente.
    let x = {
        let mut g = GpuModel::upload(c.host, c.device.new_stream()?, c.max_len)?;
        let mut ultimo = Vec::new();
        for &t in &tokens {
            ultimo = g.step(t)?.0;
        }
        ultimo
    };

    // 2. A passada de portão (a).
    let pa = portao_a(c, &tokens);
    rel += &format!(
        "portão (a): {} (logits {:.2e}, KV {:.2e}; hits M8/M1 {}/{}, cpu {}/{}, needed {}/{}) {}\n",
        if pa.ok { "OK" } else { "FALHOU" },
        pa.logits_max,
        pa.kv_max,
        pa.hits.0,
        pa.hits.1,
        pa.cpu.0,
        pa.cpu.1,
        pa.needed.0,
        pa.needed.1,
        pa.motivo
    );
    if !pa.ok {
        rel += "rótulo: INVALIDA (portão de correção; o tempo não é lido)\n";
        std::fs::write(saida.join(format!("{nome}-relatorio.txt")), &rel)?;
        return Ok(SaidaCelula { rotulo: Rotulo::Invalida, relatorio: rel });
    }

    // A1m: só a thread principal vai para o E-core; vale para as três fases cronometradas.
    let mascara_antes = moe_rt::affinity::get_current()?;
    if let Some(cpu) = afin.main_cpu() {
        moe_rt::affinity::set_current(&[cpu])?;
    }
    let mut portoes_ok = true;

    // 3. Controle C antes.
    let c_antes = controle(c, &tokens, &x, &mut portoes_ok);

    // 4. Os pares.
    let slots = c.slots_10();
    let mut linhas: Vec<LinhaPar> = Vec::new();
    let mut parado = false;
    match c.sobe(slots, BACKLOG, vec![], POSICOES_MAX) {
        Err(e) => {
            rel += &format!("motor dos pares não subiu: {e}\n");
            parado = true;
        }
        Ok(mut mt) => {
            for par in 0..PARES {
                let aquecimento = par < AQUECIMENTO;
                let m1_primeiro = aquecimento || m1_primeiro(par - AQUECIMENTO);
                let (mut tp1, mut tp8, mut e1, mut e8) = (None, None, 0.0, 0.0);
                for &com_m1 in if m1_primeiro { &[true, false] } else { &[false, true] } {
                    match mt.repeticao(&tokens, if com_m1 { 1 } else { POSICOES_MAX }) {
                        Ok((secs, lg)) => {
                            let tp = secs / BLOCO as f64;
                            let erro = erro_l2_rel(&lg, &x);
                            if !portao_repeticao(&lg, &x) {
                                portoes_ok = false;
                            }
                            if com_m1 {
                                (tp1, e1) = (Some(tp), erro);
                            } else {
                                (tp8, e8) = (Some(tp), erro);
                            }
                        }
                        Err(e) => {
                            rel += &format!("par {par}: erro do motor ({e}); os pares seguintes ficam perdidos\n");
                            parado = true;
                            break;
                        }
                    }
                }
                linhas.push(LinhaPar { par, aquecimento, m1_primeiro, tp1, tp8, erro1: e1, erro8: e8 });
                if parado {
                    break;
                }
            }
            if !parado {
                if let Err(e) = mt.eng.close().d() {
                    rel += &format!("fechamento do motor dos pares: {e}\n");
                }
            }
        }
    }

    // 5. Controle C depois (o motor dos pares já foi liberado: dois motores com pesos densos não cabem juntos).
    let c_depois = if parado { vec![None; C_REPETICOES_LADO] } else { controle(c, &tokens, &x, &mut portoes_ok) };
    moe_rt::affinity::set_current(&mascara_antes)?;

    // 7. A regra.
    let usados: Vec<&LinhaPar> = linhas.iter().filter(|l| !l.aquecimento).collect();
    let ganhos: Vec<f64> = usados.iter().filter_map(|l| ganho_do_par(l.tp1?, l.tp8?)).collect();
    let tp1s: Vec<f64> = usados.iter().filter_map(|l| l.tp1.filter(|_| l.tp8.is_some())).collect();
    let cel = Celula { portoes_ok, ganhos: ganhos.clone(), tp1_passo_mediana: mediana(&tp1s), c: ControleC { antes: c_antes.clone(), depois: c_depois.clone() } };
    let rotulo = decide(&cel);

    let mut csv = String::from("par,aquecimento,m1_primeiro,tp1_ms,tp8_ms,ganho,erro_l2_m1,erro_l2_m8\n");
    for l in &linhas {
        let g = l.tp1.zip(l.tp8).and_then(|(a, b)| ganho_do_par(a, b));
        let f = |v: Option<f64>| v.map(|x| format!("{:.6}", x * 1e3)).unwrap_or_else(|| "perdido".into());
        csv += &format!("{},{},{},{},{},{},{:.3e},{:.3e}\n", l.par + 1, l.aquecimento, l.m1_primeiro, f(l.tp1), f(l.tp8), g.map(|v| format!("{v:.4}")).unwrap_or_else(|| "perdido".into()), l.erro1, l.erro8);
    }
    std::fs::write(saida.join(format!("{nome}-pares.csv")), csv)?;
    let linha_c = |v: &[Option<f64>]| v.iter().map(|x| x.map(|t| format!("{:.6}", t * 1e3)).unwrap_or_else(|| "perdido".into())).collect::<Vec<_>>().join(" ");
    rel += &format!("controle C (ms por posição): antes [{}] depois [{}]; mediana {}\n", linha_c(&c_antes), linha_c(&c_depois), cel.c.mediana().map(|m| format!("{:.6}", m * 1e3)).unwrap_or_else(|| "incompleto".into()));
    let r = resumo(&ganhos);
    rel += &format!(
        "pares usados n = {} (de {PARES_USADOS}); ganhos: mín {:.3} q1 {:.3} mediana {:.3} q3 {:.3} máx {:.3}; positivos {} (mínimo para PASSA {}); portões das repetições {}\n",
        ganhos.len(),
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
        ganhos.iter().filter(|&&g| g > 0.0).count(),
        if ganhos.len() >= N_MINIMO { minimo_positivos(ganhos.len()).to_string() } else { "n/a".into() },
        if portoes_ok { "OK" } else { "FALHOU" }
    );
    rel += &format!("tp(1) do passo (mediana) {:.6} ms; g_M: não medido (carimbos não implementados)\n", cel.tp1_passo_mediana * 1e3);
    rel += &format!("rótulo: {}\n", rotulo.nome());
    std::fs::write(saida.join(format!("{nome}-relatorio.txt")), &rel)?;
    Ok(SaidaCelula { rotulo, relatorio: rel })
}

fn sha256(p: &Path) -> Result<String, String> {
    let o = std::process::Command::new("sha256sum").arg(p).output().map_err(|e| format!("sha256sum: {e}"))?;
    if !o.status.success() {
        return Err(format!("sha256sum falhou em {}", p.display()));
    }
    Ok(String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1).collect())?;
    let exe = std::env::current_exe()?;
    println!("E4b: células {:?}; {PARES} pares ({AQUECIMENTO} de aquecimento), bloco de {BLOCO}, C {C_REPETICOES_LADO}+{C_REPETICOES_LADO}, X {X_PCT}%, slots {}%", args.celulas.iter().map(|c| &c.0).collect::<Vec<_>>(), FRACAO_SLOTS * 100.0);
    if args.ensaio {
        println!("ensaio: nada foi aberto na GPU. critério {}; binário {}", args.criterio.display(), exe.display());
        return Ok(());
    }
    exige_autorizacao(std::env::var(VARIAVEL).ok().as_deref())?;
    std::fs::create_dir_all(&args.saida)?;
    let (h_bin0, h_crit0) = (sha256(&exe)?, sha256(&args.criterio)?);
    println!("integridade antes: binário {h_bin0}; critério {h_crit0}");
    moe_rt::affinity::set_compute_plan(args.afin.compute_plan(args.threads));
    let device = Device::new(0)?;
    let mut rotulos: Vec<(String, Rotulo)> = Vec::new();
    let mut relatorios = String::new();
    for (nome, mdir, rdir) in &args.celulas {
        let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json"))?)?)?;
        let host = HostModel::load(&Checkpoint::open(mdir)?, cfg.clone())?;
        let c = Cenario { device: &device, host: &host, cfg: &cfg, max_len: (BLOCO + 1).next_multiple_of(ATTN_BN), threads: args.threads };
        let s = celula(&c, nome, rdir, &args.saida, args.afin)?;
        println!("{}", s.relatorio);
        relatorios += &s.relatorio;
        rotulos.push((nome.clone(), s.rotulo));
    }
    let (h_bin1, h_crit1) = (sha256(&exe)?, sha256(&args.criterio)?);
    let integro = h_bin0 == h_bin1 && h_crit0 == h_crit1;
    println!("integridade depois: binário {h_bin1}; critério {h_crit1}: {}", if integro { "IGUAIS" } else { "DIFERENTES: todas as células são INVALIDA" });
    if !integro {
        for r in rotulos.iter_mut() {
            r.1 = Rotulo::Invalida;
        }
    }
    let mut dec = String::from("decisão do E4b (rótulos pela regra do critério):\n");
    for (n, r) in &rotulos {
        dec += &format!("  {n}: {}\n", r.nome());
    }
    if rotulos.len() == 2 {
        let a2 = match resultado_a2(rotulos[0].1, rotulos[1].1) {
            ResultadoA2::AbreProposta => "1B e 3B PASSAM: abre-se a proposta de próximo passo ao Breno (não é autorização)",
            ResultadoA2::FechaA2 => "1B e 3B NAO_PASSAM: fecha o A2 (parada da proposta)",
            ResultadoA2::SemConclusao => "SEM CONCLUSÃO do A2: volta ao Breno; o exec não escolhe sozinho",
        };
        dec += &format!("resultado do A2: {a2}\n");
    }
    dec += &format!("integridade: binário {h_bin0} -> {h_bin1}; critério {h_crit0} -> {h_crit1}\n");
    println!("{dec}");
    std::fs::write(args.saida.join("decisao.txt"), dec + &relatorios)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn sem_a_variavel_o_portao_recusa_antes_do_device() {
        assert!(exige_autorizacao(None).is_err());
        assert!(exige_autorizacao(Some("0")).is_err());
        assert!(exige_autorizacao(Some("")).is_err());
        assert!(exige_autorizacao(Some("true")).is_err());
        assert!(exige_autorizacao(Some("1")).is_ok());
    }

    #[test]
    fn argumentos() {
        let a = parse_args(v(&["--celula", "1b:/m:/r", "--saida", "/o", "--ensaio"])).unwrap();
        assert_eq!(a.celulas.len(), 1);
        assert!(a.ensaio && a.threads == 8);
        assert!(parse_args(v(&["--saida", "/o"])).is_err(), "sem célula");
        assert!(parse_args(v(&["--celula", "1b:/m:/r"])).is_err(), "sem saída");
        assert!(parse_args(v(&["--celula", "1b:/m", "--saida", "/o"])).is_err(), "célula sem referência");
        assert!(parse_args(v(&["--celula", "a:/m:/r", "--celula", "b:/m:/r", "--celula", "c:/m:/r", "--saida", "/o"])).is_err(), "mais de duas");
        assert!(parse_args(v(&["--celula", "1b:/m:/r", "--saida", "/o", "--pares", "3"])).is_err(), "os parâmetros do critério não se trocam por opção");
    }

    #[test]
    fn tabela_congelada_tem_chaves_unicas_espacadas_e_dentro_do_espaco() {
        for (todas, s) in [(32 * 24, 76), (40 * 32, 128), (4, 1), (10, 10)] {
            let k = chaves_congeladas(todas, s);
            assert_eq!(k.len(), s);
            let mut u = k.clone();
            u.sort_unstable();
            u.dedup();
            assert_eq!(u.len(), s, "chaves repetidas");
            assert!(k.iter().all(|&x| (x as usize) < todas));
            assert!(k.windows(2).all(|w| w[0] < w[1]));
        }
    }
}
