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
//!  4. 15 pares (2 de aquecimento), a ordem inverte a cada par a partir do primeiro usado (M1 M8 | M8 M1 | ...; leitura A, ver
//!     `decisoes-e4b-medida.txt`): uma repetição de M = 1 e uma de M = 8 sobre o mesmo bloco,
//!     as duas pelo caminho do passo (`begin_step_tokens`, `enqueue_step`, `finish_token`; M = 1 com `m = 1`). Região cronometrada:
//!     do início do primeiro `begin_step_tokens` ao fim do último `finish_token`; os logits são lidos depois dela.
//!  5. Controle C depois: nova instância, 5 repetições. Roda também quando os pares pararam por erro (o motor dos pares é solto antes).
//!  6. Portão (b) em toda repetição, também nas que a carga alheia descartou: logits da última posição contra X, L2 relativa 1e-4 e argmax
//!     igual. Falhou: INVALIDA (a correção vence a perda por carga), e o relatório diz "tempo NÃO LIDO" (o CSV sai com `-NAO-LER` no nome).
//!     O mesmo vale se a integridade falhar: relatórios e CSV são montados só depois dela (`renderiza`). `--saida` tem de ser nova ou vazia.
//!  7. Carga alheia (critério, seção 3, "sem outra carga na GPU"): antes e depois de cada repetição consulta-se `nvidia-smi` pelos PIDs com
//!     contexto na GPU; diferente do conjunto-base da célula (sem o próprio PID), a repetição é perdida e o par não é refeito (seção 4).
//!     Falha na consulta falha fechada. Não vê carga de processo que já existia e passou a trabalhar, nem o que começou e terminou
//!     dentro de uma repetição: limite registrado no relatório.
//!  8. Rótulo pela regra de `moe_rt::e4b_regra` (a mesma testada em CPU); no fim o resultado do A2 se as células 1b e 3b rodaram.
//!     Os relatórios são escritos DEPOIS da checagem de integridade, com o rótulo final.
//!
//! Descritivo (não decide): o tempo da cabeça de saída (`CudaBackend::cronometra_cabeca`), medido na instância do C de depois, com o motor
//! já fechado e fora das observações; por posição, M = 1 (uma cabeça por posição) e M = 8 (uma por passo). Se não sair, o relatório diz PENDENTE e a causa.
//!
//! Não implementado: os carimbos de g_M (5 pares extras com evento de GPU no início e no fim de cada camada). Os eventos do backend são
//! criados sem cronometragem e instrumentar as camadas é mudança de backend. O critério aprovado prevê esse item; sair sem ele é uma
//! decisão do Breno, tomada em 06/11/2026 ("aceito sem g_M", registro em `decisoes-e4b-medida.txt`). O relatório diz "g_M: não medido".
//!
//! Pontos que o critério não fixa e que este programa decide (registrados no relatório e na revisão):
//!  - cada instância do controle C faz um bloco de partida SEM cronômetro antes das 5 repetições (a instância nasce fria e o critério
//!    pede as 10 observações válidas, sem aquecimento próprio);
//!  - um erro do motor numa repetição encerra os pares dali em diante (o motor não continua depois de um erro): os pares seguintes ficam
//!    perdidos e a regra decide com o n que sobrou;
//!  - a integridade (hash do binário e do critério antes e depois) falhando, ou não calculável, torna todas as células INVALIDA;
//!  - a afinidade `a1m` fixa só a thread principal DEPOIS de o motor subir (as threads novas herdariam a máscara), e a solta antes do próximo.
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
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const VARIAVEL: &str = "E4B_MEDIDA_AUTORIZADA";
/// Fração de slots (seção 2: 10% como no A1-DIAG10).
const FRACAO_SLOTS: f64 = 0.10;
const POSICOES_MAX: usize = 8;
const BACKLOG: usize = 4;
/// Chamadas da cabeça no relatório descritivo: aquecimento e medidas.
const CABECA_AQUECIMENTO: usize = 10;
const CABECA_N: usize = 200;

struct Args {
    celulas: Vec<(String, PathBuf, PathBuf)>,
    saida: PathBuf,
    criterio: PathBuf,
    threads: usize,
    afin: moe_rt::affinity::Afin,
    ensaio: bool,
}

/// O nome da célula vira parte do nome de arquivo: só letras, números, `-` e `_`.
fn nome_valido(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn canonico(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
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
                if !nome_valido(p[0]) {
                    return Err(format!("nome de célula inválido: {} (só letras, números, - e _)", p[0]));
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
        return Err("no máximo duas células (1b e 3b)".into());
    }
    let nomes: BTreeSet<&String> = r.celulas.iter().map(|c| &c.0).collect();
    if nomes.len() != r.celulas.len() {
        return Err("nomes de célula repetidos".into());
    }
    let dirs: BTreeSet<PathBuf> = r.celulas.iter().map(|c| canonico(&c.1)).collect();
    if dirs.len() != r.celulas.len() {
        return Err("o mesmo diretório de modelo em duas células".into());
    }
    if r.saida.as_os_str().is_empty() {
        return Err("--saida é obrigatória".into());
    }
    Ok(r)
}

/// A pasta de saída tem de ser nova ou vazia: resultados de uma execução anterior (um `decisao.txt` com PASSA) não podem sobreviver a esta.
fn prepara_saida(p: &Path) -> Result<(), String> {
    if p.exists() && std::fs::read_dir(p).map_err(|e| format!("--saida {}: {e}", p.display()))?.next().is_some() {
        return Err(format!("--saida {} já existe e não está vazia", p.display()));
    }
    std::fs::create_dir_all(p).map_err(|e| format!("--saida {}: {e}", p.display()))
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

/// O resultado conjunto do A2 só existe para as células `1b` e `3b` (o critério decide sobre os dois modelos).
fn a2_conjunto(rotulos: &[(String, Rotulo)]) -> Option<ResultadoA2> {
    let get = |n: &str| rotulos.iter().find(|r| r.0 == n).map(|r| r.1);
    Some(resultado_a2(get("1b")?, get("3b")?))
}

trait D<T> {
    fn d(self) -> Result<T, String>;
}

impl<T, E: std::fmt::Debug> D<T> for Result<T, E> {
    fn d(self) -> Result<T, String> {
        self.map_err(|e| format!("{e:?}"))
    }
}

/// Fixa a thread principal no E-core da afinidade `a1m` e devolve a máscara antiga ao soltar (também em erro). As threads de um motor
/// novo herdariam a máscara: por isso só se fixa DEPOIS de o motor subir e se solta antes do próximo.
struct Fixa(Option<Vec<usize>>);

impl Fixa {
    fn nova(afin: moe_rt::affinity::Afin) -> Result<Self, String> {
        match afin.main_cpu() {
            Some(c) => {
                let antes = moe_rt::affinity::get_current().d()?;
                moe_rt::affinity::set_current(&[c]).d()?;
                Ok(Fixa(Some(antes)))
            }
            None => Ok(Fixa(None)),
        }
    }
}

impl Drop for Fixa {
    fn drop(&mut self) {
        if let Some(a) = self.0.take() {
            let _ = moe_rt::affinity::set_current(&a);
        }
    }
}

fn nvidia_smi(args: &[&str]) -> Option<String> {
    let o = std::process::Command::new("nvidia-smi").args(args).output().ok()?;
    o.status.success().then(|| String::from_utf8_lossy(&o.stdout).into_owned())
}

/// Os PIDs alheios com contexto na GPU (sem o próprio). `None` se a consulta falhou: quem chama falha fechado.
fn pids_alheios() -> Option<BTreeSet<u32>> {
    let mut s = pids_de_computacao(&nvidia_smi(&["--query-compute-apps=pid", "--format=csv,noheader"])?)?;
    s.remove(&std::process::id());
    Some(s)
}

/// O estado da GPU, em texto, para o relatório (link PCIe, memória, uso e processos de computação).
fn estado_gpu() -> String {
    let g = nvidia_smi(&["--query-gpu=pci.link.gen.current,pci.link.width.current,memory.used,utilization.gpu,clocks.sm,temperature.gpu", "--format=csv,noheader"]).unwrap_or_else(|| "nvidia-smi falhou".into());
    let p = nvidia_smi(&["--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]).unwrap_or_else(|| "nvidia-smi falhou".into());
    format!("GPU [{}]; processos [{}]", g.trim(), p.trim().replace('\n', " | "))
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

    /// A repetição com a checagem de carga alheia antes e depois. `Err` é erro do motor (ele não segue). Os logits voltam sempre: a validade
    /// numérica (portão (b)) é independente da validade do tempo (`carga_ok`), e a correção tem precedência (`classifica`).
    fn repeticao_monitorada(&mut self, tokens: &[usize], m: usize, base: &BTreeSet<u32>) -> Result<Rep, String> {
        let antes = pids_alheios();
        let (secs, logits) = self.repeticao(tokens, m)?;
        let depois = pids_alheios();
        let carga_ok = matches!((&antes, &depois), (Some(a), Some(d)) if !contaminada(base, a, d));
        Ok(Rep { secs, logits, carga_ok })
    }
}

/// Uma repetição concluída: o tempo, os logits da última posição e se a GPU ficou sem carga alheia durante ela.
struct Rep {
    secs: f64,
    logits: Vec<f32>,
    carga_ok: bool,
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
    go().unwrap_or_else(|e| PortaoA { motivo: format!("erro: {e}"), ..Default::default() })
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

/// O controle C de um lado: as 5 observações (`None` é a perdida), as ocorrências em texto e, se pedido, o tempo da cabeça de saída.
struct SaidaC {
    obs: Vec<Option<f64>>,
    ocorrencias: String,
    /// Descritivo: medido na instância do C de depois, com o motor já fechado e fora das observações (`Err` guarda a causa).
    cabeca: Option<Result<f64, String>>,
}

/// As repetições do controle C: 5 num motor por token novo, depois de um bloco de partida sem cronômetro. Cada perda e cada erro ficam em
/// `ocorrencias`, que vai ao relatório.
fn controle(c: &Cenario, tokens: &[usize], x: &[f32], base: &BTreeSet<u32>, portoes_ok: &mut bool, afin: moe_rt::affinity::Afin, lado: &str, com_cabeca: bool) -> SaidaC {
    let slots = c.slots_10();
    let mut r = SaidaC { obs: vec![None; C_REPETICOES_LADO], ocorrencias: String::new(), cabeca: None };
    let mut go = || -> Result<(), String> {
        let mut mt = c.sobe(slots, BACKLOG, vec![], 1)?;
        let fixa = Fixa::nova(afin)?;
        mt.repeticao(tokens, 0)?; // partida, sem cronômetro e sem contar
        for i in 0..C_REPETICOES_LADO {
            let rep = mt.repeticao_monitorada(tokens, 0, base)?;
            match classifica(rep.carga_ok, portao_repeticao(&rep.logits, x)) {
                Classe::Valida => r.obs[i] = Some(rep.secs / tokens.len() as f64),
                Classe::Perdida => r.ocorrencias += &format!("C {lado}, repetição {}: perdida por carga alheia na GPU (ou consulta ao nvidia-smi falhou)\n", i + 1),
                Classe::Invalida => {
                    *portoes_ok = false;
                    r.ocorrencias += &format!("C {lado}, repetição {}: logits fora do portão (b)\n", i + 1);
                }
            }
        }
        let Motor { b, compute, eng } = mt;
        let fechou = eng.close().d();
        drop(fixa);
        if com_cabeca {
            // A cabeça só depois do `close` (sem cópias de admissão em curso) e de ler os logits das observações.
            r.cabeca = Some(b.cronometra_cabeca(&compute, CABECA_AQUECIMENTO, CABECA_N).d());
        }
        fechou
    };
    if let Err(e) = go() {
        r.ocorrencias += &format!("C {lado}: erro do motor ou da fixação ({e})\n");
        if com_cabeca && r.cabeca.is_none() {
            r.cabeca = Some(Err(format!("não medida: o controle C {lado} não completou ({e})")));
        }
    }
    r
}

struct SaidaCelula {
    nome: String,
    rotulo: Rotulo,
    /// O relatório sem os tempos e sem a linha do rótulo: eles entram depois da checagem de integridade (`renderiza`).
    corpo: String,
    /// As linhas com tempos (controle C, ganhos, tp(1)) e o CSV dos pares: só são publicados se a integridade e o portão (b) passaram.
    tempos: Option<String>,
    csv: Option<String>,
    portoes_ok: bool,
}

fn invalida(nome: &str, corpo: String) -> SaidaCelula {
    SaidaCelula { nome: nome.to_string(), rotulo: Rotulo::Invalida, corpo, tempos: None, csv: None, portoes_ok: false }
}

/// O relatório final de uma célula e o CSV (nome e texto). Com a integridade falha ou o portão (b) falho o tempo NÃO É LIDO: as linhas de
/// tempo saem do relatório e o CSV vai marcado `-NAO-LER`. O rótulo e a nota de integridade entram por último.
fn renderiza(s: &SaidaCelula, integro: bool, nota: &str) -> (String, Option<(String, String)>) {
    let lido = integro && s.portoes_ok && s.tempos.is_some();
    let mut r = s.corpo.clone();
    match (&s.tempos, lido) {
        (Some(t), true) => r += t,
        (Some(_), false) => r += &format!("tempo NÃO LIDO: {} (critério, seção 3); o CSV tem -NAO-LER no nome\n", if integro { "o portão por repetição (b) falhou" } else { "a integridade falhou" }),
        (None, _) => {}
    }
    r += &format!("rótulo: {}\n{nota}\n", s.rotulo.nome());
    let csv = s.csv.as_ref().map(|c| (if lido { format!("{}-pares.csv", s.nome) } else { format!("{}-pares-NAO-LER.csv", s.nome) }, c.clone()));
    (r, csv)
}

fn celula(c: &Cenario, nome: &str, ref_dir: &Path, afin: moe_rt::affinity::Afin) -> Result<SaidaCelula, Box<dyn std::error::Error>> {
    let ref_tokens = Npy::read(&ref_dir.join("tokens.npy"))?.i32()?;
    if ref_tokens.len() < BLOCO {
        return Err(format!("a referência tem {} tokens, menos que o bloco de {BLOCO}", ref_tokens.len()).into());
    }
    let tokens: Vec<usize> = ref_tokens[..BLOCO].iter().map(|&t| t as usize).collect();
    let mut rel = format!("== célula {nome}: {} camadas, {} experts, {} slots ({}%)\n", c.cfg.layers, c.cfg.experts, c.slots_10(), FRACAO_SLOTS * 100.0);
    rel += &format!("estado da GPU antes: {}\n", estado_gpu());
    let Some(base) = pids_alheios() else {
        return Ok(invalida(nome, rel + "nvidia-smi não respondeu: sem como vigiar a carga alheia, a célula é INVALIDA (falha fechada)\n"));
    };
    rel += &format!("processos alheios na GPU (base da célula): {base:?}\n");

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
        return Ok(invalida(nome, rel + "tempo NÃO LIDO: o portão de correção (a) falhou (critério, seção 3)\n"));
    }
    let mut portoes_ok = true;

    // 3. Controle C antes.
    let c_antes = controle(c, &tokens, &x, &base, &mut portoes_ok, afin, "antes", false);
    rel += &c_antes.ocorrencias;

    // 4. Os pares.
    let slots = c.slots_10();
    let mut linhas: Vec<LinhaPar> = Vec::new();
    let mut parado = false;
    match c.sobe(slots, BACKLOG, vec![], POSICOES_MAX) {
        Err(e) => {
            rel += &format!("motor dos pares não subiu: {e}\n");
        }
        Ok(mut mt) => {
            let fixa = Fixa::nova(afin)?;
            for par in 0..PARES {
                let aquecimento = par < AQUECIMENTO;
                let primeiro_m1 = aquecimento || m1_primeiro(par - AQUECIMENTO);
                let (mut tp1, mut tp8, mut e1, mut e8) = (None, None, 0.0, 0.0);
                for &com_m1 in if primeiro_m1 { &[true, false] } else { &[false, true] } {
                    let braco = if com_m1 { "M=1" } else { "M=8" };
                    match mt.repeticao_monitorada(&tokens, if com_m1 { 1 } else { POSICOES_MAX }, &base) {
                        Ok(r) => {
                            let erro = erro_l2_rel(&r.logits, &x);
                            let tp = match classifica(r.carga_ok, portao_repeticao(&r.logits, &x)) {
                                Classe::Valida => Some(r.secs / BLOCO as f64),
                                Classe::Perdida => {
                                    rel += &format!("par {} ({braco}): repetição perdida por carga alheia na GPU (ou consulta ao nvidia-smi falhou); o par não é refeito\n", par + 1);
                                    None
                                }
                                Classe::Invalida => {
                                    portoes_ok = false;
                                    rel += &format!("par {} ({braco}): logits fora do portão (b): a célula é INVALIDA\n", par + 1);
                                    None
                                }
                            };
                            if com_m1 {
                                (tp1, e1) = (tp, erro);
                            } else {
                                (tp8, e8) = (tp, erro);
                            }
                        }
                        Err(e) => {
                            rel += &format!("par {}: erro do motor ({e}); os pares seguintes ficam perdidos\n", par + 1);
                            parado = true;
                            break;
                        }
                    }
                }
                linhas.push(LinhaPar { par, aquecimento, m1_primeiro: primeiro_m1, tp1, tp8, erro1: e1, erro8: e8 });
                if parado {
                    break;
                }
            }
            if !parado {
                if let Err(e) = mt.eng.close().d() {
                    rel += &format!("fechamento do motor dos pares: {e}\n");
                }
            }
            drop(fixa);
            // `mt` sai de escopo aqui: o motor dos pares é solto (também depois de um erro) antes do C de depois.
        }
    }

    // 5. Controle C depois (dois motores com pesos densos e 10% de slots não cabem juntos na VRAM).
    let c_depois = if linhas.is_empty() {
        SaidaC { obs: vec![None; C_REPETICOES_LADO], ocorrencias: "C depois: não rodou (o motor dos pares não subiu)\n".into(), cabeca: Some(Err("não medida: o motor dos pares não subiu".into())) }
    } else {
        controle(c, &tokens, &x, &base, &mut portoes_ok, afin, "depois", true)
    };
    rel += &c_depois.ocorrencias;
    rel += &format!("estado da GPU depois: {}\n", estado_gpu());

    // 7. A regra, sobre os pares medidos.
    let pares: Vec<ParMedido> = linhas.iter().map(|l| ParMedido { aquecimento: l.aquecimento, tp1: l.tp1, tp8: l.tp8 }).collect();
    let cel = celula_dos_pares(&pares, ControleC { antes: c_antes.obs.clone(), depois: c_depois.obs.clone() }, portoes_ok);
    let rotulo = decide(&cel);
    let ganhos = cel.ganhos.clone();

    let mut csv = String::from("par,aquecimento,m1_primeiro,tp1_ms,tp8_ms,ganho,erro_l2_m1,erro_l2_m8\n");
    for l in &linhas {
        let g = l.tp1.zip(l.tp8).and_then(|(a, b)| ganho_do_par(a, b));
        let f = |v: Option<f64>| v.map(|x| format!("{:.6}", x * 1e3)).unwrap_or_else(|| "perdido".into());
        csv += &format!("{},{},{},{},{},{},{:.3e},{:.3e}\n", l.par + 1, l.aquecimento, l.m1_primeiro, f(l.tp1), f(l.tp8), g.map(|v| format!("{v:.4}")).unwrap_or_else(|| "perdido".into()), l.erro1, l.erro8);
    }
    // O CSV e as linhas de tempo ficam na saída da célula: só `renderiza` decide se são publicados (integridade e portão (b)).
    let linha_c = |v: &[Option<f64>]| v.iter().map(|x| x.map(|t| format!("{:.6}", t * 1e3)).unwrap_or_else(|| "perdido".into())).collect::<Vec<_>>().join(" ");
    let mut tempos = format!("controle C (ms por posição): antes [{}] depois [{}]; mediana {}\n", linha_c(&c_antes.obs), linha_c(&c_depois.obs), cel.c.mediana().map(|m| format!("{:.6}", m * 1e3)).unwrap_or_else(|| "incompleto".into()));
    let r = resumo(&ganhos);
    tempos += &format!(
        "pares usados n = {} (de {PARES_USADOS}); ganhos: mín {:.3} q1 {:.3} mediana {:.3} q3 {:.3} máx {:.3}; positivos {} (mínimo para PASSA {})\n",
        ganhos.len(),
        r[0],
        r[1],
        r[2],
        r[3],
        r[4],
        ganhos.iter().filter(|&&g| g > 0.0).count(),
        if ganhos.len() >= N_MINIMO { minimo_positivos(ganhos.len()).to_string() } else { "n/a".into() }
    );
    tempos += &format!("tp(1) do passo (mediana) {:.6} ms\n", cel.tp1_passo_mediana * 1e3);
    match &c_depois.cabeca {
        Some(Ok(t)) => tempos += &format!("cabeça de saída (descritivo, medida no C depois com o motor fechado): {:.6} ms por chamada; por posição: M=1 {:.6} ms (uma por posição), M=8 {:.6} ms (uma por passo, dividida por 8)\n", t * 1e3, t * 1e3, t * 1e3 / POSICOES_MAX as f64),
        Some(Err(e)) => rel += &format!("cabeça de saída: PENDENTE ({e})\n"),
        None => rel += "cabeça de saída: PENDENTE (não chegou a ser pedida)\n",
    }
    rel += "g_M: não medido (decisão do Breno de 06/11/2026: fora desta janela; registro em decisoes-e4b-medida.txt)\n";
    rel += "limite do monitor de carga: só vê processos que apareceram ou saíram entre as consultas antes e depois de cada repetição; orientação: sem usar a máquina (ditado, vídeo) durante a janela\n";
    Ok(SaidaCelula { nome: nome.to_string(), rotulo, corpo: rel, tempos: Some(tempos), csv: Some(csv), portoes_ok })
}

fn sha256(p: &Path) -> Result<String, String> {
    let o = std::process::Command::new("sha256sum").arg(p).output().map_err(|e| format!("sha256sum: {e}"))?;
    if !o.status.success() {
        return Err(format!("sha256sum falhou em {}", p.display()));
    }
    Ok(String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string())
}

fn roda(device: &Arc<Device>, args: &Args, nome: &str, mdir: &Path, rdir: &Path) -> Result<SaidaCelula, Box<dyn std::error::Error>> {
    let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json"))?)?)?;
    let host = HostModel::load(&Checkpoint::open(mdir)?, cfg.clone())?;
    let c = Cenario { device, host: &host, cfg: &cfg, max_len: (BLOCO + 1).next_multiple_of(ATTN_BN), threads: args.threads };
    celula(&c, nome, rdir, args.afin)
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
    prepara_saida(&args.saida)?;
    std::fs::write(args.saida.join("decisao.txt"), "em execução: sem decisão\n")?;
    let (h_bin0, h_crit0) = (sha256(&exe)?, sha256(&args.criterio)?);
    println!("integridade antes: binário {h_bin0}; critério {h_crit0}");
    moe_rt::affinity::set_compute_plan(args.afin.compute_plan(args.threads));
    let device = Device::new(0)?;
    let mut saidas: Vec<SaidaCelula> = Vec::new();
    for (nome, mdir, rdir) in &args.celulas {
        // Um erro da célula (referência, memória, leitura) vira INVALIDA com o motivo; a outra célula ainda roda.
        let s = roda(&device, &args, nome, mdir, rdir).unwrap_or_else(|e| invalida(nome, format!("== célula {nome}: erro: {e}\n")));
        println!("{}", s.corpo);
        saidas.push(s);
    }
    let h_bin1 = sha256(&exe);
    let h_crit1 = sha256(&args.criterio);
    let integro = h_bin1.as_deref().ok() == Some(h_bin0.as_str()) && h_crit1.as_deref().ok() == Some(h_crit0.as_str());
    let nota = format!("integridade: binário {h_bin0} -> {}; critério {h_crit0} -> {}: {}", h_bin1.as_deref().unwrap_or("ERRO"), h_crit1.as_deref().unwrap_or("ERRO"), if integro { "IGUAIS" } else { "DIFERENTES OU NÃO CALCULÁVEIS: todas as células são INVALIDA" });
    println!("{nota}");
    if !integro {
        for s in saidas.iter_mut() {
            s.rotulo = Rotulo::Invalida;
        }
    }
    // Os relatórios e a decisão só são escritos agora, com o rótulo final de cada célula.
    let mut dec = String::from("decisão do E4b (rótulos pela regra do critério):\n");
    let rotulos: Vec<(String, Rotulo)> = saidas.iter().map(|s| (s.nome.clone(), s.rotulo)).collect();
    for s in &saidas {
        let (relatorio, csv) = renderiza(s, integro, &nota);
        std::fs::write(args.saida.join(format!("{}-relatorio.txt", s.nome)), relatorio)?;
        if let Some((nome_csv, texto)) = csv {
            std::fs::write(args.saida.join(nome_csv), texto)?;
        }
        dec += &format!("  {}: {}\n", s.nome, s.rotulo.nome());
    }
    match a2_conjunto(&rotulos) {
        Some(ResultadoA2::AbreProposta) => dec += "resultado do A2: 1B e 3B PASSAM: abre-se a proposta de próximo passo ao Breno (não é autorização)\n",
        Some(ResultadoA2::FechaA2) => dec += "resultado do A2: 1B e 3B NAO_PASSAM: fecha o A2 (parada da proposta)\n",
        Some(ResultadoA2::SemConclusao) => dec += "resultado do A2: SEM CONCLUSÃO do A2: volta ao Breno; o exec não escolhe sozinho\n",
        None => dec += "resultado do A2: não decidido (o resultado conjunto pede as células 1b e 3b)\n",
    }
    dec += &format!("{nota}\n");
    println!("{dec}");
    std::fs::write(args.saida.join("decisao.txt"), dec)?;
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
        assert!(parse_args(v(&["--celula", "a:/m:/r", "--celula", "b:/m2:/r", "--celula", "c:/m3:/r", "--saida", "/o"])).is_err(), "mais de duas");
        assert!(parse_args(v(&["--celula", "1b:/m:/r", "--saida", "/o", "--pares", "3"])).is_err(), "os parâmetros do critério não se trocam por opção");
    }

    #[test]
    fn celulas_repetidas_ou_com_nome_perigoso_sao_recusadas() {
        assert!(parse_args(v(&["--celula", "1b:/m:/r", "--celula", "1b:/m2:/r", "--saida", "/o"])).is_err(), "nome repetido");
        assert!(parse_args(v(&["--celula", "1b:/m:/r", "--celula", "3b:/m:/r", "--saida", "/o"])).is_err(), "mesmo modelo em duas células");
        // O alias de caminho: /tmp e /tmp/../tmp/ são o mesmo diretório.
        assert!(parse_args(v(&["--celula", "1b:/tmp:/r", "--celula", "3b:/tmp/../tmp:/r", "--saida", "/o"])).is_err(), "alias de caminho");
        assert!(parse_args(v(&["--celula", "../x:/m:/r", "--saida", "/o"])).is_err(), "nome com ..");
        assert!(parse_args(v(&["--celula", "a/b:/m:/r", "--saida", "/o"])).is_err(), "nome com barra");
        assert!(parse_args(v(&["--celula", "1b:/m:/r", "--celula", "3b:/m2:/r", "--saida", "/o"])).is_ok());
    }

    #[test]
    fn resultado_conjunto_so_com_1b_e_3b() {
        use Rotulo::*;
        let r = |a: &str, x, b: &str, y| vec![(a.to_string(), x), (b.to_string(), y)];
        assert_eq!(a2_conjunto(&r("1b", Passa, "3b", Passa)), Some(ResultadoA2::AbreProposta));
        assert_eq!(a2_conjunto(&r("3b", NaoPassa, "1b", NaoPassa)), Some(ResultadoA2::FechaA2));
        assert_eq!(a2_conjunto(&r("a", Passa, "b", Passa)), None, "dois nomes quaisquer não decidem o A2");
        assert_eq!(a2_conjunto(&[("1b".to_string(), Passa)]), None, "uma célula só não decide o A2");
        assert_eq!(a2_conjunto(&r("1b", Invalida, "3b", Passa)), Some(ResultadoA2::SemConclusao));
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

    #[test]
    fn a_fixa_da_afinidade_a0_nao_mexe_na_mascara() {
        let antes = moe_rt::affinity::get_current().unwrap();
        {
            let f = Fixa::nova(moe_rt::affinity::Afin::A0).unwrap();
            assert!(f.0.is_none());
        }
        assert_eq!(moe_rt::affinity::get_current().unwrap(), antes);
    }

    #[test]
    fn saida_nao_vazia_e_recusada() {
        let d = std::env::temp_dir().join(format!("e4b-medida-teste-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        assert!(prepara_saida(&d).is_ok(), "pasta nova");
        assert!(prepara_saida(&d).is_ok(), "pasta vazia");
        std::fs::write(d.join("decisao.txt"), "PASSA").unwrap();
        assert!(prepara_saida(&d).is_err(), "pasta com resultado antigo");
        std::fs::remove_dir_all(&d).unwrap();
    }

    fn saida_exemplo(rotulo: Rotulo, portoes_ok: bool) -> SaidaCelula {
        SaidaCelula { nome: "1b".into(), rotulo, corpo: "== célula 1b\n".into(), tempos: Some("pares usados n = 13; ganhos: mediana 0.200\ncabeça de saída (descritivo): 1.234000 ms por chamada\n".into()), csv: Some("par,tp1\n".into()), portoes_ok }
    }

    #[test]
    fn com_a_integridade_falha_o_tempo_nao_e_lido_e_nao_sobra_passa() {
        let ok = saida_exemplo(Rotulo::Passa, true);
        let (r, csv) = renderiza(&ok, true, "integridade: IGUAIS");
        assert!(r.contains("mediana 0.200") && r.contains("rótulo: PASSA"));
        assert!(r.contains("cabeça de saída (descritivo): 1.234000"), "{r}");
        assert_eq!(csv.unwrap().0, "1b-pares.csv");
        // Integridade falha: main troca o rótulo para INVALIDA; o relatório não pode ter o tempo nem PASSA, e o CSV leva NAO-LER.
        let mut falha = saida_exemplo(Rotulo::Passa, true);
        falha.rotulo = Rotulo::Invalida;
        let (r, csv) = renderiza(&falha, false, "integridade: DIFERENTES");
        assert!(!r.contains("mediana") && !r.contains("PASSA") && r.contains("tempo NÃO LIDO") && r.contains("INVALIDA"), "{r}");
        assert!(!r.contains("cabeça de saída") && !r.contains("1.234000"), "a cabeça escapou do tempo NÃO LIDO: {r}");
        assert_eq!(csv.unwrap().0, "1b-pares-NAO-LER.csv");
        // Portão (b) falho com integridade ok: mesma regra.
        let (r, csv) = renderiza(&saida_exemplo(Rotulo::Invalida, false), true, "n");
        assert!(!r.contains("mediana") && r.contains("portão por repetição (b)"));
        assert_eq!(csv.unwrap().0, "1b-pares-NAO-LER.csv");
        // Uma célula que nem chegou aos tempos (portão (a)) não tem CSV.
        let (r, csv) = renderiza(&invalida("3b", "== célula 3b\n".into()), true, "n");
        assert!(csv.is_none() && r.contains("INVALIDA") && !r.contains("tempo NÃO LIDO"));
    }
}
