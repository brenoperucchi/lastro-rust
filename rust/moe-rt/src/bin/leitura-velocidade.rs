//! A2 (textos longos), etapa 4(a): o programa de VELOCIDADE da leitura em blocos de até 8 posições (`le_ids`) nos modelos reais, pelo critério
//! `qual-runs/A2-LOTE-20261103/criterio-leitura-velocidade.txt` (congelado antes de qualquer dado real). NÃO RODA SEM AUTORIZAÇÃO.
//!
//! Só abre a GPU com `LEITURA_VELOCIDADE_AUTORIZADA=1` (variável própria: nenhuma guarda antiga abre este programa) e com `CUDA_VISIBLE_DEVICES`
//! igual ao UUID combinado da RTX 3080 Ti. A 4090, a 5090 e qualquer outra nunca. `--ensaio` imprime o plano e confere ids e hashes SEM abrir o device e sem a variável. `--consolida` só
//! lê resultados já gravados (sem GPU, sem a variável).
//!
//! Três modos de uso:
//!  - `--estado quente --celula 1b:<modelo>:<traços> [--celula 3b:...] --saida <dir nova>`: UM processo; por modelo (1b antes de 3b) e por
//!    cenário (C 500 ids, A 2.003 ids): 1 repetição de aquecimento descartada e 5 pares (ordem alternada), motor e backend NOVOS por repetição,
//!    uma chamada de `le_ids`, sem leitura de KV. Indicador quente = T_leitura (`le_ids`) + T_logits (`read_logits`, stream sincronizada);
//!    T_sequencia (criação de motor e backend) fica à parte. Cada célula grava `quente-<modelo>-<cenário>.txt` (bruto) e `...-parcial.txt`
//!    (com o rótulo final até ali); no fim consolida (decisao.txt e relatórios).
//!  - `--estado frio --celula 1b:<modelo>:<traços> --cenario C|A --rep 1|2|3 --saida <dir>`: UM processo = UMA repetição de UMA célula, só modo 8,
//!    medida do início do `main` até os logits lidos. Grava `frio-<modelo>-<cenário>-rep<N>.txt` e NÃO decide rótulo.
//!  - `--consolida <dir> --estado quente|frio [--modelos 1b,3b]`: lê os brutos, aplica as regras de `moe_rt::velocidade_regra` e escreve
//!    `decisao.txt` e `relatorio-<estado>-<modelo>-<cenário>.txt`.
//! Nada é repetido nem alterado depois de ver um resultado. Códigos de saída: 0 gravado e válido, 1 gravado com falha (a célula fica SEM PROVA),
//! 2 recusa antes de qualquer trabalho (portão, UUID, hash, argumentos).
use cuda_core::Device;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream, LeitorCuda};
use moe_rt::leitura_regra::{fontes_de_ids, monta_ids, Contadores, BLOCO, MAX_LEN, N_A, N_C, TOL_ENTRE_MODOS};
use moe_rt::model::{Config as ModelConfig, HostModel};
use moe_rt::motor::leitura::{le_ids, PreparaPasso};
use moe_rt::motor::{AdmitMode, Config, Engine};
use moe_rt::npy::Npy;
use moe_rt::velocidade_regra as vr;
use moe_rt::velocidade_regra::{Avaliacao, Cen, Estado, ParQuente, RegistroFrio, RegistroQuente, RepQuente, Rotulo};
use moe_rt::weights::Checkpoint;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const VARIAVEL: &str = "LEITURA_VELOCIDADE_AUTORIZADA";
/// O UUID da RTX 3080 Ti combinado para a janela: `CUDA_VISIBLE_DEVICES` tem de ser exatamente isto.
const UUID_3080TI: &str = "GPU-b22d9465-8552-9f97-812e-380b590645f8";
/// O hash do critério aprovado e congelado: um `--criterio` com outro conteúdo recusa antes de abrir a GPU.
const CRITERIO_SHA: &str = "6e0fbaadc15d5f8f9ba2b7d6bb7e1e6956226720ea5dcbf311424b017dd1c859";
const FRACAO_SLOTS: f64 = 0.10;
const CRITERIO_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-leitura-velocidade.txt";
const CENARIOS: [Cen; 2] = [Cen::C, Cen::A];

struct Args {
    estado: Option<Estado>,
    celulas: Vec<(String, PathBuf, PathBuf)>,
    saida: PathBuf,
    criterio: PathBuf,
    threads: usize,
    ensaio: bool,
    cenario: Option<Cen>,
    rep: Option<usize>,
    consolida: Option<PathBuf>,
    modelos: Vec<String>,
}

fn nome_valido(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn canonico(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

fn parse_args(v: Vec<String>) -> Result<Args, String> {
    let mut a = v.into_iter();
    let mut r = Args { estado: None, celulas: vec![], saida: PathBuf::new(), criterio: PathBuf::from(CRITERIO_PADRAO), threads: 8, ensaio: false, cenario: None, rep: None, consolida: None, modelos: vec!["1b".into(), "3b".into()] };
    let mut modelos_dados = false;
    while let Some(k) = a.next() {
        let mut val = || a.next().ok_or_else(|| format!("{k} pede um valor"));
        match k.as_str() {
            "--celula" => {
                let x = val()?;
                let p: Vec<&str> = x.splitn(3, ':').collect();
                if p.len() != 3 || p.iter().any(|s| s.is_empty()) {
                    return Err(format!("--celula nome:modelo:tracos, não {x}"));
                }
                if !nome_valido(p[0]) || fontes_de_ids(p[0]).is_none() {
                    return Err(format!("célula {}: o nome tem de ser 1b ou 3b (as fontes de ids são fixas no critério)", p[0]));
                }
                r.celulas.push((p[0].to_string(), PathBuf::from(p[1]), PathBuf::from(p[2])));
            }
            "--estado" => r.estado = Some(Estado::de_nome(&val()?).ok_or("--estado quente ou frio")?),
            "--cenario" => r.cenario = Some(Cen::de_nome(&val()?).ok_or("--cenario C ou A")?),
            "--rep" => {
                let n: usize = val()?.parse().map_err(|_| "--rep".to_string())?;
                if !(1..=vr::REPS_FRIO).contains(&n) {
                    return Err(format!("--rep de 1 a {}", vr::REPS_FRIO));
                }
                r.rep = Some(n);
            }
            "--saida" => r.saida = PathBuf::from(val()?),
            "--criterio" => r.criterio = PathBuf::from(val()?),
            "--consolida" => r.consolida = Some(PathBuf::from(val()?)),
            "--modelos" => {
                r.modelos = val()?.split(',').map(|s| s.to_string()).collect();
                modelos_dados = true;
            }
            "--threads" => r.threads = val()?.parse().map_err(|_| "--threads".to_string())?,
            "--ensaio" => r.ensaio = true,
            _ => return Err(format!("opção desconhecida: {k}")),
        }
    }
    if r.modelos.is_empty() || r.modelos.iter().any(|m| fontes_de_ids(m).is_none()) || r.modelos.iter().collect::<BTreeSet<_>>().len() != r.modelos.len() {
        return Err("--modelos: lista de 1b e 3b, sem repetir".into());
    }
    let estado = r.estado.ok_or("--estado quente ou frio é obrigatório")?;
    if r.consolida.is_some() {
        if !r.celulas.is_empty() || !r.saida.as_os_str().is_empty() || r.cenario.is_some() || r.rep.is_some() || r.ensaio {
            return Err("--consolida só aceita --estado, --modelos e --criterio".into());
        }
        return Ok(r);
    }
    if modelos_dados {
        return Err("--modelos só vale com --consolida".into());
    }
    if r.celulas.is_empty() {
        return Err("pelo menos uma --celula".into());
    }
    let nomes: BTreeSet<&String> = r.celulas.iter().map(|c| &c.0).collect();
    if nomes.len() != r.celulas.len() {
        return Err("nomes de célula repetidos".into());
    }
    let dirs: BTreeSet<PathBuf> = r.celulas.iter().map(|c| canonico(&c.1)).collect();
    if dirs.len() != r.celulas.len() {
        return Err("o mesmo diretório de modelo em duas células".into());
    }
    match estado {
        Estado::Quente => {
            if r.celulas.len() > 2 {
                return Err("no máximo duas células (1b e 3b)".into());
            }
            if r.celulas.len() == 2 && r.celulas[0].0 != "1b" {
                return Err("a ordem é 1b antes de 3b".into());
            }
            if r.cenario.is_some() || r.rep.is_some() {
                return Err("--cenario e --rep são do estado frio".into());
            }
        }
        Estado::Frio => {
            if r.celulas.len() != 1 {
                return Err("o estado frio mede uma célula por processo (uma --celula)".into());
            }
            if r.cenario.is_none() || r.rep.is_none() {
                return Err("o estado frio pede --cenario e --rep".into());
            }
        }
    }
    if r.saida.as_os_str().is_empty() && !r.ensaio {
        return Err("--saida é obrigatória".into());
    }
    Ok(r)
}

/// Quente: a pasta de saída tem de ser nova ou vazia (um resultado antigo não pode sobreviver a esta execução).
fn prepara_saida(p: &Path) -> Result<(), String> {
    if p.exists() && std::fs::read_dir(p).map_err(|e| format!("--saida {}: {e}", p.display()))?.next().is_some() {
        return Err(format!("--saida {} já existe e não está vazia", p.display()));
    }
    std::fs::create_dir_all(p).map_err(|e| format!("--saida {}: {e}", p.display()))
}

/// Frio: a pasta é compartilhada pelos 12 processos. Só aceita o que o próprio estado frio produz (`frio-*`, logs, relatórios e decisão) e
/// recusa um arquivo de resultado que já exista (nada se sobrescreve nem se repete).
fn prepara_saida_frio(dir: &Path, arquivo: &str) -> Result<PathBuf, String> {
    if dir.exists() {
        for e in std::fs::read_dir(dir).map_err(|e| format!("--saida {}: {e}", dir.display()))? {
            let n = e.map_err(|e| e.to_string())?.file_name().to_string_lossy().to_string();
            if !(n.starts_with("frio-") || n.starts_with("relatorio-frio-") || n.ends_with(".log") || n == "decisao.txt") {
                return Err(format!("--saida {} tem o arquivo estranho {n}: o estado frio não mistura com outro resultado", dir.display()));
            }
        }
    } else {
        std::fs::create_dir_all(dir).map_err(|e| format!("--saida {}: {e}", dir.display()))?;
    }
    let f = dir.join(arquivo);
    if f.exists() {
        return Err(format!("{} já existe: nada se repete nem se sobrescreve", f.display()));
    }
    Ok(f)
}

/// O portão de autorização: só `1` abre. Recusa antes de qualquer acesso ao device.
fn exige_autorizacao(valor: Option<&str>) -> Result<(), String> {
    match valor {
        Some("1") => Ok(()),
        _ => Err(format!("a medida de velocidade na GPU só roda com o 'autorizo' explícito do Breno ({VARIAVEL}=1); veja o critério em criterio-leitura-velocidade.txt")),
    }
}

/// O UUID da GPU: `CUDA_VISIBLE_DEVICES` tem de ser exatamente o da 3080 Ti combinada.
fn exige_uuid(valor: Option<&str>) -> Result<(), String> {
    match valor {
        Some(UUID_3080TI) => Ok(()),
        outro => Err(format!("CUDA_VISIBLE_DEVICES tem de ser {UUID_3080TI} (a 3080 Ti combinada); veio {outro:?}")),
    }
}

/// As chaves da tabela congelada: `slots` chaves espaçadas por igual no espaço camada·E + expert (divisão mista fixa, a da etapa 3).
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

fn sha256(p: &Path) -> Result<String, String> {
    let o = std::process::Command::new("sha256sum").arg(p).output().map_err(|e| format!("sha256sum: {e}"))?;
    if !o.status.success() {
        return Err(format!("sha256sum falhou em {}", p.display()));
    }
    Ok(String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string())
}

struct Ids {
    c: Vec<usize>,
    a: Vec<usize>,
}

impl Ids {
    fn de(&self, cen: Cen) -> &[usize] {
        match cen {
            Cen::C => &self.c,
            Cen::A => &self.a,
        }
    }
}

/// Os `tokens.npy` que a célula usa (fixos pelo critério). Os hashes saem DESTA lista antes de qualquer leitura de ids ou uso da GPU.
fn arquivos_de_ids(nome: &str, tracos: &Path) -> Result<Vec<PathBuf>, String> {
    let (conv_c, conv_a) = fontes_de_ids(nome).ok_or_else(|| format!("sem fonte de ids para {nome}"))?;
    let mut v: Vec<PathBuf> = std::iter::once(conv_c).chain(conv_a.iter().copied()).map(|n| tracos.join(format!("conversa-{n}")).join("tokens.npy")).collect();
    v.sort();
    v.dedup();
    Ok(v)
}

fn hashes_de(arqs: &[PathBuf]) -> Result<Vec<(PathBuf, String)>, String> {
    arqs.iter().map(|a| Ok((a.clone(), sha256(a)?))).collect()
}

fn ids_iguais(ids0: &[(PathBuf, String)]) -> bool {
    ids0.iter().all(|(p, h)| sha256(p).ok().as_deref() == Some(h.as_str()))
}

/// Integridade num instante: binário, critério (contra o inicial) e `tokens.npy` (contra os iniciais, tirados ANTES de usar os ids).
fn integro_agora(exe: &Path, criterio: &Path, h_bin0: &str, h_crit0: &str, ids0: &[(PathBuf, String)]) -> bool {
    sha256(exe).ok().as_deref() == Some(h_bin0) && sha256(criterio).ok().as_deref() == Some(h_crit0) && ids_iguais(ids0)
}

fn carrega_ids(nome: &str, tracos: &Path, vocab: usize) -> Result<Ids, String> {
    let (conv_c, conv_a) = fontes_de_ids(nome).ok_or_else(|| format!("sem fonte de ids para {nome}"))?;
    let arq = |n: usize| tracos.join(format!("conversa-{n}")).join("tokens.npy");
    let le = |n: usize| -> Result<Vec<i32>, String> { Npy::read(&arq(n)).and_then(|x| x.i32()).map_err(|e| format!("{}: {e:?}", arq(n).display())) };
    let c = monta_ids(&[le(conv_c)?], N_C, vocab).map_err(|e| format!("cenário C: {e}"))?;
    let a = monta_ids(&[le(conv_a[0])?, le(conv_a[1])?], N_A, vocab).map_err(|e| format!("cenário A: {e}"))?;
    Ok(Ids { c, a })
}

struct Cenario<'a> {
    device: &'a Arc<Device>,
    host: &'a HostModel,
    cfg: &'a ModelConfig,
    threads: usize,
}

impl Cenario<'_> {
    fn slots_10(&self) -> usize {
        (FRACAO_SLOTS * (self.cfg.layers * self.cfg.experts) as f64) as usize
    }
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Uma repetição medida: os tempos (ms), o instante em que os logits ficaram lidos, os logits e os contadores.
struct Rep {
    tempos: RepQuente,
    t_fim: Instant,
    logits: Vec<f32>,
    cont: Contadores,
}

/// Uma sequência medida: motor e backend NOVOS (tabela congelada, `backlog = 0`, admissão Event), UMA chamada de `le_ids` sobre todos os `ids`
/// (sem leitura de KV, sem cortes) e `read_logits`. Três tempos separados: criação de motor e backend, `le_ids`, `read_logits` (a stream é
/// sincronizada dentro dele). As conferências de posição e a leitura dos contadores vêm DEPOIS do cronômetro.
fn le_cronometrado(c: &Cenario, ids: &[usize], max_positions: usize) -> Result<Rep, String> {
    let t_seq = Instant::now();
    let slots = c.slots_10();
    let (compute, copy) = (CudaStream(c.device.new_stream().d()?), CudaStream(c.device.new_stream().d()?));
    let (b, ex) = CudaBackend::new_model_positions(c.device.clone(), c.host, slots, MAX_LEN, &compute, max_positions).d()?;
    let m = c.cfg;
    let prefill = chaves_congeladas(m.layers * m.experts, slots);
    let mcfg = Config { layers: m.layers, experts: m.experts, topk: m.k, width: m.hidden, slots, backlog: 0, admit: AdmitMode::Event, first_token: 1, prefill, max_positions };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).d()?, mcfg.first_token).d()?;
    let cpu = CpuServe { cpu: Cpu::new(c.threads), hidden: m.hidden, inter: m.inter, experts: m.experts, weights: ex.clone() };
    let mut eng = Engine::new(b.clone(), parts, mcfg, cpu).d()?;
    // A stream de cálculo do adaptador é a MESMA do motor (o clone passado a `parts`).
    let leitor = LeitorCuda::novo(b.clone(), compute.clone()).d()?;
    let sequencia_ms = ms(t_seq.elapsed());
    let t_leit = Instant::now();
    let lido = le_ids(&mut eng, &leitor, ids).map_err(|e| format!("le_ids: {e:?}"))?;
    let leitura_ms = ms(t_leit.elapsed());
    let t_log = Instant::now();
    let logits = b.read_logits(&compute).d()?;
    let t_fim = Instant::now();
    let logits_ms = ms(t_fim.duration_since(t_log));
    if lido != ids.len() {
        return Err(format!("le_ids leu {lido} em vez de {}", ids.len()));
    }
    if eng.position() != ids.len() as u64 {
        return Err(format!("posição do motor {} depois de ler {} ids", eng.position(), ids.len()));
    }
    if leitor.posicao() != ids.len() as u64 {
        return Err(format!("posição do backend {} depois de ler {} ids", leitor.posicao(), ids.len()));
    }
    let s = eng.next_published().d()?.stats().clone();
    eng.close().d()?;
    Ok(Rep { tempos: RepQuente { sequencia_ms, leitura_ms, logits_ms }, t_fim, logits, cont: Contadores { needed: s.needed, hits: s.hits, cpu: s.cpu } })
}

fn txt_rep(modo: usize, r: &Rep) -> String {
    format!("modo {modo}: sequência {:.1} ms, leitura {:.1} ms, logits {:.1} ms, indicador {:.1} ms, contadores {:?}", r.tempos.sequencia_ms, r.tempos.leitura_ms, r.tempos.logits_ms, r.tempos.indicador_ms(), r.cont)
}

/// A célula quente (modelo x cenário): aquecimento descartado e `PARES_QUENTE` pares em ordem alternada. `le(max_positions)` faz UMA repetição
/// (motor e backend novos). Para no primeiro erro (célula SEM PROVA) e no primeiro par que falha o portão de correção (alerta). Nada repete.
/// `aquece_modo1`: além do aquecimento do modo 8 (o caminho de uso), uma repetição do modo 1 descartada, para que o primeiro uso do modo 1
/// não caia dentro de um par (decisão de projeto; o critério fixa o motivo).
fn mede_quente<F: FnMut(usize) -> Result<Rep, String>>(modelo: &str, cen: Cen, aquece_modo1: bool, mut le: F) -> RegistroQuente {
    let mut rec = RegistroQuente { modelo: modelo.into(), cen, integro: true, falha: None, pares: vec![], aquecimento: vec![], extras: vec![] };
    for modo in std::iter::once(BLOCO).chain(aquece_modo1.then_some(1)) {
        match le(modo) {
            Ok(r) => {
                let t = format!("aquecimento descartado, {}", txt_rep(modo, &r));
                println!("{modelo} {} {t}", cen.nome());
                rec.aquecimento.push(t);
            }
            Err(e) => {
                rec.falha = Some(format!("aquecimento (modo {modo}): {e}"));
                return rec;
            }
        }
    }
    for indice in 1..=vr::PARES_QUENTE {
        let m1_primeiro = vr::modo1_primeiro(indice);
        let ordem = if m1_primeiro { [1, BLOCO] } else { [BLOCO, 1] };
        let mut feitas: BTreeMap<usize, Rep> = BTreeMap::new();
        for modo in ordem {
            match le(modo) {
                Ok(r) => {
                    println!("{modelo} {} par {indice}/{} {}", cen.nome(), vr::PARES_QUENTE, txt_rep(modo, &r));
                    feitas.insert(modo, r);
                }
                Err(e) => {
                    rec.falha = Some(format!("par {indice}, modo {modo}: {e}"));
                    return rec;
                }
            }
        }
        let (r1, r8) = (&feitas[&1], &feitas[&BLOCO]);
        let portao = vr::portao_par(&r8.logits, &r1.logits, r8.cont, r1.cont);
        if indice == 1 {
            rec.extras.push(("cont_m1".into(), format!("{:?}", r1.cont)));
            rec.extras.push(("cont_m8".into(), format!("{:?}", r8.cont)));
        }
        rec.pares.push(ParQuente { indice, m1_primeiro, m1: r1.tempos, m8: r8.tempos, portao });
        if !portao.ok() {
            println!("{modelo} {} par {indice}: PORTÃO DE CORREÇÃO FALHOU (erro {:.3e}, não finito {}, argmax igual {}, contadores ok {}): célula SEM PROVA, sem repetir", cen.nome(), portao.erro, portao.nao_finito, portao.argmax_ok, portao.cont_ok);
            break;
        }
    }
    rec
}

fn pct(x: Option<f64>) -> String {
    x.map(|v| format!("{:.1}%", v * 100.0)).unwrap_or_else(|| "n/d".into())
}

fn segs(x: Option<f64>) -> String {
    x.map(|v| format!("{:.1} ms ({:.2} s)", v, v / 1000.0)).unwrap_or_else(|| "n/d".into())
}

/// O texto de uma célula avaliada (relatório e parcial): o que decide e o que só descreve.
fn texto_avaliacao(estado: Estado, modelo: &str, cen: Cen, a: &Avaliacao, final_: Rotulo) -> String {
    let mut t = format!("== célula {modelo} {} {} ({} ids)\n", cen.nome(), estado.nome(), cen.n());
    t += &format!("meta: {}\n", segs(a.meta_ms));
    match estado {
        Estado::Quente => {
            t += &format!("mediana do modo 1: {}\nmediana do modo 8 (indicador = leitura + logits; DECIDE contra a meta): {}\nganho = 1 - med8/med1 (piso {:.0}%; DECIDE): {}\n", segs(a.med1_ms), segs(a.med8_ms), vr::GANHO_MINIMO * 100.0, pct(a.ganho));
            t += &format!("deriva do modo 1 (limite {:.0}%): {}\n", vr::DERIVA_MAX * 100.0, pct(a.deriva));
            t += &format!("só descrição, não decide: pior repetição do modo 8 {}; desvio entre repetições do modo 8 {}; T_sequencia mediana {}\n", segs(a.pior_ms), segs(a.desvio_ms), segs(a.med_sequencia_ms));
        }
        Estado::Frio => {
            t += &format!("mediana das repetições frias (DECIDE contra a meta): {}\nsó descrição, não decide: pior repetição {}; desvio {}\n", segs(a.med8_ms), segs(a.pior_ms), segs(a.desvio_ms));
        }
    }
    if a.rotulo != Rotulo::Atinge {
        t += &format!("motivo(s) ({}): {}\n", if a.rotulo == Rotulo::NaoAtinge { "nível 2" } else { "validade, nível 1" }, a.motivos.join("; "));
    }
    t += &format!("rótulo da célula: {}\n", final_.nome());
    t
}

fn texto_parcial(rec: &RegistroQuente, integro: bool) -> String {
    let a = vr::avalia_quente(&rec.modelo, rec.cen, &rec.pares, rec.falha.as_deref());
    let f = vr::rotulo_final(a.rotulo, integro);
    format!("{}integridade conferida no fim da célula: {}\n", texto_avaliacao(Estado::Quente, &rec.modelo, rec.cen, &a, f), if integro { "íntegra" } else { "QUEBRADA" })
}

// ---- Consolidação (sem GPU) ---------------------------------------------------------------------------------------------------------------

struct Contexto {
    exe: PathBuf,
    criterio: PathBuf,
    /// O hash esperado do critério (a constante `CRITERIO_SHA` na execução real; outro valor só nos testes).
    sha_criterio: String,
}

enum Lido<T> {
    Ausente,
    Ilegivel(String),
    Ok(T),
}

fn le_arquivo<T>(p: &Path, parse: impl Fn(&str) -> Result<T, String>) -> Lido<T> {
    if !p.exists() {
        return Lido::Ausente;
    }
    match std::fs::read_to_string(p).map_err(|e| e.to_string()).and_then(|t| parse(&t)) {
        Ok(v) => Lido::Ok(v),
        Err(e) => Lido::Ilegivel(format!("{}: {e}", p.display())),
    }
}

/// Confere as integridades do conjunto: todos os brutos concordam com o binário e o critério atuais e com os `tokens.npy` atuais.
fn integridade_do_conjunto(ctx: &Contexto, extras: &[&Vec<(String, String)>], flags: &[bool], notas: &mut Vec<String>) -> bool {
    let mut ok = true;
    if flags.iter().any(|f| !f) {
        ok = false;
        notas.push("algum registro diz que a integridade quebrou durante a execução".into());
    }
    match (sha256(&ctx.criterio), sha256(&ctx.exe)) {
        (Ok(hc), Ok(hb)) => {
            if hc != ctx.sha_criterio {
                ok = false;
                notas.push(format!("o critério agora tem sha256 {hc}, não o congelado {}", ctx.sha_criterio));
            }
            let mut ids: BTreeMap<String, String> = BTreeMap::new();
            for ex in extras {
                for (k, v) in ex.iter() {
                    if k == "hash_bin" && *v != hb {
                        ok = false;
                        notas.push(format!("o binário gravou {v}, o atual é {hb}"));
                    } else if k == "hash_criterio" && *v != ctx.sha_criterio {
                        ok = false;
                        notas.push(format!("o critério gravado foi {v}, não o congelado"));
                    } else if let Some(p) = k.strip_prefix("ids:") {
                        if ids.insert(p.to_string(), v.clone()).is_some_and(|antes| antes != *v) {
                            ok = false;
                            notas.push(format!("hash de ids diferente entre registros: {p}"));
                        }
                    }
                }
            }
            for (p, h) in &ids {
                if sha256(Path::new(p)).ok().as_deref() != Some(h.as_str()) {
                    ok = false;
                    notas.push(format!("ids {p}: o hash atual difere do gravado"));
                }
            }
            if ids.is_empty() {
                ok = false;
                notas.push("nenhum hash de ids gravado".into());
            }
        }
        _ => {
            ok = false;
            notas.push("não foi possível calcular o hash do binário ou do critério".into());
        }
    }
    ok
}

/// Lê os brutos de `dir`, aplica as regras e escreve `decisao.txt` e os relatórios. Devolve o texto da decisão e o rótulo do conjunto.
fn consolida(dir: &Path, estado: Estado, modelos: &[String], ctx: &Contexto) -> Result<(String, Rotulo), String> {
    let mut notas: Vec<String> = Vec::new();
    let mut avals: Vec<(String, Cen, Avaliacao)> = Vec::new();
    let mut extras: Vec<Vec<(String, String)>> = Vec::new();
    let mut flags: Vec<bool> = Vec::new();
    for m in modelos {
        for cen in CENARIOS {
            let a = match estado {
                Estado::Quente => match le_arquivo(&dir.join(format!("quente-{m}-{}.txt", cen.nome())), RegistroQuente::de_texto) {
                    Lido::Ok(r) => {
                        flags.push(r.integro);
                        extras.push(r.extras.clone());
                        vr::avalia_registro_quente(m, cen, &r)
                    }
                    Lido::Ausente => vr::avalia_quente(m, cen, &[], Some("célula ausente: interrompida (por exemplo tempo-limite) ou não rodada")),
                    Lido::Ilegivel(e) => vr::avalia_quente(m, cen, &[], Some(&format!("resultado ilegível: {e}"))),
                },
                Estado::Frio => {
                    let mut regs = Vec::new();
                    for n in 1..=vr::REPS_FRIO {
                        match le_arquivo(&dir.join(format!("frio-{m}-{}-rep{n}.txt", cen.nome())), RegistroFrio::de_texto) {
                            Lido::Ok(r) => {
                                flags.push(r.integro);
                                extras.push(r.extras.clone());
                                regs.push(r);
                            }
                            Lido::Ausente => notas.push(format!("frio {m} {} repetição {n}: ausente (interrompida ou não rodada)", cen.nome())),
                            Lido::Ilegivel(e) => notas.push(format!("frio {m} {} repetição {n}: resultado ilegível, conta como ausente: {e}", cen.nome())),
                        }
                    }
                    vr::avalia_frio(m, cen, &regs)
                }
            };
            avals.push((m.clone(), cen, a));
        }
    }
    if estado == Estado::Quente {
        match le_arquivo(&dir.join("quente-final.txt"), |t| Ok(t.lines().any(|l| l == "integro=1"))) {
            Lido::Ok(ok) => flags.push(ok),
            Lido::Ausente => notas.push("quente-final.txt ausente (execução interrompida): vale a integridade conferida célula a célula e a conferência agora".into()),
            Lido::Ilegivel(e) => {
                flags.push(false);
                notas.push(e);
            }
        }
    }
    let ex_ref: Vec<&Vec<(String, String)>> = extras.iter().collect();
    let integro = integridade_do_conjunto(ctx, &ex_ref, &flags, &mut notas);
    let mut dec = format!("decisão da velocidade da leitura em blocos de 8, estado {} (regra de rótulos do critério):\n", estado.nome());
    let mut rotulos = Vec::new();
    for (m, cen, a) in &avals {
        let r = vr::rotulo_final(a.rotulo, integro);
        rotulos.push(r);
        let rel = format!("{}integridade do conjunto: {}\n", texto_avaliacao(estado, m, *cen, a, r), if integro { "íntegra" } else { "QUEBRADA: todas as células SEM PROVA" });
        std::fs::write(dir.join(format!("relatorio-{}-{m}-{}.txt", estado.nome(), cen.nome())), &rel).map_err(|e| e.to_string())?;
        let resumo = match (estado, a.med8_ms) {
            (Estado::Quente, Some(m8)) => format!("mediana modo 8 {:.2} s (meta {:.0} s), ganho {}", m8 / 1000.0, a.meta_ms.unwrap_or(f64::NAN) / 1000.0, pct(a.ganho)),
            (Estado::Frio, Some(m8)) => format!("mediana {:.2} s (meta {:.0} s)", m8 / 1000.0, a.meta_ms.unwrap_or(f64::NAN) / 1000.0),
            _ => "sem tempo legível".into(),
        };
        let motivo = if r == Rotulo::Atinge { String::new() } else if integro { format!(" [{}]", a.motivos.join("; ")) } else { " [integridade quebrada]".into() };
        dec += &format!("  {m} {} {}: {}: {resumo}{motivo}\n", cen.nome(), estado.nome(), r.nome());
    }
    let conj = vr::rotulo_conjunto(&rotulos, integro);
    dec += &format!("conjunto: {}\n", conj.nome());
    dec += &format!("integridade: {}\n", if integro { "binário, critério e arquivos de ids IGUAIS aos gravados" } else { "QUEBRADA OU NÃO CALCULÁVEL: todas as células SEM PROVA" });
    for n in &notas {
        dec += &format!("nota: {n}\n");
    }
    dec += "carga externa (nvidia-smi, loadavg, timers): registrada pelo script da janela, fora deste programa; não decide rótulo\n";
    std::fs::write(dir.join("decisao.txt"), &dec).map_err(|e| e.to_string())?;
    Ok((dec, conj))
}

// ---- Estados -------------------------------------------------------------------------------------------------------------------------------

fn extras_de_integridade(h_bin0: &str, h_crit0: &str, ids0: &[(PathBuf, String)]) -> Vec<(String, String)> {
    let mut v = vec![("hash_bin".to_string(), h_bin0.to_string()), ("hash_criterio".to_string(), h_crit0.to_string())];
    v.extend(ids0.iter().map(|(p, h)| (format!("ids:{}", p.display()), h.clone())));
    v
}

fn fnv(v: &[f32]) -> u64 {
    v.iter().fold(0xcbf29ce484222325u64, |h, x| x.to_bits().to_le_bytes().iter().fold(h, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3)))
}

fn carrega_modelo(nome: &str, mdir: &Path, tracos: &Path) -> Result<(ModelConfig, Ids, HostModel), Box<dyn std::error::Error>> {
    let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json"))?)?)?;
    let ids = carrega_ids(nome, tracos, cfg.vocab)?;
    let host = HostModel::load(&Checkpoint::open(mdir)?, cfg.clone())?;
    Ok((cfg, ids, host))
}

fn quente(args: &Args, exe: &Path) -> Result<i32, String> {
    exige_autorizacao(std::env::var(VARIAVEL).ok().as_deref())?;
    exige_uuid(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
    // Todos os hashes de entrada saem ANTES de ler ids ou abrir a GPU.
    let h_bin0 = sha256(exe)?;
    let h_crit0 = sha256(&args.criterio)?;
    if h_crit0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_crit0}, não o congelado {CRITERIO_SHA}", args.criterio.display()));
    }
    let mut arquivos_ids: Vec<PathBuf> = Vec::new();
    for (nome, _, tracos) in &args.celulas {
        arquivos_ids.extend(arquivos_de_ids(nome, tracos)?);
    }
    arquivos_ids.sort();
    arquivos_ids.dedup();
    let ids0 = hashes_de(&arquivos_ids)?;
    prepara_saida(&args.saida)?;
    std::fs::write(args.saida.join("decisao.txt"), "em execução: sem decisão\n").map_err(|e| e.to_string())?;
    println!("integridade antes: binário {h_bin0}; critério {h_crit0}; {} arquivos de ids", ids0.len());
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let device = Device::new(0).map_err(|e| format!("{e:?}"))?;
    let extras = extras_de_integridade(&h_bin0, &h_crit0, &ids0);
    let mut integro = true;
    for (nome, mdir, tracos) in &args.celulas {
        let carregado = carrega_modelo(nome, mdir, tracos).map_err(|e| e.to_string());
        for (i, cen) in CENARIOS.into_iter().enumerate() {
            let mut rec = match &carregado {
                Ok((cfg, ids, host)) => {
                    let c = Cenario { device: &device, host, cfg, threads: args.threads };
                    // Aquecimento extra do modo 1 só na primeira célula de cada modelo (cenário C, o mais barato).
                    mede_quente(nome, cen, i == 0, |mp| le_cronometrado(&c, ids.de(cen), mp))
                }
                Err(e) => RegistroQuente { modelo: nome.clone(), cen, integro: true, falha: Some(format!("carga do modelo: {e}")), pares: vec![], aquecimento: vec![], extras: vec![] },
            };
            // Integridade conferida no fim de CADA célula: o parcial já sai com o rótulo final até aqui.
            integro &= integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &ids0);
            rec.integro = integro;
            rec.extras.extend(extras.iter().cloned());
            let txt = texto_parcial(&rec, integro);
            println!("{txt}");
            std::fs::write(args.saida.join(format!("quente-{nome}-{}.txt", cen.nome())), rec.para_texto()).map_err(|e| e.to_string())?;
            std::fs::write(args.saida.join(format!("quente-{nome}-{}-parcial.txt", cen.nome())), txt).map_err(|e| e.to_string())?;
        }
    }
    integro &= integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &ids0);
    std::fs::write(args.saida.join("quente-final.txt"), format!("tipo=quente-final\nintegro={}\n", integro as u8)).map_err(|e| e.to_string())?;
    let modelos: Vec<String> = args.celulas.iter().map(|c| c.0.clone()).collect();
    let ctx = Contexto { exe: exe.to_path_buf(), criterio: args.criterio.clone(), sha_criterio: CRITERIO_SHA.into() };
    let (dec, _) = consolida(&args.saida, Estado::Quente, &modelos, &ctx)?;
    println!("{dec}");
    Ok(0)
}

/// Uma repetição fria (um processo). `t0` é o primeiro instante do `main`.
fn frio(t0: Instant, args: &Args, exe: &Path) -> Result<i32, String> {
    exige_autorizacao(std::env::var(VARIAVEL).ok().as_deref())?;
    exige_uuid(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
    let (nome, mdir, tracos) = &args.celulas[0];
    let (cen, rep) = (args.cenario.ok_or("sem --cenario")?, args.rep.ok_or("sem --rep")?);
    // Todos os hashes de entrada saem ANTES de ler ids ou abrir a GPU (e entram no cronômetro frio).
    let h_bin0 = sha256(exe)?;
    let h_crit0 = sha256(&args.criterio)?;
    if h_crit0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_crit0}, não o congelado {CRITERIO_SHA}", args.criterio.display()));
    }
    let ids0 = hashes_de(&arquivos_de_ids(nome, tracos)?)?;
    let arquivo = prepara_saida_frio(&args.saida, &format!("frio-{nome}-{}-rep{rep}.txt", cen.nome()))?;
    let t_pre = ms(t0.elapsed());
    let mut partes: Vec<(String, f64)> = vec![("integridade_e_preparo".into(), t_pre)];
    let mut extras = extras_de_integridade(&h_bin0, &h_crit0, &ids0);
    // De agora em diante qualquer falha é GRAVADA (a repetição fica SEM PROVA) e nada se repete.
    let medida: Result<f64, String> = (|| {
        moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
        let mut marca = Instant::now();
        let mut passo = |nome: &str, partes: &mut Vec<(String, f64)>| {
            partes.push((nome.to_string(), ms(marca.elapsed())));
            marca = Instant::now();
        };
        let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json")).d()?).d()?).d()?;
        let ids = carrega_ids(nome, tracos, cfg.vocab)?;
        passo("config_e_ids", &mut partes);
        let device = Device::new(0).d()?;
        passo("device", &mut partes);
        let host = HostModel::load(&Checkpoint::open(mdir).d()?, cfg.clone()).d()?;
        passo("pesos_na_ram", &mut partes);
        let c = Cenario { device: &device, host: &host, cfg: &cfg, threads: args.threads };
        let r = le_cronometrado(&c, ids.de(cen), BLOCO)?;
        partes.push(("sequencia".into(), r.tempos.sequencia_ms));
        partes.push(("leitura".into(), r.tempos.leitura_ms));
        partes.push(("logits".into(), r.tempos.logits_ms));
        let total = ms(r.t_fim.duration_since(t0));
        let valido = r.cont.hits > 0 && r.cont.cpu > 0 && !r.logits.is_empty() && r.logits.iter().all(|v| v.is_finite());
        extras.push(("contadores".into(), format!("{:?}", r.cont)));
        extras.push(("logits_fnv".into(), format!("{:016x}", fnv(&r.logits))));
        extras.push(("argmax".into(), moe_rt::e4b_regra::argmax(&r.logits).to_string()));
        if !valido {
            return Err(format!("contadores sem hits ou sem cpu, ou logits não finitos ({:?}); tempo total {total:.1} ms", r.cont));
        }
        Ok(total)
    })();
    let integro = integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &ids0);
    let (total_ms, ok, falha) = match medida {
        Ok(t) => (t, true, None),
        Err(e) => (ms(t0.elapsed()), false, Some(e)),
    };
    let rec = RegistroFrio { modelo: nome.clone(), cen, rep, total_ms, ok, falha, integro, partes, extras };
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(&arquivo).map_err(|e| format!("{}: {e}", arquivo.display()))?;
    std::io::Write::write_all(&mut f, rec.para_texto().as_bytes()).map_err(|e| e.to_string())?;
    println!("frio {nome} {} rep {rep}: total {:.1} ms ({:.2} s), válido {}, integridade {}{}", cen.nome(), rec.total_ms, rec.total_ms / 1000.0, rec.ok, if integro { "íntegra" } else { "QUEBRADA" }, rec.falha.as_ref().map(|f| format!(", falha: {f}")).unwrap_or_default());
    Ok(if rec.ok && integro { 0 } else { 1 })
}

fn ensaio(args: &Args, exe: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let estado = args.estado.ok_or("sem --estado")?;
    println!("ensaio ({}): metas em segundos (1b / 3b): C {}/{}, A {}/{}", estado.nome(), vr::meta_ms(estado, "1b", Cen::C).unwrap() / 1000.0, vr::meta_ms(estado, "3b", Cen::C).unwrap() / 1000.0, vr::meta_ms(estado, "1b", Cen::A).unwrap() / 1000.0, vr::meta_ms(estado, "3b", Cen::A).unwrap() / 1000.0);
    match estado {
        Estado::Quente => println!("plano quente: por modelo ({:?}) e cenário (C {N_C} ids, A {N_A} ids): aquecimento descartado (modo 8; mais modo 1 no cenário C) e {} pares em ordem alternada (ímpar: modo 1 primeiro); uma chamada de le_ids por repetição; max_len {MAX_LEN}; limite entre modos {TOL_ENTRE_MODOS:.0e}", args.celulas.iter().map(|c| &c.0).collect::<Vec<_>>(), vr::PARES_QUENTE),
        Estado::Frio => println!("plano frio: UMA repetição ({:?}) da célula {} cenário {}, modo 8, do início do main até os logits lidos", args.rep, args.celulas[0].0, args.cenario.map(|c| c.nome()).unwrap_or("?")),
    }
    for (nome, mdir, tracos) in &args.celulas {
        let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json"))?)?)?;
        let ids = carrega_ids(nome, tracos, cfg.vocab)?;
        println!("ensaio {nome}: vocabulário {}, ids C {}, ids A {}; arquivos de ids: {:?}", cfg.vocab, ids.c.len(), ids.a.len(), hashes_de(&arquivos_de_ids(nome, tracos)?)?);
    }
    let hc = sha256(&args.criterio).unwrap_or_else(|e| e);
    println!("ensaio: nada foi aberto na GPU. critério {} ({hc}; {} o congelado); binário {} ({})", args.criterio.display(), if hc == CRITERIO_SHA { "CONFERE com" } else { "DIFERE d" }, exe.display(), sha256(exe).unwrap_or_else(|e| e));
    Ok(())
}

/// Devolve o código de saída: 0, 1 (gravado com falha) ou, por `Err`, 2 (recusa antes de qualquer trabalho).
fn principal(t0: Instant) -> Result<i32, String> {
    let args = parse_args(std::env::args().skip(1).collect())?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let estado = args.estado.ok_or("sem --estado")?;
    if let Some(dir) = &args.consolida {
        let ctx = Contexto { exe, criterio: args.criterio.clone(), sha_criterio: CRITERIO_SHA.into() };
        let (dec, _) = consolida(dir, estado, &args.modelos, &ctx)?;
        println!("{dec}");
        return Ok(0);
    }
    if args.ensaio {
        ensaio(&args, &exe).map_err(|e| e.to_string())?;
        return Ok(0);
    }
    match estado {
        Estado::Quente => quente(&args, &exe),
        Estado::Frio => frio(t0, &args, &exe),
    }
}

fn main() {
    // O primeiro instante do processo: o cronômetro frio começa aqui.
    let t0 = Instant::now();
    match principal(t0) {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("leitura-velocidade: recusado: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    fn pasta(nome: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("leitura-velocidade-{nome}-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn portao_proprio_recusa_antes_do_device() {
        assert!(exige_autorizacao(None).is_err());
        for x in ["0", "", "true", "2"] {
            assert!(exige_autorizacao(Some(x)).is_err(), "{x}");
        }
        assert!(exige_autorizacao(Some("1")).is_ok());
        assert_eq!(VARIAVEL, "LEITURA_VELOCIDADE_AUTORIZADA", "variável própria: nenhuma guarda antiga abre este programa");
        assert_ne!(VARIAVEL, "LEITURA_CORRECAO_AUTORIZADA");
        assert_ne!(VARIAVEL, "E4B_MEDIDA_AUTORIZADA");
    }

    #[test]
    fn so_o_uuid_da_3080ti_combinada_passa() {
        assert!(exige_uuid(Some(UUID_3080TI)).is_ok());
        assert_eq!(UUID_3080TI, "GPU-b22d9465-8552-9f97-812e-380b590645f8");
        assert!(exige_uuid(None).is_err());
        assert!(exige_uuid(Some("")).is_err());
        assert!(exige_uuid(Some("0")).is_err());
        assert!(exige_uuid(Some("GPU-00000000-0000-0000-0000-000000000000")).is_err(), "a 5090 e qualquer outra GPU nunca");
        assert!(exige_uuid(Some("GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd")).is_err(), "a 4090 antiga nunca");
        assert!(exige_uuid(Some(&format!("{UUID_3080TI},0"))).is_err());
    }

    #[test]
    fn argumentos() {
        let q = parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t", "--celula", "3b:/m2:/t", "--saida", "/o"])).unwrap();
        assert!(q.celulas.len() == 2 && !q.ensaio && q.threads == 8 && q.criterio == PathBuf::from(CRITERIO_PADRAO));
        assert!(parse_args(v(&["--celula", "1b:/m:/t", "--saida", "/o"])).is_err(), "sem estado");
        assert!(parse_args(v(&["--estado", "quente", "--saida", "/o"])).is_err(), "sem célula");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t"])).is_err(), "sem saída");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t", "--ensaio"])).is_ok(), "ensaio não precisa de saída");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "3b:/m:/t", "--celula", "1b:/m2:/t", "--saida", "/o"])).is_err(), "1b antes de 3b");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "3b:/m:/t", "--saida", "/o"])).is_ok(), "só o 3b é aceito");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "7b:/m:/t", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t", "--celula", "3b:/m:/t", "--saida", "/o"])).is_err(), "mesmo modelo em duas células");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t", "--saida", "/o", "--cenario", "C"])).is_err(), "cenário é do frio");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t", "--saida", "/o", "--pares", "9"])).is_err(), "parâmetros do critério não se trocam por opção");
        assert!(parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t", "--saida", "/o", "--meta", "9"])).is_err());
        // Frio.
        let f = parse_args(v(&["--estado", "frio", "--celula", "1b:/m:/t", "--cenario", "A", "--rep", "3", "--saida", "/o"])).unwrap();
        assert_eq!((f.cenario, f.rep), (Some(Cen::A), Some(3)));
        assert!(parse_args(v(&["--estado", "frio", "--celula", "1b:/m:/t", "--saida", "/o"])).is_err(), "sem cenário e rep");
        assert!(parse_args(v(&["--estado", "frio", "--celula", "1b:/m:/t", "--cenario", "A", "--rep", "4", "--saida", "/o"])).is_err(), "rep 4");
        assert!(parse_args(v(&["--estado", "frio", "--celula", "1b:/m:/t", "--cenario", "A", "--rep", "0", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--estado", "frio", "--celula", "1b:/m:/t", "--celula", "3b:/m2:/t", "--cenario", "A", "--rep", "1", "--saida", "/o"])).is_err(), "uma célula por processo");
        assert!(parse_args(v(&["--estado", "frio", "--celula", "1b:/m:/t", "--cenario", "B", "--rep", "1", "--saida", "/o"])).is_err(), "cenário B fora");
        // Consolida.
        let c = parse_args(v(&["--consolida", "/d", "--estado", "frio"])).unwrap();
        assert_eq!((c.consolida, c.modelos), (Some(PathBuf::from("/d")), v(&["1b", "3b"])));
        assert!(parse_args(v(&["--consolida", "/d"])).is_err(), "consolida pede o estado");
        assert!(parse_args(v(&["--consolida", "/d", "--estado", "frio", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--consolida", "/d", "--estado", "frio", "--modelos", "1b"])).is_ok());
        assert!(parse_args(v(&["--consolida", "/d", "--estado", "frio", "--modelos", "7b"])).is_err());
        assert!(parse_args(v(&["--consolida", "/d", "--estado", "frio", "--modelos", "1b,1b"])).is_err());
        assert!(parse_args(v(&["--estado", "quente", "--celula", "1b:/m:/t", "--saida", "/o", "--modelos", "1b"])).is_err());
    }

    #[test]
    fn saida_quente_nao_vazia_e_recusada() {
        let d = pasta("quente");
        assert!(prepara_saida(&d).is_ok(), "pasta vazia");
        std::fs::write(d.join("decisao.txt"), "x").unwrap();
        assert!(prepara_saida(&d).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn saida_fria_e_compartilhada_mas_nunca_sobrescreve_nem_mistura() {
        let d = pasta("frio");
        let n = d.join("nova");
        let a = prepara_saida_frio(&n, "frio-1b-C-rep1.txt").unwrap();
        assert!(n.is_dir() && a.ends_with("frio-1b-C-rep1.txt"));
        std::fs::write(&a, "x").unwrap();
        assert!(prepara_saida_frio(&n, "frio-1b-C-rep2.txt").is_ok(), "outra repetição na mesma pasta");
        assert!(prepara_saida_frio(&n, "frio-1b-C-rep1.txt").is_err(), "a mesma repetição não se repete");
        std::fs::write(n.join("job.log"), "x").unwrap();
        assert!(prepara_saida_frio(&n, "frio-1b-A-rep1.txt").is_ok(), "log do roteiro é aceito");
        std::fs::write(n.join("quente-1b-C.txt"), "x").unwrap();
        assert!(prepara_saida_frio(&n, "frio-1b-A-rep1.txt").is_err(), "resultado de outro estado");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn arquivos_de_ids_sao_os_do_criterio() {
        let a = arquivos_de_ids("1b", Path::new("/t")).unwrap();
        assert_eq!(a.len(), 3);
        assert!(a.iter().all(|p| p.ends_with("tokens.npy")));
        assert!(arquivos_de_ids("7b", Path::new("/t")).is_err());
    }

    #[test]
    fn hash_dos_ids_quebra_se_o_arquivo_muda() {
        let d = pasta("ids");
        let f = d.join("tokens.npy");
        std::fs::write(&f, b"um").unwrap();
        let h0 = hashes_de(&[f.clone()]).unwrap();
        assert!(ids_iguais(&h0));
        std::fs::write(&f, b"dois").unwrap();
        assert!(!ids_iguais(&h0));
        std::fs::remove_file(&f).unwrap();
        assert!(!ids_iguais(&h0), "arquivo sumiu: quebrada");
        std::fs::remove_dir_all(&d).unwrap();
    }

    fn rep_falsa(l: f64, g: f64, logits: Vec<f32>) -> Rep {
        Rep { tempos: RepQuente { sequencia_ms: 900.0, leitura_ms: l, logits_ms: g }, t_fim: Instant::now(), logits, cont: Contadores { needed: 10, hits: 4, cpu: 6 } }
    }

    #[test]
    fn celula_quente_com_motor_falso_ordem_aquecimento_e_pares() {
        let chamadas = RefCell::new(Vec::new());
        let rec = mede_quente("1b", Cen::C, true, |mp| {
            chamadas.borrow_mut().push(mp);
            // Modo 1 a 4000 ms e modo 8 a 2000 ms; logits iguais.
            Ok(rep_falsa(if mp == 1 { 3990.0 } else { 1990.0 }, 10.0, vec![0.1, 5.0, 0.2]))
        });
        // Aquecimento: 8 e 1 (descartados); depois ímpar (1, 8) e par (8, 1) alternados.
        assert_eq!(*chamadas.borrow(), vec![8, 1, 1, 8, 8, 1, 1, 8, 8, 1, 1, 8]);
        assert_eq!(rec.aquecimento.len(), 2);
        assert_eq!(rec.pares.len(), 5, "o aquecimento não vira par");
        assert!(rec.falha.is_none());
        assert_eq!(rec.pares.iter().map(|p| p.m1_primeiro).collect::<Vec<_>>(), vec![true, false, true, false, true]);
        let a = vr::avalia_quente("1b", Cen::C, &rec.pares, rec.falha.as_deref());
        assert_eq!(a.rotulo, Rotulo::Atinge, "{a:?}");
        assert_eq!(a.med8_ms, Some(2000.0));
        // Sem o aquecimento do modo 1 (células seguintes): só o modo 8.
        let c2 = RefCell::new(Vec::new());
        mede_quente("1b", Cen::A, false, |mp| {
            c2.borrow_mut().push(mp);
            Ok(rep_falsa(100.0, 10.0, vec![1.0, 2.0]))
        });
        assert_eq!(c2.borrow()[..3], [8, 1, 8]);
        assert_eq!(c2.borrow().len(), 11);
    }

    #[test]
    fn celula_quente_para_no_primeiro_erro_e_no_primeiro_portao_falho() {
        // Erro no meio: nada repete, a célula é SEM PROVA com o motivo.
        let n = RefCell::new(0);
        let rec = mede_quente("1b", Cen::C, false, |_| {
            *n.borrow_mut() += 1;
            if *n.borrow() == 5 { Err("posição do motor 499 depois de ler 500 ids".to_string()) } else { Ok(rep_falsa(100.0, 10.0, vec![1.0, 2.0])) }
        });
        assert_eq!(*n.borrow(), 5, "parou na falha");
        assert!(rec.falha.as_deref().unwrap().contains("posição do motor"));
        let a = vr::avalia_quente("1b", Cen::C, &rec.pares, rec.falha.as_deref());
        assert_eq!(a.rotulo, Rotulo::SemProva);
        // Erro no aquecimento.
        let rec = mede_quente("1b", Cen::C, true, |_| Err("sem memória".to_string()));
        assert!(rec.falha.unwrap().starts_with("aquecimento (modo 8)") && rec.pares.is_empty());
        // Portão do par 2 falha (logits do modo 8 diferentes): para ali, alerta de correção.
        let n = RefCell::new(0);
        let rec = mede_quente("1b", Cen::C, false, |mp| {
            *n.borrow_mut() += 1;
            let l = if mp == 8 && *n.borrow() > 3 { vec![9.0, 2.0] } else { vec![1.0, 2.0] };
            Ok(rep_falsa(100.0, 10.0, l))
        });
        assert_eq!(rec.pares.len(), 2);
        assert!(rec.pares[0].portao.ok() && !rec.pares[1].portao.ok());
        let a = vr::avalia_quente("1b", Cen::C, &rec.pares, rec.falha.as_deref());
        assert_eq!(a.rotulo, Rotulo::SemProva);
        assert!(a.motivos[0].contains("ALERTA DE CORREÇÃO"), "{a:?}");
    }

    fn ctx_falso(d: &Path) -> (Contexto, PathBuf) {
        let (exe, crit, ids) = (d.join("exe"), d.join("criterio.txt"), d.join("tokens.npy"));
        std::fs::write(&exe, b"binario").unwrap();
        std::fs::write(&crit, b"criterio").unwrap();
        std::fs::write(&ids, b"ids").unwrap();
        (Contexto { exe, criterio: crit.clone(), sha_criterio: sha256(&crit).unwrap() }, ids)
    }

    fn extras_falsos(ctx: &Contexto, ids: &Path) -> Vec<(String, String)> {
        extras_de_integridade(&sha256(&ctx.exe).unwrap(), &ctx.sha_criterio, &[(ids.to_path_buf(), sha256(ids).unwrap())])
    }

    fn grava_quente(d: &Path, ctx: &Contexto, ids: &Path, m: &str, cen: Cen, t1: f64, t8: f64, integro: bool) {
        let pares: Vec<ParQuente> = (1..=5)
            .map(|i| ParQuente { indice: i, m1_primeiro: vr::modo1_primeiro(i), m1: RepQuente { sequencia_ms: 1.0, leitura_ms: t1 - 10.0, logits_ms: 10.0 }, m8: RepQuente { sequencia_ms: 1.0, leitura_ms: t8 - 10.0, logits_ms: 10.0 }, portao: vr::PortaoPar { erro: 0.0, nao_finito: false, argmax_ok: true, cont_ok: true } })
            .collect();
        let r = RegistroQuente { modelo: m.into(), cen, integro, falha: None, pares, aquecimento: vec![], extras: extras_falsos(ctx, ids) };
        std::fs::write(d.join(format!("quente-{m}-{}.txt", cen.nome())), r.para_texto()).unwrap();
    }

    fn grava_frio(d: &Path, ctx: &Contexto, ids: &Path, m: &str, cen: Cen, rep: usize, total: f64) {
        let r = RegistroFrio { modelo: m.into(), cen, rep, total_ms: total, ok: true, falha: None, integro: true, partes: vec![], extras: extras_falsos(ctx, ids) };
        std::fs::write(d.join(format!("frio-{m}-{}-rep{rep}.txt", cen.nome())), r.para_texto()).unwrap();
    }

    #[test]
    fn consolida_quente_rotulos_por_celula_e_conjunto() {
        let d = pasta("cq");
        let (ctx, ids) = ctx_falso(&d);
        let sai = d.join("saida");
        std::fs::create_dir_all(&sai).unwrap();
        grava_quente(&sai, &ctx, &ids, "1b", Cen::C, 4000.0, 2000.0, true); // ATINGE
        grava_quente(&sai, &ctx, &ids, "1b", Cen::A, 20000.0, 17000.0, true); // ganho 15% ok, meta 15 s estourada: NÃO ATINGE (meta)
        grava_quente(&sai, &ctx, &ids, "3b", Cen::C, 5000.0, 4900.0, true); // NÃO ATINGE (ganho)
        // 3b A ausente: interrompida.
        let (dec, conj) = consolida(&sai, Estado::Quente, &["1b".into(), "3b".into()], &ctx).unwrap();
        assert_eq!(conj, Rotulo::NaoAtinge, "{dec}");
        assert!(dec.contains("1b C quente: ATINGE"), "{dec}");
        assert!(dec.contains("1b A quente: NÃO ATINGE") && dec.contains("[meta]"), "{dec}");
        assert!(dec.contains("3b C quente: NÃO ATINGE") && dec.contains("[ganho]"), "{dec}");
        assert!(dec.contains("3b A quente: SEM PROVA") && dec.contains("célula ausente"), "{dec}");
        assert!(sai.join("decisao.txt").exists() && sai.join("relatorio-quente-1b-C.txt").exists());
        // Só 1b e as duas atingem: ATINGE; com 3b faltando: SEM PROVA.
        let sai2 = d.join("saida2");
        std::fs::create_dir_all(&sai2).unwrap();
        grava_quente(&sai2, &ctx, &ids, "1b", Cen::C, 4000.0, 2000.0, true);
        grava_quente(&sai2, &ctx, &ids, "1b", Cen::A, 20000.0, 10000.0, true);
        assert_eq!(consolida(&sai2, Estado::Quente, &["1b".into()], &ctx).unwrap().1, Rotulo::Atinge);
        assert_eq!(consolida(&sai2, Estado::Quente, &["1b".into(), "3b".into()], &ctx).unwrap().1, Rotulo::SemProva);
        // Corte no meio: a célula fechada mantém o rótulo (quente-final ausente é só nota).
        let (dec, _) = consolida(&sai2, Estado::Quente, &["1b".into()], &ctx).unwrap();
        assert!(dec.contains("1b C quente: ATINGE") && dec.contains("quente-final.txt ausente"), "{dec}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn consolida_integridade_quebrada_deixa_tudo_sem_prova() {
        let d = pasta("ci");
        let (ctx, ids) = ctx_falso(&d);
        let sai = d.join("saida");
        std::fs::create_dir_all(&sai).unwrap();
        grava_quente(&sai, &ctx, &ids, "1b", Cen::C, 4000.0, 2000.0, true);
        grava_quente(&sai, &ctx, &ids, "1b", Cen::A, 20000.0, 10000.0, true);
        std::fs::write(sai.join("quente-final.txt"), "tipo=quente-final\nintegro=1\n").unwrap();
        assert_eq!(consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap().1, Rotulo::Atinge);
        // Flag do registro.
        grava_quente(&sai, &ctx, &ids, "1b", Cen::A, 20000.0, 10000.0, false);
        let (dec, c) = consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap();
        assert_eq!(c, Rotulo::SemProva);
        assert!(dec.contains("1b C quente: SEM PROVA") && dec.contains("integridade quebrada"), "{dec}");
        grava_quente(&sai, &ctx, &ids, "1b", Cen::A, 20000.0, 10000.0, true);
        // quente-final com integridade quebrada.
        std::fs::write(sai.join("quente-final.txt"), "tipo=quente-final\nintegro=0\n").unwrap();
        assert_eq!(consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap().1, Rotulo::SemProva);
        std::fs::write(sai.join("quente-final.txt"), "tipo=quente-final\nintegro=1\n").unwrap();
        assert_eq!(consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap().1, Rotulo::Atinge);
        // Os ids mudam depois da gravação.
        std::fs::write(&ids, b"outros").unwrap();
        assert_eq!(consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap().1, Rotulo::SemProva, "ids mudaram");
        std::fs::write(&ids, b"ids").unwrap();
        // O critério muda.
        std::fs::write(&ctx.criterio, b"criterio editado").unwrap();
        assert_eq!(consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap().1, Rotulo::SemProva, "critério mudou");
        std::fs::write(&ctx.criterio, b"criterio").unwrap();
        // O binário muda.
        std::fs::write(&ctx.exe, b"outro binario").unwrap();
        assert_eq!(consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap().1, Rotulo::SemProva, "binário mudou");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn consolida_quente_resultado_ilegivel_e_sem_prova() {
        let d = pasta("cl");
        let (ctx, ids) = ctx_falso(&d);
        let sai = d.join("saida");
        std::fs::create_dir_all(&sai).unwrap();
        grava_quente(&sai, &ctx, &ids, "1b", Cen::C, 4000.0, 2000.0, true);
        std::fs::write(sai.join("quente-1b-A.txt"), "lixo sem formato").unwrap();
        let (dec, c) = consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap();
        assert_eq!(c, Rotulo::SemProva);
        assert!(dec.contains("1b A quente: SEM PROVA") && dec.contains("ilegível"), "{dec}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn consolida_quente_registro_copiado_para_outra_celula_e_sem_prova() {
        let d = pasta("cid");
        let (ctx, ids) = ctx_falso(&d);
        let sai = d.join("saida");
        std::fs::create_dir_all(&sai).unwrap();
        grava_quente(&sai, &ctx, &ids, "1b", Cen::C, 4000.0, 2000.0, true);
        // A cópia de 1b/C no nome de 1b/A tem hashes íntegros, mas identidade gravada diferente.
        std::fs::copy(sai.join("quente-1b-C.txt"), sai.join("quente-1b-A.txt")).unwrap();
        let (dec, conj) = consolida(&sai, Estado::Quente, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, Rotulo::SemProva, "{dec}");
        assert!(dec.contains("identidade"), "{dec}");
        // 1b/C continua com o rótulo de prova (a falta de A é que deixa o conjunto sem prova).
        assert!(dec.contains("ATINGE"), "{dec}");
        // O mesmo vale entre modelos: registro de 1b no nome de 3b.
        std::fs::copy(sai.join("quente-1b-C.txt"), sai.join("quente-3b-C.txt")).unwrap();
        std::fs::copy(sai.join("quente-1b-C.txt"), sai.join("quente-3b-A.txt")).unwrap();
        let (dec3, c3) = consolida(&sai, Estado::Quente, &["3b".into()], &ctx).unwrap();
        assert_eq!(c3, Rotulo::SemProva, "{dec3}");
        assert!(dec3.matches("identidade").count() >= 2, "{dec3}");
    }

    #[test]
    fn consolida_frio_mediana_e_repeticoes() {
        let d = pasta("cf");
        let (ctx, ids) = ctx_falso(&d);
        let sai = d.join("saida");
        std::fs::create_dir_all(&sai).unwrap();
        for (cen, t) in [(Cen::C, [15000.0, 16000.0, 50000.0]), (Cen::A, [20000.0, 29000.0, 31000.0])] {
            for (i, x) in t.iter().enumerate() {
                grava_frio(&sai, &ctx, &ids, "1b", cen, i + 1, *x);
            }
        }
        // 1b C: mediana 16 s <= 20 s ATINGE (pior 50 s só descritiva); 1b A: mediana 29 s <= 30 s ATINGE.
        let (dec, c) = consolida(&sai, Estado::Frio, &["1b".into()], &ctx).unwrap();
        assert_eq!(c, Rotulo::Atinge, "{dec}");
        assert!(dec.contains("1b C frio: ATINGE") && dec.contains("1b A frio: ATINGE"), "{dec}");
        // Uma repetição a menos (processo cortado): SEM PROVA para a célula; as outras mantêm.
        std::fs::remove_file(sai.join("frio-1b-A-rep3.txt")).unwrap();
        let (dec, c) = consolida(&sai, Estado::Frio, &["1b".into()], &ctx).unwrap();
        assert_eq!(c, Rotulo::SemProva);
        assert!(dec.contains("1b C frio: ATINGE") && dec.contains("1b A frio: SEM PROVA") && dec.contains("menos de 3"), "{dec}");
        // Meta estourada: NÃO ATINGE vence.
        grava_frio(&sai, &ctx, &ids, "1b", Cen::A, 3, 60000.0);
        grava_frio(&sai, &ctx, &ids, "1b", Cen::A, 2, 40000.0);
        let (_, c) = consolida(&sai, Estado::Frio, &["1b".into()], &ctx).unwrap();
        assert_eq!(c, Rotulo::NaoAtinge);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn cronometro_e_fnv_sao_deterministicos() {
        assert_eq!(fnv(&[1.0, 2.0]), fnv(&[1.0, 2.0]));
        assert_ne!(fnv(&[1.0, 2.0]), fnv(&[2.0, 1.0]));
        assert!((ms(std::time::Duration::from_millis(1500)) - 1500.0).abs() < 1e-9);
    }

    #[test]
    fn criterio_sha_e_um_sha256_hex() {
        assert_eq!(CRITERIO_SHA.len(), 64);
        assert!(CRITERIO_SHA.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
