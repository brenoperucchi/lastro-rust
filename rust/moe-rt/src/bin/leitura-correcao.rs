//! A2 (textos longos), etapa 3: o programa de CORREÇÃO da leitura em blocos de até 8 posições (`le_ids`) nos modelos reais, pelo critério
//! `qual-runs/A2-LOTE-20261103/criterio-leitura-correcao.txt` (tolerâncias aprovadas pelo Breno em 07/11/2026, congeladas antes de qualquer
//! dado real). NÃO RODA SEM AUTORIZAÇÃO. Não mede velocidade.
//!
//! Só abre a GPU com `LEITURA_CORRECAO_AUTORIZADA=1` (variável própria: nenhuma guarda antiga abre este programa) e com
//! `CUDA_VISIBLE_DEVICES` igual ao UUID combinado da RTX 3080 Ti. A 4090, a 5090 e qualquer outra nunca. `--ensaio` imprime o plano e confere ids e hashes SEM abrir o device e sem a variável.
//!
//! Por célula (um modelo; nome `1b` ou `3b`), nesta ordem:
//!  1. ids dos traços de conversa (cenário C: 500 ids; cenário A: 2.003 ids; continuação: 1.205 e 798 dos mesmos ids de A);
//!  2. referência X: a passada residente (`GpuModel`), um token por vez, logits nas pontas de cada leitura;
//!  3. modo 1 (`max_positions = 1`) e modo 8 (`max_positions = 8`), tabela congelada, um motor e um backend NOVOS por leitura;
//!  4. comparações (a), (b) e (c) do critério, validade (nível 1), rótulos (níveis 2 e 3), tudo por funções puras de `moe_rt::leitura_regra`.
//! Uma célula que termina grava logo `<nome>-parcial.txt` (o rótulo final depende da integridade, conferida no fim); o relatório final e a
//! decisão do conjunto só saem depois dela. Nada é repetido nem alterado depois de ver um resultado.
//!
//!   leitura-correcao --celula 1b:<dir do modelo>:<dir dos traços do modelo> [--celula 3b:...] --saida <dir nova> [--criterio <arquivo>]
//!                    [--threads N] [--ensaio]
use cuda_core::Device;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream, LeitorCuda};
use moe_rt::gpu_model::GpuModel;
use moe_rt::leitura_regra::*;
use moe_rt::model::{Config as ModelConfig, HostModel};
use moe_rt::motor::leitura::{le_ids, PreparaPasso};
use moe_rt::motor::{AdmitMode, Config, Engine};
use moe_rt::npy::Npy;
use moe_rt::weights::Checkpoint;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

const VARIAVEL: &str = "LEITURA_CORRECAO_AUTORIZADA";
/// O UUID da RTX 3080 Ti combinado para a janela: `CUDA_VISIBLE_DEVICES` tem de ser exatamente isto.
const UUID_3080TI: &str = "GPU-b22d9465-8552-9f97-812e-380b590645f8";
/// O hash do critério aprovado e congelado: um `--criterio` com outro conteúdo recusa antes de abrir a GPU.
const CRITERIO_SHA: &str = "f18f659797582f45d3d8732fb89a08c8451bd28c3c5dde295e3b2b6a572eba12";
const FRACAO_SLOTS: f64 = 0.10;
const CRITERIO_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-leitura-correcao.txt";

struct Args {
    celulas: Vec<(String, PathBuf, PathBuf)>,
    saida: PathBuf,
    criterio: PathBuf,
    threads: usize,
    ensaio: bool,
}

fn nome_valido(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn canonico(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

fn parse_args(v: Vec<String>) -> Result<Args, String> {
    let mut a = v.into_iter();
    let mut r = Args { celulas: vec![], saida: PathBuf::new(), criterio: PathBuf::from(CRITERIO_PADRAO), threads: 8, ensaio: false };
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
            "--saida" => r.saida = PathBuf::from(val()?),
            "--criterio" => r.criterio = PathBuf::from(val()?),
            "--threads" => r.threads = val()?.parse().map_err(|_| "--threads".to_string())?,
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

/// A pasta de saída tem de ser nova ou vazia: um resultado antigo não pode sobreviver a esta execução.
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
        _ => Err(format!("a correção da leitura na GPU só roda com o 'autorizo' explícito do Breno ({VARIAVEL}=1); veja o critério em criterio-leitura-correcao.txt")),
    }
}

/// O UUID da GPU: `CUDA_VISIBLE_DEVICES` tem de ser exatamente o da 3080 Ti combinada.
fn exige_uuid(valor: Option<&str>) -> Result<(), String> {
    match valor {
        Some(UUID_3080TI) => Ok(()),
        outro => Err(format!("CUDA_VISIBLE_DEVICES tem de ser {UUID_3080TI} (a 3080 Ti combinada); veio {outro:?}")),
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

fn sha256(p: &Path) -> Result<String, String> {
    let o = std::process::Command::new("sha256sum").arg(p).output().map_err(|e| format!("sha256sum: {e}"))?;
    if !o.status.success() {
        return Err(format!("sha256sum falhou em {}", p.display()));
    }
    Ok(String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string())
}

/// Os arquivos de ids de uma célula (os que o critério manda usar) e os ids dos dois cenários.
struct Ids {
    c: Vec<usize>,
    a: Vec<usize>,
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

/// Verdadeiro só se TODOS os arquivos ainda têm o hash inicial (hash indisponível = quebrada).
fn ids_iguais(ids0: &[(PathBuf, String)]) -> bool {
    ids0.iter().all(|(p, h)| sha256(p).ok().as_deref() == Some(h.as_str()))
}

/// Integridade num instante: binário, critério (contra o inicial) e `tokens.npy` (contra os iniciais, tirados ANTES de usar os ids).
fn integro_agora(exe: &Path, criterio: &Path, h_bin0: &str, h_crit0: &str, ids0: &[(PathBuf, String)]) -> bool {
    sha256(exe).ok().as_deref() == Some(h_bin0) && sha256(criterio).ok().as_deref() == Some(h_crit0) && ids_iguais(ids0)
}

/// O rótulo final da célula: integridade quebrada deixa SEM PROVA, senão vale o rótulo da célula.
fn rotulo_final(s: &SaidaCelula, integro: bool) -> Rotulo {
    if integro {
        s.rotulo()
    } else {
        Rotulo::SemProva
    }
}

fn texto_parcial(s: &SaidaCelula, integro: bool) -> String {
    format!("{}integridade conferida no fim da célula: {}\nrótulo da célula: {}\n", s.corpo, if integro { "íntegra" } else { "QUEBRADA" }, rotulo_final(s, integro).nome())
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

/// O que uma leitura (uma sequência) devolve: os logits de cada ponta, o KV de todas as camadas e os contadores.
struct Leitura {
    logits: Vec<Vec<f32>>,
    /// KV de todas as camadas, lido depois de CADA chamada de `le_ids` (um item por corte).
    kv: Vec<Vec<(Vec<f32>, Vec<f32>)>>,
    cont: Contadores,
}

impl Leitura {
    fn kv_fim(&self) -> &Vec<(Vec<f32>, Vec<f32>)> {
        self.kv.last().expect("pelo menos um corte")
    }
}

/// Uma sequência: motor e backend novos (tabela congelada, `backlog = 0`, admissão Event), `ids` lidos em chamadas de `le_ids` nos `cortes`.
/// Confere a posição do motor e a do backend ao fim; devolve os logits e o KV depois de cada chamada e os contadores.
fn le(c: &Cenario, ids: &[usize], cortes: &[usize], max_positions: usize) -> Result<Leitura, String> {
    if cortes.iter().sum::<usize>() != ids.len() {
        return Err("os cortes não somam os ids".into());
    }
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
    let mut logits = Vec::new();
    let mut kvs = Vec::new();
    let mut ini = 0;
    for &n in cortes {
        let lido = le_ids(&mut eng, &leitor, &ids[ini..ini + n]).map_err(|e| format!("le_ids: {e:?}"))?;
        if lido != n {
            return Err(format!("le_ids leu {lido} em vez de {n}"));
        }
        ini += n;
        logits.push(b.read_logits(&compute).d()?);
        kvs.push((0..m.layers).map(|l| b.read_kv(&compute, l, ini).d()).collect::<Result<Vec<_>, String>>()?);
    }
    if eng.position() != ids.len() as u64 {
        return Err(format!("posição do motor {} depois de ler {} ids", eng.position(), ids.len()));
    }
    if leitor.posicao() != ids.len() as u64 {
        return Err(format!("posição do backend {} depois de ler {} ids", leitor.posicao(), ids.len()));
    }
    let s = eng.next_published().d()?.stats().clone();
    eng.close().d()?;
    Ok(Leitura { logits, kv: kvs, cont: Contadores { needed: s.needed, hits: s.hits, cpu: s.cpu } })
}

/// A referência X: a passada residente (um token por vez), com os logits nas posições pedidas (`pontas`, índices crescentes).
fn referencia(c: &Cenario, ids: &[usize], pontas: &[usize]) -> Result<Vec<Vec<f32>>, String> {
    let mut g = GpuModel::upload(c.host, c.device.new_stream().d()?, MAX_LEN).d()?;
    let mut saida = Vec::new();
    for (i, &t) in ids.iter().enumerate() {
        let l = g.step(t).d()?.0;
        if pontas.contains(&i) {
            saida.push(l);
        }
    }
    if saida.len() != pontas.len() {
        return Err("a referência não cobriu todas as pontas".into());
    }
    Ok(saida)
}

/// O resultado de uma célula: o relatório, o rótulo (antes da integridade) e as comparações.
struct SaidaCelula {
    nome: String,
    valida: bool,
    comparacoes: Vec<Comparacao>,
    corpo: String,
}

impl SaidaCelula {
    fn rotulo(&self) -> Rotulo {
        rotulo_celula(self.valida, &self.comparacoes)
    }
}

fn texto_cmp(c: &Comparacao) -> String {
    format!("  {:<44} erro {:.3e} (limite {:.0e}){}{} => {}\n", c.nome, c.erro, c.limite, if c.nao_finito { " NÃO FINITO" } else { "" }, if c.argmax_ok { "" } else { " ARGMAX DIFERENTE" }, if c.ok { "OK" } else { "FALHA NUMÉRICA" })
}

fn sem_prova(nome: &str, mut corpo: String, motivo: &str) -> SaidaCelula {
    corpo += &format!("validade (nível 1): FALHOU: {motivo}\n");
    SaidaCelula { nome: nome.into(), valida: false, comparacoes: vec![], corpo }
}

fn celula(c: &Cenario, nome: &str, ids: &Ids) -> SaidaCelula {
    let kvd = c.cfg.kv_heads * c.cfg.head_dim;
    let mut rel = format!("== célula {nome}: {} camadas, {} experts, {} slots ({}%), max_len {MAX_LEN}\n", c.cfg.layers, c.cfg.experts, c.slots_10(), FRACAO_SLOTS * 100.0);
    rel += &format!("ids: C {} (resto {}), A {} (resto {}), continuação {:?} (restos {} e {})\n", ids.c.len(), resto(N_C), ids.a.len(), resto(N_A), N_CONT, resto(N_CONT[0]), resto(N_CONT[1]));
    macro_rules! tenta {
        ($e:expr, $ctx:expr) => {
            match $e {
                Ok(v) => v,
                Err(e) => return sem_prova(nome, rel, &format!("{}: {e}", $ctx)),
            }
        };
    }
    // Referência X (passada residente) nas pontas.
    let x_c = tenta!(referencia(c, &ids.c, &[N_C - 1]), "referência X do cenário C");
    let x_a = tenta!(referencia(c, &ids.a, &[N_CONT[0] - 1, N_A - 1]), "referência X do cenário A");
    // Cenário C.
    let c1 = tenta!(le(c, &ids.c, &[N_C], 1), "modo 1, cenário C");
    let c8 = tenta!(le(c, &ids.c, &[N_C], BLOCO), "modo 8, cenário C");
    // Cenário A: modo 1 em duas leituras (para a ponta 1.204), modo 8 numa leitura só e em continuação.
    let a1 = tenta!(le(c, &ids.a, &N_CONT, 1), "modo 1, cenário A (continuação)");
    let a8 = tenta!(le(c, &ids.a, &[N_A], BLOCO), "modo 8, cenário A (uma leitura)");
    let a8c = tenta!(le(c, &ids.a, &N_CONT, BLOCO), "modo 8, cenário A (continuação)");
    rel += &format!("contadores (needed, hits, cpu): C m1 {:?} m8 {:?}; A m1 {:?} m8 {:?} m8-cont {:?}\n", c1.cont, c8.cont, a1.cont, a8.cont, a8c.cont);
    // Validade (nível 1): contadores.
    let cont_ok = contadores_validos(c8.cont, c1.cont) && contadores_validos(a8.cont, a1.cont) && contadores_validos(a8c.cont, a1.cont);
    if !cont_ok {
        return sem_prova(nome, rel, "contadores hits/cpu zerados ou diferentes entre os modos");
    }
    let cmps = vec![
        // (a) modo 8 contra modo 1.
        compara_entre_modos("a C: logits@499 m8 vs m1", &c8.logits[0], &c1.logits[0]),
        compara_kv("a C: KV m8 vs m1", c8.kv_fim(), c1.kv_fim()),
        compara_entre_modos("a A: logits@2002 m8(uma) vs m1", &a8.logits[0], &a1.logits[1]),
        compara_kv("a A: KV m8(uma) vs m1", a8.kv_fim(), a1.kv_fim()),
        compara_entre_modos("a A-cont: logits@1204 m8 vs m1", &a8c.logits[0], &a1.logits[0]),
        compara_entre_modos("a A-cont: logits@2002 m8 vs m1", &a8c.logits[1], &a1.logits[1]),
        compara_kv("a A-cont: KV@1204 m8 vs m1", &a8c.kv[0], &a1.kv[0]),
        compara_kv("a A-cont: KV@2002 m8 vs m1", a8c.kv_fim(), a1.kv_fim()),
        // (b) modo 8 contra a passada residente.
        compara_logits_x("b C: logits@499 m8 vs X", &c8.logits[0], &x_c[0]),
        compara_logits_x("b A: logits@2002 m8(uma) vs X", &a8.logits[0], &x_a[1]),
        compara_logits_x("b A-cont: logits@1204 m8 vs X", &a8c.logits[0], &x_a[0]),
        compara_logits_x("b A-cont: logits@2002 m8 vs X", &a8c.logits[1], &x_a[1]),
        // (c) continuação contra uma leitura só.
        compara_entre_modos("c: logits@2002 m8(cont) vs m8(uma)", &a8c.logits[1], &a8.logits[0]),
        compara_kv("c: KV m8(cont) vs m8(uma)", a8c.kv_fim(), a8.kv_fim()),
    ];
    rel += "comparações (nível 2):\n";
    for x in &cmps {
        rel += &texto_cmp(x);
    }
    let perfil = |g: &Leitura, w: &Leitura| perfil_kv(g.kv_fim(), w.kv_fim(), kvd);
    rel += "perfil do erro do KV por faixa de posições (só descrição; não decide): faixas 0-63, 64-511, 512-1023, 1024-2047\n";
    for (nome_p, p) in [("a C (m8 vs m1)", perfil(&c8, &c1)), ("a A (m8 uma vs m1)", perfil(&a8, &a1)), ("a A-cont (m8 vs m1)", perfil(&a8c, &a1))] {
        rel += &format!("  {nome_p}: {}\n", p.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>().join("  "));
    }
    SaidaCelula { nome: nome.into(), valida: true, comparacoes: cmps, corpo: rel }
}

fn roda(device: &Arc<Device>, args: &Args, nome: &str, mdir: &Path, tracos: &Path) -> Result<SaidaCelula, Box<dyn std::error::Error>> {
    let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json"))?)?)?;
    let ids = carrega_ids(nome, tracos, cfg.vocab)?;
    let host = HostModel::load(&Checkpoint::open(mdir)?, cfg.clone())?;
    let c = Cenario { device, host: &host, cfg: &cfg, threads: args.threads };
    Ok(celula(&c, nome, &ids))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args(std::env::args().skip(1).collect())?;
    let exe = std::env::current_exe()?;
    println!("leitura-correcao: células {:?}; C {N_C} ids, A {N_A} ids, continuação {N_CONT:?}; max_len {MAX_LEN}; limites {TOL_ENTRE_MODOS:.0e} (entre modos) e {TOL_LOGITS_CONTRA_X:.0e} (logits contra X, L2)", args.celulas.iter().map(|c| &c.0).collect::<Vec<_>>());
    if args.ensaio {
        // Sem device e sem a variável: confere ids (tamanho e vocabulário) e imprime hashes. Não é prova de nada.
        for (nome, mdir, tracos) in &args.celulas {
            let cfg = ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json"))?)?)?;
            let ids = carrega_ids(nome, tracos, cfg.vocab)?;
            println!("ensaio {nome}: vocabulário {}, ids C {}, ids A {}; arquivos de ids: {:?}", cfg.vocab, ids.c.len(), ids.a.len(), hashes_de(&arquivos_de_ids(nome, tracos)?)?);
        }
        println!("ensaio: nada foi aberto na GPU. critério {} ({}); binário {} ({})", args.criterio.display(), sha256(&args.criterio).unwrap_or_else(|e| e), exe.display(), sha256(&exe).unwrap_or_else(|e| e));
        return Ok(());
    }
    exige_autorizacao(std::env::var(VARIAVEL).ok().as_deref())?;
    exige_uuid(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())?;
    // Todos os hashes de entrada saem ANTES de ler ids ou abrir a GPU.
    let h_bin0 = sha256(&exe)?;
    let h_crit0 = sha256(&args.criterio)?;
    if h_crit0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_crit0}, não o congelado {CRITERIO_SHA}", args.criterio.display()).into());
    }
    let mut arquivos_ids: Vec<PathBuf> = Vec::new();
    for (nome, _, tracos) in &args.celulas {
        arquivos_ids.extend(arquivos_de_ids(nome, tracos)?);
    }
    arquivos_ids.sort();
    arquivos_ids.dedup();
    let hashes_ids0 = hashes_de(&arquivos_ids)?;
    prepara_saida(&args.saida)?;
    std::fs::write(args.saida.join("decisao.txt"), "em execução: sem decisão\n")?;
    println!("integridade antes: binário {h_bin0}; critério {h_crit0}; {} arquivos de ids", hashes_ids0.len());
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let device = Device::new(0)?;
    let mut saidas: Vec<SaidaCelula> = Vec::new();
    let mut integro = true;
    for (nome, mdir, tracos) in &args.celulas {
        let s = match roda(&device, &args, nome, mdir, tracos) {
            Ok(s) => s,
            Err(e) => sem_prova(nome, format!("== célula {nome}: erro\n"), &e.to_string()),
        };
        // Integridade conferida no fim de CADA célula: o parcial já sai com o rótulo final até aqui.
        let integro_celula = integro_agora(&exe, &args.criterio, &h_bin0, &h_crit0, &hashes_ids0);
        integro &= integro_celula;
        let txt = texto_parcial(&s, integro);
        println!("{txt}");
        std::fs::write(args.saida.join(format!("{}-parcial.txt", s.nome)), txt)?;
        saidas.push(s);
    }
    integro &= integro_agora(&exe, &args.criterio, &h_bin0, &h_crit0, &hashes_ids0);
    let nota = format!(
        "integridade: binário, critério ({CRITERIO_SHA}) e {} arquivos de ids: {}",
        hashes_ids0.len(),
        if integro { "IGUAIS aos iniciais" } else { "DIFERENTES OU NÃO CALCULÁVEIS: todas as células ficam SEM PROVA" }
    );
    println!("{nota}");
    let mut dec = String::from("decisão da correção da leitura (regra de rótulos do critério):\n");
    let mut rotulos = Vec::new();
    for s in &saidas {
        let r = rotulo_final(s, integro);
        rotulos.push(r);
        std::fs::write(args.saida.join(format!("{}-relatorio.txt", s.nome)), format!("{}rótulo da célula: {}\n{nota}\n", s.corpo, r.nome()))?;
        dec += &format!("  {}: {}\n", s.nome, r.nome());
    }
    let conj = rotulo_conjunto(&rotulos, integro);
    dec += &format!("conjunto: {}\n{nota}\nbinário {h_bin0}\n", conj.nome());
    for (p, h) in &hashes_ids0 {
        dec += &format!("ids {}: {h}\n", p.display());
    }
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
        for x in ["0", "", "true", "2"] {
            assert!(exige_autorizacao(Some(x)).is_err(), "{x}");
        }
        assert!(exige_autorizacao(Some("1")).is_ok());
        assert_eq!(VARIAVEL, "LEITURA_CORRECAO_AUTORIZADA", "variável própria: nenhuma guarda antiga abre este programa");
    }

    #[test]
    fn so_o_uuid_da_3080ti_combinada_passa() {
        assert!(exige_uuid(Some(UUID_3080TI)).is_ok());
        assert!(exige_uuid(None).is_err(), "sem CUDA_VISIBLE_DEVICES");
        assert!(exige_uuid(Some("")).is_err());
        assert!(exige_uuid(Some("0")).is_err());
        assert!(exige_uuid(Some("GPU-00000000-0000-0000-0000-000000000000")).is_err(), "a 5090 e qualquer outra GPU nunca");
        assert!(exige_uuid(Some("GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd")).is_err(), "a 4090 antiga nunca");
        assert!(exige_uuid(Some(&format!("{UUID_3080TI},0"))).is_err(), "lista com mais de um device");
    }

    #[test]
    fn argumentos() {
        let a = parse_args(v(&["--celula", "1b:/m:/t", "--saida", "/o", "--ensaio"])).unwrap();
        assert_eq!(a.celulas.len(), 1);
        assert!(a.ensaio && a.threads == 8 && a.criterio == PathBuf::from(CRITERIO_PADRAO));
        assert!(parse_args(v(&["--saida", "/o"])).is_err(), "sem célula");
        assert!(parse_args(v(&["--celula", "1b:/m:/t"])).is_err(), "sem saída");
        assert!(parse_args(v(&["--celula", "1b:/m", "--saida", "/o"])).is_err(), "célula sem traços");
        assert!(parse_args(v(&["--celula", "7b:/m:/t", "--saida", "/o"])).is_err(), "só 1b e 3b têm fonte de ids");
        assert!(parse_args(v(&["--celula", "1b:/m:/t", "--celula", "1b:/m2:/t", "--saida", "/o"])).is_err(), "nome repetido");
        assert!(parse_args(v(&["--celula", "1b:/m:/t", "--celula", "3b:/m:/t", "--saida", "/o"])).is_err(), "mesmo modelo em duas células");
        assert!(parse_args(v(&["--celula", "1b:/tmp:/t", "--celula", "3b:/tmp/../tmp:/t", "--saida", "/o"])).is_err(), "alias de caminho");
        assert!(parse_args(v(&["--celula", "1b:/m:/t", "--saida", "/o", "--tol", "1e-3"])).is_err(), "os limites do critério não se trocam por opção");
        assert!(parse_args(v(&["--celula", "1b:/m:/t", "--celula", "3b:/m2:/t", "--saida", "/o"])).is_ok());
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
    fn saida_nao_vazia_e_recusada() {
        let d = std::env::temp_dir().join(format!("leitura-correcao-teste-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        assert!(prepara_saida(&d).is_ok(), "pasta nova");
        assert!(prepara_saida(&d).is_ok(), "pasta vazia");
        std::fs::write(d.join("decisao.txt"), "PASSA").unwrap();
        assert!(prepara_saida(&d).is_err(), "pasta com resultado antigo");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn celula_sem_prova_nunca_vira_passa_nem_reprova() {
        let s = sem_prova("1b", "== célula 1b\n".into(), "motivo");
        assert_eq!(s.rotulo(), Rotulo::SemProva);
        assert!(s.corpo.contains("validade (nível 1): FALHOU: motivo"));
    }

    #[test]
    fn hash_dos_ids_quebra_se_o_arquivo_muda_depois_do_baseline() {
        let d = std::env::temp_dir().join(format!("leitura-correcao-ids-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&d).unwrap();
        let f = d.join("tokens.npy");
        std::fs::write(&f, b"um").unwrap();
        let h0 = hashes_de(&[f.clone()]).unwrap();
        assert!(ids_iguais(&h0));
        std::fs::write(&f, b"dois").unwrap();
        assert!(!ids_iguais(&h0), "mudou depois do baseline");
        std::fs::write(&f, b"um").unwrap();
        assert!(ids_iguais(&h0));
        std::fs::remove_file(&f).unwrap();
        assert!(!ids_iguais(&h0), "arquivo sumiu: hash indisponível vale como quebrada");
        assert!(hashes_de(&[f]).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn parcial_leva_o_rotulo_final_com_a_integridade_da_celula() {
        let boa = SaidaCelula { nome: "1b".into(), valida: true, comparacoes: vec![Comparacao { nome: "x".into(), erro: 0.0, limite: 1e-5, nao_finito: false, argmax_ok: true, ok: true }], corpo: "corpo\n".into() };
        assert_eq!(SaidaCelula { nome: "1b".into(), valida: true, comparacoes: vec![], corpo: String::new() }.rotulo(), Rotulo::SemProva, "sem comparação não há prova");
        assert_eq!(rotulo_final(&boa, true), Rotulo::Passa);
        assert_eq!(rotulo_final(&boa, false), Rotulo::SemProva, "integridade quebrada nunca passa");
        assert!(texto_parcial(&boa, true).contains("íntegra") && texto_parcial(&boa, true).contains("PASSA"));
        assert!(texto_parcial(&boa, false).contains("QUEBRADA") && texto_parcial(&boa, false).contains("SEM PROVA"));
        let ruim = SaidaCelula { nome: "1b".into(), valida: true, comparacoes: vec![Comparacao { nome: "x".into(), erro: f64::INFINITY, limite: 1e-5, nao_finito: true, argmax_ok: true, ok: false }], corpo: String::new() };
        assert_eq!(rotulo_final(&ruim, true), Rotulo::Reprova);
        assert_eq!(rotulo_final(&ruim, false), Rotulo::SemProva);
    }

    #[test]
    fn arquivos_de_ids_sao_os_do_criterio() {
        let a = arquivos_de_ids("1b", Path::new("/t")).unwrap();
        assert_eq!(a.len(), 3, "conversa-4, 0 e 1");
        assert!(a.iter().all(|p| p.ends_with("tokens.npy")));
        assert_eq!(arquivos_de_ids("3b", Path::new("/t")).unwrap().len(), 3);
        assert!(arquivos_de_ids("7b", Path::new("/t")).is_err());
    }
}
