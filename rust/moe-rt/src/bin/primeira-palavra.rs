//! A2 (textos longos), etapa 5, degrau 5b: PRIMEIRA PALAVRA a partir de texto cru, nos Granite 3.1 1B e 3B, pelo critério
//! `qual-runs/A2-LOTE-20261103/criterio-5b-primeira-palavra.txt` (congelado antes de qualquer dado de GPU). NÃO RODA SEM AUTORIZAÇÃO.
//!
//! Só abre a GPU com `PRIMEIRA_PALAVRA_AUTORIZADA=1` (variável própria: nenhuma guarda antiga abre este programa) e com `CUDA_VISIBLE_DEVICES`
//! igual ao UUID combinado da RTX 3080 Ti. A 4090, a 5090 e qualquer outra nunca. `--ensaio` imprime o plano, confere hashes e conta os ids dos textos SEM abrir o device e sem a
//! variável. `--consolida` só lê resultados já gravados (sem GPU, sem a variável).
//!
//! Modos:
//!  - `--modo livre --celula <1b|3b>:<dir do modelo> [--texto T | --pedido arquivo.json:N | (stdin)] [--data AAAA-MM-DD]`: uso de verdade, uma
//!    primeira palavra; não decide rótulo nem vale como prova. Sem `--texto` nem `--pedido`, lê o texto da entrada padrão.
//!  - `--modo correcao --celula 1b:<dir> [--celula 3b:<dir>] --saida <dir nova>`: 3 pedidos por modelo contra a referência do HF (rótulos PASSA,
//!    REPROVA e SEM PROVA). Data fixa 2026-11-07.
//!  - `--modo velocidade --estado quente --celula ... --saida <dir nova>`: UM processo; por modelo (1b antes de 3b) e cenário (C antes de A), 1
//!    aquecimento descartado e 5 repetições, motor e backend NOVOS por pedido (T_sequencia entra no indicador de produto: o Engine não tem reset).
//!  - `--modo velocidade --estado frio --celula nome:dir --cenario C|A --rep 1|2|3 --saida <dir>`: UM processo = UMA repetição; indicador do início
//!    do `main` até a palavra entregue. Não decide rótulo.
//!  - `--prova-reset --celula 1b:<dir> [--celula 3b:<dir>] --saida <dir nova> [--criterio-reset arquivo]`: PROVA 2 do reset de sequência
//!    (`criterio-reset-sequencia-20261108.md`): ordem A longo, C curto, A longo, C curto nos braços A (motor novo por pedido) e B (um motor
//!    reutilizado com `nova_sequencia`); ids, token e cada logit por `to_bits` idênticos ou REPROVA. Mesmos portões dos outros modos.
//!  - `--gera --celula 1b:<dir> [--celula 3b:<dir>] --saida <dir nova> [--criterio-5c arquivo] [--textos-5c dir] [--referencia-5c dir]`: ETAPA 5c, geração
//!    livre curta (`criterio-5c-geracao-livre-20261108.md`): 3 textos crus fixos (curto, médio, longo), template com data fixa, 64 tokens gulosos
//!    por pedido (a prova NUNCA para no EOS), um motor reutilizado entre os 3 pedidos (`nova_sequencia`), comparação com a referência do HF
//!    (`tools/ref_geracao_5c.py`; empate 1e-3). `--gera --celula nome:<dir> --texto T [--novos N] [--data AAAA-MM-DD]` é o uso normal: gera até N
//!    tokens, PARA no primeiro EOS e imprime o texto; não decide rótulo. Mesmos portões dos outros modos; `--ensaio` não abre a GPU.
//!  - `--consolida <dir> --modo correcao|velocidade [--estado quente|frio] [--modelos 1b,3b]`: relê os brutos, aplica as regras de
//!    `moe_rt::primeira_palavra_regra` e escreve `decisao.txt` e os relatórios.
//! Códigos de saída: 0 gravado e válido, 1 gravado com falha (a célula fica SEM PROVA), 2 recusa antes de qualquer trabalho. Nada repete sozinho.
use cuda_core::Device;
use moe_rt::cpu::Cpu;
use moe_rt::cuda_backend::{CpuServe, CudaBackend, CudaStream, LeitorCuda};
use moe_rt::geracao_regra as gr;
use moe_rt::leitura_regra::{fontes_de_ids, BLOCO, MAX_LEN};
use moe_rt::model::{Config as ModelConfig, HostModel};
use moe_rt::motor::{AdmitMode, Config, Engine};
use moe_rt::primeira_palavra_regra as pp;
use moe_rt::primeira_palavra_regra::{CasoCorrecao, RefCaso, RegistroCorrecao, RegistroFrioPP, RegistroQuente, RepMedida, RotuloCorrecao, Sequencia};
use moe_rt::reset_regra as rr;
use moe_rt::tokenizador::{formata_data, renderiza_chat, Mensagem, Papel, Tokenizador};
use moe_rt::velocidade_regra as vr;
use moe_rt::velocidade_regra::{Avaliacao, Cen, Estado, Rotulo};
use moe_rt::weights::Checkpoint;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const VARIAVEL: &str = "PRIMEIRA_PALAVRA_AUTORIZADA";
/// O UUID da RTX 3080 Ti combinada para a janela: `CUDA_VISIBLE_DEVICES` tem de ser exatamente isto.
const UUID_3080TI: &str = "GPU-b22d9465-8552-9f97-812e-380b590645f8";
/// O hash do critério aprovado e congelado: um `--criterio` com outro conteúdo recusa antes de abrir a GPU.
const CRITERIO_SHA: &str = "4fa895887a12ed2cdbcc1b132e568fe62c46b37c231b7158c33947ba042c5a4b";
const FRACAO_SLOTS: f64 = 0.10;
const CRITERIO_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-5b-primeira-palavra.txt";
const PROMPTS_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/tools/ref-texts/prompts-traco.json";
const REFERENCIA_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/5b-referencia";
const TEXTOS_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/5b-textos";
const CRITERIO_RESET_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-reset-sequencia-20261108.md";
/// Os cenários distintos da prova de reset (os textos congelados do 5b).
const CRITERIO_5C_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-5c-geracao-livre-20261108.md";
const TEXTOS_5C_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/5c-textos";
const REFERENCIA_5C_PADRAO: &str = "/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/5c-referencia";
/// sha256 das referências da 5c (`ref5c-<modelo>.json`): congelados DEPOIS de rodar `tools/ref_geracao_5c.py` e antes da janela. Com `None` a prova
/// recusa (sem sha congelado não há referência confiável); o ensaio mostra o sha atual.
const REF5C_SHA_1B: Option<&str> = Some("9584fcabe7d8b2a1821207267651504c72d317ce9b566cd967d390a1dbbad259");
const REF5C_SHA_3B: Option<&str> = Some("968d9b391324a99fbaf5c772a351018b90105f572b32ee2db3140f79da123124");
const ORDEM_RESET_CENARIOS: [Cen; 2] = [Cen::C, Cen::A];
const CENARIOS: [Cen; 2] = [Cen::C, Cen::A];

/// Os hashes esperados dos arquivos de entrada, pelo nome do arquivo (congelados no critério, seção 6).
fn sha_esperado(nome: &str) -> Option<&'static str> {
    match nome {
        "tokenizer.json" => Some("ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654"),
        "prompts-traco.json" => Some("e5e53c33edaa9c5c972a88ddab44f7de0b5bfef0d5be204c97527191b770ac3c"),
        "pedido-C.txt" => Some("e80bce7b40f94aa2c86cb5f2e8345f7230982711e7ec0441654df8c273157b25"),
        "pedido-A.txt" => Some("4ab72d751f8f7e67c6c93513f235ad239f0cb54c21e391dc664bc71d62a54d44"),
        "texto-curto.txt" => Some("5b6fad6f3825d430bad4f924d227105032581906739d68c726ac8e4827df2d4d"),
        "texto-medio.txt" => Some("bb0a067ae8839616f33b21ba41b362bdf16353a57d70321252a2d47851f1c32a"),
        "texto-longo.txt" => Some("821e0f15139b58173b5c17f5e9428527bb27b03dc10b35e38cbeed044defbaea"),
        "ref5c-1b.json" => REF5C_SHA_1B,
        "ref5c-3b.json" => REF5C_SHA_3B,
        "ref-1b.json" => Some("787675e72f135dc8d6072b0c60971fbcdce6ff8616698596f1fb3c26d568d59b"),
        "ref-3b.json" => Some("7425c1d8e21be1c37f66a0f950bdc46a48737321e959a3f081a20c9534d8e35f"),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Modo {
    Livre,
    Correcao,
    Velocidade,
    /// `--prova-reset`: a PROVA 2 do critério de reset de sequência (braço A x braço B).
    ProvaReset,
    /// `--gera`: a etapa 5c (geração livre curta); com `--texto` é o uso normal, sem rótulo.
    Gera,
}

struct Args {
    modo: Option<Modo>,
    estado: Option<Estado>,
    celulas: Vec<(String, PathBuf)>,
    saida: PathBuf,
    criterio: PathBuf,
    prompts: PathBuf,
    referencia: PathBuf,
    textos: PathBuf,
    threads: usize,
    ensaio: bool,
    cenario: Option<Cen>,
    rep: Option<usize>,
    consolida: Option<PathBuf>,
    modelos: Vec<String>,
    texto: Option<String>,
    pedido: Option<(PathBuf, usize)>,
    data: Option<(i32, u32, u32)>,
    /// `--reutiliza`: a lista de pedidos da correção roda num único motor, com `nova_sequencia` entre os pedidos (padrão: um motor novo por pedido).
    reutiliza: bool,
    criterio_reset: PathBuf,
    criterio_5c: PathBuf,
    textos_5c: PathBuf,
    referencia_5c: PathBuf,
    /// `--novos` (uso normal do `--gera`): tokens novos no máximo.
    novos: Option<usize>,
    /// `--so-texto` (prova `--gera`): roda só esse texto (curto, medio ou longo), em motor novo (medida FRIA).
    so_texto: Option<String>,
    /// `--repete N` (com `--so-texto`): 1 pedido de aquecimento descartado + N medidos no mesmo motor reutilizado (medida QUENTE).
    repete: Option<usize>,
    /// `--fracao-slots F` (só `--gera`): fração dos experts (camadas x experts) residentes na GPU; padrão 0,10 (comportamento atual).
    fracao: f64,
    /// `--camadas-cpu N` (só `--gera`, exclusiva com `--fracao-slots`; degrau 2): os experts das primeiras N camadas ficam só na CPU e os das camadas N.. todos na GPU.
    camadas_cpu: Option<usize>,
}

/// Limite de `--repete` (é medida de velocidade, não laço aberto).
const REPETE_MAX: usize = 50;

fn nome_valido(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn canonico(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

fn parse_data(s: &str) -> Result<(i32, u32, u32), String> {
    let p: Vec<&str> = s.split('-').collect();
    if p.len() != 3 {
        return Err(format!("--data AAAA-MM-DD, não {s}"));
    }
    let (a, m, d) = (p[0].parse::<i32>(), p[1].parse::<u32>(), p[2].parse::<u32>());
    match (a, m, d) {
        (Ok(a), Ok(m), Ok(d)) if formata_data(a, m, d).is_ok() => Ok((a, m, d)),
        _ => Err(format!("--data inválida: {s}")),
    }
}

/// Converte os argumentos do sistema em texto; um argumento que não é UTF-8 é recusado com erro tipado (sem panic).
fn args_utf8(it: impl Iterator<Item = std::ffi::OsString>) -> Result<Vec<String>, String> {
    it.map(|a| a.into_string().map_err(|o| format!("argumento que não é UTF-8: {o:?}"))).collect()
}

fn parse_args(v: Vec<String>) -> Result<Args, String> {
    let mut a = v.into_iter();
    let mut r = Args {
        modo: None, estado: None, celulas: vec![], saida: PathBuf::new(), criterio: PathBuf::from(CRITERIO_PADRAO), prompts: PathBuf::from(PROMPTS_PADRAO),
        referencia: PathBuf::from(REFERENCIA_PADRAO), textos: PathBuf::from(TEXTOS_PADRAO), threads: 8, ensaio: false, cenario: None, rep: None, consolida: None,
        modelos: vec!["1b".into(), "3b".into()], texto: None, pedido: None, data: None, reutiliza: false,
        criterio_reset: PathBuf::from(CRITERIO_RESET_PADRAO),
        criterio_5c: PathBuf::from(CRITERIO_5C_PADRAO), textos_5c: PathBuf::from(TEXTOS_5C_PADRAO), referencia_5c: PathBuf::from(REFERENCIA_5C_PADRAO), novos: None, so_texto: None, repete: None, fracao: FRACAO_SLOTS, camadas_cpu: None,
    };
    let mut modelos_dados = false;
    let mut fracao_dada = false;
    let mut prova_reset = false;
    let mut gera = false;
    while let Some(k) = a.next() {
        let mut val = || a.next().ok_or_else(|| format!("{k} pede um valor"));
        match k.as_str() {
            "--modo" => {
                r.modo = Some(match val()?.as_str() {
                    "livre" => Modo::Livre,
                    "correcao" => Modo::Correcao,
                    "velocidade" => Modo::Velocidade,
                    o => return Err(format!("--modo livre, correcao ou velocidade, não {o}")),
                })
            }
            "--celula" => {
                let x = val()?;
                let p: Vec<&str> = x.splitn(2, ':').collect();
                if p.len() != 2 || p.iter().any(|s| s.is_empty()) {
                    return Err(format!("--celula nome:modelo, não {x}"));
                }
                if !nome_valido(p[0]) || fontes_de_ids(p[0]).is_none() {
                    return Err(format!("célula {}: o nome tem de ser 1b ou 3b", p[0]));
                }
                r.celulas.push((p[0].to_string(), PathBuf::from(p[1])));
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
            "--prompts" => r.prompts = PathBuf::from(val()?),
            "--referencia" => r.referencia = PathBuf::from(val()?),
            "--textos" => r.textos = PathBuf::from(val()?),
            "--consolida" => r.consolida = Some(PathBuf::from(val()?)),
            "--modelos" => {
                r.modelos = val()?.split(',').map(|s| s.to_string()).collect();
                modelos_dados = true;
            }
            "--threads" => r.threads = val()?.parse().map_err(|_| "--threads".to_string())?,
            "--texto" => r.texto = Some(val()?),
            "--pedido" => {
                let x = val()?;
                let (f, i) = x.rsplit_once(':').ok_or("--pedido arquivo.json:N")?;
                r.pedido = Some((PathBuf::from(f), i.parse().map_err(|_| "--pedido arquivo.json:N".to_string())?));
            }
            "--data" => r.data = Some(parse_data(&val()?)?),
            "--ensaio" => r.ensaio = true,
            "--reutiliza" => r.reutiliza = true,
            "--prova-reset" => prova_reset = true,
            "--criterio-reset" => r.criterio_reset = PathBuf::from(val()?),
            "--gera" => gera = true,
            "--criterio-5c" => r.criterio_5c = PathBuf::from(val()?),
            "--textos-5c" => r.textos_5c = PathBuf::from(val()?),
            "--referencia-5c" => r.referencia_5c = PathBuf::from(val()?),
            "--novos" => {
                let n: usize = val()?.parse().map_err(|_| "--novos".to_string())?;
                if !(1..=1024).contains(&n) {
                    return Err("--novos de 1 a 1024".into());
                }
                r.novos = Some(n);
            }
            "--so-texto" => {
                let t = val()?;
                if !gr::PEDIDOS.contains(&t.as_str()) {
                    return Err(format!("--so-texto curto, medio ou longo, não {t}"));
                }
                r.so_texto = Some(t);
            }
            "--fracao-slots" => {
                let f: f64 = val()?.parse().map_err(|_| "--fracao-slots".to_string())?;
                if !(f.is_finite() && f > 0.0 && f <= 1.0) {
                    return Err("--fracao-slots em (0, 1]".into());
                }
                r.fracao = f;
                fracao_dada = true;
            }
            "--camadas-cpu" => {
                let n: usize = val()?.parse().map_err(|_| "--camadas-cpu".to_string())?;
                r.camadas_cpu = Some(n);
            }
            "--repete" => {
                let n: usize = val()?.parse().map_err(|_| "--repete".to_string())?;
                if !(1..=REPETE_MAX).contains(&n) {
                    return Err(format!("--repete de 1 a {REPETE_MAX}"));
                }
                r.repete = Some(n);
            }
            _ => return Err(format!("opção desconhecida: {k}")),
        }
    }
    if r.modelos.is_empty() || r.modelos.iter().any(|m| fontes_de_ids(m).is_none()) || r.modelos.iter().collect::<BTreeSet<_>>().len() != r.modelos.len() {
        return Err("--modelos: lista de 1b e 3b, sem repetir".into());
    }
    if gera && prova_reset {
        return Err("--gera e --prova-reset são modos diferentes".into());
    }
    let modo = if gera {
        if r.modo.is_some() {
            return Err("--gera já é um modo: não combina com --modo".into());
        }
        r.modo = Some(Modo::Gera);
        Modo::Gera
    } else if prova_reset {
        if r.modo.is_some() {
            return Err("--prova-reset já é um modo: não combina com --modo".into());
        }
        r.modo = Some(Modo::ProvaReset);
        Modo::ProvaReset
    } else {
        r.modo.ok_or("--modo é obrigatório")?
    };
    if modo == Modo::ProvaReset && r.consolida.is_some() {
        return Err("--prova-reset não tem --consolida".into());
    }
    if modo == Modo::Gera && r.consolida.is_some() {
        return Err("--gera não tem --consolida".into());
    }
    if modo != Modo::Gera && r.novos.is_some() {
        return Err("--novos é só do --gera".into());
    }
    if modo != Modo::Gera && (r.so_texto.is_some() || r.repete.is_some()) {
        return Err("--so-texto e --repete são só do --gera".into());
    }
    if modo != Modo::Gera && fracao_dada {
        return Err("--fracao-slots é só do --gera".into());
    }
    if modo != Modo::Gera && r.camadas_cpu.is_some() {
        return Err("--camadas-cpu é só do --gera".into());
    }
    if r.camadas_cpu.is_some() && fracao_dada {
        return Err("--camadas-cpu e --fracao-slots são exclusivas".into());
    }
    if r.repete.is_some() && r.so_texto.is_none() {
        return Err("--repete exige --so-texto".into());
    }
    if r.texto.is_some() && r.so_texto.is_some() {
        return Err("--so-texto é da prova --gera (com --saida ou --ensaio), não do uso normal com --texto".into());
    }
    if r.consolida.is_some() {
        if modo == Modo::Livre {
            return Err("--consolida é de correcao ou velocidade".into());
        }
        if !r.celulas.is_empty() || !r.saida.as_os_str().is_empty() || r.cenario.is_some() || r.rep.is_some() || r.ensaio || r.texto.is_some() || r.pedido.is_some() || r.data.is_some() {
            return Err("--consolida só aceita --modo, --estado, --modelos e --criterio".into());
        }
        if (modo == Modo::Velocidade) != r.estado.is_some() {
            return Err("--consolida de velocidade pede --estado; a correção não aceita --estado".into());
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
    if r.celulas.len() > 2 || (r.celulas.len() == 2 && r.celulas[0].0 != "1b") {
        return Err("no máximo duas células, 1b antes de 3b".into());
    }
    if r.reutiliza && modo != Modo::Correcao {
        return Err("--reutiliza é só do modo correcao".into());
    }
    match modo {
        Modo::Livre => {
            if r.celulas.len() != 1 || r.estado.is_some() || r.cenario.is_some() || r.rep.is_some() || !r.saida.as_os_str().is_empty() {
                return Err("o modo livre aceita uma --celula e só --texto, --pedido e --data".into());
            }
            if r.texto.is_some() && r.pedido.is_some() {
                return Err("--texto e --pedido juntos".into());
            }
        }
        Modo::Gera => {
            if r.estado.is_some() || r.cenario.is_some() || r.rep.is_some() || r.reutiliza || r.pedido.is_some() {
                return Err("--gera não tem --estado, --cenario, --rep, --reutiliza nem --pedido".into());
            }
            if r.texto.is_some() {
                // Uso normal: um texto, uma célula, sem rótulo e sem saída gravada.
                if r.celulas.len() != 1 || !r.saida.as_os_str().is_empty() || r.ensaio {
                    return Err("--gera --texto (uso normal) aceita uma --celula, sem --saida nem --ensaio".into());
                }
            } else {
                if r.data.is_some() || r.novos.is_some() {
                    return Err("a prova --gera tem data e 64 tokens fixos pelo critério (--data e --novos são do uso normal, com --texto)".into());
                }
                if r.saida.as_os_str().is_empty() && !r.ensaio {
                    return Err("--saida é obrigatória".into());
                }
            }
        }
        Modo::ProvaReset => {
            if r.texto.is_some() || r.pedido.is_some() || r.data.is_some() || r.estado.is_some() || r.cenario.is_some() || r.rep.is_some() || r.reutiliza {
                return Err("--prova-reset tem texto, data e pedidos fixos pelo critério (sem --texto, --pedido, --data, --estado, --cenario, --rep nem --reutiliza)".into());
            }
            if r.saida.as_os_str().is_empty() && !r.ensaio {
                return Err("--saida é obrigatória".into());
            }
        }
        Modo::Correcao | Modo::Velocidade => {
            if r.texto.is_some() || r.pedido.is_some() || r.data.is_some() {
                return Err("nos modos de prova o texto e a data são fixos pelo critério (sem --texto, --pedido nem --data)".into());
            }
            if r.saida.as_os_str().is_empty() && !r.ensaio {
                return Err("--saida é obrigatória".into());
            }
            if modo == Modo::Correcao {
                if r.estado.is_some() || r.cenario.is_some() || r.rep.is_some() {
                    return Err("a correção não tem --estado, --cenario nem --rep".into());
                }
            } else {
                match r.estado.ok_or("--estado quente ou frio é obrigatório na velocidade")? {
                    Estado::Quente => {
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
            }
        }
    }
    Ok(r)
}

/// Quente e correção: a pasta de saída tem de ser nova ou vazia.
fn prepara_saida(p: &Path) -> Result<(), String> {
    if p.exists() && std::fs::read_dir(p).map_err(|e| format!("--saida {}: {e}", p.display()))?.next().is_some() {
        return Err(format!("--saida {} já existe e não está vazia", p.display()));
    }
    std::fs::create_dir_all(p).map_err(|e| format!("--saida {}: {e}", p.display()))
}

/// Frio: a pasta é compartilhada pelos 12 processos. Só aceita o que o próprio estado frio produz e recusa um resultado que já exista.
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
        _ => Err(format!("a primeira palavra na GPU só roda com o 'autorizo' explícito do Breno ({VARIAVEL}=1); veja o critério em criterio-5b-primeira-palavra.txt")),
    }
}

/// O UUID da GPU: `CUDA_VISIBLE_DEVICES` tem de ser exatamente o da 3080 Ti combinada.
fn exige_uuid(valor: Option<&str>) -> Result<(), String> {
    match valor {
        Some(UUID_3080TI) => Ok(()),
        outro => Err(format!("CUDA_VISIBLE_DEVICES tem de ser {UUID_3080TI} (a 3080 Ti combinada); veio {outro:?}")),
    }
}

fn portoes() -> Result<(), String> {
    exige_autorizacao(std::env::var(VARIAVEL).ok().as_deref())?;
    exige_uuid(std::env::var("CUDA_VISIBLE_DEVICES").ok().as_deref())
}

/// As chaves da tabela congelada: `slots` chaves espaçadas por igual no espaço camada·E + expert (a divisão mista fixa das etapas 3 e 4).
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

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn sha256(p: &Path) -> Result<String, String> {
    let o = std::process::Command::new("sha256sum").arg(p).output().map_err(|e| format!("sha256sum: {e}"))?;
    if !o.status.success() {
        return Err(format!("sha256sum falhou em {}", p.display()));
    }
    Ok(String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string())
}

fn nome_do_arquivo(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()
}

/// Os arquivos de entrada de cada modo (todos têm hash esperado fixo): tokenizer.json de cada modelo, mais prompts e referências (correção) ou os
/// textos dos cenários (velocidade). Frio só usa o cenário pedido.
fn arquivos_de_entrada(args: &Args, modo: Modo) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = args.celulas.iter().map(|c| c.1.join("tokenizer.json")).collect();
    match modo {
        Modo::Livre => {}
        Modo::ProvaReset => {
            v.extend(ORDEM_RESET_CENARIOS.iter().map(|c| arquivo_pedido(&args.textos, *c)));
        }
        Modo::Gera => {
            if args.texto.is_none() {
                v.extend(gr::PEDIDOS.iter().map(|n| args.textos_5c.join(format!("texto-{n}.txt"))));
                v.extend(args.celulas.iter().map(|c| args.referencia_5c.join(format!("ref5c-{}.json", c.0))));
            }
        }
        Modo::Correcao => {
            v.push(args.prompts.clone());
            v.extend(args.celulas.iter().map(|c| args.referencia.join(format!("ref-{}.json", c.0))));
        }
        Modo::Velocidade => {
            let cens: Vec<Cen> = args.cenario.map(|c| vec![c]).unwrap_or_else(|| CENARIOS.to_vec());
            v.extend(cens.iter().map(|c| arquivo_pedido(&args.textos, *c)));
        }
    }
    v.sort();
    v.dedup();
    v
}

fn arquivo_pedido(textos: &Path, cen: Cen) -> PathBuf {
    textos.join(format!("pedido-{}.txt", cen.nome()))
}

/// Calcula o hash de cada arquivo e confere com o esperado do critério (pelo nome). Divergência recusa antes de qualquer leitura de texto.
fn hashes_conferidos(arqs: &[PathBuf]) -> Result<Vec<(PathBuf, String)>, String> {
    arqs.iter()
        .map(|a| {
            let h = sha256(a)?;
            let esp = sha_esperado(&nome_do_arquivo(a)).ok_or_else(|| format!("{}: sem hash esperado no critério", a.display()))?;
            if h != esp {
                return Err(format!("{}: sha256 {h}, o congelado é {esp}", a.display()));
            }
            Ok((a.clone(), h))
        })
        .collect()
}

fn hashes_iguais(h0: &[(PathBuf, String)]) -> bool {
    h0.iter().all(|(p, h)| sha256(p).ok().as_deref() == Some(h.as_str()))
}

/// Integridade num instante: binário, critério e arquivos de entrada contra os iniciais.
fn integro_agora(exe: &Path, criterio: &Path, h_bin0: &str, h_crit0: &str, arq0: &[(PathBuf, String)]) -> bool {
    sha256(exe).ok().as_deref() == Some(h_bin0) && sha256(criterio).ok().as_deref() == Some(h_crit0) && hashes_iguais(arq0)
}

fn extras_de_integridade(h_bin0: &str, h_crit0: &str, arq0: &[(PathBuf, String)]) -> Vec<(String, String)> {
    let mut v = vec![("hash_bin".to_string(), h_bin0.to_string()), ("hash_criterio".to_string(), h_crit0.to_string())];
    v.extend(arq0.iter().map(|(p, h)| (format!("arq:{}", p.display()), h.clone())));
    v
}

// ---- Montagem do motor real e do fluxo ----------------------------------------------------------------------------------------------------

struct Cenario<'a> {
    device: &'a Arc<Device>,
    host: &'a HostModel,
    cfg: &'a ModelConfig,
    threads: usize,
    fracao: f64,
    camadas_cpu: Option<usize>,
}

/// Linha única por processo do `--gera` (stderr): `slots {slots} de {camadas*experts}`; o extrator do degrau 1 lê esta linha (100% = os dois números iguais).
fn linha_slots(slots: usize, total: usize) -> String {
    format!("slots {slots} de {total}")
}

impl Cenario<'_> {
    fn slots_10(&self) -> usize {
        match self.camadas_cpu {
            Some(n) => slots_camadas(self.cfg.layers, self.cfg.experts, n).unwrap_or(0),
            None => ((self.fracao * (self.cfg.layers * self.cfg.experts) as f64) as usize).max(1),
        }
    }
    /// Chaves residentes: padrão = tabela espaçada; `--camadas-cpu N` = todas as chaves das camadas N...
    fn chaves(&self) -> Vec<u32> {
        match self.camadas_cpu {
            Some(n) => chaves_camadas(self.cfg.layers, self.cfg.experts, n),
            None => chaves_congeladas(self.cfg.layers * self.cfg.experts, self.slots_10()),
        }
    }
}

/// Degrau 2: slots = (camadas - N) x experts; N de 0 a camadas-1 (ao menos uma camada na GPU).
fn slots_camadas(layers: usize, experts: usize, n: usize) -> Result<usize, String> {
    if n >= layers {
        return Err(format!("--camadas-cpu {n}: de 0 a {} (camadas {layers})", layers.saturating_sub(1)));
    }
    Ok((layers - n) * experts)
}

/// Chaves camada*E+expert das camadas N.. (todas residentes na GPU; as N primeiras só na CPU, como `--n-cpu-moe N`).
fn chaves_camadas(layers: usize, experts: usize, n: usize) -> Vec<u32> {
    ((n * experts)..(layers * experts)).map(|k| k as u32).collect()
}

/// Linha única do `--camadas-cpu` (stderr): `camadas-cpu {n} de {camadas}; slots {slots} de {total}`; o extrator do degrau 2 lê esta linha.
fn linha_camadas(n: usize, layers: usize, slots: usize, total: usize) -> String {
    format!("camadas-cpu {n} de {layers}; slots {slots} de {total}")
}

/// Motor e backend NOVOS (tabela congelada de 10%, `backlog = 0`, admissão Event, max_len 2.048, blocos de 8), preparo do leitor e a leitura dos
/// logits na stream de cálculo do motor. É o `T_sequencia` do produto.
fn sequencia_real(c: &Cenario) -> Result<Sequencia<CudaBackend, LeitorCuda>, String> {
    let slots = c.slots_10();
    let (compute, copy) = (CudaStream(c.device.new_stream().d()?), CudaStream(c.device.new_stream().d()?));
    let (b, ex) = CudaBackend::new_model_positions(c.device.clone(), c.host, slots, MAX_LEN, &compute, BLOCO).d()?;
    let m = c.cfg;
    let prefill = c.chaves();
    let mcfg = Config { layers: m.layers, experts: m.experts, topk: m.k, width: m.hidden, slots, backlog: 0, admit: AdmitMode::Event, first_token: 1, prefill, max_positions: BLOCO };
    let parts = b.parts(compute.clone(), copy, b.weights(&ex).d()?, mcfg.first_token).d()?;
    let cpu = CpuServe { cpu: Cpu::new(c.threads), hidden: m.hidden, inter: m.inter, experts: m.experts, weights: ex.clone() };
    let eng = Engine::new(b.clone(), parts, mcfg, cpu).d()?;
    // A stream do adaptador é a MESMA do motor.
    let prep = LeitorCuda::novo(b.clone(), compute.clone()).d()?;
    let le_logits: Box<dyn FnMut() -> Result<Vec<f32>, String>> = Box::new(move || b.read_logits(&compute).d());
    Ok(Sequencia { eng, prep, le_logits })
}

/// Um pedido completo (texto cru -> primeira palavra entregue) com motor e backend novos; a palavra sai na saída padrão.
fn pedido_real(c: &Cenario, tok: &Tokenizador, texto: &str, t_entrada: Instant, entrada_ms: f64, data: &str) -> Result<pp::Resultado, String> {
    pp::primeira_palavra(t_entrada, entrada_ms, texto, data, tok, || sequencia_real(c), &mut std::io::stdout())
}

/// Como `pedido_real`, mas no motor guardado em `r` (criado no primeiro pedido, ou de novo depois de uma falha).
fn pedido_reutilizando(c: &Cenario, tok: &Tokenizador, texto: &str, t_entrada: Instant, entrada_ms: f64, data: &str,
                       r: &mut pp::SequenciaReutilizavel<CudaBackend, LeitorCuda>) -> Result<pp::Resultado, String> {
    pp::primeira_palavra_reutilizando(t_entrada, entrada_ms, texto, data, tok, r, || sequencia_real(c), &mut std::io::stdout())
}

/// Lê o arquivo de texto de um cenário e devolve (texto, instante em que ficou disponível, T_entrada em ms).
fn le_texto_de_arquivo(p: &Path) -> Result<(String, Instant, f64), String> {
    let t0 = Instant::now();
    let t = std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?;
    let t1 = Instant::now();
    Ok((t, t1, ms(t1.duration_since(t0))))
}

/// Quantos ids tem o pedido já renderizado (a conta que o critério confere contra a faixa).
fn conta_ids(tok: &Tokenizador, texto: &str, data: &str) -> Result<usize, String> {
    let r = renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: texto.to_string() }], true, data).map_err(|e| e.to_string())?;
    Ok(tok.codifica(&r).len())
}

fn carrega_config(mdir: &Path) -> Result<ModelConfig, String> {
    ModelConfig::from_json(&serde_json::from_slice(&std::fs::read(mdir.join("config.json")).d()?).d()?).d()
}

fn carrega_tokenizador(mdir: &Path) -> Result<Tokenizador, String> {
    Tokenizador::carrega(&mdir.join("tokenizer.json")).map_err(|e| e.to_string())
}

fn carrega_host(mdir: &Path, cfg: &ModelConfig) -> Result<HostModel, String> {
    HostModel::load(&Checkpoint::open(mdir).d()?, cfg.clone()).d()
}

// ---- Correção ----------------------------------------------------------------------------------------------------------------------------

fn le_prompts(p: &Path) -> Result<Vec<String>, String> {
    let v: Value = serde_json::from_slice(&std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| e.to_string())?;
    let l: Vec<String> = v.as_array().ok_or("prompts: não é uma lista")?.iter().map(|x| x.as_str().map(str::to_string).ok_or("prompt que não é texto")).collect::<Result<_, _>>()?;
    if l.len() < pp::PEDIDOS_CORRECAO {
        return Err(format!("{} prompts, pelo menos {} são necessários", l.len(), pp::PEDIDOS_CORRECAO));
    }
    Ok(l[..pp::PEDIDOS_CORRECAO].to_vec())
}

fn texto_correcao(modelo: &str, a: &pp::AvalCorrecao, final_: RotuloCorrecao) -> String {
    let mut t = format!("== correção {modelo}\n");
    for l in &a.linhas {
        t += &format!("{l}\n");
    }
    if a.rotulo != RotuloCorrecao::Passa {
        t += &format!("motivo(s): {}\n", a.motivos.join("; "));
    }
    t += &format!("rótulo: {}\n", final_.nome());
    t
}

/// Roda os pedidos em ordem e preenche o registro. Um erro de execução em um pedido NÃO apaga os já concluídos: o pedido com erro e os
/// restantes ficam como SEM PROVA (falha no caso) e o laço para. A REPROVA válida de um pedido anterior continua valendo (critério, seção 4).
fn executa_pedidos(rec: &mut RegistroCorrecao, prompts: &[String], refs: &[RefCaso], mut pedido: impl FnMut(usize, &str) -> Result<pp::Resultado, String>) {
    let mut parou: Option<String> = None;
    for (i, p) in prompts.iter().enumerate() {
        let referencia: Option<RefCaso> = refs.get(i).cloned();
        if let Some(e) = &parou {
            rec.casos.push(CasoCorrecao { indice: i, prompt: p.clone(), ids: vec![], medida: None, falha: Some(format!("não executado: {e}")), referencia });
            continue;
        }
        match pedido(i, p) {
            Ok(r) => rec.casos.push(CasoCorrecao { indice: i, prompt: p.clone(), ids: r.ids, medida: Some(r.medida), falha: None, referencia }),
            Err(e) => {
                rec.casos.push(CasoCorrecao { indice: i, prompt: p.clone(), ids: vec![], medida: None, falha: Some(e.clone()), referencia });
                parou = Some(format!("erro no pedido {i}: {e}"));
            }
        }
    }
    if let Some(e) = parou {
        rec.falha = Some(e);
    }
}

/// Os 3 pedidos de um modelo. Falha antes de qualquer pedido (config, pesos) deixa o registro sem casos (SEM PROVA).
fn correcao_modelo(nome: &str, mdir: &Path, args: &Args, prompts: &[String], device: &Arc<Device>) -> RegistroCorrecao {
    let mut rec = RegistroCorrecao { modelo: nome.into(), integro: true, falha: None, casos: vec![], extras: vec![] };
    let r: Result<(), String> = (|| {
        let cfg = carrega_config(mdir)?;
        let tok = carrega_tokenizador(mdir)?;
        let refs_json: Value = serde_json::from_slice(&std::fs::read(args.referencia.join(format!("ref-{nome}.json"))).d()?).d()?;
        let refs = pp::le_referencia(&refs_json)?;
        let host = carrega_host(mdir, &cfg)?;
        let c = Cenario { device, host: &host, cfg: &cfg, threads: args.threads, fracao: args.fracao, camadas_cpu: args.camadas_cpu };
        let data = pp::data_fixa_texto();
        let mut guardada = pp::SequenciaReutilizavel::nova();
        executa_pedidos(&mut rec, prompts, &refs, |i, p| {
            let r = if args.reutiliza { pedido_reutilizando(&c, &tok, p, Instant::now(), 0.0, &data, &mut guardada)? } else { pedido_real(&c, &tok, p, Instant::now(), 0.0, &data)? };
            println!("{nome} pedido {i}: {} ids, token {} {:?}, produto {:.1} ms", r.medida.ids_n, r.medida.token, r.medida.palavra, r.medida.produto_ms);
            Ok(r)
        });
        guardada.encerra()?;
        Ok(())
    })();
    if let Err(e) = r {
        rec.falha = Some(e);
    }
    rec
}

fn correcao(args: &Args, exe: &Path) -> Result<i32, String> {
    portoes()?;
    let (h_bin0, h_crit0) = (sha256(exe)?, sha256(&args.criterio)?);
    if h_crit0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_crit0}, não o congelado {CRITERIO_SHA}", args.criterio.display()));
    }
    let arq0 = hashes_conferidos(&arquivos_de_entrada(args, Modo::Correcao))?;
    let prompts = le_prompts(&args.prompts)?;
    prepara_saida(&args.saida)?;
    std::fs::write(args.saida.join("decisao.txt"), "em execução: sem decisão\n").map_err(|e| e.to_string())?;
    println!("integridade antes: binário {h_bin0}; critério {h_crit0}; {} arquivos de entrada", arq0.len());
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let device = Device::new(0).map_err(|e| format!("{e:?}"))?;
    let extras = extras_de_integridade(&h_bin0, &h_crit0, &arq0);
    let mut integro = true;
    for (nome, mdir) in &args.celulas {
        let mut rec = correcao_modelo(nome, mdir, args, &prompts, &device);
        integro &= integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &arq0);
        rec.integro = integro;
        rec.extras.extend(extras.iter().cloned());
        let a = pp::avalia_correcao(&rec);
        let txt = format!("{}integridade conferida no fim do modelo: {}\n", texto_correcao(nome, &a, pp::rotulo_final_correcao(a.rotulo, integro)), if integro { "íntegra" } else { "QUEBRADA" });
        println!("{txt}");
        grava_json(&args.saida.join(format!("correcao-{nome}.json")), &rec.para_json())?;
        std::fs::write(args.saida.join(format!("correcao-{nome}-parcial.txt")), txt).map_err(|e| e.to_string())?;
    }
    integro &= integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &arq0);
    std::fs::write(args.saida.join("correcao-final.txt"), format!("tipo=correcao-final\nintegro={}\n", integro as u8)).map_err(|e| e.to_string())?;
    let modelos: Vec<String> = args.celulas.iter().map(|c| c.0.clone()).collect();
    let ctx = Contexto { exe: exe.to_path_buf(), criterio: args.criterio.clone(), sha_criterio: CRITERIO_SHA.into(), confere_esperado: true };
    let (dec, _) = consolida_correcao(&args.saida, &modelos, &ctx)?;
    println!("{dec}");
    Ok(0)
}

// ---- Prova 2 do reset de sequência ---------------------------------------------------------------------------------------------------

/// Os dois braços de um modelo: A = motor e backend NOVOS por pedido; B = um motor reutilizado com `nova_sequencia` entre os pedidos (o braço A roda
/// inteiro antes do B, nunca dois motores vivos). Falha antes de qualquer pedido (config, pesos, texto fora da faixa) deixa o registro sem pedidos.
fn reset_modelo(nome: &str, mdir: &Path, args: &Args, device: &Arc<Device>) -> rr::RegistroReset {
    let mut rec = rr::RegistroReset { modelo: nome.into(), integro: true, falha: None, pedidos: vec![], criacoes_a: 0, criacoes_b: 0, extras: vec![] };
    let r: Result<(), String> = (|| {
        let cfg = carrega_config(mdir)?;
        let tok = carrega_tokenizador(mdir)?;
        let data = pp::data_fixa_texto();
        for cen in ORDEM_RESET_CENARIOS {
            let (t, _, _) = le_texto_de_arquivo(&arquivo_pedido(&args.textos, cen))?;
            let n = conta_ids(&tok, &t, &data)?;
            if !pp::ids_na_faixa(n, cen) {
                return Err(format!("cenário {}: {n} ids fora da faixa de 2% em torno de {}", cen.nome(), cen.n()));
            }
        }
        let host = carrega_host(mdir, &cfg)?;
        let c = Cenario { device, host: &host, cfg: &cfg, threads: args.threads, fracao: args.fracao, camadas_cpu: args.camadas_cpu };
        let mut guardada = pp::SequenciaReutilizavel::nova();
        let criou_a = std::cell::Cell::new(0usize);
        rec.pedidos = rr::executa_prova(
            |i, cen| {
                let (texto, t_entrada, entrada_ms) = le_texto_de_arquivo(&arquivo_pedido(&args.textos, cen))?;
                criou_a.set(criou_a.get() + 1);
                let r = pedido_real(&c, &tok, &texto, t_entrada, entrada_ms, &data)?;
                println!("{nome} braço A pedido {i} ({}): {} ids, token {}, T_sequencia {:.3} ms", cen.nome(), r.medida.ids_n, r.medida.token, r.medida.crono.sequencia_ms);
                Ok(r)
            },
            |i, cen| {
                let (texto, t_entrada, entrada_ms) = le_texto_de_arquivo(&arquivo_pedido(&args.textos, cen))?;
                let r = pedido_reutilizando(&c, &tok, &texto, t_entrada, entrada_ms, &data, &mut guardada)?;
                println!("{nome} braço B pedido {i} ({}): {} ids, token {}, T_sequencia {:.3} ms", cen.nome(), r.medida.ids_n, r.medida.token, r.medida.crono.sequencia_ms);
                Ok(r)
            },
        );
        rec.criacoes_a = criou_a.get();
        rec.criacoes_b = guardada.criacoes;
        guardada.encerra().map_err(|e| format!("encerramento do motor reutilizado: {e}"))
    })();
    if let Err(e) = r {
        rec.falha = Some(e);
    }
    rec
}

fn prova_reset(args: &Args, exe: &Path) -> Result<i32, String> {
    portoes()?;
    let (h_bin0, h_crit0, h_5b0) = (sha256(exe)?, sha256(&args.criterio_reset)?, sha256(&args.criterio)?);
    if h_crit0 != rr::CRITERIO_RESET_SHA {
        return Err(format!("o critério de reset {} tem sha256 {h_crit0}, não o congelado {}", args.criterio_reset.display(), rr::CRITERIO_RESET_SHA));
    }
    if h_5b0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_5b0}, não o congelado {CRITERIO_SHA}", args.criterio.display()));
    }
    let arq0 = hashes_conferidos(&arquivos_de_entrada(args, Modo::ProvaReset))?;
    prepara_saida(&args.saida)?;
    std::fs::write(args.saida.join("decisao.txt"), "em execução: sem decisão\n").map_err(|e| e.to_string())?;
    println!("integridade antes: binário {h_bin0}; critério de reset {h_crit0}; critério 5b {h_5b0}; {} arquivos de entrada", arq0.len());
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let device = Device::new(0).map_err(|e| format!("{e:?}"))?;
    let mut extras = extras_de_integridade(&h_bin0, &h_crit0, &arq0);
    extras.push(("hash_criterio_5b".into(), h_5b0.clone()));
    let agora = |integro_antes: bool| integro_antes && integro_agora(exe, &args.criterio_reset, &h_bin0, &h_crit0, &arq0) && sha256(&args.criterio).ok().as_deref() == Some(h_5b0.as_str());
    let mut integro = true;
    let mut rotulos = vec![];
    let mut linhas = vec![];
    for (nome, mdir) in &args.celulas {
        let mut rec = reset_modelo(nome, mdir, args, &device);
        integro = agora(integro);
        rec.integro = integro;
        rec.extras.extend(extras.iter().cloned());
        let a = rr::avalia(&rec);
        let final_ = rr::rotulo_final(a.rotulo, integro);
        let txt = format!("{}integridade conferida no fim do modelo: {}\n", rr::texto(&rec, final_), if integro { "íntegra" } else { "QUEBRADA" });
        println!("{txt}");
        grava_json(&args.saida.join(format!("reset-{nome}.json")), &rec.para_json())?;
        std::fs::write(args.saida.join(format!("reset-{nome}-parcial.txt")), &txt).map_err(|e| e.to_string())?;
        rotulos.push(a.rotulo);
        linhas.push(format!("{nome}: {}", final_.nome()));
    }
    integro = agora(integro);
    std::fs::write(args.saida.join("reset-final.txt"), format!("tipo=reset-final\nintegro={}\n", integro as u8)).map_err(|e| e.to_string())?;
    let conj = rr::rotulo_conjunto(&rotulos, integro);
    let dec = format!("prova 2 do reset de sequência (critério sha256 {})\n{}\nintegridade do conjunto: {}\nrótulo: {}\n", rr::CRITERIO_RESET_SHA, linhas.join("\n"), if integro { "íntegra" } else { "QUEBRADA: todos SEM PROVA" }, conj.nome());
    std::fs::write(args.saida.join("decisao.txt"), &dec).map_err(|e| e.to_string())?;
    println!("{dec}");
    Ok(0)
}

// ---- Etapa 5c: geração livre curta --------------------------------------------------------------------------------------------------

type MotorG = gr::MotorGeracao<CudaBackend, LeitorCuda>;

/// Um pedido da 5c: texto cru -> template -> ids -> laço guloso no motor guardado (criado no primeiro pedido, `nova_sequencia` nos outros).
fn pedido_gera(c: &Cenario, tok: &Tokenizador, texto: &str, t_entrada: Instant, data: &str, r: &mut MotorG, cfg: gr::CfgGera) -> Result<gr::Geracao, String> {
    gr::gera(t_entrada, texto, data, tok, r, || sequencia_real(c), cfg, &mut std::io::stdout())
}

/// Os 3 pedidos de um modelo num único motor. Falha antes dos pedidos (config, pesos, texto que não cabe) deixa o registro sem pedidos; o
/// primeiro erro de pedido para o laço e os seguintes ficam como erro ("não executado").
fn gera_modelo(nome: &str, mdir: &Path, args: &Args, device: &Arc<Device>) -> (gr::RegistroGeracao, Vec<gr::MedidaGera>) {
    let mut medidas: Vec<gr::MedidaGera> = vec![];
    let mut rec = gr::RegistroGeracao { modelo: nome.into(), integro: true, falha: None, pedidos: vec![], criacoes: 0, extras: vec![] };
    let r: Result<(), String> = (|| {
        let cfg = carrega_config(mdir)?;
        let tok = carrega_tokenizador(mdir)?;
        let data = pp::data_fixa_texto();
        let refs = std::fs::read(args.referencia_5c.join(format!("ref5c-{nome}.json")))
            .map_err(|e| e.to_string())
            .and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string()))
            .and_then(|v| gr::le_referencia(&v));
        if let Err(e) = &refs {
            rec.extras.push(("referencia_ilegivel".into(), e.clone()));
        }
        let refs = refs.unwrap_or_default();
        let mut textos = vec![];
        for (idx, n) in gr::PEDIDOS.into_iter().enumerate() {
            if args.so_texto.as_deref().is_some_and(|s| s != n) {
                continue;
            }
            let arq = args.textos_5c.join(format!("texto-{n}.txt"));
            let (t, _, _) = le_texto_de_arquivo(&arq)?;
            let ids = conta_ids(&tok, &t, &data)?;
            if ids + gr::N_NOVOS > MAX_LEN {
                return Err(format!("texto {n}: {ids} ids mais {} novos não cabem em {MAX_LEN}", gr::N_NOVOS));
            }
            textos.push((idx, n, t, sha256(&arq)?));
        }
        let host = carrega_host(mdir, &cfg)?;
        let c = Cenario { device, host: &host, cfg: &cfg, threads: args.threads, fracao: args.fracao, camadas_cpu: args.camadas_cpu };
        match args.camadas_cpu {
            Some(n) => {
                slots_camadas(cfg.layers, cfg.experts, n)?;
                eprintln!("{}", linha_camadas(n, cfg.layers, c.slots_10(), cfg.layers * cfg.experts));
            }
            None => eprintln!("{}", linha_slots(c.slots_10(), cfg.layers * cfg.experts)),
        }
        let mut guardado: MotorG = gr::MotorGeracao::novo();
        let mut parou: Option<String> = None;
        let t_repete = textos.first().map(|x| x.2.clone());
        // Com `--repete` o pedido do laço é só o aquecimento (descartado); as medidas quentes vêm depois, sem entrar no registro de prova.
        let textos_laco = if args.repete.is_some() { vec![] } else { textos };
        for (i, n, t, sha) in textos_laco.into_iter() {
            let geracao = match &parou {
                Some(e) => Err(format!("não executado: {e}")),
                None => {
                    let r = pedido_gera(&c, &tok, &t, Instant::now(), &data, &mut guardado, gr::CfgGera::prova());
                    match &r {
                        Ok(g) => println!("{nome} pedido {i} ({n}): {} ids, {} novos, prefill {:.1} ms, {:.2} ms por token (mediana), total {:.1} ms, T_sequencia {:.3} ms{}", g.ids.len(), g.gerados.len(), g.prefill_ms, g.ms_por_token_mediana(), g.total_ms, g.sequencia_ms, sufixo_media(args, g)),
                        Err(e) => {
                            println!("{nome} pedido {i} ({n}): ERRO {e}");
                            parou = Some(format!("erro no pedido {i}: {e}"));
                        }
                    }
                    r
                }
            };
            if args.so_texto.is_some() {
                medidas.push(gr::MedidaGera { rotulo: "frio".into(), descartada: false, geracao: geracao.clone() });
            }
            rec.pedidos.push(gr::PedidoGeracao { indice: i, nome: n.to_string(), texto_sha256: sha, geracao, referencia: refs.get(i).cloned() });
        }
        if let (Some(reps), Some(t)) = (args.repete, t_repete) {
            let n = args.so_texto.as_deref().unwrap_or("");
            let idx = gr::PEDIDOS.iter().position(|x| *x == n).unwrap_or(0);
            let mut saida = std::io::stdout();
            let serie = gr::gera_serie(&t, &data, &tok, &mut guardado, || sequencia_real(&c), reps, &mut saida, &mut |m| match &m.geracao {
                Ok(g) if m.descartada => println!("{nome} pedido {idx} ({n}): {} ids, {} novos, prefill {:.1} ms, {:.2} ms por token (mediana), total {:.1} ms, T_sequencia {:.3} ms{}", g.ids.len(), g.gerados.len(), g.prefill_ms, g.ms_por_token_mediana(), g.total_ms, g.sequencia_ms, sufixo_media(args, g)),
                Ok(g) => println!("{nome} {} ({n}): {} ids, {} novos, prefill {:.1} ms, {:.2} ms por token (mediana), total {:.1} ms, T_sequencia {:.3} ms{}", m.rotulo, g.ids.len(), g.gerados.len(), g.prefill_ms, g.ms_por_token_mediana(), g.total_ms, g.sequencia_ms, sufixo_media(args, g)),
                Err(e) => println!("{nome} {} ({n}): ERRO {e}", if m.descartada { format!("pedido {idx}") } else { m.rotulo.clone() }),
            });
            if let Some(m) = serie.iter().find(|m| m.geracao.is_err()) {
                let e = m.geracao.as_ref().err().cloned().unwrap_or_default();
                rec.falha = Some(if m.descartada { format!("erro no aquecimento: {e}") } else { format!("erro na repetição quente {}: {e}", m.rotulo.trim_start_matches("quente rep ")) });
            }
            medidas.extend(serie);
        }
        rec.criacoes = guardado.criacoes;
        guardado.encerra().map_err(|e| format!("encerramento do motor reutilizado: {e}"))
    })();
    if let Err(e) = r {
        rec.falha = Some(e);
    }
    (rec, medidas)
}

/// Com `--so-texto`/`--repete` a linha ganha a média por passo (soma dos passo_ms / nº de passos); sem eles a saída é a de antes.
fn sufixo_media(args: &Args, g: &gr::Geracao) -> String {
    if args.so_texto.is_some() || args.repete.is_some() {
        format!(", média por passo {:.3} ms ({} passos)", g.media_por_passo_ms(), g.passo_ms.len())
    } else {
        String::new()
    }
}

fn gera_prova(args: &Args, exe: &Path) -> Result<i32, String> {
    portoes()?;
    let (h_bin0, h_crit0, h_5b0) = (sha256(exe)?, sha256(&args.criterio_5c)?, sha256(&args.criterio)?);
    if h_crit0 != gr::CRITERIO_5C_SHA {
        return Err(format!("o critério 5c {} tem sha256 {h_crit0}, não o congelado {}", args.criterio_5c.display(), gr::CRITERIO_5C_SHA));
    }
    if h_5b0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_5b0}, não o congelado {CRITERIO_SHA}", args.criterio.display()));
    }
    let arq0 = hashes_conferidos(&arquivos_de_entrada(args, Modo::Gera))?;
    prepara_saida(&args.saida)?;
    std::fs::write(args.saida.join("decisao.txt"), "em execução: sem decisão\n").map_err(|e| e.to_string())?;
    println!("integridade antes: binário {h_bin0}; critério 5c {h_crit0}; critério 5b {h_5b0}; {} arquivos de entrada", arq0.len());
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let device = Device::new(0).map_err(|e| format!("{e:?}"))?;
    let mut extras = extras_de_integridade(&h_bin0, &h_crit0, &arq0);
    extras.push(("hash_criterio_5b".into(), h_5b0.clone()));
    let agora = |antes: bool| antes && integro_agora(exe, &args.criterio_5c, &h_bin0, &h_crit0, &arq0) && sha256(&args.criterio).ok().as_deref() == Some(h_5b0.as_str());
    let mut integro = true;
    let mut rotulos = vec![];
    let mut linhas = vec![];
    for (nome, mdir) in &args.celulas {
        let (mut rec, medidas) = gera_modelo(nome, mdir, args, &device);
        if let Some(t) = &args.so_texto {
            grava_json(&args.saida.join(format!("medidas-{nome}.json")), &gr::medidas_json(nome, t, &medidas))?;
        }
        integro = agora(integro);
        rec.integro = integro;
        rec.extras.extend(extras.iter().cloned());
        let a = gr::avalia(&rec);
        let final_ = gr::rotulo_final(a.rotulo, integro);
        let txt = format!("{}integridade conferida no fim do modelo: {}\n", gr::texto(&rec, final_), if integro { "íntegra" } else { "QUEBRADA" });
        println!("{txt}");
        grava_json(&args.saida.join(format!("gera-{nome}.json")), &rec.para_json())?;
        std::fs::write(args.saida.join(format!("gera-{nome}-parcial.txt")), &txt).map_err(|e| e.to_string())?;
        rotulos.push(a.rotulo);
        linhas.push((nome.to_string(), final_));
    }
    integro = agora(integro);
    std::fs::write(args.saida.join("gera-final.txt"), format!("tipo=gera-final\nintegro={}\n", integro as u8)).map_err(|e| e.to_string())?;
    let (dec, _conj) = gr::decisao_conjunto(&linhas, &rotulos, integro);
    std::fs::write(args.saida.join("decisao.txt"), &dec).map_err(|e| e.to_string())?;
    println!("{dec}");
    Ok(0)
}

/// Uso normal: um texto, até `--novos` tokens (padrão 64), para no primeiro EOS e imprime o texto. Não decide rótulo nem vale como prova.
fn gera_livre(args: &Args) -> Result<i32, String> {
    portoes()?;
    let (_, mdir) = &args.celulas[0];
    let texto = args.texto.clone().ok_or("sem --texto")?;
    if texto.trim().is_empty() {
        return Err("texto vazio".into());
    }
    let tok_sha = sha256(&mdir.join("tokenizer.json"))?;
    if sha_esperado("tokenizer.json") != Some(tok_sha.as_str()) {
        return Err(format!("tokenizer.json com sha256 {tok_sha}, não o do Granite 3.1 congelado"));
    }
    let (a, m, d) = args.data.unwrap_or_else(pp::data_de_hoje_utc);
    let data = formata_data(a, m, d).map_err(|e| e.to_string())?;
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let cfg = carrega_config(mdir)?;
    let tok = carrega_tokenizador(mdir)?;
    let device = Device::new(0).d()?;
    let host = carrega_host(mdir, &cfg)?;
    let c = Cenario { device: &device, host: &host, cfg: &cfg, threads: args.threads, fracao: args.fracao, camadas_cpu: args.camadas_cpu };
    let mut m: MotorG = gr::MotorGeracao::novo();
    let g = pedido_gera(&c, &tok, &texto, Instant::now(), &data, &mut m, gr::CfgGera { novos: args.novos.unwrap_or(gr::N_NOVOS), para_no_eos: true });
    m.encerra().ok();
    let g = g?;
    eprintln!("({} ids, {} novos, prefill {:.1} ms, {:.2} ms por token (mediana), total {:.1} ms; sem valor de prova)", g.ids.len(), g.gerados.len(), g.prefill_ms, g.ms_por_token_mediana(), g.total_ms);
    Ok(0)
}

/// Ensaio da 5c (sem GPU, sem a variável): hashes (a referência sem sha congelado só mostra o atual), ids do Rust contra os da referência, critérios.
fn ensaio_gera(args: &Args, exe: &Path) -> Result<(), String> {
    for p in arquivos_de_entrada(args, Modo::Gera) {
        let h = sha256(&p).unwrap_or_else(|e| e);
        let nome = nome_do_arquivo(&p);
        let estado = match sha_esperado(&nome) {
            Some(e) if e == h => "CONFERE com o congelado",
            Some(_) => "DIFERE do congelado",
            None => "SEM sha congelado (a prova recusa até congelar)",
        };
        println!("hash {estado}: {} {h}", p.display());
    }
    let data = pp::data_fixa_texto();
    for (nome, mdir) in &args.celulas {
        let tok = carrega_tokenizador(mdir)?;
        let refs: Result<Vec<gr::RefGeracao>, String> = std::fs::read(args.referencia_5c.join(format!("ref5c-{nome}.json"))).map_err(|e| e.to_string()).and_then(|b| serde_json::from_slice::<Value>(&b).map_err(|e| e.to_string())).and_then(|v| gr::le_referencia(&v));
        for (i, n) in gr::PEDIDOS.iter().enumerate() {
            let arq = args.textos_5c.join(format!("texto-{n}.txt"));
            let t = std::fs::read_to_string(&arq).map_err(|e| format!("{}: {e}", arq.display()))?;
            let ids: Vec<usize> = tok.codifica(&renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: t }], true, &data).map_err(|e| e.to_string())?).into_iter().map(|x| x as usize).collect();
            let cabe = ids.len() + gr::N_NOVOS <= MAX_LEN;
            let vs_ref = match &refs {
                Ok(r) => r.get(i).map(|r| if r.ids == ids { "IGUAIS à referência" } else { "DIFERENTES da referência" }).unwrap_or("referência sem este pedido"),
                Err(_) => "referência ausente ou ilegível",
            };
            println!("ensaio {nome} pedido {i} ({n}): {} ids pelo Rust ({} com {} novos em {MAX_LEN}), {vs_ref}", ids.len(), if cabe { "CABE" } else { "NÃO CABE" }, gr::N_NOVOS);
        }
    }
    println!("ensaio: ordem {}, {} tokens novos, motor reutilizado (nova_sequencia), a prova não para no EOS", gr::PEDIDOS.join(", "), gr::N_NOVOS);
    let hc = sha256(&args.criterio_5c).unwrap_or_else(|e| e);
    println!("critério 5c {} ({hc}; {} o congelado); binário {} ({})", args.criterio_5c.display(), if hc == gr::CRITERIO_5C_SHA { "CONFERE com" } else { "DIFERE d" }, exe.display(), sha256(exe).unwrap_or_else(|e| e));
    Ok(())
}

fn grava_json(p: &Path, v: &Value) -> Result<(), String> {
    std::fs::write(p, serde_json::to_string_pretty(v).map_err(|e| e.to_string())? + "\n").map_err(|e| format!("{}: {e}", p.display()))
}

// ---- Velocidade quente --------------------------------------------------------------------------------------------------------------------

fn texto_rep(r: &RepMedida) -> String {
    let c = &r.crono;
    format!("produto {:.1} ms (entrada {:.2}, token {:.2}, sequência {:.1}, leitura {:.1}, logits {:.1}, escolha {:.3}, detok {:.3}, entrega {:.3}); token {} {:?}; contadores {:?}", r.produto_ms, c.entrada_ms, c.token_ms, c.sequencia_ms, c.leitura_ms, c.logits_ms, c.escolha_ms, c.detok_ms, c.entrega_ms, r.token, r.palavra, r.cont)
}

/// A célula quente: aquecimento descartado e `REPS_QUENTE` repetições. `uma()` faz UMA repetição (motor e backend novos). Para no primeiro erro
/// (célula SEM PROVA). Nada repete.
fn mede_quente<F: FnMut() -> Result<RepMedida, String>>(modelo: &str, cen: Cen, ids_n: usize, mut uma: F) -> RegistroQuente {
    let mut rec = RegistroQuente { modelo: modelo.into(), cen, integro: true, falha: None, ids_n, aquecimento: vec![], reps: vec![], extras: vec![] };
    match uma() {
        Ok(r) => {
            let t = format!("aquecimento descartado, {}", texto_rep(&r));
            println!("{modelo} {} {t}", cen.nome());
            rec.aquecimento.push(t);
        }
        Err(e) => {
            rec.falha = Some(format!("aquecimento: {e}"));
            return rec;
        }
    }
    for i in 1..=pp::REPS_QUENTE {
        match uma() {
            Ok(r) => {
                println!("{modelo} {} repetição {i}/{} {}", cen.nome(), pp::REPS_QUENTE, texto_rep(&r));
                rec.reps.push(r);
            }
            Err(e) => {
                rec.falha = Some(format!("repetição {i}: {e}"));
                return rec;
            }
        }
    }
    rec
}

fn pct_ou(x: Option<f64>) -> String {
    x.map(|v| format!("{:.1} ms ({:.2} s)", v, v / 1000.0)).unwrap_or_else(|| "n/d".into())
}

/// Só descrição: quantas repetições entregaram um primeiro token sem caractere visível (só espaço ou quebra de linha). Não muda rótulo.
fn linha_nao_visivel<'a>(medidas: impl Iterator<Item = &'a RepMedida>) -> String {
    let v: Vec<&RepMedida> = medidas.collect();
    let nv: Vec<&&RepMedida> = v.iter().filter(|m| !m.visivel).collect();
    let tok = nv.first().map(|m| format!(" (token {})", m.token)).unwrap_or_default();
    format!("só descrição, não decide: repetições com primeiro token não visível: {} de {}{tok}\n", nv.len(), v.len())
}

fn texto_quente(modelo: &str, cen: Cen, a: &pp::AvalQuente, nv: &str, final_: Rotulo) -> String {
    let mut t = format!("== célula {modelo} {} quente ({} ids por pedido)\nmeta: {}\n", cen.nome(), cen.n(), pct_ou(a.meta_ms));
    t += &format!("mediana do indicador de produto COM T_sequencia (DECIDE contra a meta): {}\n", pct_ou(a.med_ms));
    t += &format!("só descrição, não decide: mediana SEM T_sequencia (o que seria com reset no Engine) {}; T_sequencia mediano {}; pior repetição {}; desvio {}\n", pct_ou(a.med_sem_seq_ms), pct_ou(a.med_sequencia_ms), pct_ou(a.pior_ms), pct_ou(a.desvio_ms));
    t += nv;
    if a.rotulo != Rotulo::Atinge {
        t += &format!("motivo(s) ({}): {}\n", if a.rotulo == Rotulo::NaoAtinge { "nível 2" } else { "validade, nível 1" }, a.motivos.join("; "));
    }
    t += &format!("rótulo da célula: {}\n", final_.nome());
    t
}

fn texto_frio(modelo: &str, cen: Cen, a: &Avaliacao, nv: &str, final_: Rotulo) -> String {
    let mut t = format!("== célula {modelo} {} frio ({} ids por pedido)\nmeta: {}\n", cen.nome(), cen.n(), pct_ou(a.meta_ms));
    t += &format!("mediana das repetições frias, do início do main até a palavra entregue (DECIDE contra a meta): {}\nsó descrição, não decide: pior repetição {}; desvio {}\n", pct_ou(a.med8_ms), pct_ou(a.pior_ms), pct_ou(a.desvio_ms));
    t += nv;
    if a.rotulo != Rotulo::Atinge {
        t += &format!("motivo(s) ({}): {}\n", if a.rotulo == Rotulo::NaoAtinge { "nível 2" } else { "validade, nível 1" }, a.motivos.join("; "));
    }
    t += &format!("rótulo da célula: {}\n", final_.nome());
    t
}

fn texto_parcial_quente(rec: &RegistroQuente, integro: bool) -> String {
    let a = pp::avalia_registro_quente(&rec.modelo, rec.cen, rec);
    let f = vr::rotulo_final(a.rotulo, integro);
    format!("{}integridade conferida no fim da célula: {}\n", texto_quente(&rec.modelo, rec.cen, &a, &linha_nao_visivel(rec.reps.iter()), f), if integro { "íntegra" } else { "QUEBRADA" })
}

fn quente(args: &Args, exe: &Path) -> Result<i32, String> {
    portoes()?;
    let (h_bin0, h_crit0) = (sha256(exe)?, sha256(&args.criterio)?);
    if h_crit0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_crit0}, não o congelado {CRITERIO_SHA}", args.criterio.display()));
    }
    let arq0 = hashes_conferidos(&arquivos_de_entrada(args, Modo::Velocidade))?;
    prepara_saida(&args.saida)?;
    std::fs::write(args.saida.join("decisao.txt"), "em execução: sem decisão\n").map_err(|e| e.to_string())?;
    println!("integridade antes: binário {h_bin0}; critério {h_crit0}; {} arquivos de entrada", arq0.len());
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let device = Device::new(0).map_err(|e| format!("{e:?}"))?;
    let extras = extras_de_integridade(&h_bin0, &h_crit0, &arq0);
    let data = pp::data_fixa_texto();
    let mut integro = true;
    for (nome, mdir) in &args.celulas {
        let carregado: Result<(ModelConfig, Tokenizador, HostModel), String> = (|| {
            let cfg = carrega_config(mdir)?;
            let tok = carrega_tokenizador(mdir)?;
            let host = carrega_host(mdir, &cfg)?;
            Ok((cfg, tok, host))
        })();
        for cen in CENARIOS {
            let arq = arquivo_pedido(&args.textos, cen);
            let mut rec = match &carregado {
                Ok((cfg, tok, host)) => {
                    let c = Cenario { device: &device, host, cfg, threads: args.threads, fracao: args.fracao, camadas_cpu: args.camadas_cpu };
                    match le_texto_de_arquivo(&arq).and_then(|(t, _, _)| conta_ids(tok, &t, &data)) {
                        Err(e) => RegistroQuente { modelo: nome.clone(), cen, integro: true, falha: Some(e), ids_n: 0, aquecimento: vec![], reps: vec![], extras: vec![] },
                        Ok(n) if !pp::ids_na_faixa(n, cen) => RegistroQuente { modelo: nome.clone(), cen, integro: true, falha: Some(format!("{n} ids fora da faixa de 2% em torno de {}", cen.n())), ids_n: n, aquecimento: vec![], reps: vec![], extras: vec![] },
                        Ok(n) => mede_quente(nome, cen, n, || {
                            let (texto, t_entrada, entrada_ms) = le_texto_de_arquivo(&arq)?;
                            pedido_real(&c, tok, &texto, t_entrada, entrada_ms, &data).map(|r| r.medida)
                        }),
                    }
                }
                Err(e) => RegistroQuente { modelo: nome.clone(), cen, integro: true, falha: Some(format!("carga do modelo: {e}")), ids_n: 0, aquecimento: vec![], reps: vec![], extras: vec![] },
            };
            integro &= integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &arq0);
            rec.integro = integro;
            rec.extras.extend(extras.iter().cloned());
            let txt = texto_parcial_quente(&rec, integro);
            println!("{txt}");
            grava_json(&args.saida.join(format!("quente-{nome}-{}.json", cen.nome())), &rec.para_json())?;
            std::fs::write(args.saida.join(format!("quente-{nome}-{}-parcial.txt", cen.nome())), txt).map_err(|e| e.to_string())?;
        }
    }
    integro &= integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &arq0);
    std::fs::write(args.saida.join("quente-final.txt"), format!("tipo=quente-final\nintegro={}\n", integro as u8)).map_err(|e| e.to_string())?;
    let modelos: Vec<String> = args.celulas.iter().map(|c| c.0.clone()).collect();
    let ctx = Contexto { exe: exe.to_path_buf(), criterio: args.criterio.clone(), sha_criterio: CRITERIO_SHA.into(), confere_esperado: true };
    let (dec, _) = consolida_velocidade(&args.saida, Estado::Quente, &modelos, &ctx)?;
    println!("{dec}");
    Ok(0)
}

// ---- Velocidade fria ----------------------------------------------------------------------------------------------------------------------

/// Uma repetição fria (um processo). `t0` é o primeiro instante do `main`.
fn frio(t0: Instant, args: &Args, exe: &Path) -> Result<i32, String> {
    portoes()?;
    let (nome, mdir) = &args.celulas[0];
    let (cen, rep) = (args.cenario.ok_or("sem --cenario")?, args.rep.ok_or("sem --rep")?);
    let (h_bin0, h_crit0) = (sha256(exe)?, sha256(&args.criterio)?);
    if h_crit0 != CRITERIO_SHA {
        return Err(format!("o critério {} tem sha256 {h_crit0}, não o congelado {CRITERIO_SHA}", args.criterio.display()));
    }
    let arq0 = hashes_conferidos(&arquivos_de_entrada(args, Modo::Velocidade))?;
    let arquivo = prepara_saida_frio(&args.saida, &format!("frio-{nome}-{}-rep{rep}.json", cen.nome()))?;
    let mut partes: Vec<(String, f64)> = vec![("integridade_e_preparo".into(), ms(t0.elapsed()))];
    let mut extras = extras_de_integridade(&h_bin0, &h_crit0, &arq0);
    let mut ids_n = 0usize;
    let mut medida: Option<RepMedida> = None;
    // De agora em diante qualquer falha é GRAVADA (a repetição fica SEM PROVA) e nada se repete.
    let r: Result<f64, String> = (|| {
        moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
        let mut marca = Instant::now();
        let mut passo = |n: &str, partes: &mut Vec<(String, f64)>| {
            partes.push((n.to_string(), ms(marca.elapsed())));
            marca = Instant::now();
        };
        let cfg = carrega_config(mdir)?;
        passo("config", &mut partes);
        let tok = carrega_tokenizador(mdir)?;
        passo("tokenizador", &mut partes);
        let device = Device::new(0).d()?;
        passo("device", &mut partes);
        let host = carrega_host(mdir, &cfg)?;
        passo("pesos_na_ram", &mut partes);
        let (texto, t_entrada, entrada_ms) = le_texto_de_arquivo(&arquivo_pedido(&args.textos, cen))?;
        let data = pp::data_fixa_texto();
        ids_n = conta_ids(&tok, &texto, &data)?;
        if !pp::ids_na_faixa(ids_n, cen) {
            return Err(format!("{ids_n} ids fora da faixa de 2% em torno de {}", cen.n()));
        }
        passo("texto_e_contagem", &mut partes);
        let c = Cenario { device: &device, host: &host, cfg: &cfg, threads: args.threads, fracao: args.fracao, camadas_cpu: args.camadas_cpu };
        let r = pedido_real(&c, &tok, &texto, t_entrada, entrada_ms, &data)?;
        let m = r.medida;
        for (k, v) in [("T_entrada", m.crono.entrada_ms), ("T_token", m.crono.token_ms), ("T_sequencia", m.crono.sequencia_ms), ("T_leitura", m.crono.leitura_ms), ("T_logits", m.crono.logits_ms), ("T_escolha", m.crono.escolha_ms), ("T_detok", m.crono.detok_ms), ("T_entrega", m.crono.entrega_ms)] {
            partes.push((k.to_string(), v));
        }
        extras.push(("contadores".into(), format!("{:?}", m.cont)));
        extras.push(("argmax".into(), m.token.to_string()));
        extras.push(("palavra".into(), format!("{:?}", m.palavra)));
        let total = ms(r.t_fim.duration_since(t0));
        medida = Some(m);
        Ok(total)
    })();
    let integro = integro_agora(exe, &args.criterio, &h_bin0, &h_crit0, &arq0);
    let (total_ms, ok, falha) = match r {
        Ok(t) => (t, true, None),
        Err(e) => (ms(t0.elapsed()), false, Some(e)),
    };
    let rec = RegistroFrioPP { modelo: nome.clone(), cen, rep, total_ms, ok, falha, integro, ids_n, partes, medida, extras };
    let mut f = std::fs::OpenOptions::new().write(true).create_new(true).open(&arquivo).map_err(|e| format!("{}: {e}", arquivo.display()))?;
    std::io::Write::write_all(&mut f, (serde_json::to_string_pretty(&rec.para_json()).map_err(|e| e.to_string())? + "\n").as_bytes()).map_err(|e| e.to_string())?;
    println!("frio {nome} {} rep {rep}: total {:.1} ms ({:.2} s), válido {}, integridade {}{}", cen.nome(), rec.total_ms, rec.total_ms / 1000.0, rec.ok, if integro { "íntegra" } else { "QUEBRADA" }, rec.falha.as_ref().map(|f| format!(", falha: {f}")).unwrap_or_default());
    Ok(if rec.ok && integro { 0 } else { 1 })
}

// ---- Modo livre --------------------------------------------------------------------------------------------------------------------------

fn livre(args: &Args) -> Result<i32, String> {
    portoes()?;
    let (nome, mdir) = &args.celulas[0];
    let _ = nome;
    let t_ini = Instant::now();
    let texto = match (&args.texto, &args.pedido) {
        (Some(t), _) => t.clone(),
        (None, Some((f, i))) => {
            let v: Value = serde_json::from_slice(&std::fs::read(f).map_err(|e| format!("{}: {e}", f.display()))?).map_err(|e| e.to_string())?;
            v.as_array().and_then(|l| l.get(*i)).and_then(Value::as_str).ok_or_else(|| format!("{}: sem pedido {i}", f.display()))?.to_string()
        }
        (None, None) => {
            let mut s = String::new();
            std::io::stdin().read_to_string(&mut s).map_err(|e| format!("entrada padrão: {e}"))?;
            s
        }
    };
    let entrada_ms = ms(Instant::now().duration_since(t_ini));
    if texto.trim().is_empty() {
        return Err("texto vazio".into());
    }
    let tok_sha = sha256(&mdir.join("tokenizer.json"))?;
    if sha_esperado("tokenizer.json") != Some(tok_sha.as_str()) {
        return Err(format!("tokenizer.json com sha256 {tok_sha}, não o do Granite 3.1 congelado"));
    }
    let (a, m, d) = args.data.unwrap_or_else(pp::data_de_hoje_utc);
    let data = formata_data(a, m, d).map_err(|e| e.to_string())?;
    moe_rt::affinity::set_compute_plan(moe_rt::affinity::Afin::A0.compute_plan(args.threads));
    let cfg = carrega_config(mdir)?;
    let tok = carrega_tokenizador(mdir)?;
    let device = Device::new(0).d()?;
    let host = carrega_host(mdir, &cfg)?;
    let c = Cenario { device: &device, host: &host, cfg: &cfg, threads: args.threads, fracao: args.fracao, camadas_cpu: args.camadas_cpu };
    // t_entrada depois da carga em todos os modos (stdin incluso): o produto nunca inclui a carga do modelo. entrada_ms é só a leitura da entrada.
    let t_entrada = Instant::now();
    let r = pedido_real(&c, &tok, &texto, t_entrada, entrada_ms, &data)?;
    eprintln!("({} ids, data {data}, produto {:.1} ms; sem valor de prova)", r.medida.ids_n, r.medida.produto_ms);
    Ok(0)
}

// ---- Consolidação (sem GPU) ---------------------------------------------------------------------------------------------------------------

struct Contexto {
    exe: PathBuf,
    criterio: PathBuf,
    /// O hash esperado do critério (a constante `CRITERIO_SHA` na execução real; outro valor só nos testes).
    sha_criterio: String,
    /// Confere os hashes gravados dos arquivos de entrada com os esperados do critério (sempre na execução real; desligado só nos testes).
    confere_esperado: bool,
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

fn le_json<T>(p: &Path, f: impl Fn(&Value) -> Result<T, String>) -> Lido<T> {
    le_arquivo(p, |t| serde_json::from_str::<Value>(t).map_err(|e| e.to_string()).and_then(|v| f(&v)))
}

/// Confere as integridades do conjunto: todos os brutos concordam com o binário, o critério e os arquivos de entrada atuais e com os esperados.
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
            let mut arqs: BTreeMap<String, String> = BTreeMap::new();
            for ex in extras {
                for (k, v) in ex.iter() {
                    if k == "hash_bin" && *v != hb {
                        ok = false;
                        notas.push(format!("o binário gravou {v}, o atual é {hb}"));
                    } else if k == "hash_criterio" && *v != ctx.sha_criterio {
                        ok = false;
                        notas.push(format!("o critério gravado foi {v}, não o congelado"));
                    } else if let Some(p) = k.strip_prefix("arq:") {
                        if arqs.insert(p.to_string(), v.clone()).is_some_and(|antes| antes != *v) {
                            ok = false;
                            notas.push(format!("hash diferente entre registros: {p}"));
                        }
                    }
                }
            }
            for (p, h) in &arqs {
                if sha256(Path::new(p)).ok().as_deref() != Some(h.as_str()) {
                    ok = false;
                    notas.push(format!("arquivo {p}: o hash atual difere do gravado"));
                }
                if ctx.confere_esperado && sha_esperado(&nome_do_arquivo(Path::new(p))) != Some(h.as_str()) {
                    ok = false;
                    notas.push(format!("arquivo {p}: o hash gravado difere do esperado no critério"));
                }
            }
            if arqs.is_empty() {
                ok = false;
                notas.push("nenhum hash de arquivo de entrada gravado".into());
            }
        }
        _ => {
            ok = false;
            notas.push("não foi possível calcular o hash do binário ou do critério".into());
        }
    }
    ok
}

fn consolida_correcao(dir: &Path, modelos: &[String], ctx: &Contexto) -> Result<(String, RotuloCorrecao), String> {
    let mut notas: Vec<String> = Vec::new();
    let mut avals: Vec<(String, pp::AvalCorrecao)> = Vec::new();
    let mut extras: Vec<Vec<(String, String)>> = Vec::new();
    let mut flags: Vec<bool> = Vec::new();
    for m in modelos {
        let a = match le_json(&dir.join(format!("correcao-{m}.json")), RegistroCorrecao::de_json) {
            Lido::Ok(r) => {
                flags.push(r.integro);
                extras.push(r.extras.clone());
                if r.modelo != *m {
                    pp::AvalCorrecao { rotulo: RotuloCorrecao::SemProva, motivos: vec![format!("identidade do registro ({}) diferente do modelo pedido ({m})", r.modelo)], linhas: vec![] }
                } else {
                    pp::avalia_correcao(&r)
                }
            }
            Lido::Ausente => pp::AvalCorrecao { rotulo: RotuloCorrecao::SemProva, motivos: vec!["modelo ausente: interrompido (por exemplo tempo-limite) ou não rodado".into()], linhas: vec![] },
            Lido::Ilegivel(e) => pp::AvalCorrecao { rotulo: RotuloCorrecao::SemProva, motivos: vec![format!("resultado ilegível: {e}")], linhas: vec![] },
        };
        avals.push((m.clone(), a));
    }
    match le_arquivo(&dir.join("correcao-final.txt"), |t| Ok(t.lines().any(|l| l == "integro=1"))) {
        Lido::Ok(ok) => flags.push(ok),
        Lido::Ausente => notas.push("correcao-final.txt ausente (execução interrompida): vale a integridade conferida modelo a modelo e a conferência agora".into()),
        Lido::Ilegivel(e) => {
            flags.push(false);
            notas.push(e);
        }
    }
    let ex_ref: Vec<&Vec<(String, String)>> = extras.iter().collect();
    let integro = integridade_do_conjunto(ctx, &ex_ref, &flags, &mut notas);
    let mut dec = "decisão da correção da primeira palavra a partir de texto cru (PASSA, REPROVA, SEM PROVA):\n".to_string();
    let mut rotulos = Vec::new();
    for (m, a) in &avals {
        let r = pp::rotulo_final_correcao(a.rotulo, integro);
        rotulos.push(r);
        let rel = format!("{}integridade do conjunto: {}\n", texto_correcao(m, a, r), if integro { "íntegra" } else { "QUEBRADA: todos SEM PROVA" });
        std::fs::write(dir.join(format!("relatorio-correcao-{m}.txt")), &rel).map_err(|e| e.to_string())?;
        dec += &format!("  {m}: {}{}\n", r.nome(), if r == RotuloCorrecao::Passa { String::new() } else if integro { format!(" [{}]", a.motivos.join("; ")) } else { " [integridade quebrada]".into() });
    }
    let conj = pp::rotulo_conjunto_correcao(&rotulos, integro);
    dec += &format!("conjunto: {}\n", conj.nome());
    dec += &format!("integridade: {}\n", if integro { "binário, critério e arquivos de entrada IGUAIS aos gravados e aos congelados" } else { "QUEBRADA OU NÃO CALCULÁVEL: tudo SEM PROVA" });
    for n in &notas {
        dec += &format!("nota: {n}\n");
    }
    dec += "se REPROVA ou SEM PROVA: a velocidade não roda sem nova decisão do Breno\n";
    std::fs::write(dir.join("decisao.txt"), &dec).map_err(|e| e.to_string())?;
    Ok((dec, conj))
}

/// A avaliação de uma célula, quente ou fria, com o que o relatório e o resumo precisam.
enum Celula {
    Q(pp::AvalQuente, String),
    F(Avaliacao, String),
}

impl Celula {
    fn rotulo(&self) -> Rotulo {
        match self {
            Celula::Q(a, _) => a.rotulo,
            Celula::F(a, _) => a.rotulo,
        }
    }
    fn motivos(&self) -> String {
        match self {
            Celula::Q(a, _) => a.motivos.join("; "),
            Celula::F(a, _) => a.motivos.join("; "),
        }
    }
    fn resumo(&self) -> String {
        let s = |x: Option<f64>| x.map(|v| format!("{:.2} s", v / 1000.0)).unwrap_or_else(|| "n/d".into());
        match self {
            Celula::Q(a, _) if a.med_ms.is_some() => format!("mediana com T_sequencia {} (meta {}); sem T_sequencia {}", s(a.med_ms), s(a.meta_ms), s(a.med_sem_seq_ms)),
            Celula::F(a, _) if a.med8_ms.is_some() => format!("mediana {} (meta {})", s(a.med8_ms), s(a.meta_ms)),
            _ => "sem tempo legível".into(),
        }
    }
    fn relatorio(&self, m: &str, cen: Cen, final_: Rotulo) -> String {
        match self {
            Celula::Q(a, nv) => texto_quente(m, cen, a, nv, final_),
            Celula::F(a, nv) => texto_frio(m, cen, a, nv, final_),
        }
    }
}

fn quente_sem_dados(m: &str, cen: Cen, falha: String) -> RegistroQuente {
    RegistroQuente { modelo: m.into(), cen, integro: true, falha: Some(falha), ids_n: 0, aquecimento: vec![], reps: vec![], extras: vec![] }
}

/// Lê os brutos de `dir`, aplica as regras e escreve `decisao.txt` e os relatórios. Devolve o texto da decisão e o rótulo do conjunto.
fn consolida_velocidade(dir: &Path, estado: Estado, modelos: &[String], ctx: &Contexto) -> Result<(String, Rotulo), String> {
    let mut notas: Vec<String> = Vec::new();
    let mut celulas: Vec<(String, Cen, Celula)> = Vec::new();
    let mut extras: Vec<Vec<(String, String)>> = Vec::new();
    let mut flags: Vec<bool> = Vec::new();
    for m in modelos {
        for cen in CENARIOS {
            let c = match estado {
                Estado::Quente => {
                    let (a, nv) = match le_json(&dir.join(format!("quente-{m}-{}.json", cen.nome())), RegistroQuente::de_json) {
                        Lido::Ok(r) => {
                            flags.push(r.integro);
                            extras.push(r.extras.clone());
                            (pp::avalia_registro_quente(m, cen, &r), linha_nao_visivel(r.reps.iter()))
                        }
                        Lido::Ausente => (pp::avalia_registro_quente(m, cen, &quente_sem_dados(m, cen, "célula ausente: interrompida (por exemplo tempo-limite) ou não rodada".into())), String::new()),
                        Lido::Ilegivel(e) => (pp::avalia_registro_quente(m, cen, &quente_sem_dados(m, cen, format!("resultado ilegível: {e}"))), String::new()),
                    };
                    Celula::Q(a, nv)
                }
                Estado::Frio => {
                    let mut regs = Vec::new();
                    for n in 1..=vr::REPS_FRIO {
                        match le_json(&dir.join(format!("frio-{m}-{}-rep{n}.json", cen.nome())), RegistroFrioPP::de_json) {
                            Lido::Ok(r) => {
                                flags.push(r.integro);
                                extras.push(r.extras.clone());
                                regs.push(r);
                            }
                            Lido::Ausente => notas.push(format!("frio {m} {} repetição {n}: ausente (interrompida ou não rodada)", cen.nome())),
                            Lido::Ilegivel(e) => notas.push(format!("frio {m} {} repetição {n}: resultado ilegível, conta como ausente: {e}", cen.nome())),
                        }
                    }
                    let nv = linha_nao_visivel(regs.iter().filter_map(|r| r.medida.as_ref()));
                    Celula::F(pp::avalia_frio_pp(m, cen, &regs), nv)
                }
            };
            celulas.push((m.clone(), cen, c));
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
    let mut dec = format!("decisão da primeira palavra a partir de texto cru, velocidade, estado {} (ATINGE, NÃO ATINGE, SEM PROVA; resultado DESCRITIVO, vai ao Breno):\n", estado.nome());
    let mut rotulos = Vec::new();
    for (m, cen, c) in &celulas {
        let r = vr::rotulo_final(c.rotulo(), integro);
        rotulos.push(r);
        let rel = format!("{}integridade do conjunto: {}\n", c.relatorio(m, *cen, r), if integro { "íntegra" } else { "QUEBRADA: todas as células SEM PROVA" });
        std::fs::write(dir.join(format!("relatorio-{}-{m}-{}.txt", estado.nome(), cen.nome())), &rel).map_err(|e| e.to_string())?;
        let motivo = if r == Rotulo::Atinge { String::new() } else if integro { format!(" [{}]", c.motivos()) } else { " [integridade quebrada]".into() };
        dec += &format!("  {m} {} {}: {}: {}{motivo}\n", cen.nome(), estado.nome(), r.nome(), c.resumo());
    }
    let conj = vr::rotulo_conjunto(&rotulos, integro);
    dec += &format!("conjunto: {}\n", conj.nome());
    dec += &format!("integridade: {}\n", if integro { "binário, critério e arquivos de entrada IGUAIS aos gravados e aos congelados" } else { "QUEBRADA OU NÃO CALCULÁVEL: todas as células SEM PROVA" });
    if estado == Estado::Quente {
        dec += "o quente por pedido INCLUI T_sequencia (o Engine não tem reset); a mediana sem T_sequencia está nos relatórios, só como descrição\n";
    }
    for n in &notas {
        dec += &format!("nota: {n}\n");
    }
    dec += "carga externa (nvidia-smi, loadavg, timers): registrada à parte, antes e depois da janela, pelo script da janela, fora deste programa; não decide rótulo\n";
    std::fs::write(dir.join("decisao.txt"), &dec).map_err(|e| e.to_string())?;
    Ok((dec, conj))
}

// ---- Ensaio e principal -------------------------------------------------------------------------------------------------------------------

fn ensaio(args: &Args, exe: &Path, modo: Modo) -> Result<(), String> {
    println!("ensaio ({modo:?}): nada será aberto na GPU; data do template {}", pp::data_fixa_texto());
    if modo == Modo::Velocidade {
        let estado = args.estado.ok_or("sem --estado")?;
        println!("metas em segundos (1b / 3b): C {}/{}, A {}/{}", vr::meta_ms(estado, "1b", Cen::C).unwrap() / 1000.0, vr::meta_ms(estado, "3b", Cen::C).unwrap() / 1000.0, vr::meta_ms(estado, "1b", Cen::A).unwrap() / 1000.0, vr::meta_ms(estado, "3b", Cen::A).unwrap() / 1000.0);
    }
    if modo == Modo::Gera {
        return ensaio_gera(args, exe);
    }
    let arqs = arquivos_de_entrada(args, modo);
    let hs = hashes_conferidos(&arqs)?;
    for (p, h) in &hs {
        println!("hash CONFERE com o congelado: {} {h}", p.display());
    }
    let data = pp::data_fixa_texto();
    for (nome, mdir) in &args.celulas {
        let tok = carrega_tokenizador(mdir)?;
        match modo {
            Modo::Correcao => {
                let prompts = le_prompts(&args.prompts)?;
                let refs = pp::le_referencia(&serde_json::from_slice(&std::fs::read(args.referencia.join(format!("ref-{nome}.json"))).d()?).d()?)?;
                for (i, p) in prompts.iter().enumerate() {
                    let n = conta_ids(&tok, p, &data)?;
                    let r = refs.get(i).ok_or("referência com menos pedidos")?;
                    println!("ensaio {nome} pedido {i}: {n} ids pelo Rust, {} na referência, ids {}", r.ids.len(), if tok.codifica(&renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: p.clone() }], true, &data).map_err(|e| e.to_string())?).iter().map(|x| *x as usize).eq(r.ids.iter().copied()) { "IGUAIS" } else { "DIFERENTES" });
                }
            }
            Modo::Velocidade => {
                for cen in args.cenario.map(|c| vec![c]).unwrap_or_else(|| CENARIOS.to_vec()) {
                    let t = std::fs::read_to_string(arquivo_pedido(&args.textos, cen)).map_err(|e| e.to_string())?;
                    let n = conta_ids(&tok, &t, &data)?;
                    println!("ensaio {nome} cenário {}: {n} ids (alvo {}, {} da faixa de 2%)", cen.nome(), cen.n(), if pp::ids_na_faixa(n, cen) { "DENTRO" } else { "FORA" });
                }
            }
            Modo::ProvaReset => {
                for cen in ORDEM_RESET_CENARIOS {
                    let t = std::fs::read_to_string(arquivo_pedido(&args.textos, cen)).map_err(|e| e.to_string())?;
                    let n = conta_ids(&tok, &t, &data)?;
                    println!("ensaio {nome} cenário {}: {n} ids (alvo {}, {} da faixa de 2%)", cen.nome(), cen.n(), if pp::ids_na_faixa(n, cen) { "DENTRO" } else { "FORA" });
                }
                println!("ensaio {nome}: ordem {}", rr::ORDEM.iter().map(|c| c.nome()).collect::<Vec<_>>().join(", "));
            }
            Modo::Livre | Modo::Gera => {}
        }
    }
    let hc = sha256(&args.criterio).unwrap_or_else(|e| e);
    println!("critério {} ({hc}; {} o congelado); binário {} ({})", args.criterio.display(), if hc == CRITERIO_SHA { "CONFERE com" } else { "DIFERE d" }, exe.display(), sha256(exe).unwrap_or_else(|e| e));
    Ok(())
}

/// Devolve o código de saída: 0, 1 (gravado com falha) ou, por `Err`, 2 (recusa antes de qualquer trabalho).
fn principal(t0: Instant) -> Result<i32, String> {
    let args = parse_args(args_utf8(std::env::args_os().skip(1))?)?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let modo = args.modo.ok_or("sem --modo")?;
    if let Some(dir) = &args.consolida {
        let ctx = Contexto { exe, criterio: args.criterio.clone(), sha_criterio: CRITERIO_SHA.into(), confere_esperado: true };
        let dec = match modo {
            Modo::Correcao => consolida_correcao(dir, &args.modelos, &ctx)?.0,
            Modo::ProvaReset => return Err("--prova-reset não tem --consolida".into()),
            Modo::Gera => return Err("--gera não tem --consolida".into()),
            _ => consolida_velocidade(dir, args.estado.ok_or("sem --estado")?, &args.modelos, &ctx)?.0,
        };
        println!("{dec}");
        return Ok(0);
    }
    if args.ensaio {
        ensaio(&args, &exe, modo)?;
        return Ok(0);
    }
    match (modo, args.estado) {
        (Modo::Livre, _) => livre(&args),
        (Modo::Correcao, _) => correcao(&args, &exe),
        (Modo::ProvaReset, _) => prova_reset(&args, &exe),
        (Modo::Gera, _) if args.texto.is_some() => gera_livre(&args),
        (Modo::Gera, _) => gera_prova(&args, &exe),
        (Modo::Velocidade, Some(Estado::Quente)) => quente(&args, &exe),
        (Modo::Velocidade, _) => frio(t0, &args, &exe),
    }
}

fn main() {
    // O primeiro instante do processo: o cronômetro frio começa aqui.
    let t0 = Instant::now();
    match principal(t0) {
        Ok(c) => std::process::exit(c),
        Err(e) => {
            eprintln!("primeira-palavra: recusado: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &[&str]) -> Vec<String> {
        s.iter().map(|x| x.to_string()).collect()
    }

    fn pasta(nome: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("primeira-palavra-{nome}-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn reutiliza_e_opcional_e_so_da_correcao() {
        let base = ["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o"];
        assert!(!parse_args(v(&base)).unwrap().reutiliza, "o padrão não reutiliza o motor");
        let mut com = base.to_vec();
        com.push("--reutiliza");
        assert!(parse_args(v(&com)).unwrap().reutiliza);
        assert!(parse_args(v(&["--modo", "livre", "--celula", "1b:/m", "--reutiliza"])).is_err());
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "quente", "--celula", "1b:/m", "--saida", "/o", "--reutiliza"])).is_err());
    }

    #[test]
    fn argumento_que_nao_e_utf8_e_recusado() {
        use std::os::unix::ffi::OsStringExt;
        let ruim = std::ffi::OsString::from_vec(vec![b'-', b'-', 0xff]);
        let e = args_utf8(vec![std::ffi::OsString::from("--modo"), ruim].into_iter()).unwrap_err();
        assert!(e.contains("UTF-8"), "{e}");
        assert_eq!(args_utf8(vec![std::ffi::OsString::from("--modo")].into_iter()).unwrap(), vec!["--modo".to_string()]);
    }

    #[test]
    fn linha_de_primeiro_token_nao_visivel_nos_relatorios() {
        let mut ruim = medida(1000.0);
        ruim.visivel = false;
        ruim.token = 220;
        let l = linha_nao_visivel([medida(1000.0), ruim.clone()].iter());
        assert!(l.contains("não visível: 1 de 2 (token 220)"), "{l}");
        assert!(linha_nao_visivel([medida(1000.0)].iter()).contains("0 de 1"));
        // Consolidação quente: a linha aparece no relatório e o rótulo não muda.
        let d = pasta("consnv");
        let (ctx, extras) = contexto_de_teste(&d);
        for cen in CENARIOS {
            let n = rec_ids(cen);
            let mut k = 0;
            let mut rec = mede_quente("1b", cen, n, || {
                k += 1;
                Ok(RepMedida { ids_n: n, visivel: k != 3, ..medida(1000.0) })
            });
            rec.extras = extras.clone();
            grava_json(&d.join(format!("quente-1b-{}.json", cen.nome())), &rec.para_json()).unwrap();
        }
        std::fs::write(d.join("quente-final.txt"), "integro=1\n").unwrap();
        let (dec, conj) = consolida_velocidade(&d, Estado::Quente, &["1b".into()], &ctx).unwrap();
        assert!(dec.contains("1b C quente: ATINGE"), "{dec}");
        let rel = std::fs::read_to_string(d.join("relatorio-quente-1b-C.txt")).unwrap();
        assert!(rel.contains("repetições com primeiro token não visível: 1 de 5 (token 705)"), "{rel}");
        let _ = conj;
        // Frio: a mesma linha, sem mudar o rótulo.
        let df = pasta("consnvf");
        let (ctxf, extrasf) = contexto_de_teste(&df);
        for rep in 1..=3 {
            let mut m = medida(3000.0);
            m.visivel = rep != 2;
            let r = RegistroFrioPP { modelo: "1b".into(), cen: Cen::C, rep, total_ms: 20_000.0, ok: true, falha: None, integro: true, ids_n: 499, partes: vec![("pesos_na_ram".into(), 1.0)], medida: Some(m), extras: extrasf.clone() };
            grava_json(&df.join(format!("frio-1b-C-rep{rep}.json")), &r.para_json()).unwrap();
        }
        let (decf, _) = consolida_velocidade(&df, Estado::Frio, &["1b".into()], &ctxf).unwrap();
        assert!(decf.contains("1b C frio: ATINGE"), "{decf}");
        let relf = std::fs::read_to_string(df.join("relatorio-frio-1b-C.txt")).unwrap();
        assert!(relf.contains("repetições com primeiro token não visível: 1 de 3 (token 705)"), "{relf}");
        std::fs::remove_dir_all(&d).unwrap();
        std::fs::remove_dir_all(&df).unwrap();
    }

    #[test]
    fn erro_tardio_nao_apaga_reprova_anterior() {
        let mk = |i: usize, token: usize| {
            let r = RefCaso { prompt: format!("p{i}"), ids: vec![1, 2], argmax: 705, top2: 9, gap: 1.0 };
            let mut m = medida(1000.0);
            m.token = token;
            m.ids_n = 2;
            (r.ids.clone(), m)
        };
        let prompts: Vec<String> = (0..3).map(|i| format!("p{i}")).collect();
        let refs: Vec<RefCaso> = (0..3).map(|i| RefCaso { prompt: format!("p{i}"), ids: vec![1, 2], argmax: 705, top2: 9, gap: 1.0 }).collect();
        let res = |ids: Vec<usize>, medida: RepMedida| pp::Resultado { medida, ids, logits: vec![], t_fim: Instant::now() };
        // (a) pedido 0 REPROVA, pedido 1 erro, pedido 2 nunca roda: caminho produtor + consolidador.
        let mut rec = RegistroCorrecao { modelo: "1b".into(), integro: true, falha: None, casos: vec![], extras: vec![] };
        let mut chamadas = 0;
        executa_pedidos(&mut rec, &prompts, &refs, |i, _| {
            chamadas += 1;
            match i {
                0 => {
                    let (ids, m) = mk(0, 5);
                    Ok(res(ids, m))
                }
                _ => Err("erro de GPU".into()),
            }
        });
        assert_eq!(chamadas, 2, "o pedido 2 não roda depois do erro");
        assert_eq!(rec.casos.len(), 3);
        assert!(rec.casos[2].falha.as_deref().unwrap().starts_with("não executado"));
        assert!(rec.falha.is_some());
        let d = pasta("conserr");
        let (ctx, extras) = contexto_de_teste(&d);
        let mut gravado = rec.clone();
        gravado.extras = extras.clone();
        grava_json(&d.join("correcao-1b.json"), &gravado.para_json()).unwrap();
        std::fs::write(d.join("correcao-final.txt"), "integro=1\n").unwrap();
        assert_eq!(pp::avalia_correcao(&rec).rotulo, RotuloCorrecao::Reprova);
        let (dec, conj) = consolida_correcao(&d, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, RotuloCorrecao::Reprova, "{dec}");
        assert!(dec.contains("1b: REPROVA"), "{dec}");
        // (c) integridade quebrada: tudo SEM PROVA.
        std::fs::write(d.join("correcao-final.txt"), "integro=0\n").unwrap();
        let (dec, conj) = consolida_correcao(&d, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, RotuloCorrecao::SemProva, "{dec}");
        assert!(dec.contains("1b: SEM PROVA"), "{dec}");
        // (b) sem reprovação válida, o mesmo erro: SEM PROVA.
        let mut rec2 = RegistroCorrecao { modelo: "1b".into(), integro: true, falha: None, casos: vec![], extras: vec![] };
        executa_pedidos(&mut rec2, &prompts, &refs, |i, _| {
            if i == 0 {
                let (ids, m) = mk(0, 705);
                Ok(res(ids, m))
            } else {
                Err("erro de GPU".into())
            }
        });
        assert_eq!(pp::avalia_correcao(&rec2).rotulo, RotuloCorrecao::SemProva);
        let mut g2 = rec2.clone();
        g2.extras = extras;
        grava_json(&d.join("correcao-1b.json"), &g2.para_json()).unwrap();
        std::fs::write(d.join("correcao-final.txt"), "integro=1\n").unwrap();
        let (dec, conj) = consolida_correcao(&d, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, RotuloCorrecao::SemProva, "{dec}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn portao_proprio_e_uuid_da_3080ti() {
        assert!(exige_autorizacao(None).is_err());
        for x in ["0", "", "true", "2"] {
            assert!(exige_autorizacao(Some(x)).is_err(), "{x}");
        }
        assert!(exige_autorizacao(Some("1")).is_ok());
        assert_eq!(VARIAVEL, "PRIMEIRA_PALAVRA_AUTORIZADA");
        for velha in ["LEITURA_VELOCIDADE_AUTORIZADA", "LEITURA_CORRECAO_AUTORIZADA", "E4B_MEDIDA_AUTORIZADA", "E4B_GPU_AUTORIZADO"] {
            assert_ne!(VARIAVEL, velha, "variável própria: nenhuma guarda antiga abre este programa");
        }
        assert_eq!(UUID_3080TI, "GPU-b22d9465-8552-9f97-812e-380b590645f8");
        assert!(exige_uuid(Some(UUID_3080TI)).is_ok());
        assert!(exige_uuid(None).is_err() && exige_uuid(Some("")).is_err() && exige_uuid(Some("0")).is_err());
        assert!(exige_uuid(Some("GPU-00000000-0000-0000-0000-000000000000")).is_err(), "a 5090 e qualquer outra GPU nunca");
        assert!(exige_uuid(Some("GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd")).is_err(), "a 4090 antiga nunca");
        assert!(exige_uuid(Some(&format!("{UUID_3080TI},0"))).is_err());
    }

    #[test]
    fn argumentos_da_correcao_e_da_velocidade() {
        let c = parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--celula", "3b:/m2", "--saida", "/o"])).unwrap();
        assert!(c.celulas.len() == 2 && c.threads == 8 && c.criterio == PathBuf::from(CRITERIO_PADRAO));
        assert!(parse_args(v(&["--celula", "1b:/m", "--saida", "/o"])).is_err(), "sem modo");
        assert!(parse_args(v(&["--modo", "correcao", "--saida", "/o"])).is_err(), "sem célula");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m"])).is_err(), "sem saída");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--ensaio"])).is_ok(), "ensaio não precisa de saída");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "3b:/m", "--celula", "1b:/m2", "--saida", "/o"])).is_err(), "1b antes de 3b");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "7b:/m", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--celula", "3b:/m", "--saida", "/o"])).is_err(), "mesmo diretório");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--texto", "oi"])).is_err(), "texto é fixo nos modos de prova");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--data", "2026-11-08"])).is_err(), "data é fixa nos modos de prova");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--estado", "quente"])).is_err());
        assert!(parse_args(v(&["--modo", "velocidade", "--celula", "1b:/m", "--saida", "/o"])).is_err(), "velocidade pede estado");
        let q = parse_args(v(&["--modo", "velocidade", "--estado", "quente", "--celula", "1b:/m", "--celula", "3b:/m2", "--saida", "/o"])).unwrap();
        assert_eq!((q.estado, q.celulas.len()), (Some(Estado::Quente), 2));
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "quente", "--celula", "1b:/m", "--saida", "/o", "--cenario", "C"])).is_err(), "cenário é do frio");
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "quente", "--celula", "1b:/m", "--saida", "/o", "--pares", "9"])).is_err(), "parâmetros do critério não se trocam por opção");
        let f = parse_args(v(&["--modo", "velocidade", "--estado", "frio", "--celula", "1b:/m", "--cenario", "A", "--rep", "3", "--saida", "/o"])).unwrap();
        assert_eq!((f.cenario, f.rep), (Some(Cen::A), Some(3)));
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "frio", "--celula", "1b:/m", "--saida", "/o"])).is_err(), "sem cenário e rep");
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "frio", "--celula", "1b:/m", "--cenario", "A", "--rep", "4", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "frio", "--celula", "1b:/m", "--cenario", "B", "--rep", "1", "--saida", "/o"])).is_err(), "cenário B fora");
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "frio", "--celula", "1b:/m", "--celula", "3b:/m2", "--cenario", "A", "--rep", "1", "--saida", "/o"])).is_err());
    }

    #[test]
    fn argumentos_do_livre_e_da_consolidacao() {
        let l = parse_args(v(&["--modo", "livre", "--celula", "1b:/m", "--texto", "oi", "--data", "2026-11-07"])).unwrap();
        assert_eq!((l.texto.as_deref(), l.data), (Some("oi"), Some((2026, 11, 7))));
        assert!(parse_args(v(&["--modo", "livre", "--celula", "1b:/m"])).is_ok(), "sem texto: entrada padrão");
        assert!(parse_args(v(&["--modo", "livre", "--celula", "1b:/m", "--pedido", "/p.json:2"])).unwrap().pedido == Some((PathBuf::from("/p.json"), 2)));
        assert!(parse_args(v(&["--modo", "livre", "--celula", "1b:/m", "--texto", "a", "--pedido", "/p.json:2"])).is_err());
        assert!(parse_args(v(&["--modo", "livre", "--celula", "1b:/m", "--data", "2026-02-30"])).is_err(), "data inválida");
        assert!(parse_args(v(&["--modo", "livre", "--celula", "1b:/m", "--saida", "/o"])).is_err());
        let c = parse_args(v(&["--consolida", "/d", "--modo", "correcao"])).unwrap();
        assert_eq!(c.modelos, v(&["1b", "3b"]));
        assert!(parse_args(v(&["--consolida", "/d", "--modo", "correcao", "--estado", "frio"])).is_err());
        assert!(parse_args(v(&["--consolida", "/d", "--modo", "velocidade"])).is_err(), "velocidade pede estado");
        assert!(parse_args(v(&["--consolida", "/d", "--modo", "velocidade", "--estado", "frio", "--modelos", "1b"])).is_ok());
        assert!(parse_args(v(&["--consolida", "/d", "--modo", "velocidade", "--estado", "frio", "--modelos", "1b,1b"])).is_err());
        assert!(parse_args(v(&["--consolida", "/d", "--modo", "livre"])).is_err());
        assert!(parse_args(v(&["--consolida", "/d", "--modo", "correcao", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--modelos", "1b"])).is_err());
    }

    #[test]
    fn saidas() {
        let d = pasta("saida");
        assert!(prepara_saida(&d).is_ok(), "pasta vazia");
        std::fs::write(d.join("decisao.txt"), "x").unwrap();
        assert!(prepara_saida(&d).is_err());
        let n = d.join("nova");
        let a = prepara_saida_frio(&n, "frio-1b-C-rep1.json").unwrap();
        std::fs::write(&a, "x").unwrap();
        assert!(prepara_saida_frio(&n, "frio-1b-C-rep2.json").is_ok());
        assert!(prepara_saida_frio(&n, "frio-1b-C-rep1.json").is_err(), "a mesma repetição não se repete");
        std::fs::write(n.join("roda-5b-frio.log"), "x").unwrap();
        assert!(prepara_saida_frio(&n, "frio-1b-A-rep1.json").is_ok(), "log do roteiro é aceito");
        std::fs::write(n.join("quente-1b-C.json"), "x").unwrap();
        assert!(prepara_saida_frio(&n, "frio-1b-A-rep1.json").is_err(), "resultado de outro estado");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn hashes_esperados_e_arquivos_de_entrada() {
        let a = parse_args(v(&["--modo", "correcao", "--celula", "1b:/m1", "--celula", "3b:/m3", "--saida", "/o"])).unwrap();
        let nomes: Vec<String> = arquivos_de_entrada(&a, Modo::Correcao).iter().map(|p| nome_do_arquivo(p)).collect();
        assert_eq!(nomes.iter().filter(|n| *n == "tokenizer.json").count(), 2);
        assert!(nomes.contains(&"prompts-traco.json".into()) && nomes.contains(&"ref-1b.json".into()) && nomes.contains(&"ref-3b.json".into()));
        let f = parse_args(v(&["--modo", "velocidade", "--estado", "frio", "--celula", "1b:/m1", "--cenario", "C", "--rep", "1", "--saida", "/o"])).unwrap();
        let nomes: Vec<String> = arquivos_de_entrada(&f, Modo::Velocidade).iter().map(|p| nome_do_arquivo(p)).collect();
        assert!(nomes.contains(&"pedido-C.txt".to_string()) && !nomes.contains(&"pedido-A.txt".to_string()));
        for n in ["tokenizer.json", "prompts-traco.json", "pedido-C.txt", "pedido-A.txt", "ref-1b.json", "ref-3b.json"] {
            assert_eq!(sha_esperado(n).map(str::len), Some(64), "{n}");
        }
        assert!(sha_esperado("outro.txt").is_none());
        // Arquivo com outro conteúdo recusa.
        let d = pasta("hash");
        let p = d.join("pedido-C.txt");
        std::fs::write(&p, "outro conteúdo").unwrap();
        assert!(hashes_conferidos(&[p]).is_err());
        std::fs::remove_dir_all(&d).unwrap();
    }

    fn medida(prod: f64) -> RepMedida {
        use moe_rt::leitura_regra::Contadores;
        RepMedida { crono: pp::Cronometros { entrada_ms: 0.1, token_ms: 1.0, sequencia_ms: 100.0, leitura_ms: 1000.0, logits_ms: 5.0, escolha_ms: 0.1, detok_ms: 0.1, entrega_ms: 0.1 }, produto_ms: prod, ids_n: 499, ids_fnv: 1, token: 705, palavra: "To".into(), visivel: true, gap: 1.0, cont: Contadores { needed: 10, hits: 1, cpu: 9 }, logits_ok: true }
    }

    #[test]
    fn quente_faz_aquecimento_mais_cinco_e_para_no_primeiro_erro() {
        let mut n = 0;
        let r = mede_quente("1b", Cen::C, 499, || {
            n += 1;
            Ok(medida(1000.0 + n as f64))
        });
        assert_eq!((n, r.reps.len(), r.aquecimento.len(), r.falha.clone()), (6, 5, 1, None));
        assert_eq!(r.reps[0].produto_ms, 1002.0, "o aquecimento (1001) não entra nas repetições");
        assert_eq!(pp::avalia_registro_quente("1b", Cen::C, &r).rotulo, Rotulo::Atinge);
        let mut n = 0;
        let r = mede_quente("1b", Cen::C, 499, || {
            n += 1;
            if n == 4 { Err("erro do motor".into()) } else { Ok(medida(1000.0)) }
        });
        assert_eq!((n, r.reps.len()), (4, 2));
        assert!(r.falha.as_deref().unwrap().contains("repetição 3"));
        assert_eq!(pp::avalia_registro_quente("1b", Cen::C, &r).rotulo, Rotulo::SemProva);
        let r = mede_quente("1b", Cen::C, 499, || Err("sem memória".into()));
        assert!(r.falha.as_deref().unwrap().contains("aquecimento") && r.reps.is_empty());
    }

    fn contexto_de_teste(d: &Path) -> (Contexto, Vec<(String, String)>) {
        let criterio = d.join("criterio.txt");
        std::fs::write(&criterio, "critério de teste").unwrap();
        let entrada = d.join("pedido-C.txt");
        std::fs::write(&entrada, "texto").unwrap();
        let exe = std::env::current_exe().unwrap();
        let (hc, hb, he) = (sha256(&criterio).unwrap(), sha256(&exe).unwrap(), sha256(&entrada).unwrap());
        let extras = vec![("hash_bin".to_string(), hb), ("hash_criterio".to_string(), hc.clone()), (format!("arq:{}", entrada.display()), he)];
        (Contexto { exe, criterio, sha_criterio: hc, confere_esperado: false }, extras)
    }

    #[test]
    fn consolidacao_quente_e_integridade() {
        let d = pasta("consq");
        let (ctx, extras) = contexto_de_teste(&d);
        for m in ["1b", "3b"] {
            for cen in CENARIOS {
                let n = rec_ids(cen);
                let mut rec = mede_quente(m, cen, n, || Ok(RepMedida { ids_n: n, ..medida(1000.0) }));
                rec.extras = extras.clone();
                grava_json(&d.join(format!("quente-{m}-{}.json", cen.nome())), &rec.para_json()).unwrap();
            }
        }
        std::fs::write(d.join("quente-final.txt"), "integro=1\n").unwrap();
        let (dec, conj) = consolida_velocidade(&d, Estado::Quente, &["1b".into(), "3b".into()], &ctx).unwrap();
        assert_eq!(conj, Rotulo::Atinge, "{dec}");
        assert!(dec.contains("INCLUI T_sequencia") && dec.contains("1b C quente: ATINGE"));
        assert!(d.join("relatorio-quente-3b-A.txt").exists() && d.join("decisao.txt").exists());
        // Meta estourada em UMA célula: NÃO ATINGE no conjunto (3b A tem meta 30 s: troca por 40 s).
        let mut rec = RegistroQuente::de_json(&serde_json::from_str(&std::fs::read_to_string(d.join("quente-3b-A.json")).unwrap()).unwrap()).unwrap();
        for r in rec.reps.iter_mut() {
            r.produto_ms = 40_000.0;
        }
        grava_json(&d.join("quente-3b-A.json"), &rec.para_json()).unwrap();
        let (dec, conj) = consolida_velocidade(&d, Estado::Quente, &["1b".into(), "3b".into()], &ctx).unwrap();
        assert_eq!(conj, Rotulo::NaoAtinge, "{dec}");
        // Célula ausente: SEM PROVA.
        std::fs::remove_file(d.join("quente-3b-A.json")).unwrap();
        let (_, conj) = consolida_velocidade(&d, Estado::Quente, &["1b".into(), "3b".into()], &ctx).unwrap();
        assert_eq!(conj, Rotulo::SemProva);
        // Critério alterado: tudo SEM PROVA.
        std::fs::write(&ctx.criterio, "mudou").unwrap();
        let (dec, conj) = consolida_velocidade(&d, Estado::Quente, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, Rotulo::SemProva, "{dec}");
        assert!(dec.contains("QUEBRADA"));
        std::fs::remove_dir_all(&d).unwrap();
    }

    fn rec_ids(cen: Cen) -> usize {
        if cen == Cen::C { 499 } else { 2003 }
    }

    #[test]
    fn consolidacao_fria_pela_mediana() {
        let d = pasta("consf");
        let (ctx, extras) = contexto_de_teste(&d);
        for (rep, total) in [(1, 19_000.0), (2, 21_000.0), (3, 20_000.0)] {
            let r = RegistroFrioPP { modelo: "1b".into(), cen: Cen::C, rep, total_ms: total, ok: true, falha: None, integro: true, ids_n: 499, partes: vec![("pesos_na_ram".into(), 1.0)], medida: Some(medida(3000.0)), extras: extras.clone() };
            grava_json(&d.join(format!("frio-1b-C-rep{rep}.json")), &r.para_json()).unwrap();
        }
        let (dec, _) = consolida_velocidade(&d, Estado::Frio, &["1b".into()], &ctx).unwrap();
        assert!(dec.contains("1b C frio: ATINGE"), "{dec}");
        assert!(dec.contains("1b A frio: SEM PROVA"), "A não rodou: {dec}");
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn consolidacao_da_correcao() {
        let d = pasta("consc");
        let (ctx, extras) = contexto_de_teste(&d);
        let mk = |i: usize, token: usize| {
            let r = RefCaso { prompt: format!("p{i}"), ids: vec![1, 2], argmax: 705, top2: 9, gap: 1.0 };
            let mut m = medida(1000.0);
            m.token = token;
            CasoCorrecao { indice: i, prompt: r.prompt.clone(), ids: r.ids.clone(), medida: Some(m), falha: None, referencia: Some(r) }
        };
        let rec = RegistroCorrecao { modelo: "1b".into(), integro: true, falha: None, casos: vec![mk(0, 705), mk(1, 705), mk(2, 705)], extras };
        grava_json(&d.join("correcao-1b.json"), &rec.para_json()).unwrap();
        std::fs::write(d.join("correcao-final.txt"), "integro=1\n").unwrap();
        let (dec, conj) = consolida_correcao(&d, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, RotuloCorrecao::Passa, "{dec}");
        let mut ruim = rec.clone();
        ruim.casos[1] = mk(1, 5);
        grava_json(&d.join("correcao-1b.json"), &ruim.para_json()).unwrap();
        let (dec, conj) = consolida_correcao(&d, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, RotuloCorrecao::Reprova, "{dec}");
        let (_, conj) = consolida_correcao(&d, &["1b".into(), "3b".into()], &ctx).unwrap();
        assert_eq!(conj, RotuloCorrecao::Reprova, "REPROVA vence o 3b ausente (SEM PROVA)");
        std::fs::remove_file(d.join("correcao-1b.json")).unwrap();
        let (_, conj) = consolida_correcao(&d, &["1b".into()], &ctx).unwrap();
        assert_eq!(conj, RotuloCorrecao::SemProva);
        std::fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn prova_reset_argumentos_e_arquivos() {
        let base = ["--prova-reset", "--celula", "1b:/m", "--saida", "/o"];
        let a = parse_args(v(&base)).unwrap();
        assert_eq!(a.modo, Some(Modo::ProvaReset));
        assert_eq!(a.criterio_reset, PathBuf::from(CRITERIO_RESET_PADRAO));
        assert!(parse_args(v(&["--prova-reset", "--celula", "1b:/m", "--ensaio"])).is_ok(), "ensaio não precisa de saída");
        assert!(parse_args(v(&["--prova-reset", "--celula", "1b:/m"])).is_err(), "sem saída");
        assert!(parse_args(v(&["--prova-reset", "--saida", "/o"])).is_err(), "sem célula");
        assert!(parse_args(v(&["--prova-reset", "--modo", "correcao", "--celula", "1b:/m", "--saida", "/o"])).is_err(), "não combina com --modo");
        for extra in [vec!["--texto", "oi"], vec!["--data", "2026-11-08"], vec!["--reutiliza"], vec!["--estado", "quente"], vec!["--cenario", "A"], vec!["--rep", "1"]] {
            let mut x = base.to_vec();
            x.extend(extra);
            assert!(parse_args(v(&x)).is_err(), "{x:?}");
        }
        let mut x = base.to_vec();
        x.extend(["--consolida", "/d"]);
        assert!(parse_args(v(&x)).is_err());
        let c = parse_args(v(&["--prova-reset", "--celula", "1b:/m1", "--celula", "3b:/m3", "--saida", "/o", "--criterio-reset", "/c.md"])).unwrap();
        assert_eq!(c.celulas.len(), 2);
        assert_eq!(c.criterio_reset, PathBuf::from("/c.md"));
        assert!(parse_args(v(&["--prova-reset", "--celula", "3b:/m", "--celula", "1b:/m2", "--saida", "/o"])).is_err(), "1b antes de 3b");
        let nomes: Vec<String> = arquivos_de_entrada(&c, Modo::ProvaReset).iter().map(|p| nome_do_arquivo(p)).collect();
        assert!(nomes.contains(&"pedido-A.txt".into()) && nomes.contains(&"pedido-C.txt".into()) && !nomes.contains(&"prompts-traco.json".into()));
        assert_eq!(nomes.iter().filter(|n| *n == "tokenizer.json").count(), 2);
        // Os portões são os dos outros modos (a recusa vem antes de qualquer acesso ao device).
        assert!(exige_autorizacao(None).is_err() && exige_uuid(Some("outra")).is_err());
        assert_eq!(rr::CRITERIO_RESET_SHA.len(), 64);
    }

    #[test]
    fn linha_slots_formato() {
        assert_eq!(linha_slots(768, 768), "slots 768 de 768");
        assert_eq!(linha_slots(77, 768), "slots 77 de 768");
    }

    #[test]
    fn gera_fracao_slots() {
        let v = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let base = ["--gera", "--celula", "1b:/m", "--so-texto", "curto", "--saida", "/o/p1/saida"];
        let com = |x: &[&str]| parse_args(v(&[&base[..], x].concat()));
        assert_eq!(com(&[]).unwrap().fracao, 0.10, "padrão = comportamento atual");
        assert_eq!(com(&["--fracao-slots", "1.0"]).unwrap().fracao, 1.0);
        assert_eq!(com(&["--fracao-slots", "0.5"]).unwrap().fracao, 0.5);
        for ruim in ["0", "-1", "1.01", "nan", "inf", "x", ""] {
            assert!(com(&["--fracao-slots", ruim]).is_err(), "fracao {ruim:?}");
        }
        assert!(com(&["--fracao-slots"]).is_err(), "sem valor");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--fracao-slots", "1.0"])).is_err());
    }

    #[test]
    fn linha_camadas_formato() {
        assert_eq!(linha_camadas(22, 24, 128, 768), "camadas-cpu 22 de 24; slots 128 de 768");
        assert_eq!(linha_camadas(0, 24, 768, 768), "camadas-cpu 0 de 24; slots 768 de 768");
    }

    #[test]
    fn camadas_cpu_slots_e_chaves() {
        // 1B: 24 camadas x 32 experts; 3B: 32 x 40
        assert_eq!(slots_camadas(24, 32, 22), Ok(64));
        assert_eq!(slots_camadas(32, 40, 29), Ok(120));
        assert_eq!(slots_camadas(24, 32, 0), Ok(768));
        assert_eq!(slots_camadas(24, 32, 23), Ok(32));
        assert!(slots_camadas(24, 32, 24).is_err() && slots_camadas(24, 32, 99).is_err());
        let ch = chaves_camadas(24, 32, 22);
        assert_eq!(ch.len(), 64);
        assert_eq!((ch[0], *ch.last().unwrap()), (22 * 32, 24 * 32 - 1));
        assert!(ch.windows(2).all(|w| w[1] == w[0] + 1), "chaves distintas e contíguas");
        assert_eq!(chaves_camadas(24, 32, 0), (0..768).collect::<Vec<u32>>());
        assert_eq!(chaves_camadas(32, 40, 29).len(), 3 * 40);
        assert!(chaves_camadas(32, 40, 29).iter().all(|&k| k as usize / 40 >= 29));
    }

    #[test]
    fn gera_camadas_cpu_parser() {
        let v = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let base = ["--gera", "--celula", "1b:/m", "--so-texto", "curto", "--saida", "/o/p1/saida"];
        let com = |x: &[&str]| parse_args(v(&[&base[..], x].concat()));
        assert_eq!(com(&[]).unwrap().camadas_cpu, None, "padrão inalterado");
        assert_eq!(com(&["--camadas-cpu", "22"]).unwrap().camadas_cpu, Some(22));
        assert_eq!(com(&["--camadas-cpu", "0"]).unwrap().camadas_cpu, Some(0));
        assert_eq!(com(&["--camadas-cpu", "22", "--repete", "5"]).unwrap().repete, Some(5));
        for ruim in ["-1", "x", "", "1.5"] {
            assert!(com(&["--camadas-cpu", ruim]).is_err(), "camadas {ruim:?}");
        }
        assert!(com(&["--camadas-cpu"]).is_err(), "sem valor");
        assert!(com(&["--camadas-cpu", "22", "--fracao-slots", "1.0"]).is_err(), "exclusiva");
        assert!(com(&["--fracao-slots", "1.0", "--camadas-cpu", "22"]).is_err(), "exclusiva");
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--camadas-cpu", "22"])).is_err());
    }

    #[test]
    fn gera_so_texto_e_repete() {
        let base = ["--gera", "--celula", "1b:/m", "--saida", "/o"];
        let com = |extra: &[&str]| {
            let mut x = base.to_vec();
            x.extend_from_slice(extra);
            parse_args(v(&x))
        };
        let p = com(&[]).unwrap();
        assert_eq!((p.so_texto.clone(), p.repete), (None, None), "padrão = comportamento atual");
        for t in ["curto", "medio", "longo"] {
            assert_eq!(com(&["--so-texto", t]).unwrap().so_texto.as_deref(), Some(t));
        }
        let q = com(&["--so-texto", "medio", "--repete", "5"]).unwrap();
        assert_eq!((q.so_texto.as_deref(), q.repete), (Some("medio"), Some(5)));
        assert_eq!(com(&["--so-texto", "curto", "--repete", &REPETE_MAX.to_string()]).unwrap().repete, Some(REPETE_MAX));
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m", "--ensaio", "--so-texto", "longo"])).is_ok(), "ensaio aceita");
        assert!(com(&["--repete", "5"]).is_err(), "--repete exige --so-texto");
        for ruim in ["0", "-1", "x", "", &(REPETE_MAX + 1).to_string()] {
            assert!(com(&["--so-texto", "curto", "--repete", ruim]).is_err(), "repete {ruim:?}");
        }
        for ruim in ["", "enorme", "Curto", "curto,medio"] {
            assert!(com(&["--so-texto", ruim]).is_err(), "so-texto {ruim:?}");
        }
        assert!(com(&["--so-texto"]).is_err() && com(&["--so-texto", "curto", "--repete"]).is_err(), "sem valor");
        // Só do --gera.
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--so-texto", "curto"])).is_err());
        assert!(parse_args(v(&["--modo", "livre", "--celula", "1b:/m", "--so-texto", "curto"])).is_err());
        assert!(parse_args(v(&["--modo", "velocidade", "--estado", "quente", "--celula", "1b:/m", "--saida", "/o", "--repete", "2"])).is_err());
        assert!(parse_args(v(&["--prova-reset", "--celula", "1b:/m", "--saida", "/o", "--so-texto", "curto"])).is_err());
        // Não do uso normal com --texto.
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m", "--texto", "oi", "--so-texto", "curto"])).is_err());
    }

    /// r111 (rev-1 P1): os argv que `roda-cmp-lastro.sh` monta (frio e quente) têm de ser ACEITOS pelo parser real; a combinação antiga (`--novos`/`--data` na prova
    /// `--gera --so-texto`) é recusada. A linha do roteiro é lida do próprio arquivo, para o teste não divergir dele.
    #[test]
    fn roteiro_cmp_lastro_monta_argv_que_o_parser_real_aceita() {
        let src = include_str!("../../../../qual-runs/A2-LOTE-20261103/roda-cmp-lastro.sh");
        let linha = src.lines().find(|l| l.trim_start().starts_with("\"$BIN\" --gera")).expect("linha de chamada do binário no roteiro").trim_start();
        let esperado = "\"$BIN\" --gera --celula \"$MODELO:$MDIR\" --so-texto \"$TEXTO\" \"${FRACAO_ARGS[@]}\" \"$@\" --saida \"$pd/saida\"";
        // FRACAO_ARGS só recebe `--fracao-slots 1.0` (degrau 1)
        let atrib: Vec<&str> = src.lines().filter(|l| l.contains("FRACAO_ARGS=(") && !l.trim_start().starts_with('#')).collect();
        assert!(atrib.iter().any(|l| l.contains("FRACAO_ARGS=(--fracao-slots 1.0)")), "atribuição de FRACAO_ARGS não achada: {atrib:?}");
        for l in &atrib {
            let r = l.replace("FRACAO_ARGS=();", "").replace("FRACAO_ARGS=(--fracao-slots 1.0)", "").replace("FRACAO_ARGS=(--camadas-cpu \"$NFIXO\")", "");
            assert!(!r.contains("FRACAO_ARGS=("), "FRACAO_ARGS recebe algo além de (--fracao-slots 1.0) ou (--camadas-cpu \"$NFIXO\"): {l}");
        }
        // degrau 2: o roteiro só passa `--camadas-cpu "$NFIXO"` com NFIXO fixo por modelo (22 no 1B, 29 no 3B)
        assert!(atrib.iter().any(|l| l.contains("FRACAO_ARGS=(--camadas-cpu \"$NFIXO\")")), "atribuição de --camadas-cpu não achada: {atrib:?}");
        assert!(src.contains("1b) NFIXO=22;; 3b) NFIXO=29;;"), "valores fixos de N por modelo mudaram");
        assert!(linha.starts_with(esperado), "o roteiro mudou a chamada; ajuste este teste junto: {linha}");
        assert!(!linha.contains("--novos") && !linha.contains("--data"), "o harness fixa 64 novos e a data: o roteiro não pode passá-los");
        let n_max = REPETE_MAX.min(50).to_string();
        for modelo in ["1b", "3b"] {
            for texto in ["curto", "medio", "longo"] {
                // frio: um processo por medida, sem --repete
                let frio = ["--gera", "--celula", &format!("{modelo}:/m"), "--so-texto", texto, "--saida", "/o/p1/saida"];
                let a = parse_args(v(&frio)).unwrap_or_else(|e| panic!("frio {modelo}/{texto} recusado pelo parser real: {e}"));
                assert_eq!((a.so_texto.as_deref(), a.repete, a.data.is_none(), a.novos.is_none()), (Some(texto), None, true, true));
                // degrau 1, frio: `--so-texto T --fracao-slots 1.0 --saida …`
                let frio1 = ["--gera", "--celula", &format!("{modelo}:/m"), "--so-texto", texto, "--fracao-slots", "1.0", "--saida", "/o/p1/saida"];
                let a = parse_args(v(&frio1)).unwrap_or_else(|e| panic!("frio L100 {modelo}/{texto} recusado pelo parser real: {e}"));
                assert_eq!((a.so_texto.as_deref(), a.repete, a.fracao), (Some(texto), None, 1.0));
                // degrau 2, frio: `--so-texto T --camadas-cpu N --saida …` (N = 22 no 1B, 29 no 3B)
                let nc = if modelo == "1b" { "22" } else { "29" };
                let frio2 = ["--gera", "--celula", &format!("{modelo}:/m"), "--so-texto", texto, "--camadas-cpu", nc, "--saida", "/o/p1/saida"];
                let a = parse_args(v(&frio2)).unwrap_or_else(|e| panic!("frio LC {modelo}/{texto} recusado pelo parser real: {e}"));
                assert_eq!((a.so_texto.as_deref(), a.repete, a.camadas_cpu), (Some(texto), None, Some(nc.parse().unwrap())));
                // quente: um processo com --repete N (N de 1 a 50 no roteiro)
                for n in ["1", "3", "5", n_max.as_str()] {
                    let quente = ["--gera", "--celula", &format!("{modelo}:/m"), "--so-texto", texto, "--repete", n, "--saida", "/o/p1/saida"];
                    let a = parse_args(v(&quente)).unwrap_or_else(|e| panic!("quente {modelo}/{texto}/{n} recusado pelo parser real: {e}"));
                    assert_eq!((a.so_texto.as_deref(), a.repete), (Some(texto), Some(n.parse().unwrap())));
                    // degrau 2, quente: `--so-texto T --camadas-cpu N --repete N --saida …`
                    let quente2 = ["--gera", "--celula", &format!("{modelo}:/m"), "--so-texto", texto, "--camadas-cpu", nc, "--repete", n, "--saida", "/o/p1/saida"];
                    let a = parse_args(v(&quente2)).unwrap_or_else(|e| panic!("quente LC {modelo}/{texto}/{n} recusado pelo parser real: {e}"));
                    assert_eq!((a.repete, a.camadas_cpu), (Some(n.parse().unwrap()), Some(nc.parse().unwrap())));
                    // degrau 1, quente: `--so-texto T --fracao-slots 1.0 --repete N --saida …`
                    let quente1 = ["--gera", "--celula", &format!("{modelo}:/m"), "--so-texto", texto, "--fracao-slots", "1.0", "--repete", n, "--saida", "/o/p1/saida"];
                    let a = parse_args(v(&quente1)).unwrap_or_else(|e| panic!("quente L100 {modelo}/{texto}/{n} recusado pelo parser real: {e}"));
                    assert_eq!((a.so_texto.as_deref(), a.repete, a.fracao), (Some(texto), Some(n.parse().unwrap()), 1.0));
                }
            }
        }
        // a combinação ANTIGA do roteiro é recusada (era o P1 da r111)
        for extra in [&["--novos", "64", "--data", "2026-11-07"][..], &["--novos", "64"][..], &["--data", "2026-11-07"][..]] {
            let mut x = vec!["--gera", "--celula", "1b:/m", "--so-texto", "curto"];
            x.extend_from_slice(extra);
            x.extend_from_slice(&["--saida", "/o/p1/saida"]);
            assert!(parse_args(v(&x)).is_err(), "{extra:?} deve ser recusado");
        }
    }

    #[test]
    fn gera_argumentos_e_arquivos() {
        let base = ["--gera", "--celula", "1b:/m", "--saida", "/o"];
        let a = parse_args(v(&base)).unwrap();
        assert_eq!(a.modo, Some(Modo::Gera));
        assert_eq!(a.criterio_5c, PathBuf::from(CRITERIO_5C_PADRAO));
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m", "--ensaio"])).is_ok(), "ensaio não precisa de saída");
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m"])).is_err(), "sem saída");
        assert!(parse_args(v(&["--gera", "--saida", "/o"])).is_err(), "sem célula");
        assert!(parse_args(v(&["--gera", "--prova-reset", "--celula", "1b:/m", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--gera", "--modo", "livre", "--celula", "1b:/m", "--saida", "/o"])).is_err());
        for extra in [vec!["--data", "2026-11-08"], vec!["--novos", "8"], vec!["--reutiliza"], vec!["--estado", "quente"], vec!["--cenario", "A"], vec!["--rep", "1"], vec!["--consolida", "/d"], vec!["--pedido", "/p.json:0"]] {
            let mut x = base.to_vec();
            x.extend(extra);
            assert!(parse_args(v(&x)).is_err(), "{x:?}");
        }
        assert!(parse_args(v(&["--modo", "correcao", "--celula", "1b:/m", "--saida", "/o", "--novos", "8"])).is_err(), "--novos é só do --gera");
        // Uso normal.
        let l = parse_args(v(&["--gera", "--celula", "1b:/m", "--texto", "oi", "--novos", "8", "--data", "2026-11-08"])).unwrap();
        assert_eq!((l.novos, l.texto.as_deref()), (Some(8), Some("oi")));
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m", "--texto", "oi", "--saida", "/o"])).is_err());
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m", "--celula", "3b:/m3", "--texto", "oi"])).is_err());
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m", "--texto", "oi", "--ensaio"])).is_err());
        assert!(parse_args(v(&["--gera", "--celula", "1b:/m", "--texto", "oi", "--novos", "0"])).is_err());
        let c = parse_args(v(&["--gera", "--celula", "1b:/m1", "--celula", "3b:/m3", "--saida", "/o", "--criterio-5c", "/c.md"])).unwrap();
        assert_eq!((c.celulas.len(), c.criterio_5c.clone()), (2, PathBuf::from("/c.md")));
        assert!(parse_args(v(&["--gera", "--celula", "3b:/m", "--celula", "1b:/m2", "--saida", "/o"])).is_err(), "1b antes de 3b");
        let nomes: Vec<String> = arquivos_de_entrada(&c, Modo::Gera).iter().map(|p| nome_do_arquivo(p)).collect();
        for n in ["texto-curto.txt", "texto-medio.txt", "texto-longo.txt", "ref5c-1b.json", "ref5c-3b.json"] {
            assert!(nomes.contains(&n.to_string()), "{n}");
        }
        assert!(!nomes.contains(&"pedido-A.txt".into()) && !nomes.contains(&"prompts-traco.json".into()));
        assert_eq!(nomes.iter().filter(|n| *n == "tokenizer.json").count(), 2);
        // Hashes: textos congelados; as referências só valem com sha congelado (recusa enquanto `None`).
        for n in ["texto-curto.txt", "texto-medio.txt", "texto-longo.txt"] {
            assert_eq!(sha_esperado(n).map(str::len), Some(64), "{n}");
        }
        assert_eq!(sha_esperado("ref5c-1b.json"), REF5C_SHA_1B);
        assert_eq!(sha_esperado("ref5c-3b.json"), REF5C_SHA_3B);
        assert_eq!(gr::CRITERIO_5C_SHA.len(), 64);
        // Os portões são os dos outros modos (a recusa vem antes de qualquer acesso ao device).
        assert!(exige_autorizacao(None).is_err() && exige_uuid(Some("outra")).is_err());
    }
}
