//! Métricas que o runtime publica para o monitor.
//!
//! Transporte: um arquivo JSON, regravado de forma atômica (temporário no mesmo diretório + `rename`), por padrão em
//! `$XDG_RUNTIME_DIR/moe-rt/metrics.json`. O monitor lê o arquivo a cada atualização e considera o runtime parado quando
//! o arquivo some ou fica velho (`ts_ms` mais antigo que [`STALE_MS`]).
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// Versão do esquema; o monitor recusa versões que não conhece.
pub const SCHEMA: u32 = 1;
/// Depois disso sem nova publicação, o runtime é mostrado como parado.
pub const STALE_MS: u64 = 3_000;

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Snapshot {
    pub schema: u32,
    /// Instante da publicação (ms desde a época Unix).
    pub ts_ms: u64,
    pub pid: u32,
    /// Início do runtime (ms desde a época Unix).
    pub started_ms: u64,
    pub model: Model,
    pub state: State,
    pub speed: Speed,
    pub context: Context,
    pub cache: ExpertCache,
    /// Régua da Fase 0(a) na última janela de decode.
    pub window: Window,
    /// Mais recentes primeiro.
    pub requests: Vec<Request>,
    /// Configuração do motor em texto (política de colocação, orçamento de VRAM, threads...), para a visão Motor.
    pub settings: BTreeMap<String, String>,
    /// Eventos recentes do runtime, mais antigos primeiro (o runtime guarda só os últimos), para a visão Logs.
    pub events: Vec<Event>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Event {
    pub ts_ms: u64,
    /// "info", "aviso" ou "erro".
    pub level: String,
    pub message: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Model {
    pub name: String,
    pub layers: u32,
    pub experts: u32,
    pub experts_per_tok: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum State {
    #[default]
    Idle,
    /// Lendo o prompt (prefill).
    Reading { done: u64, total: u64 },
    /// Gerando a resposta.
    Generating { tokens: u64, max_tokens: u64 },
    Queued { waiting: u32 },
    Error { message: String },
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Speed {
    /// tok/s do decode agora (janela recente) e da última requisição terminada.
    pub decode_now: f64,
    pub decode_last: f64,
    /// tok/s do prefill agora e da requisição em curso.
    pub prefill_now: f64,
    pub prefill_request: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Context {
    pub used: u64,
    pub max: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExpertCache {
    pub slots_total: u32,
    pub slots_resident: u32,
    pub slots_loading: u32,
    /// Fração dos experts pedidos na última janela servidos da VRAM, pela cópia e pela CPU.
    pub hit_frac: f64,
    pub copy_frac: f64,
    pub cpu_frac: f64,
    /// Cópias RAM→VRAM em voo agora e taxa recente (GB/s).
    pub copies_inflight: u32,
    pub h2d_gbs: f64,
    /// Contadores desde o início.
    pub total_hits: u64,
    pub total_copies: u64,
    pub total_cpu: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Window {
    /// Fração do tempo da janela atribuível à colocação de experts (espera por cópia + CPU além do ideal).
    pub exposure_frac: f64,
    /// Fração do tempo da janela em que o host espera a própria GPU.
    pub gpu_wait_frac: f64,
    pub ms_per_token: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Request {
    /// Início (ms desde a época Unix).
    pub started_ms: u64,
    pub prompt_tokens: u64,
    /// Tokens do prompt reaproveitados de um cache de prefixo (0 enquanto o runtime não tiver um).
    pub reused_tokens: u64,
    pub output_tokens: u64,
    pub duration_ms: u64,
    pub prefill_tok_s: f64,
    pub decode_tok_s: f64,
    /// "ok", "erro", "em curso", "cancelada".
    pub status: String,
}

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Caminho padrão: `$XDG_RUNTIME_DIR/moe-rt/metrics.json`, ou `/tmp/moe-rt-<uid>/metrics.json` sem a variável.
pub fn default_path() -> PathBuf {
    match std::env::var_os("XDG_RUNTIME_DIR") {
        Some(dir) if !dir.is_empty() => Path::new(&dir).join("moe-rt").join("metrics.json"),
        _ => {
            let uid = fs::metadata("/proc/self").map(|m| std::os::unix::fs::MetadataExt::uid(&m)).unwrap_or(0);
            PathBuf::from(format!("/tmp/moe-rt-{uid}")).join("metrics.json")
        }
    }
}

/// Grava snapshots no ponto único, sempre por arquivo temporário + `rename`, para o monitor nunca ler meio JSON.
pub struct Publisher {
    path: PathBuf,
    tmp: PathBuf,
}

impl Publisher {
    pub fn new(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        let path = path.into();
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension(format!("json.tmp-{}", std::process::id()));
        Ok(Self { path, tmp })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Publica `snap` com `schema`, `ts_ms` e `pid` preenchidos aqui.
    pub fn publish(&self, snap: &Snapshot) -> std::io::Result<()> {
        let mut s = snap.clone();
        s.schema = SCHEMA;
        s.ts_ms = now_ms();
        s.pid = std::process::id();
        let mut f = fs::File::create(&self.tmp)?;
        f.write_all(&serde_json::to_vec(&s).map_err(std::io::Error::other)?)?;
        f.sync_data()?;
        fs::rename(&self.tmp, &self.path)
    }
}

/// O que o monitor sabe do runtime numa leitura.
#[derive(Debug, PartialEq)]
pub enum Reading {
    /// Arquivo ausente: nenhum runtime publicou.
    Absent,
    /// Snapshot válido, mas sem atualização há `age_ms`.
    Stale { snap: Box<Snapshot>, age_ms: u64 },
    Live(Box<Snapshot>),
    /// Arquivo ilegível ou de outra versão do esquema.
    Invalid(String),
}

pub fn read(path: &Path, now_ms: u64) -> Reading {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Reading::Absent,
        Err(e) => return Reading::Invalid(e.to_string()),
    };
    let snap: Snapshot = match serde_json::from_slice(&bytes) {
        Ok(s) => s,
        Err(e) => return Reading::Invalid(format!("JSON inválido: {e}")),
    };
    if snap.schema != SCHEMA {
        return Reading::Invalid(format!("esquema {} (o monitor conhece {SCHEMA})", snap.schema));
    }
    let age_ms = now_ms.saturating_sub(snap.ts_ms);
    if age_ms > STALE_MS {
        Reading::Stale { snap: Box::new(snap), age_ms }
    } else {
        Reading::Live(Box::new(snap))
    }
}
