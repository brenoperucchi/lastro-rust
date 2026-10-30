//! Leitura de tensores BF16 de um checkpoint safetensors (um ou vários arquivos `*.safetensors` num diretório).
//!
//! Formato: 8 bytes little-endian com o tamanho do cabeçalho, o cabeçalho JSON (`nome → {dtype, shape, data_offsets}`) e os
//! dados. Só o índice fica em memória; cada tensor é lido do disco quando pedido.
use cutile::half::bf16;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub struct Entry {
    file: usize,
    pub dtype: String,
    pub shape: Vec<usize>,
    start: u64,
    end: u64,
}

pub struct Checkpoint {
    files: Vec<(PathBuf, File)>,
    pub index: HashMap<String, Entry>,
}

#[derive(Debug)]
pub enum WeightError {
    Io(std::io::Error),
    Format(String),
}

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WeightError::Io(e) => write!(f, "E/S: {e}"),
            WeightError::Format(m) => write!(f, "safetensors: {m}"),
        }
    }
}

impl std::error::Error for WeightError {}

impl From<std::io::Error> for WeightError {
    fn from(e: std::io::Error) -> Self {
        WeightError::Io(e)
    }
}

fn bad(m: impl Into<String>) -> WeightError {
    WeightError::Format(m.into())
}

impl Checkpoint {
    pub fn open(dir: &Path) -> Result<Self, WeightError> {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            return Err(bad(format!("nenhum *.safetensors em {}", dir.display())));
        }
        let mut files = Vec::new();
        let mut index = HashMap::new();
        for (fi, path) in paths.into_iter().enumerate() {
            let file = File::open(&path)?;
            let mut len = [0u8; 8];
            file.read_exact_at(&mut len, 0)?;
            let n = u64::from_le_bytes(len);
            if n > 100 << 20 {
                return Err(bad(format!("cabeçalho de {n} bytes em {}", path.display())));
            }
            let mut head = vec![0u8; n as usize];
            file.read_exact_at(&mut head, 8)?;
            let json: serde_json::Value = serde_json::from_slice(&head).map_err(|e| bad(e.to_string()))?;
            let obj = json.as_object().ok_or_else(|| bad("cabeçalho não é objeto"))?;
            let size = file.metadata()?.len();
            for (name, v) in obj {
                if name == "__metadata__" {
                    continue;
                }
                let dtype = v["dtype"].as_str().ok_or_else(|| bad(format!("{name}: sem dtype")))?.to_string();
                let shape: Vec<usize> = v["shape"]
                    .as_array()
                    .ok_or_else(|| bad(format!("{name}: sem shape")))?
                    .iter()
                    .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| bad(format!("{name}: shape inválido"))))
                    .collect::<Result<_, _>>()?;
                let off = v["data_offsets"].as_array().ok_or_else(|| bad(format!("{name}: sem data_offsets")))?;
                let [s, e] = &off[..] else {
                    return Err(bad(format!("{name}: data_offsets com {} valores (esperava 2)", off.len())));
                };
                let (Some(s), Some(e)) = (s.as_u64(), e.as_u64()) else {
                    return Err(bad(format!("{name}: data_offsets não numéricos")));
                };
                let (Some(start), Some(end)) = ((8 + n).checked_add(s), (8 + n).checked_add(e)) else {
                    return Err(bad(format!("{name}: offsets fora do arquivo")));
                };
                if s > e || end > size {
                    return Err(bad(format!("{name}: offsets fora do arquivo")));
                }
                index.insert(name.clone(), Entry { file: fi, dtype, shape, start, end });
            }
            files.push((path, file));
        }
        Ok(Self { files, index })
    }

    /// Tensor BF16 inteiro, achatado, com a forma.
    pub fn bf16(&self, name: &str) -> Result<(Vec<bf16>, Vec<usize>), WeightError> {
        let e = self.index.get(name).ok_or_else(|| bad(format!("tensor ausente: {name}")))?;
        if e.dtype != "BF16" {
            return Err(bad(format!("{name}: dtype {} (esperava BF16)", e.dtype)));
        }
        let bytes = e.shape.iter().try_fold(2usize, |acc, &d| acc.checked_mul(d));
        if bytes.is_none_or(|b| b as u64 != e.end - e.start) {
            return Err(bad(format!("{name}: tamanho não bate com a forma")));
        }
        let mut raw = vec![0u8; (e.end - e.start) as usize];
        self.files[e.file].1.read_exact_at(&mut raw, e.start)?;
        let v = raw.chunks_exact(2).map(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]]))).collect();
        Ok((v, e.shape.clone()))
    }
}
