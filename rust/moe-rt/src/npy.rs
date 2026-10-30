//! Leitura mínima de `.npy` (versões 1 a 3, little-endian, ordem C), para as referências gravadas pelo numpy.
use std::path::Path;

pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

fn field<'a>(head: &'a str, key: &str) -> Result<&'a str, String> {
    let at = head.find(&format!("'{key}'")).ok_or(format!("cabeçalho .npy sem {key}"))?;
    Ok(head[at + key.len() + 2..].trim_start().trim_start_matches(':').trim_start())
}

impl Npy {
    pub fn read(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let short = || format!("{}: .npy truncado", path.display());
        if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
            return Err(format!("{}: não é .npy", path.display()));
        }
        let (hlen, start) = match raw[6] {
            1 => (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10usize),
            2 | 3 => {
                let b = raw.get(8..12).ok_or_else(short)?;
                (u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize, 12)
            }
            v => return Err(format!("{}: versão .npy {v}", path.display())),
        };
        let end = start.checked_add(hlen).filter(|&e| e <= raw.len()).ok_or_else(short)?;
        let head = std::str::from_utf8(&raw[start..end]).map_err(|e| e.to_string())?;
        let descr = field(head, "descr")?.trim_start_matches('\'').split('\'').next().unwrap_or("").to_string();
        if field(head, "fortran_order")?.starts_with("True") {
            return Err(format!("{}: ordem Fortran", path.display()));
        }
        let shape_txt = field(head, "shape")?;
        let shape_txt = shape_txt.strip_prefix('(').ok_or("shape sem '('")?;
        let shape_txt = &shape_txt[..shape_txt.find(')').ok_or("shape sem ')'")?];
        let shape: Vec<usize> = shape_txt
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().map_err(|_| format!("shape inválido: {shape_txt}")))
            .collect::<Result<_, _>>()?;
        let data = raw[end..].to_vec();
        Ok(Self { descr, shape, data })
    }

    fn check(&self, descr: &str) -> Result<(), String> {
        if self.descr != descr {
            return Err(format!("dtype {} (esperava {descr})", self.descr));
        }
        if self.shape.iter().try_fold(4usize, |acc, &d| acc.checked_mul(d)) != Some(self.data.len()) {
            return Err(format!("{} bytes para a forma {:?}", self.data.len(), self.shape));
        }
        Ok(())
    }

    pub fn f32(&self) -> Result<Vec<f32>, String> {
        self.check("<f4")?;
        Ok(self.data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    pub fn i32(&self) -> Result<Vec<i32>, String> {
        self.check("<i4")?;
        Ok(self.data.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}
