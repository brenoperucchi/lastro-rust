//! Ajudantes compartilhados pelos testes: diretório temporário, valores determinísticos e escrita de safetensors.
#![allow(dead_code)]
use cutile::half::bf16;
use std::fs;
use std::path::{Path, PathBuf};

/// Diretório temporário com um único arquivo; removido sem recursão no fim.
pub struct Scratch(pub PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        if let Some(dir) = self.0.parent() {
            let _ = fs::remove_dir(dir);
        }
    }
}

pub fn scratch(name: &str) -> Scratch {
    let dir = std::env::temp_dir().join(format!("moe-rt-test-{}-{name}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    Scratch(dir.join("model.safetensors"))
}

pub fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32) * 2.0 - 1.0
        })
        .collect()
}

pub fn bf(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

/// Escreve um safetensors com os tensores dados (nome, dtype, forma, bytes).
pub fn write_safetensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
    let mut head = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, bytes) in tensors {
        let start = data.len();
        data.extend_from_slice(bytes);
        head.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
        );
    }
    let head = serde_json::to_vec(&serde_json::Value::Object(head)).unwrap();
    let mut out = (head.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&head);
    out.extend_from_slice(&data);
    fs::write(path, out).unwrap();
}

pub fn bytes(v: &[bf16]) -> Vec<u8> {
    v.iter().flat_map(|b| b.to_bits().to_le_bytes()).collect()
}

/// Dispositivo 0, ou `None` (teste pulado, com aviso) se não houver GPU. Com `MOE_RT_REQUIRE_GPU=1`, a falta de GPU é erro.
pub fn gpu_or_skip() -> Option<std::sync::Arc<cutile::prelude::Device>> {
    match cutile::prelude::Device::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            assert!(std::env::var("MOE_RT_REQUIRE_GPU").as_deref() != Ok("1"), "MOE_RT_REQUIRE_GPU=1 e sem GPU: {e:?}");
            eprintln!("sem GPU, teste pulado: {e:?}");
            None
        }
    }
}
