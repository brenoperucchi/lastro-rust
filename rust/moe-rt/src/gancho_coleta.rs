//! A2, plano de desempenho das cópias (11/11/2026, aprovado pelo Breno às 15:23, ponto 3): o GANCHO DE COLETA da prova bit a bit da O1.
//! Com `LASTRO_GANCHO_LOGITS=1` o gerador guarda, por pedido, o sha256 sobre os bits f32 dos logits (último token do prefill e cada passo
//! do decode) e o sha256 dos bits de `read_kv` por camada no fim do prefill, e o binário grava `gancho-logits-<modelo>.json` na pasta de saída.
//! DESLIGADO por padrão: sem a variável nada é lido, hasheado nem guardado, e o caminho de medida não muda. Ligado serve só à execução de
//! coleta (nunca à medida de tempo: o `read_kv` troca o cache por zeros e de volta, fora dos cronômetros mas dentro do pedido).
//!
//! Id igual não prova vetor igual (1 ULP num logit que não vence passa no argmax), por isso o comparador ([`compara`]) olha os hashes e não os
//! ids: qualquer bit diferente é divergência. Sem GPU: original contra original passa e um mutante de 1 ULP com ids iguais falha (testes abaixo).
//! O SHA-256 é implementado aqui (FIPS 180-4) para não acrescentar dependência ao crate; testado contra os vetores do padrão.
use serde_json::{json, Value};
use std::cell::RefCell;

/// A variável que liga o gancho (só o valor exato `1`).
pub const VARIAVEL_GANCHO: &str = "LASTRO_GANCHO_LOGITS";

pub fn interpreta(v: Option<&str>) -> bool {
    v == Some("1")
}

/// Lê `LASTRO_GANCHO_LOGITS` do ambiente (não UTF-8 conta como ausente).
pub fn do_ambiente() -> bool {
    interpreta(std::env::var(VARIAVEL_GANCHO).ok().as_deref())
}

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
    0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

/// SHA-256 de `dados`.
pub fn sha256(dados: &[u8]) -> [u8; 32] {
    let mut h: [u32; 8] = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let mut bloco = |b: &[u8]| {
        let mut w = [0u32; 64];
        for (i, c) in b.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([c[0], c[1], c[2], c[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7].wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let mj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(mj);
            v = [t1.wrapping_add(t2), v[0], v[1], v[2], v[3].wrapping_add(t1), v[4], v[5], v[6]];
        }
        for (a, b) in h.iter_mut().zip(v) {
            *a = a.wrapping_add(b);
        }
    };
    let mut cheios = dados.chunks_exact(64);
    for b in &mut cheios {
        bloco(b);
    }
    let resto = cheios.remainder();
    let mut fim = resto.to_vec();
    fim.push(0x80);
    while fim.len() % 64 != 56 {
        fim.push(0);
    }
    fim.extend_from_slice(&((dados.len() as u64) * 8).to_be_bytes());
    for b in fim.chunks_exact(64) {
        bloco(b);
    }
    let mut out = [0u8; 32];
    for (i, x) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// sha256 (em hexadecimal) sobre os bits f32 (`to_bits`, little endian) de `v`, na ordem: o hash muda com qualquer bit, inclusive NaN e -0.0.
pub fn hash_bits_f32(v: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    hex(&sha256(&bytes))
}

/// O que o gancho guarda de um pedido.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PedidoGancho {
    pub indice: usize,
    pub aquecimento: bool,
    /// Ids do prompt e ids gerados (o que o argmax escolheu): conferência de contexto; a prova são os hashes.
    pub ids: Vec<usize>,
    pub gerados: Vec<usize>,
    /// Posição 0: logits do último token do prefill; posição `i >= 1`: logits do passo `i` do decode.
    pub logits: Vec<String>,
    /// Por camada, `[sha256 de K, sha256 de V]` dos bits de `read_kv` de `0..len` posições no fim do prefill.
    pub kv: Vec<[String; 2]>,
}

/// O arquivo `gancho-logits-<modelo>.json`.
pub fn para_json(modelo: &str, pedidos: &[PedidoGancho]) -> Value {
    json!({
        "tipo": "gancho-logits",
        "versao": 1,
        "variavel": VARIAVEL_GANCHO,
        "modelo": modelo,
        "definicao": "sha256 (hex) sobre os bits f32 little endian; logits[0] = último token do prefill, logits[i] = passo i do decode; kv[camada] = [K, V] de 0..len posições no fim do prefill",
        "pedidos": pedidos.iter().map(|p| json!({
            "indice": p.indice, "aquecimento": p.aquecimento, "ids": p.ids, "gerados": p.gerados, "logits": p.logits, "kv": p.kv,
        })).collect::<Vec<_>>(),
    })
}

/// Lê o que [`para_json`] escreveu.
pub fn de_json(v: &Value) -> Result<Vec<PedidoGancho>, String> {
    if v.get("tipo").and_then(Value::as_str) != Some("gancho-logits") || v.get("versao").and_then(Value::as_u64) != Some(1) {
        return Err("arquivo fora do esquema (tipo gancho-logits, versao 1)".into());
    }
    let lista = v.get("pedidos").and_then(Value::as_array).ok_or("sem a lista de pedidos")?;
    let ids = |x: &Value, k: &str| -> Result<Vec<usize>, String> {
        x.get(k).and_then(Value::as_array).ok_or(format!("pedido sem {k}"))?.iter().map(|n| n.as_u64().map(|n| n as usize).ok_or(format!("{k}: valor que não é inteiro"))).collect()
    };
    let textos = |x: &Value, k: &str| -> Result<Vec<String>, String> {
        x.get(k).and_then(Value::as_array).ok_or(format!("pedido sem {k}"))?.iter().map(|s| s.as_str().map(str::to_string).ok_or(format!("{k}: valor que não é texto"))).collect()
    };
    lista
        .iter()
        .map(|x| {
            let kv = x
                .get("kv")
                .and_then(Value::as_array)
                .ok_or("pedido sem kv")?
                .iter()
                .map(|par| match par.as_array().map(|a| a.iter().map(|s| s.as_str().map(str::to_string)).collect::<Vec<_>>()).as_deref() {
                    Some([Some(k), Some(v)]) => Ok([k.clone(), v.clone()]),
                    _ => Err("kv: par [K, V] malformado".to_string()),
                })
                .collect::<Result<Vec<_>, String>>()?;
            Ok(PedidoGancho {
                indice: x.get("indice").and_then(Value::as_u64).ok_or("pedido sem indice")? as usize,
                aquecimento: x.get("aquecimento").and_then(Value::as_bool).ok_or("pedido sem aquecimento")?,
                ids: ids(x, "ids")?,
                gerados: ids(x, "gerados")?,
                logits: textos(x, "logits")?,
                kv,
            })
        })
        .collect()
}

/// Compara dois conjuntos de pedidos e devolve as divergências (vazio = idênticos bit a bit). O comparador olha os HASHES: ids iguais com um
/// logit diferente em 1 ULP é divergência. Pedidos, ids, quantidade de passos, hashes de logits e hashes de KV por camada têm de coincidir.
pub fn compara(a: &[PedidoGancho], b: &[PedidoGancho]) -> Vec<String> {
    let mut d = vec![];
    if a.is_empty() || b.is_empty() {
        d.push("conjunto sem pedidos: nada a comparar".to_string());
        return d;
    }
    if a.len() != b.len() {
        d.push(format!("{} pedidos contra {}", a.len(), b.len()));
        return d;
    }
    for (x, y) in a.iter().zip(b) {
        let p = x.indice;
        if x.indice != y.indice || x.aquecimento != y.aquecimento {
            d.push(format!("pedido {p}: índice ou marca de aquecimento diferentes"));
            continue;
        }
        if x.ids != y.ids {
            d.push(format!("pedido {p}: ids do prompt diferentes"));
        }
        if x.gerados != y.gerados {
            d.push(format!("pedido {p}: ids gerados diferentes"));
        }
        if x.logits.len() != y.logits.len() {
            d.push(format!("pedido {p}: {} hashes de logits contra {}", x.logits.len(), y.logits.len()));
        }
        if let Some(i) = x.logits.iter().zip(&y.logits).position(|(u, v)| u != v) {
            d.push(format!("pedido {p}: logits do {} diferem (primeira divergência)", if i == 0 { "último token do prefill".to_string() } else { format!("passo {i} do decode") }));
        }
        if x.kv.len() != y.kv.len() {
            d.push(format!("pedido {p}: KV de {} camadas contra {}", x.kv.len(), y.kv.len()));
        }
        for (l, (u, v)) in x.kv.iter().zip(&y.kv).enumerate() {
            for (lado, (s, t)) in ["K", "V"].iter().zip(u.iter().zip(v)) {
                if s != t {
                    d.push(format!("pedido {p}: KV da camada {l} ({lado}) difere no fim do prefill"));
                }
            }
        }
        if x.logits.is_empty() || x.kv.is_empty() {
            d.push(format!("pedido {p}: sem hashes de logits ou de KV (gancho desligado?)"));
        }
    }
    d
}

type LeKv = Box<dyn FnMut(usize) -> Result<Vec<[String; 2]>, String>>;

thread_local! {
    /// O leitor do KV da sequência em uso (registrado por quem cria o backend, só com o gancho ligado): dado `len` posições, devolve por camada
    /// `[sha256 K, sha256 V]`. Por thread porque o fecho guarda o backend e a stream, que não precisam ser `Send`.
    static LE_KV: RefCell<Option<LeKv>> = const { RefCell::new(None) };
}

/// Registra (ou limpa, com `None`) o leitor do KV desta thread.
pub fn registra_le_kv(f: Option<LeKv>) {
    LE_KV.with(|c| *c.borrow_mut() = f);
}

/// Lê o hash do KV por camada de `len` posições com o leitor registrado; `None` se não há leitor.
pub fn le_kv(len: usize) -> Option<Result<Vec<[String; 2]>, String>> {
    LE_KV.with(|c| c.borrow_mut().as_mut().map(|f| f(len)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_bate_com_os_vetores_do_padrao() {
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(hex(&sha256(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
        // 1.000.000 de 'a' (várias passagens de bloco)
        assert_eq!(hex(&sha256(&vec![b'a'; 1_000_000])), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
        // 55, 56 e 64 bytes (bordas do preenchimento)
        for n in [55usize, 56, 57, 63, 64, 65] {
            assert_eq!(sha256(&vec![b'x'; n]).len(), 32);
        }
        assert_ne!(sha256(&vec![b'x'; 55]), sha256(&vec![b'x'; 56]));
    }

    #[test]
    fn variavel_so_liga_com_o_valor_um() {
        assert_eq!(VARIAVEL_GANCHO, "LASTRO_GANCHO_LOGITS");
        assert!(interpreta(Some("1")));
        for v in [None, Some(""), Some("0"), Some("true"), Some(" 1"), Some("1 "), Some("11")] {
            assert!(!interpreta(v), "{v:?}");
        }
    }

    fn logits(n: usize) -> Vec<f32> {
        (0..n).map(|i| ((i * 37 % 101) as f32) * 0.013 - 0.4).collect()
    }

    fn pedido(l: &[f32]) -> PedidoGancho {
        let argmax = l.iter().enumerate().fold(0, |m, (i, x)| if *x > l[m] { i } else { m });
        PedidoGancho {
            indice: 0,
            aquecimento: false,
            ids: vec![1, 2, 3],
            gerados: vec![argmax],
            logits: vec![hash_bits_f32(l), hash_bits_f32(&l[..8])],
            kv: vec![[hash_bits_f32(&l[..16]), hash_bits_f32(&l[16..32])], [hash_bits_f32(&l[1..17]), hash_bits_f32(&l[2..18])]],
        }
    }

    #[test]
    fn original_contra_original_passa_e_sobrevive_ao_json() {
        let l = logits(64);
        let a = vec![pedido(&l)];
        assert!(compara(&a, &a.clone()).is_empty());
        let v = para_json("1b", &a);
        assert_eq!(de_json(&v).unwrap(), a);
        assert!(compara(&a, &de_json(&v).unwrap()).is_empty());
    }

    #[test]
    fn mutante_de_um_ulp_com_ids_iguais_falha() {
        let l = logits(64);
        // um logit que NÃO vence muda 1 ULP: o argmax (ids) é o mesmo
        let argmax = l.iter().enumerate().fold(0, |m, (i, x)| if *x > l[m] { i } else { m });
        let alvo = (argmax + 5) % l.len();
        let mut m = l.clone();
        m[alvo] = f32::from_bits(m[alvo].to_bits() + 1);
        assert_ne!(m[alvo], l[alvo]);
        let (a, b) = (pedido(&l), pedido(&m));
        assert_eq!(a.gerados, b.gerados, "os ids são iguais");
        let d = compara(&[a], &[b]);
        assert!(d.iter().any(|x| x.contains("logits do último token do prefill")), "{d:?}");
    }

    #[test]
    fn divergencia_no_decode_no_kv_ou_nos_ids_e_localizada() {
        let l = logits(64);
        let a = pedido(&l);
        let mut p = a.clone();
        p.logits[1] = hash_bits_f32(&[1.0]);
        assert!(compara(&[a.clone()], &[p]).iter().any(|x| x.contains("passo 1 do decode")));
        let mut k = a.clone();
        k.kv[1][0] = hash_bits_f32(&[2.0]);
        assert!(compara(&[a.clone()], &[k]).iter().any(|x| x.contains("camada 1 (K)")));
        let mut i = a.clone();
        i.gerados = vec![999];
        assert!(compara(&[a.clone()], &[i]).iter().any(|x| x.contains("ids gerados")));
        let mut cur = a.clone();
        cur.logits.pop();
        assert!(!compara(&[a.clone()], &[cur]).is_empty());
        assert!(!compara(&[], &[a.clone()]).is_empty(), "sem pedidos não prova nada");
        assert!(!compara(&[a.clone()], &[a.clone(), a]).is_empty());
    }

    #[test]
    fn hash_distingue_menos_zero_nan_e_ordem() {
        assert_ne!(hash_bits_f32(&[0.0]), hash_bits_f32(&[-0.0]));
        assert_ne!(hash_bits_f32(&[1.0, 2.0]), hash_bits_f32(&[2.0, 1.0]));
        assert_ne!(hash_bits_f32(&[f32::NAN]), hash_bits_f32(&[f32::from_bits(f32::NAN.to_bits() ^ 1)]));
    }

    #[test]
    fn json_fora_do_esquema_e_recusado() {
        assert!(de_json(&json!({"tipo": "outro"})).is_err());
        assert!(de_json(&json!({"tipo": "gancho-logits", "versao": 2, "pedidos": []})).is_err());
        assert!(de_json(&json!({"tipo": "gancho-logits", "versao": 1, "pedidos": [{"indice": 0}]})).is_err());
    }

    #[test]
    fn leitor_de_kv_e_por_thread_e_pode_ser_limpo() {
        assert!(le_kv(3).is_none());
        registra_le_kv(Some(Box::new(|n| Ok(vec![[n.to_string(), "v".into()]]))));
        assert_eq!(le_kv(3).unwrap().unwrap(), vec![["3".to_string(), "v".to_string()]]);
        registra_le_kv(None);
        assert!(le_kv(3).is_none());
    }
}
