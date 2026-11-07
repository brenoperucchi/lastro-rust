//! Tokenizador e detokenizador mínimos, offline, só com `std` e `serde_json` (degrau 5a da interface de texto).
//!
//! Cobre exatamente o que o `tokenizer.json` do Granite 3.1 (1B-A400M e 3B-A800M, arquivos idênticos) usa:
//! - sem normalizador;
//! - pré-tokenizador `ByteLevel` com `use_regex: true` e `add_prefix_space: false` (a regex do GPT-2);
//! - modelo BPE por rank de merges, sem dropout, sem `byte_fallback`, sem `unk_token`, `ignore_merges: false`;
//! - decodificador `ByteLevel`;
//! - bytes cujo caractere ByteLevel não está no vocabulário (0xC0, 0xC1, 0xF1, 0xF5..0xFF no Granite) são DESCARTADOS na codificação, como
//!   o `tokenizers` faz sem `unk_token`; em UTF-8 válido isso só atinge U+40000..U+7FFFF (planos não atribuídos), e a ida e volta em bytes
//!   não vale para esse caso (o HF também perde o byte);
//! - tokens adicionados (todos `special`, `normalized: false`, sem `lstrip`/`rstrip`/`single_word`), reconhecidos no texto
//!   cru por maior casamento mais à esquerda, como o `tokenizers` faz (`<|end_of_text|>` literal vira o id 0).
//!
//! O `tokenizer.json` é validado ao carregar: qualquer configuração diferente desta (normalizador, `byte_fallback`, `dropout`,
//! prefixo de espaço, token adicionado com `lstrip`/`rstrip`/`single_word`/`normalized`) vira [`ErroTokenizador::Formato`] em vez de
//! uma tokenização silenciosamente diferente da referência.
//!
//! # A regex do GPT-2 escrita à mão
//!
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, aplicada da esquerda para a direita, primeira
//! alternativa que casa vence (contrações em minúsculas; o espaço opcional é só U+0020).
//!
//! ## Onde `std` diverge de `\p{L}` e `\p{N}` (e como foi tratado)
//!
//! - `\p{N}` (Nd, Nl, No) coincide com `char::is_numeric` na versão de Unicode de cada lado quando as duas versões são iguais.
//! - `\p{L}` (Lu, Ll, Lt, Lm, Lo) NÃO coincide com `char::is_alphabetic`: esta é a propriedade Unicode `Alphabetic`, que também
//!   inclui números-letra (Nl, p. ex. algarismos romanos U+2160) e marcas "Other_Alphabetic" (vogais dependentes de escritas
//!   indianas, pontos hebraicos, marcas árabes, p. ex. U+0345, U+093E) e símbolos como as letras circuladas (U+24B6). Esses
//!   pontos de código são letra para `std` e NÃO são letra para o `\p{L}` do pré-tokenizador, então um código só com `std`
//!   produziria ids diferentes do HF em texto com devanágari, hebraico com sinais, números romanos etc.
//! - Por isso as classes `\p{L}` e `\p{N}` vêm das tabelas de faixas em `tokenizador/tabelas.rs`, geradas por
//!   `tools/gera_tabelas_5a.py` varrendo TODOS os valores escalares Unicode contra o próprio pré-tokenizador da biblioteca
//!   `tokenizers` (a que gerou os ids de referência). Isso fixa a versão de Unicode da referência e não depende da versão do
//!   `std`. O teste `std_vs_tabelas_unicode` mede e declara a diferença para `char::is_alphabetic`/`is_numeric`.
//! - `\s` usa `char::is_whitespace` (propriedade `White_Space`); a mesma varredura confere que coincide com o oracle.
//!
//! # Detokenização
//!
//! [`Tokenizador::decodifica_bytes`] devolve os bytes exatos (inverso do ByteLevel; tokens especiais viram o texto literal deles,
//! como o `decode` do HF com `skip_special_tokens=False`). [`Tokenizador::decodifica`] converte esses bytes para `String` com
//! substituição `U+FFFD` (UTF-8 inválido, p. ex. uma sequência cortada no meio de um caractere) sobre a sequência INTEIRA; o HF
//! aplica a substituição por token, então só em sequências que partem um caractere entre tokens os dois podem diferir (medido e
//! declarado no resultado da prova 5a). Não há `clean_up_tokenization_spaces` aqui: o `decode` padrão do HF apaga espaços antes de
//! pontuação (`" ."` vira `"."`); esta implementação equivale ao `decode(..., clean_up_tokenization_spaces=False)`.
//!
//! # Chat template
//!
//! [`renderiza_chat`] é a renderização FIXA do `chat_template` do Granite 3.1 (Jinja), só para mensagens `system` opcional, `user`
//! e `assistant`, sem `tools`, `documents` nem `controls`. A data de `strftime_now('%B %d, %Y')` entra como parâmetro.
mod tabelas;

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::path::Path;
use std::sync::Mutex;

/// Erros do tokenizador. Nenhuma entrada do usuário causa `panic`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErroTokenizador {
    /// Falha ao ler o arquivo.
    Io(String),
    /// JSON inválido.
    Json(String),
    /// `tokenizer.json` fora do que este código cobre (a mensagem diz o campo).
    Formato(String),
    /// Id fora do vocabulário na detokenização.
    IdDesconhecido(u32),
    /// Lista de mensagens inválida para o template (vazia ou com papel/conteúdo que o modelo fixo não cobre).
    Mensagens(String),
    /// Data inválida em [`formata_data`].
    Data(String),
}

impl fmt::Display for ErroTokenizador {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(m) => write!(f, "tokenizador: erro de leitura: {m}"),
            Self::Json(m) => write!(f, "tokenizador: JSON inválido: {m}"),
            Self::Formato(m) => write!(f, "tokenizador: tokenizer.json fora do suportado: {m}"),
            Self::IdDesconhecido(i) => write!(f, "tokenizador: id {i} fora do vocabulário"),
            Self::Mensagens(m) => write!(f, "tokenizador: mensagens inválidas: {m}"),
            Self::Data(m) => write!(f, "tokenizador: data inválida: {m}"),
        }
    }
}

impl std::error::Error for ErroTokenizador {}

fn fmt_err<T>(m: impl Into<String>) -> Result<T, ErroTokenizador> {
    Err(ErroTokenizador::Formato(m.into()))
}

/// Mapa de bytes para caracteres do GPT-2: bytes imprimíveis vão para si mesmos, os demais para U+0100 em diante, na ordem do byte.
fn bytes_para_chars() -> [char; 256] {
    let mut m = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..=255u32 {
        let imprimivel = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        m[b as usize] = if imprimivel {
            char::from_u32(b).unwrap_or('\0')
        } else {
            let c = char::from_u32(256 + n).unwrap_or('\0');
            n += 1;
            c
        };
    }
    m
}

fn em_faixas(tabela: &[(u32, u32)], c: char) -> bool {
    let v = c as u32;
    tabela.binary_search_by(|&(lo, hi)| if v < lo { std::cmp::Ordering::Greater } else if v > hi { std::cmp::Ordering::Less } else { std::cmp::Ordering::Equal }).is_ok()
}

/// `\p{L}` como o pré-tokenizador da referência (ver a documentação do módulo).
pub fn eh_letra(c: char) -> bool {
    em_faixas(tabelas::LETRA, c)
}

/// `\p{N}` como o pré-tokenizador da referência.
pub fn eh_numero(c: char) -> bool {
    em_faixas(tabelas::NUMERO, c)
}

/// Divide `texto` nos pedaços da regex do GPT-2 (fatias contíguas que cobrem o texto inteiro).
pub fn pre_tokeniza(texto: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = texto.char_indices().collect();
    let n = cs.len();
    let fim = |i: usize| if i < n { cs[i].0 } else { texto.len() };
    let mut saida = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        // contrações: 's 't 're 've 'm 'll 'd
        if c == '\'' {
            let r = |k: usize| cs.get(i + k).map(|x| x.1);
            let len = match (r(1), r(2)) {
                (Some('s' | 't' | 'm' | 'd'), _) => 2,
                (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => 3,
                _ => 0,
            };
            if len > 0 {
                saida.push(&texto[cs[i].0..fim(i + len)]);
                i += len;
                continue;
            }
        }
        // ` ?` + classe
        let j = if c == ' ' && i + 1 < n { i + 1 } else { i };
        let cj = cs[j].1;
        let classe = |ch: char| -> u8 {
            if ch.is_whitespace() {
                0
            } else if eh_letra(ch) {
                1
            } else if eh_numero(ch) {
                2
            } else {
                3
            }
        };
        let k0 = classe(cj);
        if k0 != 0 {
            let mut e = j;
            while e < n && classe(cs[e].1) == k0 {
                e += 1;
            }
            saida.push(&texto[cs[i].0..fim(e)]);
            i = e;
            continue;
        }
        // espaços: `\s+(?!\S)` (deixa o último para o pedaço seguinte) e `\s+`
        let mut e = i;
        while e < n && cs[e].1.is_whitespace() {
            e += 1;
        }
        let fim_ws = if e < n && e - i > 1 { e - 1 } else { e };
        saida.push(&texto[cs[i].0..fim(fim_ws)]);
        i = fim_ws;
    }
    saida
}

/// Token adicionado reconhecido no texto cru.
#[derive(Debug, Clone)]
struct Adicionado {
    conteudo: String,
    id: u32,
}

const CACHE_MAX: usize = 50_000;

/// Tokenizador BPE ByteLevel carregado de um `tokenizer.json`.
pub struct Tokenizador {
    /// id -> texto do token no alfabeto ByteLevel (None se o id não existe).
    id_para_token: Vec<Option<String>>,
    /// id -> bytes que o token representa (ByteLevel invertido; adicionados: o texto literal).
    id_para_bytes: Vec<Option<Vec<u8>>>,
    /// byte -> id do token de um caractere.
    /// `None` quando o vocabulário não tem o caractere do byte (o `tokenizers`, sem `unk_token`, descarta esse byte).
    id_do_byte: [Option<u32>; 256],
    /// (esquerda, direita) -> (rank, id do resultado).
    merges: HashMap<u64, (u32, u32)>,
    adicionados: Vec<Adicionado>,
    cache: Mutex<HashMap<Vec<u8>, Vec<u32>>>,
}

fn chave(a: u32, b: u32) -> u64 {
    ((a as u64) << 32) | b as u64
}

impl Tokenizador {
    /// Carrega e valida um `tokenizer.json`.
    pub fn carrega(caminho: &Path) -> Result<Self, ErroTokenizador> {
        let bruto = std::fs::read(caminho).map_err(|e| ErroTokenizador::Io(format!("{}: {e}", caminho.display())))?;
        Self::de_bytes(&bruto)
    }

    /// Como [`Self::carrega`], a partir do conteúdo já lido.
    pub fn de_bytes(bruto: &[u8]) -> Result<Self, ErroTokenizador> {
        let j: serde_json::Value = serde_json::from_slice(bruto).map_err(|e| ErroTokenizador::Json(e.to_string()))?;
        if !j["normalizer"].is_null() {
            return fmt_err("normalizer não nulo");
        }
        let pre = &j["pre_tokenizer"];
        if pre["type"] != "ByteLevel" || pre["add_prefix_space"] != false || pre["use_regex"] != true {
            return fmt_err("pre_tokenizer diferente de ByteLevel{add_prefix_space:false, use_regex:true}");
        }
        if j["decoder"]["type"] != "ByteLevel" {
            return fmt_err("decoder diferente de ByteLevel");
        }
        let m = &j["model"];
        if m["type"] != "BPE" {
            return fmt_err("model diferente de BPE");
        }
        for (campo, ok) in [
            ("dropout", m["dropout"].is_null()),
            ("unk_token", m["unk_token"].is_null()),
            ("continuing_subword_prefix", m["continuing_subword_prefix"].as_str().is_none_or(str::is_empty)),
            ("end_of_word_suffix", m["end_of_word_suffix"].as_str().is_none_or(str::is_empty)),
            ("byte_fallback", m["byte_fallback"] != true),
            ("ignore_merges", m["ignore_merges"] != true),
        ] {
            if !ok {
                return fmt_err(format!("model.{campo} fora do suportado"));
            }
        }
        let vocab = m["vocab"].as_object().ok_or_else(|| ErroTokenizador::Formato("model.vocab ausente".into()))?;
        let mut max_id = 0u32;
        let mut pares: Vec<(&str, u32)> = Vec::with_capacity(vocab.len());
        for (t, v) in vocab {
            let id = v.as_u64().filter(|&x| x <= u32::MAX as u64).ok_or_else(|| ErroTokenizador::Formato(format!("id inválido para {t:?}")))? as u32;
            max_id = max_id.max(id);
            pares.push((t, id));
        }
        let mut adicionados = Vec::new();
        for a in j["added_tokens"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
            let (Some(conteudo), Some(id)) = (a["content"].as_str(), a["id"].as_u64()) else {
                return fmt_err("added_tokens com entrada sem content/id");
            };
            if a["lstrip"] != false || a["rstrip"] != false || a["single_word"] != false || a["normalized"] != false || conteudo.is_empty() {
                return fmt_err(format!("token adicionado {conteudo:?} com lstrip/rstrip/single_word/normalized fora do suportado"));
            }
            if id > u32::MAX as u64 {
                return fmt_err("id de token adicionado grande demais");
            }
            max_id = max_id.max(id as u32);
            adicionados.push(Adicionado { conteudo: conteudo.to_string(), id: id as u32 });
        }
        // recusa ids esparsos/desproporcionais antes de alocar vetores densos indexados pelo maior id
        let n_entradas = pares.len() + adicionados.len();
        let limite = n_entradas.saturating_mul(2).saturating_add(1024);
        let tam = (max_id as usize).saturating_add(1);
        if tam > limite {
            return fmt_err(format!("ids esparsos: maior id {max_id} para {n_entradas} entradas"));
        }
        let mapa = bytes_para_chars();
        let mut char_para_byte: HashMap<char, u8> = HashMap::new();
        for (b, &c) in mapa.iter().enumerate() {
            char_para_byte.insert(c, b as u8);
        }
        let mut id_para_token: Vec<Option<String>> = vec![None; tam];
        let mut id_para_bytes: Vec<Option<Vec<u8>>> = vec![None; tam];
        for (t, id) in &pares {
            let mut bs = Vec::with_capacity(t.len());
            let mut ok = true;
            for c in t.chars() {
                match char_para_byte.get(&c) {
                    Some(&b) => bs.push(b),
                    None => ok = false,
                }
            }
            id_para_token[*id as usize] = Some((*t).to_string());
            // token com caractere fora do alfabeto ByteLevel: não é detokenizável como bytes
            id_para_bytes[*id as usize] = if ok { Some(bs) } else { None };
        }
        for a in &adicionados {
            id_para_bytes[a.id as usize] = Some(a.conteudo.as_bytes().to_vec());
            if id_para_token[a.id as usize].is_none() {
                id_para_token[a.id as usize] = Some(a.conteudo.clone());
            }
        }
        let por_texto: HashMap<&str, u32> = pares.iter().copied().collect();
        let mut id_do_byte = [None::<u32>; 256];
        for (b, &c) in mapa.iter().enumerate() {
            let mut buf = [0u8; 4];
            let s: &str = c.encode_utf8(&mut buf);
            id_do_byte[b] = por_texto.get(s).copied();
        }
        let lista = m["merges"].as_array().ok_or_else(|| ErroTokenizador::Formato("model.merges ausente".into()))?;
        let mut merges = HashMap::with_capacity(lista.len());
        for (rank, mg) in lista.iter().enumerate() {
            let (a, b) = if let Some(s) = mg.as_str() {
                let mut it = s.splitn(2, ' ');
                match (it.next(), it.next()) {
                    (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                    _ => return fmt_err(format!("merge {rank} sem espaço")),
                }
            } else if let Some(v) = mg.as_array() {
                match (v.first().and_then(|x| x.as_str()), v.get(1).and_then(|x| x.as_str()), v.len()) {
                    (Some(a), Some(b), 2) => (a.to_string(), b.to_string()),
                    _ => return fmt_err(format!("merge {rank} malformado")),
                }
            } else {
                return fmt_err(format!("merge {rank} malformado"));
            };
            let (Some(&ia), Some(&ib)) = (por_texto.get(a.as_str()), por_texto.get(b.as_str())) else {
                return fmt_err(format!("merge {rank} usa token fora do vocabulário"));
            };
            let juntos = format!("{a}{b}");
            let Some(&novo) = por_texto.get(juntos.as_str()) else {
                return fmt_err(format!("merge {rank} gera token fora do vocabulário"));
            };
            // como o `tokenizers`: um par repetido fica com o último rank
            merges.insert(chave(ia, ib), (rank as u32, novo));
        }
        Ok(Self { id_para_token, id_para_bytes, id_do_byte, merges, adicionados, cache: Mutex::new(HashMap::new()) })
    }

    /// Tamanho da tabela de ids (maior id mais um).
    pub fn tamanho_vocabulario(&self) -> usize {
        self.id_para_token.len()
    }

    /// BPE por rank de merges sobre os bytes de um pedaço pré-tokenizado.
    fn bpe(&self, pedaco: &[u8]) -> Vec<u32> {
        // bytes sem caractere no vocabulário são descartados (igual ao `tokenizers` sem unk_token); em UTF-8 válido só o 0xF1
        // (U+40000..U+7FFFF, não atribuídos) cai aqui no Granite, que não tem os caracteres de 0xC0, 0xC1, 0xF1 e 0xF5..0xFF.
        let mut id: Vec<u32> = pedaco.iter().filter_map(|&b| self.id_do_byte[b as usize]).collect();
        let n = id.len();
        if n == 0 {
            return Vec::new();
        }
        const NENHUM: usize = usize::MAX;
        let mut prox: Vec<usize> = (0..n).map(|i| if i + 1 < n { i + 1 } else { NENHUM }).collect();
        let mut ant: Vec<usize> = (0..n).map(|i| if i > 0 { i - 1 } else { NENHUM }).collect();
        let mut vivo = vec![true; n];
        let mut heap: BinaryHeap<Reverse<(u32, usize, u32, u32)>> = BinaryHeap::new();
        for i in 0..n.saturating_sub(1) {
            if let Some(&(r, _)) = self.merges.get(&chave(id[i], id[i + 1])) {
                heap.push(Reverse((r, i, id[i], id[i + 1])));
            }
        }
        while let Some(Reverse((_, pos, l, r))) = heap.pop() {
            if !vivo[pos] || id[pos] != l {
                continue;
            }
            let nx = prox[pos];
            if nx == NENHUM || !vivo[nx] || id[nx] != r {
                continue;
            }
            let Some(&(_, novo)) = self.merges.get(&chave(l, r)) else { continue };
            id[pos] = novo;
            vivo[nx] = false;
            prox[pos] = prox[nx];
            if prox[nx] != NENHUM {
                ant[prox[nx]] = pos;
            }
            if ant[pos] != NENHUM {
                let a = ant[pos];
                if let Some(&(rk, _)) = self.merges.get(&chave(id[a], id[pos])) {
                    heap.push(Reverse((rk, a, id[a], id[pos])));
                }
            }
            if prox[pos] != NENHUM {
                let p = prox[pos];
                if let Some(&(rk, _)) = self.merges.get(&chave(id[pos], id[p])) {
                    heap.push(Reverse((rk, pos, id[pos], id[p])));
                }
            }
        }
        let mut saida = Vec::new();
        let mut i = 0;
        while i != NENHUM {
            saida.push(id[i]);
            i = prox[i];
        }
        saida
    }

    fn codifica_pedaco(&self, pedaco: &str, saida: &mut Vec<u32>) {
        let bytes = pedaco.as_bytes();
        if let Ok(c) = self.cache.lock() {
            if let Some(ids) = c.get(bytes) {
                saida.extend_from_slice(ids);
                return;
            }
        }
        let ids = self.bpe(bytes);
        saida.extend_from_slice(&ids);
        if let Ok(mut c) = self.cache.lock() {
            if c.len() < CACHE_MAX {
                c.insert(bytes.to_vec(), ids);
            }
        }
    }

    /// Texto para ids, igual a `tok(text).input_ids` do HF (sem BOS: `add_bos_token` é falso e o post-processor não acrescenta nada).
    pub fn codifica(&self, texto: &str) -> Vec<u32> {
        let mut saida = Vec::new();
        let mut resto_ini = 0;
        let mut i = 0;
        let bs = texto.as_bytes();
        while i < bs.len() {
            // maior casamento mais à esquerda entre os tokens adicionados
            let mut melhor: Option<&Adicionado> = None;
            if texto.is_char_boundary(i) {
                for a in &self.adicionados {
                    if texto[i..].starts_with(a.conteudo.as_str()) && melhor.is_none_or(|m| a.conteudo.len() > m.conteudo.len()) {
                        melhor = Some(a);
                    }
                }
            }
            match melhor {
                Some(a) => {
                    for p in pre_tokeniza(&texto[resto_ini..i]) {
                        self.codifica_pedaco(p, &mut saida);
                    }
                    saida.push(a.id);
                    i += a.conteudo.len();
                    resto_ini = i;
                }
                None => i += 1,
            }
        }
        for p in pre_tokeniza(&texto[resto_ini..]) {
            self.codifica_pedaco(p, &mut saida);
        }
        saida
    }

    /// Ids para os bytes exatos (inverso do ByteLevel; tokens especiais viram o texto literal).
    pub fn decodifica_bytes(&self, ids: &[u32]) -> Result<Vec<u8>, ErroTokenizador> {
        let mut saida = Vec::new();
        for &id in ids {
            match self.id_para_bytes.get(id as usize) {
                Some(Some(b)) => saida.extend_from_slice(b),
                _ => return Err(ErroTokenizador::IdDesconhecido(id)),
            }
        }
        Ok(saida)
    }

    /// Ids para `String`; UTF-8 inválido na sequência inteira vira `U+FFFD` (ver a documentação do módulo).
    pub fn decodifica(&self, ids: &[u32]) -> Result<String, ErroTokenizador> {
        Ok(String::from_utf8_lossy(&self.decodifica_bytes(ids)?).into_owned())
    }
}

/// Papel de uma mensagem do chat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Papel {
    Sistema,
    Usuario,
    Assistente,
}

impl Papel {
    fn rotulo(self) -> &'static str {
        match self {
            Self::Sistema => "system",
            Self::Usuario => "user",
            Self::Assistente => "assistant",
        }
    }
}

/// Uma mensagem do chat.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mensagem {
    pub papel: Papel,
    pub conteudo: String,
}

const MARCA_INI: &str = "<|start_of_role|>";
const MARCA_ROLE_FIM: &str = "<|end_of_role|>";
const MARCA_FIM: &str = "<|end_of_text|>";

/// Texto da mensagem de sistema que o template injeta quando a primeira mensagem não é `system`.
pub fn sistema_padrao(data: &str) -> String {
    format!("Knowledge Cutoff Date: April 2024.\nToday's Date: {data}.\nYou are Granite, developed by IBM. You are a helpful AI assistant.")
}

/// Renderização fixa do chat template do Granite 3.1 (equivale a `apply_chat_template(..., tokenize=False)` sem `tools`,
/// `documents` nem `controls`). `data` é o texto de `strftime_now('%B %d, %Y')` (ver [`formata_data`]).
///
/// Como o template: se a primeira mensagem é `system`, ela vira a mensagem de sistema e as demais formam o laço; senão entra
/// [`sistema_padrao`] e todas formam o laço. Com `add_generation_prompt`, o aviso do assistente só é acrescentado depois da
/// última mensagem do laço (se o laço estiver vazio, não há aviso, como no template). Lista vazia é erro (o template falharia
/// em `messages[0]`).
pub fn renderiza_chat(mensagens: &[Mensagem], add_generation_prompt: bool, data: &str) -> Result<String, ErroTokenizador> {
    let Some(primeira) = mensagens.first() else {
        return Err(ErroTokenizador::Mensagens("lista vazia".into()));
    };
    let (sistema, laco) = if primeira.papel == Papel::Sistema {
        (primeira.conteudo.clone(), &mensagens[1..])
    } else {
        (sistema_padrao(data), mensagens)
    };
    let mut s = String::new();
    s.push_str(MARCA_INI);
    s.push_str("system");
    s.push_str(MARCA_ROLE_FIM);
    s.push_str(&sistema);
    s.push_str(MARCA_FIM);
    s.push('\n');
    for (i, m) in laco.iter().enumerate() {
        s.push_str(MARCA_INI);
        s.push_str(m.papel.rotulo());
        s.push_str(MARCA_ROLE_FIM);
        s.push_str(&m.conteudo);
        s.push_str(MARCA_FIM);
        s.push('\n');
        if i + 1 == laco.len() && add_generation_prompt {
            s.push_str(MARCA_INI);
            s.push_str("assistant");
            s.push_str(MARCA_ROLE_FIM);
        }
    }
    Ok(s)
}

/// Formata a data como `strftime('%B %d, %Y')` em inglês (`"November 07, 2026"`); valida mês e dia (calendário gregoriano).
pub fn formata_data(ano: i32, mes: u32, dia: u32) -> Result<String, ErroTokenizador> {
    const MESES: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
    if !(1..=12).contains(&mes) {
        return Err(ErroTokenizador::Data(format!("mês {mes}")));
    }
    let bissexto = (ano % 4 == 0 && ano % 100 != 0) || ano % 400 == 0;
    let dias = match mes {
        2 if bissexto => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    if dia == 0 || dia > dias || !(1..=9999).contains(&ano) {
        return Err(ErroTokenizador::Data(format!("{ano}-{mes}-{dia}")));
    }
    Ok(format!("{} {:02}, {:04}", MESES[mes as usize - 1], dia, ano))
}

#[cfg(test)]
mod testes {
    use super::*;
    use crate::npy::Npy;
    use std::path::PathBuf;

    /// SHA-256 do `tokenizer.json` do Granite 3.1 (1B e 3B idênticos).
    const HASH_TOKENIZER: &str = "ed944c0b3d71e2d651a5ba0f8e7b350faff30031c8e3353de70b12b9c1bf5654";
    /// SHA-256 de `casos.json`; preenchido depois de gerar a referência (critério-5a-tokenizador.txt).
    const HASH_CASOS: &str = "72a6e20ced30f0fc10db00b213c3481b69b4629b3225ad386cadc26bd3cc3efb";
    /// SHA-256 do critério congelado.
    const HASH_CRITERIO: &str = "f35f6232f3a36ebd5f48ecaa0b29b860030e3fd96ded26b7cec1004c654cedc0";
    const TOKENIZER_1B: &str = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct/tokenizer.json";

    fn raiz() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn sha256(p: &Path) -> String {
        let o = std::process::Command::new("sha256sum").arg(p).output().expect("sha256sum");
        assert!(o.status.success(), "sha256sum falhou em {}", p.display());
        String::from_utf8_lossy(&o.stdout).split_whitespace().next().unwrap_or("").to_string()
    }

    fn carrega() -> Tokenizador {
        let p = Path::new(TOKENIZER_1B);
        assert_eq!(sha256(p), HASH_TOKENIZER, "SEM PROVA: tokenizer.json mudou de hash");
        Tokenizador::carrega(p).expect("carrega tokenizer.json")
    }

    fn casos_json() -> serde_json::Value {
        let p = raiz().join("qual-runs/A2-LOTE-20261103/5a-referencia/casos.json");
        assert_eq!(sha256(&p), HASH_CASOS, "SEM PROVA: casos.json mudou de hash");
        serde_json::from_slice(&std::fs::read(&p).expect("lê casos.json")).expect("casos.json válido")
    }

    fn ids_de(v: &serde_json::Value) -> Vec<u32> {
        v.as_array().expect("lista de ids").iter().map(|x| x.as_u64().expect("id") as u32).collect()
    }

    fn msgs_de(v: &serde_json::Value) -> Vec<Mensagem> {
        v.as_array()
            .expect("mensagens")
            .iter()
            .map(|m| Mensagem {
                papel: match m["role"].as_str().expect("role") {
                    "system" => Papel::Sistema,
                    "user" => Papel::Usuario,
                    "assistant" => Papel::Assistente,
                    r => panic!("papel {r}"),
                },
                conteudo: m["content"].as_str().expect("content").to_string(),
            })
            .collect()
    }

    #[test]
    fn hashes_congelados() {
        assert_eq!(sha256(&raiz().join("qual-runs/A2-LOTE-20261103/criterio-5a-tokenizador.txt")), HASH_CRITERIO, "SEM PROVA: critério mudou de hash");
        let _ = casos_json();
        let _ = carrega();
    }

    #[test]
    fn pre_tokenizacao_gpt2_basica() {
        assert_eq!(pre_tokeniza("Hello world"), vec!["Hello", " world"]);
        assert_eq!(pre_tokeniza("don't I'll he's"), vec!["don", "'t", " I", "'ll", " he", "'s"]);
        assert_eq!(pre_tokeniza("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(pre_tokeniza("a \n b"), vec!["a", " \n", " b"]);
        assert_eq!(pre_tokeniza("x  "), vec!["x", "  "]);
        assert_eq!(pre_tokeniza(" 12ab!?"), vec![" 12", "ab", "!?"]);
        assert_eq!(pre_tokeniza(""), Vec::<&str>::new());
    }

    #[test]
    fn mapa_bytes_gpt2() {
        let m = bytes_para_chars();
        assert_eq!(m[b'A' as usize], 'A');
        assert_eq!(m[b' ' as usize], 'Ġ');
        assert_eq!(m[b'\n' as usize], 'Ċ');
        assert_eq!(m[0], 'Ā');
        assert_eq!(m[0xAD], 'Ń');
    }

    #[test]
    fn formata_data_valida() {
        assert_eq!(formata_data(2026, 11, 7).unwrap(), "November 07, 2026");
        assert_eq!(formata_data(2024, 2, 29).unwrap(), "February 29, 2024");
        assert!(formata_data(2026, 2, 29).is_err());
        assert!(formata_data(2026, 13, 1).is_err());
        assert!(formata_data(2026, 4, 31).is_err());
    }

    #[test]
    fn renderiza_chat_erros_e_forma() {
        assert!(renderiza_chat(&[], true, "x").is_err());
        let u = Mensagem { papel: Papel::Usuario, conteudo: "oi".into() };
        let r = renderiza_chat(&[u], true, "November 07, 2026").unwrap();
        assert_eq!(
            r,
            "<|start_of_role|>system<|end_of_role|>Knowledge Cutoff Date: April 2024.\nToday's Date: November 07, 2026.\nYou are Granite, developed by IBM. You are a helpful AI assistant.<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>oi<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>"
        );
    }

    #[test]
    fn entradas_estranhas_sem_panic() {
        let t = carrega();
        for s in ["", "\u{0}", "<", "<|", "<|end_of_tex", "\u{10FFFF}", "\u{FFFD}", "a\u{301}", "\r\n\r\n", "\u{200B}"] {
            let ids = t.codifica(s);
            assert_eq!(t.decodifica_bytes(&ids).unwrap(), s.as_bytes(), "ida e volta {s:?}");
        }
        assert_eq!(t.decodifica_bytes(&[u32::MAX]), Err(ErroTokenizador::IdDesconhecido(u32::MAX)));
        assert!(t.decodifica(&[49155]).is_err());
        assert!(Tokenizador::de_bytes(b"{").is_err());
        assert!(Tokenizador::de_bytes(b"{}").is_err());
    }

    /// Regressão r94 (P2): id gigante no vocabulário ou em token adicionado vira erro tipado, sem alocar vetores enormes.
    #[test]
    fn ids_esparsos_recusados_sem_alocar() {
        let base = std::fs::read_to_string(TOKENIZER_1B).expect("tokenizer.json do Granite");
        let mut j: serde_json::Value = serde_json::from_str(&base).unwrap();
        let (chave, _) = j["model"]["vocab"].as_object().unwrap().iter().next().map(|(k, v)| (k.clone(), v.clone())).unwrap();
        j["model"]["vocab"][chave.as_str()] = serde_json::json!(4294967295u64);
        let r = Tokenizador::de_bytes(serde_json::to_vec(&j).unwrap().as_slice());
        assert!(matches!(r, Err(ErroTokenizador::Formato(_))), "vocab com u32::MAX");
        let mut j: serde_json::Value = serde_json::from_str(&base).unwrap();
        j["added_tokens"][0]["id"] = serde_json::json!(4294967295u64);
        let r = Tokenizador::de_bytes(serde_json::to_vec(&j).unwrap().as_slice());
        assert!(matches!(r, Err(ErroTokenizador::Formato(_))), "added_token com u32::MAX");
        let mut j: serde_json::Value = serde_json::from_str(&base).unwrap();
        j["added_tokens"][0]["id"] = serde_json::json!(10_000_000u64);
        let r = Tokenizador::de_bytes(serde_json::to_vec(&j).unwrap().as_slice());
        assert!(matches!(r, Err(ErroTokenizador::Formato(_))), "added_token com id muito maior que o vocabulário");
        assert!(Tokenizador::de_bytes(base.as_bytes()).is_ok(), "tokenizer congelado continua carregando");
    }

    /// Mede e declara onde `std` diverge das classes `\p{L}`/`\p{N}` da referência (não falha por causa da diferença; falha se a
    /// diferença deixar de ser do tipo documentado: toda letra `\p{L}` tem de ser `is_alphabetic`).
    #[test]
    fn std_vs_tabelas_unicode() {
        let (mut alfa_nao_letra, mut letra_nao_alfa, mut num_nao_std, mut std_nao_num) = (0u32, 0u32, 0u32, 0u32);
        let (mut ex_a, mut ex_n) = (Vec::new(), Vec::new());
        for v in 0..=0x10FFFFu32 {
            let Some(c) = char::from_u32(v) else { continue };
            let (l, n) = (eh_letra(c), eh_numero(c));
            if c.is_alphabetic() && !l {
                alfa_nao_letra += 1;
                if ex_a.len() < 6 {
                    ex_a.push(format!("U+{v:04X}"));
                }
            }
            if l && !c.is_alphabetic() {
                letra_nao_alfa += 1;
            }
            if n != c.is_numeric() {
                if n {
                    num_nao_std += 1;
                } else {
                    std_nao_num += 1;
                }
                if ex_n.len() < 6 {
                    ex_n.push(format!("U+{v:04X}"));
                }
            }
            if c.is_whitespace() && (l || n) {
                panic!("espaço classificado como letra/número: U+{v:04X}");
            }
        }
        eprintln!("std_vs_tabelas: is_alphabetic sem \\p{{L}}: {alfa_nao_letra} (ex. {ex_a:?}); \\p{{L}} sem is_alphabetic: {letra_nao_alfa}; \\p{{N}} sem is_numeric: {num_nao_std}; is_numeric sem \\p{{N}}: {std_nao_num} (ex. {ex_n:?})");
        assert_eq!(letra_nao_alfa, 0, "\\p{{L}} deveria estar contido em is_alphabetic");
    }

    #[test]
    fn prova_5a_ids_template_roundtrip_decode() {
        let t = carrega();
        let j = casos_json();
        let casos = j["casos"].as_array().expect("casos");
        assert!(!casos.is_empty());
        type Contagem = std::collections::BTreeMap<String, (u32, u32)>;
        fn ocorreu(por_cat: &mut Contagem, cat: &str, ok: bool, msg: &dyn Fn() -> String, primeiro: &mut Option<String>) {
            let e = por_cat.entry(cat.to_string()).or_default();
            e.0 += 1;
            if ok {
                e.1 += 1;
            } else if primeiro.is_none() {
                *primeiro = Some(msg());
            }
        }
        let mut por_cat: Contagem = Default::default();
        let mut primeiro: Option<String> = None;
        let mut declarados: Vec<String> = Vec::new();
        let mut forcados: Vec<String> = Vec::new();
        for c in casos {
            let nome = c["nome"].as_str().expect("nome");
            let cat = c["categoria"].as_str().expect("categoria");
            let texto: String = if c["tipo"] == "chat" {
                let ms = msgs_de(&c["mensagens"]);
                let gp = c["add_generation_prompt"].as_bool().expect("agp");
                let data = c["data"].as_str().expect("data");
                let r = renderiza_chat(&ms, gp, data).expect("renderiza");
                let esperado = c["renderizado"].as_str().expect("renderizado");
                ocorreu(&mut por_cat, &format!("{cat}/template"), r == esperado, &|| format!("{nome}: template diverge\n rust: {r:?}\n   hf: {esperado:?}"), &mut primeiro);
                esperado.to_string()
            } else {
                c["texto"].as_str().expect("texto").to_string()
            };
            let ids = t.codifica(&texto);
            let ref_ids = ids_de(&c["ids"]);
            ocorreu(&mut por_cat, &format!("{cat}/ids"), ids == ref_ids, &|| {
                let i = ids.iter().zip(&ref_ids).position(|(a, b)| a != b).unwrap_or(ids.len().min(ref_ids.len()));
                format!("{nome}: ids divergem no índice {i} (rust {} ids, hf {} ids; rust[{i}]={:?} hf[{i}]={:?})", ids.len(), ref_ids.len(), ids.get(i), ref_ids.get(i))
            }, &mut primeiro);
            let volta = t.decodifica_bytes(&ids).expect("decodifica");
            ocorreu(&mut por_cat, &format!("{cat}/ida-e-volta"), volta == texto.bytes().filter(|&b| b != 0xF1).collect::<Vec<u8>>(), &|| format!("{nome}: ida e volta em bytes diverge"), &mut primeiro);
            let dec = t.decodifica(&ref_ids).expect("decodifica");
            let sem = c["decode_sem_cleanup"].as_str().expect("decode_sem_cleanup");
            ocorreu(&mut por_cat, &format!("{cat}/decode-sem-cleanup"), dec == sem, &|| format!("{nome}: decode sem clean_up diverge\n rust: {dec:?}\n   hf: {sem:?}"), &mut primeiro);
            let com = c["decode_com_cleanup"].as_str().expect("decode_com_cleanup");
            if dec != com {
                declarados.push(format!("{nome} (decode padrão do HF)"));
            }
            if dec != c["decode_com_cleanup_forcado"].as_str().expect("forcado") {
                forcados.push(nome.to_string());
            }
        }
        // decodificações de ids gerados (guloso) e de sequências cortadas no meio de um caractere
        for d in j["decodificacoes"].as_array().expect("decodificacoes") {
            let nome = d["nome"].as_str().expect("nome");
            let cat = d["categoria"].as_str().expect("categoria");
            let ids = ids_de(&d["ids"]);
            let dec = t.decodifica(&ids).expect("decodifica");
            let sem = d["decode_sem_cleanup"].as_str().expect("sem");
            if cat == "decode-cortado" {
                // só medida: o HF substitui por token, aqui por sequência
                let n = por_cat.entry("decode-cortado (medida)".into()).or_default();
                n.0 += 1;
                if dec == sem {
                    n.1 += 1;
                }
                continue;
            }
            ocorreu(&mut por_cat, &format!("{cat}/decode-sem-cleanup"), dec == sem, &|| format!("{nome}: decode diverge\n rust: {dec:?}\n   hf: {sem:?}"), &mut primeiro);
            if dec != d["decode_com_cleanup_forcado"].as_str().expect("forcado") {
                forcados.push(nome.to_string());
            }
            if let Some(g) = d["generated_text"].as_str() {
                if dec != g {
                    declarados.push(format!("{nome} (vs generated_text)"));
                }
            }
        }
        // 277 ids de texto-a (tokens.npy)
        let npy = Npy::read(&raiz().join("qual-runs/REF-granite-1b-texto-a/tokens.npy")).expect("npy");
        let esperados: Vec<u32> = match npy.descr.as_str() {
            "<i4" => npy.i32().expect("i32").iter().map(|&x| x as u32).collect(),
            "<i8" => npy.data.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32).collect(),
            d => panic!("dtype {d}"),
        };
        let texto_a = std::fs::read_to_string(raiz().join("tools/ref-texts/texto-a.txt")).expect("texto-a");
        let meus = t.codifica(&texto_a);
        ocorreu(&mut por_cat, "texto-a.npy", meus == esperados && esperados.len() == 277, &|| format!("tokens.npy de texto-a diverge: rust {} ids, npy {} ids", meus.len(), esperados.len()), &mut primeiro);

        let mut relatorio = String::new();
        for (k, (n, ok)) in &por_cat {
            relatorio.push_str(&format!("{k}: {ok}/{n}\n"));
        }
        eprintln!("PROVA 5a por categoria/verificação:\n{relatorio}diferenças só declaradas (decode padrão com clean_up != sem): {}", declarados.len());
        relatorio.push_str(&format!("diferença para o clean_up legado FORÇADO (só medida): {} (ex.: {:?})\n", forcados.len(), &forcados[..forcados.len().min(5)]));
        eprintln!("{}", relatorio.lines().last().unwrap_or(""));
        if let Ok(dir) = std::env::var("PROVA_5A_SAIDA") {
            let _ = std::fs::write(Path::new(&dir).join("contagens.txt"), format!("{relatorio}declaradas_clean_up={}\n{}\n", declarados.len(), declarados.join("\n")));
        }
        if let Some(m) = primeiro {
            panic!("primeiro caso divergente: {m}");
        }
    }
}
