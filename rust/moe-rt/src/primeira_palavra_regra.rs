//! A2 (textos longos), etapa 5, degrau 5b: a PRIMEIRA PALAVRA a partir de texto cru. Regras puras (sem GPU, testadas em CPU), o fluxo
//! texto -> template -> ids -> `le_ids` -> logits -> argmax -> detokenização -> entrega (genérico no motor, testado com o motor falso) e os
//! registros em JSON. Critério congelado: `qual-runs/A2-LOTE-20261103/criterio-5b-primeira-palavra.txt`. Recorte:
//! `qual-runs/A2-LOTE-20261103/recorte-etapa5-interface-de-texto-20261107.md`.
//!
//! Rótulos da velocidade: ATINGE, NÃO ATINGE e SEM PROVA ([`crate::velocidade_regra::Rotulo`]); da correção: PASSA, REPROVA e SEM PROVA
//! ([`RotuloCorrecao`]). O quente por pedido INCLUI `T_sequencia` (o Engine não tem reset): o resultado é descritivo e vai ao Breno.
//!
//! Escolhas de implementação que o critério não diz (registradas para a revisão):
//! - a mediana de número par é a média dos dois centrais (a de `e4b_regra`); o desvio é o amostral (n - 1), só descrição;
//! - o quente não tem "modo 1": não há ganho nem deriva; o portão por célula é a igualdade de token, hash dos ids e contadores entre as 5
//!   repetições, com hits e cpu positivos;
//! - a faixa de ids é `|n - alvo| <= 2% do alvo` (comparação em ponto flutuante, o limite entra);
//! - empate da correção: gap da referência estritamente menor que 1e-3 e o argmax do Lastro igual ao segundo colocado da referência.
use crate::e4b_regra::{argmax, mediana};
use crate::leitura_regra::{contadores_validos, Contadores};
use crate::motor::backend::Backend;
use crate::motor::engine::{Engine, Published};
use crate::motor::leitura::{le_ids_com_pendente, PreparaPasso};
use crate::tokenizador::{formata_data, renderiza_chat, Mensagem, Papel, Tokenizador};
use crate::velocidade_regra::{self as vr, Cen, Estado, RegistroFrio, Rotulo};
use serde_json::{json, Value};
use std::fmt::Debug;
use std::io::Write;
use std::time::Instant;

/// A data fixa do template nos modos de prova (correção e velocidade): 2026-11-07, "November 07, 2026".
pub const DATA_FIXA: (i32, u32, u32) = (2026, 11, 7);
/// Empate da correção: gap da referência estritamente menor que isto (a regra do M1b).
pub const GAP_EMPATE: f64 = 1e-3;
/// Pedidos da correção por modelo (os 3 primeiros de `prompts-traco.json`).
pub const PEDIDOS_CORRECAO: usize = 3;
/// Repetições válidas do quente por célula (além de 1 de aquecimento descartada).
pub const REPS_QUENTE: usize = 5;
/// Tolerância da contagem de ids dos textos de velocidade sobre o alvo do cenário.
pub const TOLERANCIA_IDS: f64 = 0.02;

/// A data fixa já formatada como o template pede.
pub fn data_fixa_texto() -> String {
    formata_data(DATA_FIXA.0, DATA_FIXA.1, DATA_FIXA.2).expect("a data fixa é válida")
}

/// Converte dias desde 1970-01-01 em (ano, mês, dia) do calendário gregoriano (algoritmo de Howard Hinnant).
pub fn civil_de_dias(dias: i64) -> (i32, u32, u32) {
    let z = dias + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((y + (m <= 2) as i64) as i32, m, d)
}

/// A data de hoje em UTC (para o modo livre; os modos de prova usam [`DATA_FIXA`]).
pub fn data_de_hoje_utc() -> (i32, u32, u32) {
    let s = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    civil_de_dias((s / 86_400) as i64)
}

/// A contagem de ids do texto de velocidade está dentro de 2% do alvo do cenário?
pub fn ids_na_faixa(n: usize, cen: Cen) -> bool {
    let alvo = cen.n() as f64;
    (n as f64 - alvo).abs() <= TOLERANCIA_IDS * alvo
}

/// Os dois maiores logits: (id do primeiro, id do segundo, diferença). `None` com menos de 2 valores ou valor não finito entre os dois.
pub fn dois_maiores(logits: &[f32]) -> Option<(usize, usize, f64)> {
    if logits.len() < 2 {
        return None;
    }
    let a = argmax(logits);
    let b = logits.iter().enumerate().filter(|(i, _)| *i != a).max_by(|x, y| x.1.total_cmp(y.1)).map(|(i, _)| i)?;
    let gap = logits[a] as f64 - logits[b] as f64;
    gap.is_finite().then_some((a, b, gap))
}

/// FNV-1a dos ids (só identidade, não é hash de segurança).
pub fn fnv_ids(ids: &[usize]) -> u64 {
    ids.iter().fold(0xcbf29ce484222325u64, |h, x| (x.to_le_bytes()).iter().fold(h, |h, b| (h ^ *b as u64).wrapping_mul(0x100000001b3)))
}

// ---- Cronômetros -----------------------------------------------------------------------------------------------------------------------

/// Os oito cronômetros de uma repetição, em milissegundos, separados.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Cronometros {
    pub entrada_ms: f64,
    pub token_ms: f64,
    pub sequencia_ms: f64,
    pub leitura_ms: f64,
    pub logits_ms: f64,
    pub escolha_ms: f64,
    pub detok_ms: f64,
    pub entrega_ms: f64,
}

impl Cronometros {
    fn campos(&self) -> [(&'static str, f64); 8] {
        [("T_entrada", self.entrada_ms), ("T_token", self.token_ms), ("T_sequencia", self.sequencia_ms), ("T_leitura", self.leitura_ms), ("T_logits", self.logits_ms), ("T_escolha", self.escolha_ms), ("T_detok", self.detok_ms), ("T_entrega", self.entrega_ms)]
    }
    /// A soma das etapas do produto (sem T_entrada, que antecede o indicador): só conferência.
    pub fn soma_etapas_ms(&self) -> f64 {
        self.token_ms + self.sequencia_ms + self.leitura_ms + self.logits_ms + self.escolha_ms + self.detok_ms + self.entrega_ms
    }
    /// Válido: todos finitos e não negativos, com leitura e logits positivos.
    pub fn valido(&self) -> bool {
        self.campos().iter().all(|(_, v)| v.is_finite() && *v >= 0.0) && self.leitura_ms > 0.0 && self.logits_ms > 0.0
    }
    pub fn para_json(&self) -> Value {
        Value::Object(self.campos().iter().map(|(k, v)| (k.to_string(), Value::from(*v))).collect())
    }
    pub fn de_json(v: &Value) -> Result<Self, String> {
        let g = |k: &str| num(v, k);
        Ok(Cronometros { entrada_ms: g("T_entrada")?, token_ms: g("T_token")?, sequencia_ms: g("T_sequencia")?, leitura_ms: g("T_leitura")?, logits_ms: g("T_logits")?, escolha_ms: g("T_escolha")?, detok_ms: g("T_detok")?, entrega_ms: g("T_entrega")? })
    }
}

fn num(v: &Value, k: &str) -> Result<f64, String> {
    v.get(k).and_then(Value::as_f64).ok_or_else(|| format!("falta ou ilegível: {k}"))
}

fn inteiro(v: &Value, k: &str) -> Result<usize, String> {
    v.get(k).and_then(Value::as_u64).map(|x| x as usize).ok_or_else(|| format!("falta ou ilegível: {k}"))
}

fn texto(v: &Value, k: &str) -> Result<String, String> {
    v.get(k).and_then(Value::as_str).map(str::to_string).ok_or_else(|| format!("falta ou ilegível: {k}"))
}

fn booleano(v: &Value, k: &str) -> Result<bool, String> {
    v.get(k).and_then(Value::as_bool).ok_or_else(|| format!("falta ou ilegível: {k}"))
}

fn lista_ids(v: &Value, k: &str) -> Result<Vec<usize>, String> {
    v.get(k).and_then(Value::as_array).ok_or_else(|| format!("falta ou ilegível: {k}"))?.iter().map(|x| x.as_u64().map(|n| n as usize).ok_or_else(|| format!("id ilegível em {k}"))).collect()
}

fn pares(v: &Value, k: &str) -> Result<Vec<(String, String)>, String> {
    match v.get(k) {
        None => Ok(vec![]),
        Some(Value::Object(m)) => m.iter().map(|(a, b)| b.as_str().map(|s| (a.clone(), s.to_string())).ok_or_else(|| format!("{k}.{a} ilegível"))).collect(),
        _ => Err(format!("{k} ilegível")),
    }
}

fn pares_json(p: &[(String, String)]) -> Value {
    Value::Object(p.iter().map(|(a, b)| (a.clone(), Value::from(b.clone()))).collect())
}

// ---- A medida de uma repetição ---------------------------------------------------------------------------------------------------------

/// O que uma repetição (um pedido, motor e backend novos) deixa para a decisão: tempos, ids, token, palavra, contadores.
#[derive(Clone, Debug, PartialEq)]
pub struct RepMedida {
    pub crono: Cronometros,
    /// Do instante em que o texto cru estava disponível até a palavra entregue (Instant, não a soma).
    pub produto_ms: f64,
    pub ids_n: usize,
    pub ids_fnv: u64,
    pub token: usize,
    pub palavra: String,
    /// A palavra tem ao menos um caractere não branco (só descrição; ver o critério, seção 2).
    pub visivel: bool,
    /// Diferença entre os dois maiores logits (só descrição); NaN se não calculável.
    pub gap: f64,
    pub cont: Contadores,
    /// Logits não vazios e todos finitos.
    pub logits_ok: bool,
}

impl RepMedida {
    pub fn para_json(&self) -> Value {
        json!({
            "crono": self.crono.para_json(), "produto_ms": self.produto_ms, "ids_n": self.ids_n, "ids_fnv": format!("{:016x}", self.ids_fnv),
            "token": self.token, "palavra": self.palavra, "visivel": self.visivel, "gap": self.gap,
            "needed": self.cont.needed, "hits": self.cont.hits, "cpu": self.cont.cpu, "logits_ok": self.logits_ok,
        })
    }
    pub fn de_json(v: &Value) -> Result<Self, String> {
        Ok(RepMedida {
            crono: Cronometros::de_json(v.get("crono").ok_or("falta crono")?)?,
            produto_ms: num(v, "produto_ms")?,
            ids_n: inteiro(v, "ids_n")?,
            ids_fnv: u64::from_str_radix(&texto(v, "ids_fnv")?, 16).map_err(|_| "ids_fnv ilegível".to_string())?,
            token: inteiro(v, "token")?,
            palavra: texto(v, "palavra")?,
            visivel: booleano(v, "visivel")?,
            // NaN vira null no JSON: lê de volta como NaN.
            gap: v.get("gap").and_then(Value::as_f64).unwrap_or(f64::NAN),
            cont: Contadores { needed: inteiro(v, "needed")? as u64, hits: inteiro(v, "hits")? as u64, cpu: inteiro(v, "cpu")? as u64 },
            logits_ok: booleano(v, "logits_ok")?,
        })
    }
    /// Validade própria da repetição (nível 1): tempos, produto, logits e contadores.
    fn invalida(&self) -> Option<String> {
        if !self.crono.valido() {
            return Some("cronômetro não finito, negativo ou zerado".into());
        }
        if !(self.produto_ms.is_finite() && self.produto_ms > 0.0) {
            return Some("indicador de produto não finito, zero ou negativo".into());
        }
        if !self.logits_ok {
            return Some("logits vazios ou não finitos".into());
        }
        if self.cont.hits == 0 || self.cont.cpu == 0 {
            return Some(format!("contadores sem hits ou sem cpu ({:?})", self.cont));
        }
        None
    }
}

// ---- O fluxo: texto cru até a primeira palavra entregue ------------------------------------------------------------------------------

/// A sequência que o fluxo usa: motor, preparo e a leitura dos logits (o backend e a stream ficam dentro do fecho).
pub struct Sequencia<B: Backend, P: PreparaPasso> {
    pub eng: Engine<B>,
    pub prep: P,
    pub le_logits: Box<dyn FnMut() -> Result<Vec<f32>, String>>,
}

/// O resultado do fluxo: a medida, a data do `Instant` da entrega (para o indicador frio) e os logits e ids (para conferência).
pub struct Resultado {
    pub medida: RepMedida,
    pub ids: Vec<usize>,
    pub logits: Vec<f32>,
    pub t_fim: Instant,
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Texto -> template (um papel `user`, sistema padrão, `data`) -> ids -> `le_ids` (blocos do motor) -> `read_logits` -> argmax -> detokeniza
/// o token -> entrega a palavra (e uma quebra de linha) em `saida` com `flush`. `t_entrada` é o instante em que o texto cru ficou disponível e
/// `entrada_ms` o que custou obtê-lo. `cria` monta motor, backend e preparo NOVOS (conta como `T_sequencia`). As conferências de posição, a
/// leitura dos contadores e o encerramento do motor vêm DEPOIS da entrega (fora do indicador). Qualquer erro devolve a mensagem; nada repete.
pub fn primeira_palavra<B, P, C>(t_entrada: Instant, entrada_ms: f64, texto: &str, data: &str, tok: &Tokenizador, cria: C, saida: &mut dyn Write) -> Result<Resultado, String>
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    C: FnOnce() -> Result<Sequencia<B, P>, String>,
{
    let mut slot = None;
    let mut pendente = None;
    let c = corpo(t_entrada, entrada_ms, texto, data, tok, &mut slot, &mut pendente, |s, _| {
        *s = Some(cria()?);
        Ok(())
    }, saida)?;
    let mut seq = slot.ok_or("sem sequência")?;
    let s = seq.eng.next_published().map_err(|e| format!("{e:?}"))?.stats().clone();
    seq.eng.close().map_err(|e| format!("{e:?}"))?;
    Ok(c.fecha(Contadores { needed: s.needed, hits: s.hits, cpu: s.cpu }))
}

/// O que o fluxo produz até a entrega, ainda sem os contadores (a leitura deles e o destino do motor diferem entre o pedido avulso e
/// o motor reutilizado).
struct Corpo {
    crono: Cronometros,
    produto_ms: f64,
    ids: Vec<usize>,
    token: usize,
    palavra: String,
    gap: f64,
    logits: Vec<f32>,
    t_fim: Instant,
}

impl Corpo {
    fn fecha(self, cont: Contadores) -> Resultado {
        let medida = RepMedida {
            crono: self.crono,
            produto_ms: self.produto_ms,
            ids_n: self.ids.len(),
            ids_fnv: fnv_ids(&self.ids),
            token: self.token,
            visivel: self.palavra.chars().any(|c| !c.is_whitespace()),
            palavra: self.palavra,
            gap: self.gap,
            cont,
            logits_ok: !self.logits.is_empty() && self.logits.iter().all(|v| v.is_finite()),
        };
        Resultado { medida, ids: self.ids, logits: self.logits, t_fim: self.t_fim }
    }
}

/// O fluxo comum. `prepara` deixa `slot` com a sequência pronta (motor novo, ou o mesmo com `nova_sequencia`) e pode deixar um `pendente`;
/// o tempo dele é o `T_sequencia`. Termina com a posição conferida e a sequência em `slot`; contadores e destino do motor são de quem chama.
#[allow(clippy::too_many_arguments)]
fn corpo<B, P, F>(t_entrada: Instant, entrada_ms: f64, texto: &str, data: &str, tok: &Tokenizador, slot: &mut Option<Sequencia<B, P>>,
                  pendente: &mut Option<Published>, prepara: F, saida: &mut dyn Write) -> Result<Corpo, String>
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    F: FnOnce(&mut Option<Sequencia<B, P>>, &mut Option<Published>) -> Result<(), String>,
{
    if texto.trim().is_empty() {
        return Err("texto vazio".into());
    }
    let passo = |marca: &mut Instant| {
        let d = ms(marca.elapsed());
        *marca = Instant::now();
        d
    };
    let renderizado = renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: texto.to_string() }], true, data).map_err(|e| e.to_string())?;
    let ids: Vec<usize> = tok.codifica(&renderizado).into_iter().map(|x| x as usize).collect();
    let token_ms = ms(t_entrada.elapsed());
    let mut marca = Instant::now();
    prepara(slot, pendente)?;
    let sequencia_ms = passo(&mut marca);
    let seq = slot.as_mut().ok_or("sem sequência")?;
    le_ids_com_pendente(&mut seq.eng, &seq.prep, &ids, pendente).map_err(|e| format!("le_ids: {e:?}"))?;
    let leitura_ms = passo(&mut marca);
    let logits = (seq.le_logits)()?;
    let logits_ms = passo(&mut marca);
    let token = argmax(&logits);
    let escolha_ms = passo(&mut marca);
    let palavra = tok.decodifica(&[token as u32]).map_err(|e| e.to_string())?;
    let detok_ms = passo(&mut marca);
    saida.write_all(palavra.as_bytes()).and_then(|_| saida.write_all(b"\n")).and_then(|_| saida.flush()).map_err(|e| format!("entrega: {e}"))?;
    let t_fim = Instant::now();
    let entrega_ms = ms(t_fim.duration_since(marca));
    let produto_ms = ms(t_fim.duration_since(t_entrada));
    // Fora do indicador: posição.
    let n = ids.len() as u64;
    if seq.eng.position() != n {
        return Err(format!("posição do motor {} depois de ler {n} ids", seq.eng.position()));
    }
    if seq.prep.posicao() != n {
        return Err(format!("posição do backend {} depois de ler {n} ids", seq.prep.posicao()));
    }
    Ok(Corpo {
        crono: Cronometros { entrada_ms, token_ms, sequencia_ms, leitura_ms, logits_ms, escolha_ms, detok_ms, entrega_ms },
        produto_ms,
        gap: dois_maiores(&logits).map(|x| x.2).unwrap_or(f64::NAN),
        ids,
        token,
        palavra,
        logits,
        t_fim,
    })
}

/// Motor que atravessa vários pedidos: a sequência, o `Published` pendente do próximo token (o motor já publicou a tabela dele e a
/// linha de serviço espera por ele) e os contadores cumulativos da linha de serviço no fim do pedido anterior (por pedido = diferença).
pub struct SequenciaReutilizavel<B: Backend, P: PreparaPasso> {
    seq: Option<Sequencia<B, P>>,
    pendente: Option<Published>,
    base: Contadores,
    /// Quantas vezes o motor foi criado (1 + recriações depois de falha).
    pub criacoes: usize,
}

impl<B: Backend, P: PreparaPasso> Default for SequenciaReutilizavel<B, P> {
    fn default() -> Self {
        Self { seq: None, pendente: None, base: Contadores { needed: 0, hits: 0, cpu: 0 }, criacoes: 0 }
    }
}

impl<B: Backend, P: PreparaPasso> SequenciaReutilizavel<B, P> {
    pub fn nova() -> Self {
        Self::default()
    }
    /// Se há motor vivo guardado para o próximo pedido.
    pub fn viva(&self) -> bool {
        self.seq.as_ref().is_some_and(|s| !s.eng.is_closed())
    }
    /// Encerra o motor guardado (se houver), sem esperar nada: o `Published` pendente é solto antes do fechamento.
    pub fn encerra(&mut self) -> Result<(), String> {
        self.pendente = None;
        self.base = Contadores { needed: 0, hits: 0, cpu: 0 };
        match self.seq.take() {
            Some(s) => s.eng.close().map_err(|e| format!("{e:?}")),
            None => Ok(()),
        }
    }
}

/// Como [`primeira_palavra`], mas o motor sobrevive ao pedido: o primeiro pedido (ou o primeiro depois de uma falha) chama `cria`; os
/// seguintes chamam `Engine::nova_sequencia` no motor guardado. Contadores por pedido pela diferença dos cumulativos. Pedido que falha fecha o
/// motor (o próximo recria). `T_sequencia` do pedido reutilizado é o custo de `nova_sequencia`.
pub fn primeira_palavra_reutilizando<B, P, C>(t_entrada: Instant, entrada_ms: f64, texto: &str, data: &str, tok: &Tokenizador,
                                              r: &mut SequenciaReutilizavel<B, P>, cria: C, saida: &mut dyn Write) -> Result<Resultado, String>
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    C: FnOnce() -> Result<Sequencia<B, P>, String>,
{
    if texto.trim().is_empty() {
        // Recusa de entrada: o motor guardado não foi tocado e segue vivo.
        return Err("texto vazio".into());
    }
    let SequenciaReutilizavel { seq, pendente, base, criacoes } = r;
    let resultado = corpo(t_entrada, entrada_ms, texto, data, tok, seq, pendente, |s, pend| {
        if s.as_ref().is_some_and(|x| !x.eng.is_closed()) && pend.is_some() {
            s.as_mut().ok_or("sem sequência")?.eng.nova_sequencia().map_err(|e| format!("nova_sequencia: {e:?}"))?;
        } else {
            // Primeiro pedido, ou motor fechado/sem pendente: recria do zero.
            if let Some(velho) = s.take() {
                let _ = velho.eng.close();
            }
            *pend = None;
            *base = Contadores { needed: 0, hits: 0, cpu: 0 };
            *s = Some(cria()?);
            *criacoes += 1;
        }
        Ok(())
    }, saida);
    let c = match resultado {
        Ok(c) => c,
        Err(e) => {
            // Falha: fecha o motor e solta o pendente; o próximo pedido recria.
            r.encerra().ok();
            return Err(e);
        }
    };
    let fim = (|| -> Result<Contadores, String> {
        let seq = r.seq.as_mut().ok_or("sem sequência")?;
        let p = match r.pendente.take() {
            Some(p) => p,
            None => seq.eng.next_published().map_err(|e| format!("{e:?}"))?,
        };
        let s = p.stats().clone();
        r.pendente = Some(p);
        Ok(Contadores { needed: s.needed, hits: s.hits, cpu: s.cpu })
    })();
    match fim {
        Ok(cum) => {
            let cont = Contadores { needed: cum.needed - r.base.needed, hits: cum.hits - r.base.hits, cpu: cum.cpu - r.base.cpu };
            r.base = cum;
            Ok(c.fecha(cont))
        }
        Err(e) => {
            r.encerra().ok();
            Err(e)
        }
    }
}

// ---- Correção --------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotuloCorrecao {
    Passa,
    Reprova,
    SemProva,
}

impl RotuloCorrecao {
    pub fn nome(self) -> &'static str {
        match self {
            RotuloCorrecao::Passa => "PASSA",
            RotuloCorrecao::Reprova => "REPROVA",
            RotuloCorrecao::SemProva => "SEM PROVA",
        }
    }
}

/// A referência do HF para um pedido: ids do prompt, argmax, segundo colocado e a diferença de logit.
#[derive(Clone, Debug, PartialEq)]
pub struct RefCaso {
    pub prompt: String,
    pub ids: Vec<usize>,
    pub argmax: usize,
    pub top2: usize,
    pub gap: f64,
}

impl RefCaso {
    pub fn de_json(v: &Value) -> Result<Self, String> {
        Ok(RefCaso { prompt: texto(v, "prompt")?, ids: lista_ids(v, "ids")?, argmax: inteiro(v, "argmax")?, top2: inteiro(v, "top2")?, gap: num(v, "gap")? })
    }
    pub fn para_json(&self) -> Value {
        json!({"prompt": self.prompt, "ids": self.ids, "argmax": self.argmax, "top2": self.top2, "gap": self.gap})
    }
}

/// Lê o arquivo de referência (`ref-<modelo>.json`): a lista `casos`.
pub fn le_referencia(v: &Value) -> Result<Vec<RefCaso>, String> {
    v.get("casos").and_then(Value::as_array).ok_or("falta a lista casos")?.iter().map(RefCaso::de_json).collect()
}

/// Um pedido da correção como o programa o grava: o prompt, os ids do texto cru, a medida (ou a falha) e a referência pareada.
#[derive(Clone, Debug, PartialEq)]
pub struct CasoCorrecao {
    pub indice: usize,
    pub prompt: String,
    pub medida: Option<RepMedida>,
    pub ids: Vec<usize>,
    pub falha: Option<String>,
    pub referencia: Option<RefCaso>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegistroCorrecao {
    pub modelo: String,
    pub integro: bool,
    pub falha: Option<String>,
    pub casos: Vec<CasoCorrecao>,
    pub extras: Vec<(String, String)>,
}

impl RegistroCorrecao {
    pub fn para_json(&self) -> Value {
        json!({
            "tipo": "correcao", "modelo": self.modelo, "integro": self.integro, "falha": self.falha, "extras": pares_json(&self.extras),
            "casos": self.casos.iter().map(|c| json!({
                "indice": c.indice, "prompt": c.prompt, "ids": c.ids, "falha": c.falha,
                "medida": c.medida.as_ref().map(RepMedida::para_json), "referencia": c.referencia.as_ref().map(RefCaso::para_json),
            })).collect::<Vec<_>>(),
        })
    }
    pub fn de_json(v: &Value) -> Result<Self, String> {
        if texto(v, "tipo")? != "correcao" {
            return Err("tipo diferente de correcao".into());
        }
        let opt = |x: &Value| if x.is_null() { None } else { Some(x.clone()) };
        let casos = v.get("casos").and_then(Value::as_array).ok_or("falta casos")?;
        let casos = casos
            .iter()
            .map(|c| {
                Ok(CasoCorrecao {
                    indice: inteiro(c, "indice")?,
                    prompt: texto(c, "prompt")?,
                    ids: lista_ids(c, "ids")?,
                    falha: c.get("falha").and_then(Value::as_str).map(str::to_string),
                    medida: c.get("medida").and_then(opt).map(|m| RepMedida::de_json(&m)).transpose()?,
                    referencia: c.get("referencia").and_then(opt).map(|m| RefCaso::de_json(&m)).transpose()?,
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        Ok(RegistroCorrecao { modelo: texto(v, "modelo")?, integro: booleano(v, "integro")?, falha: v.get("falha").and_then(Value::as_str).map(str::to_string), casos, extras: pares(v, "extras")? })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AvalCorrecao {
    pub rotulo: RotuloCorrecao,
    pub motivos: Vec<String>,
    /// Uma linha descritiva por pedido (token, palavra, ids iguais, empate).
    pub linhas: Vec<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ResCaso {
    Passa,
    Empate,
    Reprova,
    SemProva,
}

fn avalia_caso(c: &CasoCorrecao) -> (ResCaso, String) {
    let sp = |m: String| (ResCaso::SemProva, format!("pedido {}: SEM PROVA: {m}", c.indice));
    if let Some(f) = &c.falha {
        return sp(f.clone());
    }
    let (Some(m), Some(r)) = (&c.medida, &c.referencia) else { return sp("sem medida ou sem referência".into()) };
    if r.prompt != c.prompt {
        return sp("a referência é de outro prompt".into());
    }
    if let Some(x) = m.invalida() {
        return sp(x);
    }
    let ids_iguais = c.ids == r.ids;
    if !ids_iguais {
        return (ResCaso::Reprova, format!("pedido {}: REPROVA: ids do texto cru diferentes da referência (Rust {}, HF {})", c.indice, c.ids.len(), r.ids.len()));
    }
    let desc = format!("token {} palavra {:?} (referência {} gap {:.4}, segundo {}), visivel {}", m.token, m.palavra, r.argmax, r.gap, r.top2, m.visivel);
    if m.token == r.argmax {
        (ResCaso::Passa, format!("pedido {}: PASSA: ids iguais, argmax igual; {desc}", c.indice))
    } else if r.gap < GAP_EMPATE && m.token == r.top2 {
        (ResCaso::Empate, format!("pedido {}: EMPATE (vale como passou): gap da referência {:.2e} < {GAP_EMPATE:.0e}; {desc}", c.indice, r.gap))
    } else {
        (ResCaso::Reprova, format!("pedido {}: REPROVA: argmax diferente da referência; {desc}", c.indice))
    }
}

/// Avalia a correção de um modelo (níveis: validade por pedido, igualdade de ids, argmax com empate). A integridade global entra em
/// [`rotulo_final_correcao`].
pub fn avalia_correcao(r: &RegistroCorrecao) -> AvalCorrecao {
    let mut a = AvalCorrecao { rotulo: RotuloCorrecao::SemProva, motivos: vec![], linhas: vec![] };
    // Precedência do critério (seção 4): REPROVA válida de um pedido concluído vence SEM PROVA de outro (erro, ausência); só a integridade
    // global quebrada (tratada em rotulo_final_correcao) derruba tudo. Por isso cada pedido presente é avaliado mesmo com falha posterior.
    if r.casos.len() > PEDIDOS_CORRECAO {
        a.motivos.push(format!("{} pedidos em vez de {PEDIDOS_CORRECAO}", r.casos.len()));
        return a;
    }
    if r.casos.iter().enumerate().any(|(i, c)| c.indice != i) {
        a.motivos.push("índices dos pedidos fora de ordem".into());
        return a;
    }
    let mut res: Vec<(ResCaso, String)> = r.casos.iter().map(avalia_caso).collect();
    for i in r.casos.len()..PEDIDOS_CORRECAO {
        res.push((ResCaso::SemProva, format!("pedido {i}: SEM PROVA: pedido ausente (não executado)")));
    }
    a.linhas = res.iter().map(|x| x.1.clone()).collect();
    a.rotulo = if res.iter().any(|x| x.0 == ResCaso::Reprova) {
        RotuloCorrecao::Reprova
    } else if res.iter().any(|x| x.0 == ResCaso::SemProva) || r.falha.is_some() {
        RotuloCorrecao::SemProva
    } else {
        RotuloCorrecao::Passa
    };
    if let Some(f) = &r.falha {
        a.motivos.push(f.clone());
    }
    if a.rotulo != RotuloCorrecao::Passa {
        a.motivos.extend(res.iter().filter(|x| matches!(x.0, ResCaso::Reprova | ResCaso::SemProva)).map(|x| x.1.clone()));
    }
    a
}

pub fn rotulo_final_correcao(r: RotuloCorrecao, integro: bool) -> RotuloCorrecao {
    if integro {
        r
    } else {
        RotuloCorrecao::SemProva
    }
}

/// Conjunto da correção: REPROVA vence SEM PROVA, SEM PROVA vence PASSA (e a falta de modelo); integridade quebrada deixa tudo SEM PROVA.
pub fn rotulo_conjunto_correcao(v: &[RotuloCorrecao], integro: bool) -> RotuloCorrecao {
    if !integro {
        RotuloCorrecao::SemProva
    } else if v.contains(&RotuloCorrecao::Reprova) {
        RotuloCorrecao::Reprova
    } else if v.is_empty() || v.contains(&RotuloCorrecao::SemProva) {
        RotuloCorrecao::SemProva
    } else {
        RotuloCorrecao::Passa
    }
}

// ---- Velocidade quente ----------------------------------------------------------------------------------------------------------------

/// O registro bruto de uma célula quente (um modelo e um cenário): aquecimento descartado e as repetições, como gravado.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistroQuente {
    pub modelo: String,
    pub cen: Cen,
    pub integro: bool,
    pub falha: Option<String>,
    /// Ids contados pelo tokenizador Rust no texto do cenário (conferidos contra a faixa).
    pub ids_n: usize,
    pub aquecimento: Vec<String>,
    pub reps: Vec<RepMedida>,
    pub extras: Vec<(String, String)>,
}

impl RegistroQuente {
    pub fn para_json(&self) -> Value {
        json!({
            "tipo": "quente", "modelo": self.modelo, "cenario": self.cen.nome(), "integro": self.integro, "falha": self.falha, "ids_n": self.ids_n,
            "aquecimento": self.aquecimento, "extras": pares_json(&self.extras), "reps": self.reps.iter().map(RepMedida::para_json).collect::<Vec<_>>(),
        })
    }
    pub fn de_json(v: &Value) -> Result<Self, String> {
        if texto(v, "tipo")? != "quente" {
            return Err("tipo diferente de quente".into());
        }
        Ok(RegistroQuente {
            modelo: texto(v, "modelo")?,
            cen: Cen::de_nome(&texto(v, "cenario")?).ok_or("cenário ilegível")?,
            integro: booleano(v, "integro")?,
            falha: v.get("falha").and_then(Value::as_str).map(str::to_string),
            ids_n: inteiro(v, "ids_n")?,
            aquecimento: v.get("aquecimento").and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_string)).collect()).unwrap_or_default(),
            reps: v.get("reps").and_then(Value::as_array).ok_or("faltam reps")?.iter().map(RepMedida::de_json).collect::<Result<Vec<_>, _>>()?,
            extras: pares(v, "extras")?,
        })
    }
}

/// O resultado da avaliação de uma célula quente. Os descritivos saem sempre que os dados permitem, mas só valem como prova com ATINGE ou NÃO ATINGE.
#[derive(Clone, Debug, PartialEq)]
pub struct AvalQuente {
    pub rotulo: Rotulo,
    pub motivos: Vec<String>,
    pub meta_ms: Option<f64>,
    /// Mediana do indicador de produto (com T_sequencia): o número que decide contra a meta.
    pub med_ms: Option<f64>,
    /// Só descrição: mediana do produto sem T_sequencia (o que o quente seria com reset no Engine).
    pub med_sem_seq_ms: Option<f64>,
    pub pior_ms: Option<f64>,
    pub desvio_ms: Option<f64>,
    pub med_sequencia_ms: Option<f64>,
}

impl AvalQuente {
    fn sem_prova(mut self, m: impl Into<String>) -> Self {
        self.rotulo = Rotulo::SemProva;
        self.motivos.push(m.into());
        self
    }
}

/// Avalia uma célula quente lida de arquivo: a identidade gravada tem de ser a da célula pedida.
pub fn avalia_registro_quente(modelo: &str, cen: Cen, r: &RegistroQuente) -> AvalQuente {
    let mut a = avalia_quente_dados(modelo, cen, r);
    if r.modelo != modelo || r.cen != cen {
        a = AvalQuente { rotulo: Rotulo::SemProva, motivos: vec![format!("identidade do registro ({} {}) diferente da célula pedida ({modelo} {}): não vale como prova", r.modelo, r.cen.nome(), cen.nome())], meta_ms: a.meta_ms, med_ms: None, med_sem_seq_ms: None, pior_ms: None, desvio_ms: None, med_sequencia_ms: None };
    }
    a
}

fn avalia_quente_dados(modelo: &str, cen: Cen, r: &RegistroQuente) -> AvalQuente {
    let mut a = AvalQuente { rotulo: Rotulo::SemProva, motivos: vec![], meta_ms: vr::meta_ms(Estado::Quente, modelo, cen), med_ms: None, med_sem_seq_ms: None, pior_ms: None, desvio_ms: None, med_sequencia_ms: None };
    let legivel = !r.reps.is_empty() && r.reps.iter().all(|x| x.produto_ms.is_finite() && x.produto_ms > 0.0 && x.crono.sequencia_ms.is_finite());
    if legivel {
        let p: Vec<f64> = r.reps.iter().map(|x| x.produto_ms).collect();
        a.med_ms = Some(mediana(&p));
        a.med_sem_seq_ms = Some(mediana(&r.reps.iter().map(|x| x.produto_ms - x.crono.sequencia_ms).collect::<Vec<_>>()));
        a.pior_ms = Some(p.iter().cloned().fold(f64::MIN, f64::max));
        a.desvio_ms = Some(vr::desvio(&p));
        a.med_sequencia_ms = Some(mediana(&r.reps.iter().map(|x| x.crono.sequencia_ms).collect::<Vec<_>>()));
    }
    let Some(meta) = a.meta_ms else { return a.sem_prova(format!("sem meta para o modelo {modelo}")) };
    if let Some(f) = &r.falha {
        return a.sem_prova(f.clone());
    }
    if !ids_na_faixa(r.ids_n, cen) {
        return a.sem_prova(format!("{} ids fora da faixa de 2% em torno de {} (cenário {})", r.ids_n, cen.n(), cen.nome()));
    }
    if r.reps.len() < REPS_QUENTE {
        return a.sem_prova(format!("menos de {REPS_QUENTE} repetições válidas ({})", r.reps.len()));
    }
    if r.reps.len() > REPS_QUENTE {
        return a.sem_prova(format!("mais de {REPS_QUENTE} repetições ({}): fora do desenho", r.reps.len()));
    }
    for (i, x) in r.reps.iter().enumerate() {
        if let Some(m) = x.invalida() {
            return a.sem_prova(format!("repetição {}: {m}", i + 1));
        }
        if x.ids_n != r.ids_n {
            return a.sem_prova(format!("repetição {}: {} ids, esperados {}", i + 1, x.ids_n, r.ids_n));
        }
    }
    let p0 = &r.reps[0];
    if let Some((i, _)) = r.reps.iter().enumerate().find(|(_, x)| x.token != p0.token || x.ids_fnv != p0.ids_fnv || !contadores_validos(x.cont, p0.cont)) {
        return a.sem_prova(format!("ALERTA DE CORREÇÃO: a repetição {} difere da 1 em token, ids ou contadores", i + 1));
    }
    if !legivel {
        return a.sem_prova("tempo não finito, zero ou negativo");
    }
    if !(a.med_ms.unwrap_or(f64::NAN) <= meta) {
        a.motivos.push("meta".into());
    }
    a.rotulo = if a.motivos.is_empty() { Rotulo::Atinge } else { Rotulo::NaoAtinge };
    a
}

// ---- Velocidade fria --------------------------------------------------------------------------------------------------------------------

/// O registro de um processo frio: o indicador frio (do início do `main` até a palavra entregue), a medida (se chegou ao fim) e as partes.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistroFrioPP {
    pub modelo: String,
    pub cen: Cen,
    pub rep: usize,
    pub total_ms: f64,
    pub ok: bool,
    pub falha: Option<String>,
    pub integro: bool,
    pub ids_n: usize,
    pub partes: Vec<(String, f64)>,
    pub medida: Option<RepMedida>,
    pub extras: Vec<(String, String)>,
}

impl RegistroFrioPP {
    pub fn para_json(&self) -> Value {
        json!({
            "tipo": "frio", "modelo": self.modelo, "cenario": self.cen.nome(), "rep": self.rep, "total_ms": self.total_ms, "ok": self.ok, "falha": self.falha,
            "integro": self.integro, "ids_n": self.ids_n, "partes": Value::Object(self.partes.iter().map(|(k, v)| (k.clone(), Value::from(*v))).collect()),
            "medida": self.medida.as_ref().map(RepMedida::para_json), "extras": pares_json(&self.extras),
        })
    }
    pub fn de_json(v: &Value) -> Result<Self, String> {
        if texto(v, "tipo")? != "frio" {
            return Err("tipo diferente de frio".into());
        }
        let partes = match v.get("partes") {
            Some(Value::Object(m)) => m.iter().map(|(k, x)| x.as_f64().map(|n| (k.clone(), n)).ok_or_else(|| format!("parte {k} ilegível"))).collect::<Result<Vec<_>, _>>()?,
            _ => return Err("faltam partes".into()),
        };
        Ok(RegistroFrioPP {
            modelo: texto(v, "modelo")?,
            cen: Cen::de_nome(&texto(v, "cenario")?).ok_or("cenário ilegível")?,
            rep: inteiro(v, "rep")?,
            total_ms: num(v, "total_ms")?,
            ok: booleano(v, "ok")?,
            falha: v.get("falha").and_then(Value::as_str).map(str::to_string),
            integro: booleano(v, "integro")?,
            ids_n: inteiro(v, "ids_n")?,
            partes,
            medida: match v.get("medida") {
                Some(Value::Null) | None => None,
                Some(m) => Some(RepMedida::de_json(m)?),
            },
            extras: pares(v, "extras")?,
        })
    }
    /// A validade que o critério pede de um processo frio, além do que `avalia_frio` já confere: faixa de ids, logits finitos, hits e cpu positivos.
    fn motivo_invalido(&self, cen: Cen) -> Option<String> {
        if !ids_na_faixa(self.ids_n, cen) {
            return Some(format!("{} ids fora da faixa de 2% em torno de {}", self.ids_n, cen.n()));
        }
        match &self.medida {
            None => Some("sem medida (o processo não chegou à palavra)".into()),
            Some(m) => m.invalida(),
        }
    }
    /// O registro no formato de `velocidade_regra` (o `ok` já inclui a validade própria do 5b).
    pub fn para_vr(&self) -> RegistroFrio {
        let motivo = self.motivo_invalido(self.cen);
        RegistroFrio {
            modelo: self.modelo.clone(),
            cen: self.cen,
            rep: self.rep,
            total_ms: self.total_ms,
            ok: self.ok && motivo.is_none(),
            falha: self.falha.clone().or(if self.ok { motivo } else { None }),
            integro: self.integro,
            partes: self.partes.clone(),
            extras: self.extras.clone(),
        }
    }
}

/// Avalia uma célula fria (3 repetições, mediana contra a meta fria), pelas regras de `velocidade_regra::avalia_frio`.
pub fn avalia_frio_pp(modelo: &str, cen: Cen, regs: &[RegistroFrioPP]) -> vr::Avaliacao {
    let v: Vec<RegistroFrio> = regs.iter().map(RegistroFrioPP::para_vr).collect();
    vr::avalia_frio(modelo, cen, &v)
}

// ---- Testes -----------------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg};
    use crate::motor::{AdmitMode, Config};
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    const TOKENIZER_1B: &str = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct/tokenizer.json";
    const VOCAB: usize = 49155;

    fn tok() -> Tokenizador {
        Tokenizador::carrega(Path::new(TOKENIZER_1B)).expect("tokenizer.json")
    }

    // ---- regras puras ----

    #[test]
    fn data_civil_e_a_fixa() {
        assert_eq!(civil_de_dias(0), (1970, 1, 1));
        assert_eq!(civil_de_dias(20764), (2026, 11, 7));
        assert_eq!(civil_de_dias(19782), (2024, 2, 29));
        assert_eq!(civil_de_dias(-1), (1969, 12, 31));
        assert_eq!(data_fixa_texto(), "November 07, 2026");
        let (a, m, d) = data_de_hoje_utc();
        assert!(a >= 2026 && formata_data(a, m, d).is_ok());
    }

    #[test]
    fn faixa_de_ids_dois_por_cento() {
        assert!(ids_na_faixa(500, Cen::C) && ids_na_faixa(490, Cen::C) && ids_na_faixa(510, Cen::C));
        assert!(!ids_na_faixa(489, Cen::C) && !ids_na_faixa(511, Cen::C));
        assert!(ids_na_faixa(2003, Cen::A) && ids_na_faixa(1963, Cen::A) && ids_na_faixa(2043, Cen::A));
        assert!(!ids_na_faixa(1962, Cen::A) && !ids_na_faixa(2044, Cen::A));
        assert!(!ids_na_faixa(0, Cen::C));
    }

    #[test]
    fn dois_maiores_e_empate() {
        assert_eq!(dois_maiores(&[0.0, 3.0, 2.5, 1.0]), Some((1, 2, 0.5)));
        assert_eq!(dois_maiores(&[1.0]), None);
        assert!(dois_maiores(&[f32::NAN, 1.0, 0.0]).is_none() || dois_maiores(&[f32::NAN, 1.0, 0.0]).is_some_and(|x| x.2.is_finite()));
        assert_eq!(dois_maiores(&[1.0, f32::INFINITY]), None);
    }

    fn crono(seq: f64) -> Cronometros {
        Cronometros { entrada_ms: 0.1, token_ms: 2.0, sequencia_ms: seq, leitura_ms: 1000.0, logits_ms: 5.0, escolha_ms: 0.3, detok_ms: 0.01, entrega_ms: 0.05 }
    }

    fn medida(token: usize, produto: f64, seq: f64) -> RepMedida {
        RepMedida { crono: crono(seq), produto_ms: produto, ids_n: 499, ids_fnv: 7, token, palavra: "To".into(), visivel: true, gap: 1.4, cont: Contadores { needed: 100, hits: 10, cpu: 90 }, logits_ok: true }
    }

    fn quente(reps: Vec<RepMedida>) -> RegistroQuente {
        RegistroQuente { modelo: "1b".into(), cen: Cen::C, integro: true, falha: None, ids_n: 499, aquecimento: vec!["aq".into()], reps, extras: vec![("hash_bin".into(), "abc".into())] }
    }

    fn cinco(produto: f64) -> Vec<RepMedida> {
        (0..5).map(|i| medida(705, produto + i as f64, 1500.0)).collect()
    }

    #[test]
    fn cronometros_json_e_validade() {
        let c = crono(1500.0);
        assert_eq!(Cronometros::de_json(&c.para_json()).unwrap(), c);
        for k in ["T_entrada", "T_token", "T_sequencia", "T_leitura", "T_logits", "T_escolha", "T_detok", "T_entrega"] {
            assert!(c.para_json().get(k).is_some(), "{k}");
        }
        assert!(c.valido());
        assert!((c.soma_etapas_ms() - (2.0 + 1500.0 + 1000.0 + 5.0 + 0.3 + 0.01 + 0.05)).abs() < 1e-9);
        assert!(!Cronometros { leitura_ms: 0.0, ..c }.valido());
        assert!(!Cronometros { detok_ms: f64::NAN, ..c }.valido());
        assert!(!Cronometros { entrega_ms: -1.0, ..c }.valido());
    }

    #[test]
    fn quente_atinge_e_nao_atinge_pela_mediana_do_produto() {
        // Mediana 4.002 ms <= 5 s: ATINGE; o indicador com T_sequencia é o que decide.
        let a = avalia_registro_quente("1b", Cen::C, &quente(cinco(4000.0)));
        assert_eq!((a.rotulo, a.meta_ms, a.med_ms), (Rotulo::Atinge, Some(5000.0), Some(4002.0)), "{:?}", a.motivos);
        assert_eq!(a.med_sem_seq_ms, Some(4002.0 - 1500.0));
        assert_eq!(a.med_sequencia_ms, Some(1500.0));
        // Meta exata ainda atinge (as 5 iguais a 5.000).
        let exato: Vec<RepMedida> = (0..5).map(|_| medida(705, 5000.0, 1500.0)).collect();
        assert_eq!(avalia_registro_quente("1b", Cen::C, &quente(exato)).rotulo, Rotulo::Atinge);
        // Acima da meta: NÃO ATINGE com o motivo "meta", e os descritivos continuam.
        let n = avalia_registro_quente("1b", Cen::C, &quente(cinco(5100.0)));
        assert_eq!((n.rotulo, n.motivos.clone()), (Rotulo::NaoAtinge, vec!["meta".to_string()]));
        assert!(n.med_sem_seq_ms.is_some() && n.pior_ms == Some(5104.0) && n.desvio_ms.is_some());
        // Meta do 3B e do cenário A.
        let mut r = quente(cinco(9000.0));
        r.modelo = "3b".into();
        assert_eq!(avalia_registro_quente("3b", Cen::C, &r).rotulo, Rotulo::Atinge);
    }

    #[test]
    fn quente_sem_prova_nos_casos_do_nivel_um() {
        let sp = |r: RegistroQuente| {
            let a = avalia_registro_quente("1b", Cen::C, &r);
            assert_eq!(a.rotulo, Rotulo::SemProva, "{:?}", a.motivos);
            a
        };
        let mut r = quente(cinco(4000.0));
        r.falha = Some("erro do motor".into());
        sp(r);
        sp(quente(cinco(4000.0)[..4].to_vec()));
        sp(quente([cinco(4000.0), cinco(1.0)].concat()[..6].to_vec()));
        let mut r = quente(cinco(4000.0));
        r.ids_n = 489;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[2].token = 9;
        assert!(sp(r).motivos[0].contains("ALERTA DE CORREÇÃO"));
        let mut r = quente(cinco(4000.0));
        r.reps[1].ids_fnv = 8;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[4].cont.hits = 0;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[0].cont.cpu = 0;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[3].logits_ok = false;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[3].produto_ms = f64::NAN;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[3].produto_ms = 0.0;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[0].crono.leitura_ms = f64::INFINITY;
        sp(r);
        let mut r = quente(cinco(4000.0));
        r.reps[0].ids_n = 500;
        sp(r);
        // Identidade do registro diferente da célula pedida: não vale.
        let a = avalia_registro_quente("1b", Cen::A, &quente(cinco(4000.0)));
        assert_eq!(a.rotulo, Rotulo::SemProva);
        assert!(a.med_ms.is_none());
        // Modelo sem meta.
        assert_eq!(avalia_registro_quente("7b", Cen::C, &quente(cinco(4000.0))).rotulo, Rotulo::SemProva);
    }

    #[test]
    fn registro_quente_ida_e_volta() {
        let r = quente(cinco(4000.0));
        let j = serde_json::to_string_pretty(&r.para_json()).unwrap();
        assert_eq!(RegistroQuente::de_json(&serde_json::from_str(&j).unwrap()).unwrap(), r);
        let mut ruim = r.para_json();
        ruim["tipo"] = json!("frio");
        assert!(RegistroQuente::de_json(&ruim).is_err());
        // NaN no gap vira null e volta como NaN sem quebrar a leitura.
        let mut m = medida(1, 10.0, 1.0);
        m.gap = f64::NAN;
        assert!(RepMedida::de_json(&m.para_json()).unwrap().gap.is_nan());
    }

    fn frio(rep: usize, total: f64) -> RegistroFrioPP {
        RegistroFrioPP { modelo: "1b".into(), cen: Cen::C, rep, total_ms: total, ok: true, falha: None, integro: true, ids_n: 499, partes: vec![("pesos_na_ram".into(), 900.0)], medida: Some(medida(705, 3000.0, 1500.0)), extras: vec![("k".into(), "v".into())] }
    }

    #[test]
    fn frio_pela_mediana_e_validade_propria() {
        let tres = vec![frio(1, 19000.0), frio(2, 21000.0), frio(3, 20000.0)];
        assert_eq!(avalia_frio_pp("1b", Cen::C, &tres).rotulo, Rotulo::Atinge, "meta exata ainda atinge");
        let tres = vec![frio(1, 19000.0), frio(2, 21000.0), frio(3, 20001.0)];
        assert_eq!(avalia_frio_pp("1b", Cen::C, &tres).rotulo, Rotulo::NaoAtinge);
        // Frio com meta do 1B no cenário A (30 s) e do 3B (45 s).
        let a = avalia_frio_pp("1b", Cen::C, &[frio(1, 1.0), frio(2, 1.0)]);
        assert_eq!(a.rotulo, Rotulo::SemProva);
        let mut r = frio(3, 1000.0);
        r.medida.as_mut().unwrap().cont.hits = 0;
        assert_eq!(avalia_frio_pp("1b", Cen::C, &[frio(1, 1000.0), frio(2, 1000.0), r]).rotulo, Rotulo::SemProva, "hits zerados");
        let mut r = frio(3, 1000.0);
        r.ids_n = 600;
        assert_eq!(avalia_frio_pp("1b", Cen::C, &[frio(1, 1000.0), frio(2, 1000.0), r]).rotulo, Rotulo::SemProva, "ids fora da faixa");
        let mut r = frio(3, 1000.0);
        r.medida = None;
        assert_eq!(avalia_frio_pp("1b", Cen::C, &[frio(1, 1000.0), frio(2, 1000.0), r]).rotulo, Rotulo::SemProva, "sem medida");
        let mut r = frio(3, 1000.0);
        r.integro = false;
        assert_eq!(avalia_frio_pp("1b", Cen::C, &[frio(1, 1000.0), frio(2, 1000.0), r]).rotulo, Rotulo::SemProva);
        let mut r = frio(3, 1000.0);
        r.falha = Some("erro".into());
        r.ok = false;
        assert_eq!(avalia_frio_pp("1b", Cen::C, &[frio(1, 1000.0), frio(2, 1000.0), r]).rotulo, Rotulo::SemProva);
        let r = frio(2, 7.0);
        assert_eq!(RegistroFrioPP::de_json(&serde_json::from_str(&serde_json::to_string(&r.para_json()).unwrap()).unwrap()).unwrap(), r);
    }

    fn ref_caso(p: &str, argmax: usize, top2: usize, gap: f64) -> RefCaso {
        RefCaso { prompt: p.into(), ids: vec![1, 2, 3], argmax, top2, gap }
    }

    fn caso(i: usize, token: usize, r: RefCaso) -> CasoCorrecao {
        let mut m = medida(token, 3000.0, 1500.0);
        m.ids_n = 3;
        CasoCorrecao { indice: i, prompt: r.prompt.clone(), medida: Some(m), ids: r.ids.clone(), falha: None, referencia: Some(r) }
    }

    fn correcao(casos: Vec<CasoCorrecao>) -> RegistroCorrecao {
        RegistroCorrecao { modelo: "1b".into(), integro: true, falha: None, casos, extras: vec![] }
    }

    fn tres_ok() -> Vec<CasoCorrecao> {
        (0..3).map(|i| caso(i, 705, ref_caso(&format!("p{i}"), 705, 99, 1.4))).collect()
    }

    #[test]
    fn correcao_passa_reprova_empate_e_sem_prova() {
        let a = avalia_correcao(&correcao(tres_ok()));
        assert_eq!(a.rotulo, RotuloCorrecao::Passa, "{:?}", a.motivos);
        // Argmax diferente com gap grande: REPROVA.
        let mut c = tres_ok();
        c[1] = caso(1, 5, ref_caso("p1", 705, 5, 1.4));
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::Reprova);
        // Empate: gap < 1e-3 e o token é o segundo colocado da referência: vale como passou.
        let mut c = tres_ok();
        c[2] = caso(2, 5, ref_caso("p2", 705, 5, 5e-4));
        let a = avalia_correcao(&correcao(c));
        assert_eq!(a.rotulo, RotuloCorrecao::Passa);
        assert!(a.linhas[2].contains("EMPATE"));
        // Gap exatamente 1e-3 NÃO é empate (estritamente menor).
        let mut c = tres_ok();
        c[2] = caso(2, 5, ref_caso("p2", 705, 5, 1e-3));
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::Reprova);
        // Gap pequeno mas o token não é o segundo colocado: REPROVA.
        let mut c = tres_ok();
        c[2] = caso(2, 6, ref_caso("p2", 705, 5, 5e-4));
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::Reprova);
        // ids do texto cru diferentes da referência: REPROVA mesmo com argmax igual.
        let mut c = tres_ok();
        c[0].ids = vec![1, 2, 4];
        let a = avalia_correcao(&correcao(c));
        assert_eq!(a.rotulo, RotuloCorrecao::Reprova);
        assert!(a.motivos[0].contains("ids"));
        // Falha do pedido, referência ausente, prompt trocado, hits zerados, número de pedidos: SEM PROVA.
        let mut c = tres_ok();
        c[1].falha = Some("erro".into());
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::SemProva);
        let mut c = tres_ok();
        c[1].referencia = None;
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::SemProva);
        let mut c = tres_ok();
        c[1].prompt = "outro".into();
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::SemProva);
        let mut c = tres_ok();
        c[1].medida.as_mut().unwrap().cont.hits = 0;
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::SemProva);
        let mut c = tres_ok();
        c[0].medida.as_mut().unwrap().logits_ok = false;
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::SemProva);
        assert_eq!(avalia_correcao(&correcao(tres_ok()[..2].to_vec())).rotulo, RotuloCorrecao::SemProva);
        let mut r = correcao(tres_ok());
        r.falha = Some("carga do modelo".into());
        assert_eq!(avalia_correcao(&r).rotulo, RotuloCorrecao::SemProva);
        // Precedência: pedido 0 REPROVA, pedido 1 com erro, pedido 2 ausente => REPROVA; sem reprovação válida => SEM PROVA.
        let mut c = tres_ok()[..2].to_vec();
        c[0] = caso(0, 5, ref_caso("p0", 705, 9, 1.4));
        c[1].falha = Some("erro de GPU".into());
        let mut r = correcao(c);
        r.falha = Some("pedido 1: erro de GPU".into());
        let a = avalia_correcao(&r);
        assert_eq!(a.rotulo, RotuloCorrecao::Reprova, "{:?}", a.motivos);
        assert_eq!(a.linhas.len(), 3);
        assert_eq!(rotulo_final_correcao(a.rotulo, true), RotuloCorrecao::Reprova);
        assert_eq!(rotulo_final_correcao(a.rotulo, false), RotuloCorrecao::SemProva, "integridade quebrada: tudo SEM PROVA");
        let mut c = tres_ok()[..2].to_vec();
        c[1].falha = Some("erro de GPU".into());
        let mut r = correcao(c);
        r.falha = Some("pedido 1: erro de GPU".into());
        assert_eq!(avalia_correcao(&r).rotulo, RotuloCorrecao::SemProva);
        // REPROVA vence SEM PROVA entre pedidos do mesmo modelo.
        let mut c = tres_ok();
        c[0].falha = Some("erro".into());
        c[1] = caso(1, 5, ref_caso("p1", 705, 5, 1.4));
        assert_eq!(avalia_correcao(&correcao(c)).rotulo, RotuloCorrecao::Reprova);
    }

    #[test]
    fn rotulos_do_conjunto_da_correcao() {
        use RotuloCorrecao::*;
        assert_eq!(rotulo_conjunto_correcao(&[Passa, Passa], true), Passa);
        assert_eq!(rotulo_conjunto_correcao(&[Passa, SemProva], true), SemProva);
        assert_eq!(rotulo_conjunto_correcao(&[SemProva, Reprova], true), Reprova);
        assert_eq!(rotulo_conjunto_correcao(&[], true), SemProva);
        assert_eq!(rotulo_conjunto_correcao(&[Passa], false), SemProva);
        assert_eq!(rotulo_final_correcao(Reprova, false), SemProva);
        assert_eq!(rotulo_final_correcao(Passa, true), Passa);
        assert_eq!(["PASSA", "REPROVA", "SEM PROVA"], [Passa.nome(), Reprova.nome(), SemProva.nome()]);
    }

    #[test]
    fn registro_de_correcao_ida_e_volta() {
        let mut r = correcao(tres_ok());
        r.extras.push(("hash_bin".into(), "x".into()));
        r.casos[1].falha = Some("erro".into());
        r.casos[1].medida = None;
        let j = serde_json::to_string(&r.para_json()).unwrap();
        assert_eq!(RegistroCorrecao::de_json(&serde_json::from_str(&j).unwrap()).unwrap(), r);
        let ref_json = json!({"casos": [{"prompt": "a", "ids": [1, 2], "argmax": 3, "top2": 4, "gap": 0.5, "extra": 1}]});
        assert_eq!(le_referencia(&ref_json).unwrap(), vec![RefCaso { prompt: "a".into(), ids: vec![1, 2], argmax: 3, top2: 4, gap: 0.5 }]);
        assert!(le_referencia(&json!({"casos": [{"prompt": "a"}]})).is_err());
        assert!(le_referencia(&json!({})).is_err());
    }

    // ---- o fluxo com o motor falso ----

    struct Prep {
        max_len: usize,
        pos: Mutex<u64>,
        blocos: Mutex<Vec<Vec<usize>>>,
    }

    impl PreparaPasso for Prep {
        type Error = &'static str;
        fn vocab(&self) -> usize {
            VOCAB
        }
        fn max_positions(&self) -> usize {
            8
        }
        fn max_len(&self) -> usize {
            self.max_len
        }
        fn posicao(&self) -> u64 {
            *self.pos.lock().unwrap()
        }
        fn prepara(&self, ids: &[usize]) -> Result<(), &'static str> {
            self.blocos.lock().unwrap().push(ids.to_vec());
            *self.pos.lock().unwrap() += ids.len() as u64;
            Ok(())
        }
    }

    /// O preparo compartilhado: o teste guarda o `Arc` e lê os blocos depois do fluxo.
    struct PrepCompartilhado(Arc<Prep>);
    impl PreparaPasso for PrepCompartilhado {
        type Error = &'static str;
        fn vocab(&self) -> usize {
            self.0.vocab()
        }
        fn max_positions(&self) -> usize {
            8
        }
        fn max_len(&self) -> usize {
            self.0.max_len
        }
        fn posicao(&self) -> u64 {
            self.0.posicao()
        }
        fn prepara(&self, ids: &[usize]) -> Result<(), &'static str> {
            self.0.prepara(ids)
        }
    }

    fn rota(p: usize, l: usize) -> Vec<u32> {
        vec![((p + l) % 4) as u32, ((p + l + 1 + (p / 4) % 3) % 4) as u32]
    }

    fn motor_falso(total: usize) -> (Arc<FakeGpu>, Engine<FakeGpu>) {
        let routing: Vec<Vec<Vec<u32>>> = (0..total.max(1)).map(|p| (0..2).map(|l| rota(p, l)).collect()).collect();
        let gpu = FakeGpu::new_sched(
            FakeCfg { layers: 2, experts: 4, topk: 2, width: 3, routing, fail_main_at: None, copy_immediate: false, max_positions: 8 },
            SchedCfg { seed: 0xA0A, weights: [1, 1, 1, 1], budget: 300_000 },
        );
        let cfg = Config { layers: 2, experts: 4, topk: 2, width: 3, slots: 0, backlog: 0, admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions: 8, mutant: None };
        let parts = gpu.parts(&cfg);
        let eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu { weighted: true, ..FakeCpu::default() }).expect("motor");
        (gpu, eng)
    }

    fn logits_com_pico(id: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; VOCAB];
        v[id] = 5.0;
        v[id + 1] = 2.0;
        v
    }

    fn roda(texto: &str, logits: Result<Vec<f32>, String>, prep: &Arc<Prep>) -> (Result<Resultado, String>, Vec<u8>) {
        let t = tok();
        let (_gpu, eng) = motor_falso(2100);
        let p = PrepCompartilhado(prep.clone());
        let mut saida: Vec<u8> = Vec::new();
        let r = primeira_palavra(Instant::now(), 0.25, texto, &data_fixa_texto(), &t, move || Ok(Sequencia { eng, prep: p, le_logits: Box::new(move || logits.clone()) }), &mut saida);
        (r, saida)
    }

    fn prep(max_len: usize) -> Arc<Prep> {
        Arc::new(Prep { max_len, pos: Mutex::new(0), blocos: Mutex::new(vec![]) })
    }

    #[test]
    fn fluxo_texto_ids_token_palavra_com_motor_falso() {
        let t = tok();
        let aqui = t.codifica("Here")[0] as usize;
        assert_eq!(aqui, 10921, "o id de 'Here' é o da referência");
        let texto = "Explain how TCP congestion control works.";
        let p = prep(2048);
        let (r, saida) = roda(texto, Ok(logits_com_pico(aqui)), &p);
        let r = r.expect("fluxo");
        // Texto -> template -> ids: os mesmos que o 5a prova contra o HF.
        let esperado: Vec<usize> = t.codifica(&renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: texto.into() }], true, &data_fixa_texto()).unwrap()).into_iter().map(|x| x as usize).collect();
        assert_eq!(r.ids, esperado);
        assert!(esperado.len() > 40 && esperado[0] == 49152, "começa em <|start_of_role|>");
        // O preparo recebeu os ids inteiros, em blocos de 8, em ordem.
        let blocos = p.blocos.lock().unwrap().clone();
        assert!(blocos.iter().all(|b| b.len() <= 8) && blocos[..blocos.len() - 1].iter().all(|b| b.len() == 8));
        assert_eq!(blocos.concat(), esperado);
        // Primeiro token e palavra: argmax -> detokenização -> entrega com quebra de linha.
        assert_eq!((r.medida.token, r.medida.palavra.as_str(), r.medida.visivel), (aqui, "Here", true));
        assert_eq!(saida, b"Here\n");
        assert_eq!(r.medida.ids_n, esperado.len());
        assert_eq!(r.medida.ids_fnv, fnv_ids(&esperado));
        assert!((r.medida.gap - 3.0).abs() < 1e-6 && r.medida.logits_ok);
        // Cronômetros: os oito, separados, válidos; o produto cobre as etapas; T_entrada vem de fora.
        let c = r.medida.crono;
        assert!(c.valido() && c.entrada_ms == 0.25, "{c:?}");
        assert!(r.medida.produto_ms >= c.soma_etapas_ms() - 1e-6 && r.medida.produto_ms > 0.0, "{:?}", r.medida);
        assert!(c.token_ms > 0.0 && c.sequencia_ms >= 0.0 && c.escolha_ms >= 0.0);
        assert_eq!(r.logits.len(), VOCAB);
    }

    #[test]
    fn primeiro_token_so_de_espaco_e_gravado_como_nao_visivel() {
        let t = tok();
        let nl = t.codifica("\n")[0] as usize;
        let (r, saida) = roda("Olá, tudo bem?", Ok(logits_com_pico(nl)), &prep(2048));
        let r = r.unwrap();
        assert_eq!((r.medida.visivel, r.medida.palavra.as_str()), (false, "\n"));
        assert_eq!(saida, b"\n\n");
    }

    #[test]
    fn fluxo_recusa_texto_vazio_e_propaga_erros() {
        for vazio in ["", "   ", "\n\t"] {
            let (r, saida) = roda(vazio, Ok(logits_com_pico(5)), &prep(2048));
            assert!(r.err().is_some_and(|e| e.contains("vazio")), "{vazio:?}");
            assert!(saida.is_empty());
        }
        // Cache cheio: o texto estoura o max_len do preparo, o erro vem de le_ids e nada é entregue.
        let (r, saida) = roda("uma pergunta qualquer sobre protocolos", Ok(logits_com_pico(5)), &prep(8));
        assert!(r.err().is_some_and(|e| e.contains("CacheCheio")));
        assert!(saida.is_empty());
        // Falha na leitura dos logits.
        let (r, saida) = roda("outra pergunta", Err("falha dos logits".into()), &prep(2048));
        assert_eq!(r.err().as_deref(), Some("falha dos logits"));
        assert!(saida.is_empty());
        // Falha na criação da sequência.
        let t = tok();
        let mut saida: Vec<u8> = Vec::new();
        let r = primeira_palavra::<FakeGpu, PrepCompartilhado, _>(Instant::now(), 0.0, "oi", &data_fixa_texto(), &t, || Err("sem memória".into()), &mut saida);
        assert_eq!(r.err().as_deref(), Some("sem memória"));
    }

    #[test]
    fn logits_nao_finitos_ficam_marcados() {
        let mut l = logits_com_pico(7);
        l[3] = f32::NAN;
        let (r, _) = roda("pergunta", Ok(l), &prep(2048));
        assert!(!r.unwrap().medida.logits_ok);
    }

    #[test]
    fn saida_com_erro_de_escrita_falha() {
        struct Quebrada;
        impl Write for Quebrada {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(std::io::Error::new(std::io::ErrorKind::BrokenPipe, "fechada"))
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let t = tok();
        let (_g, eng) = motor_falso(2100);
        let p = PrepCompartilhado(prep(2048));
        let r = primeira_palavra(Instant::now(), 0.0, "pergunta", &data_fixa_texto(), &t, move || Ok(Sequencia { eng, prep: p, le_logits: Box::new(|| Ok(logits_com_pico(9))) }), &mut Quebrada);
        assert!(r.err().is_some_and(|e| e.contains("entrega")));
    }

    /// Preparo que zera a posição quando o backend falso recebe um `reset_sequence` novo (como o modelo real).
    struct PrepRe {
        gpu: Arc<FakeGpu>,
        visto: Mutex<usize>,
        pos: Mutex<u64>,
    }
    impl PrepRe {
        fn novo(gpu: &Arc<FakeGpu>) -> Self {
            Self { gpu: gpu.clone(), visto: Mutex::new(0), pos: Mutex::new(0) }
        }
        fn sincroniza(&self) {
            let n = self.gpu.resets().len();
            let mut v = self.visto.lock().unwrap();
            if n > *v {
                *v = n;
                *self.pos.lock().unwrap() = 0;
            }
        }
    }
    impl PreparaPasso for PrepRe {
        type Error = &'static str;
        fn vocab(&self) -> usize {
            VOCAB
        }
        fn max_positions(&self) -> usize {
            8
        }
        fn max_len(&self) -> usize {
            2048
        }
        fn posicao(&self) -> u64 {
            self.sincroniza();
            *self.pos.lock().unwrap()
        }
        fn prepara(&self, ids: &[usize]) -> Result<(), &'static str> {
            self.sincroniza();
            *self.pos.lock().unwrap() += ids.len() as u64;
            Ok(())
        }
    }

    /// Uma sequência nova (motor falso novo) cujos logits falham na `falha_na`-ésima chamada (1-based; 0 = nunca). Devolve também a GPU.
    fn seq_re(falha_na: usize, gpus: &Mutex<Vec<Arc<FakeGpu>>>) -> Result<Sequencia<FakeGpu, PrepRe>, String> {
        let (gpu, eng) = motor_falso(2100);
        gpus.lock().unwrap().push(gpu.clone());
        let n = Arc::new(Mutex::new(0usize));
        let logits = Box::new(move || {
            let mut c = n.lock().unwrap();
            *c += 1;
            if *c == falha_na {
                Err("logits falharam".to_string())
            } else {
                Ok(logits_com_pico(10921))
            }
        });
        Ok(Sequencia { eng, prep: PrepRe::novo(&gpu), le_logits: logits })
    }

    fn pedido_re(r: &mut SequenciaReutilizavel<FakeGpu, PrepRe>, texto: &str, falha_na: usize, gpus: &Mutex<Vec<Arc<FakeGpu>>>) -> Result<Resultado, String> {
        let t = tok();
        let mut saida: Vec<u8> = Vec::new();
        primeira_palavra_reutilizando(Instant::now(), 0.0, texto, &data_fixa_texto(), &t, r, || seq_re(falha_na, gpus), &mut saida)
    }

    /// Dois pedidos no mesmo motor: um só motor criado, um reset entre eles, ids e posições idênticos aos de um pedido avulso, e os
    /// contadores de cada pedido são a diferença dos cumulativos (sem slots, tudo vai para a CPU: `needed = ids * camadas * topk`).
    #[test]
    fn motor_reutilizado_entre_pedidos_zera_posicao_e_conta_por_pedido() {
        let gpus = Mutex::new(vec![]);
        let mut r = SequenciaReutilizavel::nova();
        let (t1, t2) = ("Explain how TCP congestion control works.", "Hi");
        let a = pedido_re(&mut r, t1, 0, &gpus).expect("pedido 1");
        assert!(r.viva());
        let b = pedido_re(&mut r, t2, 0, &gpus).expect("pedido 2");
        let c = pedido_re(&mut r, t1, 0, &gpus).expect("pedido 3");
        assert_eq!(r.criacoes, 1, "um motor só");
        assert_eq!(gpus.lock().unwrap().len(), 1);
        assert_eq!(gpus.lock().unwrap()[0].resets().len(), 2, "um reset por pedido depois do primeiro");
        // O pedido avulso (motor novo) dá os mesmos ids, token e contadores.
        let t = tok();
        let (_g, eng) = motor_falso(2100);
        let p = PrepCompartilhado(prep(2048));
        let mut saida: Vec<u8> = Vec::new();
        let avulso = primeira_palavra(Instant::now(), 0.0, t1, &data_fixa_texto(), &t, move || Ok(Sequencia { eng, prep: p, le_logits: Box::new(|| Ok(logits_com_pico(10921))) }), &mut saida).unwrap();
        for (x, y) in [(&a, &avulso), (&c, &avulso)] {
            assert_eq!((x.medida.ids_fnv, x.medida.ids_n, x.medida.token), (y.medida.ids_fnv, y.medida.ids_n, y.medida.token));
            assert_eq!(x.logits, y.logits);
            assert_eq!(x.medida.cont, y.medida.cont, "contadores por pedido = diferença");
        }
        assert_eq!(b.medida.cont.needed, b.medida.ids_n as u64 * 4);
        assert_eq!(a.medida.cont.needed, a.medida.ids_n as u64 * 4);
        assert_ne!(a.medida.ids_n, b.medida.ids_n);
        r.encerra().unwrap();
        assert!(!r.viva());
    }

    fn gpu_unused(g: &Mutex<Vec<Arc<FakeGpu>>>) -> &Mutex<Vec<Arc<FakeGpu>>> {
        g
    }

    /// Pedido que falha fecha o motor; o seguinte recria (e funciona), sem herdar contadores nem pendente.
    #[test]
    fn pedido_que_falha_fecha_o_motor_e_o_proximo_recria() {
        let gpus = Mutex::new(vec![]);
        let mut r = SequenciaReutilizavel::nova();
        pedido_re(&mut r, "um", 2, &gpus).expect("pedido 1 (logits na 1ª chamada)");
        let e = pedido_re(&mut r, "dois", 2, &gpus).err().expect("pedido 2 falha nos logits");
        assert!(e.contains("logits falharam"), "{e}");
        assert!(!r.viva(), "falha fecha e solta o motor");
        let c = pedido_re(&mut r, "tres", 0, &gpus).expect("pedido 3 recria");
        assert_eq!(r.criacoes, 2);
        assert_eq!(gpus.lock().unwrap().len(), 2);
        assert_eq!(c.medida.cont.needed, c.medida.ids_n as u64 * 4, "contadores do motor recriado começam do zero");
        r.encerra().unwrap();
    }

    /// Texto vazio não toca no motor guardado.
    #[test]
    fn texto_vazio_recusa_sem_fechar_o_motor_reutilizado() {
        let gpus = Mutex::new(vec![]);
        let mut r = SequenciaReutilizavel::nova();
        pedido_re(&mut r, "um", 0, &gpus).unwrap();
        let e = pedido_re(&mut r, "   ", 0, &gpus).err().unwrap();
        assert_eq!(e, "texto vazio");
        assert!(r.viva(), "entrada recusada não derruba o motor");
        pedido_re(&mut r, "tres", 0, &gpus).unwrap();
        assert_eq!(r.criacoes, 1);
        r.encerra().unwrap();
    }
}
