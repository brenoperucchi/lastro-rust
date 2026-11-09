//! A2 (textos longos), etapa 5c: GERAÇÃO LIVRE CURTA a partir de texto cru (critério `qual-runs/A2-LOTE-20261103/criterio-5c-geracao-livre-20261108.md`).
//! Regras puras (sem GPU, testadas em CPU com o motor falso): o laço guloso sobre o motor (texto -> template -> ids -> `le_ids` -> logits ->
//! argmax -> realimenta o token com `le_ids(&[token])` -> logits ...), a comparação por pedido com a referência do HF e o registro gravado.
//!
//! Regra de rótulo por pedido: PASSA se os 64 ids gerados são iguais aos da referência, ou se a PRIMEIRA divergência acontece num passo em que
//! o gap top1-top2 da referência é menor que 1e-3 (empate documentado; depois dela não se compara mais). REPROVA: ids do prompt diferentes
//! quando os dois lados rodaram, ou primeira divergência com gap >= 1e-3. SEM PROVA: erro, geração incompleta (!= 64), referência ausente ou
//! malformada, hash de entrada diferente, gap não finito no passo da divergência, integridade quebrada. REPROVA vence SEM PROVA; integridade
//! quebrada derruba tudo. Modelo PASSA se os 3 pedidos PASSAM; conjunto PASSA se 1B e 3B PASSAM. A prova gera sempre os 64 tokens (não para
//! no EOS); o uso normal para no primeiro EOS.
use crate::e4b_regra::{argmax, mediana};
use crate::leitura_regra::Contadores;
use crate::motor::backend::Backend;
use crate::motor::engine::Published;
use crate::motor::leitura::{le_ids_com_pendente, PreparaPasso};
use crate::primeira_palavra_regra::{dois_maiores, fnv_ids, RotuloCorrecao, Sequencia, GAP_EMPATE};
use crate::tokenizador::{renderiza_chat, Mensagem, Papel, Tokenizador};
use serde_json::{json, Value};
use std::fmt::Debug;
use std::io::Write;
use std::time::Instant;

/// sha256 do critério da 5c congelado (08/11/2026).
pub const CRITERIO_5C_SHA: &str = "9e3299fd9bfaac4b3d8ec2bb5c2ad941c24e2e7124070feb7dc511079a831731";
/// Tokens novos da prova.
pub const N_NOVOS: usize = 64;
/// Pedidos por modelo (curto, médio, longo), nesta ordem.
pub const PEDIDOS: [&str; 3] = ["curto", "medio", "longo"];
/// O EOS do Granite (`<|end_of_text|>`).
pub const EOS: usize = 0;

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

// ---- O laço guloso -------------------------------------------------------------------------------------------------------------------

/// O que o laço faz: quantos tokens novos e se para no primeiro EOS (a prova NÃO para).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CfgGera {
    pub novos: usize,
    pub para_no_eos: bool,
}

impl CfgGera {
    pub fn prova() -> Self {
        Self { novos: N_NOVOS, para_no_eos: false }
    }
}

/// O que um pedido produziu.
#[derive(Clone, Debug, PartialEq)]
pub struct Geracao {
    /// Ids do prompt (texto renderizado pelo template e tokenizado).
    pub ids: Vec<usize>,
    /// Tokens gerados (inclui o EOS, se o uso normal parou nele).
    pub gerados: Vec<usize>,
    /// Gap top1-top2 do lado Rust por passo (só descrição; a regra usa o da referência).
    pub gaps: Vec<f64>,
    /// Tempo de cada passo de realimentação (le_ids de 1 id + leitura dos logits + escolha), em ms.
    pub passo_ms: Vec<f64>,
    /// `T_sequencia` (criação do motor ou `nova_sequencia`), em ms.
    pub sequencia_ms: f64,
    /// Da leitura do prompt até os logits do primeiro token novo, em ms.
    pub prefill_ms: f64,
    /// Do texto cru até o último token escolhido, em ms.
    pub total_ms: f64,
    /// Texto detokenizado dos tokens gerados até o primeiro EOS (exclusive).
    pub texto: String,
    pub cont: Contadores,
}

impl Geracao {
    pub fn ids_fnv(&self) -> u64 {
        fnv_ids(&self.ids)
    }
    pub fn gerados_fnv(&self) -> u64 {
        fnv_ids(&self.gerados)
    }
    /// Mediana dos passos de realimentação (NaN sem passo).
    pub fn ms_por_token_mediana(&self) -> f64 {
        mediana(&self.passo_ms)
    }
}

/// Motor que atravessa os pedidos (a sequência, o `Published` pendente do próximo token e os contadores cumulativos do pedido anterior).
pub struct MotorGeracao<B: Backend, P: PreparaPasso> {
    seq: Option<Sequencia<B, P>>,
    pendente: Option<Published>,
    base: Contadores,
    /// Quantas vezes o motor foi criado (1 + recriações depois de falha).
    pub criacoes: usize,
}

impl<B: Backend, P: PreparaPasso> Default for MotorGeracao<B, P> {
    fn default() -> Self {
        Self { seq: None, pendente: None, base: Contadores::default(), criacoes: 0 }
    }
}

impl<B: Backend, P: PreparaPasso> MotorGeracao<B, P> {
    pub fn novo() -> Self {
        Self::default()
    }
    pub fn viva(&self) -> bool {
        self.seq.as_ref().is_some_and(|s| !s.eng.is_closed())
    }
    /// Encerra o motor guardado (solta o `Published` pendente antes de fechar).
    pub fn encerra(&mut self) -> Result<(), String> {
        self.pendente = None;
        self.base = Contadores::default();
        match self.seq.take() {
            Some(s) => s.eng.close().map_err(|e| format!("{e:?}")),
            None => Ok(()),
        }
    }
}

/// Um pedido: texto cru -> template (`data`) -> ids -> `le_ids` -> logits -> laço guloso (`cfg.novos` tokens; cada token escolhido entra por
/// `le_ids(&[token])`, um passo de uma posição, e os logits saem pelo `le_logits` da sequência). O motor guardado em `r` é criado por `cria` no
/// primeiro pedido (ou depois de uma falha) e nos demais recebe `nova_sequencia`. O último token escolhido não é realimentado (a referência
/// também não precisa dos logits depois dele). Imprime em `saida` o texto detokenizado até o primeiro EOS e uma quebra de linha. Qualquer
/// erro fecha o motor (o próximo pedido recria) e devolve a mensagem; nada repete.
pub fn gera<B, P, C>(t_entrada: Instant, texto: &str, data: &str, tok: &Tokenizador, r: &mut MotorGeracao<B, P>, cria: C, cfg: CfgGera,
                     saida: &mut dyn Write) -> Result<Geracao, String>
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    C: FnOnce() -> Result<Sequencia<B, P>, String>,
{
    if texto.trim().is_empty() {
        return Err("texto vazio".into());
    }
    if cfg.novos == 0 {
        return Err("novos = 0".into());
    }
    let corre = gera_corpo(t_entrada, texto, data, tok, r, cria, cfg);
    let g = match corre {
        Ok(g) => g,
        Err(e) => {
            r.encerra().ok();
            return Err(e);
        }
    };
    saida.write_all(g.texto.as_bytes()).and_then(|_| saida.write_all(b"\n")).and_then(|_| saida.flush()).map_err(|e| {
        r.encerra().ok();
        format!("entrega: {e}")
    })?;
    // Fora dos cronômetros: o `Published` do próximo pedido e os contadores cumulativos.
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
            Ok(Geracao { cont, ..g })
        }
        Err(e) => {
            r.encerra().ok();
            Err(e)
        }
    }
}

fn gera_corpo<B, P, C>(t_entrada: Instant, texto: &str, data: &str, tok: &Tokenizador, r: &mut MotorGeracao<B, P>, cria: C, cfg: CfgGera) -> Result<Geracao, String>
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    C: FnOnce() -> Result<Sequencia<B, P>, String>,
{
    let renderizado = renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: texto.to_string() }], true, data).map_err(|e| e.to_string())?;
    let ids: Vec<usize> = tok.codifica(&renderizado).into_iter().map(|x| x as usize).collect();
    let MotorGeracao { seq, pendente, base, criacoes } = r;
    // Sequência: motor novo (primeiro pedido ou depois de falha) ou `nova_sequencia`.
    let t_seq = Instant::now();
    if seq.as_ref().is_some_and(|x| !x.eng.is_closed()) && pendente.is_some() {
        seq.as_mut().ok_or("sem sequência")?.eng.nova_sequencia().map_err(|e| format!("nova_sequencia: {e:?}"))?;
    } else {
        if let Some(velho) = seq.take() {
            let _ = velho.eng.close();
        }
        *pendente = None;
        *base = Contadores::default();
        *seq = Some(cria()?);
        *criacoes += 1;
    }
    let sequencia_ms = ms(t_seq.elapsed());
    let s = seq.as_mut().ok_or("sem sequência")?;
    // Prefill: o prompt inteiro, até os logits do primeiro token novo.
    let t_pre = Instant::now();
    le_ids_com_pendente(&mut s.eng, &s.prep, &ids, pendente).map_err(|e| format!("le_ids: {e:?}"))?;
    let mut logits = (s.le_logits)()?;
    let prefill_ms = ms(t_pre.elapsed());
    let n = ids.len() as u64;
    if s.eng.position() != n || s.prep.posicao() != n {
        return Err(format!("posição do motor {} e do backend {} depois de ler {n} ids", s.eng.position(), s.prep.posicao()));
    }
    let mut gerados: Vec<usize> = Vec::with_capacity(cfg.novos);
    let mut gaps: Vec<f64> = Vec::with_capacity(cfg.novos);
    let mut passo_ms: Vec<f64> = Vec::with_capacity(cfg.novos);
    let mut t_passo = Instant::now();
    for i in 0..cfg.novos {
        let token = argmax(&logits);
        gaps.push(dois_maiores(&logits).map(|x| x.2).unwrap_or(f64::NAN));
        gerados.push(token);
        if i > 0 {
            passo_ms.push(ms(t_passo.elapsed()));
        }
        if cfg.para_no_eos && token == EOS {
            break;
        }
        if i + 1 < cfg.novos {
            t_passo = Instant::now();
            le_ids_com_pendente(&mut s.eng, &s.prep, &[token], &mut None).map_err(|e| format!("le_ids (passo {i}): {e:?}"))?;
            logits = (s.le_logits)()?;
        }
    }
    let total_ms = ms(t_entrada.elapsed());
    // Fora dos cronômetros: a posição tem de ser a do prompt mais os tokens realimentados.
    let esperado = n + gerados.len() as u64 - u64::from(!gerados.is_empty() && (gerados.len() == cfg.novos || cfg.para_no_eos && gerados.last() == Some(&EOS)));
    if s.eng.position() != esperado || s.prep.posicao() != esperado {
        return Err(format!("posição do motor {} e do backend {} no fim, esperada {esperado}", s.eng.position(), s.prep.posicao()));
    }
    let ate_eos: Vec<u32> = gerados.iter().take_while(|t| **t != EOS).map(|t| *t as u32).collect();
    let texto_saida = tok.decodifica(&ate_eos).map_err(|e| e.to_string())?;
    Ok(Geracao { ids, gerados, gaps, passo_ms, sequencia_ms, prefill_ms, total_ms, texto: texto_saida, cont: Contadores::default() })
}

// ---- Medidas de velocidade (comparação com o llama.cpp, emenda 2 de 08/11) -----------------------------------------------------------

/// Média por passo de realimentação = soma(passo_ms) / nº de passos (63 para 64 tokens). NaN sem passo. Nunca por diferença de totais.
pub fn media_por_passo(passo_ms: &[f64]) -> f64 {
    if passo_ms.is_empty() {
        f64::NAN
    } else {
        passo_ms.iter().sum::<f64>() / passo_ms.len() as f64
    }
}

impl Geracao {
    /// Média por passo de decode (ver `media_por_passo`); `T_sequencia`, tokenização e prefill ficam de fora.
    pub fn media_por_passo_ms(&self) -> f64 {
        media_por_passo(&self.passo_ms)
    }
}

/// Uma medida de velocidade: o pedido de aquecimento (descartada), a medida fria (única) ou uma repetição quente.
#[derive(Clone, Debug, PartialEq)]
pub struct MedidaGera {
    /// "frio", "aquecimento" ou "quente rep k".
    pub rotulo: String,
    pub descartada: bool,
    pub geracao: Result<Geracao, String>,
}

/// JSON de uma medida: ids do prompt, gerados, passo_ms, tempos, contadores e a média por passo.
pub fn medida_json(m: &MedidaGera) -> Value {
    let base = match &m.geracao {
        Ok(g) => json!({
            "ids": g.ids, "ids_n": g.ids.len(), "gerados": g.gerados, "gerados_n": g.gerados.len(), "passo_ms": g.passo_ms, "passo_n": g.passo_ms.len(),
            "prefill_ms": g.prefill_ms, "total_ms": g.total_ms, "T_sequencia_ms": g.sequencia_ms,
            "media_por_passo_ms": num_ou_nulo(g.media_por_passo_ms()), "ms_por_token_mediana_descritiva": num_ou_nulo(g.ms_por_token_mediana()),
            "cont": {"needed": g.cont.needed, "hits": g.cont.hits, "cpu": g.cont.cpu},
            "fracao_cpu_needed": num_ou_nulo(if g.cont.needed > 0 { g.cont.cpu as f64 / g.cont.needed as f64 } else { f64::NAN }),
            "falha": null,
        }),
        Err(e) => json!({"falha": e}),
    };
    let mut v = json!({"rotulo": m.rotulo, "descartada": m.descartada});
    if let (Some(a), Some(b)) = (v.as_object_mut(), base.as_object()) {
        a.extend(b.iter().map(|(k, x)| (k.clone(), x.clone())));
    }
    v
}

/// O arquivo `medidas-<nome>.json`: uma entrada por pedido, na ordem em que rodaram.
pub fn medidas_json(modelo: &str, texto: &str, medidas: &[MedidaGera]) -> Value {
    json!({
        "tipo": "medidas-gera",
        "modelo": modelo,
        "texto": texto,
        "novos": N_NOVOS,
        "definicao": "media_por_passo_ms = soma(passo_ms)/len(passo_ms); total_ms e T_sequencia_ms ficam separados e nunca entram na média",
        "medidas": medidas.iter().map(medida_json).collect::<Vec<_>>(),
    })
}

/// Aquecimento descartado + `reps` medidas no mesmo motor reutilizado (`nova_sequencia`). O primeiro erro para a série (a medida com erro é
/// guardada). `emite` é chamado a cada medida concluída, na ordem (para imprimir a linha).
pub fn gera_serie<B, P, C>(texto: &str, data: &str, tok: &Tokenizador, r: &mut MotorGeracao<B, P>, mut cria: C, reps: usize, saida: &mut dyn Write,
                           emite: &mut dyn FnMut(&MedidaGera)) -> Vec<MedidaGera>
where
    B: Backend,
    P: PreparaPasso,
    P::Error: Debug,
    C: FnMut() -> Result<Sequencia<B, P>, String>,
{
    let mut v: Vec<MedidaGera> = vec![];
    for k in 0..=reps {
        let (rotulo, descartada) = if k == 0 { ("aquecimento".to_string(), true) } else { (format!("quente rep {k}"), false) };
        let geracao = gera(Instant::now(), texto, data, tok, r, &mut cria, CfgGera::prova(), saida);
        let m = MedidaGera { rotulo, descartada, geracao };
        emite(&m);
        let erro = m.geracao.is_err();
        v.push(m);
        if erro {
            break;
        }
    }
    v
}

// ---- A referência --------------------------------------------------------------------------------------------------------------------

/// O que a referência do HF guarda de um pedido.
#[derive(Clone, Debug, PartialEq)]
pub struct RefGeracao {
    pub nome: String,
    pub texto_sha256: String,
    pub ids: Vec<usize>,
    pub gerados: Vec<usize>,
    pub gaps: Vec<f64>,
}

fn lista_usize(v: &Value, k: &str) -> Result<Vec<usize>, String> {
    v.get(k).and_then(Value::as_array).ok_or_else(|| format!("falta a lista {k}"))?.iter().map(|x| x.as_u64().map(|n| n as usize).ok_or_else(|| format!("{k}: id inválido"))).collect()
}

impl RefGeracao {
    pub fn de_json(v: &Value) -> Result<Self, String> {
        let gaps = v.get("gaps").and_then(Value::as_array).ok_or("falta a lista gaps")?.iter().map(|x| x.as_f64().ok_or("gap inválido")).collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            nome: v.get("nome").and_then(Value::as_str).ok_or("falta nome")?.to_string(),
            texto_sha256: v.get("texto_sha256").and_then(Value::as_str).ok_or("falta texto_sha256")?.to_string(),
            ids: lista_usize(v, "ids")?,
            gerados: lista_usize(v, "gerados")?,
            gaps,
        })
    }
    pub fn para_json(&self) -> Value {
        json!({"nome": self.nome, "texto_sha256": self.texto_sha256, "ids": self.ids, "gerados": self.gerados, "gaps": self.gaps})
    }
}

/// Lê o arquivo de referência (`ref5c-<modelo>.json`): a lista `casos`.
pub fn le_referencia(v: &Value) -> Result<Vec<RefGeracao>, String> {
    v.get("casos").and_then(Value::as_array).ok_or("falta a lista casos")?.iter().map(RefGeracao::de_json).collect()
}

// ---- A comparação --------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResPedido {
    Passa,
    Reprova,
    SemProva,
}

impl ResPedido {
    pub fn nome(self) -> &'static str {
        match self {
            ResPedido::Passa => "PASSA",
            ResPedido::Reprova => "REPROVA",
            ResPedido::SemProva => "SEM PROVA",
        }
    }
}

/// Um pedido da prova como o programa o grava.
#[derive(Clone, Debug, PartialEq)]
pub struct PedidoGeracao {
    pub indice: usize,
    pub nome: String,
    /// sha256 do arquivo de texto que o Rust leu.
    pub texto_sha256: String,
    pub geracao: Result<Geracao, String>,
    pub referencia: Option<RefGeracao>,
}

/// O resultado da comparação de um pedido.
#[derive(Clone, Debug, PartialEq)]
pub struct Veredito {
    pub res: ResPedido,
    pub motivo: String,
    /// Passo da primeira divergência (se houve) e gap da referência nele.
    pub divergencia: Option<(usize, f64)>,
    /// A divergência foi um empate documentado (gap < 1e-3).
    pub empate: bool,
}

/// O rótulo de um pedido (sem a integridade). Ordem: sem referência ou sem geração -> SEM PROVA; hash do texto diferente -> SEM PROVA (a
/// comparação não vale); ids do prompt diferentes -> REPROVA; referência malformada -> SEM PROVA; divergência com gap >= 1e-3 -> REPROVA
/// (vence a geração incompleta); geração incompleta -> SEM PROVA; divergência em empate ou nenhuma -> PASSA.
pub fn avalia_pedido(p: &PedidoGeracao) -> Veredito {
    let v = |res, motivo: String, divergencia, empate| Veredito { res, motivo: format!("pedido {} ({}): {motivo}", p.indice, p.nome), divergencia, empate };
    let sem = |m: String| v(ResPedido::SemProva, m, None, false);
    let Some(r) = &p.referencia else { return sem("sem referência".into()) };
    let g = match &p.geracao {
        Ok(g) => g,
        Err(e) => return sem(format!("sem geração ({e})")),
    };
    if r.texto_sha256 != p.texto_sha256 {
        return sem(format!("hash do texto diferente (Rust {}, referência {})", p.texto_sha256, r.texto_sha256));
    }
    if g.ids != r.ids {
        let pos = g.ids.iter().zip(&r.ids).position(|(a, b)| a != b);
        return v(ResPedido::Reprova, format!("ids do prompt DIFERENTES (Rust {} ids, referência {}, primeiro {:?})", g.ids.len(), r.ids.len(), pos), None, false);
    }
    if r.gerados.len() != N_NOVOS || r.gaps.len() != N_NOVOS {
        return sem(format!("referência malformada ({} ids gerados, {} gaps, esperados {N_NOVOS})", r.gerados.len(), r.gaps.len()));
    }
    let diverge = g.gerados.iter().zip(&r.gerados).position(|(a, b)| a != b);
    if let Some(i) = diverge {
        let gap = r.gaps[i];
        if !gap.is_finite() {
            return sem(format!("divergência no passo {i} com gap não finito na referência"));
        }
        if gap >= GAP_EMPATE {
            return v(ResPedido::Reprova, format!("DIVERGE no passo {i} (Rust {}, referência {}, gap da referência {gap:.6})", g.gerados[i], r.gerados[i], ), Some((i, gap)), false);
        }
    }
    if g.gerados.len() != N_NOVOS {
        return sem(format!("geração incompleta ({} tokens, esperados {N_NOVOS})", g.gerados.len()));
    }
    match diverge {
        None => v(ResPedido::Passa, format!("{N_NOVOS}/{N_NOVOS} ids iguais"), None, false),
        Some(i) => v(ResPedido::Passa, format!("igual até o passo {i}, onde a referência tinha empate (gap {:.3e}); depois dele não se compara", r.gaps[i]), Some((i, r.gaps[i])), true),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegistroGeracao {
    pub modelo: String,
    pub integro: bool,
    /// Falha fora dos pedidos (config, carga do modelo, encerramento do motor).
    pub falha: Option<String>,
    pub pedidos: Vec<PedidoGeracao>,
    /// Quantas vezes o motor foi criado (esperado 1: reutilizado entre os 3 pedidos).
    pub criacoes: usize,
    pub extras: Vec<(String, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AvalGeracao {
    pub rotulo: RotuloCorrecao,
    pub veredictos: Vec<Veredito>,
    pub motivos: Vec<String>,
}

/// O rótulo do modelo (sem a integridade): qualquer REPROVA vence; sem REPROVA, pedido faltando ou fora de ordem, SEM PROVA em pedido,
/// falha fora dos pedidos ou motor não reutilizado dá SEM PROVA.
pub fn avalia(r: &RegistroGeracao) -> AvalGeracao {
    let veredictos: Vec<Veredito> = r.pedidos.iter().map(avalia_pedido).collect();
    let mut motivos: Vec<String> = veredictos.iter().filter(|x| x.res != ResPedido::Passa).map(|x| x.motivo.clone()).collect();
    let ordem_ok = r.pedidos.len() == PEDIDOS.len() && r.pedidos.iter().enumerate().all(|(i, p)| p.indice == i && p.nome == PEDIDOS[i]);
    if !ordem_ok {
        motivos.push(format!("{} pedidos (ou fora da ordem curto, medio, longo); esperados {}", r.pedidos.len(), PEDIDOS.len()));
    }
    if let Some(f) = &r.falha {
        motivos.push(format!("falha: {f}"));
    }
    let com_erro = r.pedidos.iter().any(|p| p.geracao.is_err());
    let nao_reutilizou = !r.pedidos.is_empty() && !com_erro && r.criacoes != 1;
    if nao_reutilizou {
        motivos.push(format!("o motor não foi reutilizado ({} criações, esperada 1)", r.criacoes));
    }
    let rotulo = if veredictos.iter().any(|x| x.res == ResPedido::Reprova) {
        RotuloCorrecao::Reprova
    } else if !ordem_ok || r.falha.is_some() || nao_reutilizou || veredictos.iter().any(|x| x.res == ResPedido::SemProva) {
        RotuloCorrecao::SemProva
    } else {
        RotuloCorrecao::Passa
    };
    AvalGeracao { rotulo, veredictos, motivos }
}

/// Integridade quebrada derruba tudo para SEM PROVA.
pub fn rotulo_final(r: RotuloCorrecao, integro: bool) -> RotuloCorrecao {
    if integro {
        r
    } else {
        RotuloCorrecao::SemProva
    }
}

/// Conjunto dos modelos: PASSA se 1B e 3B PASSAM; REPROVA vence SEM PROVA; integridade quebrada ou falta de modelo é SEM PROVA.
/// Depois da integridade e da REPROVA (agregador compartilhado), exige exatamente os dois rótulos (1B e 3B): outro número é SEM PROVA.
pub fn rotulo_conjunto(v: &[RotuloCorrecao], integro: bool) -> RotuloCorrecao {
    let r = crate::primeira_palavra_regra::rotulo_conjunto_correcao(v, integro);
    if r == RotuloCorrecao::Passa && v.len() != MODELOS.len() {
        RotuloCorrecao::SemProva
    } else {
        r
    }
}

/// Os modelos exigidos no conjunto da 5c.
pub const MODELOS: [&str; 2] = ["1b", "3b"];

/// Modelos exigidos que não aparecem entre os executados.
pub fn modelos_ausentes(executados: &[&str]) -> Vec<&'static str> {
    MODELOS.iter().copied().filter(|m| !executados.contains(m)).collect()
}

/// Texto da decisão do conjunto e o rótulo global. `modelos` traz (nome, rótulo final do modelo) na ordem executada; `rotulos` os rótulos sem integridade.
pub fn decisao_conjunto(modelos: &[(String, RotuloCorrecao)], rotulos: &[RotuloCorrecao], integro: bool) -> (String, RotuloCorrecao) {
    let conj = rotulo_conjunto(rotulos, integro);
    let linhas: Vec<String> = modelos.iter().map(|(n, r)| format!("{n}: {}", r.nome())).collect();
    let nomes: Vec<&str> = modelos.iter().map(|(n, _)| n.as_str()).collect();
    let aus = modelos_ausentes(&nomes);
    let mut dec = format!("etapa 5c, geração livre curta (critério sha256 {})\n{}\nintegridade do conjunto: {}\n", CRITERIO_5C_SHA, linhas.join("\n"), if integro { "íntegra" } else { "QUEBRADA: todos SEM PROVA" });
    if !aus.is_empty() {
        dec += &format!("modelo(s) ausente(s) no conjunto: {} (o critério exige 1B e 3B; sem os dois não há PASSA)\n", aus.join(", "));
    }
    dec += "ids de prompt diferentes entre o Rust e a referência valem REPROVA (leitura da seção \"Regra de rótulo\" do critério; a seção \"Referência\" fala em SEM PROVA)\n";
    dec += &format!("rótulo: {}\n", conj.nome());
    (dec, conj)
}

// ---- Descritivos e registro ---------------------------------------------------------------------------------------------------------

/// Mediana dos passos de realimentação de todos os pedidos do modelo (NaN sem passo).
pub fn ms_por_token_do_modelo(r: &RegistroGeracao) -> f64 {
    let v: Vec<f64> = r.pedidos.iter().filter_map(|p| p.geracao.as_ref().ok()).flat_map(|g| g.passo_ms.iter().copied()).collect();
    mediana(&v)
}

/// A lista de empates usados (pedido, passo, gap da referência, token escolhido pelo Rust no passo).
pub fn empates(r: &RegistroGeracao) -> Vec<(usize, usize, f64, usize)> {
    r.pedidos.iter().filter_map(|p| {
        let v = avalia_pedido(p);
        match (v.empate, v.divergencia, &p.geracao) {
            (true, Some((i, g)), Ok(ge)) => Some((p.indice, i, g, ge.gerados[i])),
            _ => None,
        }
    }).collect()
}

fn num_ou_nulo(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}

fn geracao_json(g: &Result<Geracao, String>) -> Value {
    match g {
        Ok(g) => json!({
            "ids": g.ids, "ids_n": g.ids.len(), "ids_fnv": format!("{:016x}", g.ids_fnv()), "gerados": g.gerados, "gerados_n": g.gerados.len(),
            "gerados_fnv": format!("{:016x}", g.gerados_fnv()), "gaps_rust": g.gaps.iter().map(|x| num_ou_nulo(*x)).collect::<Vec<_>>(), "texto": g.texto,
            "T_sequencia_ms": g.sequencia_ms, "prefill_ms": g.prefill_ms, "total_ms": g.total_ms, "ms_por_token_mediana": num_ou_nulo(g.ms_por_token_mediana()),
            "passo_ms": g.passo_ms, "cont": {"needed": g.cont.needed, "hits": g.cont.hits, "cpu": g.cont.cpu}, "falha": null,
        }),
        Err(e) => json!({"falha": e}),
    }
}

impl RegistroGeracao {
    /// O resultado gravado, vinculado ao critério da 5c (sha256 dentro), com o rótulo e, por pedido, a comparação com a referência.
    pub fn para_json(&self) -> Value {
        let a = avalia(self);
        let pedidos: Vec<Value> = self.pedidos.iter().zip(&a.veredictos).map(|(p, v)| json!({
            "indice": p.indice, "nome": p.nome, "texto_sha256": p.texto_sha256, "rotulo": v.res.nome(), "motivo": v.motivo,
            "primeira_divergencia": v.divergencia.map(|d| d.0), "gap_na_divergencia": v.divergencia.map(|d| d.1), "empate_documentado": v.empate,
            "geracao": geracao_json(&p.geracao), "referencia": p.referencia.as_ref().map(RefGeracao::para_json),
        })).collect();
        json!({
            "tipo": "geracao-livre-5c",
            "criterio": "criterio-5c-geracao-livre-20261108.md",
            "criterio_sha256": CRITERIO_5C_SHA,
            "modelo": self.modelo,
            "novos": N_NOVOS,
            "gap_empate": GAP_EMPATE,
            "integro": self.integro,
            "falha": self.falha,
            "criacoes_do_motor": self.criacoes,
            "rotulo_sem_integridade": a.rotulo.nome(),
            "rotulo": rotulo_final(a.rotulo, self.integro).nome(),
            "motivos": a.motivos,
            "descritivo": {"ms_por_token_mediana": num_ou_nulo(ms_por_token_do_modelo(self)), "empates": empates(self).iter().map(|(p, i, g, t)| json!({"pedido": p, "passo": i, "gap": g, "token_rust": t})).collect::<Vec<_>>()},
            "pedidos": pedidos,
            "extras": self.extras.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
        })
    }
}

/// Texto do relatório de um modelo.
pub fn texto(r: &RegistroGeracao, final_: RotuloCorrecao) -> String {
    let a = avalia(r);
    let mut t = format!("== geração livre 5c {}\n", r.modelo);
    for (p, v) in r.pedidos.iter().zip(&a.veredictos) {
        t += &format!("{} -> {}\n", v.motivo, v.res.nome());
        if let Ok(g) = &p.geracao {
            t += &format!("  descritivo: prefill {:.1} ms ({} ids), ms por token (mediana) {:.2}, total {:.1} ms, T_sequencia {:.3} ms\n", g.prefill_ms, g.ids.len(), g.ms_por_token_mediana(), g.total_ms, g.sequencia_ms);
        }
    }
    let e = empates(r);
    if e.is_empty() {
        t += "empates usados: nenhum\n";
    } else {
        t += &format!("empates usados: {}\n", e.iter().map(|(p, i, g, t)| format!("pedido {p} passo {i} gap {g:.3e} token do Rust {t}")).collect::<Vec<_>>().join("; "));
    }
    t += &format!("ms por token do modelo (mediana de todos os passos): {:.2}\n", ms_por_token_do_modelo(r));
    if a.rotulo != RotuloCorrecao::Passa || !r.integro {
        t += &format!("motivo(s): {}\n", a.motivos.join("; "));
    }
    t += &format!("criações do motor: {} (esperado 1)\n", r.criacoes);
    t += &format!("rótulo: {}\n", final_.nome());
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg};
    use crate::motor::{AdmitMode, Config};
    use crate::primeira_palavra_regra::data_fixa_texto;
    use std::path::Path;
    use std::sync::{Arc, Mutex};

    const TOKENIZER_1B: &str = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct/tokenizer.json";
    const VOCAB: usize = 49155;
    const TEXTOS: [&str; 3] = ["Hi", "Explain how a bicycle gear system works.", "Please describe, in a few friendly sentences, how a small public library could organize its opening hours."];

    fn tok() -> Tokenizador {
        Tokenizador::carrega(Path::new(TOKENIZER_1B)).expect("tokenizer.json")
    }

    fn rota(p: usize, l: usize) -> Vec<u32> {
        vec![((p + l) % 4) as u32, ((p + l + 1 + (p / 4) % 3) % 4) as u32]
    }

    fn motor_falso() -> (Arc<FakeGpu>, crate::motor::Engine<FakeGpu>) {
        let routing: Vec<Vec<Vec<u32>>> = (0..2100).map(|p| (0..2).map(|l| rota(p, l)).collect()).collect();
        let gpu = FakeGpu::new_sched(
            FakeCfg { layers: 2, experts: 4, topk: 2, width: 3, routing, fail_main_at: None, copy_immediate: false, max_positions: 8 },
            SchedCfg { seed: 0xA0A, weights: [1, 1, 1, 1], budget: 300_000 },
        );
        let cfg = Config { layers: 2, experts: 4, topk: 2, width: 3, slots: 0, backlog: 0, admit: AdmitMode::TokenBarrier, first_token: 1, prefill: vec![], max_positions: 8, mutant: None };
        let parts = gpu.parts(&cfg);
        let eng = crate::motor::Engine::new(gpu.clone(), parts, cfg, FakeCpu { weighted: true, ..FakeCpu::default() }).expect("motor");
        (gpu, eng)
    }

    thread_local! {
        /// Os registros de ids recebidos de cada `Prep` criado nesta thread de teste.
        static RECEBIDOS: std::cell::RefCell<Vec<Arc<Mutex<Vec<Vec<usize>>>>>> = const { std::cell::RefCell::new(vec![]) };
    }

    /// Ids recebidos pelo preparo, por pedido, de todos os `Prep` criados na thread (zera a lista).
    fn recebidos_por_pedido() -> Vec<Vec<usize>> {
        RECEBIDOS.with(|r| r.borrow_mut().drain(..).flat_map(|a| a.lock().unwrap().clone()).filter(|v| !v.is_empty()).collect())
    }

    /// Preparo que zera a posição quando o backend falso recebe um `reset_sequence` novo (como o modelo real).
    struct Prep {
        gpu: Arc<FakeGpu>,
        visto: Mutex<usize>,
        pos: Mutex<u64>,
        /// Concatenação de todos os ids recebidos em `prepara`, por pedido (zera quando o backend recebe um `reset_sequence` novo).
        recebidos: Arc<Mutex<Vec<Vec<usize>>>>,
    }
    impl Prep {
        fn novo(gpu: &Arc<FakeGpu>) -> Self {
            Self { gpu: gpu.clone(), visto: Mutex::new(0), pos: Mutex::new(0), recebidos: Arc::new(Mutex::new(vec![])) }
        }
        fn sincroniza(&self) {
            let n = self.gpu.resets().len();
            let mut v = self.visto.lock().unwrap();
            if n > *v {
                *v = n;
                *self.pos.lock().unwrap() = 0;
                self.recebidos.lock().unwrap().push(vec![]);
            }
        }
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
            2048
        }
        fn posicao(&self) -> u64 {
            self.sincroniza();
            *self.pos.lock().unwrap()
        }
        fn prepara(&self, ids: &[usize]) -> Result<(), &'static str> {
            self.sincroniza();
            *self.pos.lock().unwrap() += ids.len() as u64;
            let mut r = self.recebidos.lock().unwrap();
            if r.is_empty() {
                r.push(vec![]);
            }
            r.last_mut().unwrap().extend_from_slice(ids);
            Ok(())
        }
    }

    /// Logits determinísticos pelo número de leituras DO PEDIDO: pico num id que muda a cada passo, segundo colocado a `gap` de distância;
    /// `eos_em` põe o pico no EOS (id 0) naquele passo. `atual` zera no começo de cada pedido.
    fn logits_do_passo(passo: usize, eos_em: Option<usize>, gap: f32) -> Vec<f32> {
        let mut v: Vec<f32> = (0..VOCAB).map(|i| ((i * 31 + passo * 7) % 1000) as f32 * 1e-3).collect();
        let pico = if eos_em == Some(passo) { 0 } else { 100 + (passo * 37) % 5000 };
        v[pico] = 5.0;
        v[(pico + 1) % VOCAB] = 5.0 - gap;
        v
    }

    fn sequencia(cont: &Arc<Mutex<usize>>, eos_em: Option<usize>, gap: f32, falha_a_partir: Option<usize>) -> Result<Sequencia<FakeGpu, Prep>, String> {
        let (gpu, eng) = motor_falso();
        let cont = cont.clone();
        let le: Box<dyn FnMut() -> Result<Vec<f32>, String>> = Box::new(move || {
            let mut c = cont.lock().unwrap();
            let passo = *c;
            *c += 1;
            if falha_a_partir.is_some_and(|f| passo >= f) {
                return Err("logits falharam".into());
            }
            Ok(logits_do_passo(passo, eos_em, gap))
        });
        let prep = Prep::novo(&gpu);
        RECEBIDOS.with(|r| r.borrow_mut().push(prep.recebidos.clone()));
        Ok(Sequencia { eng, prep, le_logits: le })
    }

    /// Roda os 3 pedidos num motor reutilizado; devolve as gerações, as criações e o texto impresso. O contador de logits é zerado por pedido.
    fn roda(cfg: CfgGera, eos_em: Option<usize>, gap: f32, falha_a_partir: Option<usize>) -> (Vec<Result<Geracao, String>>, usize, String) {
        let t = tok();
        let cont = Arc::new(Mutex::new(0usize));
        let mut r: MotorGeracao<FakeGpu, Prep> = MotorGeracao::novo();
        let mut saida: Vec<u8> = vec![];
        let mut v = vec![];
        for texto in TEXTOS {
            *cont.lock().unwrap() = 0;
            let c2 = cont.clone();
            v.push(gera(Instant::now(), texto, &data_fixa_texto(), &t, &mut r, move || sequencia(&c2, eos_em, gap, falha_a_partir), cfg, &mut saida));
        }
        let n = r.criacoes;
        r.encerra().unwrap();
        (v, n, String::from_utf8(saida).unwrap())
    }

    fn ped(i: usize, g: Result<Geracao, String>, r: Option<RefGeracao>) -> PedidoGeracao {
        PedidoGeracao { indice: i, nome: PEDIDOS[i].into(), texto_sha256: "aa".into(), geracao: g, referencia: r }
    }

    fn geracao_sintetica(seed: usize) -> Geracao {
        let ids: Vec<usize> = (0..20).map(|i| 100 + i + seed).collect();
        let gerados: Vec<usize> = (0..N_NOVOS).map(|i| 1000 + i * 3 + seed).collect();
        Geracao { ids, gerados, gaps: vec![1.0; N_NOVOS], passo_ms: vec![10.0; N_NOVOS - 1], sequencia_ms: 0.0, prefill_ms: 5.0, total_ms: 700.0, texto: "x".into(), cont: Contadores::default() }
    }

    fn ref_de(g: &Geracao, i: usize) -> RefGeracao {
        RefGeracao { nome: PEDIDOS[i].into(), texto_sha256: "aa".into(), ids: g.ids.clone(), gerados: g.gerados.clone(), gaps: vec![1.0; N_NOVOS] }
    }

    fn rec(p: Vec<PedidoGeracao>) -> RegistroGeracao {
        RegistroGeracao { modelo: "1b".into(), integro: true, falha: None, pedidos: p, criacoes: 1, extras: vec![] }
    }

    fn tres(muda: impl Fn(usize, &mut Geracao, &mut RefGeracao)) -> Vec<PedidoGeracao> {
        (0..3).map(|i| {
            let mut g = geracao_sintetica(i);
            let mut r = ref_de(&g, i);
            muda(i, &mut g, &mut r);
            ped(i, Ok(g), Some(r))
        }).collect()
    }

    #[test]
    fn ordem_e_constantes_do_criterio() {
        assert_eq!(PEDIDOS, ["curto", "medio", "longo"]);
        assert_eq!(N_NOVOS, 64);
        assert_eq!(GAP_EMPATE, 1e-3);
        assert_eq!(CRITERIO_5C_SHA.len(), 64);
        let r = rec(tres(|_, _, _| {}));
        assert!(avalia(&r).motivos.is_empty());
        // fora de ordem, faltando pedido: SEM PROVA
        let mut p = tres(|_, _, _| {});
        p.swap(0, 1);
        assert_eq!(avalia(&rec(p)).rotulo, RotuloCorrecao::SemProva);
        let mut p = tres(|_, _, _| {});
        p.pop();
        assert_eq!(avalia(&rec(p)).rotulo, RotuloCorrecao::SemProva);
    }

    #[test]
    fn sessenta_e_quatro_de_sessenta_e_quatro_passa() {
        let r = rec(tres(|_, _, _| {}));
        let a = avalia(&r);
        assert_eq!(a.rotulo, RotuloCorrecao::Passa, "{:?}", a.motivos);
        assert!(a.veredictos.iter().all(|v| v.res == ResPedido::Passa && !v.empate && v.divergencia.is_none()));
        let j = r.para_json();
        assert_eq!(j["criterio_sha256"], CRITERIO_5C_SHA);
        assert_eq!(j["rotulo"], "PASSA");
        assert_eq!(j["pedidos"][0]["geracao"]["gerados_n"], 64);
        assert!(empates(&r).is_empty());
        assert_eq!(ms_por_token_do_modelo(&r), 10.0);
        assert!(texto(&r, RotuloCorrecao::Passa).contains("rótulo: PASSA"));
    }

    #[test]
    fn divergencia_em_empate_documentado_passa_e_nao_compara_depois() {
        let r = rec(tres(|i, g, r| {
            if i == 1 {
                r.gaps[10] = 5e-4;
                g.gerados[10] += 1;
                // Depois do empate tudo pode ser diferente: não se compara.
                for k in 11..N_NOVOS {
                    g.gerados[k] += 7;
                }
            }
        }));
        let a = avalia(&r);
        assert_eq!(a.rotulo, RotuloCorrecao::Passa, "{:?}", a.motivos);
        assert!(a.veredictos[1].empate && a.veredictos[1].divergencia == Some((10, 5e-4)));
        assert_eq!(empates(&r), vec![(1, 10, 5e-4, 1032)]);
        assert_eq!(r.para_json()["descritivo"]["empates"][0]["passo"], 10);
        assert_eq!(r.para_json()["descritivo"]["empates"][0]["token_rust"], 1032);
        assert!(texto(&r, RotuloCorrecao::Passa).contains("token do Rust 1032"));
        // Gap exatamente 1e-3 NÃO é empate (estritamente menor).
        let r = rec(tres(|i, g, r| {
            if i == 0 {
                r.gaps[3] = 1e-3;
                g.gerados[3] += 1;
            }
        }));
        assert_eq!(avalia(&r).rotulo, RotuloCorrecao::Reprova);
    }

    #[test]
    fn divergencia_com_gap_grande_reprova_mesmo_depois_de_um_empate_nao() {
        let r = rec(tres(|i, g, r| {
            if i == 2 {
                r.gaps[0] = 0.5;
                g.gerados[0] += 1;
            }
        }));
        let a = avalia(&r);
        assert_eq!(a.rotulo, RotuloCorrecao::Reprova);
        assert_eq!(a.veredictos[2].res, ResPedido::Reprova);
        // Só a PRIMEIRA divergência conta: empate no passo 5, gap grande no 9 (não comparado): PASSA.
        let r = rec(tres(|i, g, r| {
            if i == 0 {
                r.gaps[5] = 1e-5;
                g.gerados[5] += 1;
                r.gaps[9] = 3.0;
                g.gerados[9] += 1;
            }
        }));
        assert_eq!(avalia(&r).rotulo, RotuloCorrecao::Passa);
    }

    #[test]
    fn ids_do_prompt_diferentes_reprovam() {
        let r = rec(tres(|i, g, _| {
            if i == 1 {
                g.ids[3] += 1;
            }
        }));
        let a = avalia(&r);
        assert_eq!(a.veredictos[1].res, ResPedido::Reprova);
        assert_eq!(a.rotulo, RotuloCorrecao::Reprova);
        // Mesmo com geração incompleta: os dois lados rodaram, REPROVA vence.
        let r = rec(tres(|i, g, _| {
            if i == 1 {
                g.ids.pop();
                g.gerados.truncate(5);
            }
        }));
        assert_eq!(avalia(&r).veredictos[1].res, ResPedido::Reprova);
    }

    #[test]
    fn geracao_curta_erro_e_referencia_ausente_sao_sem_prova() {
        let curta = rec(tres(|i, g, _| {
            if i == 0 {
                g.gerados.truncate(63);
            }
        }));
        let a = avalia(&curta);
        assert_eq!((a.rotulo, a.veredictos[0].res), (RotuloCorrecao::SemProva, ResPedido::SemProva));
        let erro = rec(vec![ped(0, Err("boom".into()), Some(ref_de(&geracao_sintetica(0), 0))), ped(1, Ok(geracao_sintetica(1)), Some(ref_de(&geracao_sintetica(1), 1))), ped(2, Ok(geracao_sintetica(2)), Some(ref_de(&geracao_sintetica(2), 2)))]);
        assert_eq!(avalia(&erro).rotulo, RotuloCorrecao::SemProva);
        let sem_ref = rec(vec![ped(0, Ok(geracao_sintetica(0)), None), ped(1, Ok(geracao_sintetica(1)), None), ped(2, Ok(geracao_sintetica(2)), None)]);
        assert_eq!(avalia(&sem_ref).rotulo, RotuloCorrecao::SemProva);
        // Referência com menos de 64 passos: SEM PROVA.
        let r = rec(tres(|i, _, r| {
            if i == 2 {
                r.gerados.truncate(60);
            }
        }));
        assert_eq!(avalia(&r).rotulo, RotuloCorrecao::SemProva);
        // Hash do texto diferente: SEM PROVA, sem comparar.
        let r = rec(tres(|i, _, r| {
            if i == 0 {
                r.texto_sha256 = "bb".into();
            }
        }));
        assert_eq!(avalia(&r).veredictos[0].res, ResPedido::SemProva);
        // Gap não finito no passo da divergência: SEM PROVA.
        let r = rec(tres(|i, g, r| {
            if i == 0 {
                r.gaps[4] = f64::NAN;
                g.gerados[4] += 1;
            }
        }));
        assert_eq!(avalia(&r).veredictos[0].res, ResPedido::SemProva);
        // Divergência real numa geração curta: REPROVA vence a incompleta.
        let r = rec(tres(|i, g, r| {
            if i == 0 {
                g.gerados.truncate(20);
                g.gerados[10] += 1;
                r.gaps[10] = 1.0;
            }
        }));
        assert_eq!(avalia(&r).veredictos[0].res, ResPedido::Reprova);
    }

    #[test]
    fn precedencias_modelo_conjunto_e_integridade() {
        use RotuloCorrecao::*;
        // REPROVA num pedido e SEM PROVA em outro: REPROVA.
        let r = rec(tres(|i, g, r| {
            if i == 0 {
                g.ids[0] += 1;
            }
            if i == 1 {
                r.texto_sha256 = "zz".into();
            }
        }));
        assert_eq!(avalia(&r).rotulo, Reprova);
        // Falha fora dos pedidos, ou motor recriado: SEM PROVA.
        let mut r = rec(tres(|_, _, _| {}));
        r.falha = Some("encerramento".into());
        assert_eq!(avalia(&r).rotulo, SemProva);
        let mut r = rec(tres(|_, _, _| {}));
        r.criacoes = 2;
        assert_eq!(avalia(&r).rotulo, SemProva);
        // REPROVA ainda vence uma falha fora dos pedidos.
        let mut r = rec(tres(|i, g, _| {
            if i == 2 {
                g.ids[0] += 1;
            }
        }));
        r.falha = Some("x".into());
        assert_eq!(avalia(&r).rotulo, Reprova);
        // Integridade quebrada derruba tudo (até o REPROVA).
        assert_eq!(rotulo_final(Passa, false), SemProva);
        assert_eq!(rotulo_final(Reprova, false), SemProva);
        assert_eq!(rotulo_final(Reprova, true), Reprova);
        let mut r = rec(tres(|_, _, _| {}));
        r.integro = false;
        assert_eq!(r.para_json()["rotulo"], "SEM PROVA");
        // Conjunto.
        assert_eq!(rotulo_conjunto(&[Passa, Passa], true), Passa);
        assert_eq!(rotulo_conjunto(&[Passa, SemProva], true), SemProva);
        assert_eq!(rotulo_conjunto(&[SemProva, Reprova], true), Reprova);
        assert_eq!(rotulo_conjunto(&[Passa, Passa], false), SemProva);
        assert_eq!(rotulo_conjunto(&[], true), SemProva);
        // Um só modelo (falta 1B ou 3B): nunca PASSA.
        assert_eq!(rotulo_conjunto(&[Passa], true), SemProva);
        assert_eq!(rotulo_conjunto(&[SemProva], true), SemProva);
        assert_eq!(rotulo_conjunto(&[Reprova], true), Reprova);
        assert_eq!(rotulo_conjunto(&[Passa], false), SemProva);
        assert_eq!(rotulo_conjunto(&[Passa, Passa, Passa], true), SemProva);
    }

    #[test]
    fn relatorio_com_uma_unica_celula_passa_nao_da_passa_global() {
        use RotuloCorrecao::*;
        let r = rec(tres(|_, _, _| {}));
        let a = avalia(&r);
        assert_eq!(a.rotulo, Passa);
        let (dec, conj) = decisao_conjunto(&[("1b".into(), Passa)], &[a.rotulo], true);
        assert_eq!(conj, SemProva);
        assert!(dec.contains("rótulo: SEM PROVA\n") && dec.contains("ausente(s) no conjunto: 3b"), "{dec}");
        assert!(dec.contains("ids de prompt diferentes") && dec.contains("REPROVA"));
        let (dec, conj) = decisao_conjunto(&[("3b".into(), Passa)], &[Passa], true);
        assert!(conj == SemProva && dec.contains("ausente(s) no conjunto: 1b"));
        let (dec, conj) = decisao_conjunto(&[("1b".into(), Passa), ("3b".into(), Passa)], &[Passa, Passa], true);
        assert!(conj == Passa && !dec.contains("ausente") && dec.contains("rótulo: PASSA\n"));
        let (_, conj) = decisao_conjunto(&[("1b".into(), Passa)], &[Passa], false);
        assert_eq!(conj, SemProva);
    }

    #[test]
    fn laco_com_motor_falso_gera_64_tokens_nos_3_pedidos_com_um_motor_so() {
        let (v, criacoes, saida) = roda(CfgGera::prova(), None, 1.0, None);
        assert_eq!(criacoes, 1, "um motor só nos 3 pedidos (nova_sequencia)");
        assert_eq!(v.len(), 3);
        let t = tok();
        for (i, g) in v.iter().enumerate() {
            let g = g.as_ref().unwrap_or_else(|e| panic!("pedido {i}: {e}"));
            assert_eq!(g.gerados.len(), 64);
            assert_eq!(g.gaps.len(), 64);
            assert_eq!(g.passo_ms.len(), 63, "63 realimentações (o último token não é realimentado)");
            // Cada token é o argmax dos logits do passo correspondente.
            for (k, tk) in g.gerados.iter().enumerate() {
                assert_eq!(*tk, argmax(&logits_do_passo(k, None, 1.0)), "pedido {i} passo {k}");
            }
            assert!(g.gaps.iter().all(|x| (*x - 1.0).abs() < 1e-4));
            let esperados: Vec<usize> = t.codifica(&renderiza_chat(&[Mensagem { papel: Papel::Usuario, conteudo: TEXTOS[i].into() }], true, &data_fixa_texto()).unwrap()).into_iter().map(|x| x as usize).collect();
            assert_eq!(g.ids, esperados);
            assert!(g.prefill_ms >= 0.0 && g.total_ms >= g.prefill_ms);
        }
        // O que o laço realimenta: por pedido, ids do prompt ++ gerados[..63] (o último token não é realimentado).
        let rec = recebidos_por_pedido();
        assert_eq!(rec.len(), 3);
        for (i, g) in v.iter().enumerate() {
            let g = g.as_ref().unwrap();
            let mut esp = g.ids.clone();
            esp.extend_from_slice(&g.gerados[..63]);
            assert_eq!(rec[i], esp, "pedido {i}: realimenta os tokens escolhidos, na ordem, sem o último");
        }
        let esperado_saida: String = v.iter().map(|g| format!("{}\n", g.as_ref().unwrap().texto)).collect();
        assert_eq!(saida, esperado_saida, "cada pedido imprime o texto e uma quebra de linha");
        // Os ids de prompt dos 3 pedidos diferem (textos diferentes).
        assert!(v[0].as_ref().unwrap().ids != v[1].as_ref().unwrap().ids);
        // Reprodutível: o mesmo laço com motor novo por pedido dá a mesma sequência (o laço não depende do estado do motor).
        let ids_a: Vec<Vec<usize>> = v.iter().map(|g| g.as_ref().unwrap().gerados.clone()).collect();
        assert!(ids_a.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn eos_no_meio_nao_para_a_prova_mas_para_o_uso_normal() {
        let (v, _, saida_prova) = roda(CfgGera::prova(), Some(5), 1.0, None);
        let rec_prova = recebidos_por_pedido();
        for (i, g) in v.iter().enumerate() {
            let g = g.as_ref().unwrap();
            let mut esp = g.ids.clone();
            esp.extend_from_slice(&g.gerados[..63]);
            assert_eq!(rec_prova[i], esp, "a prova realimenta 63 tokens, inclusive o EOS do passo 5");
        }
        for g in &v {
            let g = g.as_ref().unwrap();
            assert_eq!(g.gerados.len(), 64, "a prova gera sempre 64");
            assert_eq!(g.gerados[5], EOS);
            // O texto impresso para no primeiro EOS (os 5 primeiros tokens).
            let t = tok();
            let esperado = t.decodifica(&g.gerados[..5].iter().map(|x| *x as u32).collect::<Vec<_>>()).unwrap();
            assert_eq!(g.texto, esperado);
        }
        let esperado_saida: String = v.iter().map(|g| format!("{}\n", g.as_ref().unwrap().texto)).collect();
        assert_eq!(saida_prova, esperado_saida);
        let (v, criacoes, _) = roda(CfgGera { novos: 64, para_no_eos: true }, Some(5), 1.0, None);
        assert_eq!(criacoes, 1);
        for g in &v {
            let g = g.as_ref().unwrap();
            assert_eq!(g.gerados.len(), 6, "uso normal para no EOS, que entra na lista");
            assert_eq!(*g.gerados.last().unwrap(), EOS);
        }
        // Uso normal com EOS no passo 5: realimenta ids ++ gerados[..5], sem o EOS.
        let rec = recebidos_por_pedido();
        assert_eq!(rec.len(), 3);
        for (i, g) in v.iter().enumerate() {
            let g = g.as_ref().unwrap();
            let mut esp = g.ids.clone();
            esp.extend_from_slice(&g.gerados[..5]);
            assert_eq!(rec[i], esp, "pedido {i}: o EOS não é realimentado");
        }
    }

    #[test]
    fn erro_no_meio_fecha_o_motor_e_o_proximo_pedido_recria() {
        // Logits falham no passo 10 de cada pedido: todos os pedidos dão erro e cada um recria o motor.
        let (v, criacoes, _) = roda(CfgGera::prova(), None, 1.0, Some(10));
        assert!(v.iter().all(|g| g.is_err()));
        assert_eq!(criacoes, 3, "cada falha fecha o motor e o pedido seguinte cria outro");
        let r = rec(v.into_iter().enumerate().map(|(i, g)| ped(i, g, Some(ref_de(&geracao_sintetica(i), i)))).collect());
        let mut r = r;
        r.criacoes = 3;
        assert_eq!(avalia(&r).rotulo, RotuloCorrecao::SemProva);
        // Texto vazio é recusado sem tocar no motor.
        let t = tok();
        let mut m: MotorGeracao<FakeGpu, Prep> = MotorGeracao::novo();
        let cont = Arc::new(Mutex::new(0usize));
        let e = gera(Instant::now(), "  ", &data_fixa_texto(), &t, &mut m, || sequencia(&cont, None, 1.0, None), CfgGera::prova(), &mut vec![]);
        assert!(e.is_err() && m.criacoes == 0);
    }

    #[test]
    fn referencia_json_ida_e_volta() {
        let g = geracao_sintetica(1);
        let r = ref_de(&g, 1);
        let v = json!({"meta": {}, "casos": [r.para_json()]});
        assert_eq!(le_referencia(&v).unwrap(), vec![r]);
        assert!(le_referencia(&json!({"casos": [{"nome": "x"}]})).is_err());
        assert!(le_referencia(&json!({})).is_err());
    }

    #[test]
    fn serie_quente_grava_aquecimento_e_duas_medidas_e_a_media_nao_inclui_a_criacao_do_motor() {
        let t = tok();
        let cont = Arc::new(Mutex::new(0usize));
        let mut r: MotorGeracao<FakeGpu, Prep> = MotorGeracao::novo();
        let mut saida: Vec<u8> = vec![];
        let mut linhas = 0usize;
        let c2 = cont.clone();
        let c3 = cont.clone();
        let mut chamadas = 0usize;
        let v = gera_serie(TEXTOS[1], &data_fixa_texto(), &t, &mut r, move || {
            chamadas += 1;
            *c2.lock().unwrap() = 0;
            std::thread::sleep(std::time::Duration::from_millis(250)); // atraso conhecido só na criação do motor
            sequencia(&c3, None, 1.0, None)
        }, 2, &mut saida, &mut |_| linhas += 1);
        assert_eq!(r.criacoes, 1, "um motor só: o aquecimento cria, as medidas usam nova_sequencia");
        r.encerra().unwrap();
        assert_eq!((v.len(), linhas), (3, 3), "1 aquecimento + 2 medidas");
        assert!(v[0].descartada && v[0].rotulo == "aquecimento" && !v[1].descartada && !v[2].descartada);
        assert_eq!((v[1].rotulo.as_str(), v[2].rotulo.as_str()), ("quente rep 1", "quente rep 2"));
        for m in &v {
            let g = m.geracao.as_ref().unwrap();
            assert_eq!((g.gerados.len(), g.passo_ms.len()), (64, 63));
            let esperado = g.passo_ms.iter().sum::<f64>() / g.passo_ms.len() as f64;
            assert!((g.media_por_passo_ms() - esperado).abs() < 1e-12);
        }
        let (a, q) = (v[0].geracao.as_ref().unwrap(), v[1].geracao.as_ref().unwrap());
        assert!(a.sequencia_ms >= 250.0 && a.total_ms >= 250.0, "a criação do motor entra em T_sequencia e em total_ms do aquecimento");
        assert!(q.sequencia_ms < 250.0 && q.total_ms < 250.0, "a repetição quente não paga a criação");
        assert!(a.media_por_passo_ms() < 100.0, "a média de decode do aquecimento não carrega o atraso de 250 ms ({} ms)", a.media_por_passo_ms());
        // total - prefill dividido por 63 NÃO é a média (contamina com a criação do motor).
        assert!((a.total_ms - a.prefill_ms) / 63.0 > a.media_por_passo_ms() + 3.0);
        // O JSON traz tudo por pedido.
        let j = medidas_json("1b", "medio", &v);
        let ms_ = j["medidas"].as_array().unwrap();
        assert_eq!(ms_.len(), 3);
        assert_eq!(ms_[0]["descartada"], json!(true));
        assert_eq!(ms_[2]["gerados"].as_array().unwrap().len(), 64);
        assert_eq!(ms_[2]["passo_ms"].as_array().unwrap().len(), 63);
        assert!(!ms_[2]["ids"].as_array().unwrap().is_empty());
        assert!(ms_[2]["cont"].get("needed").is_some() && ms_[2]["cont"].get("hits").is_some() && ms_[2]["cont"].get("cpu").is_some());
        assert!((ms_[2]["media_por_passo_ms"].as_f64().unwrap() - media_por_passo(&v[2].geracao.as_ref().unwrap().passo_ms)).abs() < 1e-12);
        assert!(media_por_passo(&[]).is_nan() && (media_por_passo(&[1.0, 2.0, 6.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn serie_quente_para_no_primeiro_erro_e_guarda_a_medida_com_falha() {
        let t = tok();
        let cont = Arc::new(Mutex::new(0usize));
        let mut r: MotorGeracao<FakeGpu, Prep> = MotorGeracao::novo();
        let c2 = cont.clone();
        let v = gera_serie(TEXTOS[0], &data_fixa_texto(), &t, &mut r, move || sequencia(&c2, None, 1.0, Some(10)), 3, &mut vec![], &mut |_| {});
        assert_eq!(v.len(), 1, "erro no aquecimento para a série");
        assert!(v[0].geracao.is_err() && v[0].descartada);
        assert_eq!(medida_json(&v[0])["falha"].as_str().is_some(), true);
    }
}
