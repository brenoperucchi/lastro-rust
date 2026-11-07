//! A2 (textos longos), etapa 4(a): as regras PURAS da medida de velocidade da leitura em blocos de 8 nos modelos reais (sem GPU, testadas em
//! CPU). Critério: `qual-runs/A2-LOTE-20261103/criterio-leitura-velocidade.txt` (congelado antes de qualquer dado real). Recorte aprovado:
//! `qual-runs/A2-LOTE-20261103/recorte-etapa4-velocidade-textos-longos-20261107.md`.
//!
//! Célula = modelo x cenário x estado (quente ou frio). Rótulos (NÃO é o rótulo da correção nem o do E4b): ATINGE, NÃO ATINGE e SEM PROVA.
//! Três níveis: (1) validade da execução por célula (SEM PROVA, e os tempos não são lidos como prova); (2) só em célula válida, ganho e
//! meta (quente) ou meta (frio); (3) rótulo da célula e do conjunto, com NÃO ATINGE vencendo SEM PROVA e SEM PROVA vencendo ATINGE.
//!
//! Escolhas de implementação que o recorte não diz (registradas para a revisão):
//! - mediana de número par: média dos dois centrais (a de `e4b_regra`);
//! - deriva = |mediana das 2 últimas - mediana das 2 primeiras repetições do modo 1| / a menor das duas medianas; estritamente maior que
//!   10% é SEM PROVA (exatamente 10% ainda vale). O denominador pelo menor valor é o mais rígido dos dois sentidos possíveis;
//! - mais de 5 pares (quente) ou de 3 repetições (frio) não existe pelo desenho: a célula é SEM PROVA, para não decidir sobre um conjunto
//!   que o critério não previu;
//! - a ordem alternada (par ímpar: modo 1 primeiro; par par: modo 8 primeiro) é conferida na avaliação: ordem diferente é SEM PROVA;
//! - desvio entre repetições: desvio-padrão amostral (n - 1), só descrição.
use crate::e4b_regra::{argmax, mediana};
use crate::leitura_regra::{compara_entre_modos, contadores_validos, Contadores, N_A, N_C, TOL_ENTRE_MODOS};

/// Pares válidos por célula quente (um par = uma repetição de cada modo).
pub const PARES_QUENTE: usize = 5;
/// Repetições por célula fria (cada uma em processo novo, só modo 8).
pub const REPS_FRIO: usize = 3;
/// Deriva máxima entre as duas primeiras e as duas últimas repetições do modo 1 (fração).
pub const DERIVA_MAX: f64 = 0.10;
/// Piso do ganho do modo 8 sobre o modo 1 (herdado do E4b).
pub const GANHO_MINIMO: f64 = crate::e4b_regra::GANHO_MINIMO;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cen {
    C,
    A,
}

impl Cen {
    pub fn nome(self) -> &'static str {
        match self {
            Cen::C => "C",
            Cen::A => "A",
        }
    }
    pub fn de_nome(s: &str) -> Option<Cen> {
        match s {
            "C" => Some(Cen::C),
            "A" => Some(Cen::A),
            _ => None,
        }
    }
    /// Ids lidos no cenário.
    pub fn n(self) -> usize {
        match self {
            Cen::C => N_C,
            Cen::A => N_A,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Estado {
    Quente,
    Frio,
}

impl Estado {
    pub fn nome(self) -> &'static str {
        match self {
            Estado::Quente => "quente",
            Estado::Frio => "frio",
        }
    }
    pub fn de_nome(s: &str) -> Option<Estado> {
        match s {
            "quente" => Some(Estado::Quente),
            "frio" => Some(Estado::Frio),
            _ => None,
        }
    }
}

/// A meta aceita em 06/11/2026 (em milissegundos): `None` para modelo fora de 1b e 3b.
pub fn meta_ms(estado: Estado, modelo: &str, cen: Cen) -> Option<f64> {
    let s = match (estado, modelo, cen) {
        (Estado::Quente, "1b", Cen::C) => 5.0,
        (Estado::Quente, "3b", Cen::C) => 10.0,
        (Estado::Quente, "1b", Cen::A) => 15.0,
        (Estado::Quente, "3b", Cen::A) => 30.0,
        (Estado::Frio, "1b", Cen::C) => 20.0,
        (Estado::Frio, "3b", Cen::C) => 30.0,
        (Estado::Frio, "1b", Cen::A) => 30.0,
        (Estado::Frio, "3b", Cen::A) => 45.0,
        _ => return None,
    };
    Some(s * 1000.0)
}

/// Ganho = 1 - mediana(modo 8) / mediana(modo 1). `None` se a mediana do modo 1 não é positiva e finita ou a do modo 8 não é finita.
pub fn ganho(med8: f64, med1: f64) -> Option<f64> {
    if !(med1.is_finite() && med1 > 0.0 && med8.is_finite()) {
        return None;
    }
    Some(1.0 - med8 / med1)
}

/// Deriva do modo 1: mediana das duas primeiras contra a das duas últimas repetições (fração; ver a nota do módulo). `None` com menos de 4
/// repetições ou valor não positivo.
pub fn deriva(m1: &[f64]) -> Option<f64> {
    if m1.len() < 4 || m1.iter().any(|x| !(x.is_finite() && *x > 0.0)) {
        return None;
    }
    let (a, b) = (mediana(&m1[..2]), mediana(&m1[m1.len() - 2..]));
    Some((a - b).abs() / a.min(b))
}

/// Desvio-padrão amostral (n - 1); 0 com menos de 2 valores. Só descrição.
pub fn desvio(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64).sqrt()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rotulo {
    Atinge,
    NaoAtinge,
    SemProva,
}

impl Rotulo {
    pub fn nome(self) -> &'static str {
        match self {
            Rotulo::Atinge => "ATINGE",
            Rotulo::NaoAtinge => "NÃO ATINGE",
            Rotulo::SemProva => "SEM PROVA",
        }
    }
}

/// Os tempos de uma repetição quente, em milissegundos. O indicador quente é `leitura + logits`; `sequencia` fica fora dele.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RepQuente {
    pub sequencia_ms: f64,
    pub leitura_ms: f64,
    pub logits_ms: f64,
}

impl RepQuente {
    pub fn indicador_ms(&self) -> f64 {
        self.leitura_ms + self.logits_ms
    }
    fn valida(&self) -> bool {
        self.sequencia_ms.is_finite() && self.sequencia_ms >= 0.0 && self.leitura_ms.is_finite() && self.leitura_ms > 0.0 && self.logits_ms.is_finite() && self.logits_ms > 0.0
    }
}

/// O portão de correção de um par: logits do modo 8 contra os do modo 1 (norma máxima relativa <= 1e-5) e argmax igual, e contadores
/// (needed, hits, cpu) iguais, com hits e cpu positivos.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PortaoPar {
    pub erro: f64,
    pub nao_finito: bool,
    pub argmax_ok: bool,
    pub cont_ok: bool,
}

impl PortaoPar {
    pub fn ok(&self) -> bool {
        !self.nao_finito && self.erro <= TOL_ENTRE_MODOS && self.argmax_ok && self.cont_ok
    }
}

pub fn portao_par(l8: &[f32], l1: &[f32], c8: Contadores, c1: Contadores) -> PortaoPar {
    let c = compara_entre_modos("par", l8, l1);
    PortaoPar { erro: c.erro, nao_finito: c.nao_finito, argmax_ok: !c.nao_finito && argmax(l8) == argmax(l1), cont_ok: contadores_validos(c8, c1) }
}

/// A ordem do par `indice` (a partir de 1): ímpar começa pelo modo 1, par pelo modo 8.
pub fn modo1_primeiro(indice: usize) -> bool {
    indice % 2 == 1
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParQuente {
    pub indice: usize,
    pub m1_primeiro: bool,
    pub m1: RepQuente,
    pub m8: RepQuente,
    pub portao: PortaoPar,
}

/// O resultado da avaliação de uma célula (quente ou fria). Os campos descritivos saem sempre que os dados permitem, mesmo com SEM PROVA,
/// mas só valem como prova com `rotulo` ATINGE ou NÃO ATINGE.
#[derive(Clone, Debug, PartialEq)]
pub struct Avaliacao {
    pub rotulo: Rotulo,
    /// Por que SEM PROVA (nível 1) ou por que NÃO ATINGE (nível 2: "ganho", "meta"); vazio em ATINGE.
    pub motivos: Vec<String>,
    pub meta_ms: Option<f64>,
    pub med1_ms: Option<f64>,
    /// Mediana do modo 8 (quente) ou das repetições (frio): o número que decide contra a meta.
    pub med8_ms: Option<f64>,
    pub ganho: Option<f64>,
    pub deriva: Option<f64>,
    /// Só descrição.
    pub pior_ms: Option<f64>,
    pub desvio_ms: Option<f64>,
    pub med_sequencia_ms: Option<f64>,
}

impl Avaliacao {
    fn vazia(meta: Option<f64>) -> Self {
        Avaliacao { rotulo: Rotulo::SemProva, motivos: vec![], meta_ms: meta, med1_ms: None, med8_ms: None, ganho: None, deriva: None, pior_ms: None, desvio_ms: None, med_sequencia_ms: None }
    }
    fn sem_prova(mut self, motivo: impl Into<String>) -> Self {
        self.rotulo = Rotulo::SemProva;
        self.motivos.push(motivo.into());
        self
    }
}

/// Avalia uma célula QUENTE (níveis 1 a 3 da célula; a integridade global entra em [`rotulo_final`]). `falha` é qualquer condição de nível 1
/// vista pelo programa (tempo-limite, posição diferente de N, erro de motor ou GPU, ids insuficientes...).
/// Avalia um registro quente lido de arquivo: a identidade GRAVADA (modelo e cenário) tem de ser a da célula pedida. Registro de outra célula
/// (por exemplo copiado para o nome de outra) nunca vira prova: a célula fica SEM PROVA e os tempos não são lidos.
pub fn avalia_registro_quente(modelo: &str, cen: Cen, r: &RegistroQuente) -> Avaliacao {
    if r.modelo != modelo || r.cen != cen {
        return avalia_quente(modelo, cen, &[], Some(&format!("identidade do registro ({} {}) diferente da célula pedida ({modelo} {}): não vale como prova", r.modelo, r.cen.nome(), cen.nome())));
    }
    avalia_quente(modelo, cen, &r.pares, r.falha.as_deref())
}

pub fn avalia_quente(modelo: &str, cen: Cen, pares: &[ParQuente], falha: Option<&str>) -> Avaliacao {
    let mut a = Avaliacao::vazia(meta_ms(Estado::Quente, modelo, cen));
    // Descrição, sempre que os tempos são legíveis.
    let legivel = !pares.is_empty() && pares.iter().all(|p| p.m1.valida() && p.m8.valida());
    if legivel {
        let (i1, i8): (Vec<f64>, Vec<f64>) = (pares.iter().map(|p| p.m1.indicador_ms()).collect(), pares.iter().map(|p| p.m8.indicador_ms()).collect());
        a.med1_ms = Some(mediana(&i1));
        a.med8_ms = Some(mediana(&i8));
        a.ganho = ganho(mediana(&i8), mediana(&i1));
        a.pior_ms = Some(i8.iter().cloned().fold(f64::MIN, f64::max));
        a.desvio_ms = Some(desvio(&i8));
        a.med_sequencia_ms = Some(mediana(&pares.iter().flat_map(|p| [p.m1.sequencia_ms, p.m8.sequencia_ms]).collect::<Vec<_>>()));
        a.deriva = deriva(&i1);
    }
    let Some(meta) = a.meta_ms else { return a.sem_prova(format!("sem meta para o modelo {modelo}")) };
    if let Some(f) = falha {
        return a.sem_prova(f.to_string());
    }
    let ruins: Vec<usize> = pares.iter().filter(|p| !p.portao.ok()).map(|p| p.indice).collect();
    if !ruins.is_empty() {
        return a.sem_prova(format!("portão de correção falhou nos pares {ruins:?} (logits entre modos, argmax ou contadores): ALERTA DE CORREÇÃO, velocidade com resultado diferente não vale"));
    }
    if pares.len() < PARES_QUENTE {
        return a.sem_prova(format!("menos de {PARES_QUENTE} pares válidos ({})", pares.len()));
    }
    if pares.len() > PARES_QUENTE {
        return a.sem_prova(format!("mais de {PARES_QUENTE} pares ({}): fora do desenho", pares.len()));
    }
    if pares.iter().enumerate().any(|(i, p)| p.indice != i + 1 || p.m1_primeiro != modo1_primeiro(p.indice)) {
        return a.sem_prova("ordem dos pares diferente da alternada combinada (ímpar: modo 1 primeiro; par: modo 8 primeiro)");
    }
    if !legivel {
        return a.sem_prova("tempo não finito, zero ou negativo em alguma repetição");
    }
    match a.deriva {
        None => return a.sem_prova("deriva não calculável"),
        Some(d) if d > DERIVA_MAX => return a.sem_prova(format!("deriva do modo 1 de {:.1}% acima do limite de {:.0}% (a máquina mudou durante a medida)", d * 100.0, DERIVA_MAX * 100.0)),
        _ => {}
    }
    // Nível 2.
    let (g, m8) = (a.ganho.unwrap_or(f64::NAN), a.med8_ms.unwrap_or(f64::NAN));
    if !(g >= GANHO_MINIMO) {
        a.motivos.push("ganho".into());
    }
    if !(m8 <= meta) {
        a.motivos.push("meta".into());
    }
    a.rotulo = if a.motivos.is_empty() { Rotulo::Atinge } else { Rotulo::NaoAtinge };
    a
}

/// O registro de uma repetição fria (um processo): tempo total do início do `main` até os logits lidos, mais o que o processo viu.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistroFrio {
    pub modelo: String,
    pub cen: Cen,
    pub rep: usize,
    pub total_ms: f64,
    /// Posição do motor e do backend iguais a N, hits e cpu positivos, logits finitos.
    pub ok: bool,
    pub falha: Option<String>,
    /// Integridade conferida no fim deste processo (binário, critério e ids contra os do início).
    pub integro: bool,
    /// Partes do tempo, só descrição: (nome, ms).
    pub partes: Vec<(String, f64)>,
    /// Outros pares chave/valor (hashes, contadores, hash dos logits...).
    pub extras: Vec<(String, String)>,
}

/// Avalia uma célula FRIA: 3 repetições, cada uma válida, mediana contra a meta fria. Sem comparação de ganho.
pub fn avalia_frio(modelo: &str, cen: Cen, regs: &[RegistroFrio]) -> Avaliacao {
    let mut a = Avaliacao::vazia(meta_ms(Estado::Frio, modelo, cen));
    let totais: Vec<f64> = regs.iter().map(|r| r.total_ms).collect();
    if !regs.is_empty() && totais.iter().all(|t| t.is_finite() && *t > 0.0) {
        a.med8_ms = Some(mediana(&totais));
        a.pior_ms = Some(totais.iter().cloned().fold(f64::MIN, f64::max));
        a.desvio_ms = Some(desvio(&totais));
    }
    let Some(meta) = a.meta_ms else { return a.sem_prova(format!("sem meta para o modelo {modelo}")) };
    if regs.len() < REPS_FRIO {
        return a.sem_prova(format!("menos de {REPS_FRIO} repetições frias ({})", regs.len()));
    }
    if regs.len() > REPS_FRIO {
        return a.sem_prova(format!("mais de {REPS_FRIO} repetições frias ({}): fora do desenho", regs.len()));
    }
    let mut idx: Vec<usize> = regs.iter().map(|r| r.rep).collect();
    idx.sort_unstable();
    if idx != (1..=REPS_FRIO).collect::<Vec<_>>() {
        return a.sem_prova(format!("repetições {idx:?} em vez de 1 a {REPS_FRIO}"));
    }
    if regs.iter().any(|r| r.modelo != modelo || r.cen != cen) {
        return a.sem_prova("registro de outra célula na mesma célula");
    }
    for r in regs {
        if let Some(f) = &r.falha {
            return a.sem_prova(format!("repetição {}: {f}", r.rep));
        }
        if !r.ok {
            return a.sem_prova(format!("repetição {}: posição, contadores ou logits inválidos", r.rep));
        }
        if !r.integro {
            return a.sem_prova(format!("repetição {}: integridade quebrada no fim do processo", r.rep));
        }
    }
    let Some(med) = a.med8_ms else { return a.sem_prova("tempo total não finito, zero ou negativo em alguma repetição") };
    if !(med <= meta) {
        a.motivos.push("meta".into());
    }
    a.rotulo = if a.motivos.is_empty() { Rotulo::Atinge } else { Rotulo::NaoAtinge };
    a
}

/// O rótulo final da célula: integridade quebrada deixa SEM PROVA, senão vale o rótulo da célula.
pub fn rotulo_final(r: Rotulo, integro: bool) -> Rotulo {
    if integro {
        r
    } else {
        Rotulo::SemProva
    }
}

/// O rótulo do conjunto: NÃO ATINGE se alguma célula NÃO ATINGE; senão SEM PROVA se alguma estiver sem prova (ou se não houver célula);
/// senão ATINGE. Integridade quebrada: SEM PROVA em tudo.
pub fn rotulo_conjunto(celulas: &[Rotulo], integro: bool) -> Rotulo {
    if !integro {
        return Rotulo::SemProva;
    }
    if celulas.contains(&Rotulo::NaoAtinge) {
        Rotulo::NaoAtinge
    } else if celulas.is_empty() || celulas.contains(&Rotulo::SemProva) {
        Rotulo::SemProva
    } else {
        Rotulo::Atinge
    }
}

// ---- Registros em texto (um arquivo por célula quente, um por processo frio) -------------------------------------------------------------

/// O registro bruto de uma célula quente, como gravado pelo programa e relido pela consolidação.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistroQuente {
    pub modelo: String,
    pub cen: Cen,
    /// Integridade acumulada até o fim desta célula (uma vez quebrada, fica quebrada).
    pub integro: bool,
    pub falha: Option<String>,
    pub pares: Vec<ParQuente>,
    /// Linhas descritivas do aquecimento (descartado); não são lidas de volta.
    pub aquecimento: Vec<String>,
    pub extras: Vec<(String, String)>,
}

fn b(x: bool) -> u8 {
    x as u8
}

fn linha_rep(p: &str, r: &RepQuente) -> String {
    format!("{p}_seq={:?} {p}_leit={:?} {p}_log={:?}", r.sequencia_ms, r.leitura_ms, r.logits_ms)
}

fn kv_de(linha: &str) -> std::collections::HashMap<&str, &str> {
    linha.split_whitespace().filter_map(|t| t.split_once('=')).collect()
}

fn num(m: &std::collections::HashMap<&str, &str>, k: &str) -> Result<f64, String> {
    m.get(k).ok_or_else(|| format!("falta {k}"))?.parse::<f64>().map_err(|_| format!("{k} ilegível"))
}

fn flag(m: &std::collections::HashMap<&str, &str>, k: &str) -> Result<bool, String> {
    match *m.get(k).ok_or_else(|| format!("falta {k}"))? {
        "1" => Ok(true),
        "0" => Ok(false),
        _ => Err(format!("{k} ilegível")),
    }
}

fn rep_de(m: &std::collections::HashMap<&str, &str>, p: &str) -> Result<RepQuente, String> {
    Ok(RepQuente { sequencia_ms: num(m, &format!("{p}_seq"))?, leitura_ms: num(m, &format!("{p}_leit"))?, logits_ms: num(m, &format!("{p}_log"))? })
}

fn uma_linha(s: &str) -> String {
    s.replace(['\n', '\r'], " ")
}

impl RegistroQuente {
    pub fn para_texto(&self) -> String {
        let mut t = format!("tipo=quente\nmodelo={}\ncenario={}\nintegro={}\n", self.modelo, self.cen.nome(), b(self.integro));
        if let Some(f) = &self.falha {
            t += &format!("falha={}\n", uma_linha(f));
        }
        for (k, v) in &self.extras {
            t += &format!("x.{k}={}\n", uma_linha(v));
        }
        for l in &self.aquecimento {
            t += &format!("# {}\n", uma_linha(l));
        }
        for p in &self.pares {
            t += &format!(
                "par ind={} m1_primeiro={} {} {} erro={:?} nao_finito={} argmax_ok={} cont_ok={}\n",
                p.indice,
                b(p.m1_primeiro),
                linha_rep("m1", &p.m1),
                linha_rep("m8", &p.m8),
                p.portao.erro,
                b(p.portao.nao_finito),
                b(p.portao.argmax_ok),
                b(p.portao.cont_ok)
            );
        }
        t
    }

    pub fn de_texto(t: &str) -> Result<Self, String> {
        let (mut modelo, mut cen, mut integro, mut falha, mut pares, mut extras) = (None, None, None, None, vec![], vec![]);
        for l in t.lines() {
            if l.starts_with('#') || l.trim().is_empty() {
                continue;
            }
            if let Some(r) = l.strip_prefix("par ") {
                let m = kv_de(r);
                pares.push(ParQuente {
                    indice: m.get("ind").ok_or("falta ind")?.parse().map_err(|_| "ind ilegível")?,
                    m1_primeiro: flag(&m, "m1_primeiro")?,
                    m1: rep_de(&m, "m1")?,
                    m8: rep_de(&m, "m8")?,
                    portao: PortaoPar { erro: num(&m, "erro")?, nao_finito: flag(&m, "nao_finito")?, argmax_ok: flag(&m, "argmax_ok")?, cont_ok: flag(&m, "cont_ok")? },
                });
                continue;
            }
            let (k, v) = l.split_once('=').ok_or_else(|| format!("linha ilegível: {l}"))?;
            match k {
                "tipo" if v == "quente" => {}
                "tipo" => return Err(format!("tipo {v}, esperado quente")),
                "modelo" => modelo = Some(v.to_string()),
                "cenario" => cen = Some(Cen::de_nome(v).ok_or("cenário ilegível")?),
                "integro" => integro = Some(v == "1"),
                "falha" => falha = Some(v.to_string()),
                _ => {
                    if let Some(x) = k.strip_prefix("x.") {
                        extras.push((x.to_string(), v.to_string()));
                    } else {
                        return Err(format!("chave desconhecida: {k}"));
                    }
                }
            }
        }
        Ok(RegistroQuente { modelo: modelo.ok_or("falta modelo")?, cen: cen.ok_or("falta cenário")?, integro: integro.ok_or("falta integro")?, falha, pares, aquecimento: vec![], extras })
    }
}

impl RegistroFrio {
    pub fn para_texto(&self) -> String {
        let mut t = format!("tipo=frio\nmodelo={}\ncenario={}\nrep={}\ntotal_ms={:?}\nok={}\nintegro={}\n", self.modelo, self.cen.nome(), self.rep, self.total_ms, b(self.ok), b(self.integro));
        if let Some(f) = &self.falha {
            t += &format!("falha={}\n", uma_linha(f));
        }
        for (k, v) in &self.partes {
            t += &format!("parte.{k}={v:?}\n");
        }
        for (k, v) in &self.extras {
            t += &format!("x.{k}={}\n", uma_linha(v));
        }
        t
    }

    pub fn de_texto(t: &str) -> Result<Self, String> {
        let (mut modelo, mut cen, mut rep, mut total, mut ok, mut integro, mut falha, mut partes, mut extras) = (None, None, None, None, None, None, None, vec![], vec![]);
        for l in t.lines().filter(|l| !l.trim().is_empty()) {
            let (k, v) = l.split_once('=').ok_or_else(|| format!("linha ilegível: {l}"))?;
            match k {
                "tipo" if v == "frio" => {}
                "tipo" => return Err(format!("tipo {v}, esperado frio")),
                "modelo" => modelo = Some(v.to_string()),
                "cenario" => cen = Some(Cen::de_nome(v).ok_or("cenário ilegível")?),
                "rep" => rep = Some(v.parse::<usize>().map_err(|_| "rep ilegível")?),
                "total_ms" => total = Some(v.parse::<f64>().map_err(|_| "total_ms ilegível")?),
                "ok" => ok = Some(v == "1"),
                "integro" => integro = Some(v == "1"),
                "falha" => falha = Some(v.to_string()),
                _ => {
                    if let Some(x) = k.strip_prefix("parte.") {
                        partes.push((x.to_string(), v.parse::<f64>().map_err(|_| "parte ilegível")?));
                    } else if let Some(x) = k.strip_prefix("x.") {
                        extras.push((x.to_string(), v.to_string()));
                    } else {
                        return Err(format!("chave desconhecida: {k}"));
                    }
                }
            }
        }
        Ok(RegistroFrio {
            modelo: modelo.ok_or("falta modelo")?,
            cen: cen.ok_or("falta cenário")?,
            rep: rep.ok_or("falta rep")?,
            total_ms: total.ok_or("falta total_ms")?,
            ok: ok.ok_or("falta ok")?,
            falha,
            integro: integro.ok_or("falta integro")?,
            partes,
            extras,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(l: f64, g: f64) -> RepQuente {
        RepQuente { sequencia_ms: 900.0, leitura_ms: l, logits_ms: g }
    }
    fn portao_ok() -> PortaoPar {
        PortaoPar { erro: 0.0, nao_finito: false, argmax_ok: true, cont_ok: true }
    }
    /// 5 pares com modo 1 em `t1` e modo 8 em `t8` ms (indicador), já na ordem alternada.
    fn pares(t1: [f64; 5], t8: [f64; 5]) -> Vec<ParQuente> {
        (0..5).map(|i| ParQuente { indice: i + 1, m1_primeiro: modo1_primeiro(i + 1), m1: rep(t1[i] - 10.0, 10.0), m8: rep(t8[i] - 10.0, 10.0), portao: portao_ok() }).collect()
    }
    fn frio(rep: usize, total: f64) -> RegistroFrio {
        RegistroFrio { modelo: "1b".into(), cen: Cen::C, rep, total_ms: total, ok: true, falha: None, integro: true, partes: vec![("pesos".into(), 1.5)], extras: vec![("hash".into(), "abc".into())] }
    }

    #[test]
    fn mediana_e_ganho() {
        assert_eq!(mediana(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(mediana(&[4.0, 1.0, 2.0, 3.0]), 2.5);
        assert!((ganho(80.0, 100.0).unwrap() - 0.2).abs() < 1e-12);
        assert!(ganho(1.0, 0.0).is_none() && ganho(f64::NAN, 1.0).is_none() && ganho(1.0, f64::NAN).is_none());
        assert_eq!(GANHO_MINIMO, 0.15);
        assert_eq!([PARES_QUENTE, REPS_FRIO], [5, 3]);
        assert_eq!(DERIVA_MAX, 0.10);
    }

    #[test]
    fn tabela_de_metas_do_recorte() {
        let t = |e, m, c| meta_ms(e, m, c).unwrap() / 1000.0;
        assert_eq!([t(Estado::Quente, "1b", Cen::C), t(Estado::Quente, "3b", Cen::C), t(Estado::Quente, "1b", Cen::A), t(Estado::Quente, "3b", Cen::A)], [5.0, 10.0, 15.0, 30.0]);
        assert_eq!([t(Estado::Frio, "1b", Cen::C), t(Estado::Frio, "3b", Cen::C), t(Estado::Frio, "1b", Cen::A), t(Estado::Frio, "3b", Cen::A)], [20.0, 30.0, 30.0, 45.0]);
        assert!(meta_ms(Estado::Quente, "7b", Cen::C).is_none());
        assert_eq!([Cen::C.n(), Cen::A.n()], [500, 2003]);
    }

    #[test]
    fn quente_atinge_com_ganho_e_meta() {
        let p = pares([4000.0, 4100.0, 3900.0, 4050.0, 4000.0], [2000.0, 2100.0, 1900.0, 2050.0, 2000.0]);
        let a = avalia_quente("1b", Cen::C, &p, None);
        assert_eq!(a.rotulo, Rotulo::Atinge, "{a:?}");
        assert!(a.motivos.is_empty());
        assert_eq!(a.med8_ms, Some(2000.0));
        assert!((a.ganho.unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(a.pior_ms, Some(2100.0));
        assert_eq!(a.med_sequencia_ms, Some(900.0));
    }

    #[test]
    fn quente_nao_atinge_diz_o_motivo() {
        // Ganho só 10%, meta ok (4,5 s <= 5 s).
        let g = avalia_quente("1b", Cen::C, &pares([5000.0; 5], [4500.0; 5]), None);
        assert_eq!((g.rotulo, g.motivos.clone()), (Rotulo::NaoAtinge, vec!["ganho".to_string()]));
        // Ganho 40%, meta estourada (6 s > 5 s).
        let m = avalia_quente("1b", Cen::C, &pares([10000.0; 5], [6000.0; 5]), None);
        assert_eq!((m.rotulo, m.motivos.clone()), (Rotulo::NaoAtinge, vec!["meta".to_string()]));
        // Os dois.
        let ambos = avalia_quente("1b", Cen::C, &pares([6000.0; 5], [5500.0; 5]), None);
        assert_eq!(ambos.rotulo, Rotulo::NaoAtinge);
        assert_eq!(ambos.motivos, vec!["ganho".to_string(), "meta".to_string()]);
        // A meta é contra a MEDIANA: uma repetição ruim isolada não derruba (pior repetição só descritiva).
        let mut t8 = [2000.0; 5];
        t8[2] = 9000.0;
        let a = avalia_quente("1b", Cen::C, &pares([4000.0; 5], t8), None);
        assert_eq!(a.rotulo, Rotulo::Atinge);
        assert_eq!(a.pior_ms, Some(9000.0));
        // Limites exatos: ganho de 15% e mediana igual à meta ainda atingem.
        let l = avalia_quente("3b", Cen::C, &pares([10000.0; 5], [8500.0; 5]), None);
        assert_eq!(l.rotulo, Rotulo::Atinge, "{l:?}");
        let l = avalia_quente("1b", Cen::C, &pares([10000.0; 5], [5000.0; 5]), None);
        assert_eq!(l.rotulo, Rotulo::Atinge, "{l:?}");
        let l = avalia_quente("3b", Cen::C, &pares([10000.0; 5], [8501.0; 5]), None);
        assert_eq!((l.rotulo, l.motivos), (Rotulo::NaoAtinge, vec!["ganho".to_string()]));
    }

    #[test]
    fn deriva_estritamente_acima_de_10_por_cento_e_sem_prova() {
        assert!((deriva(&[100.0, 100.0, 105.0, 110.0, 110.0]).unwrap() - 0.10).abs() < 1e-12);
        let ok = avalia_quente("1b", Cen::C, &pares([4000.0, 4000.0, 4000.0, 4400.0, 4400.0], [2000.0; 5]), None);
        assert_eq!(ok.rotulo, Rotulo::Atinge, "exatamente 10% ainda vale: {ok:?}");
        let ruim = avalia_quente("1b", Cen::C, &pares([4000.0, 4000.0, 4000.0, 4500.0, 4500.0], [2000.0; 5]), None);
        assert_eq!(ruim.rotulo, Rotulo::SemProva);
        assert!(ruim.motivos[0].contains("deriva"));
        // Deriva para baixo conta igual (a máquina mudou).
        let baixo = avalia_quente("1b", Cen::C, &pares([4500.0, 4500.0, 4000.0, 4000.0, 4000.0], [2000.0; 5]), None);
        assert_eq!(baixo.rotulo, Rotulo::SemProva);
        assert!(deriva(&[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn quente_validade_nivel_1() {
        let boas = pares([4000.0; 5], [2000.0; 5]);
        // Menos e mais pares.
        assert_eq!(avalia_quente("1b", Cen::C, &boas[..4], None).rotulo, Rotulo::SemProva);
        assert!(avalia_quente("1b", Cen::C, &boas[..4], None).motivos[0].contains("menos de 5"));
        assert_eq!(avalia_quente("1b", Cen::C, &[], None).rotulo, Rotulo::SemProva);
        let mut seis = boas.clone();
        seis.push(ParQuente { indice: 6, m1_primeiro: false, ..boas[0] });
        assert!(avalia_quente("1b", Cen::C, &seis, None).motivos[0].contains("fora do desenho"));
        // Falha externa (tempo-limite, posição, erro de motor...).
        let f = avalia_quente("1b", Cen::C, &boas, Some("posição do motor 499 depois de ler 500 ids"));
        assert_eq!(f.rotulo, Rotulo::SemProva);
        assert!(f.motivos[0].contains("posição"));
        // Portão do par: logits, argmax, contadores.
        for ruim in [PortaoPar { erro: 2e-5, ..portao_ok() }, PortaoPar { nao_finito: true, erro: f64::INFINITY, ..portao_ok() }, PortaoPar { argmax_ok: false, ..portao_ok() }, PortaoPar { cont_ok: false, ..portao_ok() }] {
            let mut p = boas.clone();
            p[3].portao = ruim;
            let a = avalia_quente("1b", Cen::C, &p, None);
            assert_eq!(a.rotulo, Rotulo::SemProva);
            assert!(a.motivos[0].contains("ALERTA DE CORREÇÃO") && a.motivos[0].contains("[4]"), "{a:?}");
        }
        // Ordem fora da alternada.
        let mut o = boas.clone();
        o[1].m1_primeiro = true;
        assert!(avalia_quente("1b", Cen::C, &o, None).motivos[0].contains("ordem"));
        // Tempo inválido.
        for ruim in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
            let mut p = boas.clone();
            p[0].m8.leitura_ms = ruim;
            assert_eq!(avalia_quente("1b", Cen::C, &p, None).rotulo, Rotulo::SemProva, "{ruim}");
        }
        // Modelo sem meta.
        assert_eq!(avalia_quente("7b", Cen::C, &boas, None).rotulo, Rotulo::SemProva);
    }

    #[test]
    fn portao_do_par_usa_logits_argmax_e_contadores() {
        let c = Contadores { needed: 10, hits: 4, cpu: 6 };
        let l = vec![0.1f32, 5.0, 0.2];
        assert!(portao_par(&l, &l, c, c).ok());
        let perto: Vec<f32> = l.iter().map(|v| v + 1e-6).collect();
        assert!(portao_par(&perto, &l, c, c).ok());
        let longe: Vec<f32> = l.iter().map(|v| v + 1e-3).collect();
        assert!(!portao_par(&longe, &l, c, c).ok());
        assert!(!portao_par(&[f32::NAN, 1.0, 2.0], &l, c, c).ok());
        let outro = Contadores { needed: 10, hits: 5, cpu: 5 };
        assert!(!portao_par(&l, &l, c, outro).ok());
        assert!(!portao_par(&l, &l, Contadores { needed: 10, hits: 10, cpu: 0 }, Contadores { needed: 10, hits: 10, cpu: 0 }).ok());
        let troca = vec![0.1f32, 4.99999, 5.0];
        let p = portao_par(&troca, &l, c, c);
        assert!(!p.argmax_ok && !p.ok());
    }

    #[test]
    fn frio_mediana_contra_a_meta_e_validade() {
        let r = |a, b, c| vec![frio(1, a), frio(2, b), frio(3, c)];
        // Mediana 19 s <= 20 s mesmo com uma repetição de 40 s (pior só descritiva).
        let a = avalia_frio("1b", Cen::C, &r(18000.0, 19000.0, 40000.0));
        assert_eq!(a.rotulo, Rotulo::Atinge, "{a:?}");
        assert_eq!(a.pior_ms, Some(40000.0));
        assert_eq!(avalia_frio("1b", Cen::C, &r(18000.0, 21000.0, 40000.0)).rotulo, Rotulo::NaoAtinge);
        assert_eq!(avalia_frio("1b", Cen::C, &r(18000.0, 21000.0, 40000.0)).motivos, vec!["meta".to_string()]);
        assert_eq!(avalia_frio("1b", Cen::C, &r(1.0, 2.0, 20000.0)).rotulo, Rotulo::Atinge, "meta exata");
        // Menos, mais, índices repetidos, outra célula.
        assert_eq!(avalia_frio("1b", Cen::C, &[frio(1, 1.0), frio(2, 1.0)]).rotulo, Rotulo::SemProva);
        assert!(avalia_frio("1b", Cen::C, &[]).motivos[0].contains("menos de 3"));
        assert_eq!(avalia_frio("1b", Cen::C, &[frio(1, 1.0), frio(2, 1.0), frio(3, 1.0), frio(3, 1.0)]).rotulo, Rotulo::SemProva);
        assert_eq!(avalia_frio("1b", Cen::C, &[frio(1, 1.0), frio(1, 1.0), frio(3, 1.0)]).rotulo, Rotulo::SemProva);
        assert_eq!(avalia_frio("1b", Cen::A, &r(1.0, 1.0, 1.0)).rotulo, Rotulo::SemProva, "registros do cenário C na célula A");
        // Falha, ok falso, integridade e tempo inválido.
        let mut v = r(1000.0, 1000.0, 1000.0);
        v[1].falha = Some("erro de motor".into());
        assert!(avalia_frio("1b", Cen::C, &v).motivos[0].contains("erro de motor"));
        let mut v = r(1000.0, 1000.0, 1000.0);
        v[2].ok = false;
        assert_eq!(avalia_frio("1b", Cen::C, &v).rotulo, Rotulo::SemProva);
        let mut v = r(1000.0, 1000.0, 1000.0);
        v[0].integro = false;
        assert_eq!(avalia_frio("1b", Cen::C, &v).rotulo, Rotulo::SemProva);
        assert_eq!(avalia_frio("1b", Cen::C, &r(1000.0, f64::NAN, 1000.0)).rotulo, Rotulo::SemProva);
        assert_eq!(avalia_frio("1b", Cen::C, &r(1000.0, 0.0, 1000.0)).rotulo, Rotulo::SemProva);
    }

    #[test]
    fn rotulo_do_conjunto_e_integridade() {
        use Rotulo::*;
        assert_eq!(rotulo_conjunto(&[Atinge, Atinge], true), Atinge);
        assert_eq!(rotulo_conjunto(&[Atinge, SemProva], true), SemProva);
        assert_eq!(rotulo_conjunto(&[SemProva, NaoAtinge], true), NaoAtinge, "NÃO ATINGE vence SEM PROVA");
        assert_eq!(rotulo_conjunto(&[NaoAtinge, Atinge], true), NaoAtinge);
        assert_eq!(rotulo_conjunto(&[], true), SemProva);
        assert_eq!(rotulo_conjunto(&[Atinge, Atinge], false), SemProva);
        assert_eq!(rotulo_conjunto(&[NaoAtinge], false), SemProva);
        assert_eq!(rotulo_final(Atinge, false), SemProva);
        assert_eq!(rotulo_final(NaoAtinge, false), SemProva);
        assert_eq!(rotulo_final(NaoAtinge, true), NaoAtinge);
        assert_eq!([Atinge.nome(), NaoAtinge.nome(), SemProva.nome()], ["ATINGE", "NÃO ATINGE", "SEM PROVA"]);
    }

    #[test]
    fn registro_quente_ida_e_volta() {
        let mut p = pares([4000.5, 4100.25, 3900.0, 4050.0, 4000.0], [2000.0, 2100.0, 1900.0, 2050.0, 2000.0]);
        p[2].portao = PortaoPar { erro: f64::INFINITY, nao_finito: true, argmax_ok: false, cont_ok: false };
        let r = RegistroQuente { modelo: "3b".into(), cen: Cen::A, integro: true, falha: Some("linha1\nlinha2".into()), pares: p.clone(), aquecimento: vec!["modo 8: 1 2 3".into()], extras: vec![("hash_bin".into(), "abc".into())] };
        let t = r.para_texto();
        assert!(t.contains("# modo 8"));
        let l = RegistroQuente::de_texto(&t).unwrap();
        assert_eq!((l.modelo.as_str(), l.cen, l.integro), ("3b", Cen::A, true));
        assert_eq!(l.falha.as_deref(), Some("linha1 linha2"));
        assert_eq!(l.pares, p);
        assert_eq!(l.extras, r.extras);
        assert!(RegistroQuente::de_texto("tipo=frio\n").is_err());
        assert!(RegistroQuente::de_texto("tipo=quente\nmodelo=1b\n").is_err());
        assert!(RegistroQuente::de_texto("tipo=quente\nmodelo=1b\ncenario=C\nintegro=1\nlixo=1\n").is_err());
    }

    #[test]
    fn registro_frio_ida_e_volta() {
        let mut r = frio(2, 12345.678);
        r.falha = Some("x".into());
        let l = RegistroFrio::de_texto(&r.para_texto()).unwrap();
        assert_eq!(l, r);
        assert!(RegistroFrio::de_texto("tipo=quente\n").is_err());
        assert!(RegistroFrio::de_texto("tipo=frio\nmodelo=1b\n").is_err());
        assert!(RegistroFrio::de_texto("tipo=frio\nmodelo=1b\ncenario=C\nrep=1\ntotal_ms=abc\nok=1\nintegro=1\n").is_err());
    }

    #[test]
    fn registro_de_outra_celula_nao_vira_prova() {
        let bom = |modelo: &str, cen: Cen| RegistroQuente { modelo: modelo.into(), cen, integro: true, falha: None, pares: pares([100.0; 5], [40.0; 5]), aquecimento: vec![], extras: vec![] };
        // Registro correto de 1b/C avaliado como 1b/C: tem rótulo de prova.
        assert_ne!(avalia_registro_quente("1b", Cen::C, &bom("1b", Cen::C)).rotulo, Rotulo::SemProva);
        // Mesmo conteúdo no nome de outro cenário ou de outro modelo: SEM PROVA, sem ler os tempos.
        for (m, c) in [("1b", Cen::A), ("3b", Cen::C), ("3b", Cen::A)] {
            let a = avalia_registro_quente(m, c, &bom("1b", Cen::C));
            assert_eq!(a.rotulo, Rotulo::SemProva, "{m} {}", c.nome());
            assert!(a.motivos.iter().any(|x| x.contains("identidade")), "{:?}", a.motivos);
        }
    }
}
