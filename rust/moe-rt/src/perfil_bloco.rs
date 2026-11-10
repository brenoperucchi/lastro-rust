//! Instrumento de fase do passo de bloco do prefill (plano `qual-runs/A2-LOTE-20261103/plano-instrumento-prefill-20261110.md`, versão
//! aprovada na rodada 136). Descritivo: cronometra por fase, não decide nada do motor.
//!
//! ## O que foi decidido na implementação (o plano deixou em aberto; detalhes em `instrumento-prefill-implementacao-20261110.md`)
//!
//! * Ativação: `LASTRO_PERFIL_BLOCO` vale só com o valor exato `1`. Ausente, vazia, `0`, `true`, ` 1` ou não UTF-8 = DESLIGADO. A leitura
//!   acontece UMA vez, em `Engine::new` (cada braço do par do controle é um processo). Desligado, `Shared::pf` é `None`: nenhum `Instant`,
//!   nenhum `Vec`, nenhum `Mutex`; o caminho quente testa um `Option`/`bool` e segue.
//! * Relógio: tudo em ns inteiros desde uma origem única do processo (`agora_ns`, `Instant` monotônico), para host e serviço comporem na
//!   mesma escala e a sobreposição poder ser medida.
//! * Host (Categoria A): cinco carimbos t0..t4 por bloco em `le_ids_com_pendente`; F0..F3 são diferenças dos MESMOS carimbos, então a
//!   conservação F0+F1+F2+F3 = t4 - t0 é exata em inteiros; a tolerância é `TOL_RELOGIO_NS` = 1 µs (cobre só carimbo não monotônico por
//!   erro de leitura). Por pedido: R_ini = t0(bloco 0) - armar, R_entre, R_fim = início de `le_logits` - t4(último), F5 = `le_logits`
//!   (de `logits_ini` a `desarma`). A identidade soma(B) + F5 + R_* = desarma - armar é exata; contra o `prefill_ms` do binário (outro
//!   `Instant`, lido em pontos vizinhos) a tolerância é max(0,2 ms, 0,5% do `prefill_ms`), como no plano.
//! * Serviço (Categoria B): sete fases, em ordem de execução na thread de serviço, que ladrilham o bloco sem lacuna além de `S_fora`:
//!   - `S_espera`: de onde a camada anterior terminou (ou do início do intervalo do bloco, na camada 0) até ver `seq[l]`; inclui a leitura
//!     de ids/pesos/`x` publicados, que `read_step` faz no mesmo laço da espera e não se separa sem mexer nele.
//!   - `S_monta`: união de experts e faltas em `serve_layer`, mais `cpu_step` e a alocação de `y` em `run` (montagem do trabalho da CPU).
//!   - `S_compute`: `compute_step` (experts na CPU).
//!   - `S_answer`: `answer` (escreve a saída e a flag).
//!   - `S_livro`: `release_held` dentro de `serve_layer`, `pin_layer`, `victim` e `copy_after_reader`.
//!   - `S_end_espera`: espera de `seq[L]` em `end_token`.
//!   - `S_end_trab`: o resto de `end_token` (soltar cópias, sync da cópia no modo com barreira, `query`, tabela, `publish`). Inclui, por
//!     isso, a espera do lado do buffer que o host devolve em `finish_token`; fica declarado, não separado.
//!   O resíduo `S_fora` = I_g - soma das fases acusa o que sobrou (laço entre camadas, `diag`, escalonamento).
//! * Intervalo do serviço: I_g vai de max(fim do bloco de serviço anterior, instante de armar) até o fim de `end_token` do bloco g. O
//!   bloco só é registrado se estava armado quando a thread viu `seq[0]` (lido com Acquire DEPOIS de ver `seq[0]`: o host armou antes de
//!   enfileirar). Como o decode é desarmado antes do primeiro passo de decode, nenhum bloco de decode é registrado; com o instrumento
//!   ligado a thread ainda carimba o decode (acumuladores locais, sem alocar) e descarta; o custo disso é medido pelo `passo_ms` do par.
//! * Geração: a chave de junção é o `gen` do `Published`/`TokenEpoch` (o token do bloco), igual no host e no serviço. A geração g do plano
//!   é o índice do bloco no pedido (o merge ordena pelos blocos do host).
//! * Mecanismo de armamento: um `AtomicBool` em `Compartilhado` (Release ao armar/desarmar, Acquire no serviço), mais o instante de armar.
//!   Sem canal novo entre as linhas e sem tocar no protocolo de sincronização do motor. O serviço empurra um `RecServico` por bloco armado
//!   (um `Mutex<Vec>` com capacidade reservada no armar); o merge acontece em `Armado::fecha`, depois do decode, fora de qualquer
//!   cronômetro, esperando até `ESPERA_REGISTROS_MS` pelos registros do serviço; o que não chegou marca o pedido como "serviço aberto".
//! * Escopo: `armar` logo depois de `t_pre`, `desarma` logo depois de `le_logits`. O decode fica fora. Uma entrada por pedido, em
//!   `MotorGeracao::perfil`, gravada pelo binário em `perfil-bloco.json`.
use crate::e4b_regra::mediana;
use crate::motor::leitura::plano_blocos;
use serde_json::{json, Value};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// Nome da variável de ambiente que liga o instrumento.
pub const VARIAVEL: &str = "LASTRO_PERFIL_BLOCO";
/// Tolerância de relógio dos carimbos (ns): serialização e leitura, nada além.
pub const TOL_RELOGIO_NS: u64 = 1_000;
/// Tolerância da soma do pedido contra o `prefill_ms` do binário: o maior entre os dois.
pub const TOL_PEDIDO_MS: f64 = 0.2;
pub const TOL_PEDIDO_FRAC: f64 = 0.005;
/// Cobertura: resíduo do host / `prefill_ms` e S_fora / I_g.
pub const LIM_COBERTURA_HOST: f64 = 0.03;
pub const LIM_COBERTURA_SERVICO: f64 = 0.10;
/// Quanto o merge espera pelo registro do serviço do último bloco, no máximo.
pub const ESPERA_REGISTROS_MS: u64 = 500;
/// Tamanho do bloco do plano (BLOCO = 8 fixo).
pub const BLOCO: usize = 8;

// ---- Ativação -----------------------------------------------------------------------------------------------------------------------------

/// A regra de leitura da variável: só o valor exato `1` liga.
pub fn interpreta(v: Option<&str>) -> bool {
    v == Some("1")
}

/// Lê a variável do ambiente (não UTF-8 conta como ausente).
pub fn ligado_do_ambiente() -> bool {
    interpreta(std::env::var(VARIAVEL).ok().as_deref())
}

/// A linha que o binário imprime, com o estado lido.
pub fn linha_estado(ligado: bool) -> &'static str {
    if ligado {
        "perfil-bloco: ligado"
    } else {
        "perfil-bloco: desligado"
    }
}

static ORIGEM: OnceLock<Instant> = OnceLock::new();

/// Ns desde a origem comum do processo (monotônico).
pub fn agora_ns() -> u64 {
    ORIGEM.get_or_init(Instant::now).elapsed().as_nanos() as u64
}

fn ms(ns: u64) -> f64 {
    ns as f64 / 1e6
}

fn trava<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

// ---- Registros ----------------------------------------------------------------------------------------------------------------------------

/// Um bloco do prefill vista pelo host: carimbos t0..t4 (ns) e a marca de `Published` pendente (F0 zero por construção).
#[derive(Clone, Debug, PartialEq)]
pub struct RecBloco {
    pub gen: u32,
    pub m: usize,
    pub pos_ini: u64,
    pub t: [u64; 5],
    pub f0_pendente: bool,
}

impl RecBloco {
    /// F0..F3 (`i` de 0 a 3): diferença de dois carimbos consecutivos.
    pub fn f(&self, i: usize) -> u64 {
        self.t[i + 1].saturating_sub(self.t[i])
    }
    pub fn b(&self) -> u64 {
        self.t[4].saturating_sub(self.t[0])
    }
    pub fn monotono(&self) -> bool {
        self.t.windows(2).all(|w| w[0] <= w[1])
    }
    /// Conservação da partição: carimbos monotônicos e F0+F1+F2+F3 = B dentro de `TOL_RELOGIO_NS`.
    pub fn conserva(&self) -> bool {
        let s: u64 = (0..4).map(|i| self.f(i)).sum();
        self.monotono() && s.abs_diff(self.b()) <= TOL_RELOGIO_NS
    }
}

pub const S_ESPERA: usize = 0;
pub const S_MONTA: usize = 1;
pub const S_COMPUTE: usize = 2;
pub const S_ANSWER: usize = 3;
pub const S_LIVRO: usize = 4;
pub const S_END_ESPERA: usize = 5;
pub const S_END_TRAB: usize = 6;
pub const NOMES_S: [&str; 7] = ["s_espera_ms", "s_monta_ms", "s_compute_ms", "s_answer_ms", "s_livro_ms", "s_end_espera_ms", "s_end_trab_ms"];

/// O serviço de um bloco (geração `gen`): intervalo I_g e as sete fases somadas sobre as camadas (ns).
#[derive(Clone, Debug, PartialEq)]
pub struct RecServico {
    pub gen: u32,
    pub ini: u64,
    pub fim: u64,
    pub f: [u64; 7],
}

impl RecServico {
    pub fn i_g(&self) -> u64 {
        self.fim.saturating_sub(self.ini)
    }
    pub fn soma(&self) -> u64 {
        self.f.iter().sum()
    }
    pub fn s_fora(&self) -> u64 {
        self.i_g().saturating_sub(self.soma())
    }
    /// Fim depois do início e soma das fases nomeadas dentro de I_g (as fases do mesmo bloco não se sobrepõem: thread única).
    pub fn coerente(&self) -> bool {
        self.fim >= self.ini && self.soma() <= self.i_g() + TOL_RELOGIO_NS
    }
    /// Quanto de I_g cai dentro de [t0, t4] do bloco do host (descritivo; sem somar duas vezes).
    pub fn inter_ns(&self, b: &RecBloco) -> u64 {
        self.fim.min(b.t[4]).saturating_sub(self.ini.max(b.t[0]))
    }
}

/// I_g de blocos consecutivos (por geração crescente) não se sobrepõem.
pub fn servicos_sem_sobreposicao(v: &[&RecServico]) -> bool {
    v.windows(2).all(|w| w[1].ini + TOL_RELOGIO_NS >= w[0].fim)
}

/// Acumuladores da thread de serviço para o bloco em andamento. Escritora única: a thread de serviço. Sem alocação.
#[derive(Default, Debug)]
pub struct AccServ {
    pub armado: bool,
    pub ini: u64,
    pub prev_fim: u64,
    pub cursor: u64,
    pub f: [u64; 7],
}

impl AccServ {
    /// Depois de ver `seq[l]` no instante `tb`. Na camada 0 abre o bloco: lê o armamento e fixa I_g.ini = max(fim anterior, armar).
    pub fn apos_espera(&mut self, l: usize, tb: u64, armado: bool, t_armado: u64) {
        if l == 0 {
            self.f = [0; 7];
            self.armado = armado;
            self.ini = self.prev_fim.max(t_armado);
            self.cursor = self.ini;
        }
        self.fase(S_ESPERA, tb);
    }
    /// Fecha a fase `i` em `agora`: soma o que passou desde o cursor e avança o cursor (nunca recua).
    pub fn fase(&mut self, i: usize, agora: u64) {
        self.f[i] += agora.saturating_sub(self.cursor);
        self.cursor = self.cursor.max(agora);
    }
    /// Fim do bloco no cursor: devolve o registro se o bloco nasceu armado e zera para o próximo.
    pub fn fecha(&mut self, gen: u32) -> Option<RecServico> {
        let fim = self.cursor;
        let r = self.armado.then(|| RecServico { gen, ini: self.ini, fim, f: self.f });
        self.prev_fim = fim;
        self.f = [0; 7];
        self.armado = false;
        r
    }
}

// ---- Estado compartilhado e guarda de armamento -------------------------------------------------------------------------------------------

/// O que as duas linhas compartilham quando o instrumento está ligado (`Shared::pf`).
pub struct Compartilhado {
    armado: AtomicBool,
    t_armado: AtomicU64,
    blocos: Mutex<Vec<RecBloco>>,
    servico: Mutex<Vec<RecServico>>,
}

impl Compartilhado {
    pub fn novo() -> Self {
        let _ = agora_ns(); // fixa a origem antes de qualquer carimbo
        Self { armado: AtomicBool::new(false), t_armado: AtomicU64::new(0), blocos: Mutex::new(vec![]), servico: Mutex::new(vec![]) }
    }
    /// Leitura do host (a mesma thread que arma): relaxada.
    pub fn ativo(&self) -> bool {
        self.armado.load(Ordering::Relaxed)
    }
    /// Leitura do serviço, depois de ver `seq[0]`.
    pub fn armado_acq(&self) -> bool {
        self.armado.load(Ordering::Acquire)
    }
    pub fn t_armado(&self) -> u64 {
        self.t_armado.load(Ordering::Acquire)
    }
    pub fn registra_bloco(&self, r: RecBloco) {
        trava(&self.blocos).push(r);
    }
    pub fn registra_servico(&self, r: RecServico) {
        trava(&self.servico).push(r);
    }
}

/// Guarda do armamento de um prefill. Inerte (sem alocar) quando o instrumento não existe; ao cair, desarma e descarta o que sobrou.
pub struct Armado {
    pf: Option<Arc<Compartilhado>>,
    armado: bool,
    t_armado: u64,
    t_logits_ini: u64,
    t_desarmado: u64,
}

/// O que `Armado::fecha` precisa saber do pedido, e o binário já tem.
pub struct DadosPedido {
    pub indice: usize,
    pub aquecimento: bool,
    pub n: usize,
    pub plano: Vec<usize>,
    pub prefill_ms: f64,
    pub decode_passo_ms_mediana: f64,
}

impl Armado {
    pub fn inerte() -> Self {
        Self { pf: None, armado: false, t_armado: 0, t_logits_ini: 0, t_desarmado: 0 }
    }
    pub fn arma(pf: Option<Arc<Compartilhado>>, blocos_previstos: usize) -> Self {
        let Some(p) = pf else { return Self::inerte() };
        {
            let mut b = trava(&p.blocos);
            b.clear();
            b.reserve(blocos_previstos);
        }
        {
            let mut s = trava(&p.servico);
            s.clear();
            s.reserve(blocos_previstos);
        }
        let t = agora_ns();
        p.t_armado.store(t, Ordering::Release);
        p.armado.store(true, Ordering::Release);
        Self { pf: Some(p), armado: true, t_armado: t, t_logits_ini: 0, t_desarmado: 0 }
    }
    pub fn ativo(&self) -> bool {
        self.pf.is_some()
    }
    /// Marca o início de `le_logits` (fim do último bloco, começo de F5).
    pub fn logits_ini(&mut self) {
        if self.armado {
            self.t_logits_ini = agora_ns();
        }
    }
    /// Desarma (idempotente): o decode que vem a seguir não registra nada.
    pub fn desarma(&mut self) {
        if self.armado {
            self.t_desarmado = agora_ns();
            if let Some(p) = &self.pf {
                p.armado.store(false, Ordering::Release);
            }
            self.armado = false;
        }
    }
    /// O merge entre host e serviço, depois do decode. `None` com o instrumento ausente.
    pub fn fecha(self, d: DadosPedido) -> Option<RecPedido> {
        self.fecha_com(d, &mut || std::thread::sleep(Duration::from_millis(1)))
    }

    /// Como [`fecha`](Self::fecha), com o que fazer a cada volta da espera pelos registros do serviço (os testes com escalonador
    /// determinístico passam um `relax` do backend falso, que é quem dá a vez à linha de serviço).
    pub fn fecha_com(mut self, d: DadosPedido, espera: &mut dyn FnMut()) -> Option<RecPedido> {
        let pf = self.pf.clone()?;
        self.desarma();
        let blocos = std::mem::take(&mut *trava(&pf.blocos));
        let limite = Instant::now() + Duration::from_millis(ESPERA_REGISTROS_MS);
        while trava(&pf.servico).len() < blocos.len() && Instant::now() < limite {
            espera();
        }
        let mut sv = std::mem::take(&mut *trava(&pf.servico));
        let servico = blocos.iter().map(|b| sv.iter().position(|s| s.gen == b.gen).map(|i| sv.swap_remove(i))).collect();
        self.pf = None;
        let servico_extra = sv.len();
        Some(RecPedido {
            indice: d.indice,
            aquecimento: d.aquecimento,
            n: d.n,
            plano: d.plano,
            blocos,
            servico,
            servico_extra,
            t_armado: self.t_armado,
            t_logits_ini: self.t_logits_ini,
            t_desarmado: self.t_desarmado,
            prefill_ms: d.prefill_ms,
            decode_passo_ms_mediana: d.decode_passo_ms_mediana,
        })
    }
}

impl Drop for Armado {
    fn drop(&mut self) {
        if let Some(p) = &self.pf {
            p.armado.store(false, Ordering::Release);
            trava(&p.blocos).clear();
            trava(&p.servico).clear();
        }
    }
}

// ---- A entrada de um pedido ---------------------------------------------------------------------------------------------------------------

/// Uma entrada de `perfil-bloco.json`: tudo de um pedido (prefill apenas; o decode só entra pela mediana do `passo_ms`).
#[derive(Clone, Debug, PartialEq)]
pub struct RecPedido {
    pub indice: usize,
    pub aquecimento: bool,
    pub n: usize,
    pub plano: Vec<usize>,
    pub blocos: Vec<RecBloco>,
    /// Um por bloco, na mesma ordem; `None` = o serviço daquele bloco não tinha fechado no merge.
    pub servico: Vec<Option<RecServico>>,
    /// Registros do serviço sem bloco do host correspondente (esperado 0; > 0 acusa registro de decode ou de outro pedido).
    pub servico_extra: usize,
    pub t_armado: u64,
    pub t_logits_ini: u64,
    pub t_desarmado: u64,
    pub prefill_ms: f64,
    pub decode_passo_ms_mediana: f64,
}

impl RecPedido {
    pub fn f5_ns(&self) -> u64 {
        self.t_desarmado.saturating_sub(self.t_logits_ini)
    }
    pub fn r_ini_ns(&self) -> u64 {
        self.blocos.first().map_or(0, |b| b.t[0].saturating_sub(self.t_armado))
    }
    pub fn r_entre_ns(&self) -> Vec<u64> {
        self.blocos.windows(2).map(|w| w[1].t[0].saturating_sub(w[0].t[4])).collect()
    }
    pub fn r_fim_ns(&self) -> u64 {
        self.blocos.last().map_or(0, |b| self.t_logits_ini.saturating_sub(b.t[4]))
    }
    pub fn residuo_host_ns(&self) -> u64 {
        self.r_ini_ns() + self.r_entre_ns().iter().sum::<u64>() + self.r_fim_ns()
    }
    pub fn soma_b_ns(&self) -> u64 {
        self.blocos.iter().map(RecBloco::b).sum()
    }
    /// armar..desarmar, medido pelos carimbos do perfil.
    pub fn total_perfil_ns(&self) -> u64 {
        self.t_desarmado.saturating_sub(self.t_armado)
    }
    /// soma(B) + F5 + resíduos = armar..desarmar (exata por construção; falha só com carimbos fora de ordem).
    pub fn identidade_host(&self) -> bool {
        let s = self.soma_b_ns() + self.f5_ns() + self.residuo_host_ns();
        s.abs_diff(self.total_perfil_ns()) <= TOL_RELOGIO_NS * (self.blocos.len() as u64 + 2)
    }
    /// armar..desarmar contra o `prefill_ms` do binário.
    pub fn bate_com_prefill(&self) -> bool {
        let tol = TOL_PEDIDO_MS.max(TOL_PEDIDO_FRAC * self.prefill_ms);
        (ms(self.total_perfil_ns()) - self.prefill_ms).abs() <= tol
    }
    pub fn cobertura_host(&self) -> f64 {
        ms(self.residuo_host_ns()) / self.prefill_ms
    }
    pub fn servico_aberto(&self) -> bool {
        self.servico.iter().any(Option::is_none)
    }
    /// Frações S_fora / I_g dos blocos com serviço fechado.
    pub fn s_fora_fracoes(&self) -> Vec<f64> {
        self.servico.iter().flatten().filter(|s| s.i_g() > 0).map(|s| s.s_fora() as f64 / s.i_g() as f64).collect()
    }
    /// Os problemas de validade do pedido (vazio = válido). Cobertura NÃO entra aqui: é ressalva, não invalidade.
    pub fn problemas(&self) -> Vec<String> {
        let mut p = vec![];
        let ms_: Vec<usize> = self.blocos.iter().map(|b| b.m).collect();
        if ms_ != self.plano || self.plano != plano_blocos(self.n, BLOCO) {
            p.push(format!("blocos {ms_:?} diferem de plano_blocos({}, {BLOCO}) = {:?}", self.n, plano_blocos(self.n, BLOCO)));
        }
        for (i, b) in self.blocos.iter().enumerate() {
            if !b.conserva() {
                p.push(format!("bloco {i}: carimbos fora de ordem ou F0+F1+F2+F3 diferente de B"));
            }
        }
        if self.blocos.windows(2).any(|w| w[1].t[0] < w[0].t[4]) {
            p.push("blocos do host se sobrepõem".into());
        }
        if !self.identidade_host() {
            p.push("soma(B) + F5 + resíduos diferente de armar..desarmar".into());
        }
        if !self.bate_com_prefill() {
            p.push(format!("armar..desarmar {:.3} ms contra prefill_ms {:.3} ms fora da tolerância", ms(self.total_perfil_ns()), self.prefill_ms));
        }
        if self.servico_extra > 0 {
            p.push(format!("{} registro(s) do serviço sem bloco do host (decode ou outro pedido)", self.servico_extra));
        }
        let fechados: Vec<&RecServico> = self.servico.iter().flatten().collect();
        for s in &fechados {
            if !s.coerente() {
                p.push(format!("serviço gen {}: fim antes do início ou fases nomeadas somam mais que I_g", s.gen));
            }
        }
        if !servicos_sem_sobreposicao(&fechados) {
            p.push("I_g de blocos consecutivos do serviço se sobrepõem".into());
        }
        p
    }
    /// R6 = B(bloco final parcial) / B(último bloco cheio anterior); `None` sem bloco final parcial ou sem bloco cheio antes dele.
    pub fn r6(&self) -> Option<f64> {
        let (ult, ant) = (self.blocos.last()?, self.blocos.iter().rev().nth(1)?);
        (ult.m < BLOCO && ant.m == BLOCO && ant.b() > 0).then(|| ult.b() as f64 / ant.b() as f64)
    }

    pub fn para_json(&self) -> Value {
        let blocos: Vec<Value> = self
            .blocos
            .iter()
            .enumerate()
            .map(|(i, b)| {
                json!({
                    "indice": i, "gen": b.gen, "m": b.m, "pos_ini": b.pos_ini, "f0_pendente": b.f0_pendente,
                    "f0_ms": ms(b.f(0)), "f1_ms": ms(b.f(1)), "f2_ms": ms(b.f(2)), "f3_ms": ms(b.f(3)), "b_ms": ms(b.b()),
                    "t_ns": b.t, "conserva": b.conserva(),
                })
            })
            .collect();
        let servico: Vec<Value> = self
            .blocos
            .iter()
            .zip(&self.servico)
            .enumerate()
            .map(|(i, (b, s))| match s {
                None => json!({"indice": i, "gen": b.gen, "aberto": true}),
                Some(s) => {
                    let mut v = json!({
                        "indice": i, "gen": s.gen, "aberto": false, "ini_ns": s.ini, "fim_ns": s.fim, "i_ms": ms(s.i_g()),
                        "s_fora_ms": ms(s.s_fora()), "s_fora_frac": if s.i_g() > 0 { s.s_fora() as f64 / s.i_g() as f64 } else { 0.0 },
                        "inter_bloco_ms": ms(s.inter_ns(b)), "coerente": s.coerente(),
                    });
                    for (k, nome) in NOMES_S.iter().enumerate() {
                        v[*nome] = json!(ms(s.f[k]));
                    }
                    v
                }
            })
            .collect();
        let problemas = self.problemas();
        json!({
            "indice": self.indice, "aquecimento": self.aquecimento, "n": self.n, "plano_m": self.plano,
            "prefill_ms": self.prefill_ms, "decode_passo_ms_mediana": self.decode_passo_ms_mediana,
            "blocos": blocos, "servico": servico,
            "f5_ms": ms(self.f5_ns()), "r_ini_ms": ms(self.r_ini_ns()),
            "r_entre_ms": self.r_entre_ns().into_iter().map(ms).collect::<Vec<_>>(), "r_fim_ms": ms(self.r_fim_ns()),
            "residuo_host_ms": ms(self.residuo_host_ns()), "cobertura_host_frac": self.cobertura_host(),
            "cobertura_host_ressalva": self.cobertura_host() > LIM_COBERTURA_HOST,
            "armar_ate_desarmar_ms": ms(self.total_perfil_ns()), "r6": self.r6(),
            "servico_aberto": self.servico_aberto(), "servico_extra": self.servico_extra, "s_fora_frac_mediana": mediana_opt(&self.s_fora_fracoes()),
            "valido": problemas.is_empty(), "problemas": problemas,
        })
    }
}

fn mediana_opt(v: &[f64]) -> Option<f64> {
    (!v.is_empty()).then(|| mediana(v))
}

/// O arquivo `perfil-bloco.json`.
pub fn perfil_json(pedidos: &[RecPedido]) -> Value {
    json!({
        "tipo": "perfil-bloco",
        "versao": 1,
        "variavel": VARIAVEL,
        "bloco": BLOCO,
        "unidades": "ms nos campos *_ms; carimbos *_ns em ns desde a origem do processo",
        "tolerancias": {"relogio_ns": TOL_RELOGIO_NS, "pedido_ms": TOL_PEDIDO_MS, "pedido_frac": TOL_PEDIDO_FRAC,
                        "cobertura_host": LIM_COBERTURA_HOST, "cobertura_servico": LIM_COBERTURA_SERVICO},
        "pedidos": pedidos.iter().map(RecPedido::para_json).collect::<Vec<_>>(),
    })
}

// ---- H6 -----------------------------------------------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leitura {
    Dentro,
    Fora,
    SemLeitura,
}

/// Faixas de H6 para o resto `r` (1..=7): (custo fixo se R6 >= a, custo por posição se R6 <= b).
pub fn faixas_h6(r: usize) -> (f64, f64) {
    (0.5 + r as f64 / 16.0, (r as f64 / 8.0) * 1.25)
}

/// As faixas se tocam: o texto é só descritivo para H6.
pub fn h6_descritivo(r: usize) -> bool {
    let (a, b) = faixas_h6(r);
    a <= b
}

pub fn classifica_h6(r: usize, r6: f64) -> Option<Leitura> {
    if h6_descritivo(r) {
        return None;
    }
    let (a, b) = faixas_h6(r);
    Some(if r6 >= a {
        Leitura::Dentro
    } else if r6 <= b {
        Leitura::Fora
    } else {
        Leitura::SemLeitura
    })
}

// ---- Controle do custo do instrumento (2.1) -----------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Braco {
    D,
    L,
}

/// Um processo do par de controle, na ordem em que rodou.
#[derive(Clone, Debug)]
pub struct ProcessoControle {
    pub braco: Braco,
    /// Mediana de `prefill_ms` dos pedidos medidos do processo.
    pub prefill_ms_mediana: f64,
    /// A linha `perfil-bloco: ligado` apareceu.
    pub imprimiu_ligado: bool,
    /// O processo produziu `perfil-bloco.json`.
    pub produziu_perfil: bool,
    /// Terminou sem aborto e com os ids iguais aos da referência.
    pub valido: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ClasseControle {
    /// |desvio| <= 3%.
    Comprovada,
    /// 3% < |desvio| <= 10%.
    ComRessalva,
    /// |desvio| > 10%.
    SemProva,
    /// Par incompleto, fora de ordem, processo abortado, D que produziu perfil, L sem perfil ou orçamento insuficiente.
    Incompleto(String),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResultadoControle {
    pub classe: ClasseControle,
    /// (mediana das medianas de L - a de D) / a de D; com sinal, para a análise descritiva.
    pub desvio: Option<f64>,
    /// (maior mediana de L - menor de D) / menor de D.
    pub pior_caso: Option<f64>,
}

pub const ORDEM_CONTROLE: [Braco; 6] = [Braco::D, Braco::L, Braco::L, Braco::D, Braco::D, Braco::L];
const EPS: f64 = 1e-12;

/// Classifica o par de controle de uma célula (modelo, série). Pura. A magnitude do desvio manda; o sinal não aprova nada.
pub fn classifica_controle(procs: &[ProcessoControle], orcamento_ok: bool) -> ResultadoControle {
    let inc = |m: &str| ResultadoControle { classe: ClasseControle::Incompleto(m.into()), desvio: None, pior_caso: None };
    if !orcamento_ok {
        return inc("orçamento insuficiente");
    }
    if procs.len() != ORDEM_CONTROLE.len() {
        return inc("menos (ou mais) de 6 processos: mínimo de 3 válidos por braço");
    }
    if procs.iter().map(|p| p.braco).ne(ORDEM_CONTROLE) {
        return inc("ordem diferente de D L L D D L");
    }
    if procs.iter().any(|p| !p.valido || !p.prefill_ms_mediana.is_finite() || p.prefill_ms_mediana <= 0.0) {
        return inc("processo de controle abortado ou inválido");
    }
    if procs.iter().any(|p| p.braco == Braco::D && (p.produziu_perfil || p.imprimiu_ligado)) {
        return inc("braço D produziu perfil-bloco.json ou imprimiu perfil-bloco: ligado");
    }
    if procs.iter().any(|p| p.braco == Braco::L && (!p.produziu_perfil || !p.imprimiu_ligado)) {
        return inc("braço L sem perfil-bloco.json ou sem a linha ligado");
    }
    let de = |b: Braco| -> Vec<f64> { procs.iter().filter(|p| p.braco == b).map(|p| p.prefill_ms_mediana).collect() };
    let (d, l) = (de(Braco::D), de(Braco::L));
    let (md, ml) = (mediana(&d), mediana(&l));
    let desvio = (ml - md) / md;
    let min_d = d.iter().copied().fold(f64::INFINITY, f64::min);
    let max_l = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let classe = match desvio.abs() {
        x if x <= 0.03 + EPS => ClasseControle::Comprovada,
        x if x <= 0.10 + EPS => ClasseControle::ComRessalva,
        _ => ClasseControle::SemProva,
    };
    ResultadoControle { classe, desvio: Some(desvio), pior_caso: Some((max_l - min_d) / min_d) }
}

// ---- Testes sem GPU -----------------------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn bloco(gen: u32, m: usize, t0: u64, f: [f64; 4]) -> RecBloco {
        let mut t = [t0; 5];
        for i in 0..4 {
            t[i + 1] = t[i] + (f[i] * MS as f64) as u64;
        }
        RecBloco { gen, m, pos_ini: 0, t, f0_pendente: false }
    }

    fn serv(gen: u32, ini: u64, fim: u64, f: [u64; 7]) -> RecServico {
        RecServico { gen, ini, fim, f }
    }

    #[test]
    fn variavel_so_liga_com_o_valor_um() {
        assert!(interpreta(Some("1")));
        for v in [None, Some(""), Some("0"), Some("true"), Some(" 1"), Some("1 "), Some("11"), Some("on"), Some("sim")] {
            assert!(!interpreta(v), "{v:?}");
        }
        assert_eq!(linha_estado(true), "perfil-bloco: ligado");
        assert_eq!(linha_estado(false), "perfil-bloco: desligado");
    }

    #[test]
    fn desligado_nao_aloca_nem_grava_nada() {
        let a = Armado::arma(None, 1000);
        assert!(!a.ativo() && a.pf.is_none());
        let mut a = a;
        a.logits_ini();
        a.desarma();
        assert_eq!((a.t_armado, a.t_logits_ini, a.t_desarmado), (0, 0, 0));
        assert!(a.fecha(DadosPedido { indice: 0, aquecimento: false, n: 1, plano: vec![1], prefill_ms: 1.0, decode_passo_ms_mediana: 1.0 }).is_none());
    }

    #[test]
    fn armar_desarmar_e_guarda_que_descarta() {
        let pf = Arc::new(Compartilhado::novo());
        assert!(!pf.ativo());
        let a = Armado::arma(Some(pf.clone()), 3);
        assert!(pf.ativo() && pf.armado_acq() && a.ativo());
        pf.registra_bloco(bloco(1, 8, 0, [0.0; 4]));
        drop(a); // erro no meio do prefill: desarma e descarta
        assert!(!pf.ativo() && trava(&pf.blocos).is_empty());
    }

    #[test]
    fn conservacao_do_host_por_bloco_e_por_pedido() {
        let b = bloco(1, 8, 100, [0.1, 0.2, 4.0, 5.7]);
        assert_eq!(b.b(), 10 * MS);
        assert!(b.conserva());
        let mut ruim = b.clone();
        ruim.t[2] = ruim.t[1] - 1; // carimbo fora de ordem
        assert!(!ruim.conserva());
        // pedido: dois blocos (8 + 1), com resíduos
        let b0 = bloco(1, 8, 1 * MS, [0.1, 0.2, 4.0, 5.7]); // 1..11 ms
        let b1 = bloco(2, 1, 11 * MS + 50_000, [0.0, 0.1, 0.5, 1.4]); // resíduo entre = 0,05 ms
        let t_logits = b1.t[4] + 100_000; // R_fim 0,1 ms
        let p = RecPedido {
            indice: 0, aquecimento: false, n: 9, plano: vec![8, 1], blocos: vec![b0, b1], servico: vec![None, None], servico_extra: 0,
            t_armado: 0, t_logits_ini: t_logits, t_desarmado: t_logits + 2 * MS, prefill_ms: 0.0, decode_passo_ms_mediana: 1.0,
        };
        let mut p = p;
        p.prefill_ms = ms(p.total_perfil_ns()) + 0.05; // diferença de relógio dentro da tolerância
        assert_eq!(p.r_ini_ns(), MS);
        assert_eq!(p.r_entre_ns(), vec![50_000]);
        assert_eq!(p.r_fim_ns(), 100_000);
        assert_eq!(p.f5_ns(), 2 * MS);
        assert!(p.identidade_host() && p.bate_com_prefill());
        assert!(p.servico_aberto(), "sem registro de serviço o pedido é marcado aberto");
        assert!(p.problemas().is_empty(), "{:?}", p.problemas());
        // cobertura do host com poder de falhar: resíduo de ~1,15 ms em ~14 ms passa de 3%
        assert!(p.cobertura_host() > LIM_COBERTURA_HOST);
        assert!(p.para_json()["cobertura_host_ressalva"].as_bool().unwrap());
        // prefill_ms do binário muito diferente do armar..desarmar invalida
        p.prefill_ms += 5.0;
        assert!(!p.bate_com_prefill() && !p.problemas().is_empty());
    }

    #[test]
    fn cobertura_do_host_abaixo_do_limite_nao_tem_ressalva() {
        let b0 = bloco(1, 8, 10_000, [0.1, 0.2, 4.0, 5.7]);
        let t_logits = b0.t[4] + 10_000;
        let p = RecPedido {
            indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b0], servico: vec![None], servico_extra: 0,
            t_armado: 0, t_logits_ini: t_logits, t_desarmado: t_logits + MS, prefill_ms: ms(t_logits + MS), decode_passo_ms_mediana: 1.0,
        };
        assert!(p.cobertura_host() < LIM_COBERTURA_HOST);
    }

    #[test]
    fn servico_totalmente_sobreposto_fecha_o_host_e_soma_mais_que_o_bloco_sem_invalidar() {
        // bloco de 10 ms: F0=0,1, F1=0,2, F2=4, F3=5,7; serviço de 7 ms inteiro dentro de [t0, t4]
        let b = bloco(1, 8, 20 * MS, [0.1, 0.2, 4.0, 5.7]);
        let ini = b.t[0] + 2 * MS;
        let s = serv(1, ini, ini + 7 * MS, [MS, MS, 3 * MS, 0, MS, 0, MS]);
        assert!(s.coerente() && b.conserva());
        assert_eq!(s.inter_ns(&b), 7 * MS);
        let host: u64 = (0..4).map(|i| b.f(i)).sum();
        assert_eq!(host, 10 * MS);
        assert_eq!(host + s.i_g(), 17 * MS, "somar as categorias passa de B e não é erro");
        assert_eq!(s.s_fora(), 0);
    }

    #[test]
    fn servico_parcialmente_sobreposto_e_publicacao_depois_do_finish_token() {
        let b = bloco(1, 8, 20 * MS, [0.1, 0.2, 4.0, 5.7]);
        // começa durante F0 (antes de t0) e termina durante F3
        let s = serv(1, b.t[0] - MS, b.t[3] + 2 * MS, [MS, 2 * MS, 2 * MS, 0, 0, MS, MS]);
        assert!(s.coerente());
        assert_eq!(s.inter_ns(&b), s.fim - b.t[0]);
        // publicação depois do finish_token: o fim do serviço (g) passa de t4 do host (g); o F0 do bloco g+1 o absorve
        let s2 = serv(1, b.t[0] + MS, b.t[4] + 3 * MS, [MS, MS, 3 * MS, 0, MS, 0, 4 * MS]);
        assert!(s2.coerente() && s2.fim > b.t[4]);
        assert_eq!(s2.inter_ns(&b), b.t[4] - s2.ini, "o recorte é só a interseção com [t0, t4]");
        let b2 = bloco(2, 8, b.t[4] + MS, [2.0, 0.2, 4.0, 5.7]); // F0(g+1) espera a publicação
        assert!(b2.t[1] >= s2.fim, "o Published só sai depois do fim do serviço de g");
        assert!(b.conserva() && b2.conserva(), "nenhuma dupla contagem na Categoria A");
        let s3 = serv(2, s2.fim, s2.fim + 5 * MS, [0, 0, 0, 0, 0, 0, 5 * MS]);
        assert!(servicos_sem_sobreposicao(&[&s2, &s3]));
    }

    #[test]
    fn carimbos_incoerentes_do_servico_sao_rejeitados() {
        assert!(!serv(1, 10 * MS, 5 * MS, [0; 7]).coerente(), "fim antes do início");
        assert!(!serv(1, 0, 4 * MS, [3 * MS, 3 * MS, 0, 0, 0, 0, 0]).coerente(), "fases nomeadas sobrepostas somam mais que I_g");
        let (a, b) = (serv(1, 0, 10 * MS, [0; 7]), serv(2, 9 * MS, 20 * MS, [0; 7]));
        assert!(!servicos_sem_sobreposicao(&[&a, &b]), "I_g consecutivos sobrepostos");
        let b0 = bloco(1, 8, 0, [0.1, 0.1, 0.1, 0.1]);
        let p = RecPedido {
            indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b0.clone()], servico: vec![Some(serv(1, 10 * MS, 5 * MS, [0; 7]))], servico_extra: 0,
            t_armado: 0, t_logits_ini: b0.t[4], t_desarmado: b0.t[4], prefill_ms: ms(b0.t[4]), decode_passo_ms_mediana: 1.0,
        };
        assert!(p.problemas().iter().any(|x| x.contains("serviço gen 1")));
    }

    #[test]
    fn cobertura_do_servico_acusa_s_fora_de_15_por_cento() {
        let s = serv(1, 0, 100 * MS, [10 * MS, 20 * MS, 20 * MS, 5 * MS, 10 * MS, 10 * MS, 10 * MS]);
        assert_eq!(s.soma(), 85 * MS);
        assert!((s.s_fora() as f64 / s.i_g() as f64 - 0.15).abs() < 1e-12);
        let b = bloco(1, 8, 0, [0.1, 0.1, 0.1, 0.1]);
        let p = RecPedido {
            indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b.clone()], servico: vec![Some(s)], servico_extra: 0,
            t_armado: 0, t_logits_ini: b.t[4], t_desarmado: b.t[4], prefill_ms: ms(b.t[4]), decode_passo_ms_mediana: 1.0,
        };
        assert!(mediana(&p.s_fora_fracoes()) > LIM_COBERTURA_SERVICO);
    }

    #[test]
    fn acumulador_do_servico_ladrilha_o_bloco_e_corta_a_camada_zero_no_armar() {
        let mut a = AccServ::default();
        a.prev_fim = 1_000;
        // camada 0: o serviço já esperava desde 1_000, o prefill foi armado em 5_000, seq[0] visto em 9_000
        a.apos_espera(0, 9_000, true, 5_000);
        assert_eq!((a.ini, a.f[S_ESPERA]), (5_000, 4_000), "a espera conta só de max(fim anterior, armar) em diante");
        a.fase(S_MONTA, 9_500);
        a.fase(S_COMPUTE, 12_000);
        a.fase(S_ANSWER, 12_100);
        a.fase(S_LIVRO, 12_600);
        a.apos_espera(1, 14_000, true, 5_000);
        a.fase(S_COMPUTE, 15_000);
        a.fase(S_END_ESPERA, 16_000);
        a.fase(S_END_TRAB, 17_000);
        let r = a.fecha(7).expect("armado");
        assert_eq!((r.gen, r.ini, r.fim), (7, 5_000, 17_000));
        assert_eq!(r.soma(), r.i_g(), "sem lacuna entre as fases carimbadas");
        assert!(r.coerente());
        // próximo bloco: I_g começa no fim do anterior; bloco não armado não gera registro
        a.apos_espera(0, 18_000, false, 5_000);
        assert_eq!(a.ini, 17_000);
        a.fase(S_END_TRAB, 19_000);
        assert!(a.fecha(8).is_none());
        assert_eq!(a.prev_fim, 19_000);
    }

    fn proc(b: Braco, v: f64) -> ProcessoControle {
        let l = b == Braco::L;
        ProcessoControle { braco: b, prefill_ms_mediana: v, imprimiu_ligado: l, produziu_perfil: l, valido: true }
    }

    /// D L L D D L com D = 100 ms e L = 100 * (1 + desvio).
    fn par(desvio: f64) -> Vec<ProcessoControle> {
        ORDEM_CONTROLE.iter().map(|&b| proc(b, if b == Braco::D { 100.0 } else { 100.0 * (1.0 + desvio) })).collect()
    }

    #[test]
    fn controle_classifica_pela_magnitude_do_desvio() {
        use ClasseControle::*;
        for (d, esperado) in [(0.02, Comprovada), (0.05, ComRessalva), (0.12, SemProva), (-0.02, Comprovada), (-0.05, ComRessalva), (-0.12, SemProva)] {
            let r = classifica_controle(&par(d), true);
            assert_eq!(r.classe, esperado, "desvio {d}");
            assert!((r.desvio.unwrap() - d).abs() < 1e-9);
        }
        assert_eq!(classifica_controle(&par(0.03), true).classe, Comprovada, "3% exato");
        assert_eq!(classifica_controle(&par(-0.03), true).classe, Comprovada);
        assert_eq!(classifica_controle(&par(0.10), true).classe, ComRessalva, "10% exato");
        assert_eq!(classifica_controle(&par(-0.10), true).classe, ComRessalva);
        assert_eq!(classifica_controle(&par(0.1001), true).classe, SemProva);
        // pior caso: maior L contra menor D
        let mut p = par(0.02);
        p[1].prefill_ms_mediana = 110.0;
        p[3].prefill_ms_mediana = 95.0;
        let r = classifica_controle(&p, true);
        assert!((r.pior_caso.unwrap() - 0.15789473684210525).abs() < 1e-9);
    }

    #[test]
    fn controle_incompleto_nao_devolve_classificacao() {
        use ClasseControle::Incompleto;
        let inc = |p: &[ProcessoControle], ok| {
            let r = classifica_controle(p, ok);
            assert!(matches!(r.classe, Incompleto(_)) && r.desvio.is_none() && r.pior_caso.is_none(), "{:?}", r.classe);
        };
        let p = par(0.01);
        inc(&p[..5], true); // par truncado
        let mut dois_l = par(0.01);
        dois_l.remove(5); // um braço L com 2 processos
        inc(&dois_l, true);
        let mut troca = par(0.01);
        troca.swap(0, 1); // L D L D D L
        inc(&troca, true);
        inc(&p, false); // orçamento insuficiente
        let mut d_vaza = par(0.01);
        d_vaza[0].produziu_perfil = true;
        inc(&d_vaza, true);
        let mut d_liga = par(0.01);
        d_liga[3].imprimiu_ligado = true;
        inc(&d_liga, true);
        let mut aborta = par(0.01);
        aborta[2].valido = false;
        inc(&aborta, true);
        let mut l_sem = par(0.01);
        l_sem[1].produziu_perfil = false;
        inc(&l_sem, true);
    }

    #[test]
    fn faixas_do_h6_nao_se_tocam_nos_restos_1_2_4_e_o_6_e_descritivo() {
        for r in [1usize, 2, 4, 5] {
            assert!(!h6_descritivo(r), "resto {r}");
        }
        for r in [6usize, 7] {
            assert!(h6_descritivo(r), "resto {r}");
            assert!(classifica_h6(r, 1.0).is_none());
        }
        assert_eq!(faixas_h6(1), (0.5625, 0.15625));
        assert_eq!(faixas_h6(2), (0.625, 0.3125));
        assert_eq!(faixas_h6(4), (0.75, 0.625));
        assert_eq!(classifica_h6(4, 0.80), Some(Leitura::Dentro));
        assert_eq!(classifica_h6(4, 0.60), Some(Leitura::Fora));
        assert_eq!(classifica_h6(4, 0.70), Some(Leitura::SemLeitura));
    }

    #[test]
    fn r6_compara_o_bloco_final_com_o_ultimo_cheio() {
        let b0 = bloco(1, 8, 0, [0.0, 0.0, 5.0, 5.0]); // 10 ms
        let b1 = bloco(2, 8, 11 * MS, [0.0, 0.0, 5.0, 5.0]);
        let b2 = bloco(3, 1, 22 * MS, [0.0, 0.0, 3.0, 3.0]); // 6 ms
        let p = RecPedido {
            indice: 0, aquecimento: false, n: 17, plano: vec![8, 8, 1], blocos: vec![b0, b1, b2], servico: vec![None; 3], servico_extra: 0,
            t_armado: 0, t_logits_ini: 0, t_desarmado: 0, prefill_ms: 0.0, decode_passo_ms_mediana: 1.0,
        };
        assert!((p.r6().unwrap() - 0.6).abs() < 1e-9);
    }
}
