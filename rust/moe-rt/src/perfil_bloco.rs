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
//! * Serviço (Categoria B): sete fases. Cada operação nomeada tem carimbo de INÍCIO e de FIM; só esses intervalos somam. O que a thread faz
//!   entre operações (diagnóstico, laço, escalonamento) não é de fase alguma e aparece como `S_fora` = I_g - soma:
//!   - `S_espera`: da entrada de `serve_layer` (na camada 0, no máximo de I_g.ini) até ver `seq[l]`; inclui a leitura de ids/pesos/`x`
//!     publicados, que `read_step` faz no mesmo laço da espera e não se separa sem mexer nele.
//!   - `S_monta`: união de experts e faltas em `serve_layer` (depois de `release_held`), e `cpu_step` mais a alocação de `y` em `run`.
//!   - `S_compute`: `compute_step`.
//!   - `S_answer`: `answer` (escreve a saída e a flag).
//!   - `S_livro`: `release_held` dentro de `serve_layer`, e `pin_layer`, `victim` e `copy_after_reader` (o diagnóstico/injeção que vem antes
//!     de `pin_layer` fica de fora).
//!   - `S_end_espera`: espera de `seq[L]` em `end_token`.
//!   - `S_end_trab`: o resto de `end_token` (soltar cópias, sync da cópia no modo com barreira, `query`, tabela, `publish`). Inclui, por
//!     isso, a espera do lado do buffer que o host devolve em `finish_token`; fica declarado, não separado.
//!   As operações ficam guardadas (capacidade reservada na criação do serviço) para o recorte em [t0, t4] de cada bloco do host feito no
//!   merge, que vai a campos `*_recorte_ms` separados das somas integrais. Validade: operações ordenadas, sem sobreposição, dentro de I_g.
//! * Intervalo do serviço: I_g vai de max(fim do bloco de serviço anterior, instante de armar) até um carimbo de fim próprio, lido depois de `end_token` do bloco g
//!   (independente das somas das operações). O
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
/// Durações das mesmas fases recortadas em [t0, t4] do bloco do host (campos à parte das somas integrais).
pub const NOMES_S_RECORTE: [&str; 7] = [
    "s_espera_recorte_ms",
    "s_monta_recorte_ms",
    "s_compute_recorte_ms",
    "s_answer_recorte_ms",
    "s_livro_recorte_ms",
    "s_end_espera_recorte_ms",
    "s_end_trab_recorte_ms",
];

/// Uma operação nomeada da thread de serviço: a fase (`S_*`) e o intervalo [`ini`, `fim`] em ns, medidos nos DOIS extremos.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpS {
    pub fase: u8,
    pub ini: u64,
    pub fim: u64,
}

impl OpS {
    /// Interseção da operação com [a, b] (ns).
    pub fn inter(&self, a: u64, b: u64) -> u64 {
        self.fim.min(b).saturating_sub(self.ini.max(a))
    }
}

/// O serviço de um bloco (geração `gen`): intervalo I_g, as operações nomeadas medidas (ordenadas, na ordem em que a thread as executou) e
/// as sete somas por fase, derivadas SÓ das operações. Os limites de I_g são carimbos próprios, independentes das somas: o que sobra é `S_fora`.
#[derive(Clone, Debug, PartialEq)]
pub struct RecServico {
    pub gen: u32,
    pub ini: u64,
    pub fim: u64,
    pub f: [u64; 7],
    /// Operações nomeadas (capacidade reservada pela thread de serviço). Vazio só em registros sintéticos de teste.
    pub ops: Vec<OpS>,
    /// Duração de cada fase recortada em [t0, t4] do bloco do host de mesma geração (calculada no merge; zeros antes dele). Vai ao JSON
    /// em campos separados das somas integrais, que alimentam H3 principal.
    pub recorte: [u64; 7],
}

impl RecServico {
    /// Registro a partir das operações medidas: as somas por fase saem delas.
    pub fn novo(gen: u32, ini: u64, fim: u64, ops: Vec<OpS>) -> Self {
        let mut f = [0u64; 7];
        for o in &ops {
            f[o.fase as usize] += o.fim.saturating_sub(o.ini);
        }
        Self { gen, ini, fim, f, ops, recorte: [0; 7] }
    }
    /// Registro só com somas (sem operações): uso dos testes que não precisam do recorte.
    pub fn de_somas(gen: u32, ini: u64, fim: u64, f: [u64; 7]) -> Self {
        Self { gen, ini, fim, f, ops: vec![], recorte: [0; 7] }
    }
    pub fn i_g(&self) -> u64 {
        self.fim.saturating_sub(self.ini)
    }
    pub fn soma(&self) -> u64 {
        self.f.iter().sum()
    }
    pub fn s_fora(&self) -> u64 {
        self.i_g().saturating_sub(self.soma())
    }
    /// Fim depois do início, soma das fases dentro de I_g e, havendo operações: cada uma com fim >= início, dentro de I_g, em ordem e SEM
    /// sobreposição entre si (thread única), e as somas iguais às das operações.
    pub fn coerente(&self) -> bool {
        if self.fim < self.ini || self.soma() > self.i_g() + TOL_RELOGIO_NS {
            return false;
        }
        if self.ops.is_empty() {
            return true;
        }
        let ordem = self.ops.windows(2).all(|w| w[1].ini + TOL_RELOGIO_NS >= w[0].fim);
        let cada = self.ops.iter().all(|o| (o.fase as usize) < 7 && o.fim >= o.ini && o.ini + TOL_RELOGIO_NS >= self.ini && o.fim <= self.fim + TOL_RELOGIO_NS);
        let mut f = [0u64; 7];
        for o in self.ops.iter().filter(|o| (o.fase as usize) < 7) {
            f[o.fase as usize] += o.fim.saturating_sub(o.ini);
        }
        ordem && cada && f == self.f
    }
    /// Quanto de I_g cai dentro de [t0, t4] do bloco do host (descritivo; sem somar duas vezes).
    pub fn inter_ns(&self, b: &RecBloco) -> u64 {
        self.fim.min(b.t[4]).saturating_sub(self.ini.max(b.t[0]))
    }
    /// O recorte do merge: a parte de cada operação dentro de [t0, t4] do bloco `b`, somada por fase, guardada em `recorte`.
    pub fn recorta(&mut self, b: &RecBloco) {
        let mut r = [0u64; 7];
        for o in self.ops.iter().filter(|o| (o.fase as usize) < 7) {
            r[o.fase as usize] += o.inter(b.t[0], b.t[4]);
        }
        self.recorte = r;
    }
}

/// I_g de blocos consecutivos (por geração crescente) não se sobrepõem.
pub fn servicos_sem_sobreposicao(v: &[&RecServico]) -> bool {
    v.windows(2).all(|w| w[1].ini + TOL_RELOGIO_NS >= w[0].fim)
}

/// Acumuladores da thread de serviço para o bloco em andamento. Escritora única: a thread de serviço. Com capacidade reservada, as operações
/// de um bloco armado não alocam; só `fecha` copia as operações para o registro, depois do carimbo de fim de I_g.
#[derive(Default, Debug)]
pub struct AccServ {
    pub armado: bool,
    pub ini: u64,
    pub prev_fim: u64,
    pub f: [u64; 7],
    pub ops: Vec<OpS>,
}

impl AccServ {
    pub fn com_capacidade(n: usize) -> Self {
        Self { ops: Vec::with_capacity(n), ..Self::default() }
    }
    /// Depois de ver `seq[l]` no instante `tb`, com a espera iniciada em `t_esp` (entrada de `serve_layer`). Na camada 0 abre o bloco: lê o
    /// armamento e fixa I_g.ini = max(fim anterior, armar); a espera da camada 0 conta só de I_g.ini em diante.
    pub fn apos_espera(&mut self, l: usize, t_esp: u64, tb: u64, armado: bool, t_armado: u64) {
        let mut ini_esp = t_esp;
        if l == 0 {
            self.f = [0; 7];
            self.ops.clear();
            self.armado = armado;
            self.ini = self.prev_fim.max(t_armado);
            ini_esp = ini_esp.max(self.ini);
        }
        self.op(S_ESPERA, ini_esp, tb);
    }
    /// Registra a operação nomeada `i` no intervalo medido [`ini`, `fim`]. Nada fora das operações entra em fase alguma. Sem armamento, não
    /// guarda nada (o decode não registra).
    pub fn op(&mut self, i: usize, ini: u64, fim: u64) {
        if !self.armado {
            return;
        }
        self.f[i] += fim.saturating_sub(ini);
        self.ops.push(OpS { fase: i as u8, ini, fim });
    }
    /// Fim do bloco no carimbo `fim` (independente das operações): devolve o registro se o bloco nasceu armado e zera para o próximo.
    pub fn fecha(&mut self, gen: u32, fim: u64) -> Option<RecServico> {
        let r = self.armado.then(|| RecServico { gen, ini: self.ini, fim, f: self.f, ops: self.ops.iter().copied().collect(), recorte: [0; 7] });
        self.prev_fim = fim;
        self.f = [0; 7];
        self.ops.clear();
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
    /// `LASTRO_PERFIL_F2` ligada (só existe com o instrumento de bloco ligado).
    f2: bool,
    /// Camadas do motor (I3 da F2 do host); 0 com a coleta desligada.
    layers: usize,
    f2_blocos: Mutex<Vec<BlocoF2>>,
    f2_tb: Mutex<Vec<RecTb>>,
    /// `LASTRO_PERFIL_F2B` ligada (só existe com bloco e F2).
    f2b: bool,
    f2b_blocos: Mutex<Vec<BlocoB>>,
}

impl Compartilhado {
    pub fn novo() -> Self {
        Self::novo_f2(false, 0)
    }
    /// Como [`novo`](Self::novo), com a coleta da F2 do host (`LASTRO_PERFIL_F2`) escolhida por quem chama.
    pub fn novo_f2(f2: bool, layers: usize) -> Self {
        Self::novo_f2b(f2, false, layers)
    }
    /// Como [`novo_f2`](Self::novo_f2), com a F2 nível B (`LASTRO_PERFIL_F2B`); `f2b` só vale com `f2`.
    pub fn novo_f2b(f2: bool, f2b: bool, layers: usize) -> Self {
        let _ = agora_ns(); // fixa a origem antes de qualquer carimbo
        Self { armado: AtomicBool::new(false), t_armado: AtomicU64::new(0), blocos: Mutex::new(vec![]), servico: Mutex::new(vec![]),
               f2, layers, f2_blocos: Mutex::new(vec![]), f2_tb: Mutex::new(vec![]), f2b: f2 && f2b, f2b_blocos: Mutex::new(vec![]) }
    }
    /// A coleta da F2 nível B está ligada neste motor.
    pub fn f2b(&self) -> bool {
        self.f2b
    }
    pub fn registra_f2b_bloco(&self, r: BlocoB) {
        trava(&self.f2b_blocos).push(r);
    }
    /// A coleta da F2 do host está ligada neste motor.
    pub fn f2(&self) -> bool {
        self.f2
    }
    pub fn registra_f2_bloco(&self, r: BlocoF2) {
        trava(&self.f2_blocos).push(r);
    }
    pub fn registra_f2_tb(&self, r: RecTb) {
        trava(&self.f2_tb).push(r);
    }
    /// Publica o fim do bloco do serviço: o `RecTb` (F2) ANTES do `RecServico`, que é o marcador de conclusão esperado pelo merge. `entre` roda
    /// entre as duas publicações (os testes pausam o serviço aí).
    pub fn publica_servico(&self, srv: Option<RecServico>, tb: Option<RecTb>, entre: &mut dyn FnMut()) {
        if let Some(t) = tb {
            self.registra_f2_tb(t);
        }
        entre();
        if let Some(r) = srv {
            self.registra_servico(r);
        }
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
    /// Nome do modelo (célula) do pedido; o binário preenche depois, o motor não o conhece.
    pub modelo: String,
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
        if p.f2 {
            let mut b = trava(&p.f2_blocos);
            b.clear();
            b.reserve(blocos_previstos);
            let mut t = trava(&p.f2_tb);
            t.clear();
            t.reserve(blocos_previstos);
        }
        if p.f2b {
            let mut b = trava(&p.f2b_blocos);
            b.clear();
            b.reserve(blocos_previstos);
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
        let servico: Vec<Option<RecServico>> = blocos
            .iter()
            .map(|b| {
                sv.iter().position(|s| s.gen == b.gen).map(|i| {
                    let mut r = sv.swap_remove(i);
                    r.recorta(b);
                    r
                })
            })
            .collect();
        self.pf = None;
        let servico_extra = sv.len();
        let f2 = pf.f2.then(|| {
            let mut hb = std::mem::take(&mut *trava(&pf.f2_blocos));
            let mut tb = std::mem::take(&mut *trava(&pf.f2_tb));
            // Alinhado aos blocos do host, pela geração; o que não casa fica fora (a I4 e a I8 acusam a falta).
            let hosts: Vec<Option<BlocoF2>> = blocos.iter().map(|b| hb.iter().position(|h| h.gen == b.gen).map(|i| hb.swap_remove(i))).collect();
            let tbs: Vec<Option<RecTb>> = blocos.iter().map(|b| tb.iter().position(|h| h.gen == b.gen).map(|i| tb.swap_remove(i))).collect();
            DadosF2 { layers: pf.layers, hosts, tbs, host_extra: hb.len(), tb_extra: tb.len() }
        });
        let f2b = pf.f2b.then(|| {
            let mut bb = std::mem::take(&mut *trava(&pf.f2b_blocos));
            let blocos_b: Vec<Option<BlocoB>> = blocos.iter().map(|b| bb.iter().position(|h| h.gen == b.gen).map(|i| bb.swap_remove(i))).collect();
            DadosF2B { layers: pf.layers, blocos: blocos_b, extra: bb.len() }
        });
        Some(RecPedido {
            modelo: d.modelo,
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
            f2,
            f2b,
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
            if p.f2 {
                trava(&p.f2_blocos).clear();
                trava(&p.f2_tb).clear();
            }
            if p.f2b {
                trava(&p.f2b_blocos).clear();
            }
        }
    }
}

// ---- A entrada de um pedido ---------------------------------------------------------------------------------------------------------------

/// Uma entrada de `perfil-bloco.json`: tudo de um pedido (prefill apenas; o decode só entra pela mediana do `passo_ms`).
#[derive(Clone, Debug, PartialEq)]
pub struct RecPedido {
    /// Modelo do pedido (célula): `indice` reinicia em 0 a cada motor, então o par (modelo, indice) é a chave num arquivo com várias células.
    pub modelo: String,
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
    /// F2 do host por operação (`LASTRO_PERFIL_F2`): `None` com a variável desligada. Não entra em `perfil-bloco.json` (esquema v2 intacto).
    pub f2: Option<DadosF2>,
    /// F2 nível B (`LASTRO_PERFIL_F2B`): `None` com a variável desligada. Não entra em `perfil-bloco.json` nem em `perfil-f2.json`.
    pub f2b: Option<DadosF2B>,
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
                        "inter_bloco_ms": ms(s.inter_ns(b)), "coerente": s.coerente(), "n_ops": s.ops.len(),
                    });
                    for (k, nome) in NOMES_S.iter().enumerate() {
                        v[*nome] = json!(ms(s.f[k]));
                        v[NOMES_S_RECORTE[k]] = json!(ms(s.recorte[k]));
                    }
                    v
                }
            })
            .collect();
        let problemas = self.problemas();
        json!({
            "modelo": self.modelo, "indice": self.indice, "aquecimento": self.aquecimento, "n": self.n, "plano_m": self.plano,
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
        "versao": 2,
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

// ---- H1 a H5: classificação pré-registrada (seção 1.3 do plano) ---------------------------------------------------------------------------
//
// Funções puras sobre as medianas já calculadas (por modelo, série e texto, blocos cheios, pedidos medidos). Os limiares são os do plano,
// com `>=`/`<=` e `<`/`>` exatamente como escritos lá. Interpretações onde o texto do plano deixa margem (registradas também na nota):
//  * Série: H1, H2 e H4 só têm previsão no L100; H3 e H5 têm uma por série (L10 e L100). Em série sem previsão registrada (LC, ou L10
//    em H1/H2/H4) o resultado é `SemDado`: não há o que classificar, e isso não é "sem leitura" (faixa intermediária).
//  * "Sem leitura" é só a faixa intermediária declarada. Dado ausente ou não finito = `SemDado`. Controle que não deixa a (modelo, série)
//    válida (`SemProva` ou `Incompleto`) = `SemProva`, antes de olhar qualquer número. `Comprovada` e `ComRessalva` classificam.
//  * Previsão composta (H2, H3 no L10: "dentro se A e B; fora se A' ou B'"): "fora" decide com UMA medida que viole, mesmo faltando a
//    outra; "dentro" exige as duas; sem decidir e com falta de uma delas = `SemDado`.
//  * H1: "dentro" usa F2/B e "fora" usa F3/B, medidas diferentes. As duas condições juntas (F2/B >= 0,70 e F3/B >= 0,50) somariam mais
//    que B: dado incoerente, `SemDado`.
//  * H6: resto fora de 1..=7, ou faixas que se tocam (resto 6 e 7), é `Descritivo`; sem R6 é `SemDado`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Serie {
    L10,
    L100,
    Lc,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hip {
    H1,
    H2,
    H3,
    H4,
    H5,
    H6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClasseH {
    Dentro,
    Fora,
    /// Faixa intermediária declarada.
    SemLeitura,
    /// Dado ausente, série sem previsão registrada ou dado incoerente.
    SemDado,
    /// Controle do custo do instrumento ausente, incompleto ou com |desvio| > 10%.
    SemProva,
    /// H6: as faixas do resto se tocam (ou o resto não existe): só descrição.
    Descritivo,
}

/// As medianas que H1 a H6 usam. `None` = a medida não existe.
#[derive(Clone, Copy, Debug, Default)]
pub struct MedidasH {
    pub f1_b: Option<f64>,
    pub f2_b: Option<f64>,
    pub f3_b: Option<f64>,
    /// F3 / `passo_ms` do braço D (denominador de referência de H2).
    pub f3_passo: Option<f64>,
    /// R3 = (S_compute + S_livro) / B, soma integral (o recortado fica à parte).
    pub r3: Option<f64>,
    pub s_compute_b: Option<f64>,
    /// (S_livro + S_end_trab) / B.
    pub r5: Option<f64>,
    pub r6: Option<f64>,
    /// Resto do texto (n % 8) para H6.
    pub resto: usize,
}

fn fin(x: Option<f64>) -> Option<f64> {
    x.filter(|v| v.is_finite())
}

/// "dentro se a >= da e b >= db; fora se a < fa ou b < fb; senão sem leitura".
fn composta(a: Option<f64>, b: Option<f64>, (da, db): (f64, f64), (fa, fb): (f64, f64)) -> ClasseH {
    let (a, b) = (fin(a), fin(b));
    if a.is_some_and(|x| x < fa) || b.is_some_and(|x| x < fb) {
        return ClasseH::Fora;
    }
    match (a, b) {
        (Some(x), Some(y)) if x >= da && y >= db => ClasseH::Dentro,
        (Some(_), Some(_)) => ClasseH::SemLeitura,
        _ => ClasseH::SemDado,
    }
}

/// "dentro se x >= d; fora se x < f; senão sem leitura" (H4, H5 no L10).
fn simples_ge(x: Option<f64>, d: f64, f: f64) -> ClasseH {
    match fin(x) {
        None => ClasseH::SemDado,
        Some(v) if v >= d => ClasseH::Dentro,
        Some(v) if v < f => ClasseH::Fora,
        Some(_) => ClasseH::SemLeitura,
    }
}

/// "dentro se x <= d; fora se x > f; senão sem leitura" (H3 e H5 no L100).
fn simples_le(x: Option<f64>, d: f64, f: f64) -> ClasseH {
    match fin(x) {
        None => ClasseH::SemDado,
        Some(v) if v <= d => ClasseH::Dentro,
        Some(v) if v > f => ClasseH::Fora,
        Some(_) => ClasseH::SemLeitura,
    }
}

/// H1 (L100): dentro se F2/B >= 0,70; fora se F3/B >= 0,50.
pub fn classifica_h1(serie: Serie, f2_b: Option<f64>, f3_b: Option<f64>) -> ClasseH {
    if serie != Serie::L100 {
        return ClasseH::SemDado;
    }
    let (dentro, fora) = (fin(f2_b).is_some_and(|x| x >= 0.70), fin(f3_b).is_some_and(|x| x >= 0.50));
    match (dentro, fora) {
        (true, true) => ClasseH::SemDado,
        (true, false) => ClasseH::Dentro,
        (false, true) => ClasseH::Fora,
        (false, false) if fin(f2_b).is_some() && fin(f3_b).is_some() => ClasseH::SemLeitura,
        _ => ClasseH::SemDado,
    }
}

/// H2 (L100): dentro se F3/B >= 0,50 e F3/passo >= 3,0; fora se F3/B < 0,25 ou F3/passo < 1,5.
pub fn classifica_h2(serie: Serie, f3_b: Option<f64>, f3_passo: Option<f64>) -> ClasseH {
    if serie != Serie::L100 {
        return ClasseH::SemDado;
    }
    composta(f3_b, f3_passo, (0.50, 3.0), (0.25, 1.5))
}

/// H3. L10: dentro se R3 >= 0,50 e S_compute/B >= 0,20; fora se R3 < 0,30 ou S_compute/B < 0,10. L100: dentro se R3 <= 0,10; fora se R3 > 0,25.
pub fn classifica_h3(serie: Serie, r3: Option<f64>, s_compute_b: Option<f64>) -> ClasseH {
    match serie {
        Serie::L10 => composta(r3, s_compute_b, (0.50, 0.20), (0.30, 0.10)),
        Serie::L100 => simples_le(r3, 0.10, 0.25),
        Serie::Lc => ClasseH::SemDado,
    }
}

/// H4 (L100): dentro se F1/B >= 0,05; fora se F1/B < 0,02.
pub fn classifica_h4(serie: Serie, f1_b: Option<f64>) -> ClasseH {
    if serie != Serie::L100 {
        return ClasseH::SemDado;
    }
    simples_ge(f1_b, 0.05, 0.02)
}

/// H5. L10: dentro se (S_livro + S_end_trab)/B >= 0,10; fora se < 0,03. L100: dentro se <= 0,02; fora se > 0,05.
pub fn classifica_h5(serie: Serie, r5: Option<f64>) -> ClasseH {
    match serie {
        Serie::L10 => simples_ge(r5, 0.10, 0.03),
        Serie::L100 => simples_le(r5, 0.02, 0.05),
        Serie::Lc => ClasseH::SemDado,
    }
}

/// H6 pelo resto do texto: `Descritivo` se as faixas se tocam ou o resto não está em 1..=7; `SemDado` sem R6.
pub fn classifica_h6_medida(resto: usize, r6: Option<f64>) -> ClasseH {
    if !(1..=BLOCO - 1).contains(&resto) || h6_descritivo(resto) {
        return ClasseH::Descritivo;
    }
    match fin(r6).and_then(|x| classifica_h6(resto, x)) {
        Some(Leitura::Dentro) => ClasseH::Dentro,
        Some(Leitura::Fora) => ClasseH::Fora,
        Some(Leitura::SemLeitura) => ClasseH::SemLeitura,
        None => ClasseH::SemDado,
    }
}

/// A porta do controle: só `Comprovada` e `ComRessalva` deixam a (modelo, série) classificável.
pub fn controle_permite(c: &ClasseControle) -> bool {
    matches!(c, ClasseControle::Comprovada | ClasseControle::ComRessalva)
}

/// Classifica a hipótese `h` na `serie`, dadas as medianas e a classe do controle da (modelo, série). Controle que não permite = `SemProva`.
pub fn classifica_h(h: Hip, serie: Serie, m: &MedidasH, controle: &ClasseControle) -> ClasseH {
    if !controle_permite(controle) {
        return ClasseH::SemProva;
    }
    match h {
        Hip::H1 => classifica_h1(serie, m.f2_b, m.f3_b),
        Hip::H2 => classifica_h2(serie, m.f3_b, m.f3_passo),
        Hip::H3 => classifica_h3(serie, m.r3, m.s_compute_b),
        Hip::H4 => classifica_h4(serie, m.f1_b),
        Hip::H5 => classifica_h5(serie, m.r5),
        Hip::H6 => classifica_h6_medida(m.resto, m.r6),
    }
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

// ---- F2 do host por operação em L100 (plano `plano-f2-host-20261111.md`, versão 3, aprovada na rodada 147) -----------------------------------
//
// Descritivo, como o resto do módulo. Ligado por `LASTRO_PERFIL_F2=1` (só o valor exato `1`) e só junto com `LASTRO_PERFIL_BLOCO=1`; desligado
// não há acumulador, relógio nem gravação (`Engine::f2_acc` é `None` e `enqueue_inner` recebe `None`). Saída em `perfil-f2.json` (esquema
// próprio, versão 1); o `perfil-bloco.json` versão 2 não muda.
//
// Decisões da implementação (o plano deixou em aberto ou o código real pediu):
//  * A entrega do acumulador ao registro do bloco acontece em `le_ids_com_pendente`, DEPOIS do carimbo t4 e fora de F2 (não no fim de
//    `enqueue_step`): a cópia do vetor não entra em `R_borda`. O custo cai em R_entre do bloco seguinte e é medido pelo par de controle.
//  * `H_BEGIN_STEP` é gravada sempre (duração perto de zero quando `max_positions == 1` e `begin_step` não é chamado), como `H_WAIT_EVENT`,
//    para que I3 seja 6 x layers + 7 independente da configuração.
//  * `H_WAIT_EVENT` cobre o laço inteiro das esperas (a soma dos `wait_event` mais o custo do laço); `n_ready` fica à parte.
//  * Os carimbos do serviço (`tb_obs_l`, `tb_l`) cobrem as camadas 0..layers-1 (as que passam por `apos_espera`); `seq[layers]` é esperada em
//    `end_token` por `wait_seq` e não tem carimbo `tb`.

/// Nome da variável que liga a F2 do host.
pub const VARIAVEL_F2: &str = "LASTRO_PERFIL_F2";

/// Só o valor exato `1` liga (mesmo contrato de [`interpreta`]).
pub fn interpreta_f2(v: Option<&str>) -> bool {
    interpreta(v)
}

/// Lê `LASTRO_PERFIL_F2` do ambiente (não UTF-8 conta como ausente). Não olha a variável de bloco: [`f2_efetivo`] combina as duas.
pub fn f2_do_ambiente() -> bool {
    interpreta_f2(std::env::var(VARIAVEL_F2).ok().as_deref())
}

/// A F2 só vale com o instrumento de bloco ligado.
pub fn f2_efetivo(bloco: bool, f2: bool) -> bool {
    bloco && f2
}

/// A linha de estado do binário: `perfil-f2: ligado`, `perfil-f2: desligado` ou, pedida sem o bloco, `perfil-f2: desligado (exige perfil-bloco)`.
pub fn linha_estado_f2(bloco: bool, f2: bool) -> &'static str {
    match (bloco, f2) {
        (true, true) => "perfil-f2: ligado",
        (false, true) => "perfil-f2: desligado (exige perfil-bloco)",
        _ => "perfil-f2: desligado",
    }
}

/// Operações do host (códigos estáveis do JSON). As sete fixas e as seis por camada.
pub const H_ENTRADA: u8 = 0;
pub const H_BEGIN_STEP: u8 = 1;
pub const H_UPLOAD_TABELA: u8 = 2;
pub const H_WAIT_EVENT: u8 = 3;
pub const H_POST_LAYERS: u8 = 4;
pub const H_SEQ_FIM: u8 = 5;
pub const H_SAIDA: u8 = 6;
pub const H_PRE_LAYER: u8 = 7;
pub const H_ROUTE: u8 = 8;
pub const H_SEQ: u8 = 9;
pub const H_LAUNCH_HITS: u8 = 10;
pub const H_WAIT_FLAG: u8 = 11;
pub const H_ADD_Y: u8 = 12;
pub const NOMES_H: [&str; 13] = [
    "H_ENTRADA", "H_BEGIN_STEP", "H_UPLOAD_TABELA", "H_WAIT_EVENT", "H_POST_LAYERS", "H_SEQ_FIM", "H_SAIDA", "H_PRE_LAYER", "H_ROUTE", "H_SEQ",
    "H_LAUNCH_HITS", "H_WAIT_FLAG", "H_ADD_Y",
];
/// As seis operações de cada camada, na ordem em que o motor as enfileira.
pub const OPS_CAMADA: [u8; 6] = [H_PRE_LAYER, H_ROUTE, H_SEQ, H_LAUNCH_HITS, H_WAIT_FLAG, H_ADD_Y];
/// Marca de "sem camada" nas operações fixas.
pub const SEM_CAMADA: u32 = u32::MAX;

/// Operações do bloco: 6 x layers + 7 (I3).
pub fn n_ops_esperado(layers: usize) -> usize {
    6 * layers + 7
}

/// Capacidade do acumulador por bloco: layers x 6 + 8 (as operações mais uma folga de uma), independente de `n_ready`.
pub fn capacidade_f2(layers: usize) -> usize {
    layers * 6 + 8
}

/// Uma operação do host: o código `H_*`, a camada (ou [`SEM_CAMADA`]) e o intervalo [`ini`, `fim`] em ns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpH {
    pub op: u8,
    pub camada: u32,
    pub ini: u64,
    pub fim: u64,
}

impl OpH {
    pub fn dur(&self) -> u64 {
        self.fim.saturating_sub(self.ini)
    }
}

/// Acumulador do host para o bloco em andamento (escritor único: a linha principal). Capacidade fixa reservada na criação do motor; as
/// marcas do caminho quente só empurram em vetor com folga, sem alocar.
#[derive(Debug)]
pub struct AccHost {
    pub ops: Vec<OpH>,
    /// `agora_ns()` próprio tomado na entrada de `enqueue_step`.
    pub entrada: u64,
    /// `agora_ns()` próprio tomado antes do `Ok(())` de `enqueue_step`.
    pub saida: u64,
    pub n_ready: usize,
    ult: u64,
}

impl AccHost {
    pub fn com_capacidade(layers: usize) -> Self {
        Self { ops: Vec::with_capacity(capacidade_f2(layers)), entrada: 0, saida: 0, n_ready: 0, ult: 0 }
    }
    pub fn capacidade(&self) -> usize {
        self.ops.capacity()
    }
    /// Zera para o bloco seguinte (mantém a capacidade) e fixa o carimbo de entrada.
    pub fn reinicia(&mut self, entrada: u64) {
        self.ops.clear();
        self.entrada = entrada;
        self.saida = 0;
        self.n_ready = 0;
        self.ult = entrada;
    }
    /// Abre a cadeia interna (início de `enqueue_inner`): grava `H_ENTRADA` de `entrada` até agora.
    pub fn abre(&mut self, n_ready: usize) {
        self.n_ready = n_ready;
        let agora = agora_ns();
        self.ops.push(OpH { op: H_ENTRADA, camada: SEM_CAMADA, ini: self.entrada, fim: agora });
        self.ult = agora;
    }
    /// Fim de uma chamada ao backend: o início é o fim da anterior (carimbos encadeados).
    #[inline]
    pub fn marca(&mut self, op: u8, camada: u32) {
        let agora = agora_ns();
        self.ops.push(OpH { op, camada, ini: self.ult, fim: agora });
        self.ult = agora;
    }
    /// Fecha a cadeia no carimbo `saida` (depois de `cur_m` e `current`): grava `H_SAIDA`.
    pub fn fecha_saida(&mut self, saida: u64) {
        self.ops.push(OpH { op: H_SAIDA, camada: SEM_CAMADA, ini: self.ult, fim: saida });
        self.saida = saida;
        self.ult = saida;
    }
    /// O registro do bloco (copia as operações: aloca, fora de F2).
    pub fn para_bloco(&self, gen: u32, m: usize, pos_ini: u64, t2: u64, t3: u64) -> BlocoF2 {
        BlocoF2 { gen, m, pos_ini, n_ready: self.n_ready, ops: self.ops.clone(), entrada: self.entrada, saida: self.saida, t2, t3 }
    }
}

/// O host de um bloco: as operações encadeadas, os dois carimbos de borda e os t2/t3 externos de `leitura.rs` (os do `RecBloco`).
#[derive(Clone, Debug, PartialEq)]
pub struct BlocoF2 {
    pub gen: u32,
    pub m: usize,
    pub pos_ini: u64,
    pub n_ready: usize,
    pub ops: Vec<OpH>,
    pub entrada: u64,
    pub saida: u64,
    pub t2: u64,
    pub t3: u64,
}

impl BlocoF2 {
    pub fn f2_ns(&self) -> u64 {
        self.t3.saturating_sub(self.t2)
    }
    pub fn soma_ops_ns(&self) -> u64 {
        self.ops.iter().map(OpH::dur).sum()
    }
    /// `R_borda = (entrada - t2) + (t3 - saida)`.
    pub fn r_borda_ns(&self) -> u64 {
        self.entrada.saturating_sub(self.t2) + self.t3.saturating_sub(self.saida)
    }
    /// Soma das durações da operação `op` (todas as camadas).
    pub fn soma_op_ns(&self, op: u8) -> u64 {
        self.ops.iter().filter(|o| o.op == op).map(OpH::dur).sum()
    }
    /// Fim da operação `H_SEQ` da camada `l`.
    pub fn fim_seq(&self, l: usize) -> Option<u64> {
        self.ops.iter().find(|o| o.op == H_SEQ && o.camada as usize == l).map(|o| o.fim)
    }
    /// I1: carimbos internos não decrescentes (encadeados, sem sobreposição) e dentro de [t2, t3] com `TOL_RELOGIO_NS` nas bordas.
    pub fn i1(&self) -> bool {
        let Some(p) = self.ops.first() else { return false };
        let dentro = |x: u64| x + TOL_RELOGIO_NS >= self.t2 && x <= self.t3 + TOL_RELOGIO_NS;
        if self.entrada != p.ini {
            return false;
        }
        let mut prev = p.ini;
        for o in &self.ops {
            if o.ini < prev || o.fim < o.ini || !dentro(o.ini) || !dentro(o.fim) {
                return false;
            }
            prev = o.fim;
        }
        prev == self.saida && dentro(self.saida) && dentro(self.entrada)
    }
    /// I2: soma(operações) + `R_borda` = t3 - t2, dentro de 1 µs por carimbo de borda (2 µs).
    pub fn i2(&self) -> bool {
        (self.soma_ops_ns() + self.r_borda_ns()).abs_diff(self.f2_ns()) <= 2 * TOL_RELOGIO_NS
    }
    /// I3: 6 x layers + 7 operações na ordem do motor, camadas 0..layers-1 sem repetir nem faltar.
    pub fn i3(&self, layers: usize) -> bool {
        if self.ops.len() != n_ops_esperado(layers) {
            return false;
        }
        let esperado = std::iter::empty()
            .chain([H_ENTRADA, H_BEGIN_STEP, H_UPLOAD_TABELA, H_WAIT_EVENT].map(|o| (o, SEM_CAMADA)))
            .chain((0..layers).flat_map(|l| OPS_CAMADA.map(|o| (o, l as u32))))
            .chain([H_POST_LAYERS, H_SEQ_FIM, H_SAIDA].map(|o| (o, SEM_CAMADA)));
        self.ops.iter().zip(esperado).all(|(o, (op, cam))| o.op == op && o.camada == cam)
    }
    /// I5: soma por camada das seis operações mais as fixas = soma de todas as operações.
    pub fn i5(&self, layers: usize) -> bool {
        let por_camada: u64 = (0..layers as u32).map(|l| self.ops.iter().filter(|o| o.camada == l && OPS_CAMADA.contains(&o.op)).map(OpH::dur).sum::<u64>()).sum();
        let fixas: u64 = self.ops.iter().filter(|o| o.camada == SEM_CAMADA).map(OpH::dur).sum();
        por_camada + fixas == self.soma_ops_ns()
    }
    /// `R_borda` / F2 (0 com F2 nula).
    pub fn cobertura(&self) -> f64 {
        if self.f2_ns() == 0 {
            0.0
        } else {
            self.r_borda_ns() as f64 / self.f2_ns() as f64
        }
    }
    /// I6: ressalva (não invalida) quando `R_borda`/F2 passa de 3%.
    pub fn ressalva_borda(&self) -> bool {
        self.cobertura() > LIM_COBERTURA_HOST
    }
    /// Soma das durações do bloco nas seis operações da camada `l`.
    pub fn soma_camada_ns(&self, l: usize) -> u64 {
        self.ops.iter().filter(|o| o.camada as usize == l && OPS_CAMADA.contains(&o.op)).map(OpH::dur).sum()
    }
}

/// Carimbos do serviço de uma camada: a geração `gen` que o serviço gravou (a do token), a primeira observação de `seq[l]` e o carimbo pós-leitura.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TbL {
    pub camada: u32,
    pub gen: u32,
    pub tb_obs: u64,
    pub tb: u64,
}

/// Os carimbos do serviço de um bloco (uma entrada por camada observada).
#[derive(Clone, Debug, PartialEq)]
pub struct RecTb {
    pub gen: u32,
    pub camadas: Vec<TbL>,
}

/// Acumulador do serviço (escritor único: a thread de serviço); capacidade `layers`.
#[derive(Debug, Default)]
pub struct AccTb {
    pub camadas: Vec<TbL>,
}

impl AccTb {
    pub fn com_capacidade(layers: usize) -> Self {
        Self { camadas: Vec::with_capacity(layers) }
    }
    /// Camada 0 abre o bloco.
    pub fn grava(&mut self, camada: usize, gen: u32, tb_obs: u64, tb: u64) {
        if camada == 0 {
            self.camadas.clear();
        }
        self.camadas.push(TbL { camada: camada as u32, gen, tb_obs, tb });
    }
    pub fn fecha(&mut self, gen: u32) -> RecTb {
        let r = RecTb { gen, camadas: self.camadas.clone() };
        self.camadas.clear();
        r
    }
}

fn dif(a: u64, b: u64) -> i64 {
    (a as i128 - b as i128).clamp(i64::MIN as i128, i64::MAX as i128) as i64
}

/// Grandezas descritivas por camada, com sinal (ns). Negativos são conservados (a observação pode vir antes do retorno da chamada no host).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DefasL {
    pub camada: u32,
    /// `tb_obs_l - fim(H_SEQ_l)`.
    pub defasagem_obs: i64,
    /// `tb_l - fim(H_SEQ_l)`.
    pub defasagem: i64,
    /// `tb_obs_l - tb_obs_(l-1)` (None na camada 0).
    pub intervalo_obs: Option<i64>,
    /// `tb_l - tb_(l-1)` (None na camada 0).
    pub intervalo: Option<i64>,
}

/// Defasagens e intervalos de um bloco; camadas sem `H_SEQ` no host ficam de fora.
pub fn defasagens(h: &BlocoF2, tb: &RecTb) -> Vec<DefasL> {
    let mut out = Vec::with_capacity(tb.camadas.len());
    for (i, c) in tb.camadas.iter().enumerate() {
        let Some(fim) = h.fim_seq(c.camada as usize) else { continue };
        let ant = i.checked_sub(1).map(|j| tb.camadas[j]);
        out.push(DefasL {
            camada: c.camada,
            defasagem_obs: dif(c.tb_obs, fim),
            defasagem: dif(c.tb, fim),
            intervalo_obs: ant.map(|a| dif(c.tb_obs, a.tb_obs)),
            intervalo: ant.map(|a| dif(c.tb, a.tb)),
        });
    }
    out
}

/// I8 (só da coleta de carimbos do serviço; invalida a HI do pedido, não as outras hipóteses). Causal: `tb_obs_l >= ini(H_SEQ_l) - TOL_RELOGIO_NS`
/// (a GPU não escreve `seq[l]` antes de o host começar a chamada que a enfileira). `tb_obs == 0` é acusado à parte ("tb_obs ausente").
/// A defasagem negativa NÃO é violação.
pub fn i8_problemas(h: &BlocoF2, tb: Option<&RecTb>, layers: usize) -> Vec<String> {
    let Some(tb) = tb else { return vec![format!("bloco gen {}: sem carimbos do serviço", h.gen)] };
    let mut p = vec![];
    if tb.gen != h.gen {
        p.push(format!("bloco gen {}: registro do serviço com gen {}", h.gen, tb.gen));
    }
    if tb.camadas.len() != layers {
        p.push(format!("bloco gen {}: {} camadas com carimbo de serviço, esperadas {layers}", h.gen, tb.camadas.len()));
    }
    for (i, c) in tb.camadas.iter().enumerate() {
        if c.camada as usize != i {
            p.push(format!("bloco gen {}: camada {} fora de ordem na posição {i}", h.gen, c.camada));
        }
        if c.gen != h.gen {
            p.push(format!("bloco gen {}: camada {} com geração de serviço {}", h.gen, c.camada, c.gen));
        }
        if c.tb_obs == 0 {
            p.push(format!("bloco gen {}: camada {}: tb_obs ausente", h.gen, c.camada));
            continue;
        }
        if c.tb_obs > c.tb {
            p.push(format!("bloco gen {}: camada {} com tb_obs > tb", h.gen, c.camada));
        }
        match h.ops.iter().find(|o| o.op == H_SEQ && o.camada as usize == i) {
            Some(o) if c.tb_obs + TOL_RELOGIO_NS < o.ini => p.push(format!("bloco gen {}: camada {} com tb_obs anterior ao início de H_SEQ", h.gen, c.camada)),
            None => p.push(format!("bloco gen {}: camada {} sem H_SEQ no host", h.gen, c.camada)),
            _ => {}
        }
        if i > 0 {
            let a = tb.camadas[i - 1];
            if c.tb < a.tb || c.tb_obs < a.tb_obs {
                p.push(format!("bloco gen {}: carimbos não crescentes na camada {}", h.gen, c.camada));
            }
        }
    }
    p
}

/// O que o merge entrega de um pedido com a F2 ligada: um host e um registro de serviço por bloco (na ordem de `RecPedido::blocos`).
#[derive(Clone, Debug, PartialEq)]
pub struct DadosF2 {
    pub layers: usize,
    pub hosts: Vec<Option<BlocoF2>>,
    pub tbs: Vec<Option<RecTb>>,
    /// Registros sem bloco do host correspondente (esperado 0).
    pub host_extra: usize,
    pub tb_extra: usize,
}

impl DadosF2 {
    /// Problemas que invalidam o pedido (I1 a I5; I3 invalida o pedido inteiro). A I6 e a I8 não entram aqui.
    pub fn problemas(&self, blocos: &[RecBloco]) -> Vec<String> {
        let mut p = vec![];
        if self.hosts.len() != blocos.len() {
            p.push(format!("{} blocos do host contra {} registros de F2", blocos.len(), self.hosts.len()));
            return p;
        }
        if self.host_extra > 0 {
            p.push(format!("{} registros de F2 sem bloco correspondente", self.host_extra));
        }
        for (i, (b, h)) in blocos.iter().zip(&self.hosts).enumerate() {
            let Some(h) = h else {
                p.push(format!("bloco {i}: sem registro de F2 do host"));
                continue;
            };
            if !h.i3(self.layers) {
                p.push(format!("bloco {i}: I3 (estrutura: {} operações, esperadas {})", h.ops.len(), n_ops_esperado(self.layers)));
                continue;
            }
            if !h.i1() {
                p.push(format!("bloco {i}: I1 (monotonia/contenção)"));
            }
            if !h.i2() {
                p.push(format!("bloco {i}: I2 (soma das operações mais R_borda contra t3 - t2)"));
            }
            if h.gen != b.gen || h.m != b.m || h.pos_ini != b.pos_ini {
                p.push(format!("bloco {i}: I4 (gen/m/pos_ini do F2 diferem do perfil-bloco)"));
            }
            if h.t2 != b.t[2] || h.t3 != b.t[3] || (h.soma_ops_ns() + h.r_borda_ns()).abs_diff(b.f(2)) > TOL_RELOGIO_NS {
                p.push(format!("bloco {i}: I4 (soma das operações mais R_borda contra F2 do registro por bloco)"));
            }
            if !h.i5(self.layers) {
                p.push(format!("bloco {i}: I5 (soma por camada)"));
            }
        }
        p
    }
    /// Problemas da coleta dos carimbos do serviço (I8): invalidam só a HI.
    pub fn problemas_hi(&self, blocos: &[RecBloco]) -> Vec<String> {
        let mut p = vec![];
        for i in 0..blocos.len() {
            match self.hosts.get(i).and_then(|h| h.as_ref()) {
                Some(h) => p.extend(i8_problemas(h, self.tbs.get(i).and_then(|t| t.as_ref()), self.layers).into_iter().map(|s| format!("bloco {i}: {s}"))),
                None => p.push(format!("bloco {i}: sem host para conferir os carimbos do serviço")),
            }
        }
        if self.tb_extra > 0 {
            p.push(format!("{} registros de serviço sem bloco correspondente", self.tb_extra));
        }
        p
    }
    /// Ressalva (I6): algum bloco com `R_borda`/F2 acima de 3%.
    pub fn ressalva_borda(&self) -> bool {
        self.hosts.iter().flatten().any(|h| h.ressalva_borda())
    }
}

/// Percentil por posto mais próximo (`ceil(p * n)`; definição a aprovar pelo Breno).
pub fn percentil_posto(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = ((p * s.len() as f64).ceil() as usize).clamp(1, s.len());
    Some(s[k - 1])
}

fn dist(v: &[f64]) -> Value {
    if v.is_empty() {
        return json!({"n": 0, "min": null, "mediana": null, "p90": null, "max": null});
    }
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    json!({"n": v.len(), "min": min, "mediana": mediana(v), "p90": percentil_posto(v, 0.9), "max": max})
}

/// Relatório descritivo da HI: distribuições (mínimo, mediana, p90, máximo, COM SINAL, em ns) e medianas por camada de `defasagem_obs`,
/// `defasagem`, `intervalo_obs`, `intervalo`, junto com a mediana das seis operações do host na mesma camada. Sem classe, sem limiar, sem regime
/// e sem causa: os números e os limites de interpretação.
pub fn relatorio_hi(amostras: &[(&BlocoF2, Vec<DefasL>)], layers: usize) -> Value {
    let todas = |f: &dyn Fn(&DefasL) -> Option<i64>| -> Vec<f64> { amostras.iter().flat_map(|(_, d)| d.iter().filter_map(f)).map(|x| x as f64).collect() };
    let por_camada = |f: &dyn Fn(&DefasL) -> Option<i64>| -> Vec<Value> {
        (0..layers as u32)
            .map(|l| {
                let v: Vec<f64> = amostras.iter().flat_map(|(_, d)| d.iter().filter(|x| x.camada == l).filter_map(f)).map(|x| x as f64).collect();
                json!({"camada": l, "n": v.len(), "mediana": mediana_opt(&v)})
            })
            .collect()
    };
    let host_camada: Vec<Value> = (0..layers)
        .map(|l| {
            let v: Vec<f64> = amostras.iter().map(|(h, _)| h.soma_camada_ns(l) as f64).collect();
            json!({"camada": l, "n": v.len(), "mediana_ns": mediana_opt(&v)})
        })
        .collect();
    json!({
        "tipo": "hi", "descritivo": true, "unidade": "ns, com sinal",
        "amostras": amostras.len(),
        "distribuicao": {
            "defasagem_obs": dist(&todas(&|d| Some(d.defasagem_obs))),
            "defasagem": dist(&todas(&|d| Some(d.defasagem))),
            "intervalo_obs": dist(&todas(&|d| d.intervalo_obs)),
            "intervalo": dist(&todas(&|d| d.intervalo)),
        },
        "mediana_por_camada": {
            "defasagem_obs": por_camada(&|d| Some(d.defasagem_obs)),
            "defasagem": por_camada(&|d| Some(d.defasagem)),
            "intervalo_obs": por_camada(&|d| d.intervalo_obs),
            "intervalo": por_camada(&|d| d.intervalo),
            "host_seis_operacoes": host_camada,
        },
        "limites": "defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico",
    })
}

/// O JSON de um pedido com a F2 ligada.
pub fn pedido_f2_json(p: &RecPedido) -> Option<Value> {
    let d = p.f2.as_ref()?;
    let blocos: Vec<Value> = p
        .blocos
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let h = d.hosts.get(i).and_then(|h| h.as_ref());
            let tb = d.tbs.get(i).and_then(|t| t.as_ref());
            let Some(h) = h else { return json!({"indice": i, "gen": b.gen, "sem_host": true}) };
            let defs = tb.map(|t| defasagens(h, t));
            json!({
                "indice": i, "gen": h.gen, "m": h.m, "pos_ini": h.pos_ini, "n_ready": h.n_ready,
                "t2_ns": h.t2, "t3_ns": h.t3, "entrada_ns": h.entrada, "saida_ns": h.saida,
                "f2_ns": h.f2_ns(), "r_borda_ns": h.r_borda_ns(), "soma_ops_ns": h.soma_ops_ns(),
                "ops": h.ops.iter().map(|o| json!([o.op, if o.camada == SEM_CAMADA { Value::Null } else { json!(o.camada) }, o.ini, o.fim])).collect::<Vec<_>>(),
                "tb": tb.map(|t| t.camadas.iter().map(|c| json!({"camada": c.camada, "gen": c.gen, "tb_obs_ns": c.tb_obs, "tb_ns": c.tb})).collect::<Vec<_>>()),
                "defasagens": defs.map(|v| v.iter().map(|x| json!({"camada": x.camada, "defasagem_obs_ns": x.defasagem_obs, "defasagem_ns": x.defasagem,
                                                                     "intervalo_obs_ns": x.intervalo_obs, "intervalo_ns": x.intervalo})).collect::<Vec<_>>()),
                "i1": h.i1(), "i2": h.i2(), "i3": h.i3(d.layers), "i5": h.i5(d.layers), "i6_ressalva_borda": h.ressalva_borda(),
            })
        })
        .collect();
    let problemas = d.problemas(&p.blocos);
    let problemas_hi = d.problemas_hi(&p.blocos);
    Some(json!({
        "modelo": p.modelo, "indice": p.indice, "aquecimento": p.aquecimento, "n": p.n, "plano_m": p.plano, "layers": d.layers,
        "blocos": blocos, "host_extra": d.host_extra, "tb_extra": d.tb_extra,
        "valido": problemas.is_empty(), "problemas": problemas,
        "hi_valida": problemas_hi.is_empty(), "problemas_hi": problemas_hi,
        "i6_ressalva_borda": d.ressalva_borda(),
    }))
}

/// O arquivo `perfil-f2.json` (esquema próprio, versão 1; só os pedidos com a F2 coletada).
pub fn perfil_f2_json(pedidos: &[RecPedido]) -> Value {
    json!({
        "tipo": "perfil-f2",
        "versao": 1,
        "variavel": VARIAVEL_F2,
        "exige": VARIAVEL,
        "bloco": BLOCO,
        "unidades": "carimbos e duracoes em ns desde a origem do processo; defasagens em ns com sinal",
        "nomes_op": NOMES_H,
        "tolerancias": {"relogio_ns": TOL_RELOGIO_NS, "cobertura_borda": LIM_COBERTURA_HOST},
        "pedidos": pedidos.iter().filter_map(pedido_f2_json).collect::<Vec<_>>(),
    })
}

// ---- Classificadores das hipóteses da F2 (seção 5 do plano; limiares são rascunho até o critério congelar) ----------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HipF2 {
    Ha,
    Hb,
    Hc,
    Hd,
    He,
    Hf,
    Hg,
    Hh,
    Hi,
}

/// As medidas já calculadas (por modelo, sobre blocos cheios elegíveis). `None` = a medida não existe.
#[derive(Clone, Copy, Debug, Default)]
pub struct MedidasF2 {
    /// (`H_PRE_LAYER` + `H_ROUTE` + `H_LAUNCH_HITS` + `H_ADD_Y`) / F2.
    pub ha: Option<f64>,
    /// `H_PRE_LAYER` / F2.
    pub hb_frac: Option<f64>,
    /// A soma de `H_PRE_LAYER` no bloco é maior que a de cada uma das outras operações por camada.
    pub hb_maior: Option<bool>,
    /// (`H_SEQ` + `H_WAIT_FLAG` + `H_SEQ_FIM` + `H_UPLOAD_TABELA` + `H_WAIT_EVENT` + `H_BEGIN_STEP`) / F2.
    pub hc: Option<f64>,
    /// Maior mediana por camada / menor mediana por camada (soma das seis operações).
    pub hd: Option<f64>,
    /// Parcela de F2 em chamadas acima de 10 vezes a mediana da mesma operação na mesma camada.
    pub he: Option<f64>,
    /// F2(bloco final com resto) / F2 mediano dos cheios (só descritiva).
    pub hf_razao: Option<f64>,
    /// ms por lançamento: só existe depois de fixar a contagem de kernels.
    pub hg_ms_por_lancamento: Option<f64>,
    /// F2 do bloco 0 / F2 mediano dos cheios.
    pub hh: Option<f64>,
    /// Há números de HI publicados (a HI é só descritiva).
    pub hi_dados: bool,
}

/// Faixa prevista de HF para o resto `r`: `r/8` com 25% de margem (descritiva, sem classe).
pub fn faixa_hf(r: usize) -> (f64, f64) {
    let c = r as f64 / BLOCO as f64;
    (c * 0.75, c * 1.25)
}

pub fn classifica_ha(x: Option<f64>) -> ClasseH {
    simples_ge(x, 0.80, 0.50)
}

/// HB: dentro se frac >= 0,40 e a soma de `H_PRE_LAYER` é a maior; fora se frac < 0,25; senão sem leitura.
pub fn classifica_hb(frac: Option<f64>, maior: Option<bool>) -> ClasseH {
    match (fin(frac), maior) {
        (None, _) => ClasseH::SemDado,
        (Some(f), _) if f < 0.25 => ClasseH::Fora,
        (Some(f), Some(true)) if f >= 0.40 => ClasseH::Dentro,
        (Some(_), Some(_)) => ClasseH::SemLeitura,
        (Some(_), None) => ClasseH::SemDado,
    }
}

pub fn classifica_hc(x: Option<f64>) -> ClasseH {
    simples_le(x, 0.05, 0.15)
}

/// HD: razão <= 1,5 dentro; > 3,0 fora; entre os dois sem leitura. Só uniformidade.
pub fn classifica_hd(x: Option<f64>) -> ClasseH {
    simples_le(x, 1.5, 3.0)
}

pub fn classifica_he(x: Option<f64>) -> ClasseH {
    simples_le(x, 0.05, 0.20)
}

/// HH: dentro se F2(bloco 0)/F2 mediano >= 1,5; sem limiar de fora (descritivo abaixo disso).
pub fn classifica_hh(x: Option<f64>) -> ClasseH {
    match fin(x) {
        None => ClasseH::SemDado,
        Some(v) if v >= 1.5 => ClasseH::Dentro,
        Some(_) => ClasseH::Descritivo,
    }
}

/// Classifica `h` dado o controle do modelo (a classe do B contra A). Controle que não permite = `SemProva`, antes de olhar qualquer número.
pub fn classifica_f2(h: HipF2, m: &MedidasF2, controle: &ClasseControle) -> ClasseH {
    if !controle_permite(controle) {
        return ClasseH::SemProva;
    }
    match h {
        HipF2::Ha => classifica_ha(m.ha),
        HipF2::Hb => classifica_hb(m.hb_frac, m.hb_maior),
        HipF2::Hc => classifica_hc(m.hc),
        HipF2::Hd => classifica_hd(m.hd),
        HipF2::He => classifica_he(m.he),
        HipF2::Hf => if fin(m.hf_razao).is_some() { ClasseH::Descritivo } else { ClasseH::SemDado },
        HipF2::Hg => if fin(m.hg_ms_por_lancamento).is_some() { ClasseH::Descritivo } else { ClasseH::SemDado },
        HipF2::Hh => classifica_hh(m.hh),
        HipF2::Hi => if m.hi_dados { ClasseH::Descritivo } else { ClasseH::SemDado },
    }
}

fn med(v: Vec<f64>) -> Option<f64> {
    mediana_opt(&v)
}

fn razao(a: Option<f64>, b: Option<f64>) -> Option<f64> {
    match (a, b) {
        (Some(x), Some(y)) if y > 0.0 => Some(x / y),
        _ => None,
    }
}

/// HE de UM pedido: a mediana de referência de cada grupo (op, camada, com SEM_CAMADA para as sete fixas) é tomada sobre os blocos elegíveis DO
/// PEDIDO; HE = soma das durações estritamente acima de 10x essa mediana / soma de F2 do pedido. Conjunto elegível: todas as operações medidas.
pub fn he_do_pedido(cheios: &[&BlocoF2]) -> Option<f64> {
    let mut grupos: std::collections::BTreeMap<(u8, u32), Vec<f64>> = std::collections::BTreeMap::new();
    let mut f2_total = 0f64;
    for b in cheios {
        f2_total += b.f2_ns() as f64;
        for o in &b.ops {
            grupos.entry((o.op, o.camada)).or_default().push(o.dur() as f64);
        }
    }
    let longas: f64 = grupos.values().filter_map(|d| mediana_opt(d).map(|md| d.iter().filter(|&&x| x > 10.0 * md).sum::<f64>())).sum();
    (f2_total > 0.0).then(|| longas / f2_total)
}

/// HE da célula: mediana entre os pedidos de [`he_do_pedido`] (cada pedido com a sua própria mediana de referência). Sem pedido, `None`.
pub fn he_da_celula(pedidos: &[Vec<&BlocoF2>]) -> Option<f64> {
    let v: Vec<f64> = pedidos.iter().filter_map(|p| he_do_pedido(p)).collect();
    mediana_opt(&v)
}

/// Medianas, entre os blocos de UM pedido, da soma de cada tipo de operação por camada, na ordem PRE, ROUTE, SEQ, LAUNCH, WAIT_FLAG, ADD_Y.
fn somas_por_tipo_do_pedido(cheios: &[&BlocoF2]) -> Option<[f64; 6]> {
    let mut r = [0f64; 6];
    for (i, &o) in OPS_CAMADA.iter().enumerate() {
        r[i] = med(cheios.iter().map(|b| b.soma_op_ns(o) as f64).collect())?;
    }
    Some(r)
}

/// As medidas da célula: HA a HE calculadas POR PEDIDO ([`medidas_f2_pedido`], cada pedido com as próprias medianas) e agregadas pela mediana
/// entre os pedidos da célula. HB: a comparação de maior soma usa a mediana entre pedidos dos valores por tipo de operação (não um booleano por
/// pedido). Nunca mistura os blocos de pedidos diferentes numa mediana só. Sem blocos, tudo `None`.
pub fn medidas_f2_celula(pedidos: &[Vec<&BlocoF2>], layers: usize) -> MedidasF2 {
    let por_pedido: Vec<MedidasF2> = pedidos.iter().filter(|p| !p.is_empty()).map(|p| medidas_f2_pedido(p, layers)).collect();
    let agrega = |f: &dyn Fn(&MedidasF2) -> Option<f64>| med(por_pedido.iter().filter_map(f).collect());
    let tipos: Vec<[f64; 6]> = pedidos.iter().filter(|p| !p.is_empty()).filter_map(|p| somas_por_tipo_do_pedido(p)).collect();
    let hb_maior = (!tipos.is_empty() && tipos.len() == por_pedido.len()).then(|| {
        let m: Vec<f64> = (0..6).map(|i| mediana(&tipos.iter().map(|t| t[i]).collect::<Vec<_>>())).collect();
        (1..6).all(|i| m[0] > m[i])
    });
    MedidasF2 { ha: agrega(&|m| m.ha), hb_frac: agrega(&|m| m.hb_frac), hb_maior, hc: agrega(&|m| m.hc), hd: agrega(&|m| m.hd), he: he_da_celula(pedidos), ..MedidasF2::default() }
}

/// Calcula as medidas de HA a HE de UM PEDIDO (seus blocos cheios elegíveis; razão de medianas, como no critério anterior); a célula usa
/// [`medidas_f2_celula`], que agrega por mediana entre pedidos. Não chame com blocos de vários pedidos juntos. HF, HG, HH e HI vêm de quem monta os conjuntos. Sem blocos, tudo `None`.
pub fn medidas_f2_pedido(cheios: &[&BlocoF2], layers: usize) -> MedidasF2 {
    if cheios.is_empty() {
        return MedidasF2::default();
    }
    let f2 = med(cheios.iter().map(|b| b.f2_ns() as f64).collect());
    let grupo = |ops: &[u8]| med(cheios.iter().map(|b| ops.iter().map(|&o| b.soma_op_ns(o)).sum::<u64>() as f64).collect());
    let ha = razao(grupo(&[H_PRE_LAYER, H_ROUTE, H_LAUNCH_HITS, H_ADD_Y]), f2);
    let hb_frac = razao(grupo(&[H_PRE_LAYER]), f2);
    // Soma por tipo de operação em cada bloco; a mediana entre os blocos decide quem é maior.
    let pre = grupo(&[H_PRE_LAYER]);
    let hb_maior = pre.map(|p| [H_ROUTE, H_SEQ, H_LAUNCH_HITS, H_WAIT_FLAG, H_ADD_Y].iter().all(|&o| grupo(&[o]).is_some_and(|x| p > x)));
    let hc = razao(grupo(&[H_SEQ, H_WAIT_FLAG, H_SEQ_FIM, H_UPLOAD_TABELA, H_WAIT_EVENT, H_BEGIN_STEP]), f2);
    let por_camada: Vec<f64> = (0..layers).filter_map(|l| med(cheios.iter().map(|b| b.soma_camada_ns(l) as f64).collect())).collect();
    let hd = (por_camada.len() == layers && layers > 0).then(|| {
        let mx = por_camada.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mn = por_camada.iter().copied().fold(f64::INFINITY, f64::min);
        (mn > 0.0).then(|| mx / mn)
    }).flatten();
    let he = he_do_pedido(cheios);
    MedidasF2 { ha, hb_frac, hb_maior, hc, hd, he, ..MedidasF2::default() }
}

// ---- Planejamento da janela: teto global, tetos de processo e corte de 5 para 3 (critério, pontos 7 e 8 da seção 10) -------------------------------

/// Teto global da janela por modelo (s), compartilhado entre todas as etapas; separado dos tetos dos processos.
pub const TETO_JANELA_S: u64 = 1200;
/// Folga única da projeção (s).
pub const FOLGA_S: u64 = 180;
/// Teto do smoke (s).
pub const TETO_SMOKE_S: u64 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CelulaF2 {
    Medio,
    Curto,
    Longo,
}

/// Teto de UM processo (s): 60 (1B) e 90 (3B) no médio e no curto; 75 (1B) e 120 (3B) no longo. Passar do teto = processo inválido, sem repetição.
pub fn teto_processo_s(modelo_3b: bool, c: CelulaF2) -> u64 {
    match (c, modelo_3b) {
        (CelulaF2::Longo, false) => 75,
        (CelulaF2::Longo, true) => 120,
        (_, false) => 60,
        (_, true) => 90,
    }
}

/// Estimativa conservadora de uma célula (um único processo, 1 aquecimento + N medidas): com N = 5 é o teto do processo; com N = 3 é o
/// teto x (1 + 3) / (1 + 5), arredondado para cima. Regra de rótulo: estimativa por N PROPORCIONAL ao número de pedidos, a confirmar na revisão;
/// nunca passa do teto do processo.
pub fn estimativa_celula_s(teto_s: u64, n: u32) -> u64 {
    match n {
        5 => teto_s,
        3 => (teto_s * 4).div_ceil(6),
        _ => teto_s,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlanoCelula {
    /// Roda com 5 medidas.
    Cinco,
    /// Corte: roda com 3 medidas.
    Tres,
    /// Nem com 3 cabe: a célula não roda e nada é repetido.
    NaoRodou,
}

/// Antes de iniciar uma célula: `restantes_s` são os tetos de processo de TODAS as células ainda não iniciadas, incluindo a que vai começar (a primeira).
/// Projeção `decorrido + soma + folga` contra o teto global. O corte é ÚNICO e PERMANENTE (`ja_cortado`: depois de um corte as células seguintes não
/// voltam a 5). Sem corte prévio e cabendo todas com N = 5: `Cinco`. Senão, cabendo todas com N = 3: `Tres`. Senão, o NÃO RODOU é decidido SÓ pela célula
/// atual com N = 3 (`decorrido + estimativa(atual, 3) + folga`): cabendo, `Tres`; não cabendo, `NaoRodou` (a célula não roda, nada é repetido).
pub fn planeja_celula(decorrido_s: u64, restantes_s: &[u64], folga_s: u64, teto_global_s: u64, ja_cortado: bool) -> PlanoCelula {
    let soma = |n: u32| -> u64 { restantes_s.iter().map(|&t| estimativa_celula_s(t, n)).sum() };
    if !ja_cortado && decorrido_s + soma(5) + folga_s <= teto_global_s {
        PlanoCelula::Cinco
    } else if decorrido_s + soma(3) + folga_s <= teto_global_s {
        PlanoCelula::Tres
    } else if restantes_s.first().is_some_and(|&t| decorrido_s + estimativa_celula_s(t, 3) + folga_s <= teto_global_s) {
        PlanoCelula::Tres
    } else {
        PlanoCelula::NaoRodou
    }
}

/// O controle de 9 processos nunca é cortado: cabe inteiro (smoke + nove tetos + folga dentro do teto global) ou o modelo fica incompleto/SEM PROVA.
pub fn controle_cabe(decorrido_s: u64, tetos_dos_9_s: &[u64], folga_s: u64, teto_global_s: u64) -> bool {
    tetos_dos_9_s.len() == ORDEM_CONTROLE_F2.len() && decorrido_s + tetos_dos_9_s.iter().sum::<u64>() + folga_s <= teto_global_s
}

// ---- Controle de três braços (3.5 do plano) -------------------------------------------------------------------------------------------------

/// D: as duas variáveis desligadas; A: só `LASTRO_PERFIL_BLOCO=1`; B: as duas ligadas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Braco3 {
    D,
    A,
    B,
}

/// Um processo do controle, na ordem em que rodou.
#[derive(Clone, Debug)]
pub struct ProcessoControleF2 {
    pub braco: Braco3,
    /// Mediana de `prefill_ms` dos pedidos medidos do processo.
    pub prefill_ms_mediana: f64,
    /// Mediana do `passo_ms` do decode do processo.
    pub passo_ms_mediana: f64,
    pub imprimiu_bloco_ligado: bool,
    pub imprimiu_f2_ligado: bool,
    pub produziu_bloco: bool,
    pub produziu_f2: bool,
    /// Terminou sem aborto e com os ids iguais aos da referência (I7).
    pub valido: bool,
}

/// Quadrado latino de 9 processos por modelo: D A B, B D A, A B D.
pub const ORDEM_CONTROLE_F2: [Braco3; 9] = [Braco3::D, Braco3::A, Braco3::B, Braco3::B, Braco3::D, Braco3::A, Braco3::A, Braco3::B, Braco3::D];

#[derive(Clone, Debug, PartialEq)]
pub struct DesvioF2 {
    /// (mediana das medianas do braço alvo - a da referência) / a da referência; com sinal.
    pub desvio: f64,
    /// (maior mediana do alvo - menor da referência) / menor da referência.
    pub pior_caso: f64,
    /// Desvio em ms por bloco (diferença das medianas de `prefill_ms` dividida pelos blocos do pedido).
    pub ms_por_bloco: f64,
    /// Desvio relativo do `passo_ms` do decode.
    pub desvio_passo: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResultadoControleF2 {
    /// Classe pela magnitude de |B contra A| (a medida principal); manda em tudo.
    pub classe: ClasseControle,
    /// B contra A.
    pub principal: Option<DesvioF2>,
    /// B contra D (medida de apoio): classe só descritiva, não decide.
    pub apoio: Option<DesvioF2>,
    pub classe_apoio: Option<ClasseControle>,
}

fn classe_por_desvio(d: f64) -> ClasseControle {
    match d.abs() {
        x if x <= 0.03 + EPS => ClasseControle::Comprovada,
        x if x <= 0.10 + EPS => ClasseControle::ComRessalva,
        _ => ClasseControle::SemProva,
    }
}

/// Classifica o controle de três braços de um modelo. Pura. `blocos_por_pedido` é o número de blocos totais executados no pedido do texto médio (22 em L100: 170 = 21 x 8 + 2): `ms_por_bloco` é a "diferença de prefill por bloco executado, incluindo primeiro e resto" (não isola o custo da F2).
pub fn classifica_controle_f2(procs: &[ProcessoControleF2], orcamento_ok: bool, blocos_por_pedido: usize) -> ResultadoControleF2 {
    let inc = |m: &str| ResultadoControleF2 { classe: ClasseControle::Incompleto(m.into()), principal: None, apoio: None, classe_apoio: None };
    if !orcamento_ok {
        return inc("orçamento insuficiente");
    }
    if procs.len() != ORDEM_CONTROLE_F2.len() {
        return inc("menos (ou mais) de 9 processos: mínimo de 3 válidos por braço");
    }
    if procs.iter().map(|p| p.braco).ne(ORDEM_CONTROLE_F2) {
        return inc("ordem diferente de D A B, B D A, A B D");
    }
    if procs.iter().any(|p| !p.valido || !p.prefill_ms_mediana.is_finite() || p.prefill_ms_mediana <= 0.0 || !p.passo_ms_mediana.is_finite() || p.passo_ms_mediana <= 0.0) {
        return inc("processo de controle abortado ou inválido");
    }
    for p in procs {
        let ok = match p.braco {
            Braco3::D => !p.imprimiu_bloco_ligado && !p.imprimiu_f2_ligado && !p.produziu_bloco && !p.produziu_f2,
            Braco3::A => p.imprimiu_bloco_ligado && !p.imprimiu_f2_ligado && p.produziu_bloco && !p.produziu_f2,
            Braco3::B => p.imprimiu_bloco_ligado && p.imprimiu_f2_ligado && p.produziu_bloco && p.produziu_f2,
        };
        if !ok {
            return inc(&format!("braço {:?} com linhas de estado ou arquivos de perfil que não são os do braço", p.braco));
        }
    }
    let de = |b: Braco3, f: &dyn Fn(&ProcessoControleF2) -> f64| -> Vec<f64> { procs.iter().filter(|p| p.braco == b).map(f).collect() };
    let pre = |p: &ProcessoControleF2| p.prefill_ms_mediana;
    let passo = |p: &ProcessoControleF2| p.passo_ms_mediana;
    let desvio = |alvo: Braco3, ref_: Braco3| -> DesvioF2 {
        let (a, r) = (de(alvo, &pre), de(ref_, &pre));
        let (ma, mr) = (mediana(&a), mediana(&r));
        let min_r = r.iter().copied().fold(f64::INFINITY, f64::min);
        let max_a = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (pa, pr) = (mediana(&de(alvo, &passo)), mediana(&de(ref_, &passo)));
        DesvioF2 { desvio: (ma - mr) / mr, pior_caso: (max_a - min_r) / min_r, ms_por_bloco: (ma - mr) / blocos_por_pedido.max(1) as f64, desvio_passo: (pa - pr) / pr }
    };
    let (principal, apoio) = (desvio(Braco3::B, Braco3::A), desvio(Braco3::B, Braco3::D));
    ResultadoControleF2 { classe: classe_por_desvio(principal.desvio), classe_apoio: Some(classe_por_desvio(apoio.desvio)), principal: Some(principal), apoio: Some(apoio) }
}

// ---- F2 nível B: o pré-roteador por dentro (plano `plano-f2-nivel-b-pre-roteador-20261111.md`, versão 2, aprovada na rodada 158) -----------
//
// Descritivo. Ligado por `LASTRO_PERFIL_F2B=1` (só o valor exato `1`) e só junto com `LASTRO_PERFIL_BLOCO=1` e `LASTRO_PERFIL_F2=1`. Desligado não
// há acumulador, relógio nem gravação: `Engine::f2b_acc` é `None`, o motor chama `pre_layer` e as variantes internas de `attn_block_at`,
// `attn_block` e `attention` recebem um `Option` vazio (a promessa é só essa: um `Option` vazio por ponto de medida). Saída em `perfil-f2b.json`
// (esquema próprio, versão 1); `perfil-bloco.json` v2 e `perfil-f2.json` não mudam.
//
// Itens decididos pelo Breno em 11/11 12:03 (HB3 0,60/0,40; braços D/A/B2; rótulo "tempo da chamada de envio de trabalho"), mantidos parametrizados: limiares de HB3 ([`LimiaresHb3`]), braços do
// controle ([`ControleB`]) e o rótulo de `B_LANCA` no relatório ([`RotuloLanca`]).

/// Nome da variável que liga a F2 nível B.
pub const VARIAVEL_F2B: &str = "LASTRO_PERFIL_F2B";

/// Só o valor exato `1` liga (mesmo contrato de [`interpreta`]).
pub fn interpreta_f2b(v: Option<&str>) -> bool {
    interpreta(v)
}

/// Lê `LASTRO_PERFIL_F2B` do ambiente (não UTF-8 conta como ausente).
pub fn f2b_do_ambiente() -> bool {
    interpreta_f2b(std::env::var(VARIAVEL_F2B).ok().as_deref())
}

/// A F2B só vale com o instrumento de bloco e a F2 ligados.
pub fn f2b_efetivo(bloco: bool, f2: bool, f2b: bool) -> bool {
    bloco && f2 && f2b
}

/// A linha de estado: `perfil-f2b: ligado`, `perfil-f2b: desligado` ou, pedida sem as duas outras, `perfil-f2b: desligado (exige perfil-bloco e perfil-f2)`.
pub fn linha_estado_f2b(bloco: bool, f2: bool, f2b: bool) -> &'static str {
    if f2b_efetivo(bloco, f2, f2b) {
        "perfil-f2b: ligado"
    } else if f2b {
        "perfil-f2b: desligado (exige perfil-bloco e perfil-f2)"
    } else {
        "perfil-f2b: desligado"
    }
}

/// Operações do pré-roteador (códigos estáveis do JSON).
pub const B_LOCK: u8 = 0;
pub const B_SETUP: u8 = 1;
pub const B_GENERICS: u8 = 2;
pub const B_ATTN_PREP: u8 = 3;
pub const B_LANCA: u8 = 4;
pub const B_DTOD: u8 = 5;
pub const NOMES_B: [&str; 6] = ["B_LOCK", "B_SETUP", "B_GENERICS", "B_ATTN_PREP", "B_LANCA", "B_DTOD"];
/// Marca de "sem posição" (só `B_LOCK`, que é por camada).
pub const SEM_POS: u16 = u16::MAX;

/// Registros esperados por camada para a operação `op` num bloco de `m` posições (IB3): 1, 2m, m, m, 11m, 6m.
pub fn contagem_esperada_b(op: u8, m: usize) -> usize {
    match op {
        B_LOCK => 1,
        B_SETUP => 2 * m,
        B_GENERICS | B_ATTN_PREP => m,
        B_LANCA => 11 * m,
        B_DTOD => 6 * m,
        _ => 0,
    }
}

/// A ordem de emissão dos 21 registros de uma posição em `attn_block_at_b`: SETUP, GENERICS, 4 LANCA, 2 DTOD, 2 LANCA, 3 DTOD, ATTN_PREP, 2 LANCA,
/// 1 DTOD, 3 LANCA, SETUP (2/1/1/11/6).
pub const ORDEM_POS_B: [u8; 21] = [
    B_SETUP, B_GENERICS, B_LANCA, B_LANCA, B_LANCA, B_LANCA, B_DTOD, B_DTOD, B_LANCA, B_LANCA, B_DTOD, B_DTOD, B_DTOD, B_ATTN_PREP, B_LANCA, B_LANCA, B_DTOD,
    B_LANCA, B_LANCA, B_LANCA, B_SETUP,
];

/// Capacidade fixa do acumulador: `layers x (21 x max_m + 1)`, reservada na criação do motor (24 bytes por registro).
pub fn capacidade_f2b(layers: usize, max_m: usize) -> usize {
    layers * (21 * max_m + 1)
}

/// Um registro bruto: operação, camada, posição (ou [`SEM_POS`]) e o intervalo [`ini`, `fim`] em ns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegB {
    pub ini: u64,
    pub fim: u64,
    pub camada: u16,
    pub pos: u16,
    pub op: u8,
}

impl RegB {
    pub fn dur(&self) -> u64 {
        self.fim.saturating_sub(self.ini)
    }
}

/// Acumulador do nível B (escritor único: a thread do motor). Capacidade fixa; ao encher, o registro é contado em `perdidos` (nunca realoca) e a
/// IB3 invalida o pedido.
#[derive(Debug)]
pub struct AccB {
    pub regs: Vec<RegB>,
    /// Camada e posição do contexto em andamento (fixadas por quem chama antes de cada trecho medido).
    pub camada: u16,
    pub pos: u16,
    pub perdidos: usize,
}

impl AccB {
    pub fn com_capacidade(layers: usize, max_m: usize) -> Self {
        Self { regs: Vec::with_capacity(capacidade_f2b(layers, max_m)), camada: 0, pos: SEM_POS, perdidos: 0 }
    }
    pub fn capacidade(&self) -> usize {
        self.regs.capacity()
    }
    /// Zera para o bloco seguinte (mantém a capacidade).
    pub fn reinicia(&mut self) {
        self.regs.clear();
        self.perdidos = 0;
        self.camada = 0;
        self.pos = SEM_POS;
    }
    pub fn contexto(&mut self, camada: u16, pos: u16) {
        self.camada = camada;
        self.pos = pos;
    }
    /// Fecha um registro no instante atual, com o início `ini` tomado antes.
    #[inline]
    pub fn reg(&mut self, op: u8, ini: u64) {
        let fim = agora_ns();
        if self.regs.len() < self.regs.capacity() {
            self.regs.push(RegB { ini, fim, camada: self.camada, pos: self.pos, op });
        } else {
            self.perdidos += 1;
        }
    }
    /// O registro do bloco (copia os registros: aloca, fora de F2).
    pub fn para_bloco(&self, gen: u32, m: usize, pos_ini: u64) -> BlocoB {
        BlocoB { gen, m, pos_ini, regs: self.regs.clone(), perdidos: self.perdidos }
    }
}

/// Início de um trecho medido: lê o relógio só com o acumulador presente (desligado: `Option` vazio, devolve 0 sem relógio).
#[inline]
pub fn b_ini(a: &Option<&mut AccB>) -> u64 {
    if a.is_some() {
        agora_ns()
    } else {
        0
    }
}

/// Fim de um trecho medido: grava só com o acumulador presente.
#[inline]
pub fn b_fim(a: &mut Option<&mut AccB>, op: u8, ini: u64) {
    if let Some(x) = a.as_deref_mut() {
        x.reg(op, ini);
    }
}

/// Os registros brutos de um bloco.
#[derive(Clone, Debug, PartialEq)]
pub struct BlocoB {
    pub gen: u32,
    pub m: usize,
    pub pos_ini: u64,
    pub regs: Vec<RegB>,
    pub perdidos: usize,
}

impl BlocoB {
    pub fn n_op_camada(&self, l: usize, op: u8) -> usize {
        self.regs.iter().filter(|r| r.camada as usize == l && r.op == op).count()
    }
    pub fn soma_op_camada_ns(&self, l: usize, op: u8) -> u64 {
        self.regs.iter().filter(|r| r.camada as usize == l && r.op == op).map(RegB::dur).sum()
    }
    pub fn soma_camada_ns(&self, l: usize) -> u64 {
        self.regs.iter().filter(|r| r.camada as usize == l).map(RegB::dur).sum()
    }
    /// `B_RESIDUO` da camada `l`: `H_PRE_LAYER` menos a soma dos intervalos medidos, com sinal. Nunca ajustado.
    pub fn residuo_camada_ns(&self, l: usize, host: &BlocoF2) -> Option<i64> {
        let h = host.ops.iter().find(|o| o.op == H_PRE_LAYER && o.camada as usize == l)?;
        Some(h.dur() as i64 - self.soma_camada_ns(l) as i64)
    }
    /// IB1: registros em ordem e sem sobreposição; primeiro início e último fim de cada camada dentro do `H_PRE_LAYER` da mesma camada (1 µs).
    pub fn ib1(&self, host: &BlocoF2) -> bool {
        if self.regs.is_empty() {
            return false;
        }
        let mut prev = 0u64;
        for r in &self.regs {
            if r.fim < r.ini || r.ini < prev {
                return false;
            }
            prev = r.fim;
        }
        let mut camadas: Vec<u16> = self.regs.iter().map(|r| r.camada).collect();
        camadas.dedup();
        camadas.into_iter().all(|l| {
            let Some(h) = host.ops.iter().find(|o| o.op == H_PRE_LAYER && o.camada == l as u32) else { return false };
            let da_camada = || self.regs.iter().filter(move |r| r.camada == l);
            let (Some(p), Some(u)) = (da_camada().next(), da_camada().last()) else { return false };
            p.ini + TOL_RELOGIO_NS >= h.ini && u.fim <= h.fim + TOL_RELOGIO_NS
        })
    }
    /// IB2: soma dos intervalos + `B_RESIDUO` = `H_PRE_LAYER` (identidade por construção; vale como cobertura e excesso): resíduo < -1 µs invalida.
    pub fn ib2(&self, host: &BlocoF2, layers: usize) -> bool {
        (0..layers).all(|l| self.residuo_camada_ns(l, host).is_some_and(|r| r >= -(TOL_RELOGIO_NS as i64)))
    }
    /// IB3: a sequência esperada, na ordem de emissão: por camada (0..layers-1, em ordem) um `B_LOCK` com `SEM_POS` e, para cada posição `0..m-1`
    /// em ordem, os 21 registros de [`ORDEM_POS_B`] (SETUP 2, GENERICS 1, ATTN_PREP 1, LANCA 11, DTOD 6) com a camada e a posição certas. Nenhum
    /// registro perdido, nenhum a mais ou a menos. Troca de rótulos que preserve só as contagens não passa.
    pub fn ib3(&self, layers: usize) -> bool {
        if self.perdidos > 0 || self.m == 0 || self.regs.len() != layers * (21 * self.m + 1) {
            return false;
        }
        let mut it = self.regs.iter();
        for l in 0..layers {
            let ok = |r: Option<&RegB>, op: u8, pos: u16| r.is_some_and(|r| r.op == op && r.camada as usize == l && r.pos == pos);
            if !ok(it.next(), B_LOCK, SEM_POS) {
                return false;
            }
            for p in 0..self.m {
                for op in ORDEM_POS_B {
                    if !ok(it.next(), op, p as u16) {
                        return false;
                    }
                }
            }
        }
        it.next().is_none()
    }
    /// IB5: soma, com sinal, dos resíduos por camada (inclui os negativos de até 1 µs que a IB2 aceita) sobre a soma de `H_PRE_LAYER`; 0 sem `H_PRE_LAYER`.
    pub fn razao_residuo(&self, host: &BlocoF2, layers: usize) -> f64 {
        let (mut res, mut tot) = (0i64, 0u64);
        for l in 0..layers {
            if let Some(r) = self.residuo_camada_ns(l, host) {
                res += r;
            }
            tot += host.ops.iter().filter(|o| o.op == H_PRE_LAYER && o.camada as usize == l).map(OpH::dur).sum::<u64>();
        }
        if tot == 0 {
            0.0
        } else {
            res as f64 / tot as f64
        }
    }
}

/// O que o merge entrega de um pedido com a F2B ligada: um bloco bruto por bloco do pedido (na ordem de `RecPedido::blocos`).
#[derive(Clone, Debug, PartialEq)]
pub struct DadosF2B {
    pub layers: usize,
    pub blocos: Vec<Option<BlocoB>>,
    /// Registros sem bloco correspondente (esperado 0).
    pub extra: usize,
}

impl DadosF2B {
    /// Problemas que invalidam o pedido (IB1 a IB4; a IB3 invalida o pedido inteiro). A IB5 é só ressalva. Exige a F2 do mesmo pedido.
    pub fn problemas(&self, blocos: &[RecBloco], f2: Option<&DadosF2>) -> Vec<String> {
        let mut p = vec![];
        let Some(f2) = f2 else {
            p.push("F2B sem a F2 do host no mesmo pedido".into());
            return p;
        };
        if self.blocos.len() != blocos.len() {
            p.push(format!("{} blocos do pedido contra {} registros de F2B", blocos.len(), self.blocos.len()));
            return p;
        }
        if self.extra > 0 {
            p.push(format!("{} registros de F2B sem bloco correspondente", self.extra));
        }
        for (i, (b, bb)) in blocos.iter().zip(&self.blocos).enumerate() {
            let Some(bb) = bb else {
                p.push(format!("bloco {i}: sem registro de F2B"));
                continue;
            };
            let Some(h) = f2.hosts.get(i).and_then(|h| h.as_ref()) else {
                p.push(format!("bloco {i}: sem registro de F2 do host para conferir a F2B"));
                continue;
            };
            if !bb.ib3(self.layers) {
                p.push(format!("bloco {i}: IB3 (contagens: {} registros, {} perdidos, esperados {})", bb.regs.len(), bb.perdidos, self.layers * (21 * bb.m + 1)));
                continue;
            }
            if !bb.ib1(h) {
                p.push(format!("bloco {i}: IB1 (ordem, sobreposição ou contenção em H_PRE_LAYER)"));
            }
            if !bb.ib2(h, self.layers) {
                p.push(format!("bloco {i}: IB2 (soma dos intervalos maior que H_PRE_LAYER)"));
            }
            if bb.gen != h.gen || bb.m != h.m || bb.pos_ini != h.pos_ini || bb.gen != b.gen || bb.m != b.m || bb.pos_ini != b.pos_ini {
                p.push(format!("bloco {i}: IB4 (gen/m/pos_ini da F2B diferem da F2 ou do perfil-bloco)"));
            }
            if !h.i2() {
                p.push(format!("bloco {i}: IB4 (soma de H_PRE_LAYER e das demais operações mais R_borda contra t3 - t2)"));
            }
        }
        p
    }
    /// Ressalva (IB5): algum bloco com `B_RESIDUO` / `H_PRE_LAYER` acima de 3%.
    pub fn ressalva_residuo(&self, f2: Option<&DadosF2>) -> bool {
        let Some(f2) = f2 else { return false };
        self.blocos.iter().zip(&f2.hosts).any(|(b, h)| match (b, h) {
            (Some(b), Some(h)) => b.razao_residuo(h, self.layers) > LIM_COBERTURA_HOST,
            _ => false,
        })
    }
}

/// Como o relatório descreve `B_LANCA`. Decisão do Breno (12:03): "tempo da chamada de envio de trabalho", sem isolar `cuLaunchKernel`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotuloLanca {
    /// Valor do plano.
    SoAsyncOn,
    /// Alternativa: cita `cuLaunchKernel` como parte da expressão, sem isolá-lo.
    ComCuLaunchKernel,
}

impl RotuloLanca {
    pub fn texto(self) -> &'static str {
        match self {
            RotuloLanca::SoAsyncOn => "tempo da chamada de envio de trabalho (a expressão async_on)",
            RotuloLanca::ComCuLaunchKernel => "tempo da expressão async_on (a chamada a cuLaunchKernel da dependência externa está dentro dela e não é isolada)",
        }
    }
}

/// O JSON de um pedido com a F2B ligada.
pub fn pedido_f2b_json(p: &RecPedido, rotulo: RotuloLanca) -> Option<Value> {
    let d = p.f2b.as_ref()?;
    let f2 = p.f2.as_ref();
    let blocos: Vec<Value> = p
        .blocos
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let bb = d.blocos.get(i).and_then(|x| x.as_ref());
            let Some(bb) = bb else { return json!({"indice": i, "gen": b.gen, "sem_registro": true}) };
            let host = f2.and_then(|f| f.hosts.get(i)).and_then(|h| h.as_ref());
            json!({
                "indice": i, "gen": bb.gen, "m": bb.m, "pos_ini": bb.pos_ini, "perdidos": bb.perdidos,
                "regs": bb.regs.iter().map(|r| json!([r.op, r.camada, if r.pos == SEM_POS { Value::Null } else { json!(r.pos) }, r.ini, r.fim])).collect::<Vec<_>>(),
                "residuo_camada_ns": host.map(|h| (0..d.layers).map(|l| bb.residuo_camada_ns(l, h)).collect::<Vec<_>>()),
                "ib1": host.map(|h| bb.ib1(h)), "ib2": host.map(|h| bb.ib2(h, d.layers)), "ib3": bb.ib3(d.layers),
                "residuo_sobre_h_pre_layer": host.map(|h| bb.razao_residuo(h, d.layers)),
            })
        })
        .collect();
    let problemas = d.problemas(&p.blocos, f2);
    Some(json!({
        "modelo": p.modelo, "indice": p.indice, "aquecimento": p.aquecimento, "n": p.n, "plano_m": p.plano, "layers": d.layers,
        "blocos": blocos, "extra": d.extra, "valido": problemas.is_empty(), "problemas": problemas,
        "ib5_ressalva_residuo": d.ressalva_residuo(f2),
        "b_lanca_rotulo": rotulo.texto(),
    }))
}

/// O arquivo `perfil-f2b.json` com o rótulo de `B_LANCA` escolhido (item c pendente).
pub fn perfil_f2b_json_com(pedidos: &[RecPedido], rotulo: RotuloLanca) -> Value {
    json!({
        "tipo": "perfil-f2b",
        "versao": 1,
        "variavel": VARIAVEL_F2B,
        "exige": [VARIAVEL, VARIAVEL_F2],
        "bloco": BLOCO,
        "unidades": "carimbos e duracoes em ns desde a origem do processo",
        "nomes_op": NOMES_B,
        "campos_reg": ["op", "camada", "pos", "ini_ns", "fim_ns"],
        "contagem_por_camada": "B_LOCK 1, B_SETUP 2m, B_GENERICS m, B_ATTN_PREP m, B_LANCA 11m, B_DTOD 6m",
        "b_lanca_rotulo": rotulo.texto(),
        "tolerancias": {"relogio_ns": TOL_RELOGIO_NS, "cobertura_residuo": LIM_COBERTURA_HOST},
        "pedidos": pedidos.iter().filter_map(|p| pedido_f2b_json(p, rotulo)).collect::<Vec<_>>(),
    })
}

/// O arquivo `perfil-f2b.json` com o rótulo do plano.
pub fn perfil_f2b_json(pedidos: &[RecPedido]) -> Value {
    perfil_f2b_json_com(pedidos, RotuloLanca::SoAsyncOn)
}

// ---- Classificadores das hipóteses do nível B -----------------------------------------------------------------------------------------------

/// Limiares de HB3 (decididos pelo Breno em 12:03: 0,60 e 0,40). Dentro se parcela >= `dentro`; fora se < `fora`; no meio, sem leitura.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LimiaresHb3 {
    pub dentro: f64,
    pub fora: f64,
}

/// Rascunho do plano (seção 7): 0,60 e 0,40. Só o padrão do código, até o Breno responder.
pub const LIMIARES_HB3_PLANO: LimiaresHb3 = LimiaresHb3 { dentro: 0.60, fora: 0.40 };

pub fn classifica_hb3(x: Option<f64>, lim: &LimiaresHb3) -> ClasseH {
    simples_ge(x, lim.dentro, lim.fora)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HipB {
    Hb1,
    Hb2,
    Hb3,
    Hb4,
    Hb5,
    Hb6,
}

/// Medidas descritivas do nível B (parcelas e uniformidade), por pedido ou por célula.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MedidasF2B {
    pub hb1: Option<f64>,
    pub hb2: Option<f64>,
    pub hb3: Option<f64>,
    pub hb4: Option<f64>,
    /// Mediana, entre as camadas, de (p90 / mediana) de `H_PRE_LAYER` por camada.
    pub hb5_p90_sobre_mediana: Option<f64>,
    /// Maior sobre menor mediana por camada de `H_PRE_LAYER`.
    pub hb5_dispersao_camadas: Option<f64>,
    /// Mediana de `H_PRE_LAYER` por camada (ns), para HB5, HB6 e para o relatório ao lado do B de 11:00.
    pub h_pre_layer_mediana_camada_ns: Vec<f64>,
}

/// Parcelas por pedido: soma, entre as camadas, da mediana por camada (entre os blocos) da soma do grupo, sobre a soma, entre as camadas, da mediana
/// de `H_PRE_LAYER`. `cheios` são os blocos cheios elegíveis de UM pedido (bloco de F2B, bloco de F2 do host). Sem blocos, tudo `None`.
pub fn medidas_f2b_pedido(cheios: &[(&BlocoB, &BlocoF2)], layers: usize) -> MedidasF2B {
    if cheios.is_empty() || layers == 0 {
        return MedidasF2B::default();
    }
    let soma_med = |f: &dyn Fn(&BlocoB, &BlocoF2, usize) -> f64| -> f64 {
        (0..layers).map(|l| mediana(&cheios.iter().map(|(b, h)| f(b, h, l)).collect::<Vec<_>>())).sum()
    };
    let pre = |_: &BlocoB, h: &BlocoF2, l: usize| h.ops.iter().filter(|o| o.op == H_PRE_LAYER && o.camada as usize == l).map(OpH::dur).sum::<u64>() as f64;
    let den = soma_med(&pre);
    let grupo = |ops: &'static [u8]| soma_med(&|b: &BlocoB, _: &BlocoF2, l: usize| ops.iter().map(|&o| b.soma_op_camada_ns(l, o)).sum::<u64>() as f64);
    let parc = |ops: &'static [u8]| razao(Some(grupo(ops)), Some(den));
    let por_camada: Vec<Vec<f64>> = (0..layers).map(|l| cheios.iter().map(|(b, h)| pre(b, h, l)).collect()).collect();
    let med_camada: Vec<f64> = por_camada.iter().map(|v| mediana(v)).collect();
    let p90_sobre_med: Vec<f64> = por_camada.iter().zip(&med_camada).filter(|(_, m)| **m > 0.0).filter_map(|(v, m)| percentil_posto(v, 0.9).map(|p| p / m)).collect();
    let (mx, mn) = (med_camada.iter().copied().fold(f64::NEG_INFINITY, f64::max), med_camada.iter().copied().fold(f64::INFINITY, f64::min));
    MedidasF2B {
        hb1: parc(&[B_LOCK]),
        hb2: parc(&[B_SETUP, B_GENERICS, B_ATTN_PREP]),
        hb3: parc(&[B_LANCA]),
        hb4: parc(&[B_DTOD]),
        hb5_p90_sobre_mediana: med(p90_sobre_med),
        hb5_dispersao_camadas: if mn > 0.0 { Some(mx / mn) } else { None },
        h_pre_layer_mediana_camada_ns: med_camada,
    }
}

/// A célula: cada medida por pedido e depois a mediana entre os pedidos (nunca blocos de pedidos diferentes numa mediana só).
pub fn medidas_f2b_celula(pedidos: &[Vec<(&BlocoB, &BlocoF2)>], layers: usize) -> MedidasF2B {
    let por: Vec<MedidasF2B> = pedidos.iter().filter(|p| !p.is_empty()).map(|p| medidas_f2b_pedido(p, layers)).collect();
    let agrega = |f: &dyn Fn(&MedidasF2B) -> Option<f64>| med(por.iter().filter_map(f).collect());
    let camadas: Vec<f64> = (0..layers)
        .filter_map(|l| {
            let v: Vec<f64> = por.iter().filter_map(|m| m.h_pre_layer_mediana_camada_ns.get(l).copied()).collect();
            (!v.is_empty()).then(|| mediana(&v))
        })
        .collect();
    MedidasF2B {
        hb1: agrega(&|m| m.hb1),
        hb2: agrega(&|m| m.hb2),
        hb3: agrega(&|m| m.hb3),
        hb4: agrega(&|m| m.hb4),
        hb5_p90_sobre_mediana: agrega(&|m| m.hb5_p90_sobre_mediana),
        hb5_dispersao_camadas: agrega(&|m| m.hb5_dispersao_camadas),
        h_pre_layer_mediana_camada_ns: camadas,
    }
}

/// Classifica `h` dado o controle do modelo. Controle que não permite = `SemProva`, antes de olhar qualquer número. HB1, HB2, HB4, HB5 e HB6 são
/// descritivas (sem dado se a medida faltar); HB3 usa os limiares recebidos (item a pendente).
pub fn classifica_f2b(h: HipB, m: &MedidasF2B, controle: &ClasseControle, lim: &LimiaresHb3) -> ClasseH {
    if !controle_permite(controle) {
        return ClasseH::SemProva;
    }
    let desc = |x: Option<f64>| if fin(x).is_some() { ClasseH::Descritivo } else { ClasseH::SemDado };
    match h {
        HipB::Hb1 => desc(m.hb1),
        HipB::Hb2 => desc(m.hb2),
        HipB::Hb3 => classifica_hb3(m.hb3, lim),
        HipB::Hb4 => desc(m.hb4),
        HipB::Hb5 => if fin(m.hb5_p90_sobre_mediana).is_some() && fin(m.hb5_dispersao_camadas).is_some() { ClasseH::Descritivo } else { ClasseH::SemDado },
        HipB::Hb6 => if m.h_pre_layer_mediana_camada_ns.is_empty() { ClasseH::SemDado } else { ClasseH::Descritivo },
    }
}

// ---- Controle do custo do instrumento do nível B (item b pendente: braços parametrizados) -------------------------------------------------

/// D: nenhuma variável; A: só bloco; B1: bloco e F2 (F2B desligada); B2: as três.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracoB {
    D,
    A,
    B1,
    B2,
}

/// Um processo do controle, na ordem em que rodou.
#[derive(Clone, Debug)]
pub struct ProcessoControleB {
    pub braco: BracoB,
    pub prefill_ms_mediana: f64,
    pub passo_ms_mediana: f64,
    pub imprimiu_bloco_ligado: bool,
    pub imprimiu_f2_ligado: bool,
    pub imprimiu_f2b_ligado: bool,
    pub produziu_bloco: bool,
    pub produziu_f2: bool,
    pub produziu_f2b: bool,
    /// Terminou sem aborto e com os ids iguais aos da referência (I7).
    pub valido: bool,
}

/// A configuração do controle (decidido: D/A/B2, `CONTROLE_B_PLANO`; as outras duas ficam sem uso). `ordem` é a sequência de processos; `apoio` é o braço da medida de apoio (B2 contra ele).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ControleB {
    pub nome: &'static str,
    pub ordem: &'static [BracoB],
    pub apoio: Option<BracoB>,
}

use BracoB::{A as BA, B1, B2, D as BD};
/// Plano: D, A e B2, quadrado latino de 9 (D A B2, B2 D A, A B2 D). Padrão do código.
pub const CONTROLE_B_PLANO: ControleB = ControleB { nome: "D/A/B2", ordem: &[BD, BA, B2, B2, BD, BA, BA, B2, BD], apoio: Some(BD) };
/// Alternativa 1: só A e B2 (6 processos, 3 por braço; a ordem é derivação do critério, também pendente).
pub const CONTROLE_B_A_B2: ControleB = ControleB { nome: "A/B2", ordem: &[BA, B2, B2, BA, BA, B2], apoio: None };
/// Alternativa 2: F2=1/F2B=0 (B1) no lugar do D, na mesma ordem do plano.
pub const CONTROLE_B_B1_A_B2: ControleB = ControleB { nome: "B1/A/B2", ordem: &[B1, BA, B2, B2, B1, BA, BA, B2, B1], apoio: Some(B1) };

#[derive(Clone, Debug, PartialEq)]
pub struct ResultadoControleB {
    /// Classe pela magnitude de |B2 contra A| (a medida principal); manda em tudo.
    pub classe: ClasseControle,
    pub principal: Option<DesvioF2>,
    /// B2 contra o braço de apoio (D ou B1): classe só descritiva.
    pub apoio: Option<DesvioF2>,
    pub classe_apoio: Option<ClasseControle>,
}

/// Classifica o controle de um modelo. Pura. Mesmas regras do nível A (<= 3% comprovada, <= 10% com ressalva, > 10% SEM PROVA), com o B2 no lugar do B.
pub fn classifica_controle_f2b(cfg: &ControleB, procs: &[ProcessoControleB], orcamento_ok: bool, blocos_por_pedido: usize) -> ResultadoControleB {
    let inc = |m: &str| ResultadoControleB { classe: ClasseControle::Incompleto(m.into()), principal: None, apoio: None, classe_apoio: None };
    if !orcamento_ok {
        return inc("orçamento insuficiente");
    }
    if procs.len() != cfg.ordem.len() {
        return inc("número de processos diferente do da ordem do controle: mínimo de 3 válidos por braço");
    }
    if procs.iter().map(|p| p.braco).ne(cfg.ordem.iter().copied()) {
        return inc("ordem diferente da do controle");
    }
    if procs.iter().any(|p| !p.valido || !p.prefill_ms_mediana.is_finite() || p.prefill_ms_mediana <= 0.0 || !p.passo_ms_mediana.is_finite() || p.passo_ms_mediana <= 0.0) {
        return inc("processo de controle abortado ou inválido");
    }
    for p in procs {
        let (e, i) = match p.braco {
            BracoB::D => ((false, false, false), (false, false, false)),
            BracoB::A => ((true, false, false), (true, false, false)),
            BracoB::B1 => ((true, true, false), (true, true, false)),
            BracoB::B2 => ((true, true, true), (true, true, true)),
        };
        if (p.imprimiu_bloco_ligado, p.imprimiu_f2_ligado, p.imprimiu_f2b_ligado) != e || (p.produziu_bloco, p.produziu_f2, p.produziu_f2b) != i {
            return inc(&format!("braço {:?} com linhas de estado ou arquivos de perfil que não são os do braço", p.braco));
        }
    }
    let de = |b: BracoB, f: &dyn Fn(&ProcessoControleB) -> f64| -> Vec<f64> { procs.iter().filter(|p| p.braco == b).map(f).collect() };
    let pre = |p: &ProcessoControleB| p.prefill_ms_mediana;
    let passo = |p: &ProcessoControleB| p.passo_ms_mediana;
    let desvio = |alvo: BracoB, ref_: BracoB| -> DesvioF2 {
        let (a, r) = (de(alvo, &pre), de(ref_, &pre));
        let (ma, mr) = (mediana(&a), mediana(&r));
        let min_r = r.iter().copied().fold(f64::INFINITY, f64::min);
        let max_a = a.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let (pa, pr) = (mediana(&de(alvo, &passo)), mediana(&de(ref_, &passo)));
        DesvioF2 { desvio: (ma - mr) / mr, pior_caso: (max_a - min_r) / min_r, ms_por_bloco: (ma - mr) / blocos_por_pedido.max(1) as f64, desvio_passo: (pa - pr) / pr }
    };
    let principal = desvio(B2, BA);
    let apoio = cfg.apoio.map(|r| desvio(B2, r));
    ResultadoControleB { classe: classe_por_desvio(principal.desvio), classe_apoio: apoio.as_ref().map(|d| classe_por_desvio(d.desvio)), principal: Some(principal), apoio }
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
        RecServico::de_somas(gen, ini, fim, f)
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
        assert!(a.fecha(DadosPedido { modelo: String::new(), indice: 0, aquecimento: false, n: 1, plano: vec![1], prefill_ms: 1.0, decode_passo_ms_mediana: 1.0 }).is_none());
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
            modelo: String::new(),
            indice: 0, aquecimento: false, n: 9, plano: vec![8, 1], blocos: vec![b0, b1], servico: vec![None, None], servico_extra: 0,
            t_armado: 0, t_logits_ini: t_logits, t_desarmado: t_logits + 2 * MS, prefill_ms: 0.0, f2: None, f2b: None, decode_passo_ms_mediana: 1.0,
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
            modelo: String::new(),
            indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b0], servico: vec![None], servico_extra: 0,
            t_armado: 0, t_logits_ini: t_logits, t_desarmado: t_logits + MS, prefill_ms: ms(t_logits + MS), f2: None, f2b: None, decode_passo_ms_mediana: 1.0,
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
            modelo: String::new(),
            indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b0.clone()], servico: vec![Some(serv(1, 10 * MS, 5 * MS, [0; 7]))], servico_extra: 0,
            t_armado: 0, t_logits_ini: b0.t[4], t_desarmado: b0.t[4], prefill_ms: ms(b0.t[4]), f2: None, f2b: None, decode_passo_ms_mediana: 1.0,
        };
        assert!(p.problemas().iter().any(|x| x.contains("serviço gen 1")));
    }

    /// O acumulador REAL dos ganchos: as operações nomeadas ocupam 85 ms de um I_g de 100 ms, com uma lacuna de 15 ms entre a camada e o
    /// fim (diagnóstico, laço). A lacuna não é absorvida por nenhuma fase.
    fn bloco_com_lacuna() -> RecServico {
        let mut a = AccServ::com_capacidade(16);
        a.apos_espera(0, 0, 10 * MS, true, 0); // espera [0, 10]
        a.op(S_MONTA, 10 * MS, 25 * MS); // 15
        a.op(S_COMPUTE, 25 * MS, 55 * MS); // 30
        a.op(S_ANSWER, 55 * MS, 60 * MS); // 5
        a.op(S_LIVRO, 60 * MS, 70 * MS); // 10
        // lacuna [70, 85]: ninguém a mede
        a.op(S_END_ESPERA, 85 * MS, 95 * MS); // 10
        a.op(S_END_TRAB, 95 * MS, 100 * MS); // 5
        a.fecha(1, 100 * MS).expect("armado")
    }

    #[test]
    fn cobertura_do_servico_acusa_s_fora_de_15_por_cento_pelo_acumulador_real() {
        let s = bloco_com_lacuna();
        assert_eq!((s.ini, s.fim, s.i_g()), (0, 100 * MS, 100 * MS));
        assert_eq!(s.soma(), 85 * MS);
        assert_eq!(s.s_fora(), 15 * MS);
        assert!((s.s_fora() as f64 / s.i_g() as f64 - 0.15).abs() < 1e-12, "S_fora / I_g = 0,15");
        // as durações das operações não absorvem a lacuna
        assert_eq!(s.f, [10 * MS, 15 * MS, 30 * MS, 5 * MS, 10 * MS, 10 * MS, 5 * MS]);
        assert!(s.coerente());
        let b = bloco(1, 8, 0, [0.1, 0.1, 0.1, 0.1]);
        let p = RecPedido {
            modelo: String::new(),
            indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b.clone()], servico: vec![Some(s)], servico_extra: 0,
            t_armado: 0, t_logits_ini: b.t[4], t_desarmado: b.t[4], prefill_ms: ms(b.t[4]), f2: None, f2b: None, decode_passo_ms_mediana: 1.0,
        };
        assert!(mediana(&p.s_fora_fracoes()) > LIM_COBERTURA_SERVICO, "ressalva de cobertura do serviço");
        let j = p.para_json();
        assert!((j["servico"][0]["s_fora_frac"].as_f64().unwrap() - 0.15).abs() < 1e-12);
        assert!((j["s_fora_frac_mediana"].as_f64().unwrap() - 0.15).abs() < 1e-12);
    }

    #[test]
    fn acumulador_sem_lacuna_tem_residuo_zero() {
        let mut a = AccServ::com_capacidade(16);
        a.apos_espera(0, 0, 10 * MS, true, 0);
        a.op(S_MONTA, 10 * MS, 30 * MS);
        a.op(S_COMPUTE, 30 * MS, 60 * MS);
        a.op(S_LIVRO, 60 * MS, 90 * MS);
        a.op(S_END_TRAB, 90 * MS, 100 * MS);
        let s = a.fecha(1, 100 * MS).unwrap();
        assert_eq!((s.soma(), s.s_fora()), (100 * MS, 0));
        assert!(s.coerente());
    }

    #[test]
    fn limites_de_i_g_nao_dependem_das_operacoes_e_o_decode_nao_registra() {
        let mut a = AccServ::com_capacidade(16);
        a.prev_fim = 1_000;
        // camada 0: o serviço já esperava desde 2_000, o prefill foi armado em 5_000, seq[0] visto em 9_000
        a.apos_espera(0, 2_000, 9_000, true, 5_000);
        assert_eq!(a.ini, 5_000, "I_g.ini = max(fim anterior, armar)");
        assert_eq!(a.f[S_ESPERA], 4_000, "a espera da camada 0 conta só de I_g.ini em diante");
        a.op(S_COMPUTE, 9_500, 12_000);
        a.apos_espera(1, 13_000, 14_000, true, 5_000); // espera da camada 1 [13_000, 14_000]; lacuna [12_000, 13_000]
        a.op(S_END_TRAB, 14_000, 17_000);
        let r = a.fecha(7, 17_500).expect("armado"); // fim de I_g 500 ns depois da última operação
        assert_eq!((r.gen, r.ini, r.fim), (7, 5_000, 17_500));
        assert_eq!(r.soma(), 4_000 + 2_500 + 1_000 + 3_000);
        assert_eq!(r.s_fora(), 12_500 - 10_500, "lacunas entre operações e o resto até o fim viram S_fora");
        assert_eq!(r.ops.len(), 4);
        assert!(r.coerente());
        // próximo bloco: I_g começa no fim do anterior; bloco não armado (decode) não gera registro nem guarda operação
        a.apos_espera(0, 18_000, 19_000, false, 5_000);
        assert_eq!(a.ini, 17_500);
        a.op(S_END_TRAB, 19_000, 21_000);
        assert!(a.ops.is_empty());
        assert!(a.fecha(8, 21_000).is_none());
        assert_eq!(a.prev_fim, 21_000);
    }

    #[test]
    fn operacoes_fora_de_ordem_ou_sobrepostas_invalidam_o_servico() {
        let ok = RecServico::novo(1, 0, 10 * MS, vec![OpS { fase: S_ESPERA as u8, ini: 0, fim: 4 * MS }, OpS { fase: S_COMPUTE as u8, ini: 4 * MS, fim: 8 * MS }]);
        assert!(ok.coerente());
        let sobre = RecServico::novo(1, 0, 10 * MS, vec![OpS { fase: S_ESPERA as u8, ini: 0, fim: 5 * MS }, OpS { fase: S_COMPUTE as u8, ini: 4 * MS, fim: 8 * MS }]);
        assert!(!sobre.coerente(), "sobreposição entre operações");
        let ordem = RecServico::novo(1, 0, 10 * MS, vec![OpS { fase: S_COMPUTE as u8, ini: 5 * MS, fim: 8 * MS }, OpS { fase: S_ESPERA as u8, ini: 0, fim: 4 * MS }]);
        assert!(!ordem.coerente(), "ordem trocada");
        let invertida = RecServico::novo(1, 0, 10 * MS, vec![OpS { fase: S_ESPERA as u8, ini: 5 * MS, fim: 4 * MS }]);
        assert!(!invertida.coerente(), "fim antes do início de uma operação");
        let fora = RecServico::novo(1, 0, 10 * MS, vec![OpS { fase: S_ESPERA as u8, ini: 0, fim: 11 * MS }]);
        assert!(!fora.coerente(), "operação além de I_g");
        let mut soma_errada = ok.clone();
        soma_errada.f[S_COMPUTE] += 1_000_000;
        assert!(!soma_errada.coerente(), "somas que não vêm das operações");
    }

    fn op(fase: usize, ini_ms: f64, fim_ms: f64) -> OpS {
        OpS { fase: fase as u8, ini: (ini_ms * MS as f64) as u64, fim: (fim_ms * MS as f64) as u64 }
    }

    #[test]
    fn recorte_de_h3_separa_compute_antes_e_depois_do_inicio_do_bloco_do_host() {
        // I_g = [0, 10] ms; bloco do host [5, 10] ms
        let host = bloco(1, 8, 5 * MS, [0.5, 0.5, 2.0, 2.0]);
        assert_eq!((host.t[0], host.t[4]), (5 * MS, 10 * MS));
        let mut antes = RecServico::novo(1, 0, 10 * MS, vec![op(S_COMPUTE, 0.0, 5.0), op(S_LIVRO, 5.0, 10.0)]);
        let mut depois = RecServico::novo(1, 0, 10 * MS, vec![op(S_LIVRO, 0.0, 5.0), op(S_COMPUTE, 5.0, 10.0)]);
        antes.recorta(&host);
        depois.recorta(&host);
        assert_eq!(antes.f[S_COMPUTE], depois.f[S_COMPUTE], "mesma soma integral");
        assert_eq!(antes.f[S_COMPUTE], 5 * MS);
        assert_eq!(antes.inter_ns(&host), depois.inter_ns(&host), "mesmo inter_bloco");
        assert_eq!((antes.recorte[S_COMPUTE], depois.recorte[S_COMPUTE]), (0, 5 * MS), "recortes distintos");
        assert_eq!((antes.recorte[S_LIVRO], depois.recorte[S_LIVRO]), (5 * MS, 0));
    }

    #[test]
    fn recorte_com_servico_parcialmente_sobreposto_e_publicacao_depois_de_t4() {
        let host = bloco(1, 8, 20 * MS, [0.5, 0.5, 4.0, 5.0]); // [20, 30]
        // começa antes de t0 e termina dentro; publicação (end_trab) termina depois de t4
        let mut s = RecServico::novo(
            1,
            15 * MS,
            34 * MS,
            vec![op(S_COMPUTE, 15.0, 23.0), op(S_LIVRO, 23.0, 27.0), op(S_END_ESPERA, 27.0, 29.0), op(S_END_TRAB, 29.0, 34.0)],
        );
        assert!(s.coerente());
        s.recorta(&host);
        assert_eq!(s.recorte[S_COMPUTE], 3 * MS, "só a parcela depois de t0");
        assert_eq!(s.recorte[S_LIVRO], 4 * MS);
        assert_eq!(s.recorte[S_END_ESPERA], 2 * MS);
        assert_eq!(s.recorte[S_END_TRAB], MS, "só a parcela até t4: a publicação passa de t4");
        assert_eq!(s.f[S_END_TRAB], 5 * MS, "a soma integral mantém tudo");
        let tudo: u64 = s.recorte.iter().sum();
        assert!(tudo <= host.b(), "o recorte nunca passa do intervalo do bloco");
        // serviço inteiramente fora do bloco: recorte zero
        let mut fora = RecServico::novo(1, 40 * MS, 45 * MS, vec![op(S_COMPUTE, 40.0, 45.0)]);
        fora.recorta(&host);
        assert_eq!(fora.recorte, [0; 7]);
    }

    #[test]
    fn merge_calcula_o_recorte_e_o_json_o_separa_das_somas() {
        let pf = Arc::new(Compartilhado::novo());
        let mut a = Armado::arma(Some(pf.clone()), 1);
        let t0 = agora_ns() + 10_000_000;
        let b = RecBloco { gen: 9, m: 8, pos_ini: 0, t: [t0, t0, t0, t0 + 1_000, t0 + 5_000_000], f0_pendente: false };
        pf.registra_bloco(b.clone());
        pf.registra_servico(RecServico::novo(9, t0 - 2_000_000, t0 + 3_000_000, vec![OpS { fase: S_COMPUTE as u8, ini: t0 - 2_000_000, fim: t0 + 3_000_000 }]));
        a.logits_ini();
        let p = a.fecha_com(DadosPedido { modelo: "m".into(), indice: 0, aquecimento: false, n: 8, plano: vec![8], prefill_ms: 1.0, decode_passo_ms_mediana: 1.0 }, &mut || {}).unwrap();
        let s = p.servico[0].as_ref().unwrap();
        assert_eq!(s.recorte[S_COMPUTE], 3_000_000, "[t0, t0+3 ms] dentro do bloco");
        let j = p.para_json();
        assert_eq!(j["servico"][0]["s_compute_ms"].as_f64().unwrap(), 5.0);
        assert_eq!(j["servico"][0]["s_compute_recorte_ms"].as_f64().unwrap(), 3.0);
        assert_eq!(j["servico"][0]["s_livro_recorte_ms"].as_f64().unwrap(), 0.0);
        assert!(j["servico"][0].get("n_ops").is_some());
    }

    #[test]
    fn json_traz_o_modelo_e_dois_modelos_no_mesmo_processo_se_distinguem() {
        let b = bloco(1, 8, 0, [0.1, 0.1, 0.1, 0.1]);
        let mk = |modelo: &str, indice: usize| RecPedido {
            modelo: modelo.into(), indice, aquecimento: indice == 0, n: 8, plano: vec![8], blocos: vec![b.clone()], servico: vec![None], servico_extra: 0,
            t_armado: 0, t_logits_ini: b.t[4], t_desarmado: b.t[4], prefill_ms: ms(b.t[4]), f2: None, f2b: None, decode_passo_ms_mediana: 1.0,
        };
        // cada motor reinicia `indice` em 0: o par (modelo, indice) é a chave
        let v = perfil_json(&[mk("1b", 0), mk("1b", 1), mk("3b", 0), mk("3b", 1)]);
        let ps = v["pedidos"].as_array().unwrap();
        let chaves: Vec<(String, u64)> = ps.iter().map(|p| (p["modelo"].as_str().unwrap().to_string(), p["indice"].as_u64().unwrap())).collect();
        assert_eq!(chaves, vec![("1b".into(), 0), ("1b".into(), 1), ("3b".into(), 0), ("3b".into(), 1)]);
        let unicas: std::collections::HashSet<_> = chaves.iter().collect();
        assert_eq!(unicas.len(), 4, "nenhuma entrada repete (modelo, indice)");
        assert_eq!(v["versao"], 2);
    }

    // ---- (g) classificação de H1 a H6 ----

    use ClasseH::*;

    fn ok() -> ClasseControle {
        ClasseControle::Comprovada
    }

    #[test]
    fn h1_dentro_fora_e_sem_leitura_no_l100() {
        assert_eq!(classifica_h1(Serie::L100, Some(0.70), Some(0.10)), Dentro, "F2/B = 0,70 exato");
        assert_eq!(classifica_h1(Serie::L100, Some(0.80), Some(0.10)), Dentro);
        assert_eq!(classifica_h1(Serie::L100, Some(0.20), Some(0.50)), Fora, "F3/B = 0,50 exato");
        assert_eq!(classifica_h1(Serie::L100, Some(0.40), Some(0.30)), SemLeitura);
        assert_eq!(classifica_h1(Serie::L100, Some(0.6999), Some(0.4999)), SemLeitura, "logo abaixo dos dois limites");
        assert_eq!(classifica_h1(Serie::L100, None, Some(0.30)), SemDado, "sem F2");
        assert_eq!(classifica_h1(Serie::L100, Some(0.40), None), SemDado, "sem F3");
        assert_eq!(classifica_h1(Serie::L100, Some(f64::NAN), Some(0.3)), SemDado);
        assert_eq!(classifica_h1(Serie::L100, Some(0.80), Some(0.60)), SemDado, "as duas juntas somam mais que B: incoerente");
        assert_eq!(classifica_h1(Serie::L10, Some(0.9), Some(0.0)), SemDado, "sem previsão no L10");
        assert_eq!(classifica_h1(Serie::Lc, Some(0.9), Some(0.0)), SemDado);
    }

    #[test]
    fn h2_dentro_fora_e_sem_leitura_no_l100() {
        assert_eq!(classifica_h2(Serie::L100, Some(0.50), Some(3.0)), Dentro, "limites exatos");
        assert_eq!(classifica_h2(Serie::L100, Some(0.25), Some(1.5)), SemLeitura, "limites de fora são estritos");
        assert_eq!(classifica_h2(Serie::L100, Some(0.2499), Some(5.0)), Fora, "F3/B < 0,25");
        assert_eq!(classifica_h2(Serie::L100, Some(0.60), Some(1.4999)), Fora, "F3/passo < 1,5");
        assert_eq!(classifica_h2(Serie::L100, Some(0.40), Some(2.0)), SemLeitura);
        assert_eq!(classifica_h2(Serie::L100, Some(0.60), Some(2.0)), SemLeitura, "B alto, passo no meio");
        assert_eq!(classifica_h2(Serie::L100, Some(0.10), None), Fora, "uma medida que viola decide o fora");
        assert_eq!(classifica_h2(Serie::L100, None, Some(1.0)), Fora);
        assert_eq!(classifica_h2(Serie::L100, Some(0.60), None), SemDado, "falta o denominador e nada viola");
        assert_eq!(classifica_h2(Serie::L100, None, None), SemDado);
        assert_eq!(classifica_h2(Serie::L10, Some(0.6), Some(4.0)), SemDado);
    }

    #[test]
    fn h3_dentro_fora_e_sem_leitura_nas_duas_series() {
        // L10
        assert_eq!(classifica_h3(Serie::L10, Some(0.50), Some(0.20)), Dentro, "limites exatos");
        assert_eq!(classifica_h3(Serie::L10, Some(0.29), Some(0.30)), Fora, "R3 < 0,30");
        assert_eq!(classifica_h3(Serie::L10, Some(0.60), Some(0.0999)), Fora, "S_compute/B < 0,10");
        assert_eq!(classifica_h3(Serie::L10, Some(0.30), Some(0.10)), SemLeitura, "limites de fora estritos");
        assert_eq!(classifica_h3(Serie::L10, Some(0.40), Some(0.15)), SemLeitura);
        assert_eq!(classifica_h3(Serie::L10, Some(0.60), Some(0.15)), SemLeitura, "R3 alto, compute no meio");
        assert_eq!(classifica_h3(Serie::L10, Some(0.60), None), SemDado);
        assert_eq!(classifica_h3(Serie::L10, Some(0.10), None), Fora);
        assert_eq!(classifica_h3(Serie::L10, None, None), SemDado);
        // L100
        assert_eq!(classifica_h3(Serie::L100, Some(0.10), None), Dentro, "R3 <= 0,10 exato");
        assert_eq!(classifica_h3(Serie::L100, Some(0.2501), None), Fora);
        assert_eq!(classifica_h3(Serie::L100, Some(0.25), None), SemLeitura, "fora é estrito");
        assert_eq!(classifica_h3(Serie::L100, Some(0.18), None), SemLeitura);
        assert_eq!(classifica_h3(Serie::L100, None, Some(0.5)), SemDado);
        assert_eq!(classifica_h3(Serie::Lc, Some(0.9), Some(0.9)), SemDado);
    }

    #[test]
    fn h4_dentro_fora_e_sem_leitura_no_l100() {
        assert_eq!(classifica_h4(Serie::L100, Some(0.05)), Dentro, "limite exato");
        assert_eq!(classifica_h4(Serie::L100, Some(0.0199)), Fora);
        assert_eq!(classifica_h4(Serie::L100, Some(0.02)), SemLeitura, "fora é estrito");
        assert_eq!(classifica_h4(Serie::L100, Some(0.035)), SemLeitura);
        assert_eq!(classifica_h4(Serie::L100, None), SemDado);
        assert_eq!(classifica_h4(Serie::L100, Some(f64::INFINITY)), SemDado);
        assert_eq!(classifica_h4(Serie::L10, Some(0.5)), SemDado);
    }

    #[test]
    fn h5_dentro_fora_e_sem_leitura_nas_duas_series() {
        assert_eq!(classifica_h5(Serie::L10, Some(0.10)), Dentro, "limite exato");
        assert_eq!(classifica_h5(Serie::L10, Some(0.0299)), Fora);
        assert_eq!(classifica_h5(Serie::L10, Some(0.03)), SemLeitura);
        assert_eq!(classifica_h5(Serie::L10, Some(0.06)), SemLeitura);
        assert_eq!(classifica_h5(Serie::L10, None), SemDado);
        assert_eq!(classifica_h5(Serie::L100, Some(0.02)), Dentro, "limite exato");
        assert_eq!(classifica_h5(Serie::L100, Some(0.0501)), Fora);
        assert_eq!(classifica_h5(Serie::L100, Some(0.05)), SemLeitura);
        assert_eq!(classifica_h5(Serie::L100, Some(0.03)), SemLeitura);
        assert_eq!(classifica_h5(Serie::L100, None), SemDado);
        assert_eq!(classifica_h5(Serie::Lc, Some(0.5)), SemDado);
    }

    #[test]
    fn h6_pela_medida_cobre_dentro_fora_sem_leitura_sem_dado_e_resto_descritivo() {
        // resto 4 (texto longo): H6 se R6 >= 0,75; por posição se R6 <= 0,625
        assert_eq!(classifica_h6_medida(4, Some(0.75)), Dentro);
        assert_eq!(classifica_h6_medida(4, Some(0.625)), Fora);
        assert_eq!(classifica_h6_medida(4, Some(0.70)), SemLeitura);
        assert_eq!(classifica_h6_medida(4, None), SemDado, "sem R6");
        assert_eq!(classifica_h6_medida(4, Some(f64::NAN)), SemDado);
        // restos 1 e 2 (curto e médio)
        assert_eq!(classifica_h6_medida(1, Some(0.5625)), Dentro);
        assert_eq!(classifica_h6_medida(1, Some(0.15625)), Fora);
        assert_eq!(classifica_h6_medida(2, Some(0.40)), SemLeitura);
        // restos 6 e 7: as faixas se tocam; resto 0 ou fora de 1..=7 não tem bloco parcial
        for r in [6usize, 7, 0, 8] {
            assert_eq!(classifica_h6_medida(r, Some(1.0)), Descritivo, "resto {r}");
        }
    }

    fn med() -> MedidasH {
        MedidasH { f1_b: Some(0.06), f2_b: Some(0.75), f3_b: Some(0.1), f3_passo: Some(1.0), r3: Some(0.05), s_compute_b: Some(0.3), r5: Some(0.01), r6: Some(0.8), resto: 4 }
    }

    #[test]
    fn classifica_h_exige_controle_valido_e_aceita_comprovada_e_com_ressalva() {
        let m = med();
        let casos = [
            (Hip::H1, Serie::L100, Dentro),
            (Hip::H2, Serie::L100, Fora),
            (Hip::H3, Serie::L100, Dentro),
            (Hip::H4, Serie::L100, Dentro),
            (Hip::H5, Serie::L100, Dentro),
            (Hip::H6, Serie::L100, Dentro),
        ];
        for (h, s, esperado) in casos {
            assert_eq!(classifica_h(h, s, &m, &ok()), esperado, "{h:?}");
            assert_eq!(classifica_h(h, s, &m, &ClasseControle::ComRessalva), esperado, "{h:?} com ressalva");
            for c in [ClasseControle::SemProva, ClasseControle::Incompleto("par truncado".into())] {
                assert_eq!(classifica_h(h, s, &m, &c), SemProva, "{h:?} com controle {c:?}: nenhuma classificação");
            }
        }
        // par de controle incompleto de verdade, vindo da classificação do controle
        let truncado = classifica_controle(&par(0.01)[..4], true);
        assert!(matches!(truncado.classe, ClasseControle::Incompleto(_)));
        assert_eq!(classifica_h(Hip::H1, Serie::L100, &m, &truncado.classe), SemProva);
        let sem_prova = classifica_controle(&par(0.12), true);
        assert_eq!(sem_prova.classe, ClasseControle::SemProva);
        assert_eq!(classifica_h(Hip::H4, Serie::L100, &m, &sem_prova.classe), SemProva);
        // resto que torna H6 descritivo, mesmo com controle bom
        let mut m6 = med();
        m6.resto = 6;
        assert_eq!(classifica_h(Hip::H6, Serie::L10, &m6, &ok()), Descritivo);
        // dado ausente com controle bom
        assert_eq!(classifica_h(Hip::H4, Serie::L100, &MedidasH::default(), &ok()), SemDado);
        assert_eq!(classifica_h(Hip::H6, Serie::L10, &MedidasH { resto: 4, ..Default::default() }, &ok()), SemDado);
        // série sem previsão
        assert_eq!(classifica_h(Hip::H3, Serie::Lc, &m, &ok()), SemDado);
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
            modelo: String::new(),
            indice: 0, aquecimento: false, n: 17, plano: vec![8, 8, 1], blocos: vec![b0, b1, b2], servico: vec![None; 3], servico_extra: 0,
            t_armado: 0, t_logits_ini: 0, t_desarmado: 0, prefill_ms: 0.0, f2: None, f2b: None, decode_passo_ms_mediana: 1.0,
        };
        assert!((p.r6().unwrap() - 0.6).abs() < 1e-9);
    }
}

// ---- Testes sem GPU da F2 do host ---------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests_f2 {
    use super::*;

    const T2: u64 = 1_000_000;

    /// A ordem das operações do motor para `layers` camadas, na sequência exata da I3.
    fn ordem(layers: usize) -> Vec<(u8, u32)> {
        let mut v = vec![(H_ENTRADA, SEM_CAMADA), (H_BEGIN_STEP, SEM_CAMADA), (H_UPLOAD_TABELA, SEM_CAMADA), (H_WAIT_EVENT, SEM_CAMADA)];
        for l in 0..layers as u32 {
            v.extend(OPS_CAMADA.map(|o| (o, l)));
        }
        v.extend([(H_POST_LAYERS, SEM_CAMADA), (H_SEQ_FIM, SEM_CAMADA), (H_SAIDA, SEM_CAMADA)]);
        v
    }

    /// Bloco sintético: operações encadeadas com as durações `d` (cíclicas), `b_ini` entre t2 e a entrada e `b_fim` entre a saída e t3.
    fn bloco(layers: usize, d: &[u64], b_ini: u64, b_fim: u64) -> BlocoF2 {
        let entrada = T2 + b_ini;
        let mut t = entrada;
        let mut ops = vec![];
        for (i, (op, camada)) in ordem(layers).into_iter().enumerate() {
            let fim = t + d[i % d.len()];
            ops.push(OpH { op, camada, ini: t, fim });
            t = fim;
        }
        BlocoF2 { gen: 7, m: 8, pos_ini: 0, n_ready: 0, ops, entrada, saida: t, t2: T2, t3: t + b_fim }
    }

    fn rec_bloco(h: &BlocoF2) -> RecBloco {
        RecBloco { gen: h.gen, m: h.m, pos_ini: h.pos_ini, t: [h.t2 - 2000, h.t2 - 1000, h.t2, h.t3, h.t3 + 500], f0_pendente: false }
    }

    #[test]
    fn variavel_f2_so_liga_com_o_valor_um_e_exige_o_bloco() {
        assert_eq!(VARIAVEL_F2, "LASTRO_PERFIL_F2");
        assert!(interpreta_f2(Some("1")));
        for v in [None, Some(""), Some("0"), Some("true"), Some(" 1"), Some("1 "), Some("11"), Some("on")] {
            assert!(!interpreta_f2(v), "{v:?}");
        }
        assert!(f2_efetivo(true, true));
        assert!(!f2_efetivo(false, true) && !f2_efetivo(true, false) && !f2_efetivo(false, false));
        assert_eq!(linha_estado_f2(true, true), "perfil-f2: ligado");
        assert_eq!(linha_estado_f2(false, true), "perfil-f2: desligado (exige perfil-bloco)");
        assert_eq!(linha_estado_f2(true, false), "perfil-f2: desligado");
        assert_eq!(linha_estado_f2(false, false), "perfil-f2: desligado");
    }

    #[test]
    fn acumulador_nao_aloca_no_caminho_quente_e_o_desligado_nao_existe() {
        for layers in [24usize, 32] {
            let mut a = AccHost::com_capacidade(layers);
            let cap = a.capacidade();
            assert!(cap >= capacidade_f2(layers) && capacidade_f2(layers) == layers * 6 + 8);
            for bloco in 0..3 {
                a.reinicia(agora_ns());
                a.abre(bloco * 10);
                for (op, c) in &ordem(layers)[1..ordem(layers).len() - 1] {
                    a.marca(*op, *c);
                }
                a.fecha_saida(agora_ns());
                assert_eq!(a.ops.len(), n_ops_esperado(layers));
                assert_eq!(a.capacidade(), cap, "capacidade igual depois do bloco {bloco}");
            }
        }
        assert_eq!(n_ops_esperado(24), 151);
        assert_eq!(n_ops_esperado(32), 199);
        // Sem instrumento: `Armado::arma(None, ..)` é inerte e `Compartilhado` sem F2 não guarda nada.
        let pf = Compartilhado::novo();
        assert!(!pf.f2());
        let _ = Armado::arma(None, 1000);
    }

    #[test]
    fn i3_recusa_uma_operacao_a_mais_ou_a_menos_com_24_e_32_camadas() {
        for layers in [24usize, 32] {
            let h = bloco(layers, &[100], 100, 100);
            assert!(h.i1() && h.i2() && h.i3(layers) && h.i5(layers));
            assert!(!h.i3(layers + 1) && !h.i3(layers - 1), "estrutura de outra profundidade");
            let mut mais = h.clone();
            mais.ops.push(OpH { op: H_ADD_Y, camada: 0, ini: mais.saida, fim: mais.saida });
            assert!(!mais.i3(layers));
            let mut menos = h.clone();
            menos.ops.remove(10);
            assert!(!menos.i3(layers));
            // camada repetida no lugar de outra (mesma contagem)
            let mut rep = h.clone();
            rep.ops[4 + 6 * 3].camada = 2;
            assert!(!rep.i3(layers));
            // ordem trocada dentro da camada
            let mut tro = h.clone();
            let (a, b) = (tro.ops[4].op, tro.ops[5].op);
            tro.ops[4].op = b;
            tro.ops[5].op = a;
            assert!(!tro.i3(layers));
        }
    }

    #[test]
    fn i1_i2_e_i5_pegam_carimbo_fora_de_ordem_sobreposicao_e_soma_errada() {
        let h = bloco(2, &[500], 100, 100);
        assert!(h.i1() && h.i2());
        let mut fora = h.clone();
        fora.ops[5].ini = fora.ops[4].fim.saturating_sub(10); // volta no tempo: sobreposição
        assert!(!fora.i1());
        let mut invertido = h.clone();
        invertido.ops[3].fim = invertido.ops[3].ini - 1;
        assert!(!invertido.i1());
        let mut antes_de_t2 = h.clone();
        antes_de_t2.t2 = antes_de_t2.entrada + 5_000; // a cadeia começaria antes de t2
        assert!(!antes_de_t2.i1());
        let mut depois_de_t3 = h.clone();
        depois_de_t3.t3 = depois_de_t3.saida - 5_000;
        assert!(!depois_de_t3.i1());
        // soma não fecha: buraco de 10 µs entre duas operações (fim da cadeia não bate com a saída gravada)
        let mut buraco = h.clone();
        for o in &mut buraco.ops[6..] {
            o.ini += 10_000;
            o.fim += 10_000;
        }
        assert!(!buraco.i1() || !buraco.i2());
        // t3 muito depois da saída: R_borda cresce, a soma continua batendo contra t3 - t2 (a cobertura vira ressalva, não invalida)
        let borda = bloco(2, &[500], 100, 50_000);
        assert!(borda.i1() && borda.i2() && borda.ressalva_borda());
        assert!(h.i5(2));
    }

    #[test]
    fn residuo_de_borda_0_meio_por_cento_3_por_cento_e_acima() {
        // S = 9700 ns em 19 operações: 18 de 500 e uma de 700.
        let mut d = vec![500u64; 19];
        d[18] = 700;
        let mk = |b_fim: u64| bloco(2, &d, 0, b_fim);
        let h0 = mk(0);
        assert_eq!((h0.soma_ops_ns(), h0.r_borda_ns(), h0.cobertura()), (9700, 0, 0.0));
        let h05 = mk(50);
        assert!((h05.cobertura() - 0.005).abs() < 1e-3 && !h05.ressalva_borda());
        let h3 = mk(300);
        assert_eq!(h3.cobertura(), 0.03, "exatamente 3%");
        assert!(!h3.ressalva_borda(), "3% exato não passa do limite");
        let h4 = mk(400);
        assert!(h4.cobertura() > 0.03 && h4.ressalva_borda());
        for h in [&h0, &h05, &h3, &h4] {
            assert!(h.i1() && h.i2(), "nenhum invalida por cobertura");
            let d = DadosF2 { layers: 2, hosts: vec![Some(h.clone())], tbs: vec![None], host_extra: 0, tb_extra: 0 };
            assert!(d.problemas(&[rec_bloco(h)]).is_empty(), "{:?}", d.problemas(&[rec_bloco(h)]));
        }
        let d4 = DadosF2 { layers: 2, hosts: vec![Some(h4.clone())], tbs: vec![None], host_extra: 0, tb_extra: 0 };
        assert!(d4.ressalva_borda());
    }

    #[test]
    fn i4_junta_gen_m_e_f2_do_registro_por_bloco() {
        let h = bloco(2, &[500], 100, 100);
        let ok = rec_bloco(&h);
        let mk = |h: &BlocoF2| DadosF2 { layers: 2, hosts: vec![Some(h.clone())], tbs: vec![None], host_extra: 0, tb_extra: 0 };
        assert!(mk(&h).problemas(&[ok.clone()]).is_empty());
        let mut gen = ok.clone();
        gen.gen += 1;
        assert!(!mk(&h).problemas(&[gen]).is_empty(), "gen que não casa");
        let mut m = ok.clone();
        m.m = 3;
        assert!(!mk(&h).problemas(&[m]).is_empty());
        let mut t3 = ok.clone();
        t3.t[3] += 5_000; // F2 do registro por bloco diferente da soma
        assert!(!mk(&h).problemas(&[t3]).is_empty());
        // I3 invalida o pedido: uma operação a menos num dos blocos
        let mut menos = h.clone();
        menos.ops.pop();
        assert!(mk(&menos).problemas(&[ok]).iter().any(|s| s.contains("I3")));
    }

    fn tb_de(h: &BlocoF2, layers: usize, f: impl Fn(usize, u64) -> (u64, u64)) -> RecTb {
        let camadas = (0..layers).map(|l| {
            let fim = h.fim_seq(l).unwrap();
            let (obs, tb) = f(l, fim);
            TbL { camada: l as u32, gen: h.gen, tb_obs: obs, tb }
        });
        RecTb { gen: h.gen, camadas: camadas.collect() }
    }

    #[test]
    fn i8_valida_a_coleta_do_servico_e_conserva_delta_negativo() {
        let layers = 3;
        let h = bloco(layers, &[1_000], 100, 100);
        let bom = tb_de(&h, layers, |l, fim| (fim + 100 * (l as u64 + 1), fim + 300 * (l as u64 + 1)));
        assert!(i8_problemas(&h, Some(&bom), layers).is_empty());
        // observação anterior ao retorno da chamada no host: defasagem negativa conservada com sinal, sem invalidar
        let neg = tb_de(&h, layers, |l, fim| (fim - 700 + 5_000 * l as u64, fim - 300 + 5_000 * l as u64));
        assert!(i8_problemas(&h, Some(&neg), layers).is_empty(), "{:?}", i8_problemas(&h, Some(&neg), layers));
        let defs = defasagens(&h, &neg);
        assert_eq!(defs[0].defasagem_obs, -700);
        assert_eq!(defs[0].defasagem, -300);
        assert!(defs[0].intervalo_obs.is_none() && defs[1].intervalo.is_some());
        // tb_obs > tb
        let mut ruim = bom.clone();
        ruim.camadas[1].tb_obs = ruim.camadas[1].tb + 1;
        assert!(!i8_problemas(&h, Some(&ruim), layers).is_empty());
        // tb decrescente em l
        let mut dec = bom.clone();
        dec.camadas[2].tb = dec.camadas[1].tb - 1;
        dec.camadas[2].tb_obs = dec.camadas[2].tb;
        assert!(!i8_problemas(&h, Some(&dec), layers).is_empty());
        // anterior ao início de H_SEQ (conferência causal)
        let ini0 = h.ops.iter().find(|o| o.op == H_SEQ && o.camada == 0).unwrap().ini;
        let ant = tb_de(&h, layers, |l, fim| (if l == 0 { ini0 - 100_000 } else { fim }, fim));
        assert!(!i8_problemas(&h, Some(&ant), layers).is_empty());
        // falta camada observada
        let mut falta = bom.clone();
        falta.camadas.pop();
        assert!(!i8_problemas(&h, Some(&falta), layers).is_empty());
        assert!(!i8_problemas(&h, None, layers).is_empty());
        // geração de serviço trocada: não junta ao bloco
        let mut gen = bom.clone();
        gen.camadas[1].gen += 1;
        assert!(!i8_problemas(&h, Some(&gen), layers).is_empty());
        let mut gen2 = bom.clone();
        gen2.gen += 1;
        assert!(!i8_problemas(&h, Some(&gen2), layers).is_empty());
        // a falha da coleta invalida só a HI, não o pedido
        let d = DadosF2 { layers, hosts: vec![Some(h.clone())], tbs: vec![Some(gen2)], host_extra: 0, tb_extra: 0 };
        assert!(d.problemas(&[rec_bloco(&h)]).is_empty());
        assert!(!d.problemas_hi(&[rec_bloco(&h)]).is_empty());
    }

    #[test]
    fn hi_mesma_publicacao_custos_de_copia_diferentes_e_publicacoes_distintas_com_o_mesmo_tb() {
        let layers = 4;
        let h1 = bloco(layers, &[1_000], 100, 100);
        // mesma publicação (mesmo fim de H_SEQ): tb_obs igual, tb muda com o custo da cópia
        let barato = tb_de(&h1, layers, |_, fim| (fim + 200, fim + 400));
        let caro = tb_de(&h1, layers, |_, fim| (fim + 200, fim + 9_400));
        let (db, dc) = (defasagens(&h1, &barato), defasagens(&h1, &caro));
        assert!(db.iter().zip(&dc).all(|(a, b)| a.defasagem_obs == b.defasagem_obs && a.defasagem < b.defasagem));
        // publicações distintas (hosts com H_SEQ em instantes diferentes) com o mesmo tb: defasagens distintas
        let h2 = bloco(layers, &[2_000], 100, 100);
        let mesmo_tb = |h: &BlocoF2| RecTb { gen: h.gen, camadas: (0..layers as u32).map(|l| TbL { camada: l, gen: h.gen, tb_obs: 30_000_000 + l as u64, tb: 30_000_100 + l as u64 }).collect() };
        let (d1, d2) = (defasagens(&h1, &mesmo_tb(&h1)), defasagens(&h2, &mesmo_tb(&h2)));
        assert!(d1.iter().zip(&d2).all(|(a, b)| a.defasagem != b.defasagem));
        // o relatório da HI preserva os números com sinal, sem escolher regime nem emitir classe
        let neg = tb_de(&h1, layers, |l, fim| (fim - 500 - l as u64, fim - 100));
        let amostras = vec![(&h1, defasagens(&h1, &barato)), (&h1, defasagens(&h1, &neg)), (&h2, dc.clone())];
        let r = relatorio_hi(&amostras, layers);
        assert_eq!(r["descritivo"], true);
        for chave in ["classe", "regime", "rotulo", "veredito"] {
            assert!(r.get(chave).is_none(), "o relatório não carrega `{chave}`");
        }
        let txt = r.to_string();
        assert!(!txt.contains("GPU espera"), "sem o rótulo causal");
        let d = &r["distribuicao"]["defasagem_obs"];
        assert_eq!(d["n"], 12);
        assert_eq!(d["min"], -503.0, "o mínimo negativo é conservado com sinal");
        assert!(d["max"].as_f64().unwrap() > 0.0 && d["mediana"].is_number() && d["p90"].is_number());
        assert_eq!(r["mediana_por_camada"]["defasagem_obs"].as_array().unwrap().len(), layers);
        assert_eq!(r["mediana_por_camada"]["host_seis_operacoes"][0]["n"], 3);
        assert!(r["limites"].as_str().unwrap().contains("nao medem a GPU"));
        // o JSON do pedido nunca vira zero um delta negativo
        let pj = serde_json::to_string(&r).unwrap();
        assert!(pj.contains("-503"));
    }

    #[test]
    fn classificadores_nos_limites_exatos() {
        use ClasseH::*;
        // HA: >= 0,80 dentro; < 0,50 fora
        assert_eq!((classifica_ha(Some(0.80)), classifica_ha(Some(0.7999)), classifica_ha(Some(0.50)), classifica_ha(Some(0.4999))), (Dentro, SemLeitura, SemLeitura, Fora));
        // HB
        assert_eq!(classifica_hb(Some(0.40), Some(true)), Dentro);
        assert_eq!(classifica_hb(Some(0.40), Some(false)), SemLeitura, "40% sem ser a maior soma não é dentro");
        assert_eq!(classifica_hb(Some(0.3999), Some(true)), SemLeitura);
        assert_eq!((classifica_hb(Some(0.25), Some(true)), classifica_hb(Some(0.2499), Some(true))), (SemLeitura, Fora));
        assert_eq!(classifica_hb(None, Some(true)), SemDado);
        assert_eq!(classifica_hb(Some(0.5), None), SemDado);
        // HC: <= 0,05 dentro; > 0,15 fora
        assert_eq!((classifica_hc(Some(0.05)), classifica_hc(Some(0.0501)), classifica_hc(Some(0.15)), classifica_hc(Some(0.1501))), (Dentro, SemLeitura, SemLeitura, Fora));
        // HD: <= 1,5 dentro; > 3,0 fora
        assert_eq!((classifica_hd(Some(1.5)), classifica_hd(Some(1.5001)), classifica_hd(Some(3.0)), classifica_hd(Some(3.0001))), (Dentro, SemLeitura, SemLeitura, Fora));
        // HE: <= 0,05 dentro; > 0,20 fora
        assert_eq!((classifica_he(Some(0.05)), classifica_he(Some(0.0501)), classifica_he(Some(0.20)), classifica_he(Some(0.2001))), (Dentro, SemLeitura, SemLeitura, Fora));
        // HH: >= 1,5 dentro; sem fora
        assert_eq!((classifica_hh(Some(1.5)), classifica_hh(Some(1.4999)), classifica_hh(Some(0.1))), (Dentro, Descritivo, Descritivo));
        assert_eq!(classifica_ha(Some(f64::NAN)), SemDado);
        // as descritivas: HF, HG, HI não têm classe de dentro/fora
        let c = ClasseControle::Comprovada;
        let m = MedidasF2 { hf_razao: Some(0.25), hi_dados: true, ..MedidasF2::default() };
        assert_eq!((classifica_f2(HipF2::Hf, &m, &c), classifica_f2(HipF2::Hi, &m, &c), classifica_f2(HipF2::Hg, &m, &c)), (Descritivo, Descritivo, SemDado));
        // controle > 10% = SEM PROVA para todas
        for h in [HipF2::Ha, HipF2::Hb, HipF2::Hc, HipF2::Hd, HipF2::He, HipF2::Hf, HipF2::Hg, HipF2::Hh, HipF2::Hi] {
            assert_eq!(classifica_f2(h, &m, &ClasseControle::SemProva), SemProva);
            assert_eq!(classifica_f2(h, &m, &ClasseControle::Incompleto("x".into())), SemProva);
        }
        assert_eq!(faixa_hf(2), (0.1875, 0.3125));
    }

    #[test]
    fn medidas_de_blocos_sinteticos_dao_as_razoes_esperadas() {
        // Cada camada: pre 600, route 100, seq 10, launch 200, waitflag 10, addy 80 = 1000; fixas 7 x 10.
        let layers = 4;
        let mut d = vec![10u64; 4];
        for _ in 0..layers {
            d.extend([600, 100, 10, 200, 10, 80]);
        }
        d.extend([10, 10, 10]);
        let blocos: Vec<BlocoF2> = (0..5).map(|_| bloco(layers, &d, 0, 0)).collect();
        let refs: Vec<&BlocoF2> = blocos.iter().collect();
        let m = medidas_f2_pedido(&refs, layers);
        let f2 = (4 * 1000 + 70) as f64;
        assert!((m.ha.unwrap() - 4.0 * 980.0 / f2).abs() < 1e-9);
        assert!((m.hb_frac.unwrap() - 2400.0 / f2).abs() < 1e-9 && m.hb_maior == Some(true));
        assert!((m.hc.unwrap() - (4.0 * 20.0 + 40.0) / f2).abs() < 1e-9);
        assert_eq!(m.hd, Some(1.0), "camadas iguais");
        assert_eq!(m.he, Some(0.0));
        assert_eq!(medidas_f2_pedido(&[], layers).ha, None);
        // uma chamada 11x a mediana vira cauda
        let mut cauda = blocos.clone();
        cauda[0].ops[10].fim += 0; // sem alterar encadeamento
        let alvo = cauda[0].ops.iter().position(|o| o.op == H_ROUTE && o.camada == 1).unwrap();
        let extra = 100 * 10; // 11x a mediana (100)
        cauda[0].ops[alvo].fim += extra;
        for o in cauda[0].ops.iter_mut().skip(alvo + 1) {
            o.ini += extra;
            o.fim += extra;
        }
        cauda[0].saida += extra;
        cauda[0].t3 += extra;
        let refs: Vec<&BlocoF2> = cauda.iter().collect();
        let m = medidas_f2_pedido(&refs, layers);
        assert!(m.he.unwrap() > 0.0);
    }

    fn proc(b: Braco3, pre: f64, passo: f64) -> ProcessoControleF2 {
        ProcessoControleF2 {
            braco: b, prefill_ms_mediana: pre, passo_ms_mediana: passo,
            imprimiu_bloco_ligado: b != Braco3::D, imprimiu_f2_ligado: b == Braco3::B,
            produziu_bloco: b != Braco3::D, produziu_f2: b == Braco3::B, valido: true,
        }
    }

    fn controle(db: f64) -> Vec<ProcessoControleF2> {
        ORDEM_CONTROLE_F2.iter().map(|&b| proc(b, if b == Braco3::B { 100.0 * (1.0 + db) } else { 100.0 }, 10.0)).collect()
    }

    #[test]
    fn controle_de_tres_bracos_classifica_3_10_e_acima() {
        let c = |db: f64| classifica_controle_f2(&controle(db), true, 22);
        assert_eq!(c(0.0).classe, ClasseControle::Comprovada);
        assert_eq!(c(0.03).classe, ClasseControle::Comprovada, "3% exato");
        assert_eq!(c(0.0301).classe, ClasseControle::ComRessalva);
        assert_eq!(c(-0.10).classe, ClasseControle::ComRessalva, "10% exato, pela magnitude");
        assert_eq!(c(0.1001).classe, ClasseControle::SemProva);
        let r = c(0.05);
        let p = r.principal.unwrap();
        assert!((p.desvio - 0.05).abs() < 1e-9 && (p.ms_por_bloco - 5.0 / 22.0).abs() < 1e-9 && p.desvio_passo.abs() < 1e-12);
        assert!(p.pior_caso >= p.desvio);
        assert!(r.apoio.is_some() && r.classe_apoio == Some(ClasseControle::ComRessalva));
        // o controle com SemProva derruba as classes das H
        assert_eq!(classifica_f2(HipF2::Ha, &MedidasF2 { ha: Some(0.9), ..Default::default() }, &c(0.2).classe), ClasseH::SemProva);
    }

    #[test]
    fn controle_de_tres_bracos_recusa_ordem_contagem_e_linhas_erradas() {
        let inc = |v: &[ProcessoControleF2], orc: bool| matches!(classifica_controle_f2(v, orc, 22).classe, ClasseControle::Incompleto(_));
        assert!(inc(&controle(0.0), false), "orçamento");
        assert!(inc(&controle(0.0)[..6], true), "menos de 9");
        let mut ordem_errada = controle(0.0);
        ordem_errada.swap(0, 1);
        assert!(inc(&ordem_errada, true));
        let mut invalido = controle(0.0);
        invalido[3].valido = false;
        assert!(inc(&invalido, true), "I7 / processo abortado");
        let mut a_com_f2 = controle(0.0);
        a_com_f2[1].imprimiu_f2_ligado = true;
        assert!(inc(&a_com_f2, true), "braço A não imprime perfil-f2: ligado");
        let mut b_sem_f2 = controle(0.0);
        b_sem_f2[2].produziu_f2 = false;
        assert!(inc(&b_sem_f2, true), "braço B sem perfil-f2.json");
        let mut d_com_bloco = controle(0.0);
        d_com_bloco[0].produziu_bloco = true;
        assert!(inc(&d_com_bloco, true));
    }

    #[test]
    fn json_do_pedido_nao_altera_o_esquema_v2_do_perfil_bloco() {
        let h = bloco(2, &[500], 100, 100);
        let b = rec_bloco(&h);
        let p = RecPedido {
            modelo: "m".into(), indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b], servico: vec![None], servico_extra: 0,
            t_armado: 0, t_logits_ini: 0, t_desarmado: 0, prefill_ms: 0.0, decode_passo_ms_mediana: 1.0,
            f2: Some(DadosF2 { layers: 2, hosts: vec![Some(h)], tbs: vec![None], host_extra: 0, tb_extra: 0 }),
            f2b: None,
        };
        let v2 = perfil_json(std::slice::from_ref(&p));
        assert_eq!(v2["versao"], 2);
        assert!(v2["pedidos"][0].get("f2").is_none() && !v2.to_string().contains("H_SEQ"));
        let f2 = perfil_f2_json(std::slice::from_ref(&p));
        assert_eq!((f2["tipo"].as_str(), f2["versao"].as_u64()), (Some("perfil-f2"), Some(1)));
        assert_eq!(f2["pedidos"][0]["blocos"][0]["i3"], true);
        assert_eq!(f2["pedidos"][0]["hi_valida"], false, "sem carimbos do serviço a HI fica inválida, o resto segue");
        assert_eq!(f2["pedidos"][0]["valido"], true);
    }
}

#[cfg(test)]
mod tests_f2_rodada148 {
    use super::*;
    use crate::motor::line::observa_seq;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as O};
    use std::sync::{Arc, Barrier};

    const T2: u64 = 1_000_000;

    fn bloco(layers: usize, d: &[u64]) -> BlocoF2 {
        let mut ordem = vec![(H_ENTRADA, SEM_CAMADA), (H_BEGIN_STEP, SEM_CAMADA), (H_UPLOAD_TABELA, SEM_CAMADA), (H_WAIT_EVENT, SEM_CAMADA)];
        for l in 0..layers as u32 {
            ordem.extend(OPS_CAMADA.map(|o| (o, l)));
        }
        ordem.extend([(H_POST_LAYERS, SEM_CAMADA), (H_SEQ_FIM, SEM_CAMADA), (H_SAIDA, SEM_CAMADA)]);
        let mut t = T2;
        let mut ops = vec![];
        for (i, (op, camada)) in ordem.into_iter().enumerate() {
            ops.push(OpH { op, camada, ini: t, fim: t + d[i % d.len()] });
            t += d[i % d.len()];
        }
        BlocoF2 { gen: 7, m: 8, pos_ini: 0, n_ready: 0, ops, entrada: T2, saida: t, t2: T2, t3: t }
    }

    /// Alonga a operação `idx` em `extra` ns, deslocando o resto da cadeia.
    fn alonga(b: &mut BlocoF2, idx: usize, extra: u64) {
        b.ops[idx].fim += extra;
        for o in b.ops.iter_mut().skip(idx + 1) {
            o.ini += extra;
            o.fim += extra;
        }
        b.saida += extra;
        b.t3 += extra;
    }

    #[test]
    fn tb_obs_nunca_e_nulo_com_escritora_real_em_10000_rodadas() {
        let _ = agora_ns();
        let mut nulos = 0;
        for r in 0..10_000u32 {
            let seq = Arc::new(AtomicU32::new(0));
            let w = seq.clone();
            let h = std::thread::spawn(move || {
                for _ in 0..(r % 7) * 40 {
                    std::hint::spin_loop();
                }
                w.store(1, O::Release);
            });
            let obs = observa_seq::<()>(&seq, 1, false, &mut || true, &mut agora_ns, &mut || {
                std::hint::spin_loop();
                Ok(())
            })
            .unwrap();
            h.join().unwrap();
            if obs.unwrap_or(0) == 0 {
                nulos += 1;
            }
        }
        assert_eq!(nulos, 0, "a leitura que dá o carimbo é a que libera o read_step");
    }

    #[test]
    fn sem_armamento_a_observacao_nao_le_o_relogio() {
        let seq = AtomicU32::new(5);
        let leituras = AtomicUsize::new(0);
        let mut rel = || {
            leituras.fetch_add(1, O::Relaxed);
            42
        };
        let r = observa_seq::<()>(&seq, 5, false, &mut || false, &mut rel, &mut || Ok(())).unwrap();
        assert_eq!((r, leituras.load(O::Relaxed)), (None, 0), "decode/aquecimento: nenhuma leitura de relógio");
        let r = observa_seq::<()>(&seq, 5, false, &mut || true, &mut rel, &mut || Ok(())).unwrap();
        assert_eq!((r, leituras.load(O::Relaxed)), (Some(42), 1));
        // relógio em 0 nunca vira carimbo nulo
        assert_eq!(observa_seq::<()>(&seq, 5, false, &mut || true, &mut || 0, &mut || Ok(())).unwrap(), Some(1));
        // o erro do passo (cancelamento) sai sem carimbo
        let ainda = AtomicU32::new(0);
        assert_eq!(observa_seq::<&str>(&ainda, 1, false, &mut || true, &mut rel, &mut || Err("cancelado")), Err("cancelado"));
    }

    #[test]
    fn i8_acusa_tb_obs_ausente_com_motivo_proprio() {
        let layers = 2;
        let h = bloco(layers, &[1_000]);
        let tb = RecTb { gen: h.gen, camadas: (0..layers).map(|l| {
            let fim = h.fim_seq(l).unwrap();
            TbL { camada: l as u32, gen: h.gen, tb_obs: if l == 1 { 0 } else { fim + 1 }, tb: fim + 5 }
        }).collect() };
        let p = i8_problemas(&h, Some(&tb), layers);
        assert!(p.iter().any(|s| s.contains("tb_obs ausente")), "{p:?}");
        assert!(!p.iter().any(|s| s.contains("anterior")), "o zero não cai no motivo de 'anterior': {p:?}");
    }

    #[test]
    fn rectb_e_publicado_antes_do_recservico_e_o_merge_sem_decode_ve_os_dois() {
        let layers = 2;
        let h = bloco(layers, &[1_000]);
        let pf = Arc::new(Compartilhado::novo_f2(true, layers));
        let a = Armado::arma(Some(pf.clone()), 1);
        pf.registra_bloco(RecBloco { gen: h.gen, m: h.m, pos_ini: 0, t: [T2 - 2000, T2 - 1000, h.t2, h.t3, h.t3 + 500], f0_pendente: false });
        pf.registra_f2_bloco(h.clone());
        let tb = RecTb { gen: h.gen, camadas: (0..layers).map(|l| {
            let fim = h.fim_seq(l).unwrap();
            TbL { camada: l as u32, gen: h.gen, tb_obs: fim + 10, tb: fim + 20 }
        }).collect() };
        let (b1, b2) = (Arc::new(Barrier::new(2)), Arc::new(Barrier::new(2)));
        let (pf2, c1, c2) = (pf.clone(), b1.clone(), b2.clone());
        let srv = RecServico::de_somas(h.gen, T2 - 2000, h.t3 + 400, [0; 7]);
        let t = std::thread::spawn(move || {
            pf2.publica_servico(Some(srv), Some(tb), &mut || {
                c1.wait(); // pausa do serviço entre as duas publicações
                c2.wait();
            });
        });
        b1.wait();
        assert_eq!(trava(&pf.servico).len(), 0, "o marcador de conclusão ainda não existe");
        assert_eq!(trava(&pf.f2_tb).len(), 1, "o RecTb já está publicado quando o RecServico ainda não está");
        b2.wait();
        t.join().unwrap();
        let mut a = a;
        a.logits_ini();
        let d = DadosPedido { modelo: String::new(), indice: 0, aquecimento: false, n: 8, plano: vec![8], prefill_ms: 1.0, decode_passo_ms_mediana: 1.0 };
        let p = a.fecha_com(d, &mut || {}).unwrap();
        let f = p.f2.as_ref().unwrap();
        assert!(!p.servico_aberto());
        assert!(f.problemas_hi(&p.blocos).is_empty(), "{:?}", f.problemas_hi(&p.blocos));
        assert_eq!(f.tbs[0].as_ref().unwrap().gen, h.gen);
    }

    #[test]
    fn he_inclui_as_operacoes_fixas_e_o_limite_de_10x_e_exclusivo() {
        let layers = 2;
        let base: Vec<BlocoF2> = (0..5).map(|_| bloco(layers, &[100])).collect();
        let f2_base = base[0].f2_ns() as f64;
        let he = |bs: &[BlocoF2]| medidas_f2_pedido(&bs.iter().collect::<Vec<_>>(), layers).he;
        assert_eq!(he(&base), Some(0.0));
        // exatamente 10x a mediana: excluída
        let mut dez = base.clone();
        let i = dez[0].ops.iter().position(|o| o.op == H_UPLOAD_TABELA).unwrap();
        alonga(&mut dez[0], i, 900); // 1000 = 10 x 100
        assert_eq!(he(&dez), Some(0.0));
        // cauda em fixa (H_UPLOAD_TABELA): 1001 ns > 10x
        let mut fixa = base.clone();
        alonga(&mut fixa[0], i, 901);
        let esperado = 1001.0 / (4.0 * f2_base + fixa[0].f2_ns() as f64);
        assert!((he(&fixa).unwrap() - esperado).abs() < 1e-12, "{:?} {esperado}", he(&fixa));
        // cauda em H_POST_LAYERS (fixa)
        let mut post = base.clone();
        let j = post[1].ops.iter().position(|o| o.op == H_POST_LAYERS).unwrap();
        alonga(&mut post[1], j, 5_000);
        assert!(he(&post).unwrap() > 0.0);
        // cauda em operação por camada
        let mut cam = base.clone();
        let k = cam[2].ops.iter().position(|o| o.op == H_ROUTE && o.camada == 1).unwrap();
        alonga(&mut cam[2], k, 2_000);
        assert!(he(&cam).unwrap() > 0.0);
        // mais de 20% de F2 numa cauda: classe fora
        let mut grande = base.clone();
        alonga(&mut grande[0], i, 100_000);
        let x = he(&grande);
        assert!(x.unwrap() > 0.20, "{x:?}");
        assert_eq!(classifica_he(x), ClasseH::Fora);
    }

    #[test]
    fn he_e_por_pedido_e_agrega_por_mediana_entre_pedidos() {
        let layers = 2;
        // Pedidos de patamares distintos: A com operações de 100 ns, B com 1.000 ns. Juntos, a mediana do grupo ficaria entre os dois patamares.
        let pa: Vec<BlocoF2> = (0..4).map(|_| bloco(layers, &[100])).collect();
        let pb: Vec<BlocoF2> = (0..4).map(|_| bloco(layers, &[1_000])).collect();
        let (ra, rb): (Vec<&BlocoF2>, Vec<&BlocoF2>) = (pa.iter().collect(), pb.iter().collect());
        assert_eq!(he_do_pedido(&ra), Some(0.0));
        assert_eq!(he_do_pedido(&rb), Some(0.0), "cada pedido é lido contra a própria mediana");
        assert_eq!(he_da_celula(&[ra.clone(), rb.clone()]), Some(0.0));
        // misturados num conjunto só, o valor mudaria: a mediana do grupo (op, camada) passa a ser ~550 e as chamadas de 1.000 ainda não passam de 10x; o patamar
        // de 100 também não. Para provar a diferença, um pedido com uma cauda de 2.000 ns sobre base de 100 ns:
        let mut pc = pa.clone();
        let i = pc[0].ops.iter().position(|o| o.op == H_UPLOAD_TABELA).unwrap();
        alonga(&mut pc[0], i, 1_900);
        let rc: Vec<&BlocoF2> = pc.iter().collect();
        let hc = he_do_pedido(&rc).unwrap();
        assert!(hc > 0.0);
        // junto com um pedido de patamar alto, a mediana pooled do grupo sobe e a cauda de 2.000 deixa de passar de 10x: por isso a HE é por pedido
        let pooled: Vec<&BlocoF2> = rc.iter().chain(rb.iter()).copied().collect();
        assert_eq!(he_do_pedido(&pooled), Some(0.0), "pooled esconde a cauda");
        // célula com três pedidos (cauda, limpo, limpo): a mediana entre pedidos é 0; com dois pedidos de cauda e um limpo, é > 0
        assert_eq!(he_da_celula(&[rc.clone(), ra.clone(), rb.clone()]), Some(0.0));
        assert!(he_da_celula(&[rc.clone(), rc.clone(), ra.clone()]).unwrap() > 0.0);
        // o calculador da célula chama HE por pedido e o resto sobre o conjunto
        let m = medidas_f2_celula(&[rc.clone(), rb.clone()], layers);
        assert_eq!(m.he, he_da_celula(&[rc, rb]));
        assert!(m.ha.is_some());
    }

    #[test]
    fn celula_preserva_os_pedidos_em_ha_a_hd() {
        let layers = 1;
        // Exemplo do parecer: 9 pedidos de 20 blocos, F2 = 100. Em 6 pedidos, 11 blocos com HA = 90 e 9 com HA = 1; nos outros 3, todos com HA = 1.
        let alto = [2u64, 1, 1, 1, 90, 0, 1, 0, 1, 0, 1, 1, 1]; // HA = pre + route + launch + addy = 90; soma 100
        let baixo = [91u64, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1]; // HA = 1; soma 100
        assert_eq!((alto.iter().sum::<u64>(), baixo.iter().sum::<u64>()), (100, 100));
        let mut dados: Vec<Vec<BlocoF2>> = vec![];
        for p in 0..9 {
            dados.push((0..20).map(|i| bloco(layers, if p < 6 && i < 11 { &alto } else { &baixo })).collect());
        }
        let pedidos: Vec<Vec<&BlocoF2>> = dados.iter().map(|v| v.iter().collect()).collect();
        let m = medidas_f2_celula(&pedidos, layers);
        assert!((m.ha.unwrap() - 0.90).abs() < 1e-9, "{:?}", m.ha);
        assert_eq!(classifica_ha(m.ha), ClasseH::Dentro);
        // misturando os 180 blocos (o que o parecer apontou), a mediana cairia para 0,01: a função de pedido, chamada de propósito com tudo junto, mostra a diferença
        let todos: Vec<&BlocoF2> = pedidos.iter().flatten().copied().collect();
        assert!((medidas_f2_pedido(&todos, layers).ha.unwrap() - 0.01).abs() < 1e-9);
    }

    #[test]
    fn celula_hb_hc_hd_mesma_quantidade_de_blocos_e_distribuicoes_internas_diferentes() {
        let layers = 2;
        // Camada 1 com a primeira operação (PRE) valendo x; o resto 100. P1 e P2: [5100, 5100, 100]; P3: [100, 100, 100].
        let mk = |x: u64| {
            let mut d = vec![100u64; 19];
            d[10] = x;
            bloco(layers, &d)
        };
        let dados: Vec<Vec<BlocoF2>> = vec![
            vec![mk(5100), mk(5100), mk(100)],
            vec![mk(5100), mk(5100), mk(100)],
            vec![mk(100), mk(100), mk(100)],
        ];
        let pedidos: Vec<Vec<&BlocoF2>> = dados.iter().map(|v| v.iter().collect()).collect();
        let m = medidas_f2_celula(&pedidos, layers);
        let por: Vec<MedidasF2> = pedidos.iter().map(|p| medidas_f2_pedido(p, layers)).collect();
        let esperado = |f: &dyn Fn(&MedidasF2) -> f64| mediana(&por.iter().map(f).collect::<Vec<_>>());
        assert!((m.hd.unwrap() - esperado(&|x| x.hd.unwrap())).abs() < 1e-12);
        assert!((m.hb_frac.unwrap() - esperado(&|x| x.hb_frac.unwrap())).abs() < 1e-12);
        assert!((m.hc.unwrap() - esperado(&|x| x.hc.unwrap())).abs() < 1e-12);
        let todos: Vec<&BlocoF2> = pedidos.iter().flatten().copied().collect();
        let junto = medidas_f2_pedido(&todos, layers);
        assert!(m.hd.unwrap() > 9.0 && junto.hd.unwrap() < 1.5, "por pedido {:?}, misturado {:?}", m.hd, junto.hd);
        assert_ne!(m.hb_frac, junto.hb_frac);
        assert_eq!(m.hb_maior, Some(true), "PRE tem a maior soma pelos valores por tipo preservados");
    }

    #[test]
    fn planejamento_da_janela_3b_cortes_e_tetos() {
        // 3B: 200 s decorridos, duas células restantes (curto 90, longo 120), folga 180: 590 s, cabe com cinco medidas.
        let r = [teto_processo_s(true, CelulaF2::Curto), teto_processo_s(true, CelulaF2::Longo)];
        assert_eq!(r, [90, 120]);
        assert_eq!(200 + r.iter().sum::<u64>() + FOLGA_S, 590);
        assert_eq!(planeja_celula(200, &r, FOLGA_S, TETO_JANELA_S, false), PlanoCelula::Cinco);
        // saldo insuficiente para cinco: 5 s a mais que o limite com N=5 (decorrido 871: 871 + 210 + 180 = 1261 > 1200); com N=3 (140 + ...) = 871 + 140 + 180 = 1191 cabe
        assert_eq!(planeja_celula(871, &r, FOLGA_S, TETO_JANELA_S, false), PlanoCelula::Tres);
        assert_eq!(estimativa_celula_s(90, 3) + estimativa_celula_s(120, 3), 60 + 80);
        // NÃO RODOU só pela célula atual com 3: 900 + 60 + 180 = 1140 cabe (Tres); 961 + 60 + 180 = 1201 não (NaoRodou)
        assert_eq!(planeja_celula(900, &r, FOLGA_S, TETO_JANELA_S, false), PlanoCelula::Tres);
        assert_eq!(planeja_celula(961, &r, FOLGA_S, TETO_JANELA_S, false), PlanoCelula::NaoRodou);
        // corte único e permanente: com folga sobrando, ja_cortado nunca volta a Cinco
        assert_eq!(planeja_celula(200, &r, FOLGA_S, TETO_JANELA_S, true), PlanoCelula::Tres);
        assert_eq!(planeja_celula(200, &r, FOLGA_S, TETO_JANELA_S, false), PlanoCelula::Cinco);
        // limite exato: soma == teto global cabe
        assert_eq!(planeja_celula(1200 - 210 - 180, &r, FOLGA_S, TETO_JANELA_S, false), PlanoCelula::Cinco);
        assert_eq!(planeja_celula(1200 - 210 - 180 + 1, &r, FOLGA_S, TETO_JANELA_S, false), PlanoCelula::Tres);
        // a estimativa por N nunca passa do teto do processo
        for t in [60u64, 75, 90, 120] {
            assert!(estimativa_celula_s(t, 3) <= t && estimativa_celula_s(t, 5) == t);
        }
        // tetos dos processos: 60/90 médio e curto, 75/120 longo, separados do teto global
        assert_eq!([60, 60, 75], [(false, CelulaF2::Medio), (false, CelulaF2::Curto), (false, CelulaF2::Longo)].map(|(m, c)| teto_processo_s(m, c)));
        assert_eq!([90, 90, 120], [(true, CelulaF2::Medio), (true, CelulaF2::Curto), (true, CelulaF2::Longo)].map(|(m, c)| teto_processo_s(m, c)));
        assert_eq!((TETO_JANELA_S, FOLGA_S, TETO_SMOKE_S), (1200, 180, 60));
        // controle: nunca cortado; 9 processos de 90 s + smoke 60 s + folga cabem (1B e 3B); com decorrido alto, incompleto (não há corte)
        let nove = [90u64; 9];
        assert!(controle_cabe(TETO_SMOKE_S, &nove, FOLGA_S, TETO_JANELA_S));
        assert!(!controle_cabe(400, &nove, FOLGA_S, TETO_JANELA_S));
        assert!(!controle_cabe(0, &nove[..8], FOLGA_S, TETO_JANELA_S), "só vale com os 9 processos");
    }
}


// ---- Testes sem GPU da F2 nível B -----------------------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests_f2b {
    use super::*;

    const T2: u64 = 1_000_000;
    const REG: u64 = 20;
    const GAP: u64 = 5;

    /// A ordem das operações de uma posição, na sequência real de `attn_block_at_b`: SETUP, GENERICS, 4 LANCA, 2 DTOD, 2 LANCA, 3 DTOD, ATTN_PREP,
    /// 2 LANCA, 1 DTOD, 3 LANCA, SETUP (11 LANCA, 6 DTOD).
    fn ordem_pos() -> Vec<u8> {
        let mut v = vec![B_SETUP, B_GENERICS];
        v.extend([B_LANCA; 4]);
        v.extend([B_DTOD; 2]);
        v.extend([B_LANCA; 2]);
        v.extend([B_DTOD; 3]);
        v.push(B_ATTN_PREP);
        v.extend([B_LANCA; 2]);
        v.push(B_DTOD);
        v.extend([B_LANCA; 3]);
        v.push(B_SETUP);
        v
    }

    /// Duração mínima de H_PRE_LAYER para caber os registros de uma camada com `m` posições.
    fn pre_min(m: usize) -> u64 {
        GAP + REG + GAP + m as u64 * 21 * (REG + GAP) + GAP
    }

    /// Host sintético (encadeado, como o do nível A): `pre(l)` dá a duração de H_PRE_LAYER da camada `l`; as demais operações duram 100 ns.
    fn host(layers: usize, m: usize, pre: &dyn Fn(usize) -> u64) -> BlocoF2 {
        let entrada = T2 + 50;
        let mut t = entrada;
        let mut ops = vec![];
        let mut push = |op: u8, camada: u32, d: u64, ops: &mut Vec<OpH>, t: &mut u64| {
            ops.push(OpH { op, camada, ini: *t, fim: *t + d });
            *t += d;
        };
        for (op, d) in [(H_ENTRADA, 100)] {
            push(op, SEM_CAMADA, d, &mut ops, &mut t);
        }
        for op in [H_BEGIN_STEP, H_UPLOAD_TABELA, H_WAIT_EVENT] {
            push(op, SEM_CAMADA, 100, &mut ops, &mut t);
        }
        for l in 0..layers {
            for op in OPS_CAMADA {
                let d = if op == H_PRE_LAYER { pre(l) } else { 100 };
                push(op, l as u32, d, &mut ops, &mut t);
            }
        }
        for op in [H_POST_LAYERS, H_SEQ_FIM, H_SAIDA] {
            push(op, SEM_CAMADA, 100, &mut ops, &mut t);
        }
        BlocoF2 { gen: 7, m, pos_ini: 16, n_ready: 0, ops, entrada, saida: t, t2: T2, t3: t + 50 }
    }

    /// Registros brutos dentro de cada H_PRE_LAYER do host, na ordem do código.
    fn regs(h: &BlocoF2, layers: usize, m: usize) -> BlocoB {
        let mut v = vec![];
        for l in 0..layers {
            let pre = h.ops.iter().find(|o| o.op == H_PRE_LAYER && o.camada == l as u32).unwrap();
            let mut t = pre.ini + GAP;
            v.push(RegB { ini: t, fim: t + REG, camada: l as u16, pos: SEM_POS, op: B_LOCK });
            t += REG + GAP;
            for p in 0..m {
                for op in ordem_pos() {
                    v.push(RegB { ini: t, fim: t + REG, camada: l as u16, pos: p as u16, op });
                    t += REG + GAP;
                }
            }
        }
        BlocoB { gen: h.gen, m, pos_ini: h.pos_ini, regs: v, perdidos: 0 }
    }

    fn rec_bloco(h: &BlocoF2) -> RecBloco {
        RecBloco { gen: h.gen, m: h.m, pos_ini: h.pos_ini, t: [h.t2 - 2000, h.t2 - 1000, h.t2, h.t3, h.t3 + 500], f0_pendente: false }
    }

    fn dados_f2(h: &BlocoF2, layers: usize) -> DadosF2 {
        DadosF2 { layers, hosts: vec![Some(h.clone())], tbs: vec![None], host_extra: 0, tb_extra: 0 }
    }

    fn valida(layers: usize, _m: usize, h: &BlocoF2, b: &BlocoB) -> Vec<String> {
        DadosF2B { layers, blocos: vec![Some(b.clone())], extra: 0 }.problemas(&[rec_bloco(h)], Some(&dados_f2(h, layers)))
    }

    #[test]
    fn variavel_f2b_so_liga_com_o_valor_um_e_exige_bloco_e_f2() {
        assert_eq!(VARIAVEL_F2B, "LASTRO_PERFIL_F2B");
        assert!(interpreta_f2b(Some("1")));
        for v in [None, Some(""), Some("0"), Some("true"), Some(" 1"), Some("1 "), Some("11"), Some("on")] {
            assert!(!interpreta_f2b(v), "{v:?}");
        }
        for (b, f, x) in [(false, true, true), (true, false, true), (true, true, false), (false, false, false)] {
            assert!(!f2b_efetivo(b, f, x));
        }
        assert!(f2b_efetivo(true, true, true));
        assert_eq!(linha_estado_f2b(true, true, true), "perfil-f2b: ligado");
        assert_eq!(linha_estado_f2b(true, false, true), "perfil-f2b: desligado (exige perfil-bloco e perfil-f2)");
        assert_eq!(linha_estado_f2b(false, true, true), "perfil-f2b: desligado (exige perfil-bloco e perfil-f2)");
        assert_eq!(linha_estado_f2b(true, true, false), "perfil-f2b: desligado");
        assert_eq!(linha_estado_f2b(false, false, false), "perfil-f2b: desligado");
        // O acumulador do compartilhado só existe com as duas outras.
        assert!(!Compartilhado::novo_f2b(false, true, 4).f2b());
        assert!(Compartilhado::novo_f2b(true, true, 4).f2b());
        assert!(!Compartilhado::novo_f2b(true, false, 4).f2b());
    }

    #[test]
    fn acumulador_tem_capacidade_fixa_registro_de_24_bytes_e_nunca_realoca() {
        assert_eq!(std::mem::size_of::<RegB>(), 24);
        assert_eq!(capacidade_f2b(24, 8), 24 * 169);
        let mut a = AccB::com_capacidade(2, 8);
        let cap = a.capacidade();
        assert!(cap >= capacidade_f2b(2, 8));
        for bloco in 0..3 {
            a.reinicia();
            for l in 0..2u16 {
                a.contexto(l, SEM_POS);
                a.reg(B_LOCK, agora_ns());
                for p in 0..8u16 {
                    a.contexto(l, p);
                    for op in ordem_pos() {
                        a.reg(op, agora_ns());
                    }
                }
            }
            assert_eq!(a.regs.len(), capacidade_f2b(2, 8), "bloco {bloco}");
            assert_eq!(a.perdidos, 0);
            assert_eq!(a.capacidade(), cap, "capacidade igual depois do bloco {bloco}");
        }
        // Capacidade máxima: o excesso é contado e não realoca; a IB3 invalida.
        a.reg(B_LOCK, agora_ns());
        assert_eq!((a.regs.len(), a.perdidos, a.capacidade()), (capacidade_f2b(2, 8), 1, cap));
        assert!(!a.para_bloco(1, 8, 0).ib3(2));
    }

    #[test]
    fn option_vazio_nao_le_relogio_nem_grava() {
        // Prova só o que o plano promete: com `Option` vazio a sonda devolve 0 (sem relógio) e `b_fim` não faz nada.
        let mut vazio: Option<&mut AccB> = None;
        assert_eq!(b_ini(&vazio), 0);
        b_fim(&mut vazio, B_LANCA, 0);
        assert!(vazio.is_none());
        let mut a = AccB::com_capacidade(1, 8);
        let mut some = Some(&mut a);
        let t = b_ini(&some);
        assert!(t > 0 || agora_ns() >= t);
        b_fim(&mut some, B_LANCA, t);
        assert_eq!(a.regs.len(), 1);
        assert_eq!(a.regs[0].op, B_LANCA);
    }

    #[test]
    fn bloco_integro_passa_ib1_a_ib5_e_a_contagem_e_a_do_plano() {
        for (layers, m) in [(2usize, 8usize), (3, 2), (2, 1)] {
            let h = host(layers, m, &|_| pre_min(m) + 40);
            let b = regs(&h, layers, m);
            assert_eq!(ordem_pos().len(), 21);
            assert_eq!(b.regs.len(), layers * (21 * m + 1));
            assert!(b.ib1(&h) && b.ib2(&h, layers) && b.ib3(layers));
            assert!(valida(layers, m, &h, &b).is_empty(), "{:?}", valida(layers, m, &h, &b));
            for l in 0..layers {
                for op in 0..6u8 {
                    assert_eq!(b.n_op_camada(l, op), contagem_esperada_b(op, m));
                }
                assert!(b.residuo_camada_ns(l, &h).unwrap() >= 0);
            }
            assert!(b.razao_residuo(&h, layers) < 0.5);
        }
    }

    #[test]
    fn ib3_recusa_contagem_igual_com_camada_repetida_faltando_ou_fora_de_ordem() {
        let (layers, m) = (3usize, 2usize);
        let h = host(layers, m, &|_| pre_min(m) + 40);
        let b = regs(&h, layers, m);
        // Falta e sobra um registro de outra operação: soma igual, contagens diferentes.
        let mut troca = b.clone();
        let i = troca.regs.iter().position(|r| r.op == B_DTOD).unwrap();
        troca.regs[i].op = B_LANCA;
        assert_eq!(troca.regs.len(), b.regs.len());
        assert!(!troca.ib3(layers));
        // Camada repetida no lugar de outra (mesmo total).
        let mut rep = b.clone();
        let n = rep.regs.len();
        for r in rep.regs.iter_mut().skip(n - (21 * m + 1)) {
            r.camada = 1;
        }
        assert!(!rep.ib3(layers));
        // Um registro a menos, um a mais, um perdido.
        let mut menos = b.clone();
        menos.regs.pop();
        assert!(!menos.ib3(layers));
        let mut mais = b.clone();
        mais.regs.push(*mais.regs.last().unwrap());
        assert!(!mais.ib3(layers));
        let mut perdido = b.clone();
        perdido.perdidos = 1;
        assert!(!perdido.ib3(layers));
        // Camadas fora de ordem.
        let mut ordem = b.clone();
        let k = 21 * m + 1;
        let (a, c) = (ordem.regs[..k].to_vec(), ordem.regs[k..2 * k].to_vec());
        ordem.regs[..k].clone_from_slice(&c);
        ordem.regs[k..2 * k].clone_from_slice(&a);
        assert!(!ordem.ib3(layers));
        // Posição fora de `m` e `B_LOCK` com posição.
        let mut pos = b.clone();
        pos.regs[3].pos = m as u16;
        assert!(!pos.ib3(layers));
        let mut lock = b.clone();
        lock.regs[0].pos = 0;
        assert!(!lock.ib3(layers));
        // Mais camadas no bloco do que no motor.
        assert!(!b.ib3(layers - 1) && !b.ib3(layers + 1));
    }

    #[test]
    fn ib3_valida_a_sequencia_por_posicao_e_recusa_posicoes_trocadas_com_contagem_igual() {
        // Íntegro: m = 1, restos (2, 7) e m = 8.
        for (layers, m) in [(2usize, 1usize), (2, 2), (2, 7), (2, 8)] {
            let h = host(layers, m, &|_| pre_min(m) + 40);
            let b = regs(&h, layers, m);
            assert_eq!(ORDEM_POS_B.to_vec(), ordem_pos());
            assert!(b.ib3(layers), "m = {m}");
            assert!(valida(layers, m, &h, &b).is_empty());
        }
        let (layers, m) = (2usize, 8usize);
        let h = host(layers, m, &|_| pre_min(m) + 40);
        let b = regs(&h, layers, m);
        // Mutante: todas as posições não sentinela viram zero (totais, carimbos e identidade do bloco intactos).
        let mut zero = b.clone();
        for r in zero.regs.iter_mut().filter(|r| r.op != B_LOCK) {
            r.pos = 0;
        }
        assert_eq!(zero.regs.len(), b.regs.len());
        assert_eq!((0..layers).map(|l| zero.soma_camada_ns(l)).collect::<Vec<_>>(), (0..layers).map(|l| b.soma_camada_ns(l)).collect::<Vec<_>>());
        assert!(!zero.ib3(layers));
        assert!(valida(layers, m, &h, &zero).iter().any(|s| s.contains("IB3")));
        let rec = RecBloco { gen: h.gen, m, pos_ini: h.pos_ini, t: [h.t2 - 2000, h.t2 - 1000, h.t2, h.t3, h.t3 + 500], f0_pendente: false };
        let p = RecPedido {
            modelo: "x".into(), indice: 0, aquecimento: false, n: m, plano: vec![m], blocos: vec![rec], servico: vec![None], servico_extra: 0,
            t_armado: 0, t_logits_ini: 0, t_desarmado: 0, prefill_ms: 0.0, f2: Some(dados_f2(&h, layers)),
            f2b: Some(DadosF2B { layers, blocos: vec![Some(zero)], extra: 0 }), decode_passo_ms_mediana: 1.0,
        };
        assert_eq!(perfil_f2b_json(&[p])["pedidos"][0]["valido"], false);
        // Duas posições repetidas no lugar de outras (posição 3 vira 2, posição 5 vira 4, em todas as camadas).
        let mut rep = b.clone();
        for r in rep.regs.iter_mut().filter(|r| r.op != B_LOCK) {
            if r.pos == 3 {
                r.pos = 2;
            } else if r.pos == 5 {
                r.pos = 4;
            }
        }
        assert!(!rep.ib3(layers));
        // Fora de ordem: posições 1 e 2 trocam de rótulo.
        let mut ord = b.clone();
        for r in ord.regs.iter_mut().filter(|r| r.op != B_LOCK) {
            if r.pos == 1 {
                r.pos = 2;
            } else if r.pos == 2 {
                r.pos = 1;
            }
        }
        assert!(!ord.ib3(layers));
        // Um registro da posição 0 rotulado como da posição 1 (contagem por operação igual).
        let mut um = b.clone();
        um.regs[3].pos = 1;
        assert!(!um.ib3(layers));
        // O `m` do bloco maior que o do registro: a estrutura não casa.
        let mut mm = b.clone();
        mm.m = 7;
        assert!(!mm.ib3(layers));
    }

    #[test]
    fn ib1_recusa_ordem_sobreposicao_e_contencao_diferentes_com_soma_e_contagem_iguais() {
        let (layers, m) = (2usize, 2usize);
        let h = host(layers, m, &|_| pre_min(m) + 40);
        let b = regs(&h, layers, m);
        assert!(b.ib1(&h));
        // Ordem: troca dois registros vizinhos de lugar (mesma soma, mesma contagem).
        let mut o = b.clone();
        o.regs.swap(5, 6);
        assert_eq!(o.soma_camada_ns(0), b.soma_camada_ns(0));
        assert!(o.ib3(layers), "a contagem continua a do plano: só a ordem mudou");
        assert!(!o.ib1(&h));
        // Sobreposição: o início de um registro antes do fim do anterior.
        let mut s = b.clone();
        s.regs[7].ini = s.regs[6].fim - 1;
        assert!(!s.ib1(&h));
        // Fim antes do início.
        let mut inv = b.clone();
        inv.regs[4].fim = inv.regs[4].ini - 1;
        assert!(!inv.ib1(&h));
        // Contenção: desloca tudo da camada 0 para fora do `H_PRE_LAYER` (soma e contagem iguais).
        let mut fora = b.clone();
        let pre0 = h.ops.iter().find(|o| o.op == H_PRE_LAYER && o.camada == 0).unwrap();
        let k = 21 * m + 1;
        let desloc = pre0.dur() + 500;
        for r in fora.regs[..k].iter_mut() {
            r.ini += desloc;
            r.fim += desloc;
        }
        // O deslocamento passa por cima da camada 1: a ordem global também quebra, e a contenção da camada 0 falha por si.
        assert!(!fora.ib1(&h));
        let mut alem = b.clone();
        alem.regs[k - 1].fim = pre0.fim + 2 * TOL_RELOGIO_NS;
        assert!(!alem.ib1(&h));
        // Tolerância de 1 µs na borda vale.
        let mut tol = b.clone();
        tol.regs[0].ini = pre0.ini.saturating_sub(TOL_RELOGIO_NS);
        assert!(tol.ib1(&h));
        // Registros sem o `H_PRE_LAYER` da camada, ou nenhum registro.
        let mut sem = h.clone();
        sem.ops.retain(|o| !(o.op == H_PRE_LAYER && o.camada == 1));
        assert!(!b.ib1(&sem));
        assert!(!BlocoB { regs: vec![], ..b.clone() }.ib1(&h));
    }

    #[test]
    fn ib2_aceita_residuo_pequeno_e_recusa_excesso_acima_de_1_us() {
        let (layers, m) = (2usize, 1usize);
        let h = host(layers, m, &|_| pre_min(m) + 3000);
        let b = regs(&h, layers, m);
        assert!(b.residuo_camada_ns(0, &h).unwrap() >= 3000 - GAP as i64 - 10);
        assert!(b.ib2(&h, layers));
        // Soma maior que H_PRE_LAYER por mais de 1 µs: resíduo negativo invalida; nunca é zerado.
        let mut e = regs(&h, layers, m);
        e.regs[2].fim += 5 * TOL_RELOGIO_NS;
        assert!(e.residuo_camada_ns(0, &h).unwrap() < 3000);
        let mut gigante = b.clone();
        let l0 = gigante.regs.iter().position(|r| r.camada == 0 && r.op == B_SETUP).unwrap();
        gigante.regs[l0].fim += 10_000;
        assert!(gigante.residuo_camada_ns(0, &h).unwrap() < -(TOL_RELOGIO_NS as i64) + 1);
        assert!(!gigante.ib2(&h, layers));
        // IB5: ressalva acima de 3%, nunca invalida.
        let largo = host(layers, m, &|_| 100_000);
        let bl = regs(&largo, layers, m);
        assert!(bl.razao_residuo(&largo, layers) > LIM_COBERTURA_HOST);
        let d = DadosF2B { layers, blocos: vec![Some(bl.clone())], extra: 0 };
        assert!(d.ressalva_residuo(Some(&dados_f2(&largo, layers))));
        assert!(valida(layers, m, &largo, &bl).is_empty());
        // Registros contíguos e H_PRE_LAYER justo: resíduo pequeno, sem ressalva.
        let (l2, m2) = (2usize, 8usize);
        let h2 = host(l2, m2, &|_| pre_min(m2) + 40);
        let mut b2 = regs(&h2, l2, m2);
        for i in 0..b2.regs.len() - 1 {
            if b2.regs[i].camada == b2.regs[i + 1].camada {
                b2.regs[i].fim = b2.regs[i + 1].ini;
            }
        }
        assert!(b2.razao_residuo(&h2, l2) <= LIM_COBERTURA_HOST, "{}", b2.razao_residuo(&h2, l2));
        assert!(b2.ib1(&h2) && b2.ib2(&h2, l2) && b2.ib3(l2));
        assert!(!DadosF2B { layers: l2, blocos: vec![Some(b2)], extra: 0 }.ressalva_residuo(Some(&dados_f2(&h2, l2))));
    }

    #[test]
    fn juncao_com_outra_geracao_camada_ou_soma_diferente_invalida_pelo_ib4() {
        let (layers, m) = (2usize, 2usize);
        let h = host(layers, m, &|_| pre_min(m) + 40);
        let b = regs(&h, layers, m);
        assert!(valida(layers, m, &h, &b).is_empty());
        let mut g = b.clone();
        g.gen += 1;
        assert!(valida(layers, m, &h, &g).iter().any(|s| s.contains("IB4")));
        let mut p = b.clone();
        p.pos_ini += 8;
        assert!(valida(layers, m, &h, &p).iter().any(|s| s.contains("IB4")));
        // Registros de uma camada na posição de outra: a contenção do `H_PRE_LAYER` da camada acusa (IB1).
        let mut c = b.clone();
        let k = 21 * m + 1;
        for r in c.regs[..k].iter_mut() {
            r.camada = 1;
        }
        for r in c.regs[k..].iter_mut() {
            r.camada = 0;
        }
        assert!(!valida(layers, m, &h, &c).is_empty());
        // F2 do host com soma que não fecha t3 - t2 (I2 do nível A): IB4.
        let mut ruim = h.clone();
        ruim.ops.iter_mut().find(|o| o.op == H_ROUTE).unwrap().fim += 10 * TOL_RELOGIO_NS;
        assert!(valida(layers, m, &ruim, &b).iter().any(|s| s.contains("IB4")));
        // Sem F2 no pedido, sem registro do bloco, blocos a menos.
        assert!(DadosF2B { layers, blocos: vec![Some(b.clone())], extra: 0 }.problemas(&[rec_bloco(&h)], None).iter().any(|s| s.contains("sem a F2")));
        assert!(!DadosF2B { layers, blocos: vec![None], extra: 0 }.problemas(&[rec_bloco(&h)], Some(&dados_f2(&h, layers))).is_empty());
        assert!(!DadosF2B { layers, blocos: vec![], extra: 0 }.problemas(&[rec_bloco(&h)], Some(&dados_f2(&h, layers))).is_empty());
        assert!(!DadosF2B { layers, blocos: vec![Some(b.clone())], extra: 1 }.problemas(&[rec_bloco(&h)], Some(&dados_f2(&h, layers))).is_empty());
    }

    #[test]
    fn json_do_pedido_traz_os_registros_brutos_e_o_rotulo_do_lanca() {
        let (layers, m) = (2usize, 1usize);
        let h = host(layers, m, &|_| pre_min(m) + 40);
        let b = regs(&h, layers, m);
        let p = RecPedido {
            modelo: "x".into(), indice: 0, aquecimento: false, n: 1, plano: vec![1], blocos: vec![rec_bloco(&h)], servico: vec![None], servico_extra: 0,
            t_armado: 0, t_logits_ini: 0, t_desarmado: 0, prefill_ms: 0.0, f2: Some(dados_f2(&h, layers)),
            f2b: Some(DadosF2B { layers, blocos: vec![Some(b.clone())], extra: 0 }), decode_passo_ms_mediana: 1.0,
        };
        let j = perfil_f2b_json(&[p.clone()]);
        assert_eq!(j["tipo"], "perfil-f2b");
        assert_eq!(j["versao"], 1);
        assert_eq!(j["b_lanca_rotulo"], "tempo da chamada de envio de trabalho (a expressão async_on)");
        let ped = &j["pedidos"][0];
        assert_eq!(ped["valido"], true);
        let regs_json = ped["blocos"][0]["regs"].as_array().unwrap();
        assert_eq!(regs_json.len(), b.regs.len());
        assert_eq!(regs_json[0][2], Value::Null, "B_LOCK sem posição");
        assert_eq!(regs_json[0][0], B_LOCK);
        assert!(!j.to_string().contains("cuLaunchKernel"), "o relatório do plano não fala em cuLaunchKernel");
        let alt = perfil_f2b_json_com(&[p.clone()], RotuloLanca::ComCuLaunchKernel);
        assert!(alt.to_string().contains("cuLaunchKernel"));
        // Sem F2B no pedido, nada entra.
        let mut sem = p;
        sem.f2b = None;
        assert!(perfil_f2b_json(&[sem])["pedidos"].as_array().unwrap().is_empty());
        // O arquivo do nível A não ganha campos do nível B.
        assert!(!perfil_f2_json(&[]).to_string().contains("B_LOCK"));
    }

    fn bloco_com_pre(layers: usize, m: usize, pre: u64, gen: u32) -> (BlocoB, BlocoF2) {
        let mut h = host(layers, m, &|_| pre);
        h.gen = gen;
        let mut b = regs(&h, layers, m);
        b.gen = gen;
        (b, h)
    }

    #[test]
    fn parcelas_hb1_a_hb4_somam_no_maximo_um_e_usam_mediana_por_camada() {
        let (layers, m) = (2usize, 2usize);
        let pre = pre_min(m) + 100;
        let blocos: Vec<_> = (0..5).map(|g| bloco_com_pre(layers, m, pre + g * 10, g as u32)).collect();
        let cheios: Vec<(&BlocoB, &BlocoF2)> = blocos.iter().map(|(b, h)| (b, h)).collect();
        let r = medidas_f2b_pedido(&cheios, layers);
        let (p1, p2, p3, p4) = (r.hb1.unwrap(), r.hb2.unwrap(), r.hb3.unwrap(), r.hb4.unwrap());
        assert!(p1 > 0.0 && p2 > 0.0 && p3 > 0.0 && p4 > 0.0);
        assert!(p1 + p2 + p3 + p4 <= 1.0 + 1e-9, "as parcelas somadas não passam de 1 (o resíduo é o resto)");
        // 11 de 21 registros por posição são B_LANCA e todos têm a mesma duração.
        assert!((p3 / p4 - 11.0 / 6.0).abs() < 1e-9);
        assert_eq!(r.h_pre_layer_mediana_camada_ns.len(), layers);
        assert!(r.h_pre_layer_mediana_camada_ns.iter().all(|&x| x == (pre + 20) as f64));
        assert!(r.hb5_p90_sobre_mediana.unwrap() >= 1.0);
        assert_eq!(r.hb5_dispersao_camadas, Some(1.0));
        assert_eq!(medidas_f2b_pedido(&[], layers), MedidasF2B::default());
        // Célula: mediana entre os pedidos (um pedido muito diferente não arrasta).
        let outro: Vec<_> = (0..3).map(|g| bloco_com_pre(layers, m, 50_000, g as u32)).collect();
        let c2: Vec<(&BlocoB, &BlocoF2)> = outro.iter().map(|(b, h)| (b, h)).collect();
        let cel = medidas_f2b_celula(&[cheios.clone(), cheios.clone(), c2], layers);
        assert_eq!(cel.hb3, r.hb3);
    }

    #[test]
    fn hb3_usa_os_limiares_do_plano_por_padrao_e_aceita_outros() {
        assert_eq!(LIMIARES_HB3_PLANO, LimiaresHb3 { dentro: 0.60, fora: 0.40 });
        let c = |x: f64| classifica_hb3(Some(x), &LIMIARES_HB3_PLANO);
        assert_eq!((c(0.60), c(0.80)), (ClasseH::Dentro, ClasseH::Dentro));
        assert_eq!((c(0.39), c(0.0)), (ClasseH::Fora, ClasseH::Fora));
        assert_eq!((c(0.40), c(0.59)), (ClasseH::SemLeitura, ClasseH::SemLeitura));
        assert_eq!(classifica_hb3(None, &LIMIARES_HB3_PLANO), ClasseH::SemDado);
        assert_eq!(classifica_hb3(Some(f64::NAN), &LIMIARES_HB3_PLANO), ClasseH::SemDado);
        let outros = LimiaresHb3 { dentro: 0.5, fora: 0.2 };
        assert_eq!(classifica_hb3(Some(0.5), &outros), ClasseH::Dentro);
        assert_eq!(classifica_hb3(Some(0.3), &outros), ClasseH::SemLeitura);
        assert_eq!(classifica_hb3(Some(0.1), &outros), ClasseH::Fora);
    }

    #[test]
    fn classes_descritivas_e_sem_prova_com_controle_acima_de_10_por_cento() {
        let m = MedidasF2B { hb1: Some(0.01), hb2: Some(0.2), hb3: Some(0.7), hb4: Some(0.1), hb5_p90_sobre_mediana: Some(1.2), hb5_dispersao_camadas: Some(1.1),
                             h_pre_layer_mediana_camada_ns: vec![1.0] };
        let ok = ClasseControle::Comprovada;
        for h in [HipB::Hb1, HipB::Hb2, HipB::Hb4, HipB::Hb5, HipB::Hb6] {
            assert_eq!(classifica_f2b(h, &m, &ok, &LIMIARES_HB3_PLANO), ClasseH::Descritivo, "{h:?}");
            assert_eq!(classifica_f2b(h, &MedidasF2B::default(), &ok, &LIMIARES_HB3_PLANO), ClasseH::SemDado, "{h:?}");
            assert_eq!(classifica_f2b(h, &m, &ClasseControle::SemProva, &LIMIARES_HB3_PLANO), ClasseH::SemProva);
            assert_eq!(classifica_f2b(h, &m, &ClasseControle::Incompleto("x".into()), &LIMIARES_HB3_PLANO), ClasseH::SemProva);
        }
        assert_eq!(classifica_f2b(HipB::Hb3, &m, &ok, &LIMIARES_HB3_PLANO), ClasseH::Dentro);
        assert_eq!(classifica_f2b(HipB::Hb3, &m, &ClasseControle::ComRessalva, &LIMIARES_HB3_PLANO), ClasseH::Dentro);
        assert_eq!(classifica_f2b(HipB::Hb3, &m, &ClasseControle::SemProva, &LIMIARES_HB3_PLANO), ClasseH::SemProva);
    }

    fn proc(b: BracoB, pre: f64) -> ProcessoControleB {
        let (bl, f2, f2b) = match b {
            BracoB::D => (false, false, false),
            BracoB::A => (true, false, false),
            BracoB::B1 => (true, true, false),
            BracoB::B2 => (true, true, true),
        };
        ProcessoControleB { braco: b, prefill_ms_mediana: pre, passo_ms_mediana: 10.0, imprimiu_bloco_ligado: bl, imprimiu_f2_ligado: f2, imprimiu_f2b_ligado: f2b,
                            produziu_bloco: bl, produziu_f2: f2, produziu_f2b: f2b, valido: true }
    }

    fn serie(cfg: &ControleB, b2: f64) -> Vec<ProcessoControleB> {
        cfg.ordem.iter().map(|&b| proc(b, if b == BracoB::B2 { b2 } else { 100.0 })).collect()
    }

    #[test]
    fn controle_do_plano_d_a_b2_tem_9_processos_e_classes_por_desvio() {
        let cfg = &CONTROLE_B_PLANO;
        assert_eq!(cfg.ordem.len(), 9);
        for b in [BracoB::D, BracoB::A, BracoB::B2] {
            assert_eq!(cfg.ordem.iter().filter(|&&x| x == b).count(), 3);
        }
        let r = |b2: f64| classifica_controle_f2b(cfg, &serie(cfg, b2), true, 22);
        assert_eq!(r(102.0).classe, ClasseControle::Comprovada);
        assert_eq!(r(108.0).classe, ClasseControle::ComRessalva);
        assert_eq!(r(111.0).classe, ClasseControle::SemProva);
        assert_eq!(r(90.0).classe, ClasseControle::ComRessalva, "o sinal não conta, a magnitude sim");
        let x = r(108.0);
        assert!((x.principal.unwrap().desvio - 0.08).abs() < 1e-9);
        assert!(x.apoio.is_some() && x.classe_apoio.is_some());
        assert!(matches!(classifica_controle_f2b(cfg, &serie(cfg, 100.0), false, 22).classe, ClasseControle::Incompleto(_)));
        // Incompleto: processos a menos, ordem trocada, processo inválido, braço com arquivos do outro.
        let mut s = serie(cfg, 100.0);
        s.pop();
        assert!(matches!(classifica_controle_f2b(cfg, &s, true, 22).classe, ClasseControle::Incompleto(_)));
        let mut s = serie(cfg, 100.0);
        s.swap(0, 1);
        assert!(matches!(classifica_controle_f2b(cfg, &s, true, 22).classe, ClasseControle::Incompleto(_)));
        let mut s = serie(cfg, 100.0);
        s[3].valido = false;
        assert!(matches!(classifica_controle_f2b(cfg, &s, true, 22).classe, ClasseControle::Incompleto(_)));
        let mut s = serie(cfg, 100.0);
        s[1].produziu_f2b = true;
        assert!(matches!(classifica_controle_f2b(cfg, &s, true, 22).classe, ClasseControle::Incompleto(_)));
        let mut s = serie(cfg, 100.0);
        s[2].imprimiu_f2b_ligado = false;
        assert!(matches!(classifica_controle_f2b(cfg, &s, true, 22).classe, ClasseControle::Incompleto(_)));
    }

    #[test]
    fn controle_parametrizavel_so_a_b2_e_b1_no_lugar_do_d() {
        let a_b2 = &CONTROLE_B_A_B2;
        assert_eq!(a_b2.ordem.len(), 6);
        let r = classifica_controle_f2b(a_b2, &serie(a_b2, 105.0), true, 22);
        assert_eq!(r.classe, ClasseControle::ComRessalva);
        assert!(r.apoio.is_none() && r.classe_apoio.is_none());
        let b1 = &CONTROLE_B_B1_A_B2;
        assert_eq!(b1.ordem.len(), 9);
        let mut s = serie(b1, 101.0);
        for p in s.iter_mut().filter(|p| p.braco == BracoB::B1) {
            p.prefill_ms_mediana = 100.5;
        }
        let r = classifica_controle_f2b(b1, &s, true, 22);
        assert_eq!(r.classe, ClasseControle::Comprovada);
        assert!((r.apoio.unwrap().desvio - (101.0 - 100.5) / 100.5).abs() < 1e-9, "apoio é B2 contra B1");
        // D não vale na configuração com B1 (ordem diferente).
        assert!(matches!(classifica_controle_f2b(b1, &serie(&CONTROLE_B_PLANO, 100.0), true, 22).classe, ClasseControle::Incompleto(_)));
    }
}
