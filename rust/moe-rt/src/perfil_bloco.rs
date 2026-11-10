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
            modelo: String::new(),
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
            modelo: String::new(),
            indice: 0, aquecimento: false, n: 8, plano: vec![8], blocos: vec![b0.clone()], servico: vec![Some(serv(1, 10 * MS, 5 * MS, [0; 7]))], servico_extra: 0,
            t_armado: 0, t_logits_ini: b0.t[4], t_desarmado: b0.t[4], prefill_ms: ms(b0.t[4]), decode_passo_ms_mediana: 1.0,
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
            t_armado: 0, t_logits_ini: b.t[4], t_desarmado: b.t[4], prefill_ms: ms(b.t[4]), decode_passo_ms_mediana: 1.0,
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
            t_armado: 0, t_logits_ini: b.t[4], t_desarmado: b.t[4], prefill_ms: ms(b.t[4]), decode_passo_ms_mediana: 1.0,
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
            t_armado: 0, t_logits_ini: 0, t_desarmado: 0, prefill_ms: 0.0, decode_passo_ms_mediana: 1.0,
        };
        assert!((p.r6().unwrap() - 0.6).abs() < 1e-9);
    }
}
