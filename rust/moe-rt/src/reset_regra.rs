//! A2 (textos longos): PROVA 2 do reset de sequência no Engine (critério `qual-runs/A2-LOTE-20261103/criterio-reset-sequencia-20261108.md`).
//! Regras puras (sem GPU, testadas em CPU com o motor falso): para um modelo, a ordem A longo, C curto, A longo, C curto roda em dois
//! braços, A = motor NOVO por pedido e B = UM motor reutilizado com `nova_sequencia` entre os pedidos. Por pedido, ids (`ids_n`, `ids_fnv`,
//! ids), token escolhido e CADA logit por `f32::to_bits` têm de ser idênticos entre os braços. Qualquer diferença REPROVA (mesmo com o argmax
//! igual); erro, ausência de resultado ou integridade quebrada dá SEM PROVA; REPROVA vence SEM PROVA e integridade quebrada derruba tudo.
use crate::primeira_palavra_regra::{Resultado, RotuloCorrecao};
use crate::velocidade_regra::Cen;
use serde_json::{json, Value};

/// A ordem congelada dos pedidos de cada braço (expõe KV velho: o longo, depois o curto, de novo o longo).
pub const ORDEM: [Cen; 4] = [Cen::A, Cen::C, Cen::A, Cen::C];
/// sha256 do critério de reset congelado (08/11/2026).
pub const CRITERIO_RESET_SHA: &str = "955becbb243772087258bf2bde1c564c9cbc0630926806a0ed5a8258726662c9";

/// O que um braço guarda de um pedido.
#[derive(Clone, Debug, PartialEq)]
pub struct Medida {
    pub ids: Vec<usize>,
    pub ids_n: usize,
    pub ids_fnv: u64,
    pub token: usize,
    pub logits: Vec<f32>,
    pub sequencia_ms: f64,
}

impl Medida {
    pub fn de_resultado(r: &Resultado) -> Self {
        Self { ids: r.ids.clone(), ids_n: r.medida.ids_n, ids_fnv: r.medida.ids_fnv, token: r.medida.token, logits: r.logits.clone(), sequencia_ms: r.medida.crono.sequencia_ms }
    }
}

/// A comparação bit a bit entre os dois braços de um pedido.
#[derive(Clone, Debug, PartialEq)]
pub struct Diferenca {
    pub ids_iguais: bool,
    pub primeiro_id_divergente: Option<usize>,
    pub ids_n_igual: bool,
    pub fnv_igual: bool,
    pub token_igual: bool,
    pub logits_len: (usize, usize),
    pub primeiro_logit_divergente: Option<usize>,
    /// Posições com `to_bits` diferente; as que só existem de um lado contam como divergentes.
    pub logits_divergentes: usize,
}

impl Diferenca {
    pub fn identica(&self) -> bool {
        self.ids_iguais && self.ids_n_igual && self.fnv_igual && self.token_igual && self.logits_len.0 == self.logits_len.1 && self.logits_divergentes == 0
    }
}

/// Primeiro índice divergente e quantos divergem entre duas listas comparadas pelo critério `eq` (o excedente de uma delas diverge).
fn divergencias<T>(a: &[T], b: &[T], eq: impl Fn(&T, &T) -> bool) -> (Option<usize>, usize) {
    let n = a.len().max(b.len());
    let mut primeiro = None;
    let mut cont = 0;
    for i in 0..n {
        let igual = match (a.get(i), b.get(i)) {
            (Some(x), Some(y)) => eq(x, y),
            _ => false,
        };
        if !igual {
            cont += 1;
            primeiro.get_or_insert(i);
        }
    }
    (primeiro, cont)
}

pub fn compara(a: &Medida, b: &Medida) -> Diferenca {
    let (pid, nid) = divergencias(&a.ids, &b.ids, |x, y| x == y);
    let (plg, nlg) = divergencias(&a.logits, &b.logits, |x, y| x.to_bits() == y.to_bits());
    Diferenca {
        ids_iguais: nid == 0,
        primeiro_id_divergente: pid,
        ids_n_igual: a.ids_n == b.ids_n,
        fnv_igual: a.ids_fnv == b.ids_fnv,
        token_igual: a.token == b.token,
        logits_len: (a.logits.len(), b.logits.len()),
        primeiro_logit_divergente: plg,
        logits_divergentes: nlg,
    }
}

/// Um pedido da prova: o resultado de cada braço (ou o erro dele).
#[derive(Clone, Debug, PartialEq)]
pub struct PedidoReset {
    pub indice: usize,
    pub cen: Cen,
    pub a: Result<Medida, String>,
    pub b: Result<Medida, String>,
}

/// O registro de um modelo.
#[derive(Clone, Debug, PartialEq)]
pub struct RegistroReset {
    pub modelo: String,
    pub integro: bool,
    /// Falha fora dos pedidos (carga do modelo, encerramento do motor reutilizado).
    pub falha: Option<String>,
    pub pedidos: Vec<PedidoReset>,
    pub extras: Vec<(String, String)>,
}

/// Roda um braço: os pedidos em ordem; o primeiro erro para o laço e os restantes ficam como erro ("não executado").
fn roda_braco(mut pedido: impl FnMut(usize, Cen) -> Result<Resultado, String>) -> Vec<Result<Medida, String>> {
    let mut v: Vec<Result<Medida, String>> = Vec::with_capacity(ORDEM.len());
    let mut parou: Option<String> = None;
    for (i, cen) in ORDEM.iter().enumerate() {
        if let Some(e) = &parou {
            v.push(Err(format!("não executado: {e}")));
            continue;
        }
        match pedido(i, *cen) {
            Ok(r) => v.push(Ok(Medida::de_resultado(&r))),
            Err(e) => {
                parou = Some(format!("erro no pedido {i}: {e}"));
                v.push(Err(e));
            }
        }
    }
    v
}

/// Os dois braços, um depois do outro (A inteiro, depois B inteiro: nunca dois motores vivos ao mesmo tempo). Um erro de execução não apaga os
/// pedidos já concluídos do mesmo braço e nada repete.
pub fn executa_prova(braco_a: impl FnMut(usize, Cen) -> Result<Resultado, String>, braco_b: impl FnMut(usize, Cen) -> Result<Resultado, String>) -> Vec<PedidoReset> {
    let a = roda_braco(braco_a);
    let b = roda_braco(braco_b);
    a.into_iter().zip(b).enumerate().map(|(i, (a, b))| PedidoReset { indice: i, cen: ORDEM[i], a, b }).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResPedido {
    Passa,
    Reprova,
    SemProva,
}

/// Rótulo de um pedido: PASSA só se os dois braços existem, têm logits não vazios e são idênticos em tudo; diferença REPROVA; o resto é SEM PROVA.
pub fn avalia_pedido(p: &PedidoReset) -> (ResPedido, Option<Diferenca>, String) {
    match (&p.a, &p.b) {
        (Ok(a), Ok(b)) => {
            let d = compara(a, b);
            if !d.identica() {
                (ResPedido::Reprova, Some(d.clone()), format!("pedido {} ({}): braços DIFERENTES (ids {}, ids_n {}, fnv {}, token {}, logits {} divergentes de {}/{}, primeiro {:?})", p.indice, p.cen.nome(),
                    if d.ids_iguais { "iguais" } else { "diferem" }, if d.ids_n_igual { "iguais" } else { "diferem" }, if d.fnv_igual { "igual" } else { "diferente" }, if d.token_igual { "igual" } else { "diferente" },
                    d.logits_divergentes, d.logits_len.0, d.logits_len.1, d.primeiro_logit_divergente))
            } else if a.logits.is_empty() {
                (ResPedido::SemProva, Some(d), format!("pedido {} ({}): logits vazios nos dois braços", p.indice, p.cen.nome()))
            } else {
                (ResPedido::Passa, Some(d), format!("pedido {} ({}): idêntico bit a bit ({} ids, token {}, {} logits)", p.indice, p.cen.nome(), a.ids_n, a.token, a.logits.len()))
            }
        }
        (a, b) => {
            let mut m = vec![];
            if let Err(e) = a {
                m.push(format!("braço A: {e}"));
            }
            if let Err(e) = b {
                m.push(format!("braço B: {e}"));
            }
            (ResPedido::SemProva, None, format!("pedido {} ({}): sem resultado ({})", p.indice, p.cen.nome(), m.join("; ")))
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AvalReset {
    pub rotulo: RotuloCorrecao,
    pub linhas: Vec<String>,
    pub motivos: Vec<String>,
    pub por_pedido: Vec<ResPedido>,
}

/// O rótulo do modelo (sem a integridade): qualquer REPROVA vence; sem REPROVA, falta de pedido, erro ou falha fora dos pedidos é SEM PROVA.
pub fn avalia(r: &RegistroReset) -> AvalReset {
    let mut linhas = vec![];
    let mut motivos = vec![];
    let mut por = vec![];
    let ordem_ok = r.pedidos.len() == ORDEM.len() && r.pedidos.iter().enumerate().all(|(i, p)| p.indice == i && p.cen == ORDEM[i]);
    for p in &r.pedidos {
        let (res, _, l) = avalia_pedido(p);
        if res != ResPedido::Passa {
            motivos.push(l.clone());
        }
        linhas.push(l);
        por.push(res);
    }
    if !ordem_ok {
        motivos.push(format!("{} pedidos (ou fora da ordem A, C, A, C); esperados {}", r.pedidos.len(), ORDEM.len()));
    }
    if let Some(f) = &r.falha {
        motivos.push(format!("falha: {f}"));
    }
    let rotulo = if por.contains(&ResPedido::Reprova) {
        RotuloCorrecao::Reprova
    } else if !ordem_ok || r.falha.is_some() || por.contains(&ResPedido::SemProva) {
        RotuloCorrecao::SemProva
    } else {
        RotuloCorrecao::Passa
    };
    AvalReset { rotulo, linhas, motivos, por_pedido: por }
}

/// Integridade quebrada derruba tudo para SEM PROVA.
pub fn rotulo_final(r: RotuloCorrecao, integro: bool) -> RotuloCorrecao {
    if integro {
        r
    } else {
        RotuloCorrecao::SemProva
    }
}

/// Conjunto dos modelos: as mesmas regras do 5b (REPROVA vence SEM PROVA, SEM PROVA vence PASSA e a falta de modelo, integridade quebrada derruba tudo).
pub fn rotulo_conjunto(v: &[RotuloCorrecao], integro: bool) -> RotuloCorrecao {
    crate::primeira_palavra_regra::rotulo_conjunto_correcao(v, integro)
}

fn medida_json(m: &Result<Medida, String>) -> Value {
    match m {
        Ok(m) => json!({"ids": m.ids, "ids_n": m.ids_n, "ids_fnv": format!("{:016x}", m.ids_fnv), "token": m.token, "logits_n": m.logits.len(), "T_sequencia_ms": m.sequencia_ms, "falha": null}),
        Err(e) => json!({"falha": e}),
    }
}

impl RegistroReset {
    /// O resultado gravado, vinculado ao critério de reset (sha256 dentro), com o rótulo e, por pedido, a comparação dos braços.
    pub fn para_json(&self) -> Value {
        let a = avalia(self);
        let pedidos: Vec<Value> = self.pedidos.iter().zip(&a.por_pedido).map(|(p, res)| {
            let d = avalia_pedido(p).1;
            json!({
                "indice": p.indice,
                "cenario": p.cen.nome(),
                "rotulo": match res { ResPedido::Passa => "PASSA", ResPedido::Reprova => "REPROVA", ResPedido::SemProva => "SEM PROVA" },
                "braco_A_motor_novo": medida_json(&p.a),
                "braco_B_motor_reutilizado": medida_json(&p.b),
                "ids_iguais": d.as_ref().map(|d| d.ids_iguais),
                "primeiro_id_divergente": d.as_ref().and_then(|d| d.primeiro_id_divergente),
                "ids_n_igual": d.as_ref().map(|d| d.ids_n_igual),
                "ids_fnv_igual": d.as_ref().map(|d| d.fnv_igual),
                "token_igual": d.as_ref().map(|d| d.token_igual),
                "logits_divergentes": d.as_ref().map(|d| d.logits_divergentes),
                "primeiro_logit_divergente": d.as_ref().and_then(|d| d.primeiro_logit_divergente),
            })
        }).collect();
        json!({
            "tipo": "reset-sequencia-prova2",
            "criterio": "criterio-reset-sequencia-20261108.md",
            "criterio_sha256": CRITERIO_RESET_SHA,
            "modelo": self.modelo,
            "ordem": ORDEM.iter().map(|c| c.nome()).collect::<Vec<_>>(),
            "integro": self.integro,
            "falha": self.falha,
            "rotulo_sem_integridade": a.rotulo.nome(),
            "rotulo": rotulo_final(a.rotulo, self.integro).nome(),
            "motivos": a.motivos,
            "pedidos": pedidos,
            "extras": self.extras.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>(),
        })
    }
}

/// Texto do relatório de um modelo.
pub fn texto(r: &RegistroReset, final_: RotuloCorrecao) -> String {
    let a = avalia(r);
    let mut t = format!("== reset de sequência {}\n", r.modelo);
    for l in &a.linhas {
        t += &format!("{l}\n");
    }
    if a.rotulo != RotuloCorrecao::Passa || !r.integro {
        t += &format!("motivo(s): {}\n", a.motivos.join("; "));
    }
    for p in &r.pedidos {
        if let (Ok(x), Ok(y)) = (&p.a, &p.b) {
            t += &format!("T_sequencia pedido {} ({}): A {:.3} ms, B {:.3} ms\n", p.indice, p.cen.nome(), x.sequencia_ms, y.sequencia_ms);
        }
    }
    t += &format!("rótulo: {}\n", final_.nome());
    t
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::motor::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg};
    use crate::motor::{AdmitMode, Config};
    use crate::motor::leitura::PreparaPasso;
    use crate::primeira_palavra_regra::{data_fixa_texto, primeira_palavra, primeira_palavra_reutilizando, Sequencia, SequenciaReutilizavel};
    use crate::tokenizador::Tokenizador;
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::time::Instant;

    const TOKENIZER_1B: &str = "/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct/tokenizer.json";
    const VOCAB: usize = 49155;
    const LONGO: &str = "Explain in detail how TCP congestion control works, including slow start, congestion avoidance, fast retransmit and fast recovery.";
    const CURTO: &str = "Hi";

    fn texto_de(c: Cen) -> &'static str {
        match c {
            Cen::A => LONGO,
            Cen::C => CURTO,
        }
    }

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

    /// Preparo que zera a posição quando o backend falso recebe um `reset_sequence` novo (como o modelo real).
    struct Prep {
        gpu: Arc<FakeGpu>,
        visto: Mutex<usize>,
        pos: Mutex<u64>,
    }
    impl Prep {
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
            Ok(())
        }
    }

    /// Logits determinísticos por pedido (função do texto): pico no meio, valores distintos e um segundo colocado.
    fn logits_de(texto: &str) -> Vec<f32> {
        let mut v: Vec<f32> = (0..VOCAB).map(|i| ((i * 31 + texto.len() * 7) % 1000) as f32 * 1e-3).collect();
        v[10921] = 5.0;
        v[10922] = 2.0;
        v
    }

    /// `atual` guarda o texto do pedido em curso (o motor reutilizado leva o mesmo leitor de logits por todos os pedidos).
    fn sequencia(gpus: &Mutex<Vec<Arc<FakeGpu>>>, atual: &Arc<Mutex<&'static str>>, falha_nos_logits: bool) -> Result<Sequencia<FakeGpu, Prep>, String> {
        let (gpu, eng) = motor_falso();
        gpus.lock().unwrap().push(gpu.clone());
        let le: Box<dyn FnMut() -> Result<Vec<f32>, String>> = if falha_nos_logits { Box::new(|| Err("logits falharam".to_string())) } else { { let atual = atual.clone(); Box::new(move || Ok(logits_de(*atual.lock().unwrap()))) } };
        Ok(Sequencia { eng, prep: Prep::novo(&gpu), le_logits: le })
    }

    /// Os dois braços sobre o motor falso. `ordem` registra (braço, índice, cenário) na ordem de chamada. `mexe` altera o resultado do braço B.
    fn prova(
        ordem: &Mutex<Vec<(char, usize, Cen)>>,
        erro_a: Option<usize>,
        erro_b: Option<usize>,
        mexe: impl Fn(usize, &mut Resultado),
    ) -> (Vec<PedidoReset>, usize, usize) {
        let t = tok();
        let atual = Arc::new(Mutex::new(CURTO));
        let gpus_a = Mutex::new(vec![]);
        let gpus_b = Mutex::new(vec![]);
        let mut guardada: SequenciaReutilizavel<FakeGpu, Prep> = SequenciaReutilizavel::nova();
        let pedidos = executa_prova(
            |i, cen| {
                ordem.lock().unwrap().push(('A', i, cen));
                let texto = texto_de(cen);
                *atual.lock().unwrap() = texto;
                let mut s: Vec<u8> = vec![];
                primeira_palavra(Instant::now(), 0.0, texto, &data_fixa_texto(), &t, || sequencia(&gpus_a, &atual, erro_a == Some(i)), &mut s)
            },
            |i, cen| {
                ordem.lock().unwrap().push(('B', i, cen));
                let texto = texto_de(cen);
                *atual.lock().unwrap() = texto;
                let mut s: Vec<u8> = vec![];
                // O pedido que falha no braço B faz falhar os logits do motor que o cria (só vale para motor recém-criado) ou o leitor.
                let erro = erro_b == Some(i);
                let mut r = primeira_palavra_reutilizando(Instant::now(), 0.0, texto, &data_fixa_texto(), &t, &mut guardada, || sequencia(&gpus_b, &atual, false), &mut s)?;
                if erro {
                    return Err("falha injetada no braço B".into());
                }
                mexe(i, &mut r);
                Ok(r)
            },
        );
        guardada.encerra().unwrap();
        let (na, nb) = (gpus_a.lock().unwrap().len(), gpus_b.lock().unwrap().len());
        (pedidos, na, nb)
    }

    fn rec(p: Vec<PedidoReset>) -> RegistroReset {
        RegistroReset { modelo: "1b".into(), integro: true, falha: None, pedidos: p, extras: vec![] }
    }

    #[test]
    fn ordem_dos_quatro_pedidos_e_dos_dois_bracos() {
        let ordem = Mutex::new(vec![]);
        let (p, na, nb) = prova(&ordem, None, None, |_, _| {});
        let o = ordem.lock().unwrap().clone();
        let esperado: Vec<(char, usize, Cen)> = ['A', 'B'].iter().flat_map(|b| ORDEM.iter().enumerate().map(move |(i, c)| (*b, i, *c))).collect();
        assert_eq!(o, esperado, "braço A inteiro, depois o B, cada um em A, C, A, C");
        assert_eq!(ORDEM, [Cen::A, Cen::C, Cen::A, Cen::C]);
        assert_eq!(p.len(), 4);
        assert_eq!(na, 4, "braço A: motor novo por pedido");
        assert_eq!(nb, 1, "braço B: um motor só");
        assert!(p.iter().enumerate().all(|(i, x)| x.indice == i && x.cen == ORDEM[i]));
    }

    #[test]
    fn bracos_identicos_passam() {
        let (p, _, _) = prova(&Mutex::new(vec![]), None, None, |_, _| {});
        let r = rec(p);
        let a = avalia(&r);
        assert_eq!(a.rotulo, RotuloCorrecao::Passa, "{:?}", a.motivos);
        assert_eq!(a.por_pedido, vec![ResPedido::Passa; 4]);
        // Os longos têm mais ids que os curtos e o mesmo texto dá os mesmos ids nos dois pedidos longos.
        let ids = |i: usize| r.pedidos[i].a.as_ref().unwrap().ids_n;
        assert!(ids(0) > ids(1) && ids(0) == ids(2) && ids(1) == ids(3));
        let j = r.para_json();
        assert_eq!(j["criterio_sha256"], CRITERIO_RESET_SHA);
        assert_eq!(j["rotulo"], "PASSA");
        assert_eq!(j["pedidos"][2]["logits_divergentes"], 0);
        assert!(j["pedidos"][0]["braco_A_motor_novo"]["T_sequencia_ms"].is_number() && j["pedidos"][0]["braco_B_motor_reutilizado"]["ids"].is_array());
    }

    #[test]
    fn um_bit_de_logit_nao_maximo_reprova_com_argmax_igual() {
        let (p, _, _) = prova(&Mutex::new(vec![]), None, None, |i, r| {
            if i == 2 {
                // Um bit do último ULP de um logit que NÃO é o máximo.
                let k = 12345;
                r.logits[k] = f32::from_bits(r.logits[k].to_bits() ^ 1);
            }
        });
        let r = rec(p);
        let (res, d, _) = avalia_pedido(&r.pedidos[2]);
        let d = d.unwrap();
        assert!(d.token_igual && d.ids_iguais && d.fnv_igual, "argmax e ids iguais");
        assert_eq!((d.logits_divergentes, d.primeiro_logit_divergente), (1, Some(12345)));
        assert_eq!(res, ResPedido::Reprova);
        let a = avalia(&r);
        assert_eq!(a.por_pedido, vec![ResPedido::Passa, ResPedido::Passa, ResPedido::Reprova, ResPedido::Passa]);
        assert_eq!(a.rotulo, RotuloCorrecao::Reprova);
        assert_eq!(r.para_json()["pedidos"][2]["logits_divergentes"], 1);
        // Também reprova: ids diferentes ou token diferente.
        let (p, _, _) = prova(&Mutex::new(vec![]), None, None, |i, r| {
            if i == 1 {
                r.medida.token += 1;
            }
            if i == 3 {
                r.ids[0] += 1;
            }
        });
        let a = avalia(&rec(p));
        assert_eq!(a.por_pedido, vec![ResPedido::Passa, ResPedido::Reprova, ResPedido::Passa, ResPedido::Reprova]);
    }

    #[test]
    fn nan_e_zero_com_sinal_sao_comparados_por_bits() {
        let m = |l: Vec<f32>| Medida { ids: vec![1], ids_n: 1, ids_fnv: 1, token: 0, logits: l, sequencia_ms: 0.0 };
        assert!(compara(&m(vec![f32::NAN, 1.0]), &m(vec![f32::NAN, 1.0])).identica(), "mesmo NaN, mesmos bits");
        assert!(!compara(&m(vec![0.0]), &m(vec![-0.0])).identica(), "0.0 e -0.0 têm bits diferentes");
        let d = compara(&m(vec![1.0, 2.0]), &m(vec![1.0]));
        assert_eq!((d.logits_divergentes, d.primeiro_logit_divergente), (1, Some(1)), "tamanho diferente diverge");
    }

    #[test]
    fn erro_em_um_braco_e_sem_prova_e_o_restante_nao_executa() {
        let (p, _, _) = prova(&Mutex::new(vec![]), Some(1), None, |_, _| {});
        let r = rec(p);
        assert!(r.pedidos[0].a.is_ok() && r.pedidos[1].a.is_err() && r.pedidos[2].a.as_ref().err().unwrap().starts_with("não executado"));
        let a = avalia(&r);
        assert_eq!(a.por_pedido, vec![ResPedido::Passa, ResPedido::SemProva, ResPedido::SemProva, ResPedido::SemProva]);
        assert_eq!(a.rotulo, RotuloCorrecao::SemProva);
        // Erro no braço B.
        let (p, _, _) = prova(&Mutex::new(vec![]), None, Some(2), |_, _| {});
        assert_eq!(avalia(&rec(p)).rotulo, RotuloCorrecao::SemProva);
        // Ausência de resultado: lista vazia, ou pedidos a menos, ou falha fora dos pedidos.
        assert_eq!(avalia(&rec(vec![])).rotulo, RotuloCorrecao::SemProva);
        let (mut p, _, _) = prova(&Mutex::new(vec![]), None, None, |_, _| {});
        p.pop();
        assert_eq!(avalia(&rec(p)).rotulo, RotuloCorrecao::SemProva);
        let (p, _, _) = prova(&Mutex::new(vec![]), None, None, |_, _| {});
        let mut r = rec(p);
        r.falha = Some("encerra".into());
        assert_eq!(avalia(&r).rotulo, RotuloCorrecao::SemProva);
        assert_eq!(r.para_json()["rotulo"], "SEM PROVA");
    }

    #[test]
    fn logits_vazios_nos_dois_bracos_nao_passam() {
        let (p, _, _) = prova(&Mutex::new(vec![]), None, None, |_, _| {});
        let mut r = rec(p);
        for x in &mut r.pedidos {
            x.a.as_mut().unwrap().logits.clear();
            x.b.as_mut().unwrap().logits.clear();
        }
        assert_eq!(avalia(&r).rotulo, RotuloCorrecao::SemProva);
    }

    #[test]
    fn precedencia_reprova_sobre_sem_prova_e_integridade_derruba_tudo() {
        // Pedido 0 reprova (um bit), pedido 2 com erro no braço A: REPROVA vence.
        let (p, _, _) = prova(&Mutex::new(vec![]), Some(2), None, |i, r| {
            if i == 0 {
                r.logits[7] = f32::from_bits(r.logits[7].to_bits() ^ 1);
            }
        });
        let r = rec(p);
        let a = avalia(&r);
        assert_eq!(a.por_pedido, vec![ResPedido::Reprova, ResPedido::Passa, ResPedido::SemProva, ResPedido::SemProva]);
        assert_eq!(a.rotulo, RotuloCorrecao::Reprova);
        assert_eq!(rotulo_final(a.rotulo, true), RotuloCorrecao::Reprova);
        assert_eq!(rotulo_final(a.rotulo, false), RotuloCorrecao::SemProva, "integridade quebrada derruba tudo, até a REPROVA");
        let mut q = r.clone();
        q.integro = false;
        assert_eq!(q.para_json()["rotulo"], "SEM PROVA");
        assert_eq!(rotulo_final(RotuloCorrecao::Passa, false), RotuloCorrecao::SemProva);
        // Conjunto de modelos.
        use RotuloCorrecao::*;
        assert_eq!(rotulo_conjunto(&[Passa, Passa], true), Passa);
        assert_eq!(rotulo_conjunto(&[Passa, SemProva], true), SemProva);
        assert_eq!(rotulo_conjunto(&[SemProva, Reprova], true), Reprova);
        assert_eq!(rotulo_conjunto(&[Reprova], false), SemProva);
        assert_eq!(rotulo_conjunto(&[], true), SemProva);
    }

    #[test]
    fn motor_reutilizado_de_verdade_zera_a_posicao_entre_pedidos() {
        // O braço B do motor falso faz um reset por pedido depois do primeiro (3), e o resultado bate com o do motor novo.
        let t = tok();
        let gpus = Mutex::new(vec![]);
        let atual = Arc::new(Mutex::new(CURTO));
        let mut g: SequenciaReutilizavel<FakeGpu, Prep> = SequenciaReutilizavel::nova();
        for c in ORDEM {
            let texto = texto_de(c);
            *atual.lock().unwrap() = texto;
            primeira_palavra_reutilizando(Instant::now(), 0.0, texto, &data_fixa_texto(), &t, &mut g, || sequencia(&gpus, &atual, false), &mut Vec::new()).unwrap();
        }
        assert_eq!(gpus.lock().unwrap().len(), 1);
        assert_eq!(gpus.lock().unwrap()[0].resets().len(), 3);
        g.encerra().unwrap();
    }
}
