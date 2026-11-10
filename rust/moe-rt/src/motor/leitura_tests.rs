//! Testes simulados (backend falso, sem GPU) da leitura de N ids em blocos (`leitura.rs`). Validam a divisão em blocos, a validação
//! antes de qualquer efeito, a ordem, a posição, a continuação, o erro no meio e a preservação de M = 1. NÃO validam correção numérica
//! nem velocidade de GPU: o backend falso não calcula atenção, KV nem logits de modelo.
use super::engine::Engine;
use super::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg};
use super::leitura::{le_ids, plano_blocos, valida, ErroLeitura, PreparaPasso};
use super::{AdmitMode, Config, Error};
use std::sync::{Arc, Mutex};

const VOCAB: usize = 50;

/// O preparo de teste: guarda cada bloco com a posição em que começa, avança a própria posição como a parte densa e pode falhar no
/// n-ésimo bloco.
struct Prep {
    max_len: usize,
    pos: Mutex<u64>,
    blocos: Mutex<Vec<(u64, Vec<usize>)>>,
    falha_no_bloco: Option<usize>,
    /// Arma a falha da espera de publicação na GPU falsa quando este bloco (índice) é preparado.
    arma_publicacao_no_bloco: Option<usize>,
    max_pos: usize,
    /// A GPU falsa do motor, para registrar quantos `begin_step` ela já tinha quando cada bloco foi preparado.
    gpu: Option<Arc<FakeGpu>>,
    begins_vistos: Mutex<Vec<usize>>,
}

impl Prep {
    fn novo(max_len: usize) -> Self {
        Self::na_posicao(max_len, 0)
    }
    fn com_gpu(max_len: usize, gpu: &Arc<FakeGpu>) -> Self {
        Self { gpu: Some(gpu.clone()), ..Self::novo(max_len) }
    }
    fn na_posicao(max_len: usize, pos: u64) -> Self {
        Self { max_len, pos: Mutex::new(pos), blocos: Mutex::new(vec![]), falha_no_bloco: None, arma_publicacao_no_bloco: None, max_pos: 8, gpu: None,
               begins_vistos: Mutex::new(vec![]) }
    }
    fn blocos(&self) -> Vec<(u64, Vec<usize>)> {
        self.blocos.lock().unwrap().clone()
    }
    fn tamanhos(&self) -> Vec<usize> {
        self.blocos().iter().map(|(_, b)| b.len()).collect()
    }
}

impl PreparaPasso for Prep {
    type Error = &'static str;
    fn vocab(&self) -> usize {
        VOCAB
    }
    fn max_positions(&self) -> usize {
        self.max_pos
    }
    fn max_len(&self) -> usize {
        self.max_len
    }
    fn posicao(&self) -> u64 {
        *self.pos.lock().unwrap()
    }
    fn prepara(&self, ids: &[usize]) -> Result<(), &'static str> {
        let mut b = self.blocos.lock().unwrap();
        if Some(b.len()) == self.falha_no_bloco {
            return Err("preparo falhou");
        }
        if let (Some(g), Some(k)) = (&self.gpu, self.arma_publicacao_no_bloco) {
            if b.len() == k {
                g.arma_falha_na_publicacao();
            }
        }
        if let Some(g) = &self.gpu {
            self.begins_vistos.lock().unwrap().push(g.begin_steps().len());
        }
        let mut pos = self.pos.lock().unwrap();
        b.push((*pos, ids.to_vec()));
        *pos += ids.len() as u64;
        Ok(())
    }
}

fn rota(p: usize, l: usize) -> Vec<u32> {
    vec![((p + l) % 4) as u32, ((p + l + 1 + (p / 4) % 3) % 4) as u32]
}

/// Um motor falso para até `total` posições, com `mp` posições por passo; `fail_main_at` injeta a falha da principal.
fn motor(mp: usize, total: usize, fail_main_at: Option<u64>) -> (Arc<FakeGpu>, Engine<FakeGpu>) {
    motor_pf(mp, total, fail_main_at, false)
}

/// Como `motor`, com o instrumento de prefill escolhido (sem depender da variável de ambiente).
fn motor_pf(mp: usize, total: usize, fail_main_at: Option<u64>, perfil: bool) -> (Arc<FakeGpu>, Engine<FakeGpu>) {
    let routing: Vec<Vec<Vec<u32>>> = (0..total.max(1)).map(|p| (0..2).map(|l| rota(p, l)).collect()).collect();
    let gpu = FakeGpu::new_sched(
        FakeCfg { layers: 2, experts: 4, topk: 2, width: 3, routing, fail_main_at, copy_immediate: false, max_positions: mp },
        SchedCfg { seed: 0xA0A, weights: [1, 1, 1, 1], budget: 300_000 },
    );
    let cfg = Config { layers: 2, experts: 4, topk: 2, width: 3, slots: 0, backlog: 0, admit: AdmitMode::TokenBarrier, first_token: 1,
                       prefill: vec![], max_positions: mp, mutant: None };
    let parts = gpu.parts(&cfg);
    let eng = Engine::new_pf(gpu.clone(), parts, cfg, FakeCpu { weighted: mp > 1, ..FakeCpu::default() }, perfil).expect("motor");
    (gpu, eng)
}

fn ids(n: usize) -> Vec<usize> {
    (0..n).map(|i| (i * 7 + 3) % VOCAB).collect()
}

/// O oráculo do plano: soma N, nenhum bloco vazio, só o último pode ser menor que `mp`, nenhum passa de `mp`.
fn plano_confere(plano: &[usize], n: usize, mp: usize) -> bool {
    plano.iter().sum::<usize>() == n
        && plano.iter().all(|&b| b >= 1 && b <= mp)
        && plano.iter().rev().skip(1).all(|&b| b == mp)
}

fn fecha(gpu: &Arc<FakeGpu>, eng: Engine<FakeGpu>) {
    let _ = eng.close();
    assert!(!gpu.stalled());
    assert_eq!(gpu.violations(), vec![]);
}

#[test]
fn plano_de_blocos_tem_a_lista_exata() {
    assert_eq!(plano_blocos(1, 8), vec![1]);
    assert_eq!(plano_blocos(8, 8), vec![8]);
    assert_eq!(plano_blocos(9, 8), vec![8, 1]);
    assert_eq!(plano_blocos(23, 8), vec![8, 8, 7]);
    assert_eq!(plano_blocos(24, 8), vec![8, 8, 8]);
    assert_eq!(plano_blocos(5, 1), vec![1; 5]);
    for n in 1..=24usize {
        let p = plano_blocos(n, 8);
        assert!(plano_confere(&p, n, 8), "n = {n}: {p:?}");
    }
}

#[test]
fn restos_de_1_a_7_e_multiplos_de_8() {
    for r in 1..=7usize {
        for n in [r, 16 + r] {
            let p = plano_blocos(n, 8);
            assert!(plano_confere(&p, n, 8), "n = {n}: {p:?}");
            assert_eq!(*p.last().unwrap(), r, "o resto de n = {n}");
        }
    }
    for n in [8usize, 16, 24] {
        let p = plano_blocos(n, 8);
        assert!(plano_confere(&p, n, 8) && p.iter().all(|&b| b == 8), "n = {n}: {p:?}");
    }
    // N = 0 não é um caso de resto: é lista vazia aqui e entrada inválida em `le_ids` (teste das entradas inválidas).
    assert!(plano_blocos(0, 8).is_empty());
}

#[test]
fn leitura_de_cada_tamanho_roda_os_blocos_do_plano_e_avanca_a_posicao() {
    for n in [1usize, 5, 7, 8, 9, 15, 16, 17, 23, 24] {
        let (gpu, mut eng) = motor(8, n, None);
        let prep = Prep::novo(64);
        let entrada = ids(n);
        assert_eq!(le_ids(&mut eng, &prep, &entrada).unwrap(), n);
        assert_eq!(prep.tamanhos(), plano_blocos(n, 8), "n = {n}");
        assert_eq!(eng.position(), n as u64);
        assert_eq!(prep.posicao(), n as u64);
        // Os ids chegam inteiros e em ordem, cada bloco na posição em que começa.
        let juntos: Vec<usize> = prep.blocos().iter().flat_map(|(_, b)| b.clone()).collect();
        assert_eq!(juntos, entrada);
        let mut esperado = 0u64;
        for (pos, b) in prep.blocos() {
            assert_eq!(pos, esperado);
            esperado += b.len() as u64;
        }
        fecha(&gpu, eng);
    }
}

#[test]
fn limites_do_contexto_e_da_soma() {
    // Exatamente em `max_len`: aceito.
    let (gpu, mut eng) = motor(8, 32, None);
    let prep = Prep::novo(32);
    assert_eq!(le_ids(&mut eng, &prep, &ids(32)).unwrap(), 32);
    assert_eq!(eng.position(), 32);
    fecha(&gpu, eng);

    // Um além: recusado sem nenhum passo.
    let (gpu, mut eng) = motor(8, 40, None);
    let prep = Prep::novo(32);
    assert!(matches!(le_ids(&mut eng, &prep, &ids(33)), Err(ErroLeitura::CacheCheio { posicao: 0, n: 33, max_len: 32 })));
    assert!(prep.blocos().is_empty());
    assert_eq!((eng.position(), prep.posicao()), (0, 0));
    fecha(&gpu, eng);

    // Com posição inicial > 0: 20 e depois 12 fecham em 32; 13 passa de 32.
    let (gpu, mut eng) = motor(8, 40, None);
    let prep = Prep::novo(32);
    le_ids(&mut eng, &prep, &ids(20)).unwrap();
    let antes = prep.blocos().len();
    assert!(matches!(le_ids(&mut eng, &prep, &ids(13)), Err(ErroLeitura::CacheCheio { posicao: 20, n: 13, max_len: 32 })));
    assert_eq!(prep.blocos().len(), antes, "a recusa não preparou bloco nenhum");
    assert_eq!(eng.position(), 20);
    le_ids(&mut eng, &prep, &ids(12)).unwrap();
    assert_eq!(eng.position(), 32);
    fecha(&gpu, eng);

    // A soma que estoura é recusa (checked_add), antes de qualquer efeito, sem entrar em pânico.
    assert!(matches!(valida::<(), ()>(&[1], VOCAB, u64::MAX, 32), Err(ErroLeitura::Estouro)));
    assert!(matches!(valida::<(), ()>(&[1, 2], VOCAB, u64::MAX - 1, usize::MAX), Err(ErroLeitura::Estouro)));
    let (gpu, mut eng) = motor(8, 4, None);
    let prep = Prep::na_posicao(32, u64::MAX);
    assert!(matches!(le_ids(&mut eng, &prep, &ids(2)), Err(ErroLeitura::Estouro)));
    assert!(prep.blocos().is_empty());
    fecha(&gpu, eng);
}

#[test]
fn continuacao_comeca_na_posicao_em_que_a_primeira_parou() {
    let (gpu1, mut e1) = motor(8, 24, None);
    let p1 = Prep::novo(64);
    let todos = ids(24);
    le_ids(&mut e1, &p1, &todos[..13]).unwrap();
    assert_eq!(e1.position(), 13);
    le_ids(&mut e1, &p1, &todos[13..]).unwrap();
    assert_eq!(e1.position(), 24);
    let b = p1.blocos();
    assert_eq!(b.iter().map(|(p, x)| (*p, x.len())).collect::<Vec<_>>(), vec![(0, 8), (8, 5), (13, 8), (21, 3)]);
    let s1 = e1.next_published().unwrap().stats().clone();
    fecha(&gpu1, e1);

    // O conjunto é o de uma chamada de 24: mesmos ids nas mesmas posições e os mesmos contadores de experts pedidos.
    let (gpu2, mut e2) = motor(8, 24, None);
    let p2 = Prep::novo(64);
    le_ids(&mut e2, &p2, &todos).unwrap();
    let achatados = |p: &Prep| {
        let mut v = vec![0usize; 24];
        for (pos, b) in p.blocos() {
            for (i, &id) in b.iter().enumerate() {
                v[pos as usize + i] = id;
            }
        }
        v
    };
    assert_eq!(achatados(&p1), achatados(&p2));
    let s2 = e2.next_published().unwrap().stats().clone();
    assert_eq!((s1.needed, s1.hits, s1.cpu), (s2.needed, s2.hits, s2.cpu));
    fecha(&gpu2, e2);
}

#[test]
fn entradas_invalidas_sao_recusadas_antes_de_qualquer_efeito() {
    let (gpu, mut eng) = motor(8, 16, None);
    let prep = Prep::novo(64);
    assert!(matches!(le_ids(&mut eng, &prep, &[]), Err(ErroLeitura::Vazia)));
    let mut ruim = ids(10);
    ruim[6] = VOCAB;
    assert!(matches!(le_ids(&mut eng, &prep, &ruim), Err(ErroLeitura::IdForaDoVocabulario { indice: 6, id: VOCAB, vocab: VOCAB })));
    assert!(prep.blocos().is_empty(), "nenhum bloco preparado");
    assert_eq!((eng.position(), prep.posicao()), (0, 0));
    // O motor segue utilizável depois das recusas.
    assert_eq!(le_ids(&mut eng, &prep, &ids(9)).unwrap(), 9);
    assert_eq!(eng.position(), 9);
    fecha(&gpu, eng);
}

#[test]
fn falha_do_preparo_no_meio_encerra_o_motor() {
    let (gpu, mut eng) = motor(8, 24, None);
    let mut prep = Prep::novo(64);
    prep.falha_no_bloco = Some(1);
    let r = le_ids(&mut eng, &prep, &ids(20));
    assert!(matches!(r, Err(ErroLeitura::Preparo("preparo falhou"))), "{r:?}");
    assert_eq!(prep.tamanhos(), vec![8], "só o primeiro bloco foi preparado");
    assert!(eng.is_closed());
    let prep2 = Prep::novo(64);
    assert!(matches!(le_ids(&mut eng, &prep2, &ids(3)), Err(ErroLeitura::MotorEncerrado)));
    assert!(prep2.blocos().is_empty(), "o motor encerrado não prepara nada");
    fecha(&gpu, eng);
}

#[test]
fn falha_do_backend_no_meio_encerra_o_motor_sem_travar() {
    // A falha da principal cai depois do primeiro passo (cada passo enfileira dezenas de operações na stream de cálculo).
    let (gpu, mut eng) = motor(8, 24, Some(20));
    let prep = Prep::novo(64);
    let r = le_ids(&mut eng, &prep, &ids(20));
    assert!(matches!(r, Err(ErroLeitura::Motor(Error::Backend(_)))), "{r:?}");
    assert!(eng.is_closed());
    assert!(eng.position() < 20, "a leitura não terminou");
    let prep2 = Prep::novo(64);
    assert!(matches!(le_ids(&mut eng, &prep2, &ids(3)), Err(ErroLeitura::MotorEncerrado)));
    assert!(prep2.blocos().is_empty());
    let _ = eng.close();
    assert!(!gpu.stalled());
}

/// A ordem por bloco k (0, 1, ...): o preparo acontece com exatamente k `begin_step` já enfileirados (antes do passo dele), e os
/// `begin_step(pos, m)` que o falso recebeu são os mesmos blocos que o preparo viu.
fn ordem_confere(prep: &Prep, gpu: &Arc<FakeGpu>) -> bool {
    let vistos = prep.begins_vistos.lock().unwrap().clone();
    let blocos: Vec<(u64, usize)> = prep.blocos().iter().map(|(p, b)| (*p, b.len())).collect();
    vistos.iter().enumerate().all(|(k, &v)| v == k) && gpu.begin_steps() == blocos
}

/// A rotina de verdade, com o mesmo laço, mas preparando o bloco DEPOIS do passo (o mutante da ordem).
fn le_ids_prepara_depois(eng: &mut Engine<FakeGpu>, prep: &Prep, ids: &[usize]) {
    let mut ini = 0;
    for m in plano_blocos(ids.len(), eng.max_positions()) {
        let p = eng.next_published().unwrap();
        eng.enqueue_step(p, m).unwrap();
        eng.finish_token().unwrap();
        prep.prepara(&ids[ini..ini + m]).unwrap();
        ini += m;
    }
}

#[test]
fn o_preparo_vem_antes_do_passo_do_mesmo_bloco_e_o_falso_ve_as_mesmas_posicoes() {
    for (n1, n2) in [(13usize, 11usize), (8, 8), (5, 3), (1, 9)] {
        let (gpu, mut eng) = motor(8, n1 + n2, None);
        let prep = Prep::com_gpu(64, &gpu);
        let todos = ids(n1 + n2);
        le_ids(&mut eng, &prep, &todos[..n1]).unwrap();
        le_ids(&mut eng, &prep, &todos[n1..]).unwrap();
        assert!(ordem_confere(&prep, &gpu), "n = {n1} + {n2}: preparos {:?}, begin_step {:?}", prep.begins_vistos.lock().unwrap(), gpu.begin_steps());
        let esperado: Vec<(u64, usize)> = plano_blocos(n1, 8).into_iter().scan(0u64, |p, m| { let r = (*p, m); *p += m as u64; Some(r) })
            .chain(plano_blocos(n2, 8).into_iter().scan(n1 as u64, |p, m| { let r = (*p, m); *p += m as u64; Some(r) })).collect();
        assert_eq!(gpu.begin_steps(), esperado, "o falso viu as posições globais certas");
        fecha(&gpu, eng);
    }
    // O mutante (prepara depois do passo) é acusado pelo mesmo oráculo.
    let (gpu, mut eng) = motor(8, 20, None);
    let prep = Prep::com_gpu(64, &gpu);
    le_ids_prepara_depois(&mut eng, &prep, &ids(20));
    assert!(!ordem_confere(&prep, &gpu), "o oráculo não acusou o preparo depois do passo");
    fecha(&gpu, eng);
}

#[test]
fn backend_com_menos_posicoes_que_o_motor_e_recusado_antes_de_qualquer_passo() {
    let (gpu, mut eng) = motor(8, 8, None);
    let mut prep = Prep::novo(64);
    prep.max_pos = 4;
    assert!(matches!(le_ids(&mut eng, &prep, &ids(5)), Err(ErroLeitura::BackendComMenosPosicoes { motor: 8, backend: 4 })));
    assert!(prep.blocos().is_empty());
    assert!(!eng.is_closed(), "recusa de configuração não encerra o motor");
    assert_eq!(eng.position(), 0);
    fecha(&gpu, eng);
}

/// A espera da publicação falha (`next_published` devolve erro): o motor tem de ficar encerrado, sem preparar o bloco que esperava.
/// A GPU falsa só chega a esperar (`relax`) quando a publicação ainda não está pronta; por isso a falha é armada em três pontos (antes
/// da leitura e ao preparar o 1º e o 2º bloco) e ao menos um deles tem de derrubar a leitura pelo ramo da publicação.
#[test]
fn falha_na_espera_da_publicacao_encerra_o_motor() {
    let mut pelo_ramo = 0;
    for arma in [None, Some(0usize), Some(1usize)] {
        let (gpu, mut eng) = motor(8, 20, None);
        let mut prep = Prep::com_gpu(64, &gpu);
        prep.arma_publicacao_no_bloco = arma;
        if arma.is_none() {
            gpu.arma_falha_na_publicacao();
        }
        let r = le_ids(&mut eng, &prep, &ids(20));
        if let Err(e) = &r {
            assert!(matches!(e, ErroLeitura::Motor(Error::Backend(_))), "{arma:?}: {r:?}");
            pelo_ramo += 1;
            assert!(eng.is_closed(), "{arma:?}: o erro em next_published tem de encerrar o motor");
            assert!(prep.tamanhos().len() < 3, "{arma:?}: o bloco que esperava a publicação não foi preparado");
            let p2 = Prep::novo(64);
            assert!(matches!(le_ids(&mut eng, &p2, &ids(3)), Err(ErroLeitura::MotorEncerrado)));
            assert!(p2.blocos().is_empty());
        }
        let _ = eng.close();
        assert!(!gpu.stalled());
    }
    assert!(pelo_ramo >= 1, "nenhum caso chegou ao ramo de erro de next_published");
}

/// Falha da linha de serviço (a CPU falsa) em pontos diferentes da leitura: em qualquer caminho de erro (passo ou fim do
/// passo; a espera da publicação tem teste próprio acima) o motor fica encerrado e a chamada seguinte é recusada sem preparar nada.
#[test]
fn qualquer_falha_da_linha_de_servico_deixa_o_motor_encerrado() {
    let mut erros = 0;
    for fail_at in 0..40usize {
        let routing: Vec<Vec<Vec<u32>>> = (0..24).map(|p| (0..2).map(|l| rota(p, l)).collect()).collect();
        let gpu = FakeGpu::new_sched(
            FakeCfg { layers: 2, experts: 4, topk: 2, width: 3, routing, fail_main_at: None, copy_immediate: false, max_positions: 8 },
            SchedCfg { seed: 0xA0A, weights: [1, 1, 1, 1], budget: 300_000 },
        );
        let cfg = Config { layers: 2, experts: 4, topk: 2, width: 3, slots: 0, backlog: 0, admit: AdmitMode::TokenBarrier, first_token: 1,
                           prefill: vec![], max_positions: 8, mutant: None };
        let parts = gpu.parts(&cfg);
        let mut eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu { weighted: true, fail_at: Some(fail_at), ..FakeCpu::default() }).unwrap();
        let prep = Prep::novo(64);
        if le_ids(&mut eng, &prep, &ids(20)).is_err() {
            erros += 1;
            assert!(eng.is_closed(), "fail_at = {fail_at}: o motor ficou aberto depois do erro");
            let p2 = Prep::novo(64);
            assert!(matches!(le_ids(&mut eng, &p2, &ids(3)), Err(ErroLeitura::MotorEncerrado)));
            assert!(p2.blocos().is_empty());
        }
        let _ = eng.close();
        assert!(!gpu.stalled(), "fail_at = {fail_at}");
    }
    assert!(erros > 0, "nenhuma falha injetada chegou a derrubar a leitura");
}

#[test]
fn contadores_de_posicao_divergentes_sao_recusados_sem_efeito() {
    let (gpu, mut eng) = motor(8, 8, None);
    let prep = Prep::na_posicao(64, 5);
    assert!(matches!(le_ids(&mut eng, &prep, &ids(3)), Err(ErroLeitura::ContadoresDivergem { motor: 0, backend: 5 })));
    assert!(prep.blocos().is_empty());
    assert_eq!(eng.position(), 0);
    fecha(&gpu, eng);
}

#[test]
fn m_igual_a_um_e_n_passos_de_uma_posicao_como_o_caminho_por_token() {
    for n in 1..=5usize {
        let (gpu, mut eng) = motor(1, n, None);
        let prep = Prep::novo(64);
        assert_eq!(le_ids(&mut eng, &prep, &ids(n)).unwrap(), n);
        assert_eq!(prep.tamanhos(), vec![1; n]);
        assert_eq!(eng.position(), n as u64);
        let s = eng.next_published().unwrap().stats().clone();
        fecha(&gpu, eng);

        // O caminho de antes: n vezes `next_published` e `run_token`.
        let (gpu0, mut e0) = motor(1, n, None);
        for _ in 0..n {
            let p = e0.next_published().unwrap();
            e0.run_token(p).unwrap();
        }
        let s0 = e0.next_published().unwrap().stats().clone();
        assert_eq!((s.needed, s.hits, s.cpu), (s0.needed, s0.hits, s0.cpu), "n = {n}");
        assert_eq!(e0.position(), n as u64);
        fecha(&gpu0, e0);
    }
}

/// O oráculo tem sensibilidade: um plano que ignora o resto (último bloco sempre cheio, lendo além do pedido) é acusado.
#[test]
fn oraculo_acusa_o_plano_com_a_regra_do_resto_invertida() {
    fn plano_errado(n: usize, mp: usize) -> Vec<usize> {
        vec![mp; n.div_ceil(mp)]
    }
    let mut acusados = 0;
    for n in 1..=24usize {
        assert!(plano_confere(&plano_blocos(n, 8), n, 8));
        if !plano_confere(&plano_errado(n, 8), n, 8) {
            acusados += 1;
        }
    }
    assert_eq!(acusados, 24 - 3, "só os múltiplos de 8 (8, 16, 24) escapam do mutante");
}

// ---- Instrumento de prefill (perfil_bloco): o que liga e desliga não muda, e o escopo do registro ---------------------------------------

use crate::perfil_bloco::DadosPedido;

/// Roda `n` ids com o instrumento ligado ou desligado e devolve (blocos vistos pelo preparo, begin_step do falso, posição final, entrada do pedido).
fn roda_perfil(n: usize, perfil: bool, decode: usize) -> (Vec<(u64, Vec<usize>)>, Vec<(u64, usize)>, u64, Option<crate::perfil_bloco::RecPedido>) {
    let (gpu, mut eng) = motor_pf(8, n + decode, None, perfil);
    let prep = Prep::com_gpu(n + decode + 8, &gpu);
    let mut pf = eng.perfil_armar(plano_blocos(n, 8).len());
    assert_eq!(pf.ativo(), perfil);
    le_ids(&mut eng, &prep, &ids(n)).unwrap();
    pf.logits_ini();
    pf.desarma();
    // o decode fica fora do registro
    for k in 0..decode {
        le_ids(&mut eng, &prep, &[(k * 5 + 1) % VOCAB]).unwrap();
    }
    let d = DadosPedido { modelo: String::new(), indice: 0, aquecimento: false, n, plano: plano_blocos(n, 8), prefill_ms: 0.0, decode_passo_ms_mediana: 1.0 };
    let mut e = pf.fecha_com(d, &mut || {
        let _ = super::backend::Backend::relax(&*gpu);
    });
    if let Some(e) = e.as_mut() {
        e.prefill_ms = e.total_perfil_ns() as f64 / 1e6; // o binário mede o mesmo trecho; aqui só o fechamos
    }
    let r = (prep.blocos(), gpu.begin_steps(), eng.position(), e);
    fecha(&gpu, eng);
    r
}

#[test]
fn perfil_ligado_nao_muda_a_sequencia_do_backend_e_desligado_nao_registra() {
    for (n, decode) in [(1usize, 0usize), (8, 0), (13, 0), (24, 0), (13, 4), (89, 3)] {
        let (b_off, s_off, p_off, e_off) = roda_perfil(n, false, decode);
        let (b_on, s_on, p_on, e_on) = roda_perfil(n, true, decode);
        assert_eq!((&b_off, &s_off, p_off), (&b_on, &s_on, p_on), "n = {n}, decode = {decode}: ligar não pode mudar o que o motor faz");
        assert!(e_off.is_none(), "desligado não grava nada");
        let e = e_on.expect("ligado grava uma entrada");
        let esperado = plano_blocos(n, 8);
        assert_eq!(e.blocos.iter().map(|b| b.m).collect::<Vec<_>>(), esperado, "blocos = plano_blocos(n, 8), sem registro de decode (decode = {decode})");
        assert_eq!(e.servico.len(), esperado.len());
        assert!(!e.servico_aberto() && e.servico_extra == 0, "serviço fechado e zero registros de decode: {:?}", e.servico);
        assert!(e.problemas().is_empty(), "{:?}", e.problemas());
        assert!(e.blocos.iter().enumerate().all(|(i, b)| b.pos_ini == 8 * i as u64 && !b.f0_pendente));
        assert!(e.blocos.windows(2).all(|w| w[1].gen == w[0].gen + 1));
    }
}

#[test]
fn engine_desligado_nao_aloca_e_a_guarda_e_inerte() {
    let (gpu, eng) = motor_pf(8, 8, None, false);
    assert!(!eng.perfil_ligado() && !eng.perfil_ativo());
    let mut g = eng.perfil_armar(100);
    assert!(!g.ativo());
    g.logits_ini();
    g.desarma();
    assert!(!eng.perfil_ativo());
    fecha(&gpu, eng);
}
