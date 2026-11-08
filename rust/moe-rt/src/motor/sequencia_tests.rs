//! Reset de sequência no `Engine` (critério `criterio-reset-sequencia-20261108.md`, prova 1): testes simulados, sem GPU.
//! Validam o contrato de `nova_sequencia` (posição a 0, `t` contíguo, recusas, falha fecha) e a ordem das chamadas ao backend. NÃO
//! provam que o KV velho fica mascarado nem a igualdade bit a bit dos logits (isso é a prova 2, na GPU).
use super::engine::Engine;
use super::fake::{FakeCfg, FakeCpu, FakeGpu, SchedCfg};
use super::leitura::{le_ids, le_ids_com_pendente, plano_blocos, PreparaPasso};
use super::{AdmitMode, Config, Error};
use std::sync::{Arc, Mutex};

const VOCAB: usize = 50;
const MP: usize = 8;

/// Preparo de teste que zera a posição quando o backend falso recebeu um `reset_sequence` novo (como o `Dense::reset` do modelo).
struct Prep {
    gpu: Arc<FakeGpu>,
    visto: Mutex<usize>,
    pos: Mutex<u64>,
    blocos: Mutex<Vec<(u64, usize)>>,
    falha_no_bloco: Option<usize>,
}

impl Prep {
    fn novo(gpu: &Arc<FakeGpu>) -> Self {
        Self { gpu: gpu.clone(), visto: Mutex::new(0), pos: Mutex::new(0), blocos: Mutex::new(vec![]), falha_no_bloco: None }
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
        MP
    }
    fn max_len(&self) -> usize {
        4096
    }
    fn posicao(&self) -> u64 {
        self.sincroniza();
        *self.pos.lock().unwrap()
    }
    fn prepara(&self, ids: &[usize]) -> Result<(), &'static str> {
        self.sincroniza();
        let mut b = self.blocos.lock().unwrap();
        if Some(b.len()) == self.falha_no_bloco {
            return Err("preparo falhou");
        }
        let mut pos = self.pos.lock().unwrap();
        b.push((*pos, ids.len()));
        *pos += ids.len() as u64;
        Ok(())
    }
}

fn rota(p: usize, l: usize) -> Vec<u32> {
    vec![((p + l) % 4) as u32, ((p + l + 1 + (p / 4) % 3) % 4) as u32]
}

fn motor(total: usize) -> (Arc<FakeGpu>, Engine<FakeGpu>) {
    let routing: Vec<Vec<Vec<u32>>> = (0..total.max(1)).map(|p| (0..2).map(|l| rota(p, l)).collect()).collect();
    let gpu = FakeGpu::new_sched(
        FakeCfg { layers: 2, experts: 4, topk: 2, width: 3, routing, fail_main_at: None, copy_immediate: false, max_positions: MP },
        SchedCfg { seed: 0xA0A, weights: [1, 1, 1, 1], budget: 300_000 },
    );
    let cfg = Config { layers: 2, experts: 4, topk: 2, width: 3, slots: 0, backlog: 0, admit: AdmitMode::TokenBarrier, first_token: 1,
                       prefill: vec![], max_positions: MP, mutant: None };
    let parts = gpu.parts(&cfg);
    let eng = Engine::new(gpu.clone(), parts, cfg, FakeCpu { weighted: true, ..FakeCpu::default() }).expect("motor");
    (gpu, eng)
}

fn ids(n: usize, sem: usize) -> Vec<usize> {
    (0..n).map(|i| (i * 7 + 3 + sem) % VOCAB).collect()
}

fn passos(n: usize) -> Vec<(u64, usize)> {
    plano_blocos(n, MP).into_iter().scan(0u64, |p, m| { let r = (*p, m); *p += m as u64; Some(r) }).collect()
}

fn fecha(gpu: &Arc<FakeGpu>, eng: Engine<FakeGpu>) {
    let _ = eng.close();
    assert!(!gpu.stalled());
    assert_eq!(gpu.violations(), vec![]);
}

/// Dois pedidos no mesmo motor, com `nova_sequencia` no meio: posição volta a 0, o backend vê os mesmos `begin_step` de um motor novo
/// (posições recomeçando em 0), o reset vem entre o último passo do primeiro e o primeiro do segundo, e `t` segue contíguo.
#[test]
fn dois_pedidos_no_mesmo_motor_com_nova_sequencia_no_meio() {
    for (n1, n2) in [(13usize, 11usize), (8, 8), (5, 20), (1, 1)] {
        let (gpu, mut eng) = motor(n1.max(n2));
        let prep = Prep::novo(&gpu);
        le_ids(&mut eng, &prep, &ids(n1, 0)).unwrap();
        assert_eq!(eng.position(), n1 as u64);
        let b1 = plano_blocos(n1, MP).len() as u32;
        // O Published pendente é do token `1 + b1`: contíguo.
        let p = eng.next_published().unwrap();
        assert_eq!(p.generation(), 1 + b1);
        let mut pend = Some(p);
        eng.nova_sequencia().unwrap();
        assert_eq!(eng.position(), 0);
        assert_eq!(prep.posicao(), 0);
        le_ids_com_pendente(&mut eng, &prep, &ids(n2, 3), &mut pend).unwrap();
        assert!(pend.is_none(), "o pendente foi consumido pelo primeiro bloco");
        assert_eq!(eng.position(), n2 as u64);
        let b2 = plano_blocos(n2, MP).len() as u32;
        assert_eq!(eng.next_published().unwrap().generation(), 1 + b1 + b2, "t não é zerado pelo reset");
        // Ordem das chamadas ao backend: igual à de dois motores novos, com um único reset entre os dois pedidos.
        let esperado: Vec<(u64, usize)> = passos(n1).into_iter().chain(passos(n2)).collect();
        assert_eq!(gpu.begin_steps(), esperado, "n = {n1} + {n2}");
        assert_eq!(gpu.resets(), vec![passos(n1).len()], "um reset, depois dos passos do primeiro pedido e antes dos do segundo");
        // Os blocos do preparo recomeçam em 0 no segundo pedido, como num motor novo.
        let blocos = prep.blocos.lock().unwrap().clone();
        let esp_blocos: Vec<(u64, usize)> = passos(n1).into_iter().chain(passos(n2)).collect();
        assert_eq!(blocos, esp_blocos);
        fecha(&gpu, eng);
    }
}

/// O segundo pedido num motor reutilizado vê a mesma sequência de `begin_step` que um motor novo vê para o mesmo pedido.
#[test]
fn segundo_pedido_reutilizado_enxerga_o_mesmo_que_um_motor_novo() {
    let (n1, n2) = (17usize, 9usize);
    let (gpu_a, mut a) = motor(n2);
    let prep_a = Prep::novo(&gpu_a);
    le_ids(&mut a, &prep_a, &ids(n2, 3)).unwrap();
    let fresco = gpu_a.begin_steps();
    fecha(&gpu_a, a);
    let (gpu_b, mut b) = motor(n1.max(n2));
    let prep_b = Prep::novo(&gpu_b);
    le_ids(&mut b, &prep_b, &ids(n1, 0)).unwrap();
    let antes = gpu_b.begin_steps().len();
    let mut pend = Some(b.next_published().unwrap());
    b.nova_sequencia().unwrap();
    le_ids_com_pendente(&mut b, &prep_b, &ids(n2, 3), &mut pend).unwrap();
    assert_eq!(gpu_b.begin_steps()[antes..], fresco[..]);
    fecha(&gpu_b, b);
}

/// Token enfileirado e ainda não terminado: recusado, sem reset, sem mexer na posição; depois do `finish_token` é aceito.
#[test]
fn nova_sequencia_com_token_em_andamento_e_recusada() {
    let (gpu, mut eng) = motor(8);
    let prep = Prep::novo(&gpu);
    le_ids(&mut eng, &prep, &ids(3, 0)).unwrap();
    let p = eng.next_published().unwrap();
    prep.prepara(&ids(1, 5)).unwrap();
    eng.enqueue_step(p, 1).unwrap();
    assert!(matches!(eng.nova_sequencia(), Err(Error::Config(_))));
    assert_eq!(gpu.resets(), Vec::<usize>::new(), "recusa sem efeito no backend");
    assert_eq!(eng.position(), 3, "a posição só avança em finish_token");
    assert!(!eng.is_closed());
    eng.finish_token().unwrap();
    assert_eq!(eng.position(), 4);
    eng.nova_sequencia().unwrap();
    assert_eq!((eng.position(), gpu.resets().len()), (0, 1));
    fecha(&gpu, eng);
}

#[test]
fn nova_sequencia_em_motor_fechado_e_recusada() {
    let (gpu, mut eng) = motor(8);
    let prep = Prep::novo(&gpu);
    le_ids(&mut eng, &prep, &ids(3, 0)).unwrap();
    eng.encerra();
    assert!(matches!(eng.nova_sequencia(), Err(Error::Cancelled)));
    assert_eq!(gpu.resets(), Vec::<usize>::new());
    fecha(&gpu, eng);
}

/// Falha no meio da leitura fecha o motor; `nova_sequencia` depois dela é recusada (o chamador recria o motor).
#[test]
fn falha_no_meio_fecha_o_motor_e_nao_permite_nova_sequencia() {
    let (gpu, mut eng) = motor(30);
    let mut prep = Prep::novo(&gpu);
    prep.falha_no_bloco = Some(1);
    assert!(le_ids(&mut eng, &prep, &ids(20, 0)).is_err());
    assert!(eng.is_closed());
    assert!(matches!(eng.nova_sequencia(), Err(Error::Cancelled)));
    assert_eq!(gpu.resets(), Vec::<usize>::new());
    fecha(&gpu, eng);
}

/// Falha do backend em `reset_sequence`: o motor fecha e a posição não é tocada em silêncio.
#[test]
fn falha_do_reset_no_backend_fecha_o_motor() {
    let (gpu, mut eng) = motor(8);
    let prep = Prep::novo(&gpu);
    le_ids(&mut eng, &prep, &ids(3, 0)).unwrap();
    gpu.arma_falha_no_reset();
    assert!(matches!(eng.nova_sequencia(), Err(Error::Backend(_))));
    assert!(eng.is_closed());
    assert_eq!(gpu.resets(), Vec::<usize>::new());
    fecha(&gpu, eng);
}
