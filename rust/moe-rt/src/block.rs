//! Bloco MoE de uma camada do Granite MoE (`GraniteMoeMoE` do transformers):
//! roteador `logits = W_r·x`, top-k dos logits, softmax só sobre os k escolhidos; cada expert é
//! `down · (silu(gate) ⊙ up)` com `[gate; up] = gate_up·x`; a saída é a soma dos experts pesada pelo softmax.
use crate::cpu::{Cpu, ExpertRef};
use crate::weights::{Checkpoint, WeightError};
use cutile::half::bf16;

/// Pesos de uma camada na RAM, um par de matrizes por expert (a unidade que o cache de experts move).
pub struct HostLayer {
    pub layer: usize,
    pub hidden: usize,
    pub inter: usize,
    pub k: usize,
    /// `[E, H]`
    pub router: Vec<bf16>,
    /// por expert: `[2I, H]` e `[H, I]`
    pub gate_up: Vec<Vec<bf16>>,
    pub down: Vec<Vec<bf16>>,
}

impl HostLayer {
    pub fn experts(&self) -> usize {
        self.gate_up.len()
    }

    /// Bytes BF16 de um expert (gate_up + down).
    pub fn expert_bytes(&self) -> usize {
        (2 * self.inter * self.hidden + self.hidden * self.inter) * 2
    }

    pub fn load(ck: &Checkpoint, layer: usize, k: usize) -> Result<Self, WeightError> {
        let p = format!("model.layers.{layer}.block_sparse_moe");
        let (router, rs) = ck.bf16(&format!("{p}.router.layer.weight"))?;
        let (gu, gs) = ck.bf16(&format!("{p}.input_linear.weight"))?;
        let (dn, ds) = ck.bf16(&format!("{p}.output_linear.weight"))?;
        let (e, hidden) = (rs[0], rs[1]);
        let inter = gs[1] / 2;
        let shape_ok = gs == [e, 2 * inter, hidden] && ds == [e, hidden, inter];
        if !shape_ok {
            return Err(WeightError::Format(format!("formas inesperadas: router {rs:?}, gate_up {gs:?}, down {ds:?}")));
        }
        let per_gu = 2 * inter * hidden;
        let per_dn = hidden * inter;
        Ok(Self {
            layer,
            hidden,
            inter,
            k,
            router,
            gate_up: gu.chunks_exact(per_gu).map(|c| c.to_vec()).collect(),
            down: dn.chunks_exact(per_dn).map(|c| c.to_vec()).collect(),
        })
    }

    /// Bloco inteiro na CPU.
    pub fn forward_cpu(&self, x: &[f32], cpu: &Cpu) -> (Vec<f32>, Routing) {
        let mut logits = vec![0f32; self.experts()];
        cpu.gemv(&self.router, self.experts(), self.hidden, x, &mut logits);
        let r = Routing::from_logits(&logits, self.k);
        let y = cpu.experts(x, &self.refs(&r), self.hidden, self.inter);
        (y, r)
    }

    /// Os experts escolhidos em `r`, como referências para o caminho de CPU.
    pub fn refs(&self, r: &Routing) -> Vec<ExpertRef<'_>> {
        r.experts.iter().zip(&r.weights).map(|(&e, &g)| ExpertRef { gate_up: &self.gate_up[e], down: &self.down[e], weight: g }).collect()
    }
}

/// Decisão do roteador para um token.
#[derive(Clone, Debug, PartialEq)]
pub struct Routing {
    pub logits: Vec<f32>,
    /// Experts escolhidos, do maior logit para o menor.
    pub experts: Vec<usize>,
    pub weights: Vec<f32>,
}

impl Routing {
    /// Top-k dos logits (empate: o de menor índice primeiro) e softmax sobre os k.
    pub fn from_logits(logits: &[f32], k: usize) -> Self {
        let mut order: Vec<usize> = (0..logits.len()).collect();
        order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
        order.truncate(k);
        let top = logits[order[0]];
        let ex: Vec<f32> = order.iter().map(|&i| (logits[i] - top).exp()).collect();
        let sum: f32 = ex.iter().sum();
        Self { logits: logits.to_vec(), experts: order, weights: ex.iter().map(|v| v / sum).collect() }
    }
}
