//! O teste que fecha a stream de cópia com um portão fica sozinho neste binário: qualquer sincronização do contexto enquanto
//! a stream está fechada (a primeira carga de um kernel, ou outro teste da GPU rodando em paralelo no mesmo processo) espera
//! o portão e prende o teste até o prazo. O cargo roda os binários de teste um depois do outro, e o teste aquece os kernels
//! num modelo descartável antes de fechar o portão.
use moe_rt::cache::{AdmitMode, CacheConfig, Freq, Policy};
use moe_rt::gpu_model::{Experts, GpuModel, ATTN_BN};

mod common;
use common::{gpu_or_skip, tiny_config, tiny_model_with, values};

/// C: por evento, a frequência do `lfu_cpu` é a de uma `Freq` recalculada das rotas devolvidas (todos os k experts da camada,
/// um passo por camada), inclusive quando o expert pedido ainda está carregando. No modelo de teste a cópia termina antes de
/// o expert voltar a ser pedido; o portão na stream de cópia segura as admissões da primeira metade como `Loading`.
#[test]
fn lfu_cpu_conta_o_toque_do_expert_que_ainda_esta_carregando() {
    let Some(device) = gpu_or_skip() else { return };
    let mut json = tiny_config(33);
    json["num_local_experts"] = 16.into();
    let (_tmp, m) = tiny_model_with("lfu-portao", json);
    let toks: Vec<usize> = values(48, 141).iter().map(|v| ((v + 1.0) * 16.0) as usize % m.cfg.vocab).collect();
    let cfg = CacheConfig { slots: 3, policy: Policy::LfuCpu, backlog: 4, cpu_threads: 2, admit: AdmitMode::Event };
    {
        let mut aquece = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg.clone()).unwrap();
        for &t in &toks {
            aquece.step(t).unwrap();
        }
    }
    let mut g = GpuModel::upload_cached(&m, device.new_stream().unwrap(), ATTN_BN, cfg).unwrap();
    let mut freq = Freq::new(m.cfg.layers * m.cfg.experts, m.cfg.layers);
    let mut step = 0;
    let Experts::Cached(cache) = &g.experts else { unreachable!() };
    let mut portao = Some(cache.hold_copies().unwrap());
    for (i, &t) in toks.iter().enumerate() {
        if i == toks.len() / 2 {
            portao.take().unwrap().send(()).expect("o portão abriu pelo prazo: algo sincronizou a stream de cópia");
        }
        let (_, routes) = g.step(t).unwrap();
        for (l, r) in routes.iter().enumerate() {
            for &x in &r.experts {
                freq.touch(l * m.cfg.experts + x, step);
            }
            step += 1;
        }
    }
    let Experts::Cached(cache) = &g.experts else { unreachable!() };
    assert!(cache.stats.loading > 0, "nenhum pedido de expert carregando: o teste não cobre o toque do `Loading` {:?}", cache.stats);
    for k in 0..m.cfg.layers * m.cfg.experts {
        assert_eq!(cache.freq().score(k).to_bits(), freq.score(k).to_bits(), "chave {k}");
    }
}
