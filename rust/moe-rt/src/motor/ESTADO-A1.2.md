# Estado do A1.2 (as peças na GPU), 1/11

A1.2 aprovado pelos dois revisores (final: rodada 12) e commitado localmente, sem push. A1.1 está commitado (2a4faaf). Plano: `docs/PLANO-runtime-rust-moe-20261029.md`,
"A1 — implementação" e o desenho D1 (A0b, passos 1 a 6).

## Sub-passos

- **A1.2a, top-k na GPU (feito, aprovado no checkpoint):** `rust/moe-rt/src/route.rs`, kernel `route_topk` (um bloco; `EP`
  potência de dois, `K` = 8; sentinela menos infinito). Teste `topk_na_gpu_bate_com_o_do_host`: 800 casos (8 tipos de
  logits, E = 32 e 40) contra `Routing::from_logits`; controles do desempate (`TS = -1`) e da máscara do padding.
- **A1.2b, destino mapeado (feito, aprovado no checkpoint, rodadas 7 e 8):** `route_topk` publica ids, pesos e `x` por
  ponteiro; `LayerIo.w`; `a1-cutile-leitor` passou (30.010 voltas, N1 e N2 acusados). Registro em
  `qual-runs/A1-CUTILE-LEITOR-20261101/`. Os dois revisores aprovaram a rodada 8; o commit sai com a revisão final.
- **A1.2c, um lançamento para os acertos (feito, aprovado):** `rust/moe-rt/src/hits.rs`, teste
  `hits::tests::acertos_na_gpu_batem_com_a_cpu_a_1e4` (pior erro 3,6e-7).
- **A1.2d, backend de CUDA (feito, aprovado):** `rust/moe-rt/src/cuda_backend.rs`, teste
  `cuda_backend::tests::pilha_de_blocos_moe_pelo_d1_bate_com_a_cpu_a_1e4` (pior erro 2,2e-7). `HostExperts::compute` ganhou
  os pesos; `Pinned`/`Mapped` aceitam memória do driver; `LayerIo` ganhou `w` e palavras mapeadas (`Words` opaco).
  Rodada 9: `Words` privado, espera da stream nos erros, slot de preenchimento (banco `slots + 1`), oráculo sem NaN; rodadas 10 a 12 fecharam a posse dos tensores (partições emprestadas).

Próximo: A1.3, o motor completo com o modelo inteiro.

GPU: fixe a 4090 por UUID (`CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd`).
