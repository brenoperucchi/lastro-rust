# Estado do A1.3 (o motor D1 completo), 1/11

A1.1 e A1.2 estão aprovados e commitados (2a4faaf, 5f88db7). Plano: `docs/PLANO-runtime-rust-moe-20261029.md`, "A1 — critério
de conclusão" e "Limites que ficam do A1.2".

## Sub-passos

- **A1.3a, o motor D1 sobre o `CudaBackend` com os contadores do simulador:** as fixtures de `tests/fixtures/a1_sim_token_barreira.json`
  (modo `TokenBarrier`) rodam no `Engine` com o backend de CUDA e threads de verdade. O roteamento do simulador é forçado por
  roteadores construídos (entrada one-hot por token, peso do roteador `K - posição` nos experts da fixture) e experts de pesos
  zero, então o `h` não muda e só os contadores contam. Limite: o top-k do backend exige `K` potência de dois e `E` par
  (`GEMV_BM = 2`), então cabem 31 das 64 fixtures. Teste em `rust/moe-rt/tests/motor_gpu.rs`.
- **A1.3b, o `y` escrito pelo `Pool`:** a CPU de serviço escreve o `y` pelo `Pool` e a linha de serviço grava a flag depois
  (sem store não temporal no `y`, ou `sfence` antes da flag). O conferidor do A1.0 volta com o `y` escrito pelo `Pool`
  (critério gravado antes em `qual-runs/A1-POOL-Y-<data>/criterio.txt`; N1 acusado; zero erros nas voltas do critério de
  `qual-runs/A1-DOORBELL-20261031/criterio.txt`).
- **Checkpoint de revisão (nível 2, gatilho protocolo)** ao fechar o A1.3b: aprovado pelos dois na rodada 14 (13 pediu mudanças).
- **A1.3c, o modelo inteiro no motor:** embedding, atenção, blocos MoE pelo D1 e logits do vocabulário (reaproveita o
  `GpuModel` do M1b), pesos do softmax valendo na CPU; logits contra a referência a 1e-4.
- **A1.3d, controles negativos na GPU e o modo antigo:** janela forçada da cópia contra o último leitor (versão certa bit a bit
  em 20 de 20, sem a espera acusada), `ExpertCache` com `AdmitMode::Barrier` bit a bit igual antes e depois; motor normal, loom e
  trybuild seguem passando.
- **Revisão final (nível 2)** ao fechar o A1.3d; com os dois aprovando, plano, este arquivo e o HANDOFF e o commit local.

GPU: fixe a 4090 por UUID (`CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd`). Janela longa de medida (A1.4) só com
o aval do Breno.

## Andamento

- **A1.3a feito (aprovado no checkpoint):** `rust/moe-rt/tests/motor_gpu.rs`, `token_barrier_no_cuda_bate_com_o_simulador` (31 fixtures
  iguais ao simulador no motor de CUDA com threads de verdade; `h` intacto com experts zerados) e o controle
  `controle_roteamento_deslocado_e_acusado` (26 de 31 acusadas com o roteamento deslocado pelo índice do token).
- **A1.3b feito (aprovado no checkpoint):** `Cpu::fill_rows` (`rust/moe-rt/src/cpu.rs`; `gemv` e a soma final dos experts passaram a
  usá-lo, mesmas faixas e mesma conta), `rust/moe-rt/tests/sem_store_nao_temporal.rs` (varredura textual de `src` mais o teste do
  detector; verificado vermelho com `maskmovdqu` plantado) e o conferidor `rust/moe-rt/src/bin/a1-pool-y.rs` (critério gravado
  antes em `qual-runs/A1-POOL-Y-20261101-b/criterio.txt`, a primeira rodada em `qual-runs/A1-POOL-Y-20261101/` fica de
  histórico; 31.010 voltas conferidas sem erro; N1, N2 e N3 acusados, o N3 com as faixas de fora da 0 retidas 20 µs). Limite: no motor de produção quem escreve a área mapeada continua sendo a linha de
  serviço (`HostExperts::compute` devolve um vetor); o conferidor cobre o caso do plano, com as threads do `Pool` escrevendo a área.
