# Estado do A1.4 (a medida final do motor D1), 1/11

A1.0 a A1.3 estão aprovados, commitados e empurrados (último commit c29ef34). Plano: `docs/PLANO-runtime-rust-moe-20261029.md`,
"A1 — critério de conclusão", item **Tempo** (llama.cpp como base, decisão do Breno de 31/10) e a **Condição de reabertura da busca
antecipada**. O plano termina neste passo.

## Sub-passos

- **A1.4a, o critério gravado antes:** `qual-runs/A1-MEDIDA-<data>/criterio.txt` com as 6 células (1B e 3B × 10/25/50%), as
  políticas (motor D1 em `TokenBarrier` e por evento, `lfu_cpu` de hoje, llama.cpp cache e `-ncmoe`), 5 execuções intercaladas,
  a previsão do simulador em faixa (protocolo `token`: cobrança da rodada 51 contra a de agora), o que é inconclusivo e o
  critério de aceitação dos ajustes de `c_cpu`, `c_cpu_quiet` e `o_layer`. Nenhum limiar muda depois de ver o número.
- **A1.4b, o harness do motor D1 sobre o modelo inteiro:** um binário novo em `rust/moe-rt/src/bin/` que roda as conversas dos
  traços pelo `Engine` + `CudaBackend::new_model` (o `m2a-cache` roda o `GpuModel` com `ExpertCache`, que é o `lfu_cpu` de hoje),
  com a mesma conferência de logits (1e-4), os mesmos contadores e o mesmo formato de saída. Teste curto de sanidade na 4090.
  Granite 1B (32 experts, top-8) e 3B (40 experts, top-8) cabem nos limites do A1.3 (`K` potência de dois, `E` par).
- **Checkpoint de revisão (nível 2, gatilho protocolo)** sobre o critério e o harness, antes da janela longa.
- **A1.4c, a janela longa:** só com o aval do Breno, depois de eu informar tempo estimado, GPU e núcleos. Motor D1, `lfu_cpu` de
  hoje e llama.cpp (`tools/llama-forcado`, as linhas medidas de novo na mesma sessão), `taskset -c 0-15`, processos em
  execução registrados.
- **A1.4d, análise:** diferença motor menos `lfu_cpu` com o intervalo das 5 intercaladas contra as duas cobranças previstas
  (−0,021 e −0,038 ms no 1B, −0,027 e −0,045 ms no 3B, a 50%), cauda de latência (as voltas isoladas do plano), custo dos
  dtod do bloco denso, efeito dos limites do A1.3, reajuste dos três parâmetros e o que fica aberto.
- **Revisão final (nível 2)** ao fechar o A1.4; com os dois aprovando, plano, este arquivo e o HANDOFF e o commit local.

## Janelas (estimativa inicial, a refinar com o teste de sanidade do A1.4b)

- Sanidade: minutos, GPU livre, `taskset -c 0-15`.
- Janela longa: as conversas 4 a 7 com aquecimento nas 0 a 3, 2 modelos × 3 orçamentos × 3 políticas locais × 5 execuções, mais
  as linhas do llama.cpp. A medida do C (30/10) com 2 políticas levou perto de uma hora de relógio; espera-se de 1,5 a 3 horas.
  O número firme sai do sanity e vai ao Breno antes de começar.

GPU: fixe a 4090 por UUID (`CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd`) e confira ociosa antes e depois.

## Andamento

- Plano de sub-passos mostrado ao Breno e este arquivo criado.
- **A1.4a feito (sem commit, vai ao checkpoint):** `qual-runs/A1-MEDIDA-20261101/criterio.txt` (6 células, linhas L1 a L6, previsão das duas
  cobranças por célula, regra de decisão do plano, aceitação do motor a 1,05 x o `lfu_cpu` como proposta a confirmar, comparação
  com o llama.cpp por faixas, ajustes de `c_cpu`/`c_cpu_quiet`/`o_layer`, cauda de latência, custo do bloco denso, o que invalida a
  janela). Gravado antes de existir o harness.
- **A1.4b feito (sem commit, vai ao checkpoint):** `rust/moe-rt/src/bin/a1-motor.rs` (linhas resident, motor_cheio, motor_evento, motor_barreira; texto-a em motor à parte;
  contadores por diferença; cauda de latência; `--ms-dir` grava o ms de cada token) e `CudaBackend::reset_sequence` (teste
  `reset_da_sequencia_repete_os_logits_da_cpu` em `tests/motor_model.rs`). Sanidade na 4090 (1B, 25%, conversas 4 e 5, aquecimento 0 a 3,
  build release, `taskset -c 0-15`): logits dentro de 1e-4 nas quatro configurações e `needed` exato; motor por evento 5,73 ms/token
  e `lfu_cpu` do `m2a-cache` 5,93 (residente 2,63 nos dois; motor com tudo residente 2,20). Uma execução de cada, sem valor de
  decisão. Emenda do critério: o roteamento não é conferido por id (o motor não expõe), vale `needed` exato e os logits.
- **Checkpoint, rodada 19:** os dois pediram mudanças. Corrigido: roteamento por ids contra o traço (`Engine::layer_ids`, `mesmos_ids`, regra de
  0,01% no critério), `Check::add` recusa vetor truncado/vazio/não finito (teste), `--ms-dir` também no `m2a-cache` (hash e contagens iguais
  com e sem a opção, uma amostra por token), teste do reset por `sequence_position` (o modelo sintético tem logits insensíveis ao contexto:
  a CPU dá os mesmos bits; o mutante `reset` sem efeito fica vermelho), e dez emendas datadas no critério (estatística por repetição, consequência
  de "não passa", backlog 0 em 5 pareadas, L4 menos L3 como diferença líquida, deriva de L4, "célula suspensa", nota da sanidade, justificativa 3.1).
- **Rodada 20:** rev-1 APPROVE (P3 sobre a leitura causal da nota, acatado como hipótese); rev-2 REQUEST CHANGES por um P2 do critério (a regra "rot.igual do lfu_cpu >= residente" suspendia o
  3B a 25% por construção, pelos arquivos do C). Corrigido: mesma regra de 0,01% para as quatro linhas; `criterio-v1.txt` guarda a redação revisada na rodada 19 (sha256 no fim do critério);
  arquivos do `--ms-dir` com prefixo por harness (`a1-motor-ms-`, `m2a-ms-`); o `a1-motor` recusa linhas do motor sem `--warmup` (nota do rev-2: o primeiro token paga o `prefill`).
- **Checkpoint aprovado pelos dois na rodada 21** (rodadas 19 e 20 pediram mudanças). A1.4a e A1.4b commitados localmente. Falta: o roteiro da janela (A1.4c) e o aval do Breno para a janela longa, mais a confirmação dele do limiar de 1,05 x e da consequência de "não passa".
