# Plano — Runtime híbrido de orquestração MoE para o Strata (Rust + CUDA)

Data: 2026-10-24. Autor: strata-opencode-exec (opencode local, modelo Swift via llm-gateway).
Revisões anteriores: exec Claude do space (pane encerrado em 24/10, antes da troca para OpenCode) — achados A1–A6
aplicados na v1; rodada formal `strata-opencode-2` com rev-1 (Codex) e rev-2 (Claude), 24/10 — REQUEST_CHANGES dos
dois, aplicada nesta v2. Versão de referência do Strata (decisão do Breno, 24/10): **upstream Niko1221/Strata no
commit mais recente** — a engine Qwen é a série v0.2.x; `0.1.49-2` é o package do Control e do motor DeepSeek, não
da engine deste plano. Citações desta v2 conferidas em `v0.2.4.2` (dfa3f402), exceto onde marcado 0.1.31.

## O que mudou nesta versão (v2) com a rodada strata-opencode-2

- **Premissa central reduzida (F1/F2, P1):** o upstream já tem política por custo medido — `STRATA_RAM_ADAPT2=1`
  promove experts "ranked by the stall they are expected to save per byte" (`generate.cpp:14965` em v0.2.4.2) — mas
  o bloco exige `!multi_gpu && !peer.valid() && !remote_opt`: a política por custo está desligada exatamente no
  cenário duas GPUs que motiva o plano. A lacuna passou de "contagem vs. custo" para **"estender RAM_ADAPT2 ao
  multi-GPU (helper CUDA1–3 / peer / layer split)"**.
- **Fase 0 em dois degraus (F5):** teto primeiro pelos contadores que já existem (`--stats`, espera por helper,
  `STRATA_TRACE`, `--dump-routing`), sem tocar em código; NVTX por camada só depois, opt-in, em `--no-token-graph`
  (o regime padrão é um CUDA graph por token — faixas de host não separam tempo por camada em replay), com overhead
  medido em três braços.
- **Fase 1.5 nova (F3):** sweep das opções existentes vira o baseline que a Fase 2 precisa bater.
- **Gate da Fase 2** ganhou TTFT e prefill como guardas de não-regressão, por contexto, prompt frio e quente (F7).
- **Contrafactual do teto corrigida (F6):** `--adapt-every 100000` é residência estática, não torna tudo residente;
  teto empírico = CUDA0 sozinho vs CUDA0 + helper no mesmo modelo.
- **Determinismo corrigido (F9):** `STRATA_IQ_MT_MIN=1` trata arredondamento por tamanho de grupo (#152), não o
  tier; a flag do tier é `--adapt-every 100000`; `--adapt-async` é declarado não-bit-exact.
- **Fork correto (F4):** o fork do motor é `~/Devs/GitHub/strata` (remote `fork`, branches `fork-01431`); os
  `strata-bench-core-*` são clones do harness de bench e ficam como lugar de rodar/registrar.
- **Fase 2 delimitada (rev-1 #5):** o tier remoto é estático (`remote_experts.hpp`); "único desvio" = escolher pares
  de troca no cache adaptativo principal; migração/desativação remota é outro experimento.

## Escopo e motivação

A pergunta central não é "Rust vs C++". É: **um modelo MoE que não cabe na VRAM executa com menos espera se a
colocação de experts for decidida por custo medido em tempo de execução, em vez de ranking estático offline —
especialmente em GPUs assimétricas (RTX 5090 + RTX 4090) com RAM como extensão?**

Rust não torna nenhuma multiplicação de matriz mais rápida que C++. O interesse na linguagem é construir a camada
de decisão (scheduler, planner, profiler) com garantias de memória e concorrência mais fortes que as de um patch
C++ no meio do hot path.

## Evidências verificadas no código real (~/Devs/GitHub/strata)

Conferidas em 2026-10-24 contra a tag `v0.2.4.2` (dfa3f402, mais recente do origin; engine Qwen série 0.2.x), com
o checkout local ainda em `9259cad4` (0.1.31) marcado onde citado. Nada vem de resumo externo.

- O Strata tem ~55k linhas próprias em `src/` na 0.1.31 (136 arquivos C++/CUDA; recontar no commit de referência
  escolhido): kernels CUDA nativos customizados (`native_flash_attn.cu`, `fused_gdn.cu`, `native_moe.cu`,
  `native_qsa*.cu`, i-quants), spec decode com suffix drafter (`src/spec/`), prefill sobreposto, multi-GPU em
  pipeline. `third_party/ggml` são só headers (~140 KB) para formatos de quantização (MIT, creditado no README).
- O cache de experts JÁ existe e é sofisticado: perfil de roteamento offline formato `STRP`, tabela de residência,
  um dono por expert (CUDA0, GPU secundária, ou CPU pool), staging pinned, modos `stripe` e `--gpu-placement layer`
  (`docs/SECOND_GPU.md`; passou de 85 para 227 linhas entre 0.1.31 e origin). O perfil distribuído tem 8.000 pares
  ranked (tamanho do arquivo, não constante do leitor).
- Política de colocação em três camadas:
  - **Posse inicial estática:** ranking `STRP` + round-robin entre secundárias (`SECOND_GPU.md`).
  - **Adaptive tier por contagem:** a cada `--adapt-every` rodadas (padrão 4) troca até `--adapt-swaps` experts
    (padrão 96) dos mais roteados, com decaimento dos contadores — em 0.1.31 a lógica está em `generate.cpp:4168-4220`
    (a declaração das opções é em :345-369; corrigido na revisão).
  - **Adaptive tier por custo — CORRIGIDO NA RODADA 4 (H1; o rev-2 retrata o próprio F1/F2 da rodada 2):**
    `STRATA_RAM_ADAPT`/`RAM_ADAPT2` é política entre **RAM e SSD** (escolhe quais experts têm cópia na RAM quando a
    RAM não comporta o complemento; custo comparado = leitura de drive/página em cache). O comentário do código é
    explícito: "the VRAM tier's own rule are untouched" (`v0.2.5:generate.cpp:14992-14999`; `DETAILS.md:172-180`).
    As guardas `!multi_gpu && !peer.valid() && !remote_opt` existem porque nesses modos a cópia em RAM se organiza de
    outro jeito — não porque "a política por custo esteja desligada no cenário do plano". O **tier de VRAM continua
    100% por contagem decaída** (`--adapt-every/--adapt-swaps/--adapt-decay/--adapt-min-gain`). Não há, hoje,
    qualquer política por custo na admissão VRAM.
  - A lacuna real, reescrita (H1): **seleção por custo marginal no tier de VRAM** (custo da miss = PCIe ou CPU pool,
    não drive), primeiro em uma GPU, depois nos modos helper/peer. O código do RAM_ADAPT2 serve de molde estrutural
    (seed do prompt, pares por camada, ganho por byte), mas NÃO é o delta. Risco de duplicação: não aconteceu para a
    VRAM.
- Upstream ativo na área — **checagem remota feita em 24/10** (pré-requisito cumprido):
  - #731 e #854: FECHADOS (slots duplicados helper/primária; PCIe share com helper, 40→66 tok/s em 2×6900 XT).
  - PR #3265 (OPEN, benchmark, comunidade): **oráculo de experts residentes na VRAM = teto 3.0x prose / 4.1x code no
    decode** (Swift 1.5 IQ2_XS, RTX 5070; 46.9 de 60 ms/token são CPU computando experts fora da VRAM). Isso é a
    contrafactual do teto que a nossa Fase 0(a) pretendia medir — já medida por terceiros num cenário uma-GPU. O
    próprio plano de um prefetcher de lookahead está explícito ali.
  - PR #2720 (OPEN, atualizado 20/10): peer-device recusa `STRATA_ADAPT_OVERLAP=1`; a espera do adaptador custa
    5.0-5.2 ms de uma janela de 45 ms em modo peer — alguém já está no caminho adaptativo+multi-GPU.
  - PR #2964 (OPEN): `STRATA_ADAPT_FETCH` admite misses do PCIe share no tier VRAM (-5,6%/rodada).
  - PR #3631 (OPEN, 21/10): `STRATA_WINDOW_TRACE` — telemetria por janela de decode; **colide com a entrega F8**;
    não reinventar o formato.
  - Issue #3834 (OPEN): RFE para `--peer-device` junto de layer split — exatamente o espaço multi-GPU do nosso delta.
  - **Diff de #2720 lido (25/10):** toca `peer_experts.{hpp,cpp}`, `verify.cpp` (adapt gate ganha `peer_has`),
    `expert_source`, pool CPU e `generate.cpp`. Traz o peer tier para o `STRATA_ADAPT_OVERLAP` (cópias em ordem de
    camada no link próprio, sinal stream-ordered). É ADJACENTE ao nosso delta: resolve a espera do gate por camada
    no modo peer, não a seleção por custo. Se aceito, nossa Fase 2 parte dele, não de main puro.
  - **Diff de #3631 lido (25/10):** `STRATA_WINDOW_TRACE=<path>` grava registro binário de 216 bytes por janela do
    loop serial de serve (wait do adaptador, movimentos RAM_ADAPT, passo adapt-async, verify/commit/draft, reads de
    experts), com `tools/window_trace.py` (30 campos, --csv). Overhead medido <1%. NÃO cobre `--pipeline-windows`.
    Consequência F8: a entrega de telemetria passa a ser **camada por camada dentro do formato do WinRec** (ou campo
    extra versionado), não um formato novo; e o degrau (a) da Fase 0 pode usar #3631 diretamente quando aceito.
  - Leitura do exec: o upstream NÃO tem ainda "RAM_ADAPT2 estendido a multi-GPU" em PR aberto, mas há atividade
    adjacente (adaptador+peer, telemetria por janela, oráculo de teto). O delta continua aberto; a entrega F8 deve
    ser redesenhada como delta sobre #3631, não um formato novo.
- Superfície em v0.2.4.2: `expert_cache.cpp` 807 linhas, `remote_experts.cpp` 499, `expert_source.cpp` 5.956,
  `generate.cpp` 17.028 — as estimativas antigas (367/328/~1.061/~4.300) eram da 0.1.31 e não valem mais. O tier de
  experts remotos é estático por construção (`remote_experts.hpp`: sem migração/adaptação na interface).

## Decisões já tomadas (e por quê)

1. **NÃO portar o motor para Rust.** Os kernels (GDN, QSA, PLE, MMVQ, i-quants, flash attention) são o produto do
   Strata; reescrevê-los em Rust seria anos de trabalho sem vantagem computacional. Rejeitado.
2. **NÃO construir um runtime paralelo que reimplementa o protocolo de camadas.** O Strata não expõe biblioteca:
   cada binário tem main próprio (`concurrent_main.cpp`, `load_main.cpp`, `overlap_main.cpp`). Um orquestrador
   externo teria que reinventar o plano de execução por camada por FFI — atrito alto, valor baixo. Rejeitado.
3. **SIM: um experimento de política adaptativa de colocação de experts**, começando como código dentro do fork do
   Strata (a interface de biblioteca precisaria nascer lá de qualquer forma). Rust entra depois, se e somente se o
   experimento mostrar gargalo mensurável E a política sobreviver à revisão — movendo o algoritmo de decisão para um
   crate Rust chamado via FFI estável.

4. **Tensor parallelism: avaliado e descartado (26/10, decisão do Breno).** O Breno perguntou se valia trazer o TP
   do llama.cpp (`--split-mode row`) ou do vLLM (`--tensor-parallel-size`) para o Strata ou para o projeto Rust.
   Descartado: (i) a issue #156 do upstream foi fechada como "not planned" — o decode é limitado pelos experts, e
   dividir a parte densa ganha pouco; (ii) medição real em #3025 (branch `ZeaMQ/Strata tp2-wip`, base 0.1.44.1,
   2× RTX 3080 20 GB — o caso mais favorável, placas iguais): `--tp 2` empatou com o layer split em todos os tamanhos
   (1.515 vs 1.527 tok/s em 11,7K; 1.907 vs 1.904 em 24,8K; 2.994 vs 2.987 em 123,6K); (iii) nosso hardware é
   desigual (5090+4090 ou 4090+3080 Ti) — TP divide cada camada em partes iguais, anda no ritmo da placa mais lenta,
   com all-reduce por camada pela PCIe, sem NVLink nem P2P nas GeForce; (iv) fora da pergunta central: TP não ataca a
   espera por experts, e é trabalho de kernel — a Fase 3 proíbe o Rust de tocar memória de GPU. Reabrir só com dado
   novo (ex.: duas placas iguais), partindo do `tp2-wip` rebaseado com A/B contra o layer split.

## Onde cada artefato vive (decisão do Breno, 24/10)

Começamos dentro do space/repo atual (`strata-opencode`); a independência acontece no tooling, não no motor.

- **`~/Devs/strata-opencode` (este repo):** plano, registros de revisão, coordenação e decisão. Nada de código de motor aqui.
- **`~/Devs/GitHub/strata`, remote `fork` (brenoperucchi/Strata):** o fork do motor JÁ é o repo próprio do desenvolvimento. Branch nova do commit de referência; instrumentação da Fase 0 e o delta multi-GPU do RAM_ADAPT2 (Fase 2) nascem lá. Publicação da política é PR ao Niko1221/Strata — não existe repo independente permanente para o patch do motor (decisão 2).
- **[SUPERADA em 26/10 pela decisão "Ryzen 9" abaixo; mantida como histórico]** Máquina local (decisão do Breno,
  25/10): a Fase 0 chegou a ser sediada na estação local (RTX 4090 24 GB, ~94 GB RAM, Linux, engine atualizada
  0.1.50 linux-cuda13 — BUILD.json f-0243-rel commit 1aff6d14; Control 0.2.6/0.1.51; 0.1.48 preservada p/ rollback).
  O protocolo reapontado para local em 26/10 (`docs/PROTOCOLO-fase0a-local-2026-10-26.md`) NÃO será executado; as
  correções factuais daquele ciclo (H1–H7 da rodada 4) continuam válidas no plano.

- **Sede da Fase 0(a) — decisão do Breno, 26/10 (substitui a máquina local):** o piloto roda no **Ryzen 9** (5950X,
  DDR4 96 GB, RTX 5090 32 GB), porque a pergunta do plano é sobre a produção. Decisões: **(1) uma GPU só — nenhuma
  segunda placa**; o teto multi-GPU fica desconhecido e o piloto mede apenas a hipótese de uma GPU na máquina da
  produção. **(2) n_ctx do piloto = 131072, igual à produção; antes, subida de teste SEM geração a 262144** só para
  registrar a linha do expert cache (slots/GiB). **(3)** engine **0.1.50 em pasta nova no Windows**, sem tocar em
  `E:\strata-01431-fork`, `E:\strata-0142` nem `E:\strata-terminal-gateway-20261024`; bench em porta própria ≠ 18199;
  produção da 18199 parada pelo STOP documentado e religada no fim. Janela pedida ao llm-gateway-exec (>1 h); nenhuma
  geração antes do "janela aberta" dele (repassado via strata-exec). Protocolo: `docs/PROTOCOLO-fase0a-ryzen9-v3-2026-10-26.md`.

## Plano em fases

### Fase 0 — Instrumentação em dois degraus (sem mudança de política)
Fork do motor: branch nova em `~/Devs/GitHub/strata` (remote `fork` = brenoperucchi/Strata), criada do commit de
referência decidido (engine Qwen v0.2.x; a 0.1.31 local é só onde as citações antigas foram tiradas). Os clones
`~/Devs/strata-bench-core-*` são o harness de bench: lugar de rodar e registrar, não fork do motor. [revisão-2, F4]

**Degrau (a) — teto pelos contadores que já existem, sem mudar código:**
- `--stats` (`expert tiers`: blobs RAM/arquivo + ms por rodada; CPU pool call ms/round; adaptive tier ms/round),
  espera por helper ("ms waiting for it in this request"), `STRATA_TRACE` por etapa, `GET /metrics` (`ram_blobs`,
  `file_blobs`, `file_mb`), `--dump-routing` por camada. A espera exposta por misses (CPU pool + helpers) limita o
  ganho de qualquer política; isso basta para o gate sem NVTX. [revisão-2, F5]
**Degrau (b) — só se (a) mostrar teto acima da noise band:** decomposição por camada com NVTX atrás de flag de
compilação, medida em `--no-token-graph` (o regime padrão é um CUDA graph por token; faixas de host não separam
tempo por camada em replay), com o custo do próprio `--no-token-graph` medido à parte. Overhead em A/B de três
braços: build sem NVTX, NVTX compilado desligado, NVTX ligado. Preservar capture no piloto; medição sem capture é
diagnóstico identificado. [revisão-2, F5; rev-1 #2]

Teto do ganho (corrigido de novo na rodada 4, H2/rev-1 #3): NÃO `--adapt-every 100000` + "perfil perfeito", nem
modelo menor. E **CUDA0 vs CUDA0+helper não é teto** — é "ganho líquido do helper" (configuração finita com suas
sincronizações); renomeado e REMOVIDO do critério eliminatório. Com uma GPU local o contrafactual multi-GPU nem está
disponível: **teto = desconhecido localmente**; sanidade externa pelo oráculo #3265 (3.0x/4.1x). O gate local decide
apenas "há espera de expert que valha atacar", com atribuição demonstrada (pool + adapt wait/join, G1) e controle
repetido; conclusão inconclusiva é resultado válido. [revisão-2 F6; rodada-4 rev-1 #3, H2]

Referência: `strata bench run`, modelo IQ3_S local único pack disponível; contexto local fixado no protocolo
(131072 do config ou teste a 262144); UMA GPU na primeira janela (duas GPUs só após decisão de segunda placa). Baselines em
`~/Devs/strata-bench/results/` (série ryzen9-5090 out/2026) comparáveis só com modelo, quantização, commit, SO,
contexto e configuração fixados. [revisão, A1; revisão-2]

**Custo de janela (A6):** a Fase 0 inteira (4K/32K/128K × 1 e 2 GPUs × repetições) são horas de GPU numa máquina que
tem produção e reserva do strata-bench. Estimar as horas antes de começar e combinar a janela com llm-exec e
llm-bench-exec; se não couber numa janela, dividir a grade (prioridade: 32K em duas GPUs primeiro).

**Gate de continuidade:** só seguir se houver espera mensurável por experts (degrau (a), acima da noise band do
strata-bench) e teto empírico relevante. Se o gargalo for bandwidth de kernel ou o teto for pequeno, o projeto morre
aqui com custo baixo. Essa é a decisão mais barata e mais importante do plano inteiro.

### Fase 1 — Baseline e métricas de qualidade
Rodar `llm-bench` (já existe em ~/Devs/llm-bench) no fork instrumentado para fixar a régua de qualidade: nenhuma
mudança de política pode degradar saída. Critérios de sucesso registrados ANTES de qualquer mudança.
Determinismo — duas fontes separadas, corrigido [revisão-2, F9; rev-1 #6]:
- **Tier adaptativo:** não-bit-exact run-to-run por design; `--adapt-every 100000` fixa residência estática para
  reprodutibilidade de amostragem (`DETAILS.md`; em origin o texto migrou de linha). `--adapt-async` é declarado
  "Not bit-exact run to run".
- **Arredondamento por tamanho de grupo:** `STRATA_IQ_MT_MIN=1` uniformiza os kernels IQ multi-token de CPU (#152);
  NÃO determiniza o tier. Custo medido -1..-3% no decode IQ3_S.
A régua da Fase 1 declara quais das duas ficam fixadas e mede qualidade sob o MESMO regime que a Fase 2 vai usar;
para política dinâmica, tolerância estatística medida por repetição no mesmo regime, não flag única.

### Fase 1.5 — Sweep de configuração existente (nova) [revisão-2, F3]
Antes de qualquer código novo: varredura pequena e previamente delimitada das opções que já existem no commit de
referência — `--adapt-every`, `--adapt-swaps`, `--adapt-decay`, `--adapt-min-gain`, `--adapt-async`,
`stripe` vs `layer` vs `--peer-device` vs helper auto, `STRATA_DISJOINT_ADAPT`, `--pcie-frac`/PcieAuto, secundária
desligada. Grade mais barata primeiro (32K, duas GPUs). O melhor ponto vira o **baseline ajustado** que a Fase 2
precisa bater em prompts separados dos de ajuste; se o sweep fechar a maior parte do teto da Fase 0, a Fase 2 morre
aqui. (O upstream mostra configuração sozinha mover muito: HIP 2×6900 XT, decode 40→66 tok/s por interação
PCIe/helper, `SECOND_GPU.md`.)

### Fase 2 — Política de custo estendida ao multi-GPU (C++ no fork)
Ponto de partida (corrigido na rodada 4, H1): **hipótese própria de seleção por custo marginal no tier de VRAM** —
o seletor atual é contagem decaída; o delta propõe score com custo da miss (PCIe/CPU pool) e ganho amortizado. O
RAM_ADAPT2 (RAM↔SSD) serve só de molde estrutural. Multi-GPU (helper/peer/layer-split) entra como extensão DEPOIS de
validado em uma GPU, não como remoção de guardas.
Escopo mínimo delimitado [rev-1 #5]: selecionar pares de troca no cache já adaptativo, com score de **economia
marginal no caminho crítico** (promover + evictar + amortizar), mantendo donos/capacidades e caches remotos fixos
(o tier remoto é estático por construção). "Secundária desligada" comparada como configuração fixa de baseline.
Migração/desativação dinâmica do tier remoto é OUTRO experimento, condicionado a evidência.

**Gate:** melhoria sustentada e reproduzível, definida ANTES de rodar, com limiares separados por métrica e por
contexto (A4 + [revisão-2, F7; rev-1 #4]):
- média de decode **e** P95 inter-token como duas métricas com limiar próprio — em MoE offloaded o ganho costuma
  aparecer na cauda;
- **TTFT e throughput de prefill por contexto como guardas de não-regressão** (não são variações de nomear a mesma
  coisa: prompt empresta slots do cache, RAM_ADAPT2 usa roteamento do prompt como seed, peer computa linhas do
  prompt, prefill fica na CUDA0), medidos em prompt frio e quente (acerto de checkpoint); ganho de decode não
  compensa automaticamente regressão de TTFT;
- 4K, 32K e 128K avaliados individualmente, sem média agregada entre contextos;
- melhoria acima da noise band run-to-run do strata-bench;
- alvo ≥10–15% em pelo menos uma métrica num contexto real **contra o baseline ajustado da Fase 1.5** (não contra o
  default), sem regressão nas demais além da noise band, sem regressão de qualidade (Fase 1) nem crescimento
  excessivo de memória. Meta de engenharia, não previsão.

### Fase 3 — Extração para Rust (condicional)
Se e somente se a Fase 2 passar o gate e sobreviver à revisão: extrair a decisão pura (dado estado medido →
plano de colocação) para um crate Rust sem estado de GPU; C++ mantém posse de toda a memória usada por operações
CUDA assíncronas — o compilador Rust não prova que uma operação externa terminou de ler um buffer, então o
trait boundary é: Rust recebe métricas, devolve decisão; nunca toca ponteiro. A fronteira é chamada **a cada janela
de adaptação (cada `adapt_every` rodadas), nunca por passo**: por passo o custo da travessia FFI entraria no hot
path e contaria contra o gate. [revisão, A5]

Contribuir como PR ao upstream (Niko1221/Strata) em paralelo; texto público em inglês, sem menção a Claude ou
ferramentas auxiliares, humanizado antes de publicar, só com aval do Breno.

## Riscos principais

- O gargalo pode ser bandwidth puro (Fase 0 degrau (a) mata o projeto — risco desejado, é o barato; o teto empírico
  CUDA0 vs CUDA0+helper quantifica).
- **Duplicação com o upstream — CORRIGIDA (rodada 4, H1):** não há política por custo na admissão VRAM no upstream;
  a RAM_ADAPT2 existente é RAM↔SSD e não cobre este delta. O risco real é outro: PRs adjacentes (#2720 adapt+peer,
  #2964 ADAPT_FETCH) avançam o terreno de sincronização/misses; checar issues/PRs antes de cada fase.
- O upstream pode recusar a interface de biblioteca; sem ela, Rust fica inviável e a política continua C++. Pode
  recusar também a extensão multi-GPU do RAM_ADAPT2 em si, mesmo com o gate passado.
- **Determinismo:** duas fontes separadas (Fase 1); o tier adaptativo é não-bit-exact por design e `--adapt-async`
  declarado não-bit-exact; o upstream preserva ordem de redução em kernels críticos (`bf16_gemv.hpp`). A régua de
  qualidade declara o regime. [revisão, A6; revisão-2, F9]
- **WDDM vs Linux no caminho pinned:** sob WDDM o registro CUDA do arena de host tem teto de 8 GiB
  (`STRATA_ARENA_PIN_GIB`); no Linux o arena inteiro é registrado (`SECOND_GPU.md`). A política pode ganhar num SO e
  perder no outro, e o backend de produção (ryzen9) está no Windows — o gate precisa declarar em qual SO vale. [revisão, A6]
- GPUs assimétricas: a 4090 pode nunca pagar o custo de sincronização por camada (o doc upstream já avisa que
  `layer` placement "may be slower"). A política precisa poder decidir "secundária desligada".
- Contenção de GPU: a Fase 0 inteira consome horas de janela (grade 4K/32K/128K × 1 e 2 GPUs). Estimar e combinar com
  llm-exec/llm-bench-exec antes de começar, não só "sem janela, não roda". [revisão, A6]
- Dois projetos simultâneos na mesma cabeça: este é independente do piloto OpenCode↔gateway do HANDOFF e não deve
  competir com ele por GPU sem janela combinada.
- Licença: MIT dos dois lados; confirmar que o crate Rust herda MIT. Sem risco material. [revisão, A6]

## Fora de escopo (explícito)

- mistral.rs / Candle como base de motor próprio: servem de comparação de scheduling, não de fundação — não
  suportam a família MoE/quantização do Strata.
- vLLM: resolvido para serving multiusuário; o problema aqui é desktop com VRAM insuficiente.
- Autotuning de kernels e otimização automática (direção C da discussão original): adiada; só faz sentido depois
  que a política de colocação estiver validada.
- **Expert parallelism** (cada placa guarda experts diferentes da MESMA camada, explorável pela política por custo):
  anotado como ideia para DEPOIS da Fase 2, fora do plano atual (registro pedido por strata-exec/Breno, 26/10). Não é
  decisão de descarte — é candidata futura; o TP descartado acima não a cobre.

## Registro da revisão (Claude, 2026-10-24) — decisão do exec

| Achado | Veredito | O que mudou no plano |
|---|---|---|
| A1 baselines existentes não citados | Aceito | Fase 0 aponta os results ryzen9-5090 de out/2026 como ponto de partida |
| A2 upstream já tem adaptive tier (contagem) e roadmap ativo | Aceito — corrige a premissa "estática" do rascunho original | Evidências reescritas; lacuna reformulada para "contagem vs. custo medido"; checagem de issues/PRs virou pré-requisito da Fase 0; risco novo listado |
| A3 métricas não separam espera de kernel | Aceito integral | Fase 0 reescrita: decomposição por camada, NVTX/Nsight, banda PCIe medida, contrafactual de teto |
| A4 gate: média + P95 por contexto, contra noise band | Aceito integral | Gate da Fase 2 reescrito com limiares separados |
| A5 ordem C++→Rust certa; falta frequência da FFI | Aceito | Fronteira fixada por janela de adaptação, não por passo |
| A6 riscos: determinismo, WDDM/Linux, custo de janela, upstream recusar evicção, licença | Aceito integral | Novos riscos listados; Fase 1 declara regime de amostragem; Fase 0 estima horas de janela |

Nenhum achado em conflito; nenhum rejeitado. Verificação própria do exec antes de aplicar: o adaptive tier
(`--adapt-every`/`--adapt-swaps`, `generate.cpp:345-369`) e a não-reproduzibilidade (`DETAILS.md:502`) foram
conferidos no código/doc reais — A2 procede.

## Registro da revisão — rodada strata-opencode-2 (rev-1 Codex, rev-2 Claude; 24/10)

Ambos REQUEST_CHANGES. Classificação do exec (nenhum CONFLITO entre os dois):

| Achado | Classe | Veredito do exec | O que mudou |
|---|---|---|---|
| Contrafactual não mede teto (rev-1 #1 / rev-2 F6) | CONFIRMADO | Aceito | Fase 0: teto empírico CUDA0 vs CUDA0+helper; alternativas rebaixadas a diagnóstico |
| Instrumentação vs CUDA graphs (rev-1 #2 / rev-2 F5) | CONFIRMADO | Aceito | Fase 0 em dois degraus; NVTX só opt-in em --no-token-graph com A/B de overhead |
| Baseline ajustado dos controles (rev-1 #3 / rev-2 F3) | CONFIRMADO | Aceito | Nova Fase 1.5 (sweep); gate compara contra baseline ajustado |
| Gate sem TTFT/prefill (rev-1 #4 / rev-2 F7) | CONFIRMADO | Aceito | Guardas de TTFT e prefill por contexto, prompt frio/quente |
| STRATA_IQ_MT_MIN não determiniza o tier (rev-1 #6 / rev-2 F9) | CONFIRMADO | Aceito — erro meu da v1 | Fase 1 separa as duas fontes de não-determinismo |
| Checkout desatualizado; RAM_ADAPT2 já é política por custo (rev-2 F1/F2, P1) | ÚNICO rev-2 | Aceito — conferido pelo exec no git: origin local = 0.1.45.3; condição `!multi_gpu && !peer.valid() && !remote_opt` presente também em v0.2.4.2:14965 | Premissa reescrita: delta = estender RAM_ADAPT2 ao multi-GPU; referência virou upstream mais recente (decisão Breno) |
| Fork errado apontado (rev-2 F4) | ÚNICO rev-2 | Aceito — conferido: strata-bench-core-* é harness, fork do motor é GitHub/strata remote fork | Fase 0 corrigida |
| Fase 2 subestima multi-GPU; tier remoto estático (rev-1 #5) | ÚNICO rev-1 | Aceito — conferido `remote_experts.hpp:15` em 0.1.31 | Escopo mínimo delimitado; migração remota = outro experimento |
| Aspas escapadas no export do HANDOFF (rev-1 #7 / #9 na rodada 3) | ÚNICO rev-1 | **REJEITADO com terceira análise** — scout (consulta strata-opencode-1, 26/10): opção (b), zero bytes `5c 22` no HANDOFF atual E no diff.patch congelado da rodada 3; comando idêntico byte a byte nos dois artefatos (SHA-256 HANDOFF 975f18f7…, patch ccbde115…); o exec reconferiu com contagem binária independente e os hashes batem. A reprodução fictícia do rev-1 demonstra outro texto, não estes arquivos | Comando mantido sem alteração; achado registrado como não sustentado |
| Espaços perdidos no HANDOFF (rev-2 F10) | ÚNICO rev-2 | Aceito | Corrigido no HANDOFF |
| Seções de 24/10 contraditórias sem marcar superação (rev-2 F11) | ÚNICO rev-2 | Aceito | Marcas [resolvido depois]/[superado] no HANDOFF |
| Autor = revisor no cabeçalho do plano (rev-2 F12) | ÚNICO rev-2 | Aceito | Cabeçalho corrigido nesta v2 |

Correções factuais adotadas: `generate.cpp` tem 17.028 linhas em v0.2.4.2 (6.128 na 0.1.31; "~4.300" nunca foi
verdade); contagens de superfície reescritas para v0.2.4.2; citação da lógica do tier movida para generate.cpp:4168-4220 (0.1.31).

**Base decidida (conferência com llm-gateway-exec, 25/10; relatório completo em
`/home/brenoperucchi/Devs/llm-gateway/docs/RESPOSTA-runtime-hybrid-engine-ryzen9-2026-10-25.md`):** o ryzen9 de
produção roda a **engine oficial 0.1.48 CUDA13, commit `6f8800aefe27819ac89545c86f7686c094f099a4`** (componente de
release do Terminal — BUILD.json: source f-023-rel, CUDA 13.0, archs 75/86/89/120; exe SHA256 517ac61f... conferindo
com o manifesto; não é o fork antigo). Runtime ativo: **Qwen3.8-Flash-Next GSQ-RCO IQ3_S, n_ctx=262144, total_slots=1**
(config `iq3_s-128k.json` — nome histórico, o /props de 262144 é o atestado). Só a RTX 5090 (32.607 MiB) aparece no
Windows — nenhuma 4090 detectada, confirmando que o cenário multi-GPU ainda é futuro. Referenciar a engine pelo commit,
não pelo número do app. Resposta completa:
`/home/brenoperucchi/Devs/llm-gateway/docs/RESPOSTA-runtime-hybrid-engine-ryzen9-2026-10-25.md`.

[superado em 25/10 pela decisão "Máquina local" acima: a Fase 0 passou a rodar na estação local com v0.2.5/0.1.50;
a base de produção 0.1.48/6f8800ae fica como referência do regime a confirmar no ryzen9] Decisão derivada anterior:
Fase 0 na base de produção v0.2.3 (engine 0.1.48 = commit 6f8800ae), delta medido contra a base mais atual. Nota
histórica mantida: `0.1.49.x` é o package do Control/DeepSeek, não a engine Qwen.

## Registro da revisão — rodada strata-opencode-4 (rev-1 Codex, rev-2 Claude; 26/10)

Ambos REQUEST_CHANGES. Nenhum CONFLITO entre os dois. Aplicado nesta versão:

| Achado | Classe | Veredito do exec | O que mudou |
|---|---|---|---|
| H1/#1 (P1): RAM_ADAPT2 é política RAM↔SSD, não de VRAM; premissa da Fase 2 e o braço A3 caem; rev-2 retrata o próprio F1/F2 da rodada 2 | CONFIRMADO | Aceito — conferido no código v0.2.5: comentário "the VRAM tier's own rule are untouched" (generate.cpp:14992-14999) e DETAILS.md:172-180 | Premissa reescrita: delta = score de custo no TIER DE VRAM (hipótese própria); RAM_ADAPT2 só molde estrutural; risco de duplicação corrigido; A3 removido do piloto local (com arena inteira em RAM seria no-op) |
| H2/#2 (P1): uma GPU local não mede teto multi-GPU; plano ainda tem trechos antigos (v0.2.3, grade 1-2 GPUs) contradizendo a decisão local | CONFIRMADO | Aceito | Gate local = "há espera que valha atacar"; CUDA0+helper renomeado "ganho líquido do helper", fora do critério eliminatório; teto local declarado desconhecido; grade/prioridade atualizadas |
| H3 (P1): fatores locais empurram a espera para cima (VRAM 24 GB c/ desktop, AVX-2 híbrida, PCIe ~19,9 GB/s); gate assimétrico: negativo mata, positivo exige ryzen9 antes de código | ÚNICO rev-2 | Aceito — conferido o aviso #2065 (438 MiB livres) no test-start do update | Regra do gate assimétrico registrada; regime local fixado (IQ3_S único pack, sem --vision, vram-reserve ≥832, n_ctx a decidir) |
| H4 (P1): riscos da estação (arena page-locked 46,8 GiB + zram 25 G + mesh no mesmo host; voxtype/Chrome na GPU; run-iq3_s.sh sobe na 8080 do gateway) | ÚNICO rev-2 | Aceito — conferido: `ss` mostra llm_gateway na 8080 | Riscos registrados; bench em porta própria (8090); estado da máquina por braço no protocolo |
| H5/#2 (P2): contradições internas (trecho "Fase 0 roda na produção v0.2.3", checagem "ainda não feita", hierarquia de headings) | CONFIRMADO | Aceito | Trecho marcado [superado]; pergunta 1 reescrita; "Onde cada artefato vive" movida antes de "Plano em fases" |
| H6 (P2): sequência certa no esqueleto, mas protocolo precisa ser REAPONTADO inteiro para local (não só corrigido), e a premissa H1 vem antes | ÚNICO rev-2 | Aceito | Ordem adotada: premissa → protocolo local → checagem → degrau (a) 1 GPU sem A3 → ryzen9 curto → segunda placa |
| #3 (P2): gate ainda pode encerrar com teto não demonstrado; ms_wait lido como espera de expert | CONFIRMADO | Aceito | Mesma correção do H2 + G1 já aplicado no protocolo v2 pendente |
| #4 (P2): A3 exige manifesto local de regime (arena inteira vs resident RAM parcial); controle/tratamento no mesmo regime | ÚNICO rev-1 | Aceito — conferido: iq3_s-128k.engine.log mostra arena 46,84 GiB | A3 fora do piloto; manifesto de regime vira pré-condição do protocolo local |
| H7 (P3): run config aponta server/vision para o componente 0.1.48 com engine ativa 0.1.50 | ÚNICO rev-2 | Aceito | Registrar linha "engine started:" por braço; não apagar a 0.1.48 enquanto o config apontar para ela |

Fatos conferidos pelos revisores e pelo exec: engine local 0.1.50 (BUILD.json f-0243-rel 1aff6d14, exe SHA256 ae49d5da…),
v0.2.5 = fa663fab com src/ idêntico a 1aff6d14; Control 0.2.6/0.1.51; aviso #2065 no test-start (438 MiB livres);
pack local único IQ3_S; driver open kernel 610.57 com min 580.65 ok.

Decisões pendentes do Breno (H2/H3/H4): onde/quando entra a segunda GPU; n_ctx local (131072 ou teste 262144); o que
pode ser pausado na estação durante a janela (agents, voxtype).

Nota do exec sobre "scout aspas": é a consulta ao scout sobre a divergência factual da rodada 3 #9 (rev-1 afirma que o
HANDOFF tem `\"` escapadas no export; o grep byte-a-byte do exec não acha a sequência). Por H6, a consulta ao scout
passa a ser feita DEPOIS de reapontar o protocolo, para conferir o texto final.

## O que pedir numa revisão seguinte (se houver)

1. (reescrita na rodada 4) O delta agora é "score de custo no tier de VRAM" (H1). Algum PR aberto do upstream já
   toca esse seletor? (checagem remota feita em 24/10 e a manter a cada fase)
2. O teto empírico CUDA0 vs CUDA0+helper introduz diferença de sincronização por camada que contamina a comparação?
   Precisa de braço com helper sem experts resididos para isolar?
3. A Fase 1.5 (sweep) deve vir antes ou depois do degrau (a) da Fase 0 — o sweep já produz parte dos contadores?
4. Entrega de instrumentação como delta sobre `--stats`/`STRATA_TRACE`: qual formato de linha e quais campos, para
   não colidir com o que o upstream pode adicionar por conta própria? [revisão-2, F8]
