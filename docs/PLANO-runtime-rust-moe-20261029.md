# Plano — runtime MoE em Rust sobre o CUDA Rust da NVIDIA (v0, 29/10)

Decisão do Breno (29/10, depois do fim da Fase 0(a)): posicionar o projeto como runtime de MoE em Rust, usando o CUDA Rust da
NVIDIA (`NVIDIA/cuda-rust`: cuda-oxide, cutile-rs e os crates de host), mesmo embrionário e mais lento que o Strata no início.
Testes na RTX 4090 local (Linux). O código nasce dentro deste repositório.

## Fatos que o plano assume (conferidos em 29/10)

- **Fase 0(a), Strata 0.1.50 na 5090** (RESERVA-FASE0A-20261029-5): a exposição atribuível à colocação de experts é de 6 a 9% da
  janela de decode (limite superior); ~78% é o host esperando a própria GPU (kernel/banda). O cache adaptativo do Strata já vale
  10 a 36% de velocidade (A2 sem adaptação). Ruído A1a×A1b < 0,5 ponto.
- **CUDA Rust**: projeto de pesquisa, API instável. cutile-rs (modelo Tile, Rust estável ≥1.89, CUDA 13.3, sm_80+ incluindo Ada
  sm_89) já está no crates.io; cuda-oxide (SIMT, Rust→PTX, toolchain fixo 1.99.1 + LLVM) é alfa. As regras de posse do Rust valem
  através do lançamento: buffers mutáveis particionados antes do launch, emprestimos mantidos vivos enquanto a GPU trabalha.
- **Já existe LLM em Rust sobre o CUDA Rust**: Grout (Hugging Face; Qwen3-4B/32B densos, kernels cutile-rs, GEMM via cuBLAS,
  versão 0.1.0, sem menção a MoE, offload ou quantização) e mistral.rs (usa kernels cutile-rs; offload de experts não conferido).
  Ser "o primeiro LLM em Rust no CUDA Rust" não é mais possível.
- **Máquina local**: RTX 4090 24 GB (~2,8 GB usados pelo desktop), i7-14700KF, 94 GB RAM, driver 610.57, sem toolkit CUDA (o
  pacote `cuda` 13.3.1 está no repositório), Rust 1.97 padrão, nenhum modelo baixado.

## Posicionamento proposto

O espaço livre é o do Strata: **MoE maior que a VRAM, com experts circulando entre RAM e GPU**. É também onde Rust tem argumento
técnico e não só de linguagem: o risco central desse tipo de motor é a cópia assíncrona de experts (evictar um expert que um
kernel ainda lê; reaproveitar um buffer de staging antes de a cópia terminar). Com o CUDA Rust essa classe de erro pode virar erro
de compilação. Diferencial adicional: a colocação de experts por custo medido (hipótese da Fase 2 do plano original) desde o
começo, em vez de contagem/LRU.

Expectativa honesta: por muito tempo mais lento que o Strata, cujos kernels são ajustados à mão e onde ~78% do tempo já é GPU.

## Marcos (todos na 4090)

- **M0, ambiente**: CUDA 13.3 + um kernel cutile-rs e um cuda-oxide rodando na 4090 (prova do suporte sm_89 na prática).
- **M1, correção ponta a ponta com MoE pequeno** (candidatos: Qwen3-30B-A3B, OLMoE-1B-7B): carregador em Rust, partes densas na
  GPU, experts na RAM, saída comparada token a token com uma referência. Lento é aceitável; errado não.
- **M2, cache de experts na VRAM com prefetch assíncrono** sob as regras de posse; medido contra llama.cpp `--cpu-moe` no mesmo
  modelo e quantização.
- **M3, colocação por custo medido** contra o LRU do M2, com a mesma régua de exposição da Fase 0(a).
- **Depois**: quantizações GGUF (até o IQ3_S do Strata), multi-GPU 4090 + 5090.

O Strata e o llama.cpp são referência e baseline. Os dois são MIT (conferido em 29/10: `LICENSE` de
`/home/brenoperucchi/Devs/GitHub/strata`, commit `9259cad4`, e de `/home/brenoperucchi/Devs/GitHub/llama.cpp`, commit
`0893ec9`), então dá para estudar, portar e adaptar ideias e trechos, mantendo o aviso de copyright em código portado.
Regra de trabalho (Breno, 29/10): antes de desenhar cada marco (B0, C, B completo), olhar como os dois resolvem o mesmo
problema (cache e proteção de slots, prefetch por roteamento, custo CPU contra GPU, kernels MoE agrupados com ids) e
partir do que existe, para copiar, melhorar ou divergir com motivo registrado.

## Perguntas abertas (para a consulta de design)

1. O posicionamento ("runtime MoE com offload, seguro por construção, com colocação por custo") se sustenta frente a Grout,
   mistral.rs, llama.cpp e o próprio Strata? Há um nicho melhor?
2. cutile-rs, cuda-oxide ou os dois? Em que parte do motor cada um entra (GEMV quantizado de experts, atenção, cópias)?
3. Qual modelo para o M1, e qual referência de correção?
4. Que arquitetura mínima do runtime (carregador, grafo de camadas, cache de experts, scheduler de cópias) evita retrabalho no M2/M3?
5. Quais riscos matam o projeto cedo (API do CUDA Rust instável, desempenho de kernels Tile em GEMV quantizado, Windows) e como
   testá-los no M0/M1?

## Consulta de design ask-5 (rev-1 e rev-2, cegos) e decisões do Breno (29/10)

Respostas em `.herdr/ask/strata-opencode-5/`. Convergência: seguir como veículo de pesquisa com tese testável, não como
concorrente do Strata ou do llama.cpp. Premissa central comum (P1, não conferida): o CUDA Rust consegue tornar erro de
compilação evictar um expert que um kernel ou uma cópia em voo ainda usa.

Ajustes adotados:
- Kernels: só cutile-rs no começo; cuda-oxide adiado (só se a dequantização GGUF exigir). GEMM denso via cuBLAS; cópias via
  `cuda-core`/`cuda-async` com memória pinned.
- M1: safetensors BF16, referência `transformers` comparando logits por posição (teacher forcing), não tokens gerados. Modelo
  OLMoE-1B-7B (~14 GB) com orçamento de VRAM de experts forçado; Qwen3-30B-A3B depois e quantizado (em BF16 são ~61 GB e a RAM
  livre medida em 29/10 era 44 GB).
- Arquitetura: ExpertStore (RAM pinned), DeviceCache com slots e estados (livre, carregando, residente, emprestado) e lease preso
  ao evento da GPU; `ensure_resident` já no formato assíncrono; política de colocação como função pura; régua da Fase 0(a) e
  gravação de traços de roteamento desde o M1; forward escrito à mão; todo `unsafe` num crate fino.
- Caminho de CPU para experts frios como componente de primeira classe (rev-2: em batch 1, calcular na CPU pode custar menos que
  copiar para a GPU; medir).
- Sequência: M0 (ambiente, compile-fail de posse, GEMV BF16 vs cuBLAS, sobreposição H2D, baseline llama.cpp `--cpu-moe`), M0.5
  (traços + simulador LRU × custo × oráculo, sem GPU), M0.9 (monitor de console, ver abaixo), M1a (bloco MoE isolado), M1b (modelo completo), M2a (cache síncrono + LRU +
  instrumentação), M2b (assíncrono + prefetch), M3 (custo). Adiados: GGUF/IQ3_S, multi-GPU, Windows/5090.

Decisões do Breno:
- **Monitor de console antes dos testes do runtime (29/10, 11:07).** Antes de começar os testes do runtime (o M1a em diante), o
  projeto precisa de um monitor no próprio terminal, como o Strata Monitor: uma tela de console que se atualiza sozinha, não uma
  TUI interativa. Referência do que mostrar: o Monitor do Strata (`docs/DETAILS.md` do fork, aba Monitor; os dados vêm do
  `GET /metrics`) mostra o que o modelo está fazendo (lendo o prompt, com progresso, ou gerando, a quantos tokens/s), GPU (carga,
  VRAM, temperatura, potência, tráfego PCIe), CPU, RAM e disco, o contexto em uso e as últimas requisições. O nosso acrescenta o
  que é próprio do runtime: cache de experts (slots residentes, acertos, cópias RAM→VRAM em voo, experts calculados na CPU) e a
  régua da Fase 0(a) por janela (exposição atribuível e espera pela GPU). O runtime publica essas métricas num ponto único; o monitor
  só lê. Entra como marco próprio (M0.9), depois do M0 e antes do M1a; a versão do Strata que o Breno usa no ryzen9 deve ser
  conferida antes, para copiar a disposição da tela e não só a lista de campos.
- **Modelos leves nos testes (29/10, 11:08).** Para não sobrecarregar a 4090, os testes que precisam de modelo usam um MoE pequeno,
  não o do Strata. Menor tem prioridade sobre mais quantizado: quantização exige kernels de dequantização cedo, o que a consulta
  recomendou adiar. Candidatos, a conferir antes de baixar: um MoE de ~1B com poucos centenas de milhões de parâmetros ativos (ex.:
  a família Granite MoE da IBM, ~1B/400M ativos) e o OLMoE-1B-7B (~14 GB em BF16), sempre com orçamento de VRAM de experts forçado
  para exercitar o offload. Qwen3-30B-A3B e o modelo que o Strata usa em produção só entram quando o runtime estiver bem formatado.
- Se P1 falhar, o projeto continua como pesquisa (motor Rust + banco de testes de políticas de colocação), com a segurança checada
  em tempo de execução.
- Gate do M3: o simulador precisa mostrar redução de pelo menos 5% na espera por cópia, sobre o LRU, em pelo menos um modelo.

## M0 — resultados

- 29/10: CUDA 13.3.1 instalado (`/opt/cuda`, nvcc 13.3.73); Rust 1.99.1 e 1.98.0 via rustup (o `cargo` do Homebrew vem antes no
  PATH e ignora `rust-toolchain.toml`: usar `/usr/bin/cargo`). `cutile-examples hello_world` compilou e rodou na RTX 4090 (sm_89).
- 29/10, **P1 confirmada no desenho do cache** (`rust/p1-posse/`, cuda-rust 5759cb3, Rust 1.98.0): o cache entrega o slot ao kernel
  cutile como `&Tensor`; evictar e fazer upload exigem `&mut SlotCache`; o upload RAM→VRAM é uma operação própria (`Upload<'a>`)
  com o único `unsafe` do crate. Na 4090: upload, kernel lendo o slot e troca de expert no mesmo slot conferem os valores. Os três
  bugs (evictar com kernel lendo, sobrescrever com kernel lendo, kernel lendo com upload pendente) são recusados pelo compilador
  com E0502 (trybuild). Mutação: com o upload pedindo só `&self`, dois dos três casos passam a compilar e o teste falha.
- Limites encontrados (entram no desenho do M1/M2):
  - A garantia de compilação vale enquanto a operação ou o future existe. Depois do envio, se um future já submetido for
    esquecido, o empréstimo acaba; o cutile então retém e vaza o armazenamento dos argumentos do kernel (segurança em tempo de
    execução, sem comportamento indefinido).
  - O nosso `Upload` não registra o acesso de escrita no armazenamento do slot (o `Storage::retain` do cutile é `pub(crate)`),
    então a detecção de conflito entre streams do cutile não cobre o upload. Mitigação no M1: uploads e kernels na mesma stream.
    Possível contribuição ao upstream: API pública para operações próprias registrarem acesso a um `Tensor`.
  - O ponteiro de um `Tensor` só sai por `device_pointer()`; `cu_deviceptr()` é privado.
  - O spike usa `Vec<f32>` comum como origem, não memória pinned: o `PinnedHostBuffer` do cuda-core pertence à API SIMT
    (`CudaContext`), separada do `Device` do cutile. Resolver isso é parte do item de sobreposição de cópia do M0.
- 29/10, **GEMV BF16 de expert em batch 1: kernel Tile do cutile-rs empata ou supera o cuBLAS** (`rust/m0-gemv/`, evidência em
  `qual-runs/M0-GEMV-20261029/run{1,2}.csv`). `y = W·x` com W e x em BF16, acumulação e saída em f32; kernel Tile simples (cada
  bloco faz BM linhas, percorre K em fatias de BK, converte para f32, multiplica e reduz), sem `mma`; cuBLAS via FFI mínima
  (`cublasGemmEx`, n = 1, compute 32F). Tempo por eventos da GPU em 3000 lançamentos seguidos; os pesos giram entre cópias que
  somam 4× o L2 (72 MB), para medir leitura da VRAM. Melhor Tile da varredura BM ∈ {2..32}, BK ∈ {128..1024}, melhor de 2 rodadas:

  | forma (M×K) | cuBLAS | melhor Tile | Tile/cuBLAS |
  |---|---|---|---|
  | Qwen3-30B gate/up 768×2048 | 5,78 µs (545 GB/s) | 5,53 µs (570 GB/s), BM=4 BK=1024 | 105% |
  | Qwen3-30B down 2048×768 | 6,15 µs (513 GB/s) | 5,43 µs (581 GB/s), BM=8 BK=256 | 113% |
  | OLMoE gate/up 1024×2048 | 7,50 µs (560 GB/s) | 6,73 µs (624 GB/s), BM=2 BK=1024 | 111% |
  | OLMoE down 2048×1024 | 7,03 µs (598 GB/s) | 6,71 µs (627 GB/s), BM=8 BK=256 | 105% |
  | banda 4096×4096 | 42,0 µs (800 GB/s) | 40,4 µs (832 GB/s), BM=16 BK=256 | 104% |

  Erro relativo máximo contra referência em f64 na CPU: ≤1,4e-7 nas duas implementações. Variação entre rodadas ≤ ~4%.
  CPU para enfileirar um lançamento (lote de 256): Tile ~1,3 µs, cuBLAS ~2,5 µs; ambos abaixo do tempo de GPU, então os números
  são do kernel e não do lançamento. O critério de corte da consulta (< 60–70% do cuBLAS) fica longe: o risco "Tile lento em GEMV
  de baixo batch" cai para BF16.
- O que isso não mostra, e o que muda no desenho:
  - Num expert pequeno (3–4 MB) os dois chegam só a ~55–62% dos ~1 TB/s da 4090; na matriz grande, ~82%. O kernel é curto demais
    para encher a GPU. Caminho de melhora: um lançamento por grupo de experts (os top-k de um token, ou gate+up juntos), não um por
    expert. Isso entra no M1a.
  - A configuração ótima muda com a forma (BM/BK), e o kernel é especializado por K: o runtime precisa de uma pequena tabela de
    configuração por forma, medida na própria GPU (o cutile tem `tune.rs`, a conferir).
  - O cuBLAS foi chamado por `GemmEx` com n = 1; não testei `cublasLt` nem outros caminhos de GEMV dele. Pesos quantizados (GGUF)
    continuam sem medição: exigem dequantização dentro do kernel.
  - A GPU dividia a máquina com o compositor (Hyprland, ~25% de SM no `nvidia-smi pmon`); cuBLAS e Tile rodaram intercalados,
    então a comparação vale, mas os valores absolutos podem melhorar com a GPU livre.
- 29/10, **cópia de experts RAM→VRAM e sobreposição com cálculo** (`rust/m0-gemv/src/bin/h2d.rs`, evidência em
  `qual-runs/M0-GEMV-20261029/h2d-run{1,2}.txt`). Expert inteiro (gate + up + down em BF16), 64 cópias seguidas, origem em 16
  experts distintos, link em PCIe 4.0 x16 durante a medição. Memória pinned: o `PinnedHostBuffer` da API SIMT do cuda-core serve às
  streams do cutile, porque o `CudaContext` e o `Device` retêm o mesmo contexto primário (os bits saem idênticos no slot). Ele não
  aceita `bf16` (sem `DeviceCopy`), então o host guarda `u16`. Faixa das 2 rodadas:

  | expert | paginável | pinned | GEMV de uma matriz (residente) |
  |---|---|---|---|
  | Qwen3-30B, 9,4 MB | 553–605 µs (15,6–17,1 GB/s), CPU presa o tempo todo | 467–474 µs (~20 GB/s), CPU ~1 µs | 5,6 µs |
  | OLMoE, 12,6 MB | 750–771 µs (~16,5 GB/s), CPU presa o tempo todo | 637–638 µs (~19,8 GB/s), CPU ~1 µs | 6,7 µs |

  Cópia em uma stream e GEMVs de experts residentes em outra, enfileirados intercalados (uma cópia a cada ~90 GEMVs, para os dois
  lados durarem o mesmo): com origem pinned a sobreposição fica em 88–93% e o GEMV perde 7–11%; com origem paginável cai para
  71–81% e o GEMV perde 39–56%, porque a CPU fica presa na chamada de cópia e para de enfileirar kernels.
- O que isso muda no desenho:
  - O `ExpertStore` tem que ser pinned desde o M1, e o `Upload` do spike P1 precisa aceitar `PinnedHostBuffer` (retendo o buffer)
    em vez de `Vec`. Com origem paginável o prefetch assíncrono do M2b não existe: a CPU fica presa a cópia inteira.
  - Um acerto de cache custa ~17 µs de GPU por expert (3 GEMVs); uma falta custa ~470 µs de cópia, cerca de 28 vezes mais. Isso
    reforça a política de colocação como o centro do projeto e o caminho de CPU para experts frios (a consulta estimou 0,1–0,15 ms
    por expert na CPU em batch 1; medir no M1a).
  - ~20 GB/s está abaixo dos ~25 GB/s úteis que se espera do PCIe 4.0 x16; não investiguei (tipo de alocação pinned, plataforma).
    Também não testei cópias em pedaços menores nem várias cópias simultâneas.
  - As duas metades do cuda-core não se conhecem: o pinned exige um `CudaContext` (API SIMT) separado do `Device` do cutile. Funciona
    porque o contexto primário é o mesmo; vale virar um invólucro nosso no crate fino de `unsafe`.
- 29/10, **modelos dos testes: família Granite 3.1 MoE** (Apache 2.0, BF16, `GraniteMoeForCausalLM`), escolhida pela leveza e
  por ter dois tamanhos com a mesma arquitetura: 1B-A400M (2,5 GB; 24 camadas × 32 experts, 8 ativos; expert ~3,1 MB) para
  desenvolver e 3B-A800M (6,2 GB; 32 × 40, 8 ativos; expert ~4,7 MB) para medir. Em `/home/brenoperucchi/models/` (safetensors) e
  `/home/brenoperucchi/models/gguf/` (GGUF BF16 convertido pelo llama.cpp, mesmos pesos). OLMoE-1B-7B (experts de 12,6 MB, o tamanho
  medido no item de cópia) fica como segunda arquitetura.
- 29/10, **referência de correção**: `tools/ref_logits.py` (ambiente `tools/.venv-ref`: torch 2.15 CPU, transformers 5.22, fora do
  git) roda o modelo em float32 na CPU com teacher forcing e grava tokens, logits por posição e os experts escolhidos por camada.
  Granite 1B sobre `tools/ref-texts/texto-a.txt`: 277 tokens em 1,6 s; acerta o próximo token em 52% das posições; em média 46% dos
  8 experts de um token se repetem no token seguinte (25% ao acaso). Saída em `qual-runs/REF-granite-1b-texto-a/`.
- 29/10, **baseline llama.cpp** (commit 0893ec9, CUDA sm_89, `llama-bench -ngl 99 -t 8 -p 512 -n 128 -r 3`, GGUF BF16; evidência em
  `qual-runs/M0-LLAMACPP-20261029/`). `-ncmoe` = camadas com os experts na CPU (todas = `--cpu-moe`):

  | modelo | experts na GPU | metade na CPU | todos na CPU |
  |---|---|---|---|
  | Granite 1B, decode (tg128) | 567 tok/s | 89 tok/s | 50 tok/s |
  | Granite 1B, prompt (pp512) | 31.780 tok/s | 5.594 tok/s | 3.190 tok/s |
  | Granite 3B, decode (tg128) | 290 tok/s | 46 tok/s | 27 tok/s |
  | Granite 3B, prompt (pp512) | 14.825 tok/s | 2.318 tok/s | 1.301 tok/s |

  Leitura: no 3B com todos os experts na CPU, o decode leva 37 ms por token para 32 × 8 = 256 experts, ~0,14 ms por expert, perto da
  estimativa da consulta. Copiar um expert de 4,7 MB a ~20 GB/s custa ~0,24 ms: em batch 1, uma falta isolada sai mais barata
  calculada na CPU do que copiada, e copiar só compensa se o expert for reaproveitado ou se a cópia ficar escondida atrás do
  cálculo. Bastam metade das camadas na CPU para o decode cair ~6×. É contra esses números (com o mesmo orçamento de VRAM) que o
  runtime se mede no M2.
- **M0 concluído.** Próximo: M0.5 (traços de roteamento + simulador LRU × custo × oráculo, sem GPU), usando `tools/ref_logits.py`
  para gerar os traços com textos mais longos.

## M0.5 — traços e simulador (29/10)

- **Traços**: cada Granite responde a 8 prompts variados (`tools/ref-texts/prompts-traco.json`: código Python/Rust/JS/SQL/bash,
  redes, e-mail e resumo em pt-br) pelo `llama-server` local (GPU, greedy; `tools/gen_trace_texts.py`); a conversa, já no formato de
  chat, passa por `tools/ref_logits.py --experts-only`, que grava os experts escolhidos por camada e token. 1B: 7.347 tokens; 3B:
  7.418 tokens. Em `qual-runs/TRACOS-20261029/<modelo>/conversa-{0..7}/`.
- **Simulador** (`tools/moe_sim.py`, testes de invariantes em `tests/test_moe_sim.py`): decode em batch 1 sobre os traços, com
  orçamento de C slots de expert na VRAM. Custos do M0: cópia = tamanho do expert ÷ 20 GB/s (fila única no PCIe); CPU = custo por
  expert do llama.cpp com `--cpu-moe`, em fila na CPU e em paralelo com a GPU dentro da camada; acerto e atenção dividem o tempo do
  llama.cpp com tudo na GPU. Métrica: espera por token além do ideal "tudo residente". Treino (perfil estático e aquecimento) nas
  conversas 0–3, medida nas 4–7. Calibração contra o llama.cpp medido: 571/567 e 54/50 tok/s (1B), 289/290 e 29/27 tok/s (3B).
- **Resultado no 3B** (espera em ms/token; `qual-runs/TRACOS-20261029/sim-3b.txt`; o 1B tem a mesma forma, em `sim-1b.txt`):

  | orçamento | camadas (-ncmoe) | estática | lru_sync | lfu_cpu | **custo** | oráculo (CPU + admissão) | belady (síncrono) |
  |---|---|---|---|---|---|---|---|
  | 10% (0,6 GB) | 27,6 | 22,2 | 57,7 | 20,8 | **16,9** | 16,2 | 32,3 |
  | 25% (1,5 GB) | 22,9 | 14,6 | 28,9 | 12,0 | **9,8** | 6,8 | 13,9 |
  | 50% (3,0 GB) | 15,2 | 5,6 | 10,2 | 3,7 | **3,2** | 1,3 | 3,4 |
  | 75% (4,5 GB) | 7,6 | 1,4 | 1,8 | 0,61 | **0,56** | 0,16 | 0,54 |

  Políticas: `lru_sync` copia toda falta antes de calcular (o M2a); `lfu_cpu` calcula a falta na CPU e admite em segundo plano por
  frequência com decaimento (estilo contagem, como o cache adaptativo do Strata); `custo` decide, falta a falta, entre copiar ou
  calcular na CPU pelo que termina antes dadas as filas, e espera uma cópia em voo quando é mais barato; `oráculo` e `belady`
  conhecem o futuro (limites).
- **Gate do M3 (≥5% sobre o LRU em 1 modelo): passa com folga nos dois modelos** (custo −58 a −71% contra `lru_sync`; era −60 a
  −71% antes da correção abaixo). A comparação
  mais exigente é contra a melhor política sem custo (`lfu_cpu`): −8 a −19% no 3B e −12 a −20% no 1B. Sensibilidade no 1B (meia-vida
  32/128/512 tokens × fila de admissão 1/4/16, orçamentos 25% e 50%): custo fica 15–21% abaixo de `lfu_cpu` em todas as combinações.
- **Leituras**:
  - O LRU síncrono é a pior escolha com orçamento pequeno: com menos slots que os experts de um token inteiro, o acesso cíclico por
    camada faz o LRU despejar cada expert antes do reuso (0% de acerto a 10%). O M2a deve nascer já com o caminho de CPU.
  - Quase todo o ganho sobre o LRU vem do caminho de CPU para experts frios; o ganho próprio do custo (8–20% sobre contagem) vem de
    usar CPU e PCIe ao mesmo tempo nas faltas de uma mesma camada.
  - Ainda há espaço até o oráculo (25–50%: 9,8 contra 6,8 e 3,2 contra 1,3 ms/token no 3B): prever os experts da próxima camada é o
    próximo passo natural da política.
  - Com orçamento de 50% no 3B, o simulador dá 151 tok/s para `custo`, contra 46 tok/s medidos no llama.cpp com metade das camadas
    na CPU. É simulação, não medição: o M2 precisa confirmar.
- **Correção de 29/10 (revisão 26 do M2a)**: o expert copiado só entrava no heap de despejo quando a cópia assentava, na
  camada seguinte, com a recência dela. Ele empatava com os acertos dessa camada, e o desempate por versão despejava o acerto,
  que era mais recente. Afetava só as políticas cuja prioridade é a recência (`lru_sync`, `lru_cpu`). O runtime do M2a, que
  faz o LRU exato, mostrou a diferença: 48,1% de acerto no `lru_sync` a 25% no 1B contra 44,6% do simulador. Agora o expert
  copiado guarda o passo do pedido; teste `test_despejo_do_lru_leva_o_expert_de_uso_mais_antigo` em `tests/test_moe_sim.py`
  (falhou antes nas duas políticas). Os resultados foram regenerados (`sim-1b.txt`, `sim-3b.txt`; os antigos ficaram como
  `sim-*-v1.*`). Mudou de verdade só o `lru_sync` a 25% no 1B (44,6% → 48,1% de acerto, espera 15,40 → 14,36 ms/token, 58 →
  62 tok/s); o resto mudou na segunda casa decimal, e as tabelas do 3B acima não mudam. `lfu_cpu`, `custo`, `belady` e
  `oráculo` não usam a recência e saíram iguais, então as comparações entre elas também não mudam.
- **Limites**: o custo por expert na CPU é linear e só foi calibrado nos extremos (camadas inteiras); a cópia assume 20 GB/s
  constantes; os traços são do próprio modelo em greedy, 8 conversas seguidas com o cache persistindo entre elas; prefill não
  simulado (todo token é decode); experts do Granite são pequenos (razão cópia/CPU ~1,7–1,8); um modelo com experts maiores
  (OLMoE, Qwen3) muda essa razão.

## M0.9 — monitor de console (29/10)

- Referência: o monitor de console do Strata (captura do Breno, 29/10 13:51: visão Overview ligada ao túnel 18199 com o
  qwen3.8-flash-next-iq3_s 256K). Disposição: cabeçalho com servidor, modelo, contexto, estado e `[LIVE]`; linha de visões
  `1 Overview · 2 Serving · 3 Requests · 4 Engine · 5 Hardware · 6 Logs · 7 Connect`; linha "Model state" (`[Idle]` Reading
  Generating Queued Error) com barra; 4×2 cartões (Speed, GPU load, VRAM com experts em cache, GPU temp, Power, PCIe, CPU, Disk
  read) com número, legenda e barras; embaixo à esquerda as barras Context fill, Experts in VRAM e System RAM; à direita Recent
  requests (TIME STATUS PROMPT REUSED OUTPUT TOK/S DURATION); teclas `1-7 views / filter Space freeze r refresh t theme b open
  in browser ? help q quit`. Decisão do Breno (13:54): incluir também as teclas e as visões, não só a Overview. (Antes eu tinha
  copiado a aba Monitor da interface web, que é outra tela.)
- `rust/moe-monitor/` (ratatui 0.29): biblioteca (`snapshot`: esquema v1 e o `Publisher` que o runtime usará; `sys`: GPU pela
  NVML, CPU/RAM/disco por `/proc`; `ui`: a TUI; `fmt`: números pt-br) e dois binários:
  - `moe-monitor [--metrics <arquivo>] [--interval <s>] [--gpu <n>] [--once [--view <1-7>] [--size <c>x<l>]]`: a TUI com as
    mesmas visões e teclas do Strata, em pt-br (`1-7` visões, `/` filtro em Requisições e Logs, espaço congela, `r` atualiza,
    `t` tema claro/escuro, `b` avisa que ainda não há interface web, `?` ajuda, `q`/Ctrl+C sai). `--once` imprime uma tela em
    texto, sem TUI.
  - `moe-metrics-demo [--metrics <arquivo>] [--seconds <n>]`: métricas inventadas (modelo "demo") num ciclo ocioso → lendo →
    gerando, com eventos e requisições, para ver todas as visões antes de o runtime existir.
- Visões: 1 Visão geral (a do Strata mais uma fileira do runtime: cache de experts, cópias RAM→VRAM, experts na CPU, exposição da
  Fase 0(a)); 2 Serviço (modelo, processo, velocidades, contexto, gráfico); 3 Requisições (tabela inteira, com filtro);
  4 Motor (slots, frações VRAM/cópia/CPU, totais, régua da Fase 0(a), configuração publicada e quatro gráficos); 5 Hardware
  (GPU, CPU, RAM, disco e oito gráficos); 6 Logs (eventos do runtime, com filtro); 7 Conectar (onde e como o runtime publica).
- Identidade (Breno, 14:28: "ficou muito igual"): paleta própria em laranja-ferrugem (tema escuro e claro), no lugar do verde-água
  do Strata. A disposição fica; não há código do Strata (só a disposição, a partir da captura), e ele é MIT de qualquer forma.
  Outras diferenças visuais podem vir depois, sem parar o trabalho por isso.
- Ponto único: o runtime grava `$XDG_RUNTIME_DIR/moe-rt/metrics.json` (temporário + `rename`); o monitor só lê. Cabeçalho mostra
  `[AO VIVO]`, `[PARADO]` (última publicação com mais de 3 s), `[SEM RUNTIME]`, `[ERRO]` ou `[CONGELADO]`. Esquema v1 ganhou, com
  valor padrão para ler snapshots antigos: início do runtime, configuração, eventos, e por requisição tokens reaproveitados e
  duração.
- Testes (`cargo test`, 12): publicação atômica e estados do runtime, `/proc` com textos de exemplo, formatação, todas as visões
  (com e sem ajuda aberta) desenhadas em oito tamanhos de 200×60 a 8×3 sem pânico, conteúdo da visão geral, teclas, congelamento e
  filtro. O de tamanhos achou um desenho fora da tela em terminal de 3 linhas, corrigido. A TUI também foi conferida num
  pseudo-terminal de 140×40: desenha, troca de visão e sai com `q` devolvendo a tela e o cursor.

## M1a — bloco MoE de uma camada (29/10)

- `tools/ref_moe_block.py`: roda o Granite 1B no transformers 5.22 (CPU, float32 a partir dos pesos BF16) sobre `texto-a.txt`
  e captura por hooks a entrada e a saída de `block_sparse_moe` e a decisão do roteador nas camadas 0, 11 e 23 (277 tokens), em
  binário cru, em `qual-runs/REF-moe-block-granite-1b/`. Conferência independente em numpy float64 (camada 11, token 5): top-k
  igual, pesos com diferença ≤ 4,7e-8, saída com erro relativo 2,4e-7. Semântica do `GraniteMoeMoE`: `logits = W_r·x`, top-8 dos
  logits, softmax só sobre os 8; expert `down · (silu(gate) ⊙ up)` com `[gate; up] = input_linear·x`.
- `rust/moe-rt/` (crate novo, sem dependência além do cuda-rust, do `moe-monitor` e do `serde_json`):
  - `weights`: leitor de safetensors próprio (cabeçalho JSON + leitura posicional do tensor pedido, com conferência de dtype,
    forma e offsets).
  - `block`: `HostLayer` (roteador e um par gate_up/down por expert na RAM, a unidade que o cache vai mover), `Routing` (top-k
    com desempate pelo menor índice e softmax) e o bloco inteiro na CPU.
  - `cpu`: GEMV BF16 × f32 em várias threads, SwiGLU, expert.
  - `gpu`: `GpuLayer` com os 32 experts na VRAM e três kernels Tile: GEMV com a ativação em f32, SwiGLU (`g / (1 + exp(-g)) · u`)
    e `y = β·y + g·o`. Por token: o roteador roda na GPU e os logits voltam ao host (a escolha dos experts fica no host porque é
    ele que vai decidir o cache); os 8 experts são enfileirados na mesma stream com `async_on` e só a leitura de `y` sincroniza.
    Os buffers de trabalho são reaproveitados entre tokens.
  - `m1a-bloco --model <dir> --ref <dir> [--layers ...] [--threads N] [--metrics <arquivo>]`: passa os 277 tokens pelos dois
    caminhos, compara com a referência e publica no `moe-monitor` (estado, velocidade, cache, uma requisição por camada e
    caminho, eventos). Sai com código 1 se algum caminho passar da tolerância (erro relativo por token > 1e-4 ou mais de 1% dos
    tokens com top-k diferente).
- Resultado (`qual-runs/M1A-BLOCO-20261029/run1.txt`), 277 tokens por camada, com 8 threads na CPU:

  | camada | caminho | top-k igual | erro rel. máx / médio | Δpeso máx | ms por token (mediana) |
  |---|---|---|---|---|---|
  | 0 | GPU | 277/277 | 8,7e-7 / 3,8e-7 | 8,0e-7 | 0,108 |
  | 0 | CPU | 277/277 | 1,5e-6 / 4,7e-7 | 1,5e-6 | 2,05 |
  | 11 | GPU | 277/277 | 7,4e-7 / 5,0e-7 | 2,4e-7 | 0,108 |
  | 11 | CPU | 277/277 | 9,1e-7 / 6,4e-7 | 2,1e-7 | 2,29 |
  | 23 | GPU | 277/277 | 7,0e-7 / 3,0e-7 | 2,4e-7 | 0,106 |
  | 23 | CPU | 277/277 | 8,2e-7 / 3,7e-7 | 2,7e-7 | 2,22 |

  Leitura da camada do disco 56–104 ms; envio dos 101 MB à VRAM 13–89 ms (`Vec` paginável); compilação dos kernels no primeiro
  token, 361 ms, uma vez por processo.
- Testes (`cargo test`, 7): roteador (ordem, desempate, softmax), GEMV da CPU contra f64 (sobra de colunas, mais threads que
  linhas), SwiGLU, leitor de safetensors (dtype errado, tensor ausente, offsets além do arquivo), camada com formas trocadas, e o
  bloco inteiro numa camada sintética, na CPU e na GPU, contra uma implementação ingênua em f64.
- O que isso não mostra, e o que fica para os próximos marcos:
  - Correção: os dois caminhos batem com o transformers no nível do float32. O M1b herda o bloco como está.
  - GPU: 0,11 ms por camada e token, com duas sincronizações (logits e `y`), uma alocação para `x` e 1 + 4×8 lançamentos.
    Multiplicado por 24 camadas dá ~2,6 ms só de MoE por token, mais que o modelo inteiro no llama.cpp com tudo na GPU
    (1,76 ms/token). Não medi onde vai o tempo. Candidatos para o M1b: `x` residente em vez de alocado, um lançamento por grupo de
    experts (o item que o M0 empurrou para cá e não entrou), grafo CUDA por camada, e só os índices do top-k voltando ao host.
  - CPU: 1,8–2,3 ms para 8 experts, ou 0,22–0,29 ms por expert, contra ~0,095 ms por expert no `--cpu-moe` do llama.cpp no
    mesmo modelo. Com 1, 4, 8 e 16 threads: 3,2 / 1,9 / 1,8 / 3,6 ms. A piora com 16 threads aponta para o custo de criar
    threads a cada GEMV (17 por token), e o laço converte BF16 para f32 sem SIMD explícito; nenhuma das duas causas foi medida.
    O caminho de CPU precisa chegar perto do llama.cpp antes do M2a, senão a política de custo do simulador parte de um custo de
    CPU que o runtime não tem: um conjunto de threads fixo e um GEMV com AVX2.

## M1b — modelo inteiro (29/10)

- `rust/moe-rt/` ganhou o Granite MoE completo, decode em batch 1 com cache KV:
  - `model`: `Config` lida do `config.json` (recusa outro `model_type`, `rope_scaling`, viés na atenção e embeddings não
    amarrados), `HostModel` (todos os pesos na RAM) e `CpuSession`, o passo na CPU. Por camada: `h += 0,22·atenção(rmsnorm(h))`
    e `h += 0,22·moe(rmsnorm(h))`; atenção com 16 cabeças de Q e 8 de KV (d = 64), RoPE `rotate_half` com θ = 1,5e6, escala
    `attention_multiplier` = 1/64; entrada = embedding × 12; saída = embedding transposto ÷ 6. É a referência interna.
  - `gpu_model`: `GpuModel`, o mesmo passo com todos os pesos na VRAM (2,67 GB) e três kernels Tile novos: RMSNorm (o vetor num
    bloco), RoPE (uma cabeça por bloco) e atenção de decode (uma cabeça de Q por bloco, softmax online em fatias de 64 posições,
    máscara depois da posição atual). O cache KV é `[posições, kv·d]` por camada, escrito por cópia device→device. O MoE é o do
    M1a, agora recebendo `x` na VRAM e somando no residual (`GpuLayer::add_into`). A cabeça de saída ganha uma linha de zeros
    porque V = 49155 é ímpar e o GEMV trabalha em pares de linhas.
  - `npy`: leitor de `.npy` para as referências.
  - `m1b-modelo`: teacher forcing nos 277 tokens do `texto-a` contra os logits e os experts gravados por `tools/ref_logits.py`.
  - `m1b-gulosa`: geração gulosa livre (prompt com o chat template, 64 tokens) contra `tools/ref_greedy.py`.
- Resultado do teacher forcing (`qual-runs/M1B-MODELO-20261029/run1.txt`):

  | caminho | argmax igual | erro rel. dos logits máx / médio | \|Δ\| máx | roteamento igual | ms/token (mediana) |
  |---|---|---|---|---|---|
  | GPU | 277/277 | 9,0e-6 / 7,8e-7 | 7,8e-5 | 6648/6648 | 3,57 |
  | CPU (8 threads) | 277/277 | 8,3e-6 / 1,4e-6 | 1,1e-4 | 6648/6648 | 69,6 |

  Roteamento igual = as 24 camadas × 277 posições escolheram os mesmos 8 experts, na mesma ordem.

  Depois da revisão 25 a tolerância caiu de 1e-3 para 1e-4 (o maior erro medido é 9e-6, então 1e-3 deixava passar um erro
  100 vezes maior que o real). Rodada de novo com o caminho de CPU do M2a (`run2.txt`, `run2-cpu16.txt`): GPU igual
  (9,0e-6), 3,89 ms/token de mediana; CPU 6,0e-6, 14,2 ms/token com 8 threads e 14,5 com 16.
- Geração gulosa (`qual-runs/M1B-MODELO-20261029/gulosa-run2.txt`): os 3 primeiros prompts de `prompts-traco.json` (101–143
  tokens de prompt), 64 tokens gerados, iguais aos do transformers nos dois caminhos (64/64 em cada caso). A menor distância entre
  o primeiro e o segundo logit na referência foi 0,027, então nenhum passo dependeu de empate. GPU ~255–265 tok/s gerando, CPU
  ~14–15 tok/s.
- Defeito na referência, achado pela comparação: a primeira `ref_greedy.py` chamava `generate` sem `attention_mask`; com
  `pad_token_id = 0`, o transformers mascarava todo token 0 do prompt, e no Granite o 0 é também o `<|end_of_text|>` que o chat
  template põe entre os turnos. CPU e GPU concordavam entre si e divergiam do `generate` já no passo 0 com folga (0,33); um
  forward direto do transformers no mesmo prompt deu o mesmo token que o Rust. Corrigido com máscara explícita. Teste de
  regressão `tests/test_ref_greedy.py` (modelo Granite minúsculo aleatório, prompt com tokens 0): o primeiro passo do `generate`
  tem de bater com um forward direto, inclusive no gap entre os dois maiores logits. Falhou antes da correção (gap 0,0074 contra
  0,0120) e passa depois. Os traços do M0.5 vieram do llama-server e o `texto-a` não tem chat template, então não foram afetados.
- Testes: `cargo test` agora tem 12 (os 7 do M1a e 5 novos: configuração, RoPE contra a fórmula em f64, RMSNorm, leitor de `.npy`,
  e o modelo inteiro num Granite sintético de 2 camadas e vocabulário ímpar, GPU contra CPU por 70 posições, cruzando a fatia de
  64 da atenção, mais o `reset`). Python: 467 passam e 2 são pulados no Python do sistema; no venv de referência, 16 passam.
- O que isso não mostra, e o que fica para os próximos marcos:
  - Correção: o modelo em Rust bate com o transformers em float32 nos dois caminhos. É a base do M2.
  - Velocidade: a GPU dá ~280 tok/s no teacher forcing, cerca de metade dos 567 tok/s do llama.cpp com tudo na GPU. Cada token
    tem 24 sincronizações (os logits de cada roteador voltam ao host), ~1 150 lançamentos (48 por camada) e duas alocações. Não medi a divisão do
    tempo. A CPU dá ~14 tok/s, contra 50 tok/s do llama.cpp todo na CPU, pelo mesmo motivo do M1a.
  - O primeiro prompt na GPU leva ~6 s por causa da compilação dos kernels (JIT); um processo de serviço compila uma vez.
  - Prefill é token a token (o mesmo passo do decode); um prefill em lote fica para quando a velocidade importar.

## M2a — caminho de CPU rápido e cache de experts com orçamento (29/10)

- Caminho de CPU (`rust/moe-rt/src/pool.rs`, `cpu.rs`): conjunto fixo de threads (espera ativa de ~100 µs entre despachos,
  depois `Condvar`; tarefas pegas por contador atômico, o que equilibra núcleos P e E; pânico de tarefa propagado sem deixar
  thread presa) e produto escalar BF16 × f32 com AVX2 + FMA (BF16 → f32 por deslocamento de 16 bits; escalar se a CPU não
  tiver). Os k experts de uma camada vão em dois despachos (todas as linhas de gate_up, depois todas as de down já somando
  em `y`), contra 16 criações de threads por camada no M1a.
  - Bloco MoE de uma camada, 8 experts, camada 11: 2,3 → 0,21 ms com 8 threads (1 / 2 / 4 / 8 / 12 / 16 threads: 1,13 / 0,63 /
    0,37 / 0,21 / 0,23 / 0,16 ms; `qual-runs/M2A-CACHE-20261029/bloco-cpu-threads.txt`, mediana de uma rodada por contagem; uma
    rodada anterior, sem arquivo, deu 0,23 / 0,20 / 0,21 ms com 8 / 12 / 16, então a diferença entre rodadas chega a 0,05 ms e a
    ordem entre 8, 12 e 16 threads não é confiável). Esse número repete a mesma camada (100 MB), então parte dos pesos fica no L3 e a vazão
    aparente passa da RAM; não serve de custo de expert.
  - Modelo inteiro na CPU, teacher forcing do `texto-a`: 70 → 14,2 ms/token com 8 threads (70 tok/s), 14,5 ms com 16
    (`qual-runs/M1B-MODELO-20261029/run2.txt` e `run2-cpu16.txt`; uma rodada anterior, sem arquivo, deu 16,3 e 14,6), acima dos
    50 tok/s do llama.cpp todo na CPU, com argmax 277/277 e roteamento 6648/6648 ainda iguais ao transformers.
  - Tempo de CPU por expert dentro do runtime (`lru_cpu` a 10%, quase tudo na CPU): 11,4 ms por token para ~183 experts, ~0,062
    ms por expert em `run2.txt`; 9,6 ms para ~175, ~0,055 ms, em `run3.txt`. Não é a mesma medida dos 0,095 ms que o simulador usa: aqui é tempo de parede com 8 threads
    e os experts de uma camada juntos em dois despachos; no simulador é o custo marginal de um expert tirado do llama.cpp.
- Cache (`rust/moe-rt/src/cache.rs`): slots globais na VRAM endereçados por (camada, expert), um buffer pinned por expert na RAM
  (gate_up + down) que serve às cópias e ao caminho de CPU, despejo LRU que nunca escolhe slot com expert pedido na camada atual
  nem com cópia em andamento. Os buffers pinned não são a única cópia: o `HostModel` de onde saem continua vivo nos binários,
  então os experts ocupam a RAM duas vezes (2 × 2,4 GB no 1B). `lru_sync`: a falta é copiada na stream de cálculo, antes do cálculo. `lru_cpu`: a falta vai para a CPU,
  em paralelo com os experts da GPU (que ficam enfileirados), e o expert é admitido por uma segunda stream que primeiro espera
  um evento gravado no cálculo; o slot só vale depois do evento da cópia, e com 4 cópias na fila a admissão é descartada
  (como o `BACKLOG` do simulador). O `GpuModel` escolhe entre experts residentes (M1b) e o cache (`upload_cached`).
- A ausência de corrida entre cópias e cálculo vem do protocolo, não dos tipos (o Rust não vê as streams): a stream de cálculo
  está parada quando as vítimas são escolhidas (leitura dos logits do roteador), os experts pedidos na camada não podem ser
  despejados, a admissão em segundo plano espera um evento gravado no cálculo, e um slot carregando só é lido depois que o
  evento da cópia termina. Uma mudança nesse protocolo precisa de revisão com isso em mente; os testes não pegam corrida.
- `m2a-cache`: para cada política e orçamento, confere os logits do `texto-a` contra o transformers, esvazia o cache e passa as
  conversas 4–7 dos traços do M0.5 (3.451 tokens, as mesmas do teste do simulador; o cache persiste entre conversas), conferindo
  o roteamento contra o traço e os logits de cada token contra os da passada residente, que roda primeiro. `cache_full` é o
  caminho do cache com slots para todos os experts, preenchido antes das conversas (só acertos). Resultado depois das correções
  das revisões 25 e 26 (`qual-runs/M2A-CACHE-20261029/run3.txt`, simulador corrigido), 8 threads na CPU:

  | política | orçamento | ms/token média / mediana | tok/s | acerto (sim) | CPU | cópias/token | ms/token previstos pelo sim |
  |---|---|---|---|---|---|---|---|
  | residente | 100% | 4,37 / 4,36 | 229 | 100% | – | – | 1,76 (llama.cpp) |
  | cache_full | 100% | 4,36 / 4,34 | 230 | 100% | – | – | – |
  | lru_sync | 10% | 35,11 / 35,01 | 28 | 0% (0%) | – | 192,0 | 30,61 |
  | lru_sync | 25% | 20,26 / 20,65 | 49 | 48,1% (48,1%) | – | 99,6 | 16,11 |
  | lru_sync | 50% | 13,02 / 11,64 | 77 | 71,6% (71,7%) | – | 54,5 | 9,19 |
  | lru_sync | 75% | 6,25 / 5,86 | 160 | 93,8% (93,9%) | – | 11,9 | 3,14 |
  | lru_cpu | 10% | 13,53 / 13,42 | 74 | 8,8% (0%) | 91,2% | 60,6 em 2º plano | 18,46 |
  | lru_cpu | 25% | 10,97 / 10,67 | 91 | 41,7% (39,9%) | 58,3% | 52,2 em 2º plano | 11,20 |
  | lru_cpu | 50% | 8,05 / 7,71 | 124 | 73,9% (74,7%) | 26,1% | 35,9 em 2º plano | 5,10 |
  | lru_cpu | 75% | 5,40 / 5,22 | 185 | 93,8% (93,9%) | 6,2% | 11,0 em 2º plano | 2,34 |

  Em todas as 10 configurações: logits do `texto-a` com erro relativo ≤ 9,0e-6 contra o transformers (limite 1e-4, antes
  1e-3), logits das conversas iguais aos da passada residente (bit a bit no `lru_sync` e no `cache_full`, ≤ 7,7e-6 no `lru_cpu`,
  onde parte dos experts vai para a CPU) e argmax igual em todos os tokens. Roteamento igual ao traço em 82.824/82.824 (24
  camadas × 3.451 tokens), menos uma camada de um token no `lru_cpu` a 10% (82.823): os logits desse caminho ficaram a ≤ 5,2e-6
  dos residentes, então a hipótese é um quase empate no 8º expert do roteador, virado pela ordem de soma diferente entre CPU e
  GPU; o binário não grava a distância entre o 8º e o 9º logit, então não está conferido. A geração gulosa passou pelo cache nas
  duas políticas: `lru_cpu` a 10% e `lru_sync` a 25%, os 3 casos 64/64 iguais ao transformers
  (`qual-runs/M1B-MODELO-20261029/gulosa-run3.txt` e `gulosa-run4-lru_sync.txt`).
- Variação entre rodadas: o `lru_cpu` muda bastante de uma rodada para outra, porque a admissão em segundo plano depende do tempo
  (`run2.txt` contra `run3.txt`: a 10%, 16,03 → 13,53 ms e 4,5% → 8,8% de acerto; a 25%, 13,84 → 10,97 ms). O residente variou
  4,64 → 4,37 ms e o `lru_sync` ~0,5 ms. Uma rodada por configuração serve para a ordem entre políticas, não para diferenças de
  10–20% no `lru_cpu`; comparar políticas finas (M3) vai precisar de várias rodadas.
- Diferenças para a rodada 1 (`run1.txt`, antes das revisões): `lru_sync` a 25% foi de 39,2% para 48,1% de acerto e de 23,7 para
  20,3 ms/token. Os 5,4 pontos abaixo do simulador que a rodada 1 deixou "não investigados" eram um defeito no runtime: o despejo
  podia escolher um expert que já estava no cache e seria pedido mais adiante na mesma camada (só os já visitados estavam
  protegidos). O teste `cache_lru_sync_nao_despeja_acerto_da_mesma_camada_e_segue_o_lru_de_referencia` reproduziu (93/99 contra
  94/98 acertos e cópias da referência) e passa depois da correção. Corrigido o runtime, ele passou a ficar 3,5 pontos acima do
  simulador; essa diferença era um defeito no simulador (seção M0.5, "Correção de 29/10"), e com ele corrigido os dois dão
  48,1%. Um replay em Python do LRU do runtime sobre o traço dá os mesmos 48,1%, com e sem aquecimento nas conversas de treino e
  com os dois desempates.
- Leitura:
  - Com os dois corrigidos, o acerto de cada política fica a até ~2 pontos do previsto, menos `lru_cpu` a 10% (8,8% contra 0%;
    4,5% na rodada anterior). Hipótese, não medida: o simulador admite em segundo plano menos experts do que o runtime nesse
    orçamento (a fila de cópias dele enche antes). A ordem das políticas se confirma: `lru_cpu` é mais rápido que `lru_sync` em
    todo orçamento (2,6× a 10%, 1,8× a 25%, 1,6× a 50%, 1,2× a 75%).
  - Tempo acima do residente, runtime contra simulador (espera prevista): `lru_sync` 30,7 / 15,9 / 8,7 / 1,9 ms contra 28,9 /
    14,4 / 7,4 / 1,4; `lru_cpu` 9,2 / 6,6 / 3,7 / 1,0 contra 16,7 / 9,4 / 3,4 / 0,6. A cópia síncrona sai a ~0,16 ms por expert
    (~19 GB/s, o mesmo do M0). No `lru_cpu`, o runtime fica abaixo do simulador a 10% e 25% (CPU mais barata que a prevista) e
    acima a 50% e 75%. Hipótese, não medida: um custo fixo por camada com expert na CPU (ler `x`, enviar a soma, um despacho a
    mais) que o simulador não modela.
  - O custo fixo é o mesmo nos dois caminhos: `cache_full` (4,36 ms) empata com o residente (4,37), então o caminho do cache não
    acrescenta nada quando tudo acerta. Os dois ficam bem acima dos 1,76 ms do llama.cpp, e nos orçamentos grandes esse custo
    domina. Hipótese, não medida: as 24 sincronizações por token (logits do roteador ao host) e os ~1 150 lançamentos. Falta
    medir a divisão.
  - O tempo por token cresce com o contexto, agora medido (`run3.txt`, mediana por faixa de posição, residente): 3,59 ms nas
    posições 0–256, 4,24 em 256–512, 4,89 em 512–1024, 5,65 acima de 1024 (48 tokens). Isso explica os 3,57 / 3,89 ms do M1b
    (`texto-a`, 277 posições) contra a média de ~4,4–4,6 ms nas conversas. A causa provável é a atenção de decode, que lê o cache
    KV inteiro a cada token, mas a divisão não foi medida.
  - O monitor recebe slots residentes e chegando, frações VRAM/cópia/CPU e totais (visão Motor). Ainda não publico a vazão das
    cópias (`h2d_gbs` fica 0).
- Testes na entrega do M2a: `cargo test`, 17 (5 novos: produto escalar AVX2 contra f64 com e sem sobra de 32, pool em 2.000 despachos seguidos,
  pânico de tarefa, e o cache nas duas políticas num modelo sintético contra a CPU, incluindo orçamento menor que k e zero slots,
  com os contadores somando os experts pedidos).
- Revisão 25 (rev-1 e rev-2, cegos; nenhum conflito). Correções; cada defeito de comportamento ganhou um teste que falhou antes
  (os itens de `Drop` e sincronização não têm teste que falhe sem eles; ver revisão 26):
  - despejo de acerto não visitado da mesma camada (acima);
  - `ExpertCache`, `GpuLayer` e `GpuModel` ganharam `Drop` que sincroniza as streams (o cutile solta a memória numa stream de
    desalocação que não espera as nossas); depois de um erro, as streams são sincronizadas e o objeto devolve erro nas chamadas
    seguintes, em vez de pânico por scratch ausente; `copy_into` tira o expert antigo do mapa antes das cópias;
  - dimensões usadas como tile (H, I, qd, head_dim, head_dim/2) agora têm de ser potência de dois no upload, com mensagem clara.
    O Granite 3B (H = 1536) passava pelo upload e só falharia na compilação JIT. Testes com H = 1536 e com head_dim = 42;
  - pool: um pânico de quem chama e de uma thread no mesmo despacho deixava o despacho seguinte em pânico (teste falhou antes);
    tarefa ausente vira pânico de tarefa em vez de matar a thread; não reentrância documentada;
  - leitores de safetensors e `.npy`: data_offsets com tamanho errado, não numéricos ou que estouram u64, forma cujo tamanho
    estoura usize e cabeçalho truncado davam pânico (testes falharam antes) e agora dão erro;
  - testes de GPU pulam com aviso sem GPU e falham com `MOE_RT_REQUIRE_GPU=1`; contadores do cache conferidos contra um LRU de
    referência; testes novos de descarte por backlog e de `fill` (cache cheio só com acertos);
  - `tools/ref_greedy.py`: conferido que `eos_token_id=None` desliga a parada no EOS no transformers 5.22.0 (sem ele, o
    `generate` para no primeiro token quando o EOS é o maior logit); o teste novo segura isso numa atualização.
  - `cargo test`: 25; Python: 484 passam.
- Revisão 26 (verificação das correções; rev-1 e rev-2 aprovaram, nenhum conflito, só P3):
  - Nos caminhos de erro, a sincronização acontecia depois de os tensores locais caírem (os dois revisores). Agora cada `?`
    depois de enfileirar trabalho passa por `sync_on_err`, que espera a stream antes de o erro sair da função. Conferido no
    cuda-rust: `async_on` não segura os recursos depois de enfileirar, e `DeviceBuffer::drop` libera com `cuMemFreeAsync` numa
    stream de desalocação própria (`cuda-async/src/device_buffer.rs:128`).
  - NaN nos logits não reprovava (`f64::max` descarta NaN, e um -NaN fora do argmax não muda o argmax), nos três binários;
    testes unitários em `m2a-cache` e `m1b-modelo` falharam antes.
  - O simulador tinha o defeito de recência descrito no M0.5; corrigido com teste, resultados regenerados.
  - Testes novos: passo com token fora do vocabulário e com cache KV cheio devolve erro e o modelo segue utilizável; cache
    `lru_cpu` solto logo depois de admissões em segundo plano. A suíte de modelo passou sob `compute-sanitizer --tool memcheck`
    com 0 erros (14 testes), o que inclui esse último. Não há teste de corrida entre as streams (os experts dos testes são
    pequenos demais para a cópia ficar em voo por muito tempo).
  - Texto do plano conferido contra as saídas; mediana por posição no contexto acrescentada ao `m2a-cache`.
  - `cargo test`: 29; Python: 486 passam.
- Pendências registradas pela revisão 26:
  - Granite 3B: resolvido logo depois (seção seguinte), por decisão do Breno de adaptar os kernels antes de medir o custo fixo.
  - O protocolo sem corrida do cache depende de a stream de cálculo estar parada quando as vítimas são escolhidas, o que a
    leitura dos logits do roteador garante hoje. Tirar essa sincronização (o passo 1 abaixo) quebra a premissa: antes disso, o
    protocolo tem de ser refeito e testado com experts grandes sob `compute-sanitizer`.
- Próximos passos que isso sugere: (1) medir a divisão do custo fixo por token e cortá-lo antes de comparar políticas mais
  finas, porque ele já é maior que a diferença entre `lru_cpu` e o simulador nos orçamentos grandes; o candidato principal é não
  voltar os logits do roteador ao host em toda camada (top-k na GPU, ou só os índices); (2) M2b, admissão por previsão (prefetch) em vez de só
  depois da falta; (3) M3, a política `custo` do simulador, que lá ficou 15–21% abaixo de `lfu_cpu`.

## Kernels em fatias e Granite 3B (29/10)

- O cuTile só aceita tiles em potência de dois, e o GEMV e o RMSNorm carregavam o vetor de entrada inteiro num tile. O 3B tem
  H = 1536 e qd = 24·64 = 1536, então não rodava. Mudanças em `rust/moe-rt/src/gpu.rs`:
  - GEMV: percorre K em fatias de `tile_k(K)`, a maior potência de dois que divide K, até 1024 (1536 → 512, 768 → 256). No 1B
    nada muda (1024 e 512 já eram uma fatia só).
  - RMSNorm: em fatias de `tile_k(H)`, com H / fatia blocos; cada bloco soma os quadrados do vetor inteiro e escreve a própria
    fatia.
  - O upload só exige potência de dois em `head_dim` e `head_dim/2` (RoPE e atenção ainda carregam a cabeça inteira).
  - Testes: bloco na GPU contra f64 e modelo inteiro na GPU (residente, `lru_sync`, `lru_cpu`) contra a CPU com H = 768 (12
    cabeças de Q, 4 de KV), que falharam antes; teste do `tile_k`. `cargo test`: 31.
- Referências do 3B gravadas como as do 1B: `qual-runs/REF-granite-3b-texto-a/` (`tools/ref_logits.py`) e
  `qual-runs/REF-greedy-granite-3b/greedy.json` (`tools/ref_greedy.py`, 3 prompts, 64 tokens).
- 1B depois da mudança (`qual-runs/M1B-MODELO-20261029/run3-tiles.txt`): GPU com os mesmos 9,03e-6 de erro e roteamento
  6648/6648; 4,01 ms/token de mediana (3,57 / 3,89 / 4,01 nas três rodadas, dentro da variação já vista).
- 3B, teacher forcing do `texto-a` (`qual-runs/M1B-MODELO-3B-20261029/run1.txt`):

  | caminho | argmax igual | erro rel. dos logits máx / médio | \|Δ\| máx | roteamento igual | ms/token (mediana) |
  |---|---|---|---|---|---|
  | GPU | 277/277 | 2,7e-6 / 7,8e-7 | 3,8e-5 | 8864/8864 | 6,01 |
  | CPU (8 threads) | 277/277 | 6,1e-6 / 1,3e-6 | 8,8e-5 | 8864/8864 | 33,7 |

  A CPU dá ~30 tok/s, acima dos 27–29 tok/s do llama.cpp todo na CPU no 3B.
- 3B, geração gulosa (`gulosa-run1.txt`): os 3 casos 64/64 iguais ao transformers nos três caminhos, GPU (~165–174 tok/s), CPU
  (~28–33 tok/s) e cache `lru_cpu` a 25% (~45–49 tok/s).
- 3B, cache nas conversas 4–7 (3.814 tokens; `qual-runs/M2A-CACHE-3B-20261029/run1.txt`, simulador corrigido), 8 threads:

  | política | orçamento | ms/token média / mediana | tok/s | acerto (sim) | CPU | cópias/token | ms/token previstos pelo sim |
  |---|---|---|---|---|---|---|---|
  | residente | 100% | 7,06 / 6,95 | 142 | 100% | – | – | 3,45 (llama.cpp) |
  | cache_full | 100% | 7,05 / 7,00 | 142 | 100% | – | – | – |
  | lru_sync | 10% | 68,68 / 68,48 | 15 | 0% (0%) | – | 256,0 | 61,17 |
  | lru_sync | 25% | 38,86 / 38,47 | 26 | 47,7% (48%) | – | 133,9 | 32,40 |
  | lru_sync | 50% | 18,91 / 17,45 | 53 | 79,7% (80%) | – | 52,0 | 13,66 |
  | lru_sync | 75% | 9,10 / 8,52 | 110 | 95,8% (96%) | – | 10,7 | 5,25 |
  | lru_cpu | 10% | 23,15 / 22,78 | 43 | 20,3% (0%) | 79,7% | 71,3 em 2º plano | 33,87 |
  | lru_cpu | 25% | 17,98 / 17,65 | 56 | 49,4% (49%) | 50,6% | 57,5 em 2º plano | 17,48 |
  | lru_cpu | 50% | 11,97 / 11,47 | 84 | 80,2% (81%) | 19,8% | 36,5 em 2º plano | 7,68 |
  | lru_cpu | 75% | 7,72 / 7,50 | 130 | 95,7% (96%) | 4,3% | 10,2 em 2º plano | 4,05 |

  Logits do `texto-a` ≤ 3,6e-6 contra o transformers e os das conversas iguais aos residentes (bit a bit no `lru_sync` e no
  `cache_full`, ≤ 6,9e-6 no `lru_cpu`), argmax igual em todos os tokens. Roteamento igual ao traço em 122.047/122.048 (32
  camadas × 3.814 tokens) já na passada residente: a diferença está entre o runtime e o traço do transformers, não no cache, e
  a hipótese é um quase empate no 8º expert (não conferido; os logits ficam dentro da tolerância). O `lru_cpu` varia de 122.046
  a 122.048 conforme o que foi para a CPU.
- Leitura:
  - O 3B confirma o 1B: acerto a até ~1 ponto do simulador, menos `lru_cpu` a 10% (20,3% contra 0%, maior que no 1B), e
    `lru_cpu` mais rápido que `lru_sync` em todo orçamento (3,0× a 10%, 2,2× a 25%, 1,6× a 50%, 1,2× a 75%).
  - O custo fixo também se repete: `cache_full` empata com o residente (7,05 contra 7,06 ms), ~2,0× os 3,45 ms do llama.cpp
    (no 1B, ~2,5×). Por camada, na faixa de contexto curto (posições 0–256): 0,15 ms no 1B (3,59 ms / 24) e 0,18 ms no 3B (5,84
    ms / 32). O custo fixo cresce pouco com o tamanho das camadas, o que é compatível com sincronização e lançamentos
    dominarem, mas é a medição do passo seguinte que vai dizer.
  - Nos orçamentos grandes do `lru_cpu`, o runtime fica bem acima do simulador (50%: 11,97 contra 7,68; 75%: 7,72 contra 4,05),
    e a diferença tem a mesma ordem da diferença entre o residente e o llama.cpp. É o mesmo custo fixo.


## M2b — de onde vem o custo fixo (29/10)

- Passo 1 do corte do custo fixo: medir antes de mexer. `rust/moe-rt/src/bin/m2b-custo-fixo.rs` passa os 277 tokens do
  `texto-a` com todos os experts residentes, 3 rodadas por variante, em variantes do passo (`GpuModel::step_probe`, com
  `Probe { routes, skip_attn, skip_moe }`):
  - `normal`: o passo de sempre, com 2 sincronizações por camada para ler os logits do roteador e mais 3 por token;
  - `sem_sync`: o roteamento gravado na passada de aquecimento entra pronto, sem GEMV do roteador, sem leitura por camada e
    sem top-k no host;
  - `so_atencao` e `so_moe`: só metade da camada, com o roteamento gravado (com `skip_attn`, o MoE lê um `xn` velho; só serve
    para tempo);
  - `base`: nem atenção nem MoE (envio do embedding e do cos/sin, RMSNorm final, cabeça e leitura dos logits).
- Tempos por token: total (parede), host (parede até o último lançamento voltar) e GPU (eventos no começo do passo e antes da
  leitura dos logits; inclui a GPU parada à espera do host e os intervalos entre kernels). Operações e sincronizações são
  contadas no código do passo (`StepTimes::count`), com teste: por camada, 12 de atenção, 4 por expert, o AXPBY residual e,
  sem rotas dadas, 2 do roteador; mais 5 por token. Dá 1.133 operações e 51 sincronizações no 1B, 1.509 e 67 no 3B. (A
  primeira versão usava uma fórmula escrita à mão com 13 operações de atenção, uma a mais que o código; a rodada 27 achou.)
- Rodadas: `run1-*` deu GPU = 0 no `normal` (eventos só eram criados com `Probe` não padrão; corrigido com o parâmetro
  `timing` de `step_with`). `run2-*` e `run3-*` são válidos; `run3-*` tem a contagem medida e `run3-*-longo` passa o `texto-a`
  4 vezes seguidas (1.108 posições). Tudo em `qual-runs/M2B-CUSTO-FIXO-20261029/`.
- Medianas, ms (total / host / GPU), posições 0–277:

  | variante | 1B run2 | 1B run3 | 3B run2 | 3B run3 |
  |---|---|---|---|---|
  | normal | 3,557 / 3,398 / 3,511 | 3,940 / 3,769 / 3,892 | 5,421 / 5,198 / 5,376 | 5,768 / 5,534 / 5,723 |
  | sem_sync | 3,277 / 2,444 / 3,231 | 3,521 / 2,634 / 3,472 | 4,947 / 3,268 / 4,901 | 5,228 / 3,436 / 5,179 |
  | so_atencao | 1,274 / 0,421 / 1,232 | 1,309 / 0,440 / 1,265 | 1,877 / 0,551 / 1,836 | 1,896 / 0,566 / 1,850 |
  | so_moe | 2,180 / 1,984 / 2,135 | 2,395 / 2,161 / 2,351 | 3,291 / 2,645 / 3,245 | 3,540 / 2,890 / 3,495 |
  | base | 0,163 / 0,015 / 0,124 | 0,163 / 0,015 / 0,124 | 0,215 / 0,015 / 0,177 | 0,216 / 0,015 / 0,177 |

  Os logits do `sem_sync` saem iguais aos do `normal` bit a bit nos dois modelos e nas duas rodadas (o binário agora exige
  igualdade exata). As variantes parciais somam o todo (1B run2: 1,232 + 2,135 − 0,124 = 3,243 contra 3,231 do `sem_sync`).
  Entre run2 e run3 (processos diferentes, mesmo binário de kernels) o `normal` variou 6–11%, a mesma ordem da deriva entre
  processos já vista no M2a; o `base` não variou.
- GPU por faixa de posição, `--repeat 4` (ms, medianas):

  | variante | 1B 0–256 | 256–512 | 512–1024 | 1024–1108 | 3B 0–256 | 256–512 | 512–1024 | 1024–1108 |
  |---|---|---|---|---|---|---|---|---|
  | normal | 3,848 | 4,535 | 5,432 | 6,100 | 5,375 | 6,117 | 7,258 | 8,061 |
  | sem_sync | 3,399 | 4,010 | 5,011 | 5,688 | 4,908 | 5,657 | 6,788 | 7,566 |
  | so_atencao | 1,246 | 1,839 | 2,807 | 3,530 | 1,835 | 2,591 | 3,724 | 4,475 |
  | so_moe | 2,345 | 2,350 | 2,361 | 2,355 | 3,253 | 3,252 | 3,253 | 3,252 |
  | base | 0,124 | 0,124 | 0,124 | 0,124 | 0,177 | 0,177 | 0,177 | 0,177 |

- Piso de banda por token (bytes que o passo precisa ler, bf16, a ~900 GB/s): atenção 0,17 ms (1B) e 0,45 ms (3B); os k
  experts 0,67 e 1,34 ms; com a cabeça, 0,95 e 1,96 ms. O KV em f32 soma ~98 KB por posição no 1B (24 camadas × 2 × 512 × 4
  B), ~0,11 µs por posição.
- Leitura:
  - As sincronizações não são o custo principal. O `sem_sync` economiza 0,28–0,42 ms no 1B (8–11%) e 0,47–0,54 ms no 3B
    (9%), e isso é um teto para elas: a variante também tira o GEMV do roteador (24 ou 32 lançamentos, ~0,07 ms) e o top-k no
    host. O teto vale para o passo de hoje; com o passo em ~1,5–2 ms depois dos cortes, as mesmas esperas passariam a 14–19%.
  - O MoE não depende da posição (2,35 ms no 1B em todas as faixas) e a atenção cresce linearmente com ela: 1,25 → 3,53 ms no
    1B e 1,84 → 4,48 ms no 3B entre as faixas 0–256 e 1024–1108, ~2,4 µs por posição por token, ~20× o piso do KV. A partir de
    ~1.000 posições a atenção passa o MoE no 1B. A causa está no kernel: `attn_decode` roda um bloco por cabeça de Q (16 no 1B,
    24 no 3B, numa placa de 128 SMs) e cada bloco percorre o KV inteiro em série.
  - Em posições curtas, a GPU leva ~3,0–3,6 µs por operação no `sem_sync` (MoE ~2,5–3,0, atenção ~3,8–4,3), contra ~0,85 µs de banda por
    operação no MoE. Hipótese, não medida: o resto é intervalo entre kernels dependentes e kernels que não ocupam a placa;
    `gpu_ms` não separa as duas coisas, e elas pedem cortes diferentes (grafo contra fusão). O `base` mostra que um GEMV
    grande chega a ~90% da banda (0,124 ms contra o piso de 0,11 da cabeça), então o GEMV em si não é o problema.
  - Os tempos curtos batem com a faixa 0–256 do residente no M2a (3,59 ms), o que indica que o passo medido é o de produção.
- Com a atenção em duas passadas (M2d), o passo tem 13 operações de atenção por camada e 6 por token: 1.158 operações e 52
  sincronizações no 1B, 1.542 e 68 no 3B.
- Rodada 27 (rev-1 e rev-2, cegos): os dois aprovaram, sem P0 nem P1; nenhum conflito. Corrigido nesta seção e no código:
  contagem de operações medida no código; o "8–9%" virou teto; a causa por operação virou hipótese; medição de contexto
  longo; `step_probe` recusa rotas com número de camadas diferente do modelo (antes entrava em pânico no meio do passo); teste
  da recusa com cache, da contagem e de que `step()` não cria eventos; igualdade exata no `m2b`; doc do `Probe` com `skip_attn`.
  Pendente: conferir a hipótese do quase empate no 8º expert (122.047/122.048 no 3B) medindo a distância entre o 8º e o 9º logit
  do roteador no ponto que diverge.
- Consequência para o passo 2 (análise do exec e dos dois revisores, convergente):
  - Antes de mexer em kernels, uma medição curta que separa intervalo de ocupação: o mesmo GEMV de um expert repetido 1.000
    vezes na stream, com e sem `CudaGraph`; tempo por kernel de uma camada com eventos; `host_overhead` do cutile nesta máquina
    (lançamento normal contra escalar que muda a cada lançamento, caso do `len` da atenção).
  - Atenção com divisão do KV (flash-decoding: blocos por cabeça e fatia de posições, e uma passada que combina `(m, l, acc)`)
    sobe na ordem: é o maior item em contexto longo e o regime real é de conversas acima de 1.000 posições.
  - MoE: primeiro as fusões locais, sem índices dinâmicos: SwiGLU na saída do GEMV gate_up (o bloco carrega as linhas `r` e
    `I + r`) e `y = β·y + g·acc` no fim do GEMV down; 4 kernels por expert viram 2 (1.133 → ~750 operações no 1B). Depois o
    agrupamento dos k experts: pesos empilhados por camada (`[E·2I, H]`, `[E·H, I]`), índices e pesos dos experts num tensor da
    GPU, gate_up com grade `(I/BM, k)` e down com grade `(H/BM)` em que cada bloco percorre os k experts em ordem fixa e soma
    o resíduo (sem atomics, mesma ordem de soma de hoje). O cutile tem o que precisa (índice lido da GPU e usado como índice
    de tile, como no `grouped_fc1_kernel` de `cutile-examples/examples/nvfp4_moe.rs`); conferir numa compilação mínima e o
    limite de 2³¹ elementos num tensor empilhado de slots.
  - Q/K/V num GEMV só, com RoPE escrevendo K e V direto no cache: ganho pequeno (~0,07 ms), por último.
  - CUDA graph só para o caminho residente e `cache_full`, e só depois de top-k na GPU (desempate pelo menor índice, como
    `Routing::from_logits`), `pos`/`len` e a posição de escrita do KV num tensor da GPU (o `CudaGraph::update` não troca
    argumentos de kernel capturados) e uma tabela de cos/sin na GPU. Critério: rodar o `m2b` depois do MoE agrupado; se o host
    passar a GPU, o grafo é o próximo.
  - Caminho com cache (`lru_sync`, `lru_cpu`): mantém a sincronização por camada, que é onde o host decide acerto, falta,
    vítima e o que vai para a CPU. Ganha com as fusões e o agrupamento (slots num tensor empilhado, ids por cópia pequena por
    camada na stream de cálculo). Qualquer relaxamento futuro do protocolo pede antes um teste com experts do tamanho real e
    `compute-sanitizer --tool racecheck`.

## M2c — custo de cada kernel, com e sem grafo (29/10)

- Passo 1.5, pedido pela rodada 27 antes de fixar a ordem do passo 2. `rust/moe-rt/src/bin/m2c-kernels.rs`: cada kernel de uma
  camada com as dimensões do modelo (pesos sintéticos), 500 lançamentos seguidos na mesma stream, lançados um a um (`fila`,
  como no passo de hoje) e capturados num `CudaGraph` (`grafo`); eventos nas pontas; o host mede o custo de enfileirar.
  Saídas em `qual-runs/M2C-KERNELS-20261029/run1-{1b,3b}.txt`.
- Ressalva: o mesmo kernel repetido relê os mesmos pesos, que cabem nos 72 MB de L2 da 4090 (a matriz Q do 3B tem 4,7 MB). Os
  GEMV pequenos aparecem acima da banda da DRAM (976–1.566 GB/s) e são um limite inferior; no passo, os pesos de cada camada
  vêm da DRAM. O mesmo vale para o KV da atenção (um par `[4.096, kvd]`, 16,8 MB no 1B, contra 403 MB das 24 camadas no
  passo a 4.096 posições): o número de contexto longo que vale é o do `m2b --repeat`. Cada linha é uma medida só (a versão do
  M2d repete 3 vezes). Em cada execução, um kernel sai com host ~2,7 µs, estável nas 3 repetições, e é um kernel diferente a
  cada vez (o `axpby expert` no `run1`, o `rope q` no `r2/`, o `swiglu` no `r3/m2c-1b-sem-taskset`): o i7-14700KF é híbrido e
  a thread cai num E-core durante uma das medidas. Com `taskset -c 4` (um P-core) o ponto fora some (maior host 1,50 µs,
  `r3/m2c-1b-taskset-maxlen1024.txt`); as medidas do M2e em diante rodam nos P-cores. A cabeça (101 e 151 MB) não cabe na L2 e dá 896–916 GB/s, perto do piso.
- µs por lançamento (1B; o 3B é parecido, com os GEMV um pouco maiores):

  | kernel | blocos | fila | grafo | fila − grafo |
  |---|---|---|---|---|
  | axpby de 1 bloco (o menor kernel possível) | 1 | 1,52 | 0,89 | 0,63 |
  | rmsnorm, rope, axpby, swiglu, cópia para o KV | 1–16 | 1,4–2,1 | 0,8–1,5 | 0,4–0,6 |
  | GEMV q, o, gate_up (L2) | 512 | 2,5–2,8 | 1,9–2,2 | ~0,6 |
  | GEMV k, down (L2) | 256–512 | 2,2–2,5 | 1,6–1,8 | ~0,6 |
  | GEMV roteador | 16 | 1,78 | 1,20 | 0,58 |
  | atenção len = 256 | 16 | 26,9 | 26,6 | 0,3 |
  | atenção len = 1.024 | 16 | 102,8 | 94,3 | 8,5 |
  | atenção len = 4.096 | 16 | 369,8 | 372,4 | ~0 |
  | GEMV cabeça (DRAM) | 24.578 | 109,2 | 110,1 | ~0 |

- Custo de enfileirar no host no laço do microbenchmark: ~1,2–1,3 µs por lançamento. No passo é o dobro: o `sem_sync` do
  `m2b` (M2d) gasta 2,4 µs de host por operação no 1B e no 3B, e 2,7 µs no MoE (contra 2,9 µs de GPU); a razão host/GPU do
  passo é 79% no 1B e 66% no 3B (rodada 28). A primeira explicação (o `partition`/`unpartition`, `reshape` e `Arc` em volta
  de cada lançamento) não se sustentou: com partições emprestadas o custo quase não muda (2,49 contra 2,62 µs). A causa é o
  cache do ponto de lançamento do cutile, uma entrada por função de kernel, que o GEMV do gate_up (K = H) e o do down (K = I)
  derrubam um ao outro a cada expert (rodada 29; medido e corrigido no M2e). Exceção no lançamento em si: até o M2c, o `len` da atenção, que muda a cada token,
  custava 4,75 µs (escalar fixo: 1,31; desde o M2d o `len` vem de um tensor da GPU). Confirma o P3-6 da rodada 27 (o escalar inteiro que muda derruba o cache do ponto de
  lançamento do cutile). O peso `g` em f32 não tem esse efeito (1,18 contra 1,16).
- Leitura:
  - A atenção de decode é o kernel mais lento, e de longe: ~45 GB/s em qualquer comprimento, 20–25× abaixo da banda. A 1.024
    posições são 103 µs por camada, 2,5 ms por token no 1B, o que bate com o `so_atencao` do M2b nessa faixa (3,53 ms, junto
    com as outras 11 operações). Com 16 ou 24 blocos, a placa fica quase vazia; nem o grafo ajuda, porque o tempo é do
    kernel e não do lançamento. A divisão do KV é o corte de maior ganho, em qualquer contexto (já 27 µs por camada a 256
    posições, 0,65 ms por token).
  - O intervalo entre lançamentos que um grafo tira é ~0,6 µs por lançamento, medido limpo só nos GEMV (onde a GPU limita);
    nos kernels pequenos a fila fica perto do host e a diferença mede em parte a taxa do host. Nas ~1.085 operações do passo sem sincronização
    seriam ~0,65 ms no 1B (19% dos 3,47 ms), menos depois das fusões. Os kernels pequenos (rmsnorm, rope, axpby, swiglu, cópia)
    custam ~1,5 µs na fila sem quase ler memória: é custo de existir, e só fusão ou grafo o tiram.
  - O host fica logo abaixo da GPU: no passo, 2,4 µs contra ~3,0 µs por operação (no MoE, 2,7 contra 2,9). Depois das
    fusões, o host pode passar a ser o gargalo, e aí o grafo sobe na ordem; acompanhar a razão host/GPU do `m2b` por etapa.
  - SwiGLU e os dois AXPBY por expert somam ~3 µs por expert, 8 × 24 × 3 ≈ 0,58 ms por token no 1B, sem trabalho útil de
    banda: as fusões locais pagam isso.
- Ordem do passo 2 a partir disto:
  1. Atenção com divisão do KV e `len` num tensor da GPU (tira os 3,5 µs de host por camada e já é pré-requisito do grafo).
     Estimativa de antes do M2d: "de ~2,5 ms para algumas dezenas de µs por token a 1.024 posições". Errada em uma ordem de
     grandeza: eram dezenas de µs por camada. Medido no M2d: 27 µs por camada, ~0,65 ms por token a 1.024 posições, e ganho
     de ~1,5× a 256.
  2. Fusões locais do MoE (SwiGLU na saída do gate_up, AXPBY no fim do down): ~0,5 ms por token no 1B.
  3. Agrupamento dos k experts: o que sobra de lançamentos e de ocupação do MoE.
  4. Grafo, quando o host virar o gargalo (rodar o `m2b` depois de cada etapa).

## M2d — atenção de decode com divisão do KV (29/10)

- Primeiro corte do passo 2, na ordem que o M2c deu. O `attn_decode` (um bloco por cabeça de Q percorrendo o KV inteiro em
  série) virou duas passadas em `rust/moe-rt/src/gpu.rs`, como no flash-decoding:
  - `attn_split`: grade fixa `cabeças × ATTN_NS` (64 blocos por cabeça); as `nb = ⌈len/ATTN_BN⌉` fatias de 64 posições se
    dividem entre os blocos da cabeça, `⌈nb/64⌉` por bloco, então a grade não depende de `len` nem de `max_len` e um contexto
    curto continua com um bloco por fatia. Cada bloco faz o softmax online das suas posições e escreve a saída normalizada e
    o `lse = m + ln l`; um bloco sem posições sai com 0 e −∞ (piso de 1e-30 em `l` para não dar 0/0).
  - `attn_combine`: um bloco por cabeça junta, em ordem, só os blocos com posições (`⌈nb / ⌈nb/64⌉⌉`), com o softmax online
    sobre os `lse`.
  - A primeira versão (rodada 28) dividia por `max_len` (`ns = max_len/64`, fatia fixa de 64 posições): a grade e a segunda
    passada cresciam com o `max_len` alocado, o que num servidor de contexto longo (256k) daria 65 mil blocos e 33 MB de
    `part` por camada. Os dois revisores apontaram; a divisão por `len` resolve sem depender do `max_len` alvo.
  - `len` vem de um tensor da GPU (enviado por token junto do embedding e do cos/sin: +1 operação e +1 sincronização), então a
    grade não depende da posição (pré-requisito do grafo) e o host não paga mais os 4,75 µs do escalar que mudava.
  - `gpu_model::attention` é a função pública que o passo e o teste usam.
- Teste novo `atencao_em_duas_passadas_bate_com_a_referencia_em_toda_fronteira_de_fatia` (`tests/model.rs`), contra a atenção
  em f64 com |Δ| < 1e-5, 4 cabeças de Q e 2 de KV, posições depois de `len` com valores finitos quaisquer:
  - `ns` = 8 com `max_len` = 2.048 (divisão desigual: 9 fatias dão 2 por bloco e 5 blocos com posições) e `ns` = 64 com
    `max_len` = 8.192 (4.097 posições dão 2 fatias por bloco); `len` nas fronteiras de fatia e de bloco;
  - cada caso também ampliado (`q` ×4 e uma fatia no meio ×4: scores de dezenas, fatia dominante depois da primeira);
  - os blocos sem posições têm saída 0 e `lse` = −∞ em `part`.
  Conferido que reprova com a segunda passada lendo só o primeiro bloco e com `den = den + w` (sem o fator de reescala).
  A contagem do teste do `Probe` foi para 13 operações de atenção e 6 por token. `cargo test`: 31 de integração + 2 dos
  binários.
- Validação: `qual-runs/M2D-ATENCAO-20261029/` é a primeira versão; `r2/` é a versão com blocos fixos por cabeça, com os
  mesmos resultados de precisão (as velocidades abaixo são da primeira versão; as do `r2/` estão na coluna "r2" da tabela):
  - `m1b-modelo`: 1B argmax 277/277, erro dos logits 7,0e-6 (antes 9,0e-6), roteamento 6648/6648; 3B 277/277, 2,9e-6 (antes
    2,7e-6), 8864/8864.
  - `m1b-gulosa`: 64/64 nos 3 casos, GPU e CPU, nos dois modelos; GPU a 266–272 tok/s no 1B e 175–179 no 3B (no `r2/`,
    280–282 e 180–181).
  - `m2a-cache`: logits dentro da tolerância em todas as políticas e orçamentos; roteamento como antes (82.824/82.824 no 1B,
    122.047/122.048 no 3B); `sem_sync` igual ao `normal` bit a bit no `m2b`.
- Tempo (rodadas em sessões diferentes das anteriores; a deriva entre processos é de 6–11%):

  | | 1B antes | 1B 1ª versão | 1B r2 | 3B antes | 3B 1ª versão | 3B r2 |
  |---|---|---|---|---|---|---|
  | `attn_decode` → 1ª + 2ª passada, len 256 (µs, fila) | 26,9 | 16,0 + 2,1 | 13,8 + 2,1 | 26,6 | 20,2 + 2,2 | 17,9 + 2,1 |
  | len 1.024 | 102,8 | 23,9 + 3,1 | 20,5 + 2,9 | 99,3 | 29,9 + 3,0 | 26,0 + 2,8 |
  | len 4.096 | 369,8 | 51,7 + 6,4 | 44,4 + 5,9 | 369,9 | 74,7 + 5,8 | 63,3 + 5,9 |
  | `m2b` `so_atencao` GPU, posições 1024–1108 (ms) | 3,53 | 1,38 | 1,37 | 4,48 | 2,09 | 2,19 |
  | `m2b` `normal` GPU, posições 1024–1108 (ms) | 6,10 | 4,06 | 3,74 | 8,06 | 6,10 | 5,82 |
  | `m2a` residente, média nas conversas (ms/token) | 4,37 | 3,85 | 3,66 | 7,06 | 5,86 | 5,71 |
  | `m2a` residente, posições 1024+ | 5,65 | 4,16 | 3,76 | 8,99 | 6,12 | 5,86 |
  | `m2a` `lru_cpu` 25% | 10,97 | 10,14 | 9,23 | 17,98 | 16,11 | 15,30 |
  | `m2a` `lru_cpu` 50% | 8,05 | 7,21 | 6,67 | 11,97 | 9,86 | 10,10 |

- Versão com blocos fixos por cabeça (`r2/`, mesma sessão para as linhas do M2d de agora; µs por chamada na fila, mediana de
  3, faixas de ±1%):

  | len | 1 | 64 | 256 | 1.024 | 4.096 | 16.384 |
  |---|---|---|---|---|---|---|
  | 1B 1ª passada | 13,8 | 12,6 | 13,8 | 20,5 | 44,4 | 144,1 |
  | 1B 2ª passada | 1,8 | 1,8 | 2,1 | 2,9 | 5,9 | 5,9 |
  | 3B 1ª passada | 16,8 | 15,5 | 17,9 | 26,0 | 63,3 | 210,0 |

  A segunda passada não cresce mais com o contexto além de 64 fatias. A primeira passada tem um piso de ~13 µs (1B) mesmo com
  uma posição (a frase anterior, "uma fatia de 64 posições leva ~13 µs", era extrapolação). Separado na rodada 29, variando os
  blocos por cabeça com len = 1 (grafo, `r3/m2c-1b.txt`): 7,6 / 10,3 / 9,7 / 12,2 µs com ns = 8 / 16 / 32 / 64. O custo
  cresce com os blocos, de 7,6 µs (128 blocos) a 12,2 µs (1.024), mas sem forma linear nos pontos do meio (ns = 16 custa mais
  que ns = 32), então não dá para separar em latência de bloco mais custo por bloco. O que a varredura mostra: com 128 blocos
  o piso já é 7,6 µs, então menos blocos em contexto curto tiraria no máximo ~4,6 µs por camada. Com len = 256, mesma forma. `m2a` residente: 3,66 ms no 1B e 5,71 no 3B nas conversas; 3,76 e
  5,86 em 1024+. `m2b` `normal` GPU em 1024–1108: 3,74 (1B) e 5,82 (3B).
- Custo de host (rodada 28, P2 do rev-2): os 4 lançamentos de um expert enfileirados como em `GpuLayer::run` custam 2,62 µs
  por lançamento no host, e com partições emprestadas 2,49 µs, contra 1,2–1,4 µs repetindo um mesmo kernel. A diferença não
  é o `partition`/`unpartition` do nosso código. A conclusão que estava aqui ("é alternar kernels diferentes no lançador; só
  menos lançamentos ou grafo resolvem") não tinha prova, e os dois revisores da rodada 29 a contestaram com o contraexemplo
  da atenção (~8 kernels alternados a 1,43 µs por operação). A causa medida está no M2e.
- Leitura:
  - A atenção quase deixou de crescer com a posição (1B: 1,10 → 1,38 ms entre as faixas 0–256 e 1024+, antes 1,25 → 3,53), e
    o residente nas conversas caiu 12% no 1B e 17% no 3B (16% e 19% no `r2/`). O MoE passou a ser o maior item (`so_moe`
    2,21 ms no 1B e 3,23 no 3B no `r2/`).
  - Em posições curtas o ganho é pequeno: o piso de ~13 µs por camada da primeira passada (~0,33 ms por token no 1B) domina.
    Ajustes possíveis depois: menos blocos quando há poucas fatias (exige saber o `len` no host ou um lançamento indireto),
    mais de uma cabeça de Q por bloco (as `G` cabeças que dividem uma cabeça de KV relendo o mesmo K e V), KV em bf16.
  - O residente segue ~2,1× o llama.cpp no 1B (3,66 contra 1,76 ms, `r2/`) e ~1,7× no 3B (5,71 contra 3,45).
- Rodada 28 (rev-1 e rev-2, cegos): os dois aprovaram, sem P0 nem P1, sem conflito. Aplicado: divisão por `len` em vez de
  `max_len` (P2 do rev-2, P3 do rev-1), teste ampliado (os dois), custo de host do passo no plano e medição do lançador (os
  dois), previsão do M2c marcada como superada (os dois), ressalva da L2 também para o KV, repetições no `m2c`, piso da
  primeira passada medido. Pendente: os três envios sincronizados por token (embedding, cos/sin, `len`) viram um buffer só
  na etapa do grafo.
- Rodada 29 (rev-1 e rev-2, cegos): os dois aprovaram, sem P0 nem P1, sem conflito; os dois conferiram os limites de
  `per`/`j0`/`j1`/`used`. Confirmado pelos dois: a explicação do custo de host estava errada ou sem prova (corrigida no M2e),
  um ponto fora novo no `m2c` (o `rope q`) e números do M2d defasados em relação ao `r2/` (coluna nova na tabela). Únicos e
  aplicados: o piso da primeira passada separado em latência de bloco e custo da grade (varredura de ns), asserções de forma
  em `gpu_model::attention` com o teste `atencao_recusa_part_com_menos_blocos_que_a_grade_pede`, o bullet do M2c que culpava o
  `partition`/`reshape`/`Arc` e o `cargo-test.txt` regravado com comando, ambiente e lista de testes (`r3/`).

## M2e — um lançador para o GEMV do down (29/10)

- Causa do custo de host do MoE (rodada 29, os dois revisores): o lançador que o `#[cutile::entry]` gera guarda a última
  especialização resolvida num `static LaunchSite` por função de kernel, com uma entrada só, que compara generics e bits de
  forma (`cutile-rs`, `tile_kernel.rs`, "single entry: last one wins"). Um acerto custa ~1,25 µs; uma falta refaz a chave
  e consulta o cache global. No passo, todo GEMV tem K = H (q, k, v, o, roteador, gate_up, cabeça; qd = H nos dois modelos),
  menos o down (K = I), então cada expert dava duas faltas. Rope de q e de k não alternam: o `DivHint` satura em 16, e todas
  as formas são múltiplas de 16.
- Controles no `m2c-kernels` (P-core fixo, `r3/m2c-{1b,3b}.txt`), µs de host por lançamento nos 4 de um expert:

  | | 1B | 3B |
  |---|---|---|
  | `gemv` nos dois GEMVs (como até o M2d) | 2,52 | 2,53 |
  | down com o mesmo K do gate_up | 1,23 | 1,26 |
  | down por um kernel de outro nome (`gemv_down`) | 1,23 | 1,27 |
  | agrupado por kernel (8 gate_up, 8 swiglu, 8 down, 8 axpby) | 1,39 | 1,43 |
  | referência: só o gate_up, repetido | 1,23 | 1,27 |
  | como em `GpuLayer::run` agora (partições próprias, `gemv_down`) | 1,25 | 1,29 |

- Correção: `kernels::gemv_down` em `rust/moe-rt/src/gpu.rs`, com o corpo idêntico ao do `gemv` e um lançador próprio, usado
  no down de `GpuLayer::run` e do cache (`cache.rs`). Testes novos em `tests/block.rs`: `gemv_down_tem_o_mesmo_corpo_do_gemv`
  (compara o texto dos dois kernels, do `#[cutile::entry]` ao fim do corpo, trocando o nome, para um não mudar sem o outro),
  `gemv_down_na_gpu_da_os_mesmos_bits_do_gemv` (K = 768, 512 e 256) e `down_dos_experts_usa_gemv_down` (o down de `gpu.rs`
  e de `cache.rs` chama `gemv_down`; voltar ao `gemv` compila e dá o mesmo resultado, só o host piora). `cargo test`: 37 testes, todos passam (`r3/cargo-test.txt`). Alternativa não feita: um `LaunchSite` com mais de
  uma entrada no cutile-rs, mudança na dependência.
- Medido na mesma sessão, binários de antes (a mesma árvore com o down por `gemv` em `gpu.rs` e `cache.rs`) e de depois
  intercalados, duas rodadas de cada, nos P-cores (`r3/rodar.sh`, com a receita dos binários e os sha256 em
  `r3/bin/SHA256SUMS`; `r3/m2b-*-longo-{antes,depois}-{1,2}.txt`, `r3/gulosa-*`). `m2b` com 1.108 tokens, ms por token:

  | | 1B antes | 1B depois | 3B antes | 3B depois |
  |---|---|---|---|---|
  | `normal` total | 3,68 / 3,67 | 3,60 / 3,61 | 5,70 / 5,70 | 5,70 / 5,70 |
  | `sem_sync` host | 2,60 / 2,60 | 1,51 / 1,53 | 3,40 / 3,38 | 2,04 / 2,04 |
  | `sem_sync` total (GPU limita) | 3,34 | 3,35 / 3,34 | 5,24 / 5,23 | 5,24 |
  | `so_moe` host | 2,14 / 2,13 | 1,06 / 1,08 | 2,77 / 2,76 | 1,42 / 1,43 |
  | `so_moe` GPU | 2,25 / 2,24 | 2,14 | 3,23 | 3,24 |
  | razão host/GPU do `sem_sync` | 79% | 46% | 65% | 39% |

  `m1b-gulosa`: 64/64 nos 3 casos, antes e depois, nos dois modelos; 277–284 tok/s no 1B e 179–181 no 3B, sem diferença
  fora do ruído. `m1b-modelo` depois: 1B argmax 277/277, erro dos logits 7,0e-6, roteamento 6648/6648; 3B 277/277, 2,9e-6,
  8864/8864. Os logits do `sem_sync` seguem iguais aos do `normal` bit a bit.
- `m2a-cache`, mesma sessão e P-cores, binários de antes rodados depois dos de depois (`r3/m2a-{1b,3b}{,-antes}.txt`), média
  nas conversas em ms por token:

  Uma execução por lado, e a política `lru_cpu` não é determinística (a taxa de acerto e as cópias em segundo plano mudam
  entre execuções: no 1B/50%, acerto 74,4% contra 74,8%). Com a deriva entre processos de até ~3% vista no `m2b`, as
  diferenças de até 3% ficam dentro do ruído; só as de 6–11% saem dele. Repetir antes e depois intercalados (2 de cada)
  fica para quando o caminho de cache voltar a ser o foco.

  | | 1B antes | 1B depois | 3B antes | 3B depois |
  |---|---|---|---|---|
  | residente | 3,63 | 3,58 (−1%) | 5,72 | 5,68 (−1%) |
  | `lru_cpu` 10% | 10,46 | 10,37 (−1%) | 18,61 | 18,10 (−3%) |
  | `lru_cpu` 25% | 8,55 | 8,38 (−2%) | 14,61 | 13,62 (−7%) |
  | `lru_cpu` 50% | 6,42 | 5,72 (−11%) | 9,63 | 8,79 (−9%) |
  | `lru_cpu` 75% | 4,52 | 4,15 (−8%) | 6,88 | 6,48 (−6%) |

  Logits dentro de 1e-4 e roteamento como antes (82.824/82.824; 3B 122.047/122.048, e 122.046 no `lru_cpu` 75% do binário
  de antes, o mesmo quase empate de sempre). Os números de antes já são menores que os do `r2/` (25% no 1B: 8,55 contra
  9,23): fixar nos P-cores também acelera o caminho de CPU, então a comparação válida é só antes contra depois desta sessão.
- Leitura:
  - O host do passo sem sincronização (`sem_sync`) caiu ~40% nos dois modelos (1,1 ms por token no 1B, 1,35 no 3B); no
    `normal`, que espera a GPU no roteador de cada camada, o host cai só 3% (3,51 → 3,40 ms no 1B). A GPU já limitava: com
    experts residentes, o tempo de parede cai 2% no 1B e nada no 3B. No caminho de cache, onde o host espera a GPU a cada
    camada e ainda calcula experts na CPU, as quedas vão de −1% a −11% conforme o orçamento (6–11% com 50–75% dos experts na
    VRAM), numa execução por lado e sem controlar a taxa de acerto.
  - Monitor desta propriedade: o `m2b` `sem_sync` host (~1,5 ms no 1B e ~2,0 no 3B; voltar a ~2,6 / ~3,4 é sinal de que um
    GEMV com outro K voltou a dividir o nome de kernel) e o teste `down_dos_experts_usa_gemv_down` (abaixo). Um modelo com
    `qd ≠ H` reintroduz duas faltas por camada (o GEMV `o` e o `q` seguinte): pequenas perto das ~16 por camada que o MoE
    tinha, mas a conferir ao trocar de modelo.
  - O grafo desce na ordem: a razão host/GPU foi para 46% e 39%, longe dos ~90% que o colocariam na frente. As fusões e o
    agrupamento tiram tempo de GPU e vão subir essa razão de novo; medir depois de cada etapa.
  - O agrupamento por kernel também evita as faltas, mas fica um pouco acima (1,39 contra 1,23 µs) e pede k buffers; o
    `gemv_down` resolve com uma linha por chamada.
- Rodada 30 (rev-1 e rev-2, cegos): os dois aprovaram, só P3. Aplicado: números da varredura de ns e da gulosa conferidos
  com as saídas e a leitura da varredura sem modelo linear (os dois); o `m2a` marcado como uma execução por lado, com as
  diferenças de até 3% dentro do ruído (os dois; a repetição intercalada ficou para depois); "host do passo" trocado por
  "host do `sem_sync`"; o `len` escalar no M2c no passado; o teste do texto do kernel passou a comparar atributo e
  assinatura; teste novo `down_dos_experts_usa_gemv_down`; o teste de forma da atenção deixa de passar vazio sem GPU
  (`catch_unwind` e pulo, como os outros); `sync_on_err` também na 1ª passada da atenção; receita e sha256 dos binários
  de antes e de depois no `rodar.sh`. Conferido que os dois testes de texto reprovam com `unchecked_accesses = false` só no
  `gemv_down` e com o down de `cache.rs` de volta ao `gemv`.
- Próximo: fusões locais do MoE (SwiGLU na saída do GEMV gate_up, AXPBY no fim do GEMV down). Um GEMV com epílogo diferente
  já ganha um nome e um lançador próprios.

## M2f — fusões locais do MoE: 4 → 2 lançamentos por expert (29/10)

- O que mudou (`rust/moe-rt/src/gpu.rs`): `kernels::gemv_swiglu` (GEMV do gate_up com o SwiGLU na saída, sem escrever o
  vetor `h` de `2I`) e `kernels::gemv_axpby` (GEMV do down com `y = β·y + g·(W·a)` na saída, sem o vetor `o` de `H`). Os
  dois lançamentos de um expert ficam em `enqueue_expert`, usado por `GpuLayer::run` e pelo cache (`cache.rs`). Saíram
  `swiglu`, `gemv_down` e `SWIGLU_B`. Cada kernel novo tem um K só por modelo, então a lição do M2e (um nome de kernel por
  especialização) segue valendo.
- Evidência em `qual-runs/M2F-FUSOES-20261029/`, em duas medidas, nos P-cores:
  - primeira (`rodar.sh`, 20:30–21:34): `antes` = a árvore do M2e, `depois` = a fusão com BM = 2, intercalados; `m2a` 2× por
    lado no 1B e 1× no 3B. `cargo test` terminou às 20:30:15, antes da medida (`cargo-test-medida.txt`); fontes do
    `depois` em `bin/FONTES-depois.sha256`;
  - segunda (`rodar-bm1.sh`, 21:37–22:05): `depois` contra `final` (o `gemv_swiglu` com BM = 1 e as correções da rodada
    31), intercalados; `m2a` só do `final`, 1× por modelo. `cargo test` da árvore final (37 testes) às 21:37, antes da
    medida (`cargo-test.txt`); fontes em `bin/FONTES-final.sha256`; sha256 dos binários em `bin/SHA256SUMS`;
  - terceira (`rodar-r3.sh`, 22:13–22:39, pedida pela rodada 32): `lru_cpu` do `depois` e do `final` intercalados, 3
    execuções cada, e `--backlog 0`. Depois dela só mudou um comentário (o doc de `gemv_swiglu`, rodada 32), então
    `src/gpu.rs` não bate mais com `bin/FONTES-final.sha256` por essa linha.
- Kernels de um expert (`m2c-kernels`, P-core fixo, µs; grafo = capturado num CudaGraph, fila = lançados um a um, como no
  passo), cada linha do binário que ela nomeia (`m2c-{1b,3b}-{antes,depois,final}.txt`):

  | | 1B grafo | 1B fila | 3B grafo | 3B fila |
  |---|---|---|---|---|
  | antes: `gemv` gate_up + `swiglu` + `gemv` down + `axpby` | 2,00 + 0,95 + 1,57 + 0,84 = 5,36 | 7,75 | 2,42 + 0,95 + 1,72 + 0,84 = 5,93 | 8,30 |
  | depois, BM = 2: `gemv_swiglu` + `gemv_axpby` | 2,54 + 1,64 = 4,18 | 5,37 | 2,93 + 1,82 = 4,75 | 5,90 |
  | final, `gemv_swiglu` com BM = 1 | 2,16 + 1,64 = 3,80 | 4,96 | 2,28 + 1,82 = 4,10 | 5,19 |
  | host por lançamento nos 2 de um expert (final) | 1,29 | | 1,27 | |

  O passo paga a fila, não o grafo: (7,75 − 4,96) µs × 192 experts = 0,54 ms por token no 1B e (8,30 − 5,19) × 256 = 0,80
  ms no 3B, contra 0,59 e 0,76 ms medidos no `so_moe` (abaixo). A diferença entre fila e grafo (~0,6 µs por lançamento, o
  intervalo entre lançamentos que o M2c mediu) é o que um grafo ainda tiraria.

  Os dois revisores da rodada 31 apontaram que a fusão cortava a grade do gate_up pela metade (`I/BM` blocos, cada um com
  duas matrizes). Medido: com BM = 2 o `gemv_swiglu` ficava em 829 GB/s no 1B, contra 1.055 do GEMV de `2I` linhas. Com
  BM = 1 (`SWIGLU_BM`, `I` blocos) volta a 974 GB/s no 1B e chega a 1.385 GB/s no 3B, acima do GEMV de `2I` linhas
  (1.302). O `gemv_axpby` com BM = 1 não muda (1,64 → 1,61 e 1,82 → 1,84 µs) e fica com `GEMV_BM` = 2. A escolha veio de
  uma medida exploratória (`m2c-{1b,3b}-bm1.txt`, com as linhas de BM = 1 acrescentadas ao `m2c` sobre a árvore do
  `depois`; esse binário não foi guardado) e se confirmou no binário `final`, que tem as duas variantes lado a lado. Observação para depois: o próprio `gemv` com BM = 1 talvez ganhe no 3B (q, k, v, o, roteador);
  o caminho residente está congelado (ask-6), então isso fica anotado, não medido.
- Passo inteiro, primeira medida (`m2b`, 1.108 tokens, duas execuções por lado, ms por token):

  | | 1B antes | 1B depois | 3B antes | 3B depois |
  |---|---|---|---|---|
  | `normal` total | 3,60 / 3,65 | 3,10 / 3,11 | 5,73 / 5,77 | 5,03 / 5,08 |
  | `sem_sync` host | 1,53 / 1,53 | 1,05 / 1,05 | 1,99 / 2,00 | 1,39 / 1,39 |
  | `sem_sync` GPU | 3,30 / 3,33 | 2,80 / 2,79 | 5,22 / 5,25 | 4,60 / 4,60 |
  | `so_moe` GPU | 2,14 / 2,14 | 1,61 / 1,61 | 3,24 / 3,24 | 2,61 / 2,61 |
  | `so_atencao` GPU (controle) | 1,23 / 1,23 | 1,23 / 1,23 | 2,04 / 2,04 | 2,04 / 2,04 |
  | razão host/GPU do `sem_sync` | 46% | 38% | 38% | 30% |
  | operações por token (`normal`) | 1.158 | 774 | 1.542 | 1.030 |

  O ganho de parede veio da GPU: o `so_moe` caiu 25% no 1B e 19% no 3B, e a atenção, que não mudou, ficou igual. O host do
  `sem_sync` também caiu 0,5 ms por token no 1B e 0,6 no 3B (384 e 512 lançamentos a menos), mas a GPU já limitava. A razão host/GPU desceu
  para 38% e 30%, longe dos ~80% que reabririam o grafo (ask-6).
- Segunda medida, BM = 2 contra BM = 1 (`m2b-*-r2-*`, `gulosa-*-r2-*`, duas execuções por lado):

  | | 1B depois | 1B final | 3B depois | 3B final |
  |---|---|---|---|---|
  | `normal` total | 3,09 / 3,09 | 3,04 / 3,03 | 5,02 / 5,03 | 4,88 / 4,89 |
  | `sem_sync` host / GPU | 1,03 / 2,79 | 1,04 / 2,72 | 1,37 / 4,60 | 1,39 / 4,48 |
  | `so_moe` GPU | 1,61 | 1,55 | 2,61 | 2,48 |
  | gulosa, tok/s (3 casos × 2) | 328–332 | 332–339 | 202–204 | 210–213 |

  BM = 1 tira mais 2% no 1B e 3% no 3B. No total do M2f, o passo residente foi de 3,62 para 3,03 ms no 1B (−16%) e de
  5,75 para 4,88 ms no 3B (−15%); a razão host/GPU do `sem_sync` fica em 38% e 31%.
- Correção: `m1b-gulosa` 64/64 nos 3 casos, antes e depois, nos dois modelos (1B ~282 → ~329 tok/s; 3B ~175 → ~204, o `antes` do 3B de uma execução que variou de 170 a 180).
  `m1b-modelo` depois e final (os mesmos números nos dois): 1B argmax 277/277, erro dos logits 5,6e-6 (era 7,0e-6),
  roteamento 6648/6648; 3B 277/277, 2,9e-6, 8864/8864. A gulosa do `final` também dá 64/64 nos 3 casos nos dois modelos. O `sem_sync` segue igual ao `normal` bit a bit. A fórmula é a mesma, os bits não: o compilador contrai as contas
  de outro jeito, e o doc de `gemv_swiglu` diz isso.
- `m2a-cache`, média nas conversas em ms por token (1B: duas execuções por lado na primeira medida; 3B: uma; `final`: uma por
  modelo, na segunda medida, ~1 h depois):

  | | 1B antes | 1B depois | 1B final | 3B antes | 3B depois | 3B final |
  |---|---|---|---|---|---|---|
  | residente | 3,60 / 3,58 | 3,06 / 3,10 | 3,05 | 5,72 | 5,00 | 4,86 |
  | `lru_cpu` 10% | 10,56 / 11,09 | 10,02 / 10,60 | 10,13 | 18,42 | 19,10 | 17,53 |
  | `lru_cpu` 25% | 8,10 / 8,41 | 7,66 / 8,14 | 7,78 | 13,68 | **16,26** | 13,33 |
  | `lru_cpu` 50% | 5,53 / 6,12 | 5,15 / 5,20 | 5,19 | 9,04 | **12,90** | 8,16 |
  | `lru_cpu` 75% | 4,11 / 4,25 | 3,66 / 3,69 | 3,61 | 6,52 | 6,17 | 5,73 |
  | `lru_sync` 50% | 12,30 / 12,26 | 11,72 / 11,77 | 11,58 | 18,26 | 17,42 | 17,16 |

  Na primeira medida, o `depois` do 3B piorou a 25% e 50% (19% e 43%), com a mesma taxa de acerto, mais cópias em segundo
  plano (42,8 → 53,6 e 29,3 → 35,4 por token) e mais CPU pelos mesmos experts (9,05 → 10,92 e 3,77 → 4,77 ms por token).
  O `final` não repetiu isso. Como o `final` é outro binário, a rodada 32 (P2-1 do rev-1) pediu para rodar de novo o
  próprio `depois`: terceira medida, `rodar-r3.sh`, só `lru_cpu` 25% e 50%, `depois` e `final` intercalados, 3 execuções
  cada (`m2a-{1b,3b}-{depois,final}-r3-{1,2,3}.txt`), ms por token, com as cópias em segundo plano por token entre
  parênteses:

  | | 3B 25% | 3B 50% | 1B 25% | 1B 50% |
  |---|---|---|---|---|
  | `depois` | 13,14 (41,2) / **15,51 (51,3)** / 13,20 (41,6) | 8,15 / **9,21** / 8,09 | 8,38 / 7,89 / 7,82 | 5,33 / 5,33 / 5,20 |
  | `final` | 13,18 (41,5) / 13,40 (42,0) / 13,10 (41,3) | 8,27 / 8,30 / 8,04 | 8,03 / 7,88 / 7,79 | 5,25 / 5,47 / 5,23 |
  | mediana `depois` / `final` | 13,20 / 13,18 | 8,15 / 8,27 | 7,89 / 7,88 | 5,33 / 5,25 |

  As medianas dos dois binários são iguais: a piora não é efeito das fusões. A execução lenta reapareceu uma vez em três
  no `depois` (e nenhuma em quatro do `final` no 3B), de novo com mais cópias em segundo plano (51,3 contra ~41) e mais CPU
  por expert (0,080 contra 0,070 ms), com o mesmo acerto (49,4% contra 50,2%). É um modo lento que aparece em parte das
  execuções do `lru_cpu`, não um valor que deriva. Fora dele, a faixa entre execuções do mesmo binário é de 2–3% no 3B e
  5–8% no 1B. O `lru_sync`, sem admissão em segundo plano, varia 0,3% (1B 50%: 12,30 e 12,26).
- Causa da correlação "mais cópias em segundo plano ↔ mais CPU por expert": duas leituras possíveis. (a) As cópias disputam
  a banda de memória do host com os experts na CPU. (b) Causalidade inversa (P3-2 do rev-2): uma CPU mais lenta naquela
  execução alonga a camada, e a admissão, que depende do tempo, completa mais cópias no mesmo token; entre orçamentos, em que
  as cópias variam 4×, o CPU por expert fica quase plano (1B 0,046–0,050 ms; 3B 0,069–0,077), o que favorece (b). A
  tentativa de separar as duas com `--backlog 0` (`m2a-*-final-backlog0-{1,2}.txt`) não decide: sem admissão o cache fica
  vazio, 100% dos experts vão à CPU, e o lote maior por camada muda o CPU por expert por outro motivo (3B 0,061 contra
  0,070 ms; 1B 0,039 contra 0,048). O teste limpo é o C: cache preenchido de antemão, admissão desligada, mesmo lote. A
  medida serviu para outra coisa: com `--backlog 0` o `lru_cpu` é determinístico (acerto 0%, zero cópias), e o caminho de
  CPU sozinho varia 0,2–4% entre execuções (3B 25%: 19,10 e 19,14; 1B 50%: 9,64 e 10,04).
- Consequências para o B0 e o C: (1) a variância do `lru_cpu` vem da admissão em segundo plano, e cada tempo publicado tem
  de vir com acerto, `cóp.bg` e CPU ms; (2) a admissão é parte da política, e a histerese e o limite de trocas do Strata são
  uma linha de base para ela; (3) o critério de não regressão do B0 muda (abaixo, na ordem aprovada).
- Parâmetros do modelo de custo na velocidade final (pedido da ask-6; `m2a-*-final-1.txt`):

  | | 1B | 3B |
  |---|---|---|
  | tempo de parede por token, residente (`m2a`, média nas conversas) | 3,05 ms | 4,86 ms |
  | GPU por token, residente (`m2b` `normal`, 1.108 tokens; `sem_sync` entre parênteses) | 2,99 ms (2,72) | 4,83 ms (4,48) |
  | cópia de um expert, síncrona (`lru_sync` 10%, 0% de acerto: espera ÷ cópias) | 0,157 ms (3,15 MB, 20 GB/s) | 0,233 ms (4,72 MB, 20 GB/s) |
  | um expert na CPU, 8 threads (`lru_cpu` 10%: CPU ms ÷ experts na CPU) | 0,046 ms | 0,069 ms |
  | razão CPU ÷ cópia | 0,30 | 0,30 |

  Calcular um expert frio na CPU custa ~30% de copiá-lo pela PCIe, nos dois modelos. É um dos motivos de o `lru_cpu` ganhar
  do `lru_sync` em todo orçamento; o outro é que o `lru_cpu` sobrepõe CPU, cópia em segundo plano e GPU, e o `lru_sync`
  espera a cópia no caminho crítico. Para o simulador (passo 3), o tempo de GPU é a segunda linha, não a primeira. O custo de CPU por expert sobe ~20% nas execuções com mais cópias em segundo plano (acima),
  então ele não é constante e o simulador tem de saber disso.

- Rodada 31 (rev-1 e rev-2, cegos, sobre o código antes da medida): os dois aprovaram, só P3. Os dois apontaram a grade do
  gate_up cortada pela metade (medido acima: BM = 1 adotado) e o `β = 0` que não apaga um NaN do `y` anterior (documentado
  no `gemv_axpby`; depois de um erro a camada fica `POISONED`, então um `y` com NaN não é reaproveitado). Aplicado também:
  asserts de forma em `enqueue_expert` (pesos, `x`, `a` e `y`) com o teste `enqueue_expert_recusa_a_com_tamanho_errado`,
  que reprova com o assert comentado (mutação conferida); formas (1536, 512) e (256, 768) no teste numérico, cobrindo o H do
  3B e um down com três fatias, e o teste agora roda o `gemv_swiglu` com BM = 1 e 2; o doc de `gemv_swiglu` diz "mesma
  fórmula, não os mesmos bits"; o teste de texto diz que a contagem de um `gemv_axpby` é intencional; hashes das fontes e
  horário do `cargo test` registrados.
- Rodada 32 (rev-1 e rev-2, cegos, sobre o código final e cada número desta seção): os dois aprovaram (rev-1 com um P2,
  rev-2 sem ressalvas); todos os números conferem com os arquivos. Aplicado: o P2-1 do rev-1 (o critério de 2% do B0 estava
  abaixo do ruído do `lru_cpu` e o `depois` do 3B não tinha sido repetido): terceira medida acima e critério novo no passo
  2 da ordem aprovada. Os dois: a linha final da tabela de kernels vinha da medida exploratória (`bm1`), agora vem do
  binário `final`, com a coluna de fila e a conta que fecha com o `so_moe`; o host do `sem_sync` caiu 0,5 ms no 1B e 0,6 no
  3B. Do rev-1: a linha de GPU do modelo de custo era tempo de parede (agora as duas, rotuladas); "é por isso" trocado por
  "um dos motivos", com a sobreposição; o ponto do `d_res` do Strata marcado como lido e não executado, com o comentário que
  diz o contrário. Do rev-2: o doc de `gemv_swiglu` dizia "contra f64" (é contra o transformers); a causalidade inversa
  como leitura alternativa da correlação; a gulosa do 3B escrita como ~175 → ~204. Não aplicado: a linha da mutação
  "repetida" no `cargo-test.txt` (o arquivo tem uma só); os arquivos citados e não congelados nesta rodada entram na
  próxima que tocar o M2f.
- M2f encerrado. O caminho residente fica congelado (ask-6): a próxima etapa é o B0.

## Consulta ask-6 e decisão do Breno: o que vem depois das fusões (29/10)

Respostas em `.herdr/ask/strata-opencode-6/`. Os dois consultores, cegos, convergiram na decisão e nas premissas:

- **O custo fixo do caminho residente para no M2f.** Agrupamento dos k experts, top-k na GPU e grafo ficam para depois. A
  diferença que sobra para o llama.cpp é trabalho de GPU e não decide nenhuma das duas teses. As políticas se comparam pela
  espera além do residente, e congelar o residente deixa os parâmetros do modelo de custo parados. Premissa que reabre isso: a
  razão host/GPU do `m2b` passar de ~80%, ou o simulador mostrar que a ordem entre políticas muda com o tempo de GPU.
- **O agrupamento vem só junto do prefetch completo.** Com ids de slot lidos da GPU, o kernel toma emprestado o tensor de
  slots inteiro, e a garantia por slot (a tese 1) cai para "por tensor". A granularidade do lease (por slot ou por bloco de
  slots) se decide no B0. O limite de 2³¹ elementos pede blocos de slots em experts grandes (OLMoE, ~4 M elementos).
- **Aviso sobre a tese 1:** `async_on` solta o empréstimo ao enfileirar, então o compilador sozinho não impede despejar um
  slot que a GPU ainda lê. O resultado realista é "API segura sobre um núcleo `unsafe` fino, com o lease preso ao evento da
  GPU". O texto do posicionamento se ajusta ao que o B0 mostrar.

Ordem aprovada pelo Breno (29/10, 21:07):

1. **Fechar o M2f** (medida, testes, plano). Registrar a razão host/GPU e os parâmetros do modelo de custo na velocidade
   final: ms por cópia de expert, ms de CPU por expert, ms de GPU por token.
2. **B0, tese 1 no protocolo síncrono de hoje, sem mudar o comportamento:** estado do slot em tipos que mudam de dono (livre,
   carregando, residente, em voo preso ao evento do kernel); testes de compilação que falham (`trybuild`: despejar slot em
   voo, lançar sobre slot carregando, reusar slot movido, reaproveitar staging antes do evento); `compute-sanitizer`
   `racecheck` e `memcheck` no `lru_cpu` com a janela de corrida alargada (experts grandes ou kernel de espera) e um controle
   negativo (mutação sem a espera do evento, que tem de ser acusada). O desenho dos estados vai para o plano antes do código.
   Encerra com: os casos de compile-fail reprovando pelo motivo certo; 0 conflitos na versão tipada e a mutação acusada; o
   `m2a` sem piorar além do ruído, pelo critério abaixo; o `unsafe` restante listado por módulo e motivo; uma frase de
   veredito sobre a tese 1. Critério de não regressão: o da seção "B0 — desenho revisado depois da consulta ask-7"
   (modo `Barrier` bit a bit em todas as políticas; residente e `lru_sync` ≤ 1%; `lru_cpu` do 3B pela mediana de 5;
   ≤ 3 µs por camada de host). O `racecheck` saiu: ele não enxerga esta corrida.
3. **Recalibrar o simulador** com o custo do nosso caminho de CPU (sem GPU) e refazer `custo` contra `lru_cpu`/`lfu_cpu`. Se
   o ganho cair abaixo de 5%, avisar antes de implementar.
4. **C (M3), política `custo` no protocolo síncrono:** traços de treino e de teste separados, execuções intercaladas com
   núcleos fixos (3 a 7 por configuração, conforme o ruído), taxa de acerto e cópias publicadas junto de cada tempo. Encerra
   com ≥ 5% de redução da espera sobre `lru_cpu` em pelo menos um modelo e um orçamento, com intervalos sem sobreposição, ou
   com o resultado negativo e a causa.
5. **B completo** (prefetch sem a sincronização por camada) junto com o agrupamento e o top-k na GPU, usando a decisão de
   granularidade do B0.

## B0 — desenho proposto (para revisão antes do código, 29/10)

Hoje a segurança do cache vem do protocolo descrito em `ExpertCache::add_into`, não dos tipos. `Slot` tem `key`, `loading:
bool` e o evento `ready`; `copy_into` é `unsafe` e a garantia está no comentário ("nenhuma leitura pendente roda depois da
cópia"). O B0 passa cada passo do protocolo para um tipo, sem mudar a ordem das operações nem os números.

**Estados de um slot** (enum guardado no `Vec`, transições por métodos que consomem o estado anterior):

| estado | o que pode | como entra | como sai |
|---|---|---|---|
| `Free` | ser destino de cópia | início; `clear` | `begin_copy(ordem)` → `Loading` |
| `Loading { ready: Event }` | nada (não lê, não despeja) | `begin_copy` | `settle` com `ready.query()` verdadeiro → `Resident` |
| `Resident { key }` | ser lido por um kernel (`&Resident`); ser despejado | `settle`; cópia na stream de cálculo | `evict(ordem)` → `Free` (e em seguida `begin_copy`) |

**O que os tipos passam a garantir (erro de compilação):**

1. `enqueue_expert` recebe `&ResidentSlot`, que só existe dentro de `Resident`. Lançar sobre um slot carregando não compila.
2. Despejar ou copiar para um slot pede um `Ordering`, que só sai de dois construtores: `Ordering::same_stream(&compute)` (a
   cópia vai para a stream de cálculo, depois das leituras já enfileiradas; caso `lru_sync`) e `Ordering::after(&copy,
   &order_event)`, que grava o evento no cálculo e faz a stream de cópia esperá-lo (caso `lru_cpu`). Sem `Ordering`, não
   compila; o `Ordering` não é `Clone` e é consumido pela cópia.
3. `begin_copy` consome o `Free` e devolve o `Loading`: reusar o slot antigo depois de mandá-lo para a cópia não compila (E0382).
4. Os tensores com a saída da CPU (`keep`) só são soltos com um `ComputeIdle`, que só a sincronização da stream de cálculo
   produz (hoje: a leitura dos logits do roteador). Soltar antes não compila.

**O que fica no núcleo `unsafe`** (um módulo `cache::core`, listado no plano por função e motivo): os dois `memcpy_htod_async`,
a gravação e a espera dos eventos dentro de `Ordering::after`, e o `ComputeIdle` produzido depois de uma sincronização real.
Esse núcleo é o que os tipos não provam; é ali que entram o `racecheck` e o controle negativo.

**O que não muda:** a escolha das vítimas continua com a stream de cálculo parada, os experts pedidos na camada continuam
protegidos pelo `last_use`, e as estatísticas e a ordem das operações ficam iguais. O `m2a` antes e depois segue o critério de não
regressão da ordem aprovada (passo 2): bit a bit e ≤ 2% onde não há admissão em segundo plano (residente, `cache_full`,
`lru_sync`), mediana de 5 execuções no `lru_cpu`.

**Testes:**

- `trybuild` (4 casos que têm de reprovar pelo motivo certo): lançar sobre `Loading`; despejar sem `Ordering`; reusar slot
  movido para a cópia; soltar `keep` sem `ComputeIdle`. Cada caso com a mensagem de erro conferida, e uma mutação que faz o
  caso compilar (como no spike P1) para provar que o teste enxerga o erro.
- `compute-sanitizer --tool racecheck` e `memcheck` no `lru_cpu` com backlog cheio e a janela de corrida alargada: experts
  sintéticos grandes (dezenas de MB, poucos slots), nos quais a cópia fica em voo enquanto o cálculo roda.
- Controle negativo: uma versão com `Ordering::after` sem a espera do evento (mutação no núcleo). Ela tem de ser acusada pelo
  `racecheck` ou por um slot "envenenado" (preenchido com NaN no despejo, logits comparados bit a bit ao residente) em 5 de 5
  execuções.

**Como o Strata e o llama.cpp fazem** (lido em 29/10, só leitura, `strata` `9259cad4` e `llama.cpp` `0893ec9`):

- **llama.cpp, `--moe-cache-mib`** (experimental, `src/llama-moe-cache.cpp`): um banco de slots por tensor de expert
  (gate, up, down), compartilhado por camadas com o mesmo layout, e um `slot_map` por camada. A vítima sai de um SLRU
  (probation e protected, protected limitado à metade) e pula os experts que o top-k atual já escolheu (`victim`, linhas
  89–100): é o nosso `last_use`. A segurança é sincronização total: o callback lê os ids, chama `ggml_backend_synchronize`
  (566–567) e faz as cópias na própria stream de cálculo, antes do `mul_mat_id`. Não há stream de cópia nem evento: é o
  nosso `lru_sync`. O nosso `lru_cpu` (cópia em outra stream, experts frios na CPU) já vai além do que ele faz.
- **Strata, tier de VRAM** (`src/program/generate.cpp:5689–5756`): não há LRU. Os slots começam de um perfil
  (`profile.bin`) e mudam por trocas entre rodadas: LFU com decaimento (`×0,7` a cada adapt), troca só se o candidato
  passar a vítima por 1,5, no máximo 96 trocas a cada 4 rodadas. O protocolo é o mesmo do nosso `lru_cpu`, com outros nomes:
  despejo lógico na hora (`host_res[out] = -1`, a CPU passa a calcular o expert), cópia numa stream própria com evento
  gravado depois, e admissão preguiçosa no começo da janela seguinte por `cudaEventQuery` (`apply_pending`), que é o nosso
  `settle`. A garantia de que nenhum kernel ainda lê o slot vem de graça da sincronização no fim de cada janela, como a nossa
  vem da leitura dos logits do roteador.
- **Leitura para o B0:** os estados propostos casam com os dois (`pending` do Strata = `Loading`; `host_res` = tabela de
  `Resident`). Nenhum dos dois põe o protocolo em tipos: o que o B0 acrescenta é exatamente isso, não um protocolo novo.
- **Ponto que o Strata deixa aberto e vale para o B completo** (lido, não executado): durante uma troca pendente, a cópia da
  tabela que mora na GPU (`d_res`) ainda aponta o slot em reescrita; só os caminhos que leem a tabela na GPU
  (`moe_hit_select`, plano na GPU) correriam esse risco. O comentário da cópia (`generate.cpp:5733`) diz que "a janela
  seguinte espera" as cópias, mas o código da janela seguinte só consulta o evento (`apply_pending(false)`), sem
  `cudaStreamWaitEvent`. Não conferi se o `moe_hit_select` roda durante uma troca; o mesmo bloco está repetido em
  `generate.cpp:4139–4218`. Quando os ids de slot forem lidos pela GPU, a tabela da GPU só pode mudar depois do `ready` e em ordem na
  stream de cálculo; isso entra no lease por bloco.
- **llama.cpp, kernel com ids** (`ggml/src/ggml-cuda/mmvq.cu:623–664`): no decode, os ids do top-k ficam num tensor na GPU
  e cada bloco lê `ids[blockIdx.y]` para indexar o expert num tensor 3D `[K, N, n_expert]`, com gate, up e GLU fundidos. Isso
  confirma a premissa da ask-6: agrupar com ids na GPU empresta o banco inteiro.
- **Para o C:** o SLRU do llama.cpp e o LFU com decaimento e histerese do Strata entram como linhas de base, além do
  `lru_cpu`. A regra do Strata para os misses (os últimos `m` vão à GPU pela PCIe, com `m` vindo da banda medida por
  `probe_pcie_h2d_gbps`; `src/core/expert_source.cpp:1613–1690`) é o protótipo mais simples de política por custo, e a nossa
  `custo` tem de ganhar dela também. Nenhum dos dois mede tempo de CPU por expert em execução.
- **Para a comparação do M2:** o `--moe-cache-mib` é uma linha de base mais justa para o nosso cache do que o `-ncmoe`
  (cache contra cache, em vez de camadas inteiras na CPU). O `llama-cli` local (build de 29/10, `0893ec9`) aceita
  `--moe-cache-mib`; o `llama-bench` não. Medir com o mesmo orçamento de VRAM, junto com `--cpu-moe`, quando o caminho de
  cache voltar a ser o foco (C).

**Perguntas para a revisão do desenho:**

- A granularidade é por slot. O agrupamento futuro (B completo) com ids de slot na GPU pede um lease por bloco de slots;
  vale já desenhar o `Ordering` por bloco, ou fica para o B completo?
- `ComputeIdle` como token de sincronização é suficiente, ou o `keep` deveria ser preso a um evento próprio (como o `ready`)?
- O Strata e o llama.cpp chegam ao mesmo protocolo sem tipos (acima). Alguma escolha deles muda este desenho: o despejo
  lógico imediato do Strata (a CPU calcula o expert enquanto a cópia anda), o banco por tensor do llama.cpp, ou o SLRU que
  pula o top-k atual em vez de um `last_use` por camada?
- O enum no `Vec` deixa casos que são checados em tempo de execução (procurar o slot de uma chave e encontrá-lo `Loading`).
  Isso não causa comportamento indefinido, mas não é erro de compilação. O texto da tese tem de separar as duas coisas.

### B0 — desenho revisado depois da consulta ask-7 (29/10; substitui o acima onde diverge)

Respostas em `.herdr/ask/strata-opencode-7/`. Os dois consultores, cegos, aprovaram o tamanho do B0 e a ordem (B0 antes do
C) e convergiram nas correções abaixo. Onde divergiram, a divergência vinha de premissas diferentes, resolvidas pelos fatos.

**Correções em que os dois concordam:**

- **`racecheck` sai da lista de evidências.** Ele só vê memória compartilhada de um bloco (conferido no
  `compute-sanitizer` local: "racecheck: Shared memory hazard checking"; o `--racecheck-memcpy-async` trata das cópias
  para a memória compartilhada dentro do kernel). A corrida da tese 1 é em memória global, entre uma cópia H2D e um kernel
  de outra stream. O `memcheck` fica, para acesso inválido e uso depois de liberar.
- **"Em voo" por slot não entra no B0.** Hoje o evento gravado no `begin_copy` cobre todas as leituras já enfileiradas, e o
  estado `Loading` impede leituras novas. O B completo troca esse evento global por um por slot (`last_read`); o B0 só
  garante que essa troca fica dentro do núcleo.
- **Granularidade por slot**, com `SlotId` opaco (newtype, para virar `(banco, índice)` no B completo sem tocar nos
  chamadores) e o núcleo sem `&Tensor` público, só o `ResidentSlot` que o `enqueue_expert` consome.
- **Modo de admissão determinístico, só para teste e reprodução** (`AdmitMode::Barrier`, padrão desligado): sincroniza a
  stream de cópia antes do `settle`. Com ele, antes e depois do B0 dão os mesmos acertos, as mesmas cópias e logits bit a
  bit também no `lru_cpu`, sem depender da mediana de 5. Admissão contada em tokens como política fica para o C. Entra
  primeiro, num commit separado, e o `antes` do B0 é medido com ele.
- **Custo de host medido no próprio módulo:** ≤ 3 µs por camada nas transições, eventos e verificações (≤ 0,1 ms por token
  a 32 camadas). É o tipo de custo que uma API tipada introduz e que o tempo de parede não enxerga.
- **`unsafe` só no núcleo**, com um teste que falha se aparecer `unsafe` fora dele (`#![forbid(unsafe_code)]` nos módulos de
  fora) e a lista dos blocos do núcleo com o motivo de cada um (hoje há 16 ocorrências em `cache.rs`).

**Correções de um só, adotadas:**

- **`CopyMode` no lugar de `Ordering` (rev-2).** Um `Ordering` construído antes e consumido depois deixa um buraco: um
  lançamento feito entre os dois fica fora do evento. O `begin_copy(slot, key, CopyMode::SameStream |
  CopyMode::AfterCompute { copy })` grava o evento ele mesmo, e o evento deixa de aparecer na API.
- **`Pins` (rev-2) com conferência de geração (rev-1).** A regra "não despejar o que a camada pediu" é a que o runtime já
  violou (M2a, rodada 1). `let pins = cache.pin_layer(&routing)`, e a escolha de vítima só compila com `&Pins` na mão;
  `Pins` guarda o relógio da camada e recusa um `Pins` velho (verificação em tempo de execução). Em debug, cada
  `Resident` leva uma geração, e no fim da camada nenhum slot lançado pode ter mudado de geração.
- **Mapa só de `Resident` e `pending` à parte (rev-1, copiado do Strata).** Hoje `slot_of[key]` recebe a chave nova já no
  `copy_into`, e um acerto em slot carregando cai num ramo de tempo de execução (`Some(_)` → CPU). Como no `apply_pending`
  do Strata, a chave nova só entra no mapa depois do `ready`; até lá é uma falta com admissão pendente, e o ramo some.
- **A escolha de vítima e o `release_uploads` (o `keep`) pedem `&ComputeIdle` (rev-1).** É a forma barata de tipar "a vítima
  é escolhida com a stream parada". A versão mais forte do rev-1 (um `Epoch` em que lançar com um `Idle` vivo não
  compila) fica como tentativa: se pesar com o cutile, cai para o `ComputeIdle` por valor.

**Divergência resolvida pelos fatos: como fazer o controle negativo.** O rev-2 propôs um teste diferencial com experts
grandes, em que a build sem a espera do evento tem de divergir da referência. O rev-1 notou que, no protocolo de hoje, a
stream de cálculo está parada quando a vítima é escolhida (leitura dos logits do roteador), então o evento já está completo
quando é gravado e tirar a espera não muda nada observável: o mutante passaria em silêncio, com experts de qualquer tamanho.
O fato decide pelo rev-1 (é o que o "O que não muda" acima diz). Logo, o controle negativo tem de forçar a janela fora do
protocolo: um teste do núcleo na GPU que enfileira na stream de cálculo um kernel de espera lendo o slot e depois a cópia
pelo `begin_copy`, com e sem a espera. O critério de repetição é o do rev-2 (≥ 19 de 20, e não 5 de 5, que deixa o limite
inferior de 95% em ~48%).

**Não adotado no B0, registrado:** o teste de propriedade com backend simulado do rev-1 (algumas centenas de linhas; os
mutantes de evento e de `Pins` já são cobertos pelo teste de janela forçada, pelo teste de LRU de referência que pegou o
defeito da rodada 1 e pelo modo `Barrier`). Volta se o veredito sair fraco.

**Testes revisados:**

1. `trybuild`, 5 casos, cada um com o gêmeo corrigido como caso que compila (controle positivo) e o `rustc` fixado (a
   mensagem faz parte do teste): lançar sobre `Loading`; reusar o slot movido para a cópia (E0382); escolher vítima sem
   `&Pins`; escolher vítima sem `&ComputeIdle`; soltar os uploads sem `&ComputeIdle`.
2. Teste do núcleo na GPU com janela forçada (acima): versão tipada bit a bit em 20 de 20; sem a espera, acusada em ≥ 19 de
   20; `memcheck` com 0 erros nele e na suíte de modelo.
3. Modo `Barrier`: acertos, cópias, experts na CPU e logits iguais bit a bit antes e depois, em residente, `cache_full`,
   `lru_sync` e `lru_cpu`, nos dois modelos, a 25% e 50%.
4. Tempo: residente e `lru_sync` ≤ 1% (variam 0,3%); `lru_cpu` sem o `Barrier` só no 3B, mediana de 5 intercaladas dentro
   da faixa de antes, com o modo lento publicado; ≤ 3 µs por camada de host no módulo.
5. `unsafe`: 0 fora do núcleo, com teste; lista do núcleo por bloco e motivo.

**O que a tese 1 vai afirmar, em três camadas:** estática (ordem das operações, slot movido, estado `Loading`, `Pins`,
`ComputeIdle`), verificada em tempo de execução (o `Pins` velho, a geração, o estado de um slot achado no `Vec`) e núcleo
`unsafe` (as cópias H2D, a gravação e espera dos eventos, o `ComputeIdle` produzido por uma sincronização real). Com a
nota de que a prova dinâmica da espera do evento só fica completa no B completo, quando a espera passar a ser necessária.

**Ordem de implementação:** (1) `AdmitMode::Barrier` em commit separado e o `antes` medido com ele; (2) núcleo, `SlotId`
e o mapa só de `Resident`, sem mudar o comportamento; (3) `begin_copy` com `CopyMode`; (4) `Pins`; (5) `ComputeIdle` na
escolha de vítima e nos uploads; (6) os testes; (7) as medidas e o veredito.
