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

### B0 — implementação (29/10)

Material em `qual-runs/B0-SEGURANCA-20261029/` (binários `antes`, `depois` e `final` com sha256, fontes com sha256,
saídas do `cargo test`, do `memcheck` (com o comando e a versão do `compute-sanitizer` no topo) e do `m2a`, scripts
`rodar-antes.sh`, `rodar-depois.sh` e `rodar-final.sh`). `depois` é o código revisado na rodada 33; `final` é o mesmo com
as correções dela (abaixo).

**O que mudou no código.** `src/cache.rs` ficou com a política (LRU, `LruSync`/`LruCpu`, backlog, contadores) e
`#![deny(unsafe_code)]` (não `forbid`, como o desenho dizia: o `forbid` não deixaria o `#[allow(unsafe_code)]` do
submódulo; um teste segura o resto); o protocolo foi para `src/cache/core.rs`, o único módulo do cache com
`#[allow(unsafe_code)]`:

- `Slots`: estados `Free`, `Reserved`, `Loading{key, last_use}` e `Resident{key, last_use}`, com geração por slot. O mapa
  chave → slot só lista `Resident`; as cópias em segundo plano ficam em `pending` até o evento terminar (`settle`), como no
  `apply_pending` do Strata. `Lookup::{Hit(SlotId), Loading, Miss}` substitui o ramo `Some(_)` do slot carregando.
- `pin_layer(&keys) -> Pins` marca os experts pedidos na camada; `victim(&Pins, &Idle)` recusa um `Pins` de outra camada
  (tempo de execução) e devolve um `Victim` (`#[must_use]`, sem `Clone`) que a cópia consome. A cópia confere que a
  vítima ainda está reservada na mesma geração (um `clear` no meio a invalida) e que a chave é falta.
- `Slots` e `Epoch` só nascem juntos (`core::new_pair`), na mesma stream de cálculo e com um id de instância que `Pins`,
  `Idle`, `SlotId`, `Victim` e `ResidentSlot` carregam; usar um deles noutra instância dá erro. Sem isso (rodada 33), dois
  objetos criados com streams diferentes deixavam a cópia na stream de cálculo correr contra os lançamentos por uma API
  segura.
- As duas formas de cópia são métodos que gravam o próprio evento: `copy_on_compute` (na stream de cálculo, slot
  `Resident` na hora) e `copy_in_background` (grava o evento de ordem no cálculo, a stream de cópia espera, slot `Loading`).
  O `CopyMode` do desenho virou esses dois métodos; o evento não aparece na API.
- `resident(SlotId) -> ResidentSlot` confere estado e geração; `Epoch::launch_expert` só aceita `ResidentSlot`.
- `Epoch::router_logits` devolve os logits e um `Idle<'_>`, que toma a `Epoch` emprestada (`&mut`): enquanto ele vive não
  há lançamento, e só ele escolhe vítima, solta os uploads da CPU (`release_uploads`) e lê `x` para a CPU. A versão forte
  do rev-1 coube no cutile sem o recuo para `ComputeIdle` por valor.
- O `ExpertCache` fica inutilizável depois de um erro (flag `poisoned`, as duas streams sincronizadas), como antes.

Duas diferenças de comportamento apareceram na releitura linha a linha contra o código antigo e foram corrigidas antes de
compilar: o código antigo atualizava o `last_use` de um slot carregando quando a chave era pedida (conta para o LRU depois
do `settle`; o `pin_layer` faz o mesmo agora), e o `Drop` do `ExpertCache` sincroniza as streams antes de os campos caírem.

**Testes novos (6; a suíte passou de 38 para 44, todos verdes e sem warnings, `cargo-test-final.txt`; os números dos
testes do núcleo estão em `nucleo-nocapture-final.txt`):**

- `tests/tipos.rs`, `trybuild`: os 6 casos de `tests/ui/` falham pelo motivo esperado e os 6 gêmeos de `tests/ui-pass/`
  compilam. Criar `Slots` sem o par (rodada 34): E0624, construtor privado; o gêmeo usa `new_pair`. Lançar um slot carregando: E0308 (`LoadingSlot` onde se pede `ResidentSlot`). Reusar a vítima: E0382. Vítima
  sem `pin_layer`: "cannot construct `Pins` ... due to private fields" (era E0451 com um campo só; com o id de
instância o rustc muda a mensagem, mesmo motivo). Vítima depois de um lançamento: E0499 (o `Idle` ainda empresta a
  `Epoch`). Soltar os uploads depois de enfileirar a saída da CPU: E0499. Mensagens geradas com rustc 1.97.0.
- `tests/tipos.rs`: falha se os atributos com `unsafe_code` em `cache.rs` não forem exatamente o `#![deny(unsafe_code)]` e
  o `#[allow(unsafe_code)]` do `pub mod core` (pega também `allow(dead_code, unsafe_code)` e `expect(unsafe_code)`).
- `cache::core::tests::copia_em_segundo_plano_espera_a_leitura_do_slot`, a janela forçada: um callback de host
  (`cuLaunchHostFunc`) bloqueia a stream de cálculo, um expert que lê o slot entra na fila atrás dele, e só então vem a
  cópia de outro expert para o mesmo slot. Pela `copy_in_background`, a saída é bit a bit a do expert antigo em **20 de
  20**; pelo mutante sem a espera do evento, é bit a bit a do expert novo em **20 de 20** (o teste exige 20: a cópia do
  mutante é esperada antes de o portão abrir, então o resultado não depende de tempo). Sem kernel de espera. A primeira
  versão travou: o `h0` era criado com `sync_on` na stream bloqueada e o portão só abria pelo prazo de 10 s; agora o `h0`
  vem antes do portão e o `send` falha com essa explicação.
- `cache::core::tests::slot_carregando_fica_fora_do_mapa_e_ganha_o_uso_da_camada` (rodada 33): portão na stream de cópia,
  então a cópia em segundo plano fica presa e o slot `Loading` de verdade, estado que a barreira nunca alcança (ela drena a
  fila antes do `pin_layer`). Confere que a chave dá `Lookup::Loading` e fica fora do mapa, que `settle(false)` não promove
  antes do evento, e que um slot carregando pedido numa camada ganha o uso dela: depois do `settle`, a vítima é o outro
  slot. O controle negativo (sem o pedido) escolhe o slot carregando. A primeira versão passou sozinha e falhou na suíte em
  paralelo e no `memcheck`, com o portão aberto pelo prazo: o `router_logits` dentro da janela aloca e sincroniza o device,
  inclusive a stream presa; agora ele vem antes de fechar o portão.
- `cache::core::tests::conferencias_em_tempo_de_execucao` (rodada 33): `Pins` de outra camada, `SlotId` de um slot
  sobrescrito, vítima reservada antes de um `clear`, cópia de chave já residente, e `Pins`, `Idle`, `SlotId` e
  `ResidentSlot` de outra instância: todos dão erro.
- Mutantes (`mutantes.sh`, saída em `mutantes-nucleo.txt`, com o trecho trocado e a linha `FAILED` de cada um; o script
  guarda o `core.rs`, aplica um de cada vez e confere o sha256 ao restaurar): os 7 são reprovados pelo teste esperado.
  `pin_layer` sem o laço dos slots carregando e `settle` sem a consulta do evento (teste do slot carregando);
  `copy_in_background` sem gravar e esperar o evento de ordem (janela forçada, agora pelo caminho real e não por um
  mutante escrito no teste); `resident` sem a geração, `victim` sem o relógio do `Pins`, `victim` sem conferir a
  instância e a cópia sem conferir a vítima (conferências).
- `cache::core::tests::protocolo_custa_pouco_no_host`: `settle` com 4 cópias prontas na fila (as `query()` dos eventos),
  `pin_layer`, 8 consultas e a conferência de estado, geração e instância dos acertos custam **0,35 µs por camada** (64
  slots; 0,56 sob o `memcheck`; critério ≤ 3 µs). A primeira versão, sem eventos na fila, media só a contabilidade (0,028
  µs). Escolha de vítima e cópias ficam de fora (custam o mesmo que antes do B0); o tempo de parede abaixo as cobre.
- `memcheck` (`compute-sanitizer` 2026.2.1, comando no topo de `memcheck-nucleo-final.txt` e `memcheck-model-final.txt`): 0
  erros nos 4 testes do núcleo e na suíte de modelo (19 testes).

**`unsafe` no núcleo fora dos testes, por função e motivo (15 ocorrências; `cache.rs` tinha 17 linhas com `unsafe`, 16
antes do `synchronize` do modo `Barrier`, e agora tem 0):**

| Função | Quantas | O que | Por que é seguro |
|---|---|---|---|
| `Slots::weights` | 1 | `u16` pinned lido como `bf16` | `bf16` é `repr(transparent)` sobre `u16` |
| `Slots::clear`, `Slots::settle` | 2 | `synchronize` da stream de cópia | só espera |
| `copy_on_compute`, `copy_in_background` | 2 | chamam `copy_raw` | mesma stream das leituras, ou espera do evento gravado ali mesmo |
| `Slots::copy_raw` | 2 | `unsafe fn` e as duas cópias H2D assíncronas | tamanhos conferidos em `new`; o `Drop` sincroniza antes de soltar o buffer pinned |
| `Slots::sync_all` | 1 | `synchronize` das duas streams | só espera |
| `Epoch::router_logits` | 1 | GEMV do roteador | formas múltiplas dos blocos (conferido em `ExpertCache::new`) |
| `Epoch::launch_expert` | 1 | `enqueue_expert` | slot `Resident` (o `ResidentSlot` prova); formas conferidas |
| `Epoch::finish_gpu`, `Epoch::add_cpu_out` | 3 | `axpby` e o upload da saída da CPU | upload retido em `keep` até o próximo `Idle` |
| `Epoch::sync` | 1 | `synchronize` | só espera |
| `Idle::read_x` | 1 | cópia D2H de `x` e `synchronize` | `x` vive até a espera |

**Medida 1, admissão em barreira (critério bit a bit).** O `antes` rodou duas vezes por modelo e as duas deram as mesmas
contagens exatas e o mesmo hash dos logits em todas as políticas, inclusive `lru_cpu` (o modo `Barrier` serve de referência).
O `depois` rodou uma vez por modelo: acertos, cópias síncronas, cópias em segundo plano, descartes, experts na CPU e hash
dos logits **iguais aos do antes** em residente, `cache_full`, `lru_sync` e `lru_cpu`, a 25% e 50%, no 1B e no 3B. Logits
dentro da tolerância contra o transformers em todas as configurações. Referência do 3B:

| política | orçam. | acertos | cóp.s | cóp.bg | descart. | CPU | hash |
|---|---|---|---|---|---|---|---|
| lru_sync | 25% | 465777 | 510607 | 0 | 0 | 0 | 162ca54cd7716a68 |
| lru_sync | 50% | 778193 | 198191 | 0 | 0 | 0 | 162ca54cd7716a68 |
| lru_cpu | 25% | 439404 | 0 | 414057 | 122923 | 536980 | 1a68c295e433deef |
| lru_cpu | 50% | 778977 | 0 | 188904 | 8503 | 197407 | deddc039191cc684 |

**Medida 2, tempo com a admissão por evento (o padrão),** 5 pares antes/depois intercalados por modelo (`tempo-evento.txt`),
ms/token, média nas conversas:

| modelo | política | antes (5) | depois (5) | mediana | critério |
|---|---|---|---|---|---|
| 1B | residente | 3,02 3,04 3,01 3,04 3,01 | 3,01 3,01 3,01 3,01 3,03 | 3,02 → 3,01 (−0,3%) | ≤ 1% ✓ |
| 1B | lru_sync 25% | 18,94 18,68 18,62 18,69 18,66 | 18,69 18,63 18,80 18,60 19,14 | 18,68 → 18,69 (+0,1%) | ≤ 1% ✓ |
| 1B | lru_sync 50% | 11,75 11,66 11,61 11,66 11,63 | 11,68 11,61 11,69 11,58 11,90 | 11,66 → 11,68 (+0,2%) | ≤ 1% ✓ |
| 3B | residente | 4,84 4,88 4,83 4,86 4,94 | 4,87 4,84 4,84 4,83 5,06 | 4,86 → 4,84 (−0,4%) | ≤ 1% ✓ |
| 3B | lru_sync 25% | 36,05 36,08 35,92 36,06 36,31 | 36,21 36,00 36,03 36,05 36,21 | 36,06 → 36,05 (−0,0%) | ≤ 1% ✓ |
| 3B | lru_sync 50% | 17,02 17,02 16,98 16,98 17,17 | 17,03 16,99 17,04 17,00 17,02 | 17,02 → 17,02 (+0,0%) | ≤ 1% ✓ |
| 3B | lru_cpu 25% | 13,07 13,37 13,11 12,97 14,22 | 13,32 13,53 13,23 13,24 13,47 | 13,11 → 13,32 (+1,6%) | na faixa do antes (12,97–14,22) ✓ |
| 3B | lru_cpu 50% | 8,25 8,14 8,13 8,02 8,84 | 8,10 8,00 8,09 8,08 8,50 | 8,14 → 8,09 (−0,6%) | na faixa do antes (8,02–8,84) ✓ |

O quinto par saiu mais lento nos dois binários (residente 3B: 4,94 e 5,06), e o antes 5 teve o modo lento do `lru_cpu`
(14,22 a 25%). A faixa do antes que o critério usa inclui esse par; sem ele, ela é 12,97–13,37 e dois dos cinco depois
(13,47 e 13,53) ficam acima (rodada 33). No `lru_cpu` a 25%, o depois ficou acima do antes em 4 dos 5 pares, com a
diferença quase toda no tempo dos experts na CPU.

**Medida 3, o viés do `lru_cpu` com trabalho idêntico (rodada 33, `rodar-final.sh`, `tempo-barreira-lrucpu.txt`).** Em
barreira, antes e depois fazem exatamente o mesmo trabalho (as 6 execuções de cada modelo dão a mesma tabela de contagens),
então uma diferença de tempo seria custo do código e não da admissão. 3 pares antes/final intercalados, ms/token:

| modelo | orçam. | antes (3) | final (3) | mediana | experts na CPU, ms/token |
|---|---|---|---|---|---|
| 1B | 25% | 15,51 15,92 15,73 | 15,99 15,69 15,69 | 15,73 → 15,69 (−0,3%) | 5,35 → 5,32 |
| 1B | 50% | 9,55 9,65 9,68 | 9,87 9,62 9,55 | 9,65 → 9,62 (−0,3%) | 2,61 → 2,59 |
| 3B | 25% | 30,44 31,20 31,27 | 31,07 30,66 30,55 | 31,20 → 30,66 (−1,7%) | 10,80 → 10,27 |
| 3B | 50% | 15,16 15,32 15,40 | 15,42 15,21 15,17 | 15,32 → 15,21 (−0,7%) | 3,93 → 3,83 |

Com o mesmo trabalho, o final não saiu mais lento (de −0,3% a −1,7%), e o mesmo binário varia até 2,7% entre execuções
(3B a 25%: 30,44–31,27). Com n = 3 pares e o antes sempre primeiro, isso quer dizer "sem diferença detectável acima de
~2–3%", não "sem viés": juntando todas as execuções em barreira do 3B a 25% (antes 30,47 30,27 30,44 31,20 31,27; depois e
final 30,61 31,07 30,66 30,55), as medianas são 30,47 e 30,64 (+0,6%) (rodada 34). O que sustenta que o B0 não custa
tempo é o micro-teste de host abaixo, mais a igualdade bit a bit do trabalho. O custo do protocolo
no host é o do micro-teste: 0,35 µs por camada com 4 eventos consultados, ~11 µs por token no 3B.

**Medida 4, barreira com os binários finais:** contagens exatas e hash dos logits iguais aos do antes nas quatro políticas,
nos dois modelos (`m2a-{1b,3b}-final-barreira-1.txt`). As correções da rodada 33 não mudaram o comportamento.

**Medida 5, tempo com os binários finais (rodada 34, `rodar-final-tempo.sh`, `tempo-final.txt`).** O `final` acrescentou
as conferências de instância e da vítima ao caminho de cópia e de lançamento; os critérios da medida 2 foram refeitos
com ele: 5 pares antes/final intercalados, admissão por evento, ms/token, média nas conversas.

| modelo | política | antes (5) | final (5) | mediana | critério |
|---|---|---|---|---|---|
| 1B | residente | 3,07 3,02 3,06 3,03 3,03 | 3,04 3,03 3,05 3,03 3,04 | 3,03 → 3,04 (+0,3%) | ≤ 1% ✓ |
| 1B | lru_sync 25% | 18,80 18,79 18,88 18,87 18,90 | 18,82 18,95 18,82 18,91 19,28 | 18,87 → 18,91 (+0,2%) | ≤ 1% ✓ |
| 1B | lru_sync 50% | 11,75 11,77 11,68 11,72 11,82 | 11,75 11,79 11,71 11,87 11,71 | 11,75 → 11,75 (+0,0%) | ≤ 1% ✓ |
| 3B | residente | 4,89 4,90 4,85 4,83 5,00 | 4,85 4,89 4,84 4,90 4,90 | 4,89 → 4,89 (+0,0%) | ≤ 1% ✓ |
| 3B | lru_sync 25% | 36,38 36,53 36,51 36,49 36,41 | 36,28 36,25 36,47 36,58 36,64 | 36,49 → 36,47 (−0,1%) | ≤ 1% ✓ |
| 3B | lru_sync 50% | 17,29 17,14 17,29 17,13 17,28 | 17,20 17,23 17,17 17,53 17,46 | 17,28 → 17,23 (−0,3%) | ≤ 1% ✓ |

O 3,41 ms/token do residente 1B na única amostra anterior do final (`m2a-1b-final-barreira-1.txt`) não se repetiu: as 5
execuções do final ficaram em 3,03–3,05, iguais às do antes. O caminho residente nem passa pelo `ExpertCache`; foi ruído
daquela execução. Logits dentro da tolerância nas 20 saídas.

**Veredito da tese 1 no B0 (reescrito depois das rodadas 33 e 34).** Dentro do `ExpertCache`, os cinco erros de ordem de
uso e a criação de `Slots` e `Epoch` separados deixam de compilar; o estado, a geração, o relógio da camada e a instância são conferidos em tempo de execução, com teste
para cada conferência; e todo o `unsafe` do cache está em 15 blocos de um módulo. Em três camadas:

- **Estático (o compilador recusa):** lançar um slot que não seja `Resident`; reusar uma vítima já consumida; forjar o
  `Pins` (só `pin_layer` o cria); escolher vítima ou soltar os uploads da CPU depois de um lançamento na camada;
  criar `Slots` e `Epoch` separados (só `new_pair`, na mesma stream). Os 6 casos de `trybuild` falham com a mensagem
  esperada e os 6 gêmeos compilam. O `unsafe` fora do núcleo é recusado pelo `deny(unsafe_code)`, e o teste de texto
  impede que um `allow` o desligue.
- **Verificado em tempo de execução, com teste e mutante:** `Pins` de outra camada; geração e estado do slot ao lançar;
  vítima reservada antes de um `clear`; cópia de chave que não é falta; objetos de outra instância; slot carregando fora do
  mapa até o evento, e com o uso das camadas que o pedem.
- **Núcleo `unsafe`:** as cópias H2D, a gravação e espera dos eventos, os lançamentos e a sincronização que produz o
  `Idle` (tabela acima).

O que os tipos não provam, e o texto não deve dizer: o `Pins` prova que a camada foi fixada, não que a lista de chaves
estava completa (`pin_layer(&[])` compila); o que impede lançar um acerto que foi despejado é a geração conferida em
`resident`. O `Idle` prova que nada foi lançado desde a leitura dos logits, não que a stream está parada (em `LruSync` as
cópias da camada já estão na fila dela). Quem ordena cópia e kernel é a ordem das streams e o evento dentro do núcleo; os
tipos impedem usar esses dois mecanismos fora de ordem. O controle negativo do evento só existe no teste forçado, porque
no protocolo de hoje o `Idle` vem de uma sincronização e a espera nunca é exercitada; ela passa a ser necessária no B
completo, quando a escolha de vítima deixar de esperar o roteador, e aí a janela forçada vira o teste de regressão dela.

O comportamento não mudou (barreira bit a bit nos dois modelos, antes, depois e final). O tempo ficou dentro dos critérios
(medida 2, no `depois`; medida 5, no `final`) e sem diferença detectável com trabalho idêntico (medida 3).

**Rodada 33** (`.herdr/review/strata-opencode-33/`, completa; rev-1 aprova com ressalvas, rev-2 aprova; sem P0/P1). Todos
os achados procederam e foram corrigidos: os dois revisores acharam que `Slots` e `Epoch` com streams ou instâncias
diferentes passavam pela API segura (construtor único e id de instância); vítima sem geração e cópia de chave que não é
falta (conferidas); as linhas de `eprintln!` com os números não estavam em arquivo (`nucleo-nocapture-final.txt`);
`forbid`/`deny`, 16/17 e o teste de texto fraco (corrigidos); a faixa do `lru_cpu` esticada pelo par lento (medida 3). O
rev-1 notou que a barreira nunca exercita `Loading`/`pending` (teste do slot carregando) e que o mutante da janela
dependia de tempo (agora determinístico); o rev-2, que as conferências em tempo de execução não tinham teste e que `Pins`
e `Idle` provavam menos do que o texto dizia (veredito reescrito acima).

**Rodada 34** (`.herdr/review/strata-opencode-34/`, completa; rev-1 aprova com ressalvas, rev-2 aprova; sem P0/P1/P2). Os
dois conferiram, um a um, que os achados da rodada 33 estão resolvidos e que o veredito tem o alcance que as provas
sustentam. Ressalvas, todas tratadas: o caso de criar `Slots` sem o par não tinha `trybuild` (sexto caso) e os mutantes não
tinham registro (`mutantes.sh`, `mutantes-nucleo.txt`; os dois revisores); os critérios de tempo foram medidos no `depois`
e não no `final`, e o residente do 1B marcou 3,41 ms/token na única amostra do final, contra 3,09 e 3,03 do antes (rev-1;
medida 5); "sem viés" era forte demais para 3 pares (rev-2; texto da medida 3).

## Simulador recalibrado com o runtime (passo 3 da ordem aprovada, 30/10)

O M0.5 usava custos do llama.cpp e um protocolo que o runtime não segue. O passo 3 troca os dois e confere o simulador contra o
`m2a` medido antes de refazer a comparação das políticas. A primeira versão desta seção concluía que o `custo` acrescentava
até 4% sobre o `lfu_cpu`. A rodada 35 mostrou que isso dependia de uma escolha de modelagem que a validação não testava, e as
medidas com backlog 0 decidiram a escolha no sentido contrário (abaixo).

**Mudanças em `tools/moe_sim.py`** (o conjunto antigo continua disponível: `--params llamacpp --protocol m05` reproduz
`sim-1b.txt` e `sim-3b.txt`, iguais fora do cabeçalho, que ganhou campos):

- **Parâmetros do runtime** (`PARAMS_RUNTIME`, o padrão): `h` = os dois kernels de expert do M2c (1B 4,96 µs, 3B 5,19 µs);
  `g` = GPU por token do M2b ÷ camadas − 8·`h` (1B 84,9 µs, 3B 109,4 µs).
- **Protocolo do runtime** (`--protocol runtime`, o padrão): no `lru_cpu`, a chave em voo vai à CPU (o `ExpertCache` devolve
  `Lookup::Loading`), e a admissão é recusada com `backlog` cópias não assentadas, contadas como no runtime, em vez do tempo de
  fila do M0.5. O `backlog` virou parâmetro do `Sim` (padrão 4). O `custo` mantém a regra dele de esperar a cópia quando ela
  termina antes da CPU.
- **Camada com tráfego de cópia** = começa com cópia em voo (depois do `settle` do começo da camada, como o
  `loading_count() > 0` do runtime) ou enfileira uma. Contada em `stats["layers_traffic"]`. É ela que separa os três regimes
  medidos:
  - **CPU em dois níveis**: por expert, 1B 46,4 µs com tráfego e 40,1 µs sem; 3B 69,4 e 61,0 µs (a CPU e o DMA disputam a
    memória). Vêm do CPU ms medido por token contra os experts na CPU simulados em cada tipo de camada (`cpu_traffic`,
    `cpu_quiet`), por mínimos quadrados sem pesos nos 4 `lru_cpu` e nos 4 `lru_cpu` com backlog 0 de cada modelo, iterados até
    o ponto fixo (as contagens dependem do próprio `c_cpu`; 2 iterações). Resíduo máximo 0,22 e 0,32 ms/token; um custo fixo
    por camada com CPU, a outra explicação, deixa 0,70 e 0,99.
  - **`o_layer` = 36,7 µs** só nas camadas com tráfego e algum expert fora da GPU. É uma sobra medida, sem causa isolada:
    aparece no `lru_sync` e no `lru_cpu` (10–36 µs por camada com tráfego, quase sempre entre 24 e 36) e some no `lru_cpu`
    com backlog 0. Ajustado só no `lru_sync` do 1B (mínimos quadrados pela origem, 4 orçamentos). O tempo de despachar e
    juntar as threads da CPU já está no `c_cpu`; a primeira versão do comentário o contava também aqui.
  - **`pcie_gbs_cpu` = 14,1 GB/s**: com a CPU calculando experts, a cópia vai mais devagar. No `lru_cpu` a 10% do 1B a fila de
    admissão fica sempre cheia, então a vazão medida é a banda efetiva (10,13 ms/token ÷ 45,3 cópias = 0,224 ms por 3,15 MB).
    Vale para toda cópia em segundo plano e para a cópia síncrona do `custo` numa camada em que a CPU já trabalha. O 3B, fora
    do ajuste, dá 14,2 GB/s pela mesma conta.
- `--validar <saídas do m2a>` simula as condições de cada arquivo (cache vazio, conversas de teste 4–7, sem aquecimento, o
  backlog do cabeçalho) e compara. O cabeçalho é lido nos dois formatos (com e sem `; admissão X`); cabeçalho sem backlog ou
  com admissão por barreira é erro, e um arquivo sem nenhuma linha comparável reprova. `--cpu-scale` multiplica os dois custos
  de CPU, para a sensibilidade.

**Validação** (`qual-runs/TRACOS-20261029/validacao-{1b,3b}-runtime.txt`; critério combinado antes: acerto a ±2 pontos e
ms/token a ±10%). Dentro do critério nas 24 linhas, 12 por modelo. Desvio de ms/token (e de acerto, no `lru_cpu`):

| modelo | política | 10% | 25% | 50% | 75% |
|---|---|---|---|---|---|
| 1B | `lru_sync` | +0,0% | +0,1% | +0,1% | −1,9% |
| 1B | `lru_cpu` | +3,7% (−0,4 pt) | +2,0% (+0,4 pt) | −1,5% (+0,5 pt) | −2,5% (+0,1 pt) |
| 1B | `lru_cpu` backlog 0, 2 execuções | | +2,1% / +0,6% | +1,0% / −3,0% | |
| 3B | `lru_sync` | +1,3% | +0,5% | −0,4% | −1,8% |
| 3B | `lru_cpu` | +3,8% (+0,3 pt) | +1,2% (+0,2 pt) | 0,0% (+0,3 pt) | −2,9% (0,0 pt) |
| 3B | `lru_cpu` backlog 0, 2 execuções | | +0,1% / −0,1% | +2,7% / −0,6% | |

O caminho até aqui: só com os parâmetros novos, o tempo saía baixo, de −0,5 a −8% no `lru_sync` e até −16% no `lru_cpu`. O
`o_layer` em toda camada com CPU fechou as 16 linhas com backlog 4, mas previa +23% (1B) e +20% (3B) no backlog 0, onde não
há cópia nenhuma; com ele restrito às camadas com tráfego e a CPU em dois níveis, o backlog 0 fecha também. A banda disputada
fechou o `lru_cpu` a 10%, que saía com acerto baixo (1B: 3,3% contra 15,2%) porque, a 20 GB/s, o simulador assentava cópias
demais.

**Comparação das políticas** (`sim-{1b,3b}-runtime.{txt,json}`; sensibilidade com a CPU 15% mais rápida e 20% mais lenta em
`sim-*-runtime-cpu085.*` e `sim-*-runtime-cpu12.*`). Espera em ms/token, como no M0.5, com aquecimento nas conversas 0–3:

| modelo | orçamento | `lru_cpu` | `lfu_cpu` | `custo` | oráculo | `lfu_cpu` × `lru_cpu` | `custo` × `lru_cpu` | `custo` × `lfu_cpu` |
|---|---|---|---|---|---|---|---|---|
| 1B | 10% | 7,52 | 4,95 | 6,14 | 5,41 | −34,2% | −18,3% | +24,1% |
| 1B | 25% | 4,95 | 3,06 | 3,52 | 3,21 | −38,2% | −28,9% | +15,0% |
| 1B | 50% | 2,12 | 1,02 | 1,06 | 0,92 | −51,7% | −49,7% | +4,2% |
| 1B | 75% | 0,52 | 0,18 | 0,19 | 0,15 | −64,3% | −63,8% | +1,2% |
| 3B | 10% | 13,37 | 9,98 | 11,50 | 10,24 | −25,4% | −14,0% | +15,3% |
| 3B | 25% | 8,66 | 5,92 | 6,38 | 5,38 | −31,6% | −26,3% | +7,8% |
| 3B | 50% | 3,31 | 1,98 | 2,01 | 1,37 | −40,1% | −39,1% | +1,6% |
| 3B | 75% | 0,70 | 0,40 | 0,40 | 0,20 | −42,7% | −42,5% | +0,3% |

Fração das camadas com tráfego (contador `layers_traffic`, todas as camadas das conversas 4–7, com aquecimento), de 10% a
50%: `lru_cpu` 98–100%, `lfu_cpu` 6–12%, `custo` 14–86%, oráculo 68–98%.

- **Portão do passo 3 (`custo` ≥ 5% sobre o `lru_cpu`): passa no texto, mas não serve mais de guia.** O `custo` reduz a
  espera em 14–50% de 10% a 50%. Só que o `lfu_cpu`, que é o `custo` sem a cópia síncrona, reduz mais: 25–52% (+19 a +32% de
  tok/s). Com a CPU 15% mais rápida ou 20% mais lenta, a ordem é a mesma (`custo` × `lfu_cpu` de +0,8 a +24%).
- **Por que o `lfu_cpu` ganha:** acerta mais (1B 10%: 24,9% contra 14,8%; 25%: 50,1% contra 43,7%) e copia muito menos (1,5–3
  cópias por token contra 23–54 do `lru_cpu` de 10% a 50%), então só 6–12% das camadas têm tráfego. Nas outras, a CPU custa o
  nível baixo e não há `o_layer`.
- **Por que o `custo` perde para o `lfu_cpu`:** a cópia síncrona que ele escolhe quando "termina antes da CPU" põe tráfego em
  14–86% das camadas, e a regra de decisão dele não vê esse custo (a CPU disputada nas camadas seguintes e o `o_layer`).
- **O que isso sugere para o C:** implementar o despejo e a admissão do `lfu_cpu` (frequência com decaimento, admite só se
  vence o residente mais fraco), sem modelo de tempo no runtime, e medir contra o `lru_cpu`. A cópia síncrona por custo não
  entra como está; se voltar, a regra dela precisa contar o tráfego que cria.
- **O oráculo não é mais limite:** no 1B a 10% e 25% e no 3B a 10%, o `lfu_cpu` fica abaixo dele, porque o oráculo admite
  pelo futuro e copia muito (tráfego em 68–98% das camadas; o simulador não tem oráculo de colocação que conte o custo do
  tráfego). A distância até ele deixa de medir o que o prefetch pode ganhar.

**Medida com backlog 1 e 2** (`qual-runs/BACKLOG-INTERMEDIARIO-20261030/`, aprovada pelo Breno em 30/10; `lru_cpu`, backlog
0, 1, 2 e 4 intercalados, 2 repetições, 10/25/50%, os dois modelos; mesmo binário das referências; análise em `analise.py` e
`analise.txt`). Previsões e critério estão em `previsao.txt`, gravado às 10:34:17 pelo mtime (o cabeçalho diz 10:34:05, a hora
em que o script começou). Nesse instante só a 1ª execução de controle, 1B backlog 0, tinha terminado (10:34:09), e eu não a
tinha lido; a 1ª com backlog 1 terminou às 10:36:04. Os metadados confirmam o registro antes de qualquer resultado com
backlog 1 ou 2, não antes do primeiro controle. Ela testa a metade da hipótese que a fração de camadas não testa: se uma cópia em voo pesa como quatro.

- **O ambiente mudou desde as referências.** O driver da NVIDIA foi atualizado às 03:22 de 30/10 sem recarregar o módulo
  (módulo 610.57, `libcuda` 615.78; o CUDA funciona, o NVML não), e a máquina tinha carga de fundo (média de carga ~5 antes da
  medida). Nesta sessão o residente ficou 6,6–7,6% mais rápido (1B 2,85 contra 3,05 ms/token; 3B 4,49 contra 4,86) e a CPU
  13–23% mais lenta por expert (1B 45,9 µs sem tráfego e 55,8 com backlog 4, contra 40,1 e 46,4; 3B 69–70 e 85, contra 61 e
  69). Por isso o critério absoluto ("ms/token mais de 10% abaixo da previsão") não se aplica: tudo ficou acima da previsão.
  A conta que vale é dentro da sessão, contra os controles 0 e 4.
- **Pelo critério registrado, o binário não cai.** Nenhum ms/token ficou abaixo da previsão, e o CPU por expert com
  backlog 1 fica a 61–90% do caminho entre o nível quieto e o nível com tráfego (backlog 2: 87–130%, o que dá uma ideia do
  ruído com 2 repetições e carga de fundo). Mas o binário não é exato: os 10–39% que a 1ª cópia não traz contam, abaixo.
- **Custos que o modelo binário não tem.** O `lru_cpu` com backlog 1 ficou 6–10% mais rápido que com backlog 4 (1B 25%:
  6,3%; as outras 5 células, 9–10%); o simulador previa 0–4%. Da diferença em ms/token (0,57–2,03), 39–66% está no CPU ms
  (0,26–1,33) e 22–44% na espera do roteador (0,18–0,45), que é parede do host até os logits, não tempo isolado da GPU. Os
  dois não particionam o total, porque se sobrepõem. A espera do roteador quase não muda de backlog 0 para 1 em 4 das 6
  células (1B 10%: 0,74 → 0,75; 3B 25%: 1,21 → 1,21) e cresce de 1 para 4. Ela acompanha as cópias simultâneas, não as
  cópias por token. Uma hipótese, não testada: a leitura dos logits (D2H na stream de cálculo) espera na fila do motor de
  cópia atrás das H2D em voo.
- **Efeito na conclusão: a margem do `lfu_cpu` sobre o `custo` diminui, sem virar o sinal.** No simulador, 94–97% das camadas
  com tráfego do `lfu_cpu` e 93–96% das do `custo` têm uma cópia em voo só. O termo do roteador, que só aparece com várias
  simultâneas, quase não pega nenhum dos dois. O que pega é a CPU não binária: o modelo cobra o nível cheio numa camada de uma
  cópia que, medida, custa 61–90% do aumento. O `custo` tem mais camadas assim (tráfego em 14–86% contra 6–12%), então
  corrigir isso ajuda mais a ele. Cobrando só a fração S do aumento de CPU e do `o_layer` nas camadas de uma cópia (S = 1 é a
  tabela acima; cobrar S também no `o_layer` é o caso mais favorável ao `custo`, então é um limite), `custo` × `lfu_cpu` fica:

  | S | 1B 10% | 1B 25% | 1B 50% | 1B 75% | 3B 10% | 3B 25% | 3B 50% | 3B 75% |
  |---|---|---|---|---|---|---|---|---|
  | 1 | +24,1% | +15,0% | +4,2% | +1,2% | +15,3% | +7,8% | +1,6% | +0,3% |
  | 0,75 | +17,5% | +11,1% | +3,2% | +1,0% | +10,8% | +5,6% | +1,2% | +0,2% |
  | 0,6 | +13,6% | +8,9% | +2,6% | +0,8% | +8,7% | +5,3% | +1,3% | +0,2% |

  O `lfu_cpu` × `lru_cpu` fica igual ou melhor (1B 10%: −34,2% → −34,8%). A recomendação do C se mantém porque o sinal
  aguenta S = 0,6 em toda célula e porque o `lfu_cpu` não precisa de modelo de tempo no runtime. Mas a margem sobre o `custo`
  nos orçamentos baixos é até ~45% menor que a da tabela (1B 10%: 24,1 → 13,6; 3B 10%: 15,3 → 8,7), e no 3B a 25% (+5,3%) chega perto do erro do modelo nesta medida
  (backlog 1 ÷ backlog 4 medido 0,90–0,94, previsto 0,96–1,00). A tabela de S saiu de uma cópia instrumentada do simulador
  feita pelo rev-2 na rodada 37 (contador de cópias em voo por camada e fator S); conferi que ela só acrescenta isso ao
  `moe_sim.py` atual, rodei de novo e os números batem. Ela não entrou no `moe_sim.py`. A medida do C não decide o
  S (no `lfu_cpu` frio, S = 1 e S = 0,6 preveem quase o mesmo, ver a seção do C); a margem sobre o `custo` fica com a
  incerteza desta tabela.
- A execução 1B backlog 0 n.º 1 teve um ajuste em Python rodando em paralelo por ~1,5 min; ela bate com a n.º 2 (10,89/10,93/
  11,22 contra 10,81/11,07/11,18 ms/token) e entrou na conta.

**Testes** (`tests/test_moe_sim.py`, 46 passam): o da calibração por camadas usa `c_cpu_quiet` sem `o_layer` no caso sem
nada residente; dois fixam o protocolo do runtime (chave em voo vai à CPU; admissão recusada pela contagem com a fila vazia
em tempo), cada um com o `m05` como controle; dois fixam os regimes (camada com e sem tráfego, por backlog 4 e 0; cópia
síncrona do `custo` com a CPU ociosa e ocupada, com a ociosa de controle); um confere a banda disputada contra a cópia do
`lru_sync`; quatro fixam o leitor do `m2a` (backlog nos dois formatos de cabeçalho, cabeçalho que o simulador não reproduz é
erro, validação sem linha reprova, os três reprovando no leitor antigo; e a linha `lfu_cpu` lida, preparação da validação do
C, com o cabeçalho de aquecimento recusado porque a validação simula o cache vazio); `test_simulador_reproduz_o_m2a` roda a validação nos
6 arquivos (pula sem os traços). Mutantes, um de cada vez, cada um reprovado: cópia do `custo` sempre com a banda livre;
`o_layer` sem a condição de tráfego; CPU num nível só.

**Limites:**

- **A maior incerteza é o regime do `lfu_cpu`**: a validação cobre camadas sem tráfego (backlog 0) e quase todas com tráfego
  (`lru_cpu`, 98–100% de 10% a 50%), e o `lfu_cpu` vive em 6–12%. A medida com backlog 1 e 2 mostrou que, numa camada com
  uma cópia só, a CPU por expert sobe 61–90% do aumento entre o nível quieto e o de backlog 4, e não 100% (custo
  `c_cpu_quiet + S · (c_cpu − c_cpu_quiet)`, com S medido de 0,61 a 0,90). Isso estreita a distância para o `custo` (tabela de S)
  e não muda a do `lru_cpu`. O C mede o `lfu_cpu` de verdade e valida o simulador nesse regime.
- Os parâmetros são da sessão das referências (29/10). A sessão de 30/10 teve CPU 13–23% mais lenta e residente 6,6–7,6% mais rápido;
  comparação entre políticas tem de ser feita dentro da mesma sessão, com controles.
- A validação cobre só `lru_sync` e `lru_cpu`, as políticas que o runtime tinha no passo 3. `lfu_cpu` e `custo` são extrapolação do mesmo
  modelo de custo.
- `o_layer` e banda vieram do 1B e o 3B os confere sem reajuste, mas o critério de ±10% distingue pouco o `o_layer` (2–8% do
  tempo): o ajuste próprio do 3B no `lru_sync` daria 25 µs. A sobra por camada com tráfego varia de 10 a 36 µs entre
  orçamentos.
- Uma execução por configuração com backlog 4 nas referências: o `lru_cpu` a 10% do 1B variou de 10,02 a 11,09 ms/token entre
  as execuções `antes`/`depois`/`final` do B0, dispersão da ordem do critério.
- Os limites do M0.5 continuam: traços do próprio modelo em greedy, 8 conversas seguidas, prefill fora, experts pequenos.

**Rodada 35** (`.herdr/review/strata-opencode-35/`, só o rev-2: o dispatcher marcou "material mudou" porque um script do
rev-2 importou o simulador congelado sem `-B` e o Python gravou um `__pycache__` lá; nenhum arquivo congelado mudou de
conteúdo, e o rev-1 não chegou a ser disparado). Achados do rev-2, todos procedentes e tratados: o `o_layer` incidia em toda
camada com CPU por hipótese, e os `backlog0` já medidos desmentiam isso, o que invertia a conclusão (P2; CPU em dois níveis,
`o_layer` só com tráfego, `backlog0` no `--validar`); a sensibilidade só olhava a CPU mais lenta (`--cpu-scale 0.85`); "deixa a
PCIe livre" não era o mecanismo no modelo antigo (texto refeito com acerto e tráfego); o 3B confere pouco o `o_layer` e o
"3–15%" do histórico estava errado (limites e caminho); o ramo da banda disputada do `custo` não tinha teste (teste novo e
mutante).

**Rodada 36** (`.herdr/review/strata-opencode-36/`, completa; nenhum P0/P1; os dois mantêm a recomendação do C). rev-1: dois
P2 no leitor do `m2a`, o backlog que caía para 4 em silêncio com o cabeçalho atual (`; admissão evento`) e a validação sem
linhas que aprovava (corrigidos, com testes de regressão que reprovavam antes). Os dois: as frações de tráfego citadas não
saíam do predicado do simulador (contavam antes do `settle`; agora saem do contador `layers_traffic`), o `c_cpu` do 1B não era
o ponto fixo do próprio ajuste (0,0472 → 0,0464, e 0,80 → 0,70 na alternativa) e a medida com backlog 1 e 2 testa a
intensidade, não a fração (texto refeito, previsão registrada antes da medida). rev-2: a célula do oráculo no 1B a 50% estava
trocada pela de 10%.

**Rodada 37** (`.herdr/review/strata-opencode-37/`, completa; nenhum P0/P1; os dois confirmam as correções da 36 e mantêm a
recomendação do C). Os dois (P2): a leitura da medida com backlog 1 e 2 estava errada. A diferença não vinha quase toda do
roteador (22–44%; a CPU dá 39–66%), e a medida não aumenta a vantagem do `lfu_cpu` sobre o `custo`. O rev-1 apontou que ela
não fora medida, e o rev-2 mostrou que a CPU não binária a estreita (tabela de S, conferida). Texto refeito; o simulador não
muda antes do C. rev-2 (P3): faixas erradas (6–10%, 0,18–0,45 ms, 13–23%, 6,6–7,6%), o 2º ramo do critério não estava
escrito, e eram 46 testes, com quatro do leitor. rev-1: o horário da previsão. O arquivo foi gravado às 10:34:17, depois do
primeiro controle com backlog 0; texto corrigido.

## C (M3) — desenho: despejo e admissão por frequência (`lfu_cpu`) no runtime (30/10)

Aprovado pelo Breno em 30/10 (10:30), depois do passo 3: o C passa a ser o `lfu_cpu` do simulador, sem a cópia síncrona por
custo, que o simulador recalibrado põe abaixo do `lfu_cpu`.

**Como as referências fazem** (leitura de 30/10):

- **Strata** (`/home/brenoperucchi/Devs/GitHub/strata`): "adaptive tier" em `src/program/generate.cpp:5703-5756` (e
  `:4167-4222` no `--serve`). Contagem por (camada, expert), +1 por entrada roteada (`src/core/expert_source.cpp:1610-1612`),
  ×0,7 em todo o vetor a cada chamada, que roda a cada 4 rodadas (`generate.cpp:347`, `:5752`). Candidato só com contagem
  ≥ 2,0 e troca só se vence o residente mais frio **da mesma camada** por 1,5 (`:5718`, `:5727`); as trocas de todas as camadas
  são ordenadas pelo ganho e cortadas em 96 por lote, num único lote em voo (`:369`, `:5708`). O cache sai cheio de um perfil
  offline (`data/expert-profile.bin`, `tools/make_profile.py`), e a contagem começa zerada.
- **llama.cpp** (`/home/brenoperucchi/Devs/GitHub/llama.cpp`): `--moe-cache-mib`, experimental, `src/llama-moe-cache.cpp`.
  SLRU por recência (duas listas, `:16-34`, `:82-87`), todo miss admitido e copiado de forma síncrona antes do split
  (`:109-154`, `:574-589`), sem perfil, sem prefetch. É o nosso `lru_sync` com proteção contra varredura.

**O que copiamos e onde divergimos:**

- Frequência com decaimento e filtro de admissão contra o residente mais fraco: como o Strata.
- **Slots globais**, não por camada: o orçamento é um só e a vítima pode ser de qualquer camada, como no simulador que
  validamos. O Strata compete dentro da camada porque a cota dele por camada vem do perfil.
- **Decisão por falta, não em lote**: a falta já vai à CPU e a cópia entra na fila de admissão (backlog 4) como no `lru_cpu`,
  que mediu e validou esse protocolo. O lote do Strata existe porque a troca dele roda numa thread entre janelas do decode
  especulativo.
- **Pontuação do simulador**, para o runtime e o simulador darem as mesmas escolhas: por chave, log₂ do valor e o passo da
  última atualização; tocar soma 1 ao valor decaído com meia-vida de 128 tokens (`HALF_LIFE · L` passos de camada); a ordem
  entre chaves é `log₂ v + passo / (HALF_LIFE · L)`, que não muda com o tempo. Todos os k experts pedidos na camada são
  tocados antes das decisões. Sem limiar e sem margem na primeira versão: admite se a pontuação vence a da vítima
  estritamente. A margem do Strata vira uma varredura no simulador (barata) antes de virar parâmetro no runtime.
- Perfil persistido fica fora do C: a medida aquece com as conversas de treino, como a comparação do simulador.

**Mudanças:**

1. `cache/core.rs`: `victim_min_by(&pins, &idle, prio, teto)` devolve um slot livre se houver; senão o `Resident` de menor
   prioridade fora dos pinos, e só o reserva se essa prioridade for menor que o teto. Recusar não muda estado nenhum. Empate
   pelo índice do slot. O `victim` do LRU continua igual (o `lru_cpu` não muda bit a bit no modo barreira).
2. `cache.rs`: `Policy::LfuCpu`, com a pontuação num tipo pequeno (`Freq`) sem GPU; `Loading` vai à CPU e o backlog é o
   mesmo do `lru_cpu`.
3. `m2a-cache`: política `lfu_cpu` e `--warmup <conversas>`, que passa as conversas de treino (0–3) pelo cache antes das
   medidas, sem cronometrar, para as duas políticas.

**Testes** (feitos; 49 testes Rust passam, sem warnings):

- `freq_da_as_mesmas_pontuacoes_do_simulador` (unidade): a `Freq` dá as pontuações do `moe_sim.py` numa sequência fixa, a
  1e-12, e o `clear` dela zera.
- `vitima_por_prioridade_recusa_sem_mudar_estado` (núcleo, GPU): slot livre primeiro, recusa no empate com o teto sem mudar
  estado nem geração, pino nunca sai, empate pelo menor índice.
- `cache_lfu_cpu_segue_o_lfu_de_referencia_e_bate_com_a_cpu`: modelo de teste com 16 experts, duas conversas de 48 tokens,
  modo barreira, 4 configurações de slots e backlog; os cinco contadores iguais aos de uma referência LFU em Rust
  (`reference_lfu`, a mesma `Freq`), recusas pelo teto > 0, logits contra a CPU.
- `cache_lfu_cpu_por_evento_bate_com_a_cpu_e_clear_zera_as_frequencias`: por evento, logits contra a CPU e orçamento; o
  `clear` zera todas as frequências, e B depois de A e do `clear` dá os mesmos contadores que B num cache novo.
- `lfu_cpu_conta_o_toque_do_expert_que_ainda_esta_carregando` (`tests/portao.rs`): por evento, a frequência do cache igual
  bit a bit à de uma `Freq` recalculada das rotas devolvidas, com um portão de teste na stream de cópia (`hold_copies`,
  `#[doc(hidden)]`) que segura as admissões da primeira metade e garante pedidos de chave em `Loading` (contador novo
  `loading` > 0). Fica num binário só dele e aquece os kernels num modelo descartável antes de fechar o portão: qualquer
  sincronização do contexto com a stream fechada (a primeira carga de um kernel, ou outro teste da GPU em paralelo) espera
  o portão até o prazo. Foi o que aconteceu nas duas primeiras versões, e o mutante "chave `Loading` sem toque" chegou a
  reprovar pelo prazo, não pela `Freq`; refeito, ele reprova pela `Freq`, e o teste certo passou 5 vezes seguidas.
- Mutantes, um de cada vez, todos reprovados: M1 sem teto, M2 sem o toque, M3 vítima LRU no LFU, M4 `clear` sem zerar a
  `Freq` (`mutantes-c.txt` no scratchpad da sessão); e, do rev-2 na rodada 38, empate pelo maior índice, passo não
  incrementado, toque depois das decisões, teto não estrito e chave `Loading` sem toque (este só reprova com o portão).
- Em barreira (`qual-runs/C-SEGURANCA-20261030/rodar-barreira.sh`): `resident`, `cache_full`, `lru_sync` e `lru_cpu` com
  contagens exatas e hash dos logits iguais aos do binário final do B0, nos dois modelos; `lfu_cpu` aquecido nas conversas
  0–3, duas execuções por modelo com contagens e hash iguais.

**Medida (encerra o C):**

- `lru_cpu` contra `lfu_cpu`, aquecidos nas conversas 0–3 e medidos nas 4–7, nos dois modelos, a 10%, 25% e 50%; 5
  execuções intercaladas por configuração, núcleos fixos (`taskset -c 0-15`) e nada mais pesado na máquina durante a medida.
  Espera = ms/token − ms/token do residente da mesma execução.
- **Portão do C:** ≥ 5% de redução da espera do `lfu_cpu` sobre o `lru_cpu` em pelo menos um modelo e um orçamento, com as
  faixas observadas (mín–máx das 5, não intervalo de confiança) sem sobreposição; ou o resultado negativo e a causa.
- **Teste do simulador na comparação aquecida**, que o portão não faz (ele passaria com o simulador errando por um fator 5):
  a redução da espera prevista por célula (hoje 25–52%) é registrada antes de qualquer medida, com os parâmetros atuais do
  `moe_sim.py` (sha no registro; os reajustados da fase A servem só à validação fria), e o simulador cai para a espera
  relativa se a redução medida, calculada das médias das 5 execuções (a mesma grandeza do portão), ficar fora de [metade,
  dobro] da prevista em qualquer célula.
- **Validação do simulador no regime esparso**, que o passo 3 não cobria: o `lfu_cpu` medido sem aquecimento (cache vazio,
  conversas 4–7, admissão por evento) contra o `--validar`, pelo mesmo critério (±2 pontos, ±10%). A sessão de 30/10 teve
  CPU 13–23% mais lenta e residente 6,6–7,6% mais rápido que a das referências, então os parâmetros são reajustados na
  própria sessão, em duas fases, com o procedimento fixado antes:
  1. **Fase A:** `lru_cpu` com backlog 0 e 4, sem aquecimento, 10/25/50%, os dois modelos, 3 repetições intercaladas.
     Reajuste: `c_cpu` e `c_cpu_quiet` pelo mesmo mínimo quadrado sem pesos do passo 3 (ponto fixo, as linhas de backlog 0 e
     4 da fase A); a GPU pelo residente da sessão, pela mesma calibração por camadas do passo 3; `o_layer` e banda ficam.
     Os parâmetros reajustados e as previsões do `lfu_cpu` frio, com S = 1 (o modelo atual) e S = 0,6 (a cópia
     instrumentada da rodada 37, guardada na pasta da medida com o sha), vão para um `previsao.txt` gravado antes da fase B.
  2. **Fase B:** `lfu_cpu` frio, 3 repetições, cada uma intercalada com um `lru_cpu` backlog 4 frio de controle (deriva da
     sessão: o controle tem de ficar a ±5% da média da fase A, ou isso é relatado).
  Regra: a validação passa ou reprova pelo critério comum (±2 pontos, ±10%) com S = 1, o modelo atual. A previsão com
  S = 0,6 vai junto, mas a medida não decide o S: no `lfu_cpu` frio as duas previsões diferem só 0,6–1,7% de ms/token
  (testado com os dados da medida com backlog 1 e 2 no lugar da fase A), abaixo da dispersão entre repetições. O S pesa onde
  há muitas camadas de uma cópia, como no `custo`, que o runtime não tem; ele continua o da medida com backlog 1 e 2
  (0,61–0,90). A medida testa a extrapolação para o tráfego esparso. Ela não isola o mecanismo do aumento da espera do roteador com cópias
  simultâneas (hipótese da fila D2H/H2D), que pediria um contador de tempo próprio.
- Registrar também, como na medida com backlog 1 e 2, `ambiente.txt` (driver, carga média, sha do binário) e os horários.
  O binário sai de `cargo build --release --bin m2a-cache` e é copiado para a pasta da medida antes de tudo: o `cargo test`
  troca o `target/release/m2a-cache` por um build do perfil de teste, com outro sha.
- Antes da medida, pedir ao Breno para aliviar a carga de fundo da máquina (navegador, indexação, AnyDesk), que dividiu a CPU
  com o caminho dos experts na sessão de 30/10.

**Rodada 38** (`.herdr/review/strata-opencode-38/`, completa; nenhum P0/P1/P2; os dois não acharam defeito funcional no C e
confirmam as correções da 37). Os dois (P3): a medida do C não "separa o custo na GPU"; texto refeito, e a validação fria
passa a prever S = 1 e S = 0,6 antes de medir. rev-2 (P3): o mutante "chave `Loading` sem toque" sobrevivia, porque no
modelo de teste a cópia termina antes de o expert voltar; teste novo com portão na stream de cópia e contador `loading`, num
binário próprio, e o mutante reprova pela `Freq`. rev-2 (P3): o protocolo do C ganhou o teste do simulador na comparação aquecida ([metade, dobro] da redução
prevista), o reajuste em duas fases fixado antes e 3 repetições do frio e dos controles; o bloco de testes do C descreve o
que existe, e o `lfu_cpu` entrou na conferência em barreira (duas execuções). rev-2 mediu também que o desempate diferente
(índice do slot no runtime, versão do heap no simulador) não muda nenhum resultado do simulador. rev-1 (P3): o limite
confundia fração do aumento com fração do custo. rev-2 (P3): "~40%" era até ~45%, "as políticas que o runtime tem" e uma
quebra de linha.

**Rodada 39** (`.herdr/review/strata-opencode-39/`, nível 1 com o rev-2, verificação; os seis achados da 38 resolvidos). O
teste do portão do C mata o mutante pela `Freq`, falha alto se o portão não segurar (mutante do rev-2) e passou 60 vezes,
também com outro processo de GPU em paralelo. Achados P3: (1) os dois testes de portão do núcleo, do B0, falhavam em ~1–3%
das execuções paralelas dos testes de unidade, pelo mesmo mecanismo; agora todo teste de unidade com GPU segura uma trava
do crate (`gpu_test_lock`, só em `cfg(test)`), e 200 execuções paralelas passaram sem falha; (2) três escolhas do protocolo
escritas (distância para escolher o S, parâmetros da redução prevista, redução das médias).
Depois da rodada 39, ao montar os roteiros da medida (`qual-runs/C-MEDIDA-20261030/`), o `ajuste_fase_a.py` testado com os
dados da medida com backlog 1 e 2 mostrou que S = 1 e S = 0,6 preveem o `lfu_cpu` frio com só 0,6–1,7% de diferença: a
regra "fica o S mais perto" não teria poder de decisão. Ela saiu; a validação fria usa S = 1, e o S fica em aberto.

**Rodada 40** (`.herdr/review/strata-opencode-40/`, nível 1 com o rev-1, roteiros da medida antes de rodar). Previsão
aquecida, ajuste da fase A (reproduzido), recalibração da GPU, a decisão sobre o S e a trava dos testes conferem. Dois P2
no `rodar.sh`, corrigidos: uma execução do `m2a-cache` que falhasse era registrada e o roteiro seguia (agora `set -euo
pipefail` e `run` devolve o código do binário), e uma nova chamada sobrescreveria a medida e a previsão (agora trava com
`flock` e recusa se a pasta já tiver resultados ou previsões, sem apagar nada). Testado numa cópia com binário falso: a
falha na 1ª execução para o roteiro com saída 1; a 2ª chamada recusa sem mudar um byte; a trava ocupada recusa; o caminho
de sucesso faz as 44 chamadas.

**Resultado da medida do C** (`qual-runs/C-MEDIDA-20261030/`, 30/10, 13:33–16:15; 44 execuções, todas com código 0;
`analise.py` → `analise.txt`; previsões em `previsao.txt`, gravado às 13:34 antes da primeira execução, e
`previsao-fase-b.txt`, gravado às 15:45:40 antes da fase B). **O C passa.**

- **Comparação aquecida** (espera = ms/token − residente da mesma execução, médias das 5):

  | modelo | orçamento | espera `lru_cpu` (mín–máx) | espera `lfu_cpu` (mín–máx) | redução | prevista | tok/s `lru_cpu` → `lfu_cpu` |
  |---|---|---|---|---|---|---|
  | 1B | 10% | 9,48 (9,00–9,94) | 6,09 (5,45–6,96) | 35,8% | 34,2% | 81 → 112 (+38%) |
  | 1B | 25% | 5,90 (5,34–6,56) | 4,15 (3,92–4,62) | 29,7% | 38,2% | 114 → 143 (+26%) |
  | 1B | 50% | 2,81 (2,36–3,31) | 1,64 (1,50–1,84) | 41,7% | 51,7% | 176 → 223 (+27%) |
  | 3B | 10% | 16,28 (14,32–19,20) | 12,08 (11,03–13,45) | 25,8% | 25,4% | 48 → 61 (+26%) |
  | 3B | 25% | 10,25 (9,43–11,21) | 7,37 (6,65–7,97) | 28,1% | 31,6% | 68 → 85 (+25%) |
  | 3B | 50% | 4,45 (4,02–4,99) | 2,68 (2,47–2,96) | 39,8% | 40,1% | 112 → 141 (+26%) |

  Portão: passa nas 6 células (redução ≥ 5% e faixas sem sobreposição). Teste do simulador: passa (toda redução dentro de
  [metade, dobro] da prevista; no 3B a até 3,5 pontos, no 1B a 25% e 50% abaixo da prevista por 8,5 e 10 pontos). O
  `lfu_cpu` acerta mais (1B 24,9 / 50,1 / 78,5% contra 12,5 / 42,9 / 75,0%) e copia 1,5–3,0 experts por token contra 27–59 do
  `lru_cpu`. O tok/s sai do bloco final do `analise.txt` (1000 ÷ média do ms/token das 5 aquecidas); o residente das
  execuções aquecidas dá 349 tok/s no 1B e 225 no 3B.
- **Validação fria do `lfu_cpu`** (médias das 3, previsão com os parâmetros reajustados na fase A, S = 1): passa. O acerto
  bate a 0,1 ponto em todas as células, porque a regra de admissão do simulador é a do runtime. O ms/token medido fica de
  +0,7% a +7,5% acima do previsto, sempre acima: um viés de ~5%, dentro do critério. No 1B a deriva média dos controles da fase B (+1,8 a +2,8%) é da
  mesma ordem, então os dados não separam o modelo da sessão; no 3B a deriva não o explica (a 25% o controle caiu 2,9%). A
  previsão com S = 0,6 fica mais longe nas 6 células, mas a diferença (~1%) é menor que o viés; o S continua em aberto.
  Parâmetros reajustados: 1B `c_cpu` 0,0524 e `c_cpu_quiet` 0,0447 (13% e 11% acima dos de 29/10), 3B 0,0802 e 0,0662 (16% e 9%
  acima); residente da fase A (2,69 e 4,23 ms) 12% (1B) e 13% (3B) mais rápido que nas referências.
- **Deriva:** os 6 controles `lru_cpu` frios (18 células) da fase B ficaram de −4,2% a +5,3% da média da fase A; dois passaram de ±5%
  (1B 25%: +5,1%; 1B 50%: +5,3%). A carga média da máquina variou de 10,2 a 17,4 durante a medida (o `m2a-cache` usa 8 threads),
  contra 5,2 no `ambiente.txt` logo antes. O alívio da carga de fundo foi pedido ao Breno antes da medida, mas nenhum
  registro diz quais processos rodavam durante ela: carga de fundo não identificada. O portão passa nas 6 células e
  bastava uma; a folga mais estreita entre as faixas é de 0,5 ms (1B 50%), da ordem de uma deriva de 5%.
- **Limites:** uma sessão só, com carga de fundo; 5 repetições aquecidas e 3 frias; os modelos são pequenos (experts de 3–5
  MB) e o contexto vai até ~1.100 posições. O residente ficou ~6% mais rápido entre a parte aquecida e a fria (1B 2,86 →
  2,69 ms; 3B 4,45 → 4,23 ms), deriva dentro da sessão. O viés frio de ~5% e o S ficam para a próxima recalibração.

**Próximos passos (Breno, 30/10):** comparar o runtime com o llama.cpp nos Granite 1B e 3B, com os mesmos parâmetros:
mesmas conversas regeneradas em greedy pelo `llama-server` (seed 1, como os traços), o `--moe-cache-mib` com o mesmo
orçamento de VRAM dos nossos slots junto de `--cpu-moe` (e o `-ncmoe` como segunda linha), GGUF BF16, 8 threads,
`taskset -c 0-15`, execuções intercaladas e previsão registrada antes. A comparação com o Strata fica para depois, com um
modelo menor: hoje o Strata só roda o Qwen3.8-Flash-Next (125B, GGUF IQ2/IQ3), e o runtime só os Granite em BF16.

## Comparação com o llama.cpp nos Granite 1B e 3B (desenho, 30/10, revisto nas rodadas 42 e 43)

**Pergunta:** com o mesmo modelo, os mesmos tokens, o mesmo número de experts na VRAM, a mesma precisão de KV e a mesma
CPU, quanto tempo por token o nosso runtime leva contra o llama.cpp (`0893ec9`, build local de 29/10)? Nenhuma decisão do
runtime depende desta medida. É uma comparação, e cada afirmação de "mais rápido" só vale numa célula com faixas mín–máx sem
sobreposição.

**Problema de método.** O `m2a-cache` não gera texto. Ele passa as conversas já prontas (`tokens.npy` dos traços), um token
por vez em batch 1, contando os tokens do prompt (3.451 no 1B e 3.814 no 3B, nas conversas 4 a 7), e o cache persiste entre
as conversas. O tempo de cada token é o `g.step` inteiro, com a leitura dos logits no host; as conferências ficam fora do
cronômetro (`src/bin/m2a-cache.rs:293-307`). O `llama-server` processa o prompt num batch e gera a resposta. Se a resposta
divergir do traço, o texto, o roteamento e o acerto do cache mudam. Comparar o ms/token do servidor com o nosso misturaria
motor e método.

**Linha principal: harness de decode forçado no llama.cpp** (`tools/llama-forcado/forcado.cpp`, compilado com `g++`
contra `build/bin/libllama.so` e `libllama-common.so` do build existente; o repositório do llama.cpp não muda).
- Inicializa por `common_params_parse` e `common_init_from_params`, o único caminho que leva o `--moe-cache-mib` ao contexto;
  um contexto montado à mão rodaria sem cache e sem erro (rodada 42).
- Flags fixos em todas as linhas: `-c 2048` (as conversas vão até 1.127 tokens), `--fit off`, `--no-warmup` (o cache começa
  vazio, como o nosso), `-t 8`, `-ctk f32 -ctv f32 -fa off` (o nosso KV é F32, `gpu_model.rs:189`; o padrão do llama.cpp é
  F16). O flash attention fica desligado nas linhas F32 porque, com `-fa on`, o CUDA aceita K/V em F32 mas converte para F16
  antes de calcular (`ggml-cuda/fattn.cu:452-453, 759-773`): a linha "F32" faria a conta do F16 com uma conversão a mais
  (rodada 43; os arquivos de argmax F32 com FA e F16 saíram idênticos byte a byte).
- Lê os tokens de cada conversa, exportados dos `tokens.npy` para texto. Limpa o KV (`llama_memory_clear`) no começo de cada
  conversa e chama `llama_decode` com um token por vez, pedindo os logits e conferindo o retorno. O cache MoE persiste entre
  as conversas, como o nosso.
- Aquecimento só na conversa 3, sem cronometrar (Breno, 30/10, para encurtar a medida); medida nas 4 a 7. O nosso
  `m2a-cache` roda com `--warmup 3`. Por isso o resultado não se compara direto com o da medida do C, que aqueceu nas 0 a 3:
  o `lfu_cpu` chega com menos histórico.
- O tempo de cada token é `llama_decode` + `llama_get_logits_ith(-1)`, que sincroniza e entrega a linha inteira do
  vocabulário no host, o mesmo volume do `g.step`. O argmax e as conferências ficam fora do cronômetro. O harness imprime a
  média e a mediana do ms/token por conversa e no total, nas mesmas faixas de posição do `m2a-cache`, e grava o argmax e a
  margem top-1 − top-2 de cada posição para a conferência.

**Validação antes da medida** (`validacao.txt`):
- Portão de correção (Breno, 30/10, depois da rodada 43): com tudo na GPU F32, o argmax do harness é igual ao do
  transformers (float32, teacher forcing nos mesmos ids, `tools/llama-forcado/confere_logits.py`) em toda posição do
  `texto-a` com margem do transformers > 0,1 logit, e em ≥ 99% das posições com margem > 0,1 nas conversas 4 a 7, prompt
  incluído. O erro relativo L2 por posição é relatado (mediana, p99, máximo e onde).
- Portão por configuração, na medida: o argmax de cada linha do llama.cpp é igual ao da passada tudo na GPU F32 do mesmo
  harness (1ª repetição) em ≥ 99% das posições cronometradas com margem da referência > 0,1. O nosso `m2a-cache` passa pela
  conferência própria (logits contra o transformers no `texto-a` e contra a passada residente nas conversas, limite 1e-4).
  Uma configuração que não passa fica suspensa, e as afirmações que dependem dela não são confirmadas nem refutadas. O 99% é
  um detector de defeito do harness (que derrubaria muito mais que 1%), e não do ruído de BF16 nem da conta dos experts na
  CPU, que no piloto do `--cpu-moe` deu 99,34% (1B) e 99,73% (3B).
- O portão registrado antes, argmax contra o próximo token do traço (≥ 99% e margem < 0,5), reprovou no 1B (abaixo) e foi
  trocado: o traço é o texto do servidor tokenizado de novo pelo HF, então o prefixo forçado não é o que o servidor viu ao
  gerar, e o próximo token dele não é referência do argmax. A igualdade do tokenizador também deixou de ser condição, porque
  os dois motores recebem os mesmos ids. As duas conferências continuam relatadas.
- Cache ativo e flags efetivos: o `analise.py` lê o log de cada execução do llama.cpp e confere FA desligado nas linhas F32,
  o tipo do KV, `MoE cache size = …` com um grupo só (`24 layers` / `32 layers`) e o N de slots igual ao nosso, e o hit rate
  do destrutor diferente de zero a 25%, 50% e no cache cheio; o que não bate sai como `FALHA`.

**Validação feita** (30/10, `qual-runs/LLAMACPP-COMPARA-20261030/validacao.txt`, gravado por `validacao.sh`; harness em
`tools/llama-forcado/`): passa nos portões registrados.
- **Logits contra o transformers, tudo na GPU F32 sem FA:** no `texto-a`, argmax igual em 266/266 (1B) e 264/264 (3B)
  posições com margem > 0,1; erro relativo L2 mediano 3,2e-3 e 2,5e-3, máximo 6,0e-2 e 1,1e-1. Nas conversas 4 a 7, argmax
  igual em 99,64% (3.346/3.358) e 99,81% (3.736/3.743); erro L2 mediano 2,0e-2 e 6,5e-3, máximo 3,1e-1 e 2,7e-1. A maior
  divergência (margem 1,9 no 1B) é a posição 51, a mesma nas quatro conversas, porque o prompt de sistema é igual: depois de
  `<|end_of_text|>\n<|start_of_role|>`, onde o modelo adivinha o nome do papel ("2" no transformers, "I" no llama.cpp; o
  certo é "user"). O resto são quase-empates. O nosso runtime fica em ~6e-6 contra a mesma referência no `texto-a`; o
  llama.cpp calcula com ativações em precisão reduzida.
- **Erro L2 por faixa de posição** (conversas 4 a 7, mediana nas faixas 0–256, 256–512, 512–1024 e 1024+; rodada 44): 1B
  2,4e-2 / 2,0e-2 / 1,9e-2 / 1,4e-2 e 3B 7,9e-3 / 5,9e-3 / 6,2e-3 / 6,6e-3. O erro fica estável e não cresce com a posição;
  a faixa 1024+ tem só 48 e 155 posições.
- **Entre configurações** (argmax contra a passada tudo na GPU F32, todas as posições): F16 com FA 99,91% e 99,84%;
  `--cpu-moe` 99,34% e 99,73% (8 e 1 divergências com margem ≥ 0,5, da conta dos experts na CPU).
- **Registro do portão antigo** (argmax contra o próximo token do traço): reprova no 1B (99,24%, uma divergência de
  retokenização com margem ≥ 0,5) e, sem FA, também no 3B (99,35%, um quase-empate com margem ≥ 0,5).
- **Tokenizador:** difere do HF em 14 dos 16 textos, sempre no mesmo caso: o HF corta `\n\n` em dois `Ċ` e o GGUF usa um
  token só, `ĊĊ` (44 vezes no 1B, 46 no 3B). O `validacao.sh` conta os casos com o difflib e grava a lista no
  `validacao.txt`.
- **Calibração do MiB:** 231 / 579 / 1.155 / 2.307 MiB no 1B e 581 / 1.445 / 2.885 / 5.765 no 3B dão exatamente 76 / 192 /
  384 / 768 e 128 / 320 / 640 / 1.280 slots, com um grupo só. Os experts têm 3 MiB e 4,5 MiB.
- **Banda H2D síncrona** (`banda.cu`): pageable 13,5 GB/s por tensor de 1 MiB e 14,6 GB/s por tensor de 1,5 MiB; pinned
  19–22 GB/s. Uma falta do cache do llama.cpp custa ~0,23 ms no 1B e ~0,32 ms no 3B (três tensores por expert).
- **Pilotos de tempo:** tudo na GPU F32 1,88–2,00 ms (1B) e 3,53–3,75 ms (3B), variando ~6% entre passadas; F16 1,71–1,80 e
  3,21–3,51; `--cpu-moe` 12,0–13,4 e 22,1–25,4, variando ~11–15% entre passadas. A previsão usa a passada gravada no
  `validacao.txt` (1,94 / 3,58; 1,76 / 3,27; 12,02 / 22,12), lida pelo `previsao.py`, que grava o sha do arquivo (rodada 44).

**Linha secundária: `llama-server` em uso normal**, uma vez por configuração do núcleo. Regenera as conversas 4 a 7 em greedy
(temperatura 0, seed 1, max_tokens 1024, mesmo template de `tools/gen_trace_texts.py`, mesmos flags de KV e contexto),
confere os tokens contra os traços, registra a primeira divergência e grava o `timings.predicted_per_token_ms`, só da
resposta. É "uso real", sem comparação célula a célula com o nosso, porque o método é outro.

**Configurações** (por modelo; o orçamento é a fração dos experts na VRAM, igual à dos nossos slots):

| linha | 1B (768 experts de 3,15 MB, 24 camadas) | 3B (1.280 de 4,72 MB, 32 camadas) | repetições |
|---|---|---|---|
| llama.cpp tudo na GPU, KV F32 | 1 | 1 | 3 |
| llama.cpp `--cpu-moe --moe-cache-mib`, 10/25/50% | 76 / 192 / 384 slots | 128 / 320 / 640 slots | 3 |
| nosso `m2a-cache --policies resident,lru_cpu,lfu_cpu` | 10/25/50% | 10/25/50% | 3 |
| llama.cpp tudo na GPU, KV F16 (padrão do produto) | 1 | 1 | 1 |
| llama.cpp `--cpu-moe` sem cache | 1 | 1 | 1 |
| llama.cpp `-ncmoe`, mesma fração de camadas ou menos na GPU | 22 / 18 / 12 (8,3% / 25% / 50%) | 29 / 24 / 16 (9,4% / 25% / 50%) | 1 |
| llama.cpp cache cheio (`--moe-cache-mib` com 768 / 1.280 slots) | 1 | 1 | 1 |
| nosso `lru_sync` (mesmo protocolo síncrono do cache do llama.cpp) e `cache_full` | 10/25/50% e 100% | 10/25/50% e 100% | 1 |

- O núcleo repete 3 vezes, e as linhas de base, que são lentas e previsíveis, rodam uma vez (Breno, 30/10). As diferenças
  esperadas entre o núcleo e as linhas de base são de 2 a 5 vezes.
- O cache do llama.cpp (`src/llama-moe-cache.cpp`) tem slots globais por grupo de camadas com o mesmo layout, como os nossos.
  É um SLRU com o protegido limitado à metade, admite toda falta e copia de forma síncrona na stream de cálculo: é o
  protocolo do nosso `lru_sync`, com outra política. Ele lê os ids e sincroniza em toda camada com cache, com ou sem falta.
- O MiB de cada célula sai de uma calibração antes da medida: subir o harness com MiB crescentes e ler a linha de log
  `N slots (x%)` até N ser igual ao nosso número de slots. O banco reserva um slot extra por tensor (o MMQ pode ler além do
  último expert) e alinha cada tensor, então o llama.cpp fica com a mesma capacidade útil e um pouco mais de VRAM física, a
  favor dele. O MiB de cada célula fica no `rodar.sh`; a linha de slots e o tamanho dos bancos ficam no log de cada execução.
- `-ncmoe` põe camadas inteiras na CPU. A 10% não existe número inteiro de camadas que dê a mesma fração: 22 camadas no 1B
  deixam 8,3% dos experts na GPU, e 29 no 3B deixam 9,4%.
- O cache do llama.cpp copia de memória pageable (o mmap do GGUF, com `--cpu-moe`), e o nosso copia de memória pinned. É o
  comportamento real dele, então a medida fica assim, e a diferença vai para os limites.

**Roteiro** (`qual-runs/LLAMACPP-COMPARA-20261030/rodar.sh`, no molde do C: `set -euo pipefail`, `flock`, recusa se já
houver resultados, `carga.log` por execução, e um `ambiente.txt` com:
- o commit do llama.cpp e o número de arquivos alterados na árvore dele;
- os sha das `libllama*`, das `libggml*`, do `llama-server`, dos GGUF e do simulador (os do harness e do `m2a-cache` vão para
  o `sha-binarios.txt`, conferidos contra o `bin/SHA256SUMS` antes de começar);
- o driver, a `libcuda` e a carga média;
- a lista dos processos acima de 5% de CPU no começo e no fim.)
- Os flags efetivos e as linhas de log dos buffers e do cache não vão para o `ambiente.txt`: ficam no `.log` de cada execução,
  e o `analise.py` confere em cada um o FA, o tipo do KV, o grupo único com o número de slots da célula e o hit rate (rodada
  44).

- Em cada repetição do núcleo, por modelo, as linhas do llama.cpp numa ordem que gira a cada repetição e uma execução do
  `m2a-cache`; os modelos alternam. Depois, as linhas de base, uma vez cada; por fim, a linha do servidor.
- Tudo com `taskset -c 0-15` e 8 threads. Nada compila durante a medida, e o Python roda só nos E-cores.
- Duração estimada pela previsão de cada linha (tokens de aquecimento e medida × ms/token previsto, mais 8 s de carga):
  2,14 h, mais ~15 min da linha do servidor. A conta por linha vai para o `previsao.txt`. A linha mais longa é o cache do
  llama.cpp a 10% no 3B (~87 ms/token, ~21 min nas 3 repetições).
- Antes: pedir ao Breno para aliviar a carga de fundo.
- O servidor não herda a trava e é encerrado pelo PID exato também em falha ou interrupção (`trap`); a porta é conferida no
  começo e antes de cada servidor, e um servidor que morre antes de ficar pronto falha a linha sem medir (rodada 43). Testado
  com binários falsos em sete casos: sucesso (36 + 8 + 8 chamadas, argmax em 20), falha que para, segunda chamada e trava que
  recusam, porta ocupada (recusa sem tocar no serviço alheio), servidor que morre e interrupção durante uma requisição
  (servidor encerrado, trava livre).

**Previsão** (gravada em `previsao.txt` antes da primeira execução):
- Tudo na GPU e `--cpu-moe`: os pilotos do harness, lidos do `validacao.txt`, e não mais os tg128 do `llama-bench` de 29/10.
  Com os P-cores (`taskset -c 0-15`), o `--cpu-moe` leva 12,0 ms (1B) e 22,1 ms (3B), contra 20 e 37 ms no `llama-bench`, que
  rodou sem prender os núcleos. A cópia por tensor sai do `validacao/banda.txt`.
- `-ncmoe`: interpolação linear pelas camadas na CPU entre tudo na GPU F32 e o `--cpu-moe`. O tg128 com metade na CPU
  ficava 3% abaixo dessa reta no 1B e ~7% no 3B.
- Cache do llama.cpp: tudo na GPU + faltas por token × bytes do expert ÷ banda H2D a partir de memória pageable (medida antes,
  com uma cópia de 3–5 MB de memória comum) + camadas × custo de sincronização por camada (fixo, com ou sem falta). O acerto
  sai de uma réplica do SLRU do llama.cpp sobre os traços (script guardado na pasta, com sha), aquecida na conversa 3. Com o
  aquecimento nas 0 a 3, a réplica da rodada 42 deu 0% / 16,5% / 74,4% no 1B e 0% / 26,7% / 81,1% no 3B: com 76 slots
  globais e 192 experts pedidos por token, o SLRU despeja cada expert antes de o token seguinte voltar à mesma camada, e a
  10% nunca acerta.
- Com o aquecimento só na conversa 3, a réplica dá os mesmos acertos (0% / 16,5% / 74,4% no 1B e 0% / 26,7% / 81,1% no
  3B), e o simulador também, para as nossas políticas: com o SLRU e o LRU o estado final depende só dos acessos mais
  recentes, e com o LFU as conversas 0 a 2 já decaíram para menos de 1% do peso. A ressalva de comparação com o C fica fraca.
- Números previstos (`previsao.py`, com SYNC = 20 µs por camada, estimado): cache do llama.cpp 47,2 / 39,8 / 13,9 ms no 1B e
  86,8 / 64,8 / 19,8 no 3B; `--cpu-moe` 12,0 e 22,1; `-ncmoe` 11,2 / 9,5 / 7,0 e 20,4 / 17,5 / 12,9; o nosso `lfu_cpu`
  7,9 / 6,0 / 4,0 e 14,8 / 10,8 / 6,8; o `lru_sync` 33,2 / 18,7 / 11,6 e 65,2 / 36,4 / 17,1.
- A previsão conta três cópias por falta. O llama.cpp junta faltas com expert e slot consecutivos; pela réplica do rev-2
  (rodada 43), isso economiza ~5% no 1B a 10%, sem mudar nenhuma afirmação. Fica como viés conhecido da previsão.
- O contador `hit rate` do destrutor soma o aquecimento, então a réplica é comparada somando a conversa 3.
- O nosso: o simulador, com o mesmo aquecimento, para `lru_cpu`, `lfu_cpu` e `lru_sync`.
- Afirmações registradas antes:
  - (a) o nosso residente é mais lento que o llama.cpp tudo na GPU com KV F32 (o custo fixo já conhecido: ~1,6× no 1B e
    ~1,3× no 3B contra os tg128 F16);
  - (b) o nosso `lfu_cpu` é mais rápido que o cache do llama.cpp nas 6 células do núcleo;
  - (c) o cache do llama.cpp é mais lento que o `--cpu-moe` sem cache a 10% e a 25%; a 50% a previsão fica ~16% acima do
    `--cpu-moe` no 1B e ~11% abaixo no 3B; o `--cpu-moe` variou 11–15% entre as passadas da validação, e a célula é só
    relatada;
  - (d) com o mesmo protocolo, o nosso `lru_sync` contra o cache do llama.cpp: a diferença acompanha a do acerto das duas
    políticas na réplica (a 25%, o LRU acerta mais que o SLRU) e, a 10%, onde os dois acertam 0%, mede só o motor.

**Critério e análise** (`criterio.txt` gravado antes; `analise.py` → `analise.txt`, reproduzível byte a byte):
- Por célula: média, mediana e faixa mín–máx do ms/token, tok/s = 1000 ÷ média e a razão nosso ÷ llama.cpp.
- No núcleo, "mais rápido" só com faixas sem sobreposição; com sobreposição, "empate dentro do ruído".
- Com uma linha de base (uma execução): o valor único é "mais lento" acima de 1,1 × o máximo da faixa do núcleo e "mais
  rápido" abaixo de mínimo ÷ 1,1; no resto, "sem diferença decidida". Entre duas linhas de base, diferença acima de 10%. Os
  10% são uma margem operacional escolhida (o dobro da deriva tolerada), e não uma incerteza medida, porque a dispersão de
  uma linha que roda uma vez não é observada (rodada 43; o texto anterior falava em 2 vezes e o critério em 10%).
- Cada afirmação (a)–(d) é relatada como confirmada, não confirmada (alguma célula contra) ou não decidida (alguma célula
  suspensa e nenhuma contra; rodada 44). A previsão numérica é relatada com o erro por célula, sem portão.
- Deriva: a linha tudo-na-GPU do llama.cpp e o residente do nosso, por repetição; relatar quem passar de ±5% da média.

**Limites já conhecidos:**
- O decode forçado do prompt não é o uso normal do llama.cpp; a linha do servidor cobre esse lado.
- O cache do llama.cpp é experimental, e não ajustei flags além dos equivalentes aos nossos.
- O cache do llama.cpp copia de memória pageable, o nosso de pinned.
- Batch 1, contexto até ~1.100 posições, uma sessão com carga de fundo.
- O aquecimento numa conversa só não se compara com o do C.
- A amostragem do servidor não é igualada no harness, que só lê os logits.

## Comparação com o llama.cpp: resultados (31/10)

Medida de 01h10 a 03h27 de 31/10 (`qual-runs/LLAMACPP-COMPARA-20261030/rodar.sh`), com as 52 execuções saindo com código 0
(`rodar.log`, a saída do roteiro). O `analise.py` reproduz o `analise.txt` byte a byte. O `criterio.txt` foi gravado
em 30/10 às 18h55 e o `previsao.txt`, que leva uma cópia dele, às 01h11min02s, antes do laço de execuções (a primeira
terminou às 01h11min12s), depois das rodadas 42 a 46. Carga de fundo: a
carga de 1 min estava em 0,74 no começo. No começo e no fim, o `ps` (média de CPU desde o início de cada processo, não o uso
durante a medida) mostrava `quickshell` e `localsearch-3` perto de 19% e outros processos entre 6% e 18% (`ambiente.txt`).
Entre execuções, a carga de 1 min ficou entre 1,5 e 10,7 (`carga.log`), com os picos logo depois das execuções que
calculam na CPU.

**Correção.** Nenhuma configuração foi suspensa. As 36 execuções do llama.cpp passam na conferência do log (FA, tipos de K e
V, grupo único com os slots da célula, hit rate), e todas as configurações passam no portão de 99% contra a passada tudo na
GPU F32: o cache fica em 99,40% (1B) e 99,68% (3B) nas três células, com as mesmas argmax nas três, porque o cálculo na GPU
não muda com o acerto; o `--cpu-moe` e o `-ncmoe` ficam entre 99,31% e 99,79%. O nosso `m2a-cache` passa na própria
conferência de logits (limite 1e-4). O hit rate do log do llama.cpp é igual ao da réplica do SLRU em cinco das seis células;
no 3B a 25% dá 26,67% contra 27,45%. A deriva da linha tudo na GPU e do residente ficou dentro de ±0,2%.

**Núcleo** (ms/token, média das 3 repetições, faixa mín–máx; tok/s = 1000 ÷ média):

| modelo | orçamento | llama.cpp cache | nosso `lru_cpu` | nosso `lfu_cpu` | llama.cpp ÷ `lfu_cpu` |
|---|---|---|---|---|---|
| 1B | 10% | 39,20 (38,91–39,42), 25,5 tok/s | 9,75 (9,59–9,97) | 7,76 (7,69–7,88), 128,9 tok/s | 5,05× |
| 1B | 25% | 33,41 (33,19–33,59), 29,9 tok/s | 7,44 (7,31–7,56) | 5,93 (5,89–5,99), 168,7 tok/s | 5,64× |
| 1B | 50% | 12,59 (12,57–12,62), 79,4 tok/s | 4,83 (4,79–4,89) | 3,92 (3,89–3,94), 255,1 tok/s | 3,21× |
| 3B | 10% | 89,94 (89,32–90,69), 11,1 tok/s | 17,27 (16,82–17,73) | 14,23 (14,17–14,30), 70,3 tok/s | 6,32× |
| 3B | 25% | 72,36 (72,02–72,93), 13,8 tok/s | 12,61 (12,40–12,74) | 10,38 (10,12–10,59), 96,3 tok/s | 6,97× |
| 3B | 50% | 22,16 (22,06–22,33), 45,1 tok/s | 7,50 (7,44–7,59) | 6,34 (6,27–6,38), 157,6 tok/s | 3,49× |

Tudo na GPU com KV F32: llama.cpp 1,89 ms (1B) e 3,52 ms (3B); o nosso residente 2,68 e 4,21 (1,42× e 1,20× mais lento).

**Linhas de base** (uma execução, ms/token): llama.cpp GPU F16 1,71 e 3,19; `--cpu-moe` 11,36 e 21,38; `-ncmoe` a 10% (8,3% e
9,4% dos experts na GPU) / 25% / 50%: 10,67 / 9,04 / 6,70 no 1B e 19,75 / 17,01 / 12,64 no 3B; cache cheio 2,54 e 4,78. O
nosso `lru_sync` 33,16 / 18,46 / 11,34 e 63,95 / 35,51 / 16,52.

**Afirmações registradas antes:**
- (a) **confirmada**: o nosso residente é mais lento que o llama.cpp tudo na GPU F32 nos dois modelos (1,42× e 1,20×; contra
  o F16, 1,57× e 1,32×).
- (b) **confirmada**: o nosso `lfu_cpu` é mais rápido que o cache do llama.cpp nas 6 células, de 3,2× a 7,0×.
- (c) **confirmada** a 10% e 25%: o cache do llama.cpp é mais lento que o `--cpu-moe` sem cache. A 50%, só relatada: mais
  lento no 1B (12,59 contra 11,36, acima da margem de 10%) e sem diferença decidida no 3B (22,16 contra 21,38).
- (d) **confirmada** a 10%: com o mesmo protocolo síncrono e 0% de acerto nas duas políticas, o nosso `lru_sync` é 1,18× (1B)
  e 1,41× (3B) mais rápido. O custo incremental efetivo por falta, inferido pela diferença dos tempos totais a 10% (não medido
  por componente), é (33,16 − 2,68) ÷ 192 = 0,159 ms no 1B e (63,95 − 4,21) ÷ 256 = 0,233 ms no 3B no nosso, compatível com
  uma cópia pinned do expert inteiro (0,159 e 0,239 ms no `banda.txt`); no llama.cpp, a mesma conta dá 0,194 e 0,338 ms.
  A 25% o LRU acerta mais (48,1% e 47,7% contra 16,4% e 26,7%) e a diferença cresce para 1,81× e 2,04×; a 50%, com acertos
  parecidos (71,6% contra 73,3% no 1B), fica em 1,11× e 1,34×.

**Relatado, sem afirmação registrada antes:** a melhor linha do llama.cpp em cada orçamento é o `-ncmoe`, e não o cache.
Contra ela, pela regra das linhas de base (valor único acima de 1,1 × o máximo da faixa do núcleo), o nosso `lfu_cpu` é
mais rápido nas 6 células: 1,38× / 1,52× / 1,71× no 1B e 1,39× / 1,64× / 1,99× no 3B. A 10% o `-ncmoe` usa um pouco menos
VRAM (8,3% e 9,4%), a favor do nosso.

**Previsão contra o medido.** As nossas linhas ficaram de 0,2% a 7,9% abaixo da previsão. O cache do llama.cpp errou mais:
−17% / −16% / −9,5% no 1B e +3,6% / +11,7% / +12,0% no 3B. Nada abaixo foi isolado por medida; são contas sobre os tempos
totais (`analise.txt`, `previsao.txt`, `banda.txt`), na mesma base para os dois modelos (cópias mais a sincronização por
camada):
- A 10%, o custo incremental por falta é 0,194 ms contra 0,236 previstos no 1B (−18%) e 0,338 contra 0,325 no 3B (+4%). No
  1B, a junção de cópias consecutivas explicava ~5% pela réplica do rev-2 (rodada 43), e o resto não está explicado.
- O cache cheio mede a sobrecarga por camada sem falta: 2,54 − 1,89 = 0,65 ms no 1B (0,027 ms por camada) e 4,78 − 3,52 =
  1,26 ms no 3B (0,039 ms por camada), contra os 0,020 ms por camada estimados. É o erro de +5,0% e +13,3% dessa linha.
- Descontada essa sobrecarga e usado o hit rate do log, o custo por falta fica em 0,191 / 0,192 / 0,196 ms no 1B e sobe de
  0,333 a 10% para 0,360 a 25% e 0,346 a 50% no 3B. Isso é coerente com a réplica da rodada 43, em que o llama.cpp junta
  cópias a 10% (0,83 cópia por tensor) e não junta a 25% e 50%, mas não prova a causa. O hit rate do log inclui o
  aquecimento, e os tempos são só da medida. Sem os contadores separados por fase, não dá para saber o valor nem a direção
  dessa correção no custo por falta: na réplica do 3B a 25%, o aquecimento acerta mais que a medida (30,6% contra 26,7%;
  rodada 48).

As linhas de base do llama.cpp ficaram de 1,6% a 5,5% abaixo da previsão, exceto o cache cheio.

**Linha do servidor** (`llama-server`, greedy, temperatura 0, seed 1, max_tokens 1024; ms/token só da resposta): tudo na GPU
1,96 e 3,59; cache a 10% / 25% / 50%: 38,90 / 36,80 / 12,75 no 1B e 90,16 / 63,17 / 21,68 no 3B. Ficam a até 13% do
decode forçado (1B a 25% +10%, 3B a 25% −13%), com outros textos e outra sequência de experts. As quatro conversas divergem
do texto do traço em todas as configurações; o caractere da primeira divergência fica no `servidor-*.txt` (tudo na GPU no 1B:
295, 1.873, 220 e 227). Os textos dos traços também saíram do `llama-server` (M0.5, `tools/gen_trace_texts.py`, por
`/v1/chat/completions`), e a conversa foi retokenizada pelo HF; esta linha manda os ids do prompt do traço para
`/completion`. Os dados não isolam a causa da divergência: a diferença de ids depois da retokenização, a configuração do
servidor e o cálculo numérico são possibilidades, e a margem no ponto de divergência não foi medida.

**Limites:**
- Método: decode forçado, um token por chamada, com o prompt incluído e cronometrado como decode nos dois motores. O
  prefill em lote do llama.cpp não foi medido; a linha do servidor mede só a resposta.
- Precisão: KV F32 e FA desligado no núcleo, para igualar ao nosso. O padrão do llama.cpp (KV F16 com FA) é 9–10% mais
  rápido tudo na GPU (1,71 e 3,19 ms), e essa vantagem não entra nas células com cache.
- Amostragem: o harness só lê o argmax dos logits; a linha do servidor usa o amostrador do llama.cpp em greedy, sem
  igualar ao nosso, e a divergência contra os traços é registrada pelo caractere, não pela posição do token.
- O cache do llama.cpp (commit 0893ec9) é experimental, copia de memória pageable (o nosso, de pinned) e não teve flags
  ajustados além dos equivalentes aos nossos.
- As linhas de base rodaram uma vez; o `--cpu-moe` variou 11–15% entre as passadas da validação (12,0–13,4 e 22,1–25,4 ms) e
  saiu 11,36 e 21,38 na medida, abaixo de todas elas. A margem de 10% é operacional, e não uma incerteza medida.
- Batch 1, contexto até ~1.100 posições, aquecimento só na conversa 3, uma sessão com carga de fundo.
- A comparação com o Strata ficou fora deste escopo.

## Decisão do Breno (31/10, 12:11): terminar o plano com o llama.cpp como base, e o Strata depois

O plano segue até o fim com o llama.cpp como única base de comparação de velocidade. A comparação com o Strata volta
depois, como etapa própria. O Strata continua como referência de código (regra de 29/10).

**Por que a comparação com o Strata não cabe agora** (pesquisa de 31/10, sem executar nada; código do Strata em
`origin/main` 46c152a1):
- Nenhum modelo roda hoje nos dois motores sem mudança. O nosso aceita só `granitemoe` em BF16
  (`rust/moe-rt/src/model.rs:37–38`, `rust/moe-rt/src/weights.rs:111–112`). O Strata aceita só `qwen4exp` e `qwen35moe`, sempre
  quantizado em GGUF, e não lê F16 nem BF16 nas matrizes (`src/kernels/cuda/native_mmvq.cu:3443–3448`).
- No `qwen35moe`, a geometria vem do GGUF (`include/strata/core/family_layout.hpp:31–94`), com estas exigências:
  - cabeça 256, estado do SSM 128, conv 4 (`src/qwen35/q36_engine.cu:972`, `constexpr int HD = 256` em `:304`);
  - camadas GDN, shared expert e `output.weight` separado;
  - a chave `qwen35moe.nextn_predict_layers` (`include/strata/artifact/gguf_reader.hpp:645–664`). O MTP em si é opcional
    (`q36_engine.cu:1031`).
- **Candidato para quando voltarmos:** Groot-19B-A3B-261023 (`qwen35moe`, 19,4B no total, ~3B ativos, 128 experts, 40
  camadas).
  - O GGUF Q8_0 publicado (20,2 GB) tem as chaves e os tipos que o Strata exige, então deve rodar sem mudança. É leitura
    do código, ninguém executou, e com 128 experts o roteador cai num caminho genérico nunca testado com esse número
    (`q36_engine.cu:1799,2007`).
  - No nosso lado o esforço é grande: suporte a `qwen35moe` (Gated DeltaNet só para decode, atenção com porta, Q/K norm,
    RoPE parcial, shared expert, `lm_head` separado, carregador). Os experts empilhados têm o formato que o Granite já usa.
  - Alternativa com tudo residente: o REAP-64 do Qwen3.6 (10,9B, só BF16), que exige converter para GGUF com a chave
    `nextn`.
- **O mesmo modelo não basta para uma comparação justa.** O decode é limitado por memória, e Q8_0 (~1,06 byte por
  parâmetro) contra BF16 (2 bytes) mede o formato, não o motor. Uma das pontas precisa de outro tipo de peso: Q8_0 no
  nosso runtime ou BF16 no Strata, cada uma com esforço médio.

## A0 — teto da busca antecipada e desenho do motor sem parada por camada (31/10)

Origem: consulta ask-8 (`.herdr/ask/strata-opencode-8/`). Os dois consultores convergiram na etapa 4 da ordem aprovada,
dividida como o rev-2 propôs: A0 sem GPU (A0a, teto da busca antecipada no simulador; A0b, desenho do protocolo), A1 (motor
sem parada por camada, top-k na GPU, um lançamento para todos os experts, admissão do `lfu_cpu`, sem previsão) e A2 (busca
antecipada, só se a A0a mostrar ≥ 5% de redução da espera com um preditor realista). Controles pedidos pelo rev-1: uma
linha com a busca desligada, os ids dos traços sem servir de oráculo para a estratégia medida, a ausência de relato do
`racecheck` não conta como prova, e as tolerâncias gravadas antes.

### A0a — teto da busca antecipada no simulador

Material em `qual-runs/A0-PREFETCH-SIM-20261031/`: `criterio.txt` (gravado antes de rodar nos traços de teste, com a
previsão), `prefetch_sim.py` (o roteiro do critério) e `resultado.txt`, que saiu do simulador `0aee964c`. Depois da
primeira versão do protocolo `token` (A0c), a A0a e o diagnóstico foram reexecutados com o simulador `493e5dd6` e saíram
iguais byte a byte, tirando a linha do sha (`reexecucao.txt`). As correções seguintes do simulador (rodadas 51 e 52) só
tocam o ramo `token`. Na rodada 51, o rev-2 conferiu a igualdade exata fora dele, em 432 combinações contra o `HEAD`.

**Extensão do simulador** (`tools/moe_sim.py`, opção `prefetch`; sem ela o simulador dá os mesmos números de `fddc2cc`, com
um teste de valores fixos): a cada camada, o preditor aponta os experts da camada `d` passos à frente, que entram na fila
do PCIe como cópia em segundo plano (mesmo backlog e mesma banda disputada da admissão) e ficam reservados até o alvo
passar. Preditores: `anterior` (o roteamento do token anterior na camada do alvo, só o passado), `frequente` (os k de maior
frequência com decaimento na camada do alvo) e `oraculo` (os experts do alvo, só como limite). Admissão `livre` ou `lfu`.
`espera`: a stream de cálculo espera o evento de uma cópia em voo que termina antes da CPU, o que o protocolo sem parada
por camada permite; é relatada sozinha também, porque é parte do A1 com ou sem busca. Testes novos em
`tests/test_moe_sim.py`: orçamento nunca excedido com busca, o oráculo com camada longa usa o que traz, o preditor
`anterior` só lê o passado, um expert reservado não é despejado antes do uso, e a espera do evento nos dois sentidos.

**Resultado (31/10, `resultado.txt`, simulador `0aee964c97e8ba8a`, critério `b979b9b2e46eed71`): o portão não passa, como a
previsão dizia.** A busca antecipada sai do caminho, e o A1 segue sem ela.

- A configuração escolhida no treino foi a mesma nos dois modelos: `anterior`, d = 8, admissão `lfu`. No teste ela reduz a
  espera em +0,30 / +0,25 / +0,33% no 1B e +0,16 / +0,14 / +0,18% no 3B (10 / 25 / 50%), longe dos 5%.
- O próprio oráculo, que é o limite superior, não chega a 5% em nenhuma célula:
  - com admissão `lfu`, o melhor é +1,30% (1B a 50%, d = 8);
  - com admissão `livre`, ele perde em todas as células, de −9,9% (3B a 50%, d = 8) a −85% (1B a 50%, d = 1). Ele acerta
    mais (até +7 pontos de acerto), mas põe tráfego em 98–100% das camadas, e o simulador cobra de cada uma o `o_layer` e
    a CPU disputada. Com d = 1 quase nada chega a tempo (0,2–0,5 cópia útil por token).
- A espera do evento sozinha dá +0,00% em todas as células (previsão: < 2%).

Conferências feitas depois (`diagnostico.py`, `diagnostico.txt`; não decidem o portão):
- **Por que a busca com `lfu` quase não copia.** De ~500 mil propostas no 1B a 10%, sobram 157 cópias com o `anterior` e 6
  com o oráculo. As outras caem por backlog cheio ou porque a frequência do candidato não vence a do residente mais fraco:
  o `lfu_cpu` já guarda os experts mais frequentes. O ganho de +0,1–0,3% da escolhida não vem das cópias da busca. Os
  números apontam para as reservas barrando admissões: no 1B a 10%, as cópias de admissão caem de 1,5 para 1,3 por token
  e as camadas com tráfego de 6,4% para 5,8%, ou seja, menos tráfego (sem contador próprio que isole a causa).
- **Por que a espera do evento não muda nada.** Em nenhuma célula uma chave pedida estava em voo: 0 de 662.592 pedidos no
  1B e 0 de 976.384 no 3B. Toda cópia de admissão termina antes de a chave voltar a ser pedida, o que só acontece um
  token depois, na mesma camada.
- **Para a A0c.** Na virada do token, ainda estão em voo de 1,0% a 3,6% das cópias de admissão (no máximo 345 em 3.814
  tokens no 3B a 50%). Com a residência publicada só no começo do token, essas cópias atrasam um token. Se cada uma custar
  um expert na CPU (~0,05 ms), isso dá menos de 0,005 ms por token, contra uma espera de 1–10 ms por token. Fica de fora
  desta conta o atraso das cópias para vítimas de camadas ainda não passadas no token, que a A0c mede.

**Limites:**
- **O resultado depende de dois termos do custo do tráfego, e só vira se os dois sumirem** (`sensibilidade.py`,
  `sensibilidade.txt`). Os termos são o `o_layer` (uma sobra medida por camada com tráfego, sem causa isolada) e a CPU
  disputada (`c_cpu` contra `c_cpu_quiet`, +14–16% por expert, medida). Com a busca livre e d = 8:
  - com `o_layer = 0` e a disputa mantida, o oráculo passa de 5% só a 50% (+5,3% no 1B e +12,2% no 3B), e os preditores
    realistas ficam todos ≤ 0;
  - com a disputa zerada e o `o_layer` calibrado (f = 0), nada passa, nem o oráculo;
  - só com os dois zerados os realistas passam de 5%: `anterior` a 50% (+5,8% no 1B, +6,2% no 3B) e `frequente` a 10%
    (+5,2% no 1B, +5,4% no 3B).
  - As duas primeiras variantes são os mesmos termos que o rev-2 mediu na rodada 50 (achado 5), aqui com a espera
    desligada. Os dele, com a espera ligada, diferem até 0,8 ponto no 3B. A varredura da disputa com o `o_layer` calibrado
    é nova.
  - A varredura é evidência sobre as variantes amostradas (f = 0; 0,25; 0,5; 0,75), e não prova que todo limiar exige os
    dois termos exatamente nulos.
- **Duas regras de admissão testadas** (`livre` e `lfu`). Uma regra intermediária, como a de admitir só no lugar de um
  residente cujo próximo uso está mais longe que o alvo, não foi testada. O oráculo `livre` mostra que acertar mais não
  paga o tráfego neste modelo de custo, e não que nenhuma regra pagaria.
- **Os preditores usam só o roteamento.**
- **Reabertura:** a decisão volta à mesa se a medida do A1 mostrar os dois termos perto de zero no motor novo (critério do
  A1, abaixo).

### A0b — o protocolo sem a parada por camada

**O que para hoje, e por quê.** Cada camada do `ExpertCache` sincroniza a stream de cálculo para ler os logits do roteador
(`Epoch::router_logits`). É dessa parada que vêm o top-k no host, a escolha do que vai à CPU, a vítima com a stream parada
(o `Idle`) e a garantia de que nenhum kernel ainda lê o slot que a cópia vai sobrescrever. Tirar a parada tira as quatro
coisas ao mesmo tempo.

**Como o Strata faz** (`/home/brenoperucchi/Devs/GitHub/strata`, 9259cad4, lido e não executado):
- `include/strata/core/session.hpp:384–420` e `src/core/session.cpp:820–900`: o token inteiro vira um grafo. Em cada
  camada, o roteador e o top-k rodam na GPU e um kernel publica `x`, ids e pesos em memória host mapeada e toca um número
  de sequência (o "doorbell", `include/strata/core/layer.hpp:396–414`). O host espera a sequência, calcula as faltas na
  CPU, escreve a saída em memória mapeada e escreve uma flag; um kernel de uma thread (`doorbell_wait`) segura a GPU até a
  flag chegar. Não há sincronização de stream no laço das camadas.
- A residência é **fixa durante o token** (`TokenHits`, `session.hpp:402–420`): uma tabela `n_layers × n_expert` na GPU
  (slot ou −1); o `moe_hit_select` monta a lista de acertos da camada na GPU a partir dela, e o host lê a própria cópia da
  mesma tabela para saber o que calcular.
- As trocas só entram entre janelas (`generate.cpp:5689–5756`): cópia numa stream própria, despejo lógico na hora, e
  `apply_pending` no começo da janela seguinte por `cudaEventQuery`, seguido de um `cudaMemcpy` síncrono da tabela para a
  GPU. A garantia de que nenhum kernel lê o slot reescrito vem de as cópias rodarem entre rodadas, com a GPU parada; o
  buraco anotado no B0 (a cópia da tabela na GPU ainda aponta o slot reescrito enquanto a troca está pendente) continua
  lá.
- Os comentários registram duas medidas que valem para nós como cuidado, não como fato: no Windows (WDDM) uma espera do
  host que só lê memória nunca deixa o driver submeter o trabalho, e um `cudaEventRecord` dentro de uma captura de grafo é
  descartado em silêncio. Não usamos grafo; no Linux a primeira é para conferir no A1, não para assumir.

**Fato que simplifica o desenho.** A chave do cache é (camada, expert). Um expert admitido na camada `l` do token `t` só
volta a ser pedido na camada `l` de um token seguinte. Fixar a residência durante o token não atrasa o uso de nenhuma
admissão que termine antes do fim do token; só perde as cópias que terminam entre o começo do token `t+1` e a camada `l`
dele, que o `settle` por camada de hoje aproveitaria. A A0c (abaixo) mediu esse custo no simulador.

**Desenho proposto para o A1 (D1, residência fixa por token; revisto na rodada 51):**

1. **Doorbell em memória mapeada, sem kernel de espera.** As bindings já têm `cuStreamWaitValue32_v2` e
   `cuStreamWriteValue32_v2`, e o `cuda-async` já usa memória host mapeada com `cuStreamWriteValue32`
   (`/home/brenoperucchi/Devs/GitHub/cuda-rust/cuda-async/src/reactor.rs:96–113,243`).
   - **Por camada:**
     - o roteador e o top-k na GPU escrevem ids, pesos e `x` numa área pinned mapeada da camada;
     - `cuStreamWriteValue32(seq[l], t)` avisa o host;
     - a GPU enfileira os acertos;
     - `cuStreamWaitValue32(flag[l], t, GEQ)` segura a soma da saída da CPU.
   - **No fim do token,** depois dos experts da última camada, um `seq[L]` avisa o host de que a GPU passou dela.
   - **Sem anel:** a área de cada camada só é reescrita no token seguinte, depois de o host ter servido a camada (a GPU
     esperou a flag dele).
   - **O `GEQ` compara de forma cíclica** (`(int32_t)(*addr − value) >= 0`, `/opt/cuda/include/cuda.h:535`). Por isso o
     contador de tokens começa em 1, com as flags zeradas (com 0, a espera do primeiro token passaria antes de a CPU
     responder), e todo valor escrito fica na meia faixa à frente do valor esperado (passo 6).
   - **Ordem de memória:**
     - do device para o host, o `cuStreamWriteValue32` sem `NO_MEMORY_BARRIER` torna visíveis as escritas anteriores da
       stream antes da sequência (a v2 nem aceita a flag);
     - do host para o device, a saída da CPU fica em memória host mapeada, então a flag `FLUSH` do `cuStreamWaitValue32`,
       que trata de escritas remotas na memória do device, não é o mecanismo em jogo. O host escreve a saída e depois a
       flag com `Release`.
   - **Primeiro portão do A1, num micro-teste antes de qualquer outra peça:**
     - o atributo do device para esperar valor em memória host;
     - a latência da volta;
     - a GPU ler o `y` novo, e não um antigo, logo depois da flag, com um padrão diferente escrito a cada rodada.
2. **Duas linhas no host, com posse separada.**
   - **A linha principal é dona da stream de cálculo** e é a única que enfileira nela: a tabela do token, as camadas e a
     leitura dos logits.
   - **A linha de serviço é dona do estado do cache** (slots, tabela, frequências, admissões), da stream de cópia (a única
     que enfileira nela) e do `TokenEpoch`.
   - **Por token, a linha de serviço:**
     - espera `seq[l]` para cada camada;
     - lê ids e pesos publicados pela GPU, que são a fonte única do roteamento (o host não recalcula o top-k);
     - calcula as faltas;
     - escreve a saída e a flag;
     - só então decide as admissões, fora do caminho crítico da GPU.
   - **A virada do token:** ao ver `seq[L]`, a linha de serviço enfileira as cópias pendentes da última camada (passo 4),
     faz o `settle` por consulta, monta a tabela do token seguinte e manda à principal, por canal, um `Published` (o lado
     do buffer pinned com a tabela, a geração e os eventos `ready` dos slots recém-admitidos).
   - **A principal,** depois de ler os logits do token, espera o `Published`, enfileira o envio da tabela e as esperas
     pelos `ready` e só então enfileira o token seguinte.
   - **As duas streams se ordenam por observação no host:** a cópia depois de o host ver o doorbell (passo 4), o uso
     depois de o host ver o `ready` completo (passo 3).
   - **A única espera por evento entre as linhas é a do `ready`, pela principal.** Ela é segura por dois motivos:
     - a linha de serviço grava o `ready` antes de mandar o `Published`, e o canal ordena;
     - o mesmo `ready` só pode ser regravado depois de a principal fazer a espera. Para virar vítima no token `t+1`, o slot
       precisa de uma camada servida, e o `seq[0]` de `t+1` só vem depois de a principal enfileirar a tabela, as esperas e
       a camada 0.
     A espera de um evento capturado na hora errada foi o P1 da rodada 50, por isso o argumento fica escrito, e o backend
     falso tem um mutante que manda o `Published` antes de gravar o `ready`.
3. **Tabela de residência por token.**
   - A tabela vai para a GPU na stream de cálculo antes do primeiro trabalho do token, a partir de um buffer pinned em
     dois lados. Um lado só volta para a linha de serviço depois de o envio que o leu terminar (a principal devolve o
     lado depois da leitura dos logits do token que o usou).
   - A tabela do host é a mesma e fica imutável até o fim do token: um acerto do host é um acerto da GPU, sem corrida.
   - Uma chave só entra na tabela se o `ready` dela já terminou na consulta.
   - Por precaução, a stream de cálculo também espera o `ready` de cada slot recém-admitido. O evento foi gravado pela
     linha de serviço antes de mandar o `Published`, e o canal dá a ordem entre as duas linhas. Custa nada quando o evento
     já terminou.
4. **A sobrescrita espera o último leitor, pelo doorbell.**
   - Com a tabela fixa, o último leitor do slot de uma chave da camada `l'` no token `t` é o lançamento dos experts da
     camada `l'` desse token.
   - **Se a vítima é de uma camada `l' < l`** (já passada no token), a linha de serviço enfileira a cópia na hora: ela viu
     `seq[l]`, gravado depois dos experts de `l'` na mesma stream.
   - **Se `l' ≥ l`,** a vítima continua na tabela do token, e a linha de serviço guarda a cópia `(l', chave)`. Ela só a
     enfileira ao ver `seq[l'+1]` (ou `seq[L]` para a última camada), que a GPU só escreve depois dos experts de `l'`.
   - Não há evento de ordem entre as streams. A fila da stream de cópia segue a ordem em que a linha de serviço
     enfileira, e o simulador modela essa ordem (A0c).
   - A vítima sai da tabela do token `t+1`, e a chave nova só entra numa tabela depois do `ready`.
5. **Um lançamento para todos os acertos.**
   - Os slots viram um banco único (`SlotId` vira índice no banco, como o B0 previu).
   - Um kernel de seleção monta na GPU a lista de acertos da camada (slot, peso, contagem) a partir da tabela e dos ids,
     e um kernel agrupado lê a lista. O cutile já lê dados do device como escalar (`attn_split` com `lens`) e tem views
     de gather.
   - A soma dos experts muda de ordem: os logits deixam de ser bit a bit iguais aos de hoje e passam a ser comparados pela
     tolerância (1e-4).
6. **Erro, cancelamento e `Drop`.** Uma GPU presa num `cuStreamWaitValue32`, ou uma linha de serviço presa esperando um
   `seq` que nunca vem, trava o processo. Por isso:
   - **Valor de soltura:** é `t + 2^30` para o token corrente `t`, dentro da meia faixa à frente de toda espera em curso.
     Nunca um valor "máximo", que o `GEQ` cíclico não aceita como maior.
   - **Cancelamento:** um atômico que as duas linhas olham. A espera da linha de serviço por `seq[l]` também o consulta e
     volta com erro quando ele é marcado.
   - **Se a linha de serviço falha** (erro devolvido ou `panic`), uma guarda dela escreve todas as flags com o valor de
     soltura, marca o cancelamento e envenena o cache. A principal, que pode estar parada na leitura dos logits, é solta
     pela GPU e vê o envenenamento.
   - **A linha de serviço sincroniza a própria stream de cópia ao sair,** por qualquer caminho (o normal, o cancelamento
     ou a guarda do `panic`), depois da última cópia que enfileirou. Ela é a única que enfileira nessa stream, então é ela
     quem drena. O cancelamento é cooperativo e não prova sozinho que ela parou de enfileirar.
   - **Se a principal falha no meio da enfileiração,** ela marca o cancelamento, escreve as flags com o valor de soltura,
     faz o `join` da linha de serviço e sincroniza a stream de cálculo.
   - **No `Drop`,** nessa ordem:
     1. cancelamento;
     2. flags com o valor de soltura;
     3. `join` da linha de serviço, que já drenou a stream de cópia;
     4. sincronização da stream de cálculo;
     5. só então a liberação das regiões mapeadas, dos buffers pinned e do banco de slots.
   - **A drenagem do núcleo de hoje fica.** O `Slots::drop` sincroniza as streams antes de soltar o armazenamento
     (`rust/moe-rt/src/cache/core.rs:433–437`, com o `sync_all` em `:424–429`). A drenagem não pode sumir na troca para o banco de slots.
   - **Testes:**
     - no `loom`, as quatro saídas: a linha de serviço morre, a principal morre, e o `Drop` no meio do token nos dois
       casos. Toda espera enfileirada é solta e toda linha termina. Não se exige que nenhuma flag fique abaixo do valor de
       soltura: a linha de serviço, antes de ver o cancelamento, pode responder uma camada com `t` depois da soltura, e só
       o token `t` tem esperas em fila;
     - no backend falso e no `loom`, o `Drop` com a linha de serviço parada entre a flag e a enfileiração de uma cópia.
       Nenhuma operação fica em fila em stream nenhuma quando a memória cai. O mutante com a ordem antiga (sincronizar
       antes do `join`) tem de ser acusado;
     - na GPU, um teste com `t` perto de `2^31`, para o valor de soltura;
     - na GPU, um teste em que a linha de serviço entra em `panic` no meio do token, e a chamada da principal volta com
       erro em menos de 1 s.
7. **O modo antigo fica.**
   - O `ExpertCache` com a parada por camada continua como está, inclusive o `AdmitMode::Barrier`, e tem de sair bit a bit
     igual antes e depois do A1.
   - O motor novo ganha um modo determinístico próprio (`AdmitMode::TokenBarrier`): a linha de serviço sincroniza a stream
     de cópia antes do `settle` da virada, e o backlog conta todas as cópias do token. Nele, acertos, cópias e experts na
     CPU têm de bater exatamente com o simulador no protocolo `token` com barreira (A0c).

**Tipos (o que a tese 1 ganha ou perde).** O `Idle` deixa de existir, porque a vítima não é mais escolhida com a stream
parada. No lugar, todos na linha de serviço, que é dona do cache:
- **`TokenEpoch`:** sai de `begin_token` e é consumido por `end_token`. É a tabela do token. `lookup` e a escolha de vítima
  pedem `&TokenEpoch`, e usá-lo depois do `end_token` é E0382.
- **`LayerView<'t>`:** sai de `serve_layer`, que espera `seq[l]`, e empresta o `TokenEpoch`. Traz ids, pesos e `x`
  publicados, e prova que a GPU passou pelo roteador de `l` neste token.
- **`answer(view, y) -> Served<'t>`:** consome a `LayerView` e escreve a saída e a flag uma vez só (E0382 na segunda vez).
  Devolve um `Served`, que guarda a camada e os ids e continua provando que a GPU passou pelo roteador de `l`.
- **A escolha de vítima pede `&Served` e `&Pins`.** As admissões vêm depois da flag, fora do caminho crítico.
- **A cópia, `copy_after_reader(victim, key, &Served)`,** compara a camada da vítima com a do `Served`: se ela já passou,
  enfileira na hora; senão, guarda a cópia, que `serve_layer(l'+1)` (ou o `seq[L]`) enfileira. A comparação é em tempo de
  execução, porque a camada é um número, e fica dentro do núcleo.
- **A principal só vê um `Published`.** Enfileirar um token pede o `Published` dele, e um `Published` de outra geração é
  recusado em tempo de execução.

Casos de `trybuild` novos, cada um com o gêmeo que compila:
- vítima sem `Served` (o gêmeo: vítima depois do `answer`);
- `lookup` depois do `end_token`;
- responder a mesma camada duas vezes;
- enfileirar um token sem `Published`.
Ficam os 6 de hoje que ainda se aplicam.

**Controles negativos** (janela forçada, como no B0, com o portão de `cuLaunchHostFunc`; 20 de 20 nos dois sentidos, sem
depender da velocidade relativa das streams):
- **Cópia enfileirada na escolha da vítima.** O mutante é a cópia para uma vítima de camada `l' > l` enfileirada na hora,
  como no desenho da rodada 50. Com o portão segurando a stream de cálculo antes dos experts de `l'` e a cópia livre para
  terminar:
  - no mutante, a saída de `l'` sai bit a bit a do expert novo;
  - na versão certa, a cópia não é enfileirada enquanto o portão segura a GPU antes de `seq[l'+1]`, e a saída sai bit a
    bit a do expert antigo.
- **Tabela publicada sem o `ready`.** São dois testes, com contratos separados (achados da rodada 50):
  - **(a) Estado:** a publicação recusa slot `Loading`. É um teste de unidade da montagem da tabela, sem GPU.
  - **(b) GPU:** o mutante tira as duas proteções, a consulta na publicação e a espera da stream de cálculo pelo `ready`.
    Com o `hold_copies` segurando a stream de cópia antes da cópia, o mutante lê o slot com o expert antigo inteiro, e a
    saída sai bit a bit a do expert antigo. A versão certa não lista a chave, que vai à CPU, e a saída sai bit a bit a
    do expert novo calculado na CPU.
  - Tirar só a consulta não basta para errar: a espera pelo `ready` segura a GPU até a cópia inteira terminar. O teste
    registra esse mascaramento, em vez de tratá-lo como insensibilidade.
  - A leitura de um slot pela metade (um portão entre os dois `memcpy` de `copy_raw`) fica como extensão opcional, e não
    como critério.
- O `memcheck` fica, para acesso inválido. O `racecheck` não prova nada aqui (memória global entre streams, como
  registrado no B0).

**O que o Rust dá a mais, e entra no A1** (pedido do Breno, 31/10, depois de comparar com o que o Strata e o llama.cpp
garantem à mão):
- **Tipos por espaço de memória.**
  - `Pinned<T>` (origem das cópias H2D), `Mapped<T>` (pinned e mapeada, com um endereço do host e outro do device) e
    `Device<T>`, cada um com o seu construtor. As funções de cópia e o doorbell só aceitam o tipo certo.
  - O Strata confere isso em tempo de execução (`src/core/session.cpp:836–839`, "the parts staging is not mapped pinned
    memory"), e o cache do llama.cpp copia de memória pageable sem aviso.
  - As flags do doorbell são `Mapped<AtomicU32>`, alinhadas e vivas até o `join` (passo 6), com ordem de memória
    explícita no lado do host: escrita da saída da CPU e depois a flag com `Release`; leitura da sequência com `Acquire`
    antes de ler ids, pesos e `x`. Isso substitui o `volatile` do Strata.
  - O lado do device fica com a semântica do `cuStreamWaitValue32`/`cuStreamWriteValue32` (passo 1), documentada no
    núcleo `unsafe`.
- **`loom` nas duas linhas do host.**
  - As duas linhas compartilham só o doorbell, o canal do `Published`, a devolução dos lados do buffer e o cancelamento.
    Esse estado fica atrás de uma interface pequena e genérica sobre os atômicos (`std` na build normal, `loom` no teste).
  - A GPU entra no modelo como uma terceira thread, que segue a ordem da stream de cálculo: publica ids e `x`, escreve
    `seq` e espera a flag com o `GEQ` cíclico de verdade.
  - O modelo carrega os dados (ids, `x`, `y`) além das flags, para que o mutante com `Relaxed` apareça como dado velho
    lido, e não só como ordem de chamadas.
  - O teste percorre todas as intercalações de um token curto (2–3 camadas, dois tokens) e exige:
    - cada camada respondida uma vez e na ordem;
    - nenhuma leitura de ids antes da sequência;
    - nenhuma flag antes da saída da CPU, e a GPU lê o `y` daquela camada e daquele token;
    - nenhuma cópia para uma vítima antes do `seq` da camada seguinte à dela;
    - as saídas de erro do passo 6, com a invariante de lá (toda espera solta, toda linha termina, nada em fila quando a
      memória cai).
  - Ele não prova a ponte de memória do CUDA, que fica com o micro-teste do passo 1.
- **Backend falso por trait, com teste de propriedade.**
  - O protocolo (tabela por token, vítima, cópias guardadas e liberadas pelo doorbell, `ready`, publicação) fica genérico
    sobre um `trait Backend` com as operações de stream: copiar, gravar evento, esperar evento, consultar evento, lançar,
    publicar a tabela e escrever e esperar valor.
  - A implementação de teste é uma GPU de mentira, que guarda o conteúdo de cada slot como o id do expert e marca um slot
    em cópia como inválido. Cada stream é uma fila FIFO. A espera por um evento captura a última gravação feita **antes da
    chamada**, como no CUDA, e não uma gravação futura. A consulta de evento segue a mesma regra: olha a última gravação
    feita antes da chamada, e um evento nunca gravado responde completo, como o `cudaEventQuery`. O `ready` de cada slot
    é criado uma vez (`rust/moe-rt/src/cache/core.rs:154`), então ligar a consulta à cópia, e não à gravação, esconderia
    o caso em que ela vê uma gravação velha (achado do rev-2 na rodada 52). As filas avançam em qualquer ordem que
    respeite as esperas.
  - As duas linhas do host rodam intercaladas pelo `proptest`, que gera roteamentos, orçamentos, a intercalação das linhas
    e a ordem das filas.
  - Duas invariantes:
    - todo lançamento lê o expert que a tabela do token diz, nunca um slot reescrito ou em cópia;
    - no modo com barreira, acertos e cópias batem com o simulador no protocolo `token`.
  - Esta é a proposta do rev-1 que ficou fora do B0. Com a D1, as ordens possíveis entre as streams e entre as linhas
    passam a importar, e ela entra.
  - Os mutantes dos controles negativos (a cópia enfileirada na escolha da vítima; a tabela sem as duas proteções do
    `ready`) têm de cair nele também, assim como o `Published` mandado antes de gravar o `ready`, a consulta do `ready`
    antes da gravação dele e a ordem antiga do `Drop`.

O que fica de fora do Rust e continua com testes na GPU: a ordem real entre streams e os kernels. Os controles negativos
com janela forçada continuam. E os tipos do Rust impedem o uso duplo, mas não obrigam o uso: uma camada nunca respondida
trava a GPU, o que continua com o cancelamento e o valor de soltura do passo 6.

**Alternativas pesadas e não adotadas:**
- **D2, a GPU decide os acertos por camada,** com flags de `ready` escritas pela stream de cópia, e publica a decisão para o
  host. Ela admite mais cedo, no meio do token, mas a tabela passa a ser escrita pelo host enquanto a GPU lê, e a decisão
  da GPU vira a única fonte da verdade sobre quem calcula cada expert.
  - O ganho sobre a D1 se limita às cópias que terminam entre o começo do token e a camada delas. A vítima que segue
    legível até o fim do token, de onde vem o ganho da A0c sobre o protocolo de hoje, a D2 também teria.
  - O teto desse ganho, pela A0c (as cópias em voo na virada, cada uma a um expert na CPU), fica entre 0,07% e 1,06% da espera
    (`resultado.txt`, linha "teto da D2" de cada célula).
  - Mais corridas a fechar para esse teto: a D2 fica de fora.
- **Copiar na hora só para vítimas de camadas já passadas.** Restringe a política de despejo (na camada 0 só sobram os
  experts não pedidos dela) e muda a `lfu_cpu` validada. Guardar a cópia até o doorbell da camada seguinte dá o mesmo para
  `l' < l` e não restringe nada.
- **Evento `E[t, l']` gravado pela principal e esperado pela linha de serviço** (o desenho da rodada 50). Exige uma
  barreira de submissão entre as duas linhas (a principal publicaria "gravado até (t, l)" com `Release`, e a de serviço só
  esperaria depois de ver esse valor com `Acquire`). O doorbell já dá a mesma garantia sem estado novo.

### A0c — o custo da residência por token no simulador (31/10; corrigida nas rodadas 51 e 52)

Material em `qual-runs/A0C-TOKEN-SIM-20261031/`:
- `criterio.txt`, gravado antes de rodar, com a previsão;
- `token_sim.py`, `resultado.txt`, `diagnostico.py` e `diagnostico.txt`;
- `mutantes.py` e `mutantes.txt`;
- `decomposicao.py` e `decomposicao.txt` (rodada 52: o protocolo de hoje, o `token` da rodada 51 e o de agora), com
  `moe_sim-rodada51.py`, a cópia do simulador da rodada 51 (`286e06c713556cc1`);
- o histórico: `resultado-rodada50.txt` e `diagnostico-rodada50.txt` (a primeira execução) e `resultado-rodada51.txt`.

**Protocolo `token` no simulador** (`tools/moe_sim.py`, só `lru_cpu` e `lfu_cpu`, sem busca):
- a tabela de residência é publicada no começo do token, antes do `g` da primeira camada, e uma cópia que termina no meio
  do token só vale no seguinte;
- a vítima de uma camada `l'` maior ou igual à corrente continua na tabela até o fim do token;
- a cópia para o slot dessa vítima só entra na fila do PCIe quando o host vê a GPU passar dos experts de `l'`, isto é,
  no doorbell da camada `l'+1` (depois do `g` dela) ou, para a última camada, no doorbell do fim do token;
- a fila do PCIe segue a ordem em que o host enfileira;
- as admissões de uma camada vêm depois da resposta da CPU dela, como o `Served` da D1 exige (achado 1 do rev-1 na
  rodada 51). Elas entram no fim da CPU da camada, com o backlog daquele instante, e a cópia que começa ali não é cobrada
  como disputa na CPU da própria camada. Até a rodada 51, a admissão saía no instante do roteador e corria junto com a CPU;
- o backlog conta as cópias ainda não terminadas, porque o host consulta o evento;
- com `token_barreira`, o começo do token espera todas as cópias, a parada entra na espera (`barrier_ms` à parte), e o
  backlog conta todas as cópias do token. Esse é o modo `TokenBarrier` do A1, em que acertos e cópias não dependem do
  tempo da cópia.

O tráfego da camada passou a contar as cópias que entraram na fila do PCIe, e não as admitidas, porque uma cópia adiada
é admitida numa camada e só corre depois. Fora do protocolo `token`, as duas contagens coincidem: o teste de valores
fixos de `fddc2cc` continua passando, e o rev-2 conferiu a igualdade exata contra o `HEAD` em 432 combinações (as 6
políticas, os protocolos `runtime` e `m05`, 3 orçamentos, os dois conjuntos de parâmetros e 6 traços).

**Testes novos (a suíte passou de 56 para 71):**
- uma cópia terminada no meio do token só vale no seguinte;
- uma cópia que termina dentro do primeiro `g` não entra na tabela;
- a vítima de camada à frente continua sendo lida no token, e a cópia espera o doorbell da camada seguinte;
- a vítima da última camada sai no doorbell do fim do token;
- o orçamento nunca é excedido e nada fica pendente no fim do token, com e sem barreira;
- a barreira não depende da banda da cópia;
- a parada da barreira entra na espera;
- (rodada 52) a primeira cópia admitida termina `c_copy` depois do fim da CPU da camada no `token`, e `c_copy` depois do
  começo da camada no protocolo de hoje (um teste com os dois casos);
- (rodada 52) o backlog do `token` é o do fim da CPU: quatro cópias que terminam no meio da CPU já não barram as
  admissões da camada.

**Mutantes (`mutantes.txt`).** Cada um dos 8 derruba pelo menos um teste:
- cópia adiada começando na hora;
- acerto pelo residente em vez da tabela;
- `settle` a cada camada;
- backlog da barreira pelo tempo;
- tabela publicada depois do `g`;
- parada da barreira fora da espera;
- adiada liberada no fim da própria camada;
- (rodada 52) admissão no começo da camada, antes da CPU.
O teste da barreira usa o `lru_cpu` com um traço sem conjunto quente: no `lfu_cpu`, quase todo descarte é recusa pela
frequência, e o backlog não chegava a encher.

**Resultado (`resultado.txt`, simulador `802ad2061bb15f21`, critério `9a78e4353fd788db`): o portão passa, e a D1 fica como
desenho do A1.** O protocolo `token` espera **menos** que o de hoje em todas as células:

| | 10% | 25% | 50% |
|---|---|---|---|
| 1B | −0,81% | −1,21% | −3,73% |
| 3B | −0,70% | −0,87% | −2,31% |

- **O modo com barreira**, com a parada contada, fica de +0,12% a +0,62% acima do `token` no tempo total por token.
- **O histórico.**
  - Rodada 51 (`resultado-rodada51.txt`, admissão no instante do roteador): −0,81 / −1,02 / −2,09% e
    −0,67 / −0,70 / −1,39%.
  - Rodada 50, com a tabela publicada depois do `g`, a adiada liberada no fim da própria camada e a parada da barreira
    fora da espera: −0,83 / −1,35 / −3,57% e −0,70 / −0,96 / −2,13%.
  - As correções vieram dos achados dos revisores. O sinal e o portão não mudaram em nenhuma delas. O rev-2 também rodou
    uma variante FIFO do desenho da rodada 50, e o portão passava nela.
- **De onde vem a mudança da rodada 52 (`decomposicao.txt`).** As cópias por token não mudam (1B a 50%: 1,64 nas duas). Cai
  o número de experts calculados na CPU em camada com tráfego, porque a cópia admitida depois da CPU não corre mais junto
  com ela: 4,87 → 4,00 por token no 1B a 50% e 6,67 → 5,76 no 3B a 50%; a 10%, 7,23 → 7,15 e 10,77 → 10,40. Antes, a
  cópia corria com a CPU da camada que a admitiu, que sempre tem ao menos a falta que gerou a admissão. Agora ela (0,22 ms
  no 1B, 0,33 ms no 3B) corre no `g` da camada seguinte, quando a CPU espera a GPU, e na CPU dessa camada, que pode não
  ter expert nenhum.
- **A previsão acertou o portão e errou o tamanho:**
  - a faixa prevista era de −1% a +1% a 10% e 25% e de −2% a +1% a 50%; saíram fora dela −1,21% no 1B a 25%, −3,73% no 1B
    a 50% e −2,31% no 3B a 50%;
  - as cópias em voo na virada ficaram em 5,2–12,8%, abaixo ou na borda de baixo dos 10–25% previstos;
  - 53–55% das cópias foram adiadas, perto da metade prevista.
- **A causa, conferida (`diagnostico.txt`, igual ao da rodada 50, porque só usa o protocolo de hoje).** No protocolo de
  hoje, metade das vítimas é de camadas à frente, e uma parte delas é pedida de novo no mesmo token e readmitida na hora:
  0,39 / 0,41 / 0,21 cópia por token no 1B e 0,51 / 0,43 / 0,20 no 3B, sempre igual ao número de vítimas pedidas de novo.
  No protocolo `token`, a vítima segue como acerto até o fim do token, e esse vaivém some. A queda de cópias por token
  bate com ele (1B a 10%: 1,53 → 1,17, −0,36 contra 0,39 de vaivém), e com menos cópias há menos camadas com tráfego.
- **O vaivém também diz algo sobre o `lfu_cpu` de hoje:** a frequência de um expert de uma camada à frente ainda não
  recebeu o uso deste token quando a vítima é escolhida. É um defeito pequeno da política, que o protocolo `token`
  contorna de graça.
- **O teto da D2** (as cópias em voo na virada, cada uma a um expert na CPU) fica entre 0,07% e 1,06% da espera do
  `token`.

**Limites:**
- Os mesmos da A0a: o modelo de custo do tráfego foi calibrado com o protocolo de hoje, e o simulador é determinístico.
- O protocolo `token` do simulador não cobra o doorbell nem a linha de serviço do host, que o micro-teste do A1 mede.
- Usa o mesmo `g` para as duas pontas, e o `g` é um bloco só (atenção, roteador e o resto), então "depois do `g` da
  camada seguinte" aproxima o instante do doorbell.
- A parte do ganho que veio na rodada 52 depende da granularidade do modelo de custo. Ele cobra a disputa por camada
  (todos os experts na CPU de uma camada com cópia em voo pagam `c_cpu`), e não pelo tempo de sobreposição de fato. Também
  supõe que a cópia no `g` da camada seguinte não atrasa a GPU. O `o_layer` não tem causa isolada (A0a): se ele vier do
  trabalho do host para enfileirar a cópia, e não da sobreposição, parte desse ganho some. A medida de tempo do A1 confere,
  contra a previsão gravada antes.

### A1 — critério de conclusão (proposto; revisto na rodada 51)

- **Micro-teste do doorbell antes de qualquer outra peça:** o atributo do device, a latência da volta e a GPU lendo o `y`
  novo depois da flag, com um padrão diferente a cada rodada. Feito no A1.0. O mesmo conferidor volta duas vezes (rev-2, rodada 54):
  - no A1.2, com o kernel cutile que lê o `y` como quarto leitor;
  - no A1.3, com o `y` escrito pelo `Pool` da CPU, e a condição "sem store não temporal no `y`, ou `sfence` antes da
    flag".
  Zero erros nos dois, com o N1 acusado.
- **`trybuild`:** os casos falham pelo motivo esperado, e os gêmeos compilam.
- **Controles negativos:**
  - os dois acusados em 20 de 20, e a versão certa bit a bit em 20 de 20;
  - o teste de estado da publicação recusa slot `Loading`.
- **Erro e cancelamento:**
  - a linha de serviço em `panic` no meio do token faz a chamada da principal voltar com erro em menos de 1 s;
  - o `Drop` com a linha de serviço entre a flag e a enfileiração de uma cópia não libera memória com operação em fila (no
    backend falso e no `loom`), e o mutante com a ordem antiga é acusado;
  - o valor de soltura funciona com `t` perto de `2^31`.
- **`loom`:**
  - todas as intercalações de um token curto passam nas invariantes das linhas do host, inclusive as quatro saídas de
    erro;
  - três mutantes são acusados: a flag escrita antes da saída da CPU, a sequência lida com `Relaxed` (aparece como dado
    velho) e a cópia de vítima enfileirada antes do `seq` da camada seguinte à dela.
- **Backend falso com `proptest`:** as duas invariantes passam, e os dois mutantes dos controles negativos caem nele.
- **Tipos por espaço de memória:** casos de `trybuild` (copiar de memória pageable; passar `Device<T>` onde se pede
  `Mapped<T>`), cada um com o gêmeo que compila.
- **Logits e roteamento:**
  - logits dentro de 1e-4 contra a passada residente, nos dois modelos e nos três orçamentos;
  - roteamento igual ao do host: os ids da GPU contra `Routing::from_logits` nos mesmos logits, com desempate por índice.
- **O modo antigo:** `ExpertCache` com `AdmitMode::Barrier` bit a bit igual antes e depois.
- **`TokenBarrier`:** acertos, cópias e experts na CPU iguais aos do simulador no protocolo `token` com barreira.
- **Tempo:** o motor novo contra o `lfu_cpu` de hoje, mediana de 5 intercaladas, com a previsão gravada antes pelo
  simulador. O A2 saiu com a A0a, então não há linha de busca.
  - **O llama.cpp como base** (decisão do Breno de 31/10, 12:15): a comparação de 31/10 se refaz com o motor novo, no
    mesmo harness (`tools/llama-forcado`), nas mesmas 6 células, com as mesmas regras e as linhas do llama.cpp medidas de
    novo na mesma sessão. O plano fecha com esse número.
  - A previsão do `token` é uma faixa, e não um ponto: vai da cobrança da rodada 51 (`moe_sim-rodada51.py`, a cópia
    correndo com a CPU da camada) à de agora, por exemplo 1,001–0,984 ms de espera por token no 1B e 1,955–1,937 ms no 3B
    a 50%. A faixa fica como previsão gravada antes (sugestão do rev-2 na rodada 52).
  - A medida só aponta uma das duas cobranças se o intervalo das 5 intercaladas da diferença medida (motor novo menos o
    `lfu_cpu` de hoje) excluir a outra diferença prevista: −0,021 contra −0,038 ms no 1B e −0,027 contra −0,045 ms no 3B,
    a 50%. Se cobrir as duas, o resultado é inconclusivo, e vale o limite sobre a granularidade escrito na A0c. Com o
    ruído registrado (até 2,7% entre execuções do mesmo binário, de 6 a 20 vezes a distância entre as pontas), o mais
    provável é inconclusivo, coerente com a regra do empate dentro do ruído (achado do rev-2 na rodada 53).
- **Condição de reabertura da busca antecipada** (achados do rev-2 nas rodadas 50 e 51). A medida de tempo do A1 refaz os
  três parâmetros com execuções que o motor novo consegue rodar (ele só tem a admissão do `lfu_cpu`):
  - `c_cpu` e `c_cpu_quiet`: mínimos quadrados do ms de CPU por token contra os experts na CPU em camadas com e sem
    tráfego, nas execuções `lfu_cpu` com backlog 4 e com backlog 0 (sem cópia), iterado até o ponto fixo, como no passo 3;
  - `o_layer`: a sobra do tempo medido além do simulado contra as camadas com cópia, nas execuções `lfu_cpu` com backlog 4;
  - o critério de aceitação de cada ajuste (resíduo máximo e validação contra o tempo medido, ±10% como no passo 3) é
    gravado antes da medida, com a incerteza do ajuste (o intervalo entre as 5 intercaladas);
  - o `o_layer` mal se distingue do ruído: com 1,5–3,9 camadas com cópia por token, ele vale de 0,4% a 3,6% do tempo por
    token, da ordem da variação entre execuções (`tools/moe_sim.py:86–89` já registra que o ±10% distingue pouco esse
    termo). Se o intervalo cobrir tanto 0 quanto 0,0367 ms, o ajuste conta como inconclusivo e não reabre nada sozinho
    (achado do rev-2 na rodada 52).
  Com esses parâmetros, roda de novo o `prefetch_sim.py` da A0a (cerca de 15 minutos de CPU). Se o portão da A0a passar,
  o A2 volta. Pela sensibilidade da A0a, nas variantes amostradas isso só acontece com os dois termos perto de zero.

**Perguntas para a revisão (rodada 52):**
- A drenagem da stream de cópia pela própria linha de serviço, antes de ela terminar, e a ordem do `Drop` (cancelamento,
  flags, `join`, sincronização da stream de cálculo, liberação) garantem que nada fica em fila quando a memória cai,
  inclusive com a linha de serviço parada entre a consulta do cancelamento e a enfileiração?
- O argumento da espera do `ready` (gravado antes do `Published`; só regravado depois do `seq[0]` do token seguinte, que
  vem depois da espera) está certo, e o mutante do backend falso pega a ordem trocada?
- O simulador agora admite depois da CPU da camada, como o `Served` exige. A explicação do ganho maior a 25% e 50%
  (`decomposicao.txt`: a cópia sai da CPU da camada que a admitiu) e o limite escrito sobre a granularidade do modelo de
  custo são suficientes, ou a medida do A1 precisa de um controle próprio para essa parte?
- A reabertura da busca antecipada com os ajustes que o motor novo consegue rodar (`lfu_cpu` com backlog 4 e 0, critério
  de aceitação gravado antes) fecha o achado 3 do rev-2?

## A1 — implementação (a partir de 31/10)

Ordem aprovada pelo Breno (31/10, 12:15), cada passo com revisão de nível 2:
- A1.0, micro-teste do doorbell;
- A1.1, o protocolo sem GPU (tipos por espaço de memória, `trait Backend`, backend falso com `proptest`, `loom`,
  `trybuild` e mutantes);
- A1.2, as peças na GPU (roteador e top-k, um lançamento que lê a tabela do token);
- A1.3, o motor D1 (linha de serviço, cópias guardadas até o doorbell, encerramento, controles negativos na GPU, modo
  com barreira igual ao simulador, modo de hoje bit a bit);
- A1.4, a medida final contra o `lfu_cpu` de hoje e contra o llama.cpp.

### A1.0 — micro-teste do doorbell (31/10; corrigido na rodada 54): o portão passa

Material em `qual-runs/A1-DOORBELL-20261031/`:
- `criterio.txt`, gravado antes de rodar, com a previsão;
- `resultado.txt`, a execução que vale (o topo imprime o sha256 do binário e do fonte);
- o histórico: `resultado-execucao1.txt` e `resultado-execucao2.txt`.

Binário `rust/moe-rt/src/bin/a1-doorbell.rs`, API do driver direto, sem cutile. Tem 4 testes de unidade do conferidor
(`cargo test --release --bin a1-doorbell`): a conclusão da soltura antes do prazo conta, vista depois do prazo ou nunca
reprova e termina, e o GEQ cíclico.

**Montagem.** A linha principal enfileira na stream; a de serviço só toca memória host mapeada, como na D1. A cada volta
`i`, a GPU escreve `seq = i`, espera `flag >= i` (GEQ cíclico), três leitores copiam o `y` (1.536 palavras, a largura do
3B) e a GPU escreve `done = i`:
- (a) um kernel PTX com `ld.global`;
- (b) um kernel PTX com `ld.global.nc`;
- (c) uma cópia pelo motor de cópia.
O host espera o `seq`, escreve um padrão novo no `y`, grava a flag com `Release`, espera o `done` e confere as três
saídas. Em uma volta a cada quatro, o host para 20 µs antes de escrever, e a GPU fica parada na espera. Essas voltas são
um controle positivo embutido: uma espera que não segurasse daria erro nelas. As voltas que cruzam 2^31 e o zero também
têm a parada, então um GEQ sem sinal ou sem ciclo daria erro garantido (rev-2, rodada 54).

**Resultado (`resultado.txt`, terceira execução):**
- **Atributos:** `CAN_MAP_HOST_MEMORY` 1 e `UNIFIED_ADDRESSING` 1. O atributo antigo `CAN_USE_STREAM_MEM_OPS_V1` dá 0,
  mas ele é da API v1, que está obsoleta; as chamadas `_v2` funcionaram em todas as voltas.
- **Correção:** zero saídas erradas nos três leitores em 31.000 voltas: as 1.000 de aquecimento, as 20.000 a partir de
  1, e 5.000 atravessando 2^31 e 5.000 atravessando o zero. O leitor (a) também ficou sem erro nas 21.000 voltas da
  latência. O `ld.global.nc` lê o dado novo, como previsto.
- **Soltura:** com a GPU esperando `t + 1`, o `done` estava em `t − 1` antes da flag (a espera segurou), e a gravação de
  `t + 2^30` a soltou em 6,6 µs, dentro do prazo de 1 s. Se o prazo vencer, o binário sai com falha, sem o
  `cuStreamSynchronize`, que esperaria para sempre.
- **Controles negativos:** o conferidor acusou dado velho em 100% das 2.000 voltas sem a espera da flag (N1) e em 99,6%
  com a flag gravada antes do `y` (N2).
- **Latência** (20.000 voltas, só o leitor (a)):

  | | mediana | p90 | p99 | máximo |
  |---|---|---|---|---|
  | doorbell (flag gravada → `done` visto) | 5,41 µs | 5,70 µs | 6,09 µs | 521 µs |
  | evento (lançar → `cuEventSynchronize`) | 6,85 µs | 7,15 µs | 7,43 µs | 423 µs |

  A razão das medianas é 0,79, dentro da régua (até 20 µs e até 1,5× o evento). O doorbell ficou dentro da previsão (3 a
  8 µs); o evento, não (previsto de 8 a 15 µs, saiu 6,85 µs), o que não muda o portão, cuja régua é relativa. Os máximos
  são voltas isoladas, provavelmente preempção do host; a cauda entra na medida de tempo do A1.4.

**Histórico das execuções:**
- A primeira (`resultado-execucao1.txt`) deu "não passa" por um erro do teste: a conferência da soltura comparava o
  `done` com `t`, mas o `reset(t)` o deixa em `t − 1`, e "segurou" dava falso sempre. Os outros itens do portão passavam
  (zero erros, controles em 100%, mediana de 5,49 µs).
- A segunda (`resultado-execucao2.txt`), com a conferência corrigida, passou. A rodada 54 achou duas falhas no
  conferidor, corrigidas para a terceira:
  - o prazo de 1 s da soltura não entrava no portão, e uma soltura que nunca viesse travaria no `cuStreamSynchronize`
    (achado do rev-1);
  - o aquecimento e as voltas da latência não eram conferidos, e o critério pede 31.000 voltas (os dois revisores).
- Entre as execuções, os números fora do portão variaram: o N1 (c) foi 1.921, 1.928 e 1.938 de 2.000; o N2, 2.000,
  2.000 e 1.992 (os dois controles dependem do tempo, e em 8 voltas a GPU não leu dentro da janela de 20 µs); o p99 do
  doorbell, 6,09, 8,40 e 6,09 µs; o máximo, 38,5, 1.280 e 521 µs. Os dois revisores aprovaram na rodada 55.

**Limites, e onde cada um volta a ser conferido:**
- **O cutile lendo memória mapeada.** O teste de logits do A1.2 pega um `y` sempre velho, mas não uma leitura velha que
  só aparece numa janela de tempo. Por isso o A1.2 repete este conferidor com um quarto leitor, o kernel cutile que vai
  ler o `y` (padrão novo por volta, parada em uma de cada quatro, N1 acusado) (rev-2, rodada 54).
- **O `y` escrito por outras linhas.** Aqui uma linha só escreve o `y` e grava a flag. No A1.3, o `y` vem do `Pool` da
  CPU (`rust/moe-rt/src/cpu.rs:96`), escrito por outras linhas, e a de serviço grava a flag depois. No x86, a garantia
  passa adiante pela sincronização do `pool.run` mais o `Release`, desde que nenhuma escrita do `y` use store não
  temporal, que o `Release` não ordena. Hoje não há nenhum em `rust/moe-rt/src`. A condição fica escrita: sem store não
  temporal no `y`, ou um `sfence` antes da flag. O A1.3 repete este conferidor com o `y` escrito pelo `Pool` (rev-2,
  rodada 54).
- O lado GPU → host só é conferido pelas saídas lidas depois do `done`, sem controle negativo próprio (a v2 não aceita
  `NO_MEMORY_BARRIER`).
- Uma GPU, sem disputa de CPU.

### A1.1 — o protocolo sem GPU (31/10): aprovado

O protocolo D1 do A0b, escrito em Rust contra um backend falso, sem GPU. Código em `rust/moe-rt/src/motor/`; o estado
de trabalho está em `rust/moe-rt/src/motor/ESTADO-A1.1.md`. As rodadas: `lastro-rust-1` parcial (só o rev-2) e `lastro-rust-2` com
mudanças pedidas; na `lastro-rust-3` o rev-2 aprovou e o rev-1 pediu mudanças; na `lastro-rust-4` o rev-2 aprovou de novo
e o rev-1 pediu o `loom` da falha do serviço sem ajuda da principal. Na `lastro-rust-5` o rev-1 achou a ordem do estado de falha, corrigida.
Na `lastro-rust-6` os dois revisores aprovaram, sem achados. A unidade está **aprovada** dentro dos limites declarados abaixo.

**O que existe**
- `Engine` (linha principal) e `Service` (linha de serviço), `Backend` como trait, tipos distintos para memória pinned,
  mapeada e do device, e a tabela do token que só aceita slot residente.
- Backend falso (`fake.rs`) com filas FIFO, eventos com a regra da última gravação antes da chamada, um conferidor de
  violações e um escalonador por semente e pesos por ator. Com ele rodam o `proptest` (invariantes nos dois modos de
  admissão, saídas de erro, `panic` da linha de serviço) e a paridade com o simulador: 64 fixtures de
  `tools/a1_fixtures_sim.py` contra os contadores do motor, em 8 intercalações cada.
- Modelos do `loom` (`loom_tests.rs`) para as ordens de memória e para o encerramento, e 16 casos de `trybuild`
  (`tests/ui/motor_*.rs`, cada um com o gêmeo em `tests/ui-pass/`).

**Resultados (31/10, depois da rodada 2)**
- Suíte normal: 20 testes, cerca de 3 s. Suíte do `loom`: 18 testes, cerca de 8 s (os de encerramento levam até 4 s). `trybuild`: os 16 casos novos falham
  pelo motivo esperado e os gêmeos compilam (cerca de 1 min).
- Paridade com o simulador: sem diferença de bit nos empates entre `exp2` do Rust e `2.0 ** x` do Python.
- Bug achado e corrigido: se a linha de serviço dava `panic`, a flag `done` nunca subia e o `join` do motor girava para
  sempre. Agora uma guarda a sobe também no unwind. A regressão é `panic_da_linha_de_servico_solta_a_principal`, que roda
  o motor numa thread com teto de 5 s e fica vermelha, em vez de travar, sem a guarda.
- O escalonador agora é reproduzível: a linha de serviço entra no sorteio desde o `spawn` (antes, só quando a thread
  começava, e a ordem dependia do tempo real). Com a mesma semente e os mesmos pesos, 10 execuções dão o mesmo resultado
  (`mesma_semente_mesma_execucao`). Os números abaixo se repetiram em três execuções seguidas.

**Mutantes, e quem acusa cada um** (os testes exigem acusação maior que zero)

| Mutante | Acusado por |
|---|---|
| `CopyAtVictim` (cópia na escolha da vítima) | lançamento, 35 de 200 |
| `NoReadyProtection` (sem as duas proteções do `ready`) | lançamento 6 e tabela 39, de 200 |
| `PublishedBeforeRecord` (`Published` antes de gravar o `ready`) | lançamento 20 e tabela 66, de 400, com pesos que favorecem a principal |
| `QueryBeforeRecord` (consulta antes da gravação) | tabela, 18 de 200; o lançamento fica mascarado (0) pela espera da principal |
| `OldDropOrder` (ordem antiga do `Drop`) | memória liberada com fila, 10 de 200. No `loom` ele é acusado pela falta de drenagem que vem junto (`drain: !OldDropOrder`); a ordem antiga sozinha, com drenagem, sai limpa no modelo |
| `HeldOneLayerEarly` (cópia guardada solta uma camada cedo) | lançamento, 27 de 200 (em 171 o caso termina em `Err(Config)`, cópia guardada que não saiu; em 2 não há nem acusação nem erro); no `loom`, acesso concorrente ao slot |
| `SeqRelaxed` (`seq` lida com `Relaxed`) | `loom`: o host lê ids e `x` velhos |
| `FlagBeforeY` (flag antes do `y`) | `loom`: a GPU lê `y` velho |
| Linha de serviço sem drenagem (`drain` desligado) | `loom`: memória liberada com cópia pendente |
| Fechamento sem soltar as flags | `loom`: a GPU fica presa na flag |
| Falha do serviço sem soltar as flags (`FalhaSemSoltura`) | `loom`: a GPU fica presa na flag |
| Falha do serviço sem envenenar (`FalhaSemPoison`) | `loom`: a GPU termina o token sem a falha publicada |
| Falha do serviço sem cancelar (`FalhaSemCancel`) | `loom`: a GPU termina o token sem o cancelamento publicado |
| Falha do serviço publicando o estado depois da soltura (`FalhaEstadoDepois`) | `loom`: a GPU termina o token com `poison` ainda falso |
| Linha de serviço sem o sinal de término | `loom`: o `join` não termina |

O mascaramento só vale para o `QueryBeforeRecord`: a gravação vem antes do `Published`, então toda espera da principal
a captura. No `PublishedBeforeRecord` a gravação vem depois, a espera pode não segurar nada, e o lançamento lê o slot em
cópia. Por isso gravar o `ready` antes do `Published` é necessário para a segurança (rodada 1, rev-2).

**O `loom`:** o modelo com o `Engine` inteiro e o backend falso (três threads com espera ativa) estoura o limite de
ramos mesmo com `preemption_bound` 1, porque cada volta de espera ativa é um ramo e ceder a vez não conta como
preempção. Ficou com modelos pequenos, de duas ou três threads, que chamam a composição de produção: `read_layer` (espera
`seq` e só então lê ids e `x`), `write_answer`, `release_layer` e `held_due` (a cópia guardada) e `close_order` (a lista
de passos do fechamento que `Engine::shutdown` percorre). O de encerramento cobre a principal fechando enquanto a linha
de serviço enfileira uma cópia, sai cedo ou falha, contra uma GPU que espera a flag. Os passos de saída são os de produção (`exit_service`, `release_flags`, `signal_done` e `close_order`), e o modelo exige GPU solta, serviço terminado e memória liberada sem cópia pendente. Esperas do modelo são limitadas: falhar a espera é violação, não estouro de ramos. Isso só vale com uma thread esperando em giro de cada vez (com duas, uma cede à outra até o limite e acusa falsamente); por isso o cenário da falha do serviço sem ajuda da principal deixa a principal só juntando as threads. Nele quem solta a GPU, envenena e cancela é `exit_service`, e o modelo omite cada uma dessas ações para mostrar que pesa (`FalhaSemSoltura`, `FalhaSemPoison`, `FalhaSemCancel`). A principal observa a GPU terminada e confere `poison` e `cancel` antes de juntar o serviço, como `finish_token` faz. A ordem das filas
e as saídas de erro (`CpuErr`, `CpuPanic`, `MainErr`, queda no meio do token) ficam com o `proptest`. Um mutante só conta
como acusado se o panic tiver a marca `MOTOR_VIOLACAO` ou for a violação de causalidade do próprio `loom`; estouro de
ramos reprova o teste. Fora do `loom`: a corrida entre o valor de soltura do `shutdown` e o `answer`.

**Achados da rodada 1 (rev-2) e estado**
- P1-1, `acusado()` do `loom` aceitava qualquer panic: corrigido pelo filtro de payload acima.
- P1-2, `vitima_a_frente` não lia o slot da vítima: o teste novo do escalonador usa o roteamento que lê (chave 2 na
  camada 1) e o `assert!` estrutural virou `Err(Config)`.
- P1-3, o mascaramento do `PublishedBeforeRecord` não era estrutural: o teste agora exige lançamento acusado.
- P2-1, travamento depois de `WrongGeneration`: o motor cai e as chamadas seguintes voltam com `Cancelled`; `finish_token`
  sem token enfileirado é recusado.
- P3: o `prefill` é validado em `Engine::new`; a regressão do `panic` ganhou teto e a estratégia ganhou `Fault::CpuPanic`.

**Achados da rodada 2 e estado** (rev-1 e rev-2)
- P1 (rev-1), `Published` de outro motor passava quando a geração coincidia: ele agora leva a identidade do motor de
  origem e `enqueue_token` o recusa (`Config`, motor derrubado) antes de qualquer operação de backend. Regressão:
  `published_de_outro_motor_recusado`, vermelha sem a checagem.
- P2 (rev-2), escalonador não reproduzível: corrigido como descrito acima.
- P2 (rev-1), `loom` sem o encerramento: modelo novo e dois mutantes, na seção do `loom`.
- P2 e P3 (os dois), faltavam casos de `trybuild`: entraram sete (virada com `Served` vivo, `answer` duas vezes,
  `Published` duas vezes, `lookup` depois da virada, `Device` sem acesso do host, doorbell em `Device`, `Pinned` forjado).
- P3-2 (rev-2), `#[must_use]` não atravessa `Option`: o descarte de `Option<Victim>` (`victim()?;`) continua sem aviso, e o
  atributo no método era redundante (saiu). A proteção real é o `run` interno; o estado `Evicted` no lugar de `Free`
  continua não feito. O candidato fora das faltas, ou que já tem slot, volta com `Err(Config)` em `victim` e em
  `copy_after_reader` (regressão em `line.rs`).
- P3-3 (rev-2), composição reescrita no modelo: `read_layer`, `held_due` e `close_order` foram extraídos e são usados
  pela produção e pelo `loom`.
- P3-4 (rev-2): o comentário de teste voltou ao lugar e a linha do `HeldOneLayerEarly` diz o que acontece nos outros casos.

**Achados da rodada 3 e estado** (rev-1 REQUEST CHANGES, rev-2 APPROVE)
- P2 (rev-1), `loom` do encerramento reescrevia a saída do serviço: `release_flags`, `exit_service` e `signal_done` foram
  extraídos e o modelo os chama, com uma GPU esperando a flag; quatro mutantes (ordem antiga, sem drenagem, sem soltura
  das flags, sem sinal de término).
- P2 (rev-1), faltavam dois pares de `trybuild` (`upload_de_memoria_pageable`, `enfileira_sem_published`).
- P3 (rev-2): a chave que já tem slot é recusada em `victim` e `copy_after_reader`; o `#[must_use]` redundante saiu; o
  texto e o comentário de `close_order` dizem que a ordem antiga só é acusada com a falta de drenagem.

**Achados da rodada 4 e estado** (rev-1 REQUEST CHANGES, rev-2 APPROVE)
- P2 (rev-1), a soltura da principal mascarava o ramo de falha de `exit_service` no `loom`: cenário novo sem ajuda da
  principal e três mutantes (acima).
- P3 (rev-2): a espera limitada só vale com uma thread em giro (comentário e texto acima); a regressão da chave com slot
  agora também cobre `copy_after_reader` com uma vítima obtida antes da primeira cópia; entrou o par
  `motor_published_forjado` (campos privados).

**Achados da rodada 5 e estado** (rev-1 REQUEST CHANGES, rev-2 APPROVE)
- P1 (rev-1), `exit_service` soltava as flags antes de publicar `poison` e `cancel`: a GPU podia terminar o token e a
  principal devolver `Ok` antes de enxergar a falha. A ordem agora é estado de falha primeiro, soltura depois. O cenário
  sem ajuda da principal confere `poison` e `cancel` depois de juntar a GPU e antes de juntar o serviço; a ordem inversa
  é o mutante `FalhaEstadoDepois`. Não foi feito o teste dirigido no backend falso com um portão entre a soltura e o
  estado (a janela só existia entre duas escritas da guarda; o `loom` a cobre).
- P3 (rev-2), dois `Victim` do mesmo slot livre passavam: `copy_after_reader` recusa um slot que não está `Free`
  (regressão em `line.rs`). Os itens soltos da rodada 2 voltaram para a lista dela e a tabela de mutantes ganhou as
  linhas da falha do serviço.

**Limites que ficam**
- O motor roda só contra o backend falso; o backend de CUDA está só na interface (`backend.rs`). Fica para o A1.2.
- Os tipos de memória guardam `Arc<T>` no A1.1, para o backend falso poder ler uma região liberada cedo sem
  comportamento indefinido e anotar a violação.
- No `loom` a cópia roda na hora em que é enfileirada; a ordem das filas é coberta só pelo `proptest`.
- Dois tipos chamados `Victim` e `Pins` agora existem no crate (`cache::core` e `motor`), então dois `.stderr` antigos do
  `trybuild` mudaram só no nome qualificado do tipo, sem mudar o erro nem a linha.

### A1.2 — as peças na GPU (1/11): aprovado (rodada 12)

Quatro sub-passos (estado em `rust/moe-rt/src/motor/ESTADO-A1.2.md`): a, top-k na GPU; b, destino mapeado e quarto leitor
do `y`; c, um lançamento para os acertos; d, backend de CUDA. Checkpoint de revisão depois do b (rodadas 7 e 8, os dois aprovaram),
revisão final depois do d.

**A1.2a e A1.2b (checkpoint)**
- `rust/moe-rt/src/route.rs`: o kernel cutile `route_topk` (um bloco) tira os `K` maiores logits (empate: o menor índice, como
  `Routing::from_logits`), calcula o softmax sobre eles e escreve ids, pesos e o `x` da camada, por `DevicePointer`, em
  memória host pinned e mapeada. A sentinela do padding e da remoção é menos infinito. Teste `topk_na_gpu_bate_com_o_do_host`:
  oito tipos de logits (distintos, com empates, negativos, espalhados em dezenas, todos iguais a -1e30 e a -2e30, dois
  máximos, empate no corte de `K`) em E = 32 e E = 40 (EP = 64), 50 casos por tipo e por E, ids iguais, pesos a 1e-6 e `x`
  bit a bit. Pré-condições escritas no módulo: logits finitos, `K <= E <= EP`, `-0` e `+0` empatam e NaN não é coberto.
- Controles do `route` (cada um verificado vermelho): com a sentinela antiga de -1e30 os casos todos iguais a -1e30 e -2e30
  falham (200 de 800); o desempate pelo maior índice (parâmetro `TS = -1`) é acusado só nos tipos com empate; sem a máscara
  do padding (o kernel recebe `E = EP`) os casos negativos de E = 40 são acusados.
- `LayerIo` ganhou `w` (os pesos do softmax, na ordem dos ids). O backend falso publica pesos iguais; o motor ainda não os
  usa, e passam a valer no A1.3 (a CPU precisa do peso `g` de cada expert).
- Quarto leitor do `y` (`rust/moe-rt/src/bin/a1-cutile-leitor.rs`, critério gravado antes em
  `qual-runs/A1-CUTILE-LEITOR-20261101/criterio.txt`, resultado em `resultado.txt`): o kernel cutile `copy_y` lê o `y` em
  memória mapeada por uma vista sobre o ponteiro, depois de `cuStreamWaitValue32`, com a parada de 20 µs em uma volta a cada
  quatro. Zero saídas erradas em 30.010 voltas (10 de compilação do JIT, de 1, atravessando 2^31 e atravessando o zero); N1 (sem a espera) acusado em 1.993 de 2.000 voltas e N2 (flag
  antes do `y`) em 2.000 de 2.000.
- Limites: o `y` é escrito pela própria linha de serviço (o `y` escrito pelo `Pool` fica para o A1.3); uma GPU, sem disputa de
  CPU; o lado GPU para host só é conferido pelas saídas lidas depois do `done`.

**A1.2c — um lançamento para os acertos**
- `rust/moe-rt/src/hits.rs`: três kernels cutile sobre o banco único de slots (`[S·2I, H]` e `[S·H, I]`, BF16, lidos por vista
  sobre ponteiro): `hit_select` monta, a partir dos ids e pesos do top-k e da linha da camada na tabela do token (slot ou -1), a
  lista compacta de acertos (slot, peso, contagem) na ordem do top-k; `moe_gate_up` calcula o SwiGLU de cada acerto (grade
  `K·I/BM`); `moe_down` soma `peso · (W_down · a)` sobre os acertos, em ordem, com máscara pela contagem.
- Teste `hits::tests::acertos_na_gpu_batem_com_a_cpu_a_1e4` (1.536 de largura, 512 intermediária, 40 experts, 8 residentes, banco
  de 9 slots): 40 casos com 0 a 8 acertos contra o caminho de CPU (`Cpu::experts`); pior erro relativo 3,6e-7, tolerância 1e-4.
  O slot 0 do banco fica com NaN e nunca entra na tabela: as posições além da contagem o leem, e a máscara tem de impedir que
  chegue ao `y`. Verificado vermelho: sem a máscara o teste acusa o NaN; sem a compactação (escrita na posição do top-k) acusa
  os slots fora de ordem; a conferência com um acerto a menos fica fora da tolerância (o teste distingue 1e-4 de um acerto).
  Esses dois mutantes foram aplicados à mão e revertidos, não são testes permanentes.
- A soma dos experts muda de ordem em relação ao caminho de hoje (como o plano previa): os logits passam a valer pela tolerância.

**A1.2d — backend de CUDA**
- `rust/moe-rt/src/cuda_backend.rs`: `CudaBackend` implementa `Backend` sobre o driver e os kernels acima: `copy_expert`
  (`cuMemcpyHtoDAsync` do expert pinned para o slot do banco), eventos (`cuEventRecord`, `cuStreamWaitEvent`, `cuEventQuery`),
  `upload_table`, `route` (GEMV do roteador e `route_topk`), `launch_hits`, `add_y` (soma da GPU e da saída da CPU no estado `h`),
  `write_value` e `wait_value` (`cuStreamWriteValue32_v2`/`cuStreamWaitValue32_v2`, `GEQ` cíclico) e `sync`. Os tipos de
  memória ganharam o interior real: `Pinned` e `Mapped` aceitam conteúdo em memória do driver (`cuMemHostAlloc`) além do `Arc`
  do backend falso, e `LayerIo` agora tem os pesos do softmax e pode ser montada sobre palavras mapeadas (`Words` opaco, `LayerIo::from_raw` interno).
- O trait `HostExperts::compute` recebeu os pesos do softmax das faltas (`weights`), publicados pela GPU; a linha de serviço
  os lê depois do `Acquire` de `seq`, como ids e `x`. As suítes do A1.1 passam sem mudança de comportamento.
- Teste `cuda_backend::tests::pilha_de_blocos_moe_pelo_d1_bate_com_a_cpu_a_1e4`: uma pilha de 3 blocos MoE (16 experts, topk 4,
  largura 512, 14 slots) pelo motor D1 completo (duas linhas do host, doorbell, cópias, tabela por token) contra a mesma pilha
  inteira na CPU, 16 tokens com duas entradas alternadas (o cache enche: 90 acertos da GPU e 90 faltas da CPU nos contadores
  acumulados até o último `Published`). Pior erro relativo 2,2e-7, tolerância 1e-4. Verificado vermelho: sem o `cuStreamWaitValue32` da flag
  o erro é de 19% no token 0; sem a saída da GPU na soma o teste acusa a partir do token 1 (o token 0 não tem acertos).
  Mutantes aplicados à mão e revertidos.
- Correções da rodada 9 (revisão final): (1) `Words` virou tipo opaco (representação privada, `LayerIo::from_raw` é
  `pub(crate) unsafe`), porque a variante pública `Raw` permitia ler memória arbitrária por código seguro; novo par
  `trybuild` (`motor_words_forjado`, falha por construtor privado; gêmeo em `ui-pass` com `LayerIo::new`), 17 pares. (2) Os
  tensores de trabalho ficam donos do estado e os lançadores recebem partições emprestadas (`&mut`, `PartitionMut`), então um
  lançamento que falha derruba só o empréstimo: o cutile libera tensor com `cuMemFreeAsync` em outra stream, e antes isso
  acontecia com os argumentos movidos (rodada 10, rev-1 P1, rev-2 P3-1). Os `?` de `route`/`launch_hits`/`add_y` esperam a
  stream (`fail_k`) antes de a função retornar. Teste `erro_apos_enfileirar_espera_a_stream_antes_de_soltar_os_tensores`: a
  stream fica presa numa espera de valor, `moe_down` falha na compilação (`FAULT_DOWN`) e `launch_hits` não pode voltar até a
  flag; verificado vermelho trocando o `cuStreamSynchronize` por nada. O teste prende a espera; que o dono continue vivo
  vale pelo desenho (sem argumento movido), não por um teste de liberação. (3) O banco tem `slots + 1` slots: o último é um slot de preenchimento zerado, lido pelos
  grupos inativos de `hit_select` (parâmetro `pad`), para que nenhum kernel leia slots administrados pela linha de cópia.
  (4) O banco é zerado e o contexto esperado depois do `memset` (stream nula não ordena as streams do motor). (5) O oráculo do teste integrado recusa NaN e tamanho diferente (`erro_rel` devolve infinito), com teste próprio.
- O `trybuild` do A1.1 passa; mudou só o texto de um `.stderr` (`motor_view_forjada`: o compilador abrevia a lista de campos
  privados porque `LayerView` ganhou o campo `w`).

**Limites que ficam do A1.2**
- Mutante equivalente no D1: contar uma falta como acerto no `hit_select` não muda a saída do teste integrado (o slot -1 lê o zero do
  preenchimento); só o teste do `hits` o pega.
- O modelo do teste é uma pilha de blocos MoE (`h ← h + moe(h)`), sem atenção, embedding ou logits do vocabulário; o "teste de
  logits" do plano com o modelo inteiro passa a ser o do A1.3. Aqui o que se compara é a ativação depois dos blocos.
- Uma GPU, sem disputa de CPU; o `y` é escrito pela linha de serviço (o `y` escrito pelo `Pool` e a condição do store não
  temporal são do A1.3). Nenhuma medida de tempo: o A1.4 mede.
- A seleção dos acertos roda um bloco só e os kernels calculam os `K` slots mesmo com poucos acertos (a bandwidth das posições
  além da contagem é gasta e descartada); otimizar fica para depois da medida.
- O backend roda `route`, `launch_hits` e `add_y` com um mutex no estado de trabalho; só a linha principal os chama.
- Não há teste de portão ou sanitizador para a corrida do slot de preenchimento: a garantia vem do desenho (o slot nunca é
  escrito pela linha de cópia), não de uma medida.
- `-0` e `+0`, NaN e logits não finitos não são cobertos no top-k (pré-condições escritas em `route.rs`).

### A1.3 — o motor D1 completo (1/11): feito e aprovado (checkpoint na rodada 14, final na 17, regressão do `read_logits` na 18)

Quatro sub-passos (estado em `rust/moe-rt/src/motor/ESTADO-A1.3.md`): a, as fixtures do simulador no motor de CUDA; b, o `y`
escrito pelo `Pool`; c, o modelo inteiro no motor; d, controles negativos na GPU e o modo antigo. Checkpoint de revisão depois
do b, revisão final depois do d.

**A1.3a — o motor de CUDA com os contadores do simulador**
- `rust/moe-rt/tests/motor_gpu.rs`: as fixtures de `tests/fixtures/a1_sim_token_barreira.json` (protocolo `token` com barreira,
  `AdmitMode::TokenBarrier`) rodam no `Engine` com o `CudaBackend`, com as duas linhas do host em threads de verdade e a GPU de
  verdade. O roteamento do simulador é forçado por roteadores construídos: a entrada do token `t` é o vetor com 1 na posição
  `t`, e o roteador da camada `l` dá ao expert na posição `p` da lista de `routing[t][l]` o logit `K - p` e zero aos demais.
  Os experts têm pesos zero, então o `h` não muda (conferido a cada token) e só os contadores contam.
- Resultado: as 31 fixtures que cabem batem com o simulador (acertos, cópias, experts na CPU, recusas e cópias guardadas), e
  somam acertos, cópias, CPU e cópias guardadas diferentes de zero. Limite: o top-k do backend exige `K` potência de dois e `E`
  par (`GEMV_BM = 2`), então as fixtures com `K = 3` ou `E` ímpar (33 das 64) não rodam na GPU; elas seguem no backend falso.
- Controle: com o expert deslocado pelo índice do token (um deslocamento igual para todos só renomearia os experts), 26 das 31
  fixtures saem com contadores diferentes. Mostra que o roteamento que chega ao motor é o dos roteadores construídos.

**A1.3b — o `y` escrito pelo `Pool`**
- `Cpu::fill_rows` (`rust/moe-rt/src/cpu.rs`) é o caminho de escrita das saídas da CPU: faixas de 16 linhas repartidas pelas
  threads do `Pool`, stores comuns, retorno só depois de todas as tarefas. `gemv` e a soma final dos experts passaram a usá-lo
  (mesmas faixas, mesma conta; os testes de CPU e de modelo seguem iguais).
- A condição do plano ("sem store não temporal no `y`, ou `sfence` antes da flag") ficou como teste permanente em
  `rust/moe-rt/tests/sem_store_nao_temporal.rs`: uma varredura textual de `rust/moe-rt/src`, sem exceção de arquivo e sem
  distinguir caixa, atrás das famílias `stream`, `movnt`, `maskmov` (`maskmovdqu`, `_mm_maskmove`; o `vmaskmovps` do AVX é um
  store mascarado comum e fica de fora), `movdir` e `directstore`, mais um teste do detector com trechos de código. Verificado
  vermelho com `maskmovdqu` plantado em `src/route.rs` e em `src/motor/cpu.rs`. Cobertura textual, não prova sobre código
  gerado nem sobre dependências. Os dois furos que a rodada 13 achou (família `maskmov` e a exceção de qualquer `cpu.rs` por
  nome) estão fechados.
- Conferidor `rust/moe-rt/src/bin/a1-pool-y.rs`: o kernel cutile lê o `y` mapeado depois da flag, mas quem escreve o `y` são as
  threads do `Pool` (`fill_rows`) e a linha de serviço grava a flag com `Release` depois do retorno. Duas rodadas, cada uma
  com critério gravado antes: `qual-runs/A1-POOL-Y-20261101/` (o N3 deu 201 de 2.000, a uma volta do mínimo, e não isolava o
  término das outras faixas; mantida como histórico) e `qual-runs/A1-POOL-Y-20261101-b/` (a oficial). Na segunda, o N3 segura as
  faixas diferentes da 0 por 20 µs, publica a flag de dentro da faixa 0 e só conta o erro que alcança uma faixa de outra tarefa.
  31.010 voltas conferidas (30.000 medidas, 1.000 de aquecimento e 10 de JIT), zero saídas erradas; N1 acusado em 1.999 de 2.000,
  N2 em 2.000 de 2.000 e N3 em 2.000 de 2.000 (mínimo 50% nos três).
- Limites: no motor de produção quem escreve a área mapeada continua sendo a linha de serviço (`HostExperts::compute` devolve um
  vetor, e `answer` o copia para o `y` mapeado e grava a flag); o conferidor cobre o caso em que as threads do `Pool` escrevem a
  área mapeada, que é o que o plano pede. Uma GPU, sem disputa de CPU além do próprio `Pool`.

**A1.3c — o modelo inteiro no motor**
- `rust/moe-rt/src/cuda_dense.rs` (`Dense`): embedding, atenção (rmsnorm, q/k/v, rope, cache KV, atenção em duas passadas, o),
  residual, rmsnorm do MoE e o `lm_head`, na stream de cálculo, reaproveitando os kernels do M1b. O `Backend` ganhou dois
  ganchos com implementação vazia por padrão (`pre_layer`, antes do roteador da camada, e `post_layers`, antes do último
  `seq`); o backend de CUDA os usa para rodar o bloco denso. Os pesos do softmax valem na CPU (`g`) e o resíduo soma
  `residual_multiplier * g * (yg + yh)`.
- `rust/moe-rt/tests/motor_model.rs`: modelo Granite pequeno, 14 tokens, com 2, 5 e 8 slots nos modos por evento e
  `TokenBarrier`; logits do motor contra a `CpuSession` a 1e-4 (pior erro ~4e-7).

**A1.3d — controles negativos na GPU e o modo antigo**
- Janela da cópia na vítima (`janela_da_copia_na_vitima_...`, em `src/cuda_backend.rs`): a stream de cálculo fica presa na
  camada 1 enquanto as cópias decididas nas camadas 0 e 1 correm. A versão certa sai a ~2e-6 da CPU; o mutante `CopyAtVictim`
  é acusado em 60 de 60 voltas (3 cenas, 20 voltas cada, atrasos de teste de 80 e 400 ms alternados); a versão certa repete os mesmos bits em todas as voltas. Só 3 fixtures do simulador viram cena elegível (o mutante só lê dado errado quando a vítima é um
  slot que uma camada futura ainda lê no mesmo token); o teste exige as 3 cenas e 100% das voltas acusadas, com a saída do mutante diferente, em bits, da execução sem janela; a versão certa com janela é bit a bit igual a essa execução. No controle do `ready` (modo por evento) a execução sem janela muda a divisão GPU/CPU e não serve de referência: ali valem a repetição de bits entre voltas e a tolerância contra a CPU.
- Janela do `ready` (`janela_do_ready_...`): com a stream de cópia presa, a versão certa não lista a chave e bate com a CPU
  (erros 0, 0, 0); o mutante `NoReadyProtection` perde a contribuição do expert (erro 1.0 no token 1). 20 de 20 voltas, a versão certa com os mesmos bits.
- Os kernels precisam estar compilados (JIT) antes de prender uma stream: carregar um módulo é uma sincronização implícita do
  contexto e travaria; os testes rodam a cena uma vez sem prender antes.
- Regressão: `erro_na_copia_dos_logits_nao_estraga_os_tokens_seguintes` (`tests/motor_model.rs`) injeta uma falha na cópia dos logits; antes da correção os tokens seguintes saíam errados (erro `inf`).
- Modo antigo: os testes de `tests/model.rs` (`ExpertCache` com `AdmitMode::Barrier`, contadores e logits) seguem iguais. Motor
  normal (38 na lib), loom (18) e trybuild (17 pares) passam; zero avisos no build normal.
- Limites: na GPU, `K` potência de dois e `E` par; o bloco denso copia entre buffers por dtod (custo a medir no A1.4); nenhuma
  medida de tempo; o controle da janela da cópia tem poucas cenas; os controles não têm sinal explícito de progresso da linha de serviço (decisão do Breno em 01/11: aceitar; a independência da velocidade vem de repetir com atrasos de 80 e 400 ms, e o mutante da cópia é acusado até sem janela), e a comparação é bit a bit entre as voltas da versão certa e por tolerância contra a CPU, não contra uma referência GPU do expert antigo/novo. `Dense` é interno ao crate (`pub(crate)`), porque os lançamentos não sincronizam.

## Mudança para o lastro-rust (31/10)

A pedido do Breno, o runtime saiu do strata-opencode (o projeto que liga o OpenCode ao Strata, onde ele nasceu) para um
repositório e um space próprios: `/home/brenoperucchi/Devs/lastro-rust`, space `lastro-rust` no Herdr.
- O histórico veio junto, filtrado aos caminhos do runtime (`git filter-branch`): os 8 commits de 3eb221f a 3bacc1e no
  strata-opencode viraram 9dda2df a ff639fd aqui, com as mesmas datas e mensagens (mais o primeiro commit do
  strata-opencode, que só traz o começo do `.gitignore`). As árvores de `rust/`, `qual-runs/`, `tools/`, `tests/`, `docs/`
  e o `.gitignore` conferem byte a byte com o 3bacc1e. Os hashes citados neste plano são os do strata-opencode.
- A estrutura de pastas é a mesma, na mesma profundidade: os caminhos relativos deste plano e os do `Cargo.toml` para o
  cuda-rust (`../../../GitHub/cuda-rust`) seguem valendo.
- Os registros das `qual-runs` ficam como foram gravados, com os caminhos absolutos antigos, porque os critérios têm sha
  citados nos resultados. Para executar de novo, troque `/home/brenoperucchi/Devs/strata-opencode` por este repositório.
- As rodadas de revisão e de consulta do runtime citadas acima (revisões `strata-opencode-25` a `55`, consultas `5` a `8`)
  foram copiadas para `.herdr/` deste repositório, fora do git, então os caminhos relativos do plano resolvem aqui; o
  original continua em `/home/brenoperucchi/Devs/strata-opencode/.herdr/`. As novas são `lastro-rust-N`.
- O plano que deu origem ao runtime (o runtime híbrido para o Strata, 24/10) está em
  `docs/historico/PLANO-runtime-hybrid-strata-2026-10-24.md`.
- Ficou no strata-opencode a parte do OpenCode e todo o material da Fase 0A.
