# Plano: instrumento de fase no passo de bloco do prefill (10/11/2026, rascunho para revisão nível 2)

Autorização do Breno (18:01): mudar código só para instrumentar, com plano e critério antes dos dados, revisão nível 2 (até 6 rodadas), janelas na 3080 Ti sob autorização permanente (voxtype parado, critério congelado antes dos dados, limite de tempo, sem repetição automática após SEM PROVA). Este arquivo é só o plano. Nada foi editado, compilado ou rodado para escrevê-lo; tudo abaixo vem de leitura do código em `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src`.

Base empírica (perfil 10/11, 3080 Ti): prefill linear nas posições. 1B: L10 ~9,8 ms/posição, L100 ~2,44 (decode 23,1 e 2,8 ms/passo). 3B: L10 ~20, L100 ~4,1 (decode ~45 e ~4,6). Razão ms/posição ÷ decode: ~0,42 a 0,45 no L10, ~0,87 a 0,90 no L100. Com blocos de 8, um bloco custa ~3,4 a 3,6 passos de decode no L10 e ~7 no L100. O L100 não tem CPU de experts em jogo (100% dos experts na GPU), e mesmo assim o bloco custa ~7 passos: isso já sugere que a GPU não amortiza o lote.

## 1. Mapa do caminho do prefill

Cronômetro atual: `geracao_regra.rs:203` (`prefill_ms` = `le_ids_com_pendente` + `(s.le_logits)()`, de `t_pre` em :198 até :203). O decode é cronometrado por passo em `geracao_regra.rs:209` a :225 (`passo_ms`). A impressão está em `bin/primeira-palavra.rs:920,940,1037`.

Laço de blocos (thread principal, uma por bloco de até 8 ids):
1. `motor/leitura.rs:~82` `le_ids_com_pendente`; `plano_blocos` em :53; laço em :98. Por bloco: `eng.next_published()` (:101), `prep.prepara(ids)` (:111), `eng.enqueue_step(p, m)` (:115), `eng.finish_token()` (:116).
2. `Engine::next_published` (`motor/engine.rs:170`): espera (laço com `relax`) o `Published` que a linha de serviço publicou no fim do bloco anterior. Mede a folga de serviço entre blocos.
3. `LeitorCuda::prepara` (`cuda_backend.rs:650`) chama `begin_step_tokens` (:499) que chama `CudaDense::begin_step` (`cuda_dense.rs:180`). Por posição: embedding no host, `rope_cos_sin` no host, `copy_host_vec_to_device(...).sync_on` de `lens` (:~192), `f32_dev` de cos/sin, e depois em `begin_step_tokens` outro `copy_host_vec_to_device(...).sync_on` do `h` (:~507). Ou seja, vários H2D síncronos por posição, na stream de cálculo, com a GPU ociosa. Não confirmei se `f32_dev` também é síncrono.
4. `Engine::enqueue_step` (`engine.rs:255`) chama `enqueue_inner` (:185). Só enfileira, não espera: `begin_step`, `upload_table` (tabela de slots, H2D), `wait_event` por slot pronto, e para cada camada `pre_layer`, `route`, `write_value(seq[l])`, `launch_hits`, `wait_value(flag[l])`, `add_y`; depois `post_layers` e `write_value(seq[L])`. O tempo de host aqui é o custo de lançamento (JIT/launch do cutile com `generics` em `String`, alocações) e fica sobreposto à GPU e à linha de serviço, salvo se o lançador for mais lento que a GPU.
5. Por camada, por posição `p` em `0..m`, os kernels são lançados por posição, sem lote na GPU:
   * `pre_layer` (`cuda_backend.rs:733`) chama `attn_block_at` (`cuda_dense.rs:206`) e `attn_block` (:~232): rmsnorm, 3 gemv (q,k,v), 2 rope, vários `dtod`, atenção (`attention`, ATTN_NS partições), gemv o, axpby, rmsnorm. Em torno de 15 a 20 lançamentos por posição por camada.
   * `route` (:764): gemv do roteador, `route_topk` (grid 1x1x1), por posição; no fim `cuStreamWriteValue32` com `m`.
   * `launch_hits` (:797): `hit_select`, `moe_gate_up`, `moe_down` por posição (3 kernels por posição por camada). Os pesos dos experts em GPU são relidos da memória por posição (gemv por posição), então o L100 não ganha com M=8 na parte de experts.
   * `add_y` (:850): `add_y_scaled` por posição.
   * `post_layers` (:749): rmsnorm e gemv do `lm_head` só da última posição do bloco. Mas é executado em todos os blocos, não só no último (custo do vocabulário inteiro por bloco; não confirmei o tamanho do vocab no 1B/3B).
6. Linha de serviço (outra thread), `motor/line.rs:708` `run`, por camada: `serve_layer` (:435, espera `seq[l]` com `relax`, monta a união de experts do bloco e as faltas), `cpu.compute_step` (`cuda_backend.rs:942` que chama `Cpu::experts_step`, `cpu.rs:239`, dois despachos do `Pool` sobre a união), `answer`, `pin_layer`, `victim`/`copy_after_reader` (:562, cópias H2D de experts na stream de cópia, ou `Held`), depois a flag de resposta. No fim do bloco `end_token` (:587): espera `seq[L]`, `query` dos eventos de cópia, tabela, `publish`. O prefill de blocos de 8 faz a GPU esperar a CPU em toda camada (round trip por camada por bloco), e a CPU só começa a camada depois de `seq[l]`.
7. `finish_token` (`engine.rs:285`): `cuStreamSynchronize` da stream de cálculo (`cuda_backend.rs:889`). Um por bloco. Depois `read_logits` (:611, outro `cuStreamSynchronize` e D2H dos logits, `cuda_dense.rs:319`), uma vez, no fim do prompt.
8. Já existe a feature de compilação `diag` (`diag.rs`, `Cargo.toml`): relógios da linha de serviço por camada (T_espera, T_servico, T_cpu, T_conv, T_livro) somados por token, sem impressão no caminho quente. Serve de molde, mas é por token (aqui precisamos por bloco e inclui o lado da GPU/lançamento) e mudar de feature muda o binário.

Onde provavelmente está o custo por bloco (sem afirmar causa): na repetição por posição de toda a parte densa e dos experts em GPU (centenas de lançamentos por bloco e leitura repetida de pesos), no round trip GPU↔CPU por camada (espera de `seq[l]`, CPU dos experts da união, flag), no preparo síncrono por posição, e no custo de lançamento do cutile. Nenhuma está medida.

### Hipóteses pré-registradas (H1..H7), previsões sobre o que o instrumento mostraria

Fases de referência (por bloco, em ms, medianas): F0 `next_published`; F1 `prepara`; F2 host de `enqueue_step` (todos os lançamentos); F3 espera da stream em `finish_token` (de F2 acabar até o sync voltar); F4 por camada na linha de serviço: espera de `seq[l]`, `compute_step`, `answer`+livro; F5 `read_logits`. Mais, por camada, o tempo de GPU de cada seção (eventos) em modo opcional.

* H1 (lançamento domina o host, GPU esperando o host): se verdadeira, F2 ≈ prefill por bloco e F3 ≈ 0; a soma de F2 cresce linear em m (lançamentos por posição). No L100 espera-se F2 ≥ 70% do bloco. Falsificada se F3 ≥ 50% do bloco.
* H2 (a GPU é o limite por falta de lote: gemv por posição relê pesos): F3 grande (≥ 50% do bloco no L100) e o tempo de GPU de `launch_hits` + `pre_layer` por bloco ≈ m × o de m=1 (razão 6 a 8 entre bloco de 8 e bloco de 1, medida no último bloco do prompt ou no decode). Falsificada se a razão ficar abaixo de 3.
* H3 (round trip por camada GPU↔CPU no L10): no L10, a soma sobre camadas de "espera de `seq[l]` + serviço" domina o bloco (≥ 50%), e `compute_step` (CPU dos experts) é a maior parcela do serviço. Prevê-se que no L100 (sem faltas) essa parcela caia a ≤ 10% do bloco. Falsificada se `compute_step` for < 20% do bloco no L10.
* H4 (preparo síncrono por posição): F1 por bloco ≥ 5% do bloco no L100 (vários H2D síncronos por posição). Falsificada se F1 < 2% do bloco.
* H5 (cópias de experts, `copy_after_reader`, `Held`, `query`): no L10 o livro (`victim`/`copy`) + `end_token` somam ≥ 10% do bloco; no L100 ≈ 0. Falsificada se < 3% no L10.
* H6 (custo fixo por bloco, não por posição): o bloco final parcial (resto 1 a 7 posições) custa quase o mesmo que um bloco cheio. Previsão ordinal: custo(bloco de 1) / custo(bloco de 8) ≥ 0,5 se H6; ≤ 0,25 se o custo é por posição (H2). Isto separa "fixo por bloco" de "por posição". Só testável nos prompts cujo resto de n % 8 seja 1 a 3 (conferir os três textos: 89 % 8 = 1, 170 % 8 = 2, 364 % 8 = 4).
* H7 (cabeça `lm_head` por bloco): F do `post_layers` por bloco ≥ 5% do bloco no L100 (vocabulário grande × 2 kernels por bloco). Falsificada se < 2%. Requer evento ao redor de `post_layers`.

As previsões são ordinais e de fração do bloco; os limiares foram fixados aqui, antes dos dados, e só valem se a validade (seção 4) ficar OK.

## 2. Desenho mínimo do instrumento

Princípio: cronometrar por fase sem mudar nada do que o motor faz quando desligado.

Ativação: variável de ambiente nova `LASTRO_PERFIL_BLOCO=1` (lida UMA vez, na criação do motor, para um `bool` no `Engine`/backend; nunca no caminho quente). Desligada por padrão. Decisão em aberto para o Breno (pergunta 1): variável de ambiente em tempo de execução (um `if` por ponto, branch previsível, sem relógio) versus feature de compilação (zero código no binário, mas é outro binário e outro sha para a medida). Recomendo a variável de ambiente, porque o par com/sem instrumento sai do mesmo binário, e a medida do custo do instrumento fica honesta.

Fases e pontos (todas com `Instant`, relógio monotônico do host, exceto onde marcado):
* Lado do host (thread principal), por bloco, em `le_ids_com_pendente` (`motor/leitura.rs`): t0 antes de `next_published`; t1 depois (F0); t2 depois de `prepara` (F1); t3 depois de `enqueue_step` (F2, só lançamento); t4 depois de `finish_token` (F3, inclui o `cuStreamSynchronize` que já existe, sem sync novo). Zero sincronização adicional: F0 a F3 usam só as chamadas que já bloqueiam. Gravar em `Vec<RecBloco>` pré-alocado (capacidade dada pelo `plano_blocos`), sem impressão no caminho quente.
* Lado da linha de serviço, por camada, em `motor/line.rs::run`: aproveitar o `diag::now`/`wait_done`/`compute_parts` já existentes como estrutura, mas com acumuladores por bloco (não por token) e ligados por `bool` em vez de feature. Fases: espera de `seq[l]` (de `serve_layer` entrar até ver o seq; folga do host), `compute_step` (CPU dos experts, já separado em conversão e `Cpu::experts_step` por `compute_parts`), `answer`, livro (`pin`, `victim`, `copy_after_reader`), e `end_token` (espera de `seq[L]` + `query` + `publish`). Escritora única (a thread de serviço); o merge com os registros do host só ao fim do pedido, fora do cronômetro de prefill.
* Lado da GPU (opcional, segunda passada, porque exige sincronização/eventos): eventos CUDA (`cuEventRecord`) na stream de cálculo ao redor de `pre_layer`, `route`, `launch_hits`, `add_y` e `post_layers` de CADA camada, lidos com `cuEventElapsedTime` só depois do `cuStreamSynchronize` do `finish_token` (que já existe), portanto sem sync extra. Custo: 2 a 10 `cuEventRecord` por camada (~µs cada, enfileirados, mas inserem pontos na stream; podem impedir fusão/sobreposição de lançamento e tornam a medição "com instrumento" mais lenta). Eventos temporizados exigem `CU_EVENT_DEFAULT` (sem `DISABLE_TIMING`); conferir como `CudaBackend::event()` (`cuda_backend.rs:409`) cria os eventos: NÃO confirmei se são criados com timing desligado, então pode ser preciso um construtor novo.
* Cada bloco grava: índice, m, posição inicial, e as fases acima. Cada pedido grava um resumo (somas e medianas por fase, por m) em arquivo `perfil-bloco.json` ao lado da saída, só se a variável estiver ligada.

Garantia de desligado: sem a variável, nenhum `Instant::now`, nenhum `cuEventRecord`, nenhum `Vec` alocado; só `bool`s falsos testados. Os ids gerados e os logits não podem mudar: o instrumento não toca no estado do modelo, nas posições, nas tabelas nem na ordem das operações da stream (os eventos só entram quando ligado). Prova: os ids da referência 5c (os mesmos 3 textos, 1B e 3B, L10, L100 e LC) têm de sair idênticos com a variável desligada E ligada. Ligada, a ordem dos lançamentos é a mesma (eventos são extras na stream, não alteram dependências), então ids iguais também; se divergir, o instrumento é inválido (SEM PROVA).

Custo do próprio instrumento: três níveis, medidos em par intercalado (sem, com host+serviço, com host+serviço+GPU), mesma célula, mesma ordem aleatorizada fixada antes, N medidas cada (seção 4). Reporta-se o desvio do prefill_ms mediano com e sem; se o desvio passar de 3% (limiar fixado aqui), a decomposição "com instrumento" é descritiva com ressalva e as frações são reportadas relativas ao tempo medido com instrumento, nunca ao sem; se passar de 10%, o nível GPU é SEM PROVA.

## 3. Mudanças de código previstas

Estimativa (linhas aproximadas, sem contar testes):
* `rust/moe-rt/src/perfil_bloco.rs` (novo): tipos `RecBloco`, `RecServico`, `Perfil` (acumuladores, resumo em JSON com `serde_json`, que já é dependência), leitura da variável. ~150 a 200 linhas. Lógica pura (agregar, mediana, resumo) separada do relógio para testar em CPU.
* `motor/leitura.rs::le_ids_com_pendente`: ganha um parâmetro de perfil opcional (ou o `Engine` carrega um `Option<Perfil>`); ~25 linhas. Cuidado: a assinatura é usada em `geracao_regra.rs` e nos testes (`leitura_tests.rs`, `sequencia_tests.rs`); preferir guardar no `Engine` para não mudar a assinatura pública.
* `motor/engine.rs`: campo `perfil: Option<...>`, ligado em `Engine::new` (:101); fases F2 e F3 em `enqueue_step`/`finish_token`; ~30 linhas.
* `motor/line.rs::run` e `Service`: acumuladores por camada e por bloco ligados por `bool`; ~50 linhas.
* `cuda_backend.rs` e `cuda_dense.rs`: eventos CUDA ao redor das seções (nível GPU); `Backend` ganha métodos de marca opcionais com implementação vazia no `FakeGpu` (`motor/fake.rs`) e no trait (`motor/backend.rs`); ~60 a 90 linhas. Mudar o trait `Backend` obriga a atualizar `fake.rs`; para reduzir a superfície, os eventos podem ficar dentro de `CudaBackend` (campo `perfil`) sem tocar no trait. Decisão do exec na implementação, a revisar.
* `geracao_regra.rs` e `bin/primeira-palavra.rs`: coletar o resumo do perfil depois do prefill (fora de `t_pre`..:203) e gravá-lo; ~30 linhas. A saída do resumo vai em arquivo à parte (não altera o registro de geração nem o formato do que o parser dos critérios lê).
* Testes sem GPU (CPU, `FakeGpu` do `motor/fake.rs` e `cargo test` normal, que já roda sem GPU): (a) `perfil_bloco`: agregação, mediana, resumo, ordem das fases, soma das fases ≤ tempo do bloco; (b) com o `FakeGpu`, rodar `le_ids` com perfil ligado e desligado e comparar os eventos/ordem das operações do backend falso (mesmos ids, mesma sequência de chamadas, mesma posição final): é o teste de "ligar não muda resultado"; (c) o resto de n % 8 produz o bloco final de tamanho esperado no registro; (d) variável de ambiente inválida ou ausente = desligado; (e) um `Engine` com perfil desligado não aloca (contador de alocações no registro, ou apenas `is_none()`).
* O que só a GPU testa: validade real dos eventos CUDA (ordem, `cuEventElapsedTime` coerente com o relógio do host), custo real do instrumento (par com/sem), ids iguais à referência 5c nos modelos reais, e as próprias hipóteses.

Compatibilidade com o portão da 3080 Ti e o FIXO: o binário `primeira-palavra` muda de sha (e o de qualquer `lastro-*` que dependa do crate). O `roda-perfil-janela.sh` guarda os shas do FIXO no array `FIXO` (:81), com `--verifica-fixos` (:7). Precisa de emenda: (1) o sha do binário novo no FIXO, e o do binário da janela 17:05 (`65faf6e7...`) mantido como referência do "sem instrumento" (ele é o ponto de comparação para o par); (2) o roteiro novo (o `roda-perfil-janela.sh` serve de molde, mas é arquivo congelado; criar `roda-instrumento-janela.sh` em vez de editar o antigo) com o sha do próprio roteiro, do critério novo, e do JSON/resumo de saída; (3) o critério de ids iguais à 5c com a variável ligada e desligada; (4) o portão de 370 W, a prova de GPU livre (voxtype parado), os timers e `systemctl` do roteiro anterior, sem mudança; (5) o `criterio-perfil-prefill-20261109.md` diz "nenhuma linha de código muda": esta frente é nova, com critério novo, sem reabrir aquele. Não consegui conferir o conteúdo completo do array `FIXO`; a lista exata de chaves a emendar sai da leitura do script na revisão.

## 4. Rascunho do critério pré-registrado

Pergunta: dentro de um bloco de prefill de 8 posições, quanto do tempo de parede vai para cada fase (preparo, lançamento no host, espera da GPU, round trip com a linha de serviço, CPU dos experts, cópias de experts, cabeça de saída)? Descritivo, sem meta, sem PASSA.

Hipóteses H1..H7 como na seção 1, com os limiares ali. Previsão geral a conferir (só "dentro/fora da previsão"): a soma das fases medidas cobre ≥ 90% do tempo do bloco (se menos, há uma fase não vista e o resultado é descrito como incompleto).

Séries e células (por modelo, 1B e 3B, uma janela cada):
* Séries: L10 (padrão), L100 (`--fracao-slots 1.0`). LC (camadas inteiras na CPU) só se sobrar tempo; é candidata a corte primeiro.
* Textos: curto (89 posições), médio (170), longo (364), os mesmos da 5c (o binário só aceita os textos pelo sha, e o critério anterior já provou isso).
* Para H6 usa-se o bloco final parcial de cada texto (restos 1, 2 e 4; o 4 e o 2 tem poder baixo; declarar isso).
* Rodadas por célula: `--so-texto` com `--repete`, 1 pedido de aquecimento descartado + 5 medidos, como a janela de 17:05; corte único permitido: 5 para 3 medidas nas células ainda não iniciadas, como no roteiro anterior.
* Par para o custo do instrumento: em cada modelo, na série L100 e no texto médio, intercalar sem/com(host+serviço)/com(host+serviço+GPU), 3 repetições de cada, ordem fixada antes. Só entra o que couber nos 20 min.

Validade (SEM PROVA em tudo se quebrar): shas congelados antes e depois (modelos, textos, referência, critério, roteiro, binário novo), ids de cada pedido iguais à referência 5c, tanto desligado quanto ligado; potência de 370 W conferida; GPU livre; timers restaurados; amostras abaixo do limite de tempo; nenhuma repetição automática; `perfil-bloco.json` presente e com o número de blocos igual ao `plano_blocos(n, 8)` do texto; fases somando ≤ tempo do bloco. Célula individual com falha = SEM PROVA só dela; as demais valem.

O que NÃO conclui: não prova causa (só localiza fase com tempo, não diz por que a fase custa); não compara com a 4090 nem com o llama.cpp; não vale para outro modelo, outra placa ou outro tamanho de bloco (BLOCO=8 fixo); o tempo "com instrumento" não é o tempo de produção; frações do nível GPU (eventos) podem estar distorcidas pelo próprio instrumento; não diz que reduzir uma fase reduz o prefill na mesma proporção (as fases se sobrepõem, GPU, lançador e serviço correm em paralelo, então a soma de tempos de fase pode passar do tempo de parede); não decide nenhuma mudança de motor.

Reporte: tabela por série x texto com, por fase, mediana e faixa em ms por bloco e % do bloco; razão bloco de 1 / bloco de 8 (H6); par com/sem e desvio; cada H marcada "dentro da previsão", "fora da previsão" ou "SEM PROVA", sem veredito geral.

Janela e duração estimadas: por modelo, 2 séries x 3 textos x (1 aquecimento + 5 medidas): no 1B, o L10 dura ~0,9 + 1,7 + 3,5 s por repetição no 1B (soma ~6 s por rodada de textos) e ~12 s no 3B, L100 ~1,2 s (1B) e ~2,6 s (3B), mais carga do modelo, criação do motor, o par com/sem e os 180 s de encerramento do roteiro. Estimativa: ~8 a 10 min no 1B e ~12 a 15 min no 3B, ambos dentro de 20 min, com o limite duro do roteiro. Estas estimativas vêm dos tempos da janela de 17:05 e não incluem o nível GPU com eventos nem o par; se somados, rever.

## 5. Riscos e perguntas em aberto ao Breno (só o que muda a decisão)

Riscos:
* O instrumento altera o tempo (eventos na stream, relógios): tratado pelo par com/sem e pelos limiares de 3% e 10%; se o custo passar, só o nível host+serviço vale.
* Pouca visibilidade da GPU sem eventos: sem o nível GPU, F3 mistura "GPU trabalhando" com "GPU esperando a CPU". Para separar H2 de H3 de forma limpa é preciso o nível GPU, ou então usar a comparação L10 contra L100 (o L100 não tem CPU de experts).
* Mudar o crate muda o binário: o resultado de 17:05 deixa de ser o do mesmo sha; o par "sem instrumento" no binário novo é a nova linha de base, não a de 17:05.
* O motor tem teste `loom` e propriedade (`loom_tests.rs`, `proptest`); mexer em `line.rs` e `engine.rs` exige rodar essas suítes (não rodei nada). Se o `Backend` trait mudar, `fake.rs` muda junto.
* Não confirmei: o número de camadas e o tamanho do vocabulário do 1B e do 3B; como `CudaBackend::event()` cria eventos (com ou sem timing); se `f32_dev` é síncrono; o conteúdo do array `FIXO`.

Perguntas ao Breno (cada uma muda o desenho):
1. Variável de ambiente (mesmo binário, par honesto) ou feature de compilação como a `diag` (binário zero custo, mas outro binário para o par)? Recomendo variável de ambiente.
2. O nível GPU (eventos CUDA por seção e por camada) entra já nesta rodada, ou primeiro só host + serviço e depois, se os dados pedirem, a GPU? Recomendo dois degraus: host + serviço primeiro (barato, sem mexer em `cuda_dense`), GPU só se F3 dominar e H2 versus H3 ficar sem separar.
3. Posso criar roteiro novo (`roda-instrumento-janela.sh`) em vez de editar o `roda-perfil-janela.sh` congelado? Recomendo roteiro novo.
4. Há limite de tempo de janela por modelo além dos 20 min (por exemplo, sem o par e sem LC no 3B se o tempo apertar)?

## 6. Decisões do exec sobre as perguntas da seção 5 (10/11/2026, antes de qualquer código ou dado)

Autorização do Breno (10/11 18:01): instrumentar o motor, com plano e critério antes dos dados, revisão nível 2 e janelas na 3080 Ti sob a autorização permanente. As quatro perguntas abaixo são de desenho e não mudam o que o Breno autorizou, então o exec decide e a revisão confere:

1. Ativação: variável de ambiente `LASTRO_PERFIL_BLOCO`, desligada por padrão. Não usa feature de compilação, para que o mesmo binário rode com e sem instrumento no par intercalado.
2. Níveis: primeiro host e serviço, sem sincronização nova. Eventos CUDA só entram em uma segunda unidade, se o primeiro nível não separar as hipóteses.
3. Roteiro: roteiro novo, com FIXO próprio e UUID da 3080 Ti, em vez de editar o roteiro congelado do perfil.
4. Tempo: duas janelas (1B e 3B), até 20 min cada, como no perfil; voxtype parado antes e religado ao fim; sem repetição automática.

O sha do binário muda: o FIXO do novo roteiro congela o binário instrumentado, e o binário atual (`65faf6e7…`) continua sendo o do perfil e da comparação já medidos.
