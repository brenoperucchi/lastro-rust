# Plano: instrumento de fase no passo de bloco do prefill (10/11/2026, versão 2, resposta à rodada 134)

Autorização do Breno (18:01): mudar código só para instrumentar, com plano e critério antes dos dados, revisão nível 2 (até 6 rodadas), janelas na 3080 Ti sob autorização permanente (voxtype parado, critério congelado antes dos dados, limite de tempo, sem repetição automática após SEM PROVA). Este arquivo é só o plano. Nada foi editado, compilado ou rodado para escrevê-lo; tudo abaixo vem de leitura do código em `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src` e dos pareceres da rodada 134. A seção "Mudanças desta versão" no fim diz onde cada achado foi tratado.

Base empírica (perfil 10/11, 3080 Ti, janela de 17:05): prefill linear nas posições. 1B: L10 ~9,8 ms/posição, L100 ~2,44 (decode 23,1 e 2,8 ms/passo). 3B: L10 ~20, L100 ~4,1 (decode ~45 e ~4,6). Razão ms/posição ÷ decode: ~0,42 a 0,45 no L10, ~0,87 a 0,90 no L100. Com blocos de 8, um bloco custa ~3,4 a 3,6 passos de decode no L10 e ~7 no L100. O L100 não tem CPU de experts em jogo (100% dos experts na GPU), e mesmo assim o bloco custa ~7 passos: isso sugere que a GPU não amortiza o lote. Nota: naquela janela a célula 1B L10 médio ficou SEM PROVA; a base empírica do 1B L10 vem dos outros textos.

## 1. Mapa do caminho do prefill

Cronômetro atual: `geracao_regra.rs:203` (`prefill_ms` = `le_ids_com_pendente` + `(s.le_logits)()`, de `t_pre` em :198 até :203). O decode é cronometrado por passo em `geracao_regra.rs:209` a :225 (`passo_ms`), e cada passo de decode chama `le_ids_com_pendente` com um único id (:224). A impressão está em `bin/primeira-palavra.rs:920,940,1037`.

Laço de blocos (thread principal, uma por bloco de até 8 ids):
1. `motor/leitura.rs:~82` `le_ids_com_pendente`; `plano_blocos` em :53; laço em :98. Por bloco: `eng.next_published()` (:101), `prep.prepara(ids)` (:111), `eng.enqueue_step(p, m)` (:115), `eng.finish_token()` (:116). No motor reutilizado (pedidos seguintes do `--repete`), o primeiro bloco usa o `Published` guardado (:102 a :105), então a espera do bloco 0 é zero por construção.
2. `Engine::next_published` (`motor/engine.rs:170`): espera (laço com `relax`) o `Published` que a linha de serviço publicou no fim do bloco anterior.
3. `LeitorCuda::prepara` (`cuda_backend.rs:650`) chama `begin_step_tokens` (:499) que chama `CudaDense::begin_step` (`cuda_dense.rs:180`). Por posição, três H2D síncronos (`sync_on`): `lens` e `cos/sin` em `cuda_dense.rs:193-194` (`f32_dev` é síncrono, `cuda_dense.rs:64-66`) e `h` em `cuda_backend.rs:~512`, na stream de cálculo, com a GPU ociosa.
4. `Engine::enqueue_step` (`engine.rs:255`) chama `enqueue_inner` (:185). Só enfileira, não espera (`wait_value(flag[l])` é operação da stream, não bloqueia o host): `begin_step`, `upload_table`, `wait_event` por slot, e por camada `pre_layer`, `route`, `write_value(seq[l])`, `launch_hits`, `wait_value(flag[l])`, `add_y`; depois `post_layers` e `write_value(seq[L])`.
5. Por camada, por posição `p` em `0..m`, os kernels são lançados por posição, sem lote na GPU: `pre_layer` (`cuda_backend.rs:733`, ~15 a 20 lançamentos por posição por camada), `route` (:764), `launch_hits` (:797, 3 kernels por posição por camada, pesos relidos por posição), `add_y` (:850). `post_layers` (:749) roda em todo bloco, só na última posição.
6. Linha de serviço (outra thread), `motor/line.rs:708` `run`, por camada: `serve_layer` (:435, espera `seq[l]` com `relax`, monta a união de experts do bloco e as faltas), `cpu.compute_step` (`cuda_backend.rs:942`, `Cpu::experts_step`, `cpu.rs:239`), `answer`, `pin_layer`, `victim`/`copy_after_reader` (:562), flag de resposta. No fim do bloco `end_token` (:587): espera `seq[L]`, `query` dos eventos de cópia, tabela, `publish`. A thread de serviço só faz isso (`line.rs:708-735`), então o tempo dela é espera ou serviço. O `end_token` do bloco g pode terminar depois do `finish_token` do bloco g no host, e então aparece como espera do host em F0 do bloco g+1.
7. `finish_token` (`engine.rs:285`): `cuStreamSynchronize` da stream de cálculo (`cuda_backend.rs:889`). Um por bloco. Depois `read_logits` (:611), uma vez, no fim do prompt.
8. Já existe a feature `diag` (`diag.rs`): relógios da linha de serviço por camada somados por token. Serve de molde, mas é por token e muda de feature (outro binário).

Onde provavelmente está o custo por bloco (sem afirmar causa): repetição por posição da parte densa e dos experts em GPU, round trip GPU e CPU por camada, preparo síncrono por posição, custo de lançamento do cutile. Nenhuma está medida.

### 1.1 Duas categorias de métrica (contrato do critério)

Esta é a regra que organiza todo o resto. Nenhuma métrica pertence às duas categorias.

**Categoria A, partição disjunta do tempo do host (thread principal).** Por bloco g do prefill, com carimbos t0 a t4 lidos na ordem: F0 = t1 menos t0 (`next_published`), F1 = t2 menos t1 (`prepara`), F2 = t3 menos t2 (`enqueue_step`, só lançamento), F3 = t4 menos t3 (`finish_token`, inclui o `cuStreamSynchronize` que já existe). Intervalo do bloco B_g = t4 menos t0. Por pedido: F5 = tempo de `read_logits`, e o `prefill_ms` que o binário já mede. Os intervalos não nomeados do host são o resíduo: do início do cronômetro até t0 do bloco 0 (R_ini), entre t4 do bloco g e t0 do bloco g+1 (R_entre), e de t4 do último bloco até o início de `read_logits` (R_fim).

Conservação (testável em CPU e conferida em cada pedido medido):
* por bloco, F0+F1+F2+F3 = B_g, com tolerância de 1 µs (os carimbos são gravados em ns inteiros e as fases são diferenças dos mesmos carimbos; a tolerância cobre só a serialização);
* por pedido, soma de B_g + F5 + R_ini + R_entre + R_fim = `prefill_ms`, com tolerância de 0,2 ms ou 0,5% do `prefill_ms`, o que for maior (o `prefill_ms` e os carimbos do perfil vêm de `Instant` distintos, lidos em pontos vizinhos).

**Categoria B, intervalos do serviço (thread de serviço), reportados à parte, com sobreposição permitida.** Por bloco g (geração g, definida em 1.2) e somados sobre as camadas: S_espera (espera de `seq[l]` em `serve_layer`), S_monta (união de experts e faltas em `serve_layer`), S_compute (`compute_step`), S_answer, S_livro (`pin_layer`, `victim`, `copy_after_reader`), S_end_espera (espera de `seq[L]` em `end_token`), S_end_trab (`query`, tabela, `publish`). Carimbos em ns desde uma origem comum com o host, para permitir sobreposição. O intervalo do serviço do bloco g, I_g, vai do fim da publicação do bloco g menos 1 (ou do instante em que o perfil foi armado, para o bloco 0 do pedido, o que for mais tarde) até o fim da publicação do bloco g.

Conservação do serviço (testável em CPU): a thread de serviço é única, então os intervalos nomeados do mesmo I_g não se sobrepõem entre si; a soma das fases nomeadas é menor ou igual a I_g, e S_fora = I_g menos a soma das fases nomeadas. Os I_g de blocos consecutivos não se sobrepõem. Carimbo incoerente (fim antes do início, ou intervalos nomeados sobrepostos entre si) invalida o pedido.

Não há regra de soma entre categorias. O serviço corre em paralelo com o host: somar F0..F3 com S_* pode passar de B_g sem erro. Quando for preciso falar de sobreposição, reporta-se a interseção do intervalo do serviço com [t0, t4] (sem somar duas vezes), como dado descritivo à parte.

Cobertura, com poder de falhar:
* cobertura do host: (R_ini + R_entre + R_fim) / `prefill_ms` ≤ 3%. Acima disso há tempo do host sem fase nomeada: resultado marcado "com ressalva de cobertura do host".
* cobertura do serviço: mediana de S_fora / I_g ≤ 10% por (modelo, série). Acima disso há trecho do serviço sem fase nomeada: resultado marcado "com ressalva de cobertura do serviço". A lista de fases do serviço acima vem da leitura de `serve_layer` e `end_token`; se a implementação achar trecho relevante fora da lista, ele ganha fase própria antes de congelar o critério, e o resíduo continua sendo o teste.

### 1.2 Atribuição por geração

* A geração g de um bloco é o índice dele no plano do pedido (`plano_blocos(n, 8)`), contado a partir de 0.
* O serviço do bloco g é tudo o que a thread de serviço faz de `serve_layer(0)` até o fim de `publish`, incluindo o `end_token`, mesmo que termine depois de t4 do bloco g no host.
* O `Published` do bloco g é consumido pela espera F0 do bloco g+1. Esse tempo conta em F0(g+1) (Categoria A) e em S_end_espera e S_end_trab da geração g (Categoria B). É a mesma espera vista por dois lados, e por isso as categorias não se somam.
* Bloco 0 em motor reutilizado: F0 é zero por construção (`Published` pendente). Cada bloco carrega a marca `f0_pendente`. O F0 dos blocos com `f0_pendente` é reportado à parte e excluído da mediana de F0.
* O `end_token` do último bloco do prefill pertence à geração do último bloco. Ele pode terminar depois do fim do prefill; por isso o merge entre host e serviço é feito ao fim do pedido (depois do decode), fora de qualquer cronômetro, e um I_g ainda aberto no merge marca o pedido como "serviço aberto" e o tira das estatísticas de serviço (contado no relatório).

### 1.3 Hipóteses pré-registradas (H1 a H7)

Todas as medidas abaixo são medianas por (modelo, série, texto), de pedidos medidos (sem aquecimento), de blocos cheios (m = 8), salvo onde dito. Cada H diz o campo de `perfil-bloco.json` que a mede. Cada previsão tem três saídas possíveis: "dentro da previsão", "fora da previsão" (um resultado que a contradiz) e "sem leitura" (faixa intermediária declarada). Todos os limiares foram fixados aqui, antes dos dados, e só valem se a validade (seção 4) e o controle do custo do instrumento (seção 2.1) deixarem a (modelo, série) em estado válido.

* H1 (lançamento domina o host; a GPU espera o host). Campos: `f2_ms`, `b_ms`, `f3_ms`. No L100: dentro se F2/B ≥ 0,70; fora se F3/B ≥ 0,50; senão sem leitura.
* H2 (a GPU é o limite por falta de lote), reformulada com medida do host. Campos: `f3_ms` por bloco e `decode_passo_ms_mediana` do pedido (a mediana dos `passo_ms` que o binário já grava; o decode fica fora do perfil, ver 1.4). Denominador de referência: o `passo_ms` do braço desligado do controle (seção 2.1) da mesma (modelo, série), porque é o decode sem o custo do instrumento. No L100: dentro se F3/B ≥ 0,50 e F3 / passo de decode ≥ 3,0; fora se F3/B < 0,25 ou F3 / passo de decode < 1,5; senão sem leitura. Limite declarado: F3 é a espera do host depois de todos os lançamentos, não o tempo de GPU; H2 mostra que a GPU domina a cauda do bloco, não separa "relê pesos por posição" de outras causas de GPU lenta. A medida direta do tempo de GPU por seção fica para a segunda unidade.
* H3 (round trip por camada GPU e CPU no L10), sem a parcela de espera. Campos: `s_compute_ms`, `s_livro_ms`, `b_ms` (a razão usa o intervalo do bloco do host como denominador, e as durações do serviço atribuídas pela geração, sem recorte; a versão recortada na interseção com [t0, t4] sai à parte). Definição: R3 = (S_compute + S_livro) / B. Previsão no L10: dentro se R3 ≥ 0,50 e S_compute / B ≥ 0,20; fora se R3 < 0,30 ou S_compute / B < 0,10; senão sem leitura. Previsão no L100 (sem faltas): dentro se R3 ≤ 0,10; fora se R3 > 0,25; senão sem leitura. A parcela de espera não entra em H3 porque, em uma thread que só espera ou serve, ela fecha o intervalo com ou sem a hipótese.
* H4 (preparo síncrono por posição). Campos: `f1_ms`, `b_ms`. No L100: dentro se F1/B ≥ 0,05; fora se F1/B < 0,02; senão sem leitura.
* H5 (cópias de experts e fechamento do bloco). Campos: `s_livro_ms`, `s_end_trab_ms`, `b_ms`. No L10: dentro se (S_livro + S_end_trab) / B ≥ 0,10; fora se < 0,03; senão sem leitura. No L100: dentro se ≤ 0,02; fora se > 0,05; senão sem leitura. A espera de `seq[L]` (S_end_espera) fica fora de H5, por ser espera da GPU.
* H6 (custo fixo por bloco, não por posição), com limiar escalado pelo resto. Campos: `m` e `b_ms` do bloco final parcial, `b_ms` do último bloco cheio imediatamente anterior no mesmo pedido, por pedido; a razão R6 = B(bloco final de r posições) / B(último bloco cheio anterior) é calculada por pedido e se toma a mediana. Comparar com o último bloco cheio anterior (posição de contexto mais próxima) evita misturar tamanhos de contexto; o viés residual (o bloco final lê um cache poucas posições mais longo) é declarado e favorece levemente H6. Previsões, para resto r: dentro (H6, custo fixo) se R6 ≥ 0,5 + r/16; fora (custo por posição) se R6 ≤ (r/8) × 1,25; senão sem leitura. Verificação prévia das faixas, para os três textos (os restos vêm de 89 % 8 = 1, 170 % 8 = 2, 364 % 8 = 4):

  | texto | n | resto r | H6 se R6 ≥ | por posição se R6 ≤ | faixas se tocam? |
  |---|---|---|---|---|---|
  | curto | 89 | 1 | 0,5625 | 0,1563 | não |
  | médio | 170 | 2 | 0,625 | 0,3125 | não |
  | longo | 364 | 4 | 0,75 | 0,625 | não |

  Regra: se para algum texto as faixas se tocarem, o texto é descritivo para H6. Com a fórmula, restos 6 e 7 tocam (r = 6: 0,875 contra 0,9375; r = 7: 0,9375 contra 1,094) e o resto 5 não; os três textos atuais têm resto 1, 2 e 4, então os três são testáveis, com poder maior no curto e no médio que no longo (a folga entre as faixas no longo é 0,125). A divergência anterior entre as seções 1 e 4 sobre o resto 4 está resolvida: o 4 entra, com a folga declarada.
* H7 (cabeça `lm_head` por bloco). Não testável nesta unidade: mede-se com evento CUDA ao redor de `post_layers`, que não entra aqui (decisão 2 da seção 6). Adiada para o critério da segunda unidade. Nenhum resultado desta unidade é classificado como H7.

### 1.4 Escopo do registro (o que o perfil grava e o que não)

* O perfil só é armado no prefill: um `armar` imediatamente depois de ler `t_pre` em `geracao_regra.rs` e um `desarmar` imediatamente depois de `(s.le_logits)()`, antes de parar o cronômetro em :203. O custo dos dois (um store de bool) fica dentro do `prefill_ms` e dentro do desvio medido pelo par (2.1). O decode fica FORA do perfil: nenhum registro de decode, nenhuma alocação durante o decode. O decode não serve de referência de m = 1 para H6 (H6 usa o bloco final do prefill). O `passo_ms` do decode entra só em H2, como número que o binário já grava.
* Cada pedido produz uma entrada própria em `perfil-bloco.json`: índice do pedido, marca `aquecimento` (o pedido 0 do `--repete`), `n`, o plano de blocos (`m` de cada bloco), os blocos com F0..F3 e `f0_pendente`, os S_* por geração, F5, os resíduos, `prefill_ms` e `decode_passo_ms_mediana`. Com `--repete 2` o arquivo tem 3 entradas (1 aquecimento e 2 medidas).
* Os carimbos são gravados em `Vec` pré-alocado por pedido (capacidade dada por `plano_blocos`), sem impressão no caminho quente.

## 2. Desenho mínimo do instrumento (host e serviço apenas)

Princípio: cronometrar por fase sem mudar nada do que o motor faz quando desligado.

Ativação: variável de ambiente `LASTRO_PERFIL_BLOCO=1`, lida UMA vez, na criação do motor, para um `bool` no `Engine` e na linha de serviço. Desligada por padrão. Mesmo binário com e sem instrumento, para que o par do controle (2.1) seja honesto. Como a variável é lida na criação do motor, cada braço do par é um processo separado.

Pontos (todos com `Instant`, relógio monotônico do host):
* Host (thread principal), em `le_ids_com_pendente` (`motor/leitura.rs`) ou guardados no `Engine` para não mudar a assinatura pública: t0 antes de `next_published`; t1 depois; t2 depois de `prepara`; t3 depois de `enqueue_step`; t4 depois de `finish_token`. Zero sincronização adicional.
* Serviço, em `motor/line.rs::run`: carimbos de início e fim de cada fase da Categoria B, por camada, somados por geração, ligados por `bool`. Escritora única (a thread de serviço); merge com o host só ao fim do pedido. A geração e a marca de armado chegam ao serviço junto com a publicação de `seq[0]` do bloco (o host escreve antes do `write_value`, o serviço lê depois de ver `seq[0]`). Se a implementação carimbar o serviço também fora do prefill (por exemplo na espera de `seq[0]`, que começa antes do armamento), as marcas de blocos não armados são descartadas, e o par reporta também o desvio do `passo_ms` do braço ligado. O critério exige só: nenhum registro nem alocação de decode, e a geração atribuída como em 1.2.
* Cada bloco grava índice, m, posição inicial e as fases acima. Cada pedido grava a entrada descrita em 1.4, só se a variável estiver ligada.

Garantia de desligado: sem a variável, nenhum `Instant::now`, nenhum `Vec` alocado; só `bool`s falsos testados. Os ids e os logits não mudam: o instrumento não toca no estado do modelo, nas posições, nas tabelas nem na ordem das operações da stream. Prova: os ids da referência 5c (os mesmos 3 textos, 1B e 3B, L10 e L100; LC se rodada) têm de sair idênticos com a variável desligada E ligada. Se divergir, o instrumento é inválido (SEM PROVA).

Fora desta unidade (segunda unidade, só se os dados pedirem): nível GPU, com eventos CUDA (`cuEventRecord`) ao redor de `pre_layer`, `route`, `launch_hits`, `add_y` e `post_layers`. Pendências já respondidas pela revisão: `CudaBackend::event()` (`cuda_backend.rs:409-413`) cria eventos com `CU_EVENT_DISABLE_TIMING`, então haverá construtor novo com timing; os eventos entram só na segunda unidade, com critério próprio, inclusive o controle do custo deles. H7 e a medida direta de GPU de H2 pertencem a essa unidade.

### 2.1 Controle do custo do instrumento (obrigatório)

O par ligado e desligado do nível host mais serviço é parte obrigatória da janela, não uma célula opcional. O nível GPU não faz parte do par desta unidade.

* Braços: D (binário novo, `LASTRO_PERFIL_BLOCO` ausente) e L (o mesmo binário, variável ligada). Ids iguais à 5c nos dois.
* Células de controle, por modelo: (L10, texto médio) e (L100, texto médio). Cada célula de controle roda 6 processos, em ordem intercalada fixada antes dos dados: D L L D D L. Cada processo faz 1 aquecimento descartado e 3 medidas (`--repete 3`: o aquecimento vem a mais do N; 4 pedidos por processo). Mínimo inviolável: 3 processos válidos por braço por célula. Os dados dos processos L das células de controle valem também como os dados principais dessas duas células (9 pedidos medidos: 3 processos L x 3 medidas), o que reduz o custo da janela. Todo processo D roda com `env -u LASTRO_PERFIL_BLOCO` e não pode produzir `perfil-bloco.json`; o binário imprime uma linha `perfil-bloco: ligado` ou `perfil-bloco: desligado` com o estado lido da variável, e a validade confere essa linha em cada processo (D desligado, L ligado).
* Orçamento reservado antes de tudo: a ordem de execução da janela é (1) preflight, (2) pares de controle, (3) as demais células obrigatórias (L10 e L100 nos textos curto e longo, 1 processo de 1 aquecimento e 5 medidas, com o corte único de 5 para 3 do roteiro anterior), (4) células opcionais (LC). O roteiro estima o tempo do bloco (2), a partir do tempo de carga e de prefill da janela de 17:05, antes de começar, e se o bloco (2) mais o encerramento não couber nos 20 min, a janela não começa e fica registrada como SEM PROVA da unidade inteira (sem repetição automática). Cortes por falta de tempo só atingem (3) e (4). Célula obrigatória sem dado não é SEM PROVA do par; as H que dependem dela ficam "sem dado".
* Cálculo: desvio_prefill = (mediana das medianas por processo de `prefill_ms` em L menos a de D) / a de D, por célula de controle. Reporta-se também o pior caso (maior mediana de L menos menor de D, sobre a menor de D), o desvio em ms por bloco (para checar se é constante entre séries) e o desvio de `passo_ms` do decode.
* Limiares (fixados agora), aplicados à MAGNITUDE |desvio_prefill| (o sinal continua no relatório para a análise descritiva, mas não aprova nada): |desvio| ≤ 3%: fidelidade temporal comprovada para a (modelo, série); entre 3% e 10%: descrição com ressalva, frações relativas ao tempo medido com instrumento, nunca ao do braço D; > 10%: todas as H dessa (modelo, série) ficam SEM PROVA, e os dados brutos se preservam como descrição. Sinais opostos com a mesma magnitude recebem a mesma classe (uma aceleração grande do braço L também reprova).
* Escopo do controle: vale por (modelo, série). Do médio para curto e longo da mesma (modelo, série), a extrapolação é justificada porque o número de carimbos por bloco é fixo (só muda a quantidade de blocos) e o custo do bloco é linear nas posições (base empírica); como checagem, o desvio em ms por bloco medido no médio tem de bater com o dos outros textos da célula, que também são pedidos L e não têm D, então a checagem é só interna (variação do ms por bloco entre textos ≤ 10%). Entre séries a extrapolação NÃO vale (L10 tem livro e cópias, L100 não): cada série tem seu par. A série LC, se rodada, não tem controle e é descritiva, sempre com ressalva.
* Controle ausente ou incompleto (menos de 3 processos válidos por braço em uma célula de controle, ordem diferente da fixada, processo de controle abortado): a avaliação do custo do instrumento e todas as classificações H1 a H6 da (modelo, série) ficam SEM PROVA, com os dados brutos preservados como descrição e a limitação escrita no relatório. Os testes que não dependem de tempo (ids, contagem de blocos, conservação) continuam valendo.

## 3. Mudanças de código previstas

Estimativa (linhas aproximadas, sem contar testes):
* `rust/moe-rt/src/perfil_bloco.rs` (novo): tipos `RecBloco`, `RecServico`, `RecPedido`, `Perfil` (acumuladores, conservação, cobertura, classificação do controle, resumo JSON com `serde_json`, que já é dependência), leitura da variável. ~200 a 260 linhas. Lógica pura (agregar, mediana, conservação, cobertura, resíduos, classificação das H e do controle) separada do relógio para testar em CPU.
* `motor/leitura.rs::le_ids_com_pendente` e `motor/engine.rs`: carimbos do host; o perfil fica no `Engine` (`Option<Perfil>`), ligado em `Engine::new` (:101), para não mudar a assinatura usada em `geracao_regra.rs`, `leitura_tests.rs` e `sequencia_tests.rs`; ~50 linhas.
* `motor/line.rs::run` e `Service`: carimbos das fases da Categoria B por geração; ~60 linhas.
* `geracao_regra.rs` e `bin/primeira-palavra.rs`: `armar` depois de `t_pre`, `desarmar` depois de `le_logits`; coletar o resumo depois do pedido (fora de `t_pre`..:203) e gravar `perfil-bloco.json` com uma entrada por pedido; ~40 linhas. O arquivo vai à parte e não altera o registro de geração nem o que o parser dos critérios lê.
* O trait `Backend` e `fake.rs` não mudam nesta unidade (sem eventos CUDA).

Testes sem GPU (CPU, `FakeGpu` do `motor/fake.rs`, `cargo test` normal):
* (a) `perfil_bloco`: agregação, mediana, resumo, ordem das fases; conservação da Categoria A (F0+F1+F2+F3 = B_g dentro de 1 µs; soma de B_g + F5 + resíduos = `prefill_ms` na tolerância); conservação do serviço (soma das fases nomeadas ≤ I_g, S_fora); cobertura do host e do serviço com resultado que pode falhar (resíduo de 4% no host marca "com ressalva"; S_fora de 15% marca "com ressalva").
* (a2) casos sintéticos determinísticos de sobreposição, todos devem ser aceitos quando os carimbos são coerentes: serviço totalmente sobreposto a [t0, t4] (bloco de 10 ms com F0=0,1, F1=0,2, F2=4, F3=5,7 e 7 ms de serviço dentro do intervalo: F0..F3 fecham 10 ms e a soma com o serviço dá 17 ms, sem invalidar); serviço parcialmente sobreposto (começa durante F0, termina durante F3); publicação depois do `finish_token` (fim de `publish` depois de t4, atribuído à geração g, visível como F0 do bloco g+1 sem dupla contagem na Categoria A). Casos inválidos (devem ser rejeitados): fim antes do início, intervalos nomeados do serviço sobrepostos entre si, I_g de blocos consecutivos sobrepostos.
* (b) com o `FakeGpu`, rodar `le_ids` com perfil ligado e desligado e comparar a sequência de chamadas do backend falso (mesmos ids, mesma sequência, mesma posição final): é o teste de "ligar não muda resultado".
* (c) escopo do registro: o resto de n % 8 produz o bloco final de tamanho esperado; com perfil ligado e decode de k tokens (k > 0), o número de registros de prefill continua igual a `plano_blocos(n, 8)` e há zero registros de decode; com `--repete 2` o arquivo tem 3 entradas separadas, a primeira com `aquecimento = true`; o bloco 0 de um motor reutilizado vem com `f0_pendente = true` e fica fora da mediana de F0.
* (d) variável inválida ou ausente = desligado.
* (e) um `Engine` com perfil desligado não aloca (`is_none()` ou contador).
* (f) classificação do controle, função pura: par completo (3 e 3, ordem D L L D D L) com desvios de +2%, +5% e +12% cai em "comprovada", "com ressalva" e "SEM PROVA", e com -2%, -5% e -12% cai nas mesmas três classes (a magnitude manda); os limites exatos de 3% e 10% ficam em "comprovada" e "com ressalva" respectivamente; um braço D que produza `perfil-bloco.json` ou imprima `perfil-bloco: ligado` também cai em "controle incompleto", e um D com a variável exportada no ambiente pai continua desligado por causa do `env -u`; par truncado (2 processos em um braço), ordem trocada e orçamento insuficiente caem em "controle incompleto": H1 a H6 SEM PROVA e a avaliação do custo SEM PROVA, sem que a função devolva nenhuma classificação de H.
* (g) classificação das H, função pura: para cada H, valores sintéticos que caem em "dentro", em "fora" e em "sem leitura"; para H6, a verificação de que as faixas dos restos 1, 2 e 4 não se tocam e que o resto 6 marca o texto como descritivo.

O que só a GPU testa: custo real do instrumento (o par), ids iguais à referência 5c nos modelos reais, e as próprias hipóteses.

Compatibilidade com o portão da 3080 Ti e o FIXO:
* Binário de referência preservado: o binário atual (`65faf6e7...`) está copiado em `/home/brenoperucchi/Devs/lastro-rust/.herdr/binarios/primeira-palavra-65faf6e7` (sha conferido, fora do git). O binário instrumentado é compilado com `CARGO_TARGET_DIR` próprio, para não sobrescrever `rust/moe-rt/target/release/primeira-palavra`, que os roteiros congelados usam por padrão.
* O "sem instrumento" do par é o binário NOVO com a variável desligada (mesmo sha, mesmo binário que o "com"). O `65faf6e7` é a referência das séries já medidas na janela de 17:05 e o binário das repetições das 8 células SEM PROVA, passado por `--bin` com o caminho preservado acima. Justificativa: os roteiros congelados dessas células guardam o sha `65faf6e7` no FIXO (`roda-perfil-janela.sh:81-103`, 21 chaves) e aceitam `--bin` (:69); repetir com o mesmo sha mantém as células comparáveis às medidas e não depende de o instrumento estar correto. Validação: `roda-perfil-janela.sh --verifica-fixos --bin <cópia>` deve sair com código 0 antes da janela.
* Roteiro novo `roda-instrumento-janela.sh` (o `roda-perfil-janela.sh` é arquivo congelado), com FIXO próprio contendo o sha do binário novo, do próprio roteiro, do critério novo, e do resumo de saída; o critério de ids iguais à 5c com a variável ligada e desligada; o portão de 370 W, a prova de GPU livre (voxtype parado), os timers e `systemctl` do roteiro anterior, sem mudança; e o orçamento do bloco (2) da seção 2.1.
* O `criterio-perfil-prefill-20261109.md` diz "nenhuma linha de código muda": esta frente é nova, com critério novo, sem reabrir aquele.

## 4. Rascunho do critério pré-registrado

Pergunta: dentro de um bloco de prefill de 8 posições, quanto do tempo de parede do host vai para preparo, lançamento e espera da GPU, e quanto tempo de serviço (espera, CPU dos experts, cópias, fechamento do bloco) corre em paralelo? Descritivo, sem meta, sem PASSA.

Hipóteses H1 a H7 como na seção 1.3, com os limiares e os campos ali. Cada uma tem as saídas "dentro", "fora" e "sem leitura"; H7 é "não testável nesta unidade". Cobertura: a da seção 1.1 (resíduo do host ≤ 3% do `prefill_ms`; S_fora ≤ 10% do I_g), que pode falhar. Não há regra de soma entre host e serviço.

Séries e células (por modelo, 1B e 3B, uma janela cada):
* Séries: L10 (padrão), L100 (`--fracao-slots 1.0`). LC só se sobrar tempo; é o primeiro corte e, sem controle, é sempre descritiva.
* Textos: curto (89 posições), médio (170), longo (364), os mesmos da 5c.
* H6 usa o bloco final parcial de cada texto (restos 1, 2 e 4), com a verificação de faixas da seção 1.3.
* Rodadas: as células de controle (L10 e L100, médio) seguem a seção 2.1 (6 processos, `--repete 3`); as demais células obrigatórias usam 1 aquecimento descartado e 5 medidas por `--so-texto` com `--repete`, com o corte único de 5 para 3 nas células ainda não iniciadas.

Validade (SEM PROVA em tudo, se quebrar): shas congelados antes e depois (modelos, textos, referência, critério, roteiro, binário novo, e o `65faf6e7` quando usado), ids de cada pedido iguais à referência 5c, desligado e ligado; potência de 370 W conferida; GPU livre; timers restaurados; amostras abaixo do limite de tempo; nenhuma repetição automática; `perfil-bloco.json` presente, com uma entrada por pedido e, por pedido medido, a sequência de `m` dos blocos igual a `plano_blocos(n, 8)`; conservação da Categoria A e do serviço (1.1) dentro das tolerâncias; carimbos coerentes. Célula individual com falha = SEM PROVA só dela; as demais valem.

Validade temporal (separada da anterior): depende do controle da seção 2.1. Sem controle completo e dentro do limiar, as H da (modelo, série) ficam SEM PROVA, e a avaliação do custo do instrumento também.

O que NÃO conclui: não prova causa (localiza onde há tempo, não diz por que a fase custa); não compara com a 4090 nem com o llama.cpp; não vale para outro modelo, outra placa ou outro tamanho de bloco (BLOCO=8 fixo); o tempo "com instrumento" não é o tempo de produção; não mede tempo de GPU por seção (isso é a segunda unidade); não diz que reduzir uma fase reduz o prefill na mesma proporção (host e serviço correm em paralelo); H7 não é avaliada; não decide nenhuma mudança de motor.

Reporte: tabela por série e texto com, por fase da Categoria A, mediana e faixa em ms por bloco e % de B; os S_* à parte (em ms por bloco e como razão de B, mais a interseção com [t0, t4]); F0 do bloco 0 reutilizado à parte; razão R6 e o resto de cada texto; o par (desvio de `prefill_ms`, pior caso, desvio por bloco e de `passo_ms`); cada H marcada "dentro da previsão", "fora da previsão", "sem leitura" ou "SEM PROVA", sem veredito geral.

Janela e duração: o orçamento exato sai do roteiro (seção 2.1, bloco 2), porque depende do tempo de carga do modelo, que a janela de 17:05 registra e este plano não repete sem conferir. Por modelo, o bloco obrigatório do par tem 12 processos (2 células, 2 braços, 3 processos), cada um com carga do modelo, criação do motor e 4 pedidos do texto médio (1 aquecimento e 3 medidas, `--repete 3`) (prefill de 170 posições: ~1,7 s por pedido no 1B L10, ~3,4 s no 3B L10, ~0,4 s e ~0,7 s no L100), mais os 180 s de encerramento do roteiro. Se o bloco do par não couber nos 20 min, a janela não começa. As estimativas anteriores de 8 a 10 min (1B) e 12 a 15 min (3B) valiam sem o par e sem processos separados por braço e ficam substituídas pela conta do roteiro.

## 5. Riscos e perguntas em aberto ao Breno (só o que muda a decisão)

Riscos:
* O instrumento altera o tempo (relógios no host e no serviço): tratado pelo par obrigatório e pelos limiares de 3% e 10% (2.1); sem par completo, nada é classificado.
* Sem eventos CUDA, F3 mistura "GPU trabalhando" com "GPU esperando a CPU", e H2 mede só a cauda do bloco. Separar H2 de H3 com limpeza exige a segunda unidade; nesta, usa-se a comparação L10 contra L100 (o L100 não tem CPU de experts).
* Mudar o crate muda o binário. O `65faf6e7` está preservado em `/home/brenoperucchi/Devs/lastro-rust/.herdr/binarios/primeira-palavra-65faf6e7` e o instrumentado vai para um `CARGO_TARGET_DIR` próprio; um rebuild do commit antigo não garantiria o mesmo sha.
* O motor tem teste `loom` e propriedade (`loom_tests.rs`, `proptest`); mexer em `line.rs` e `engine.rs` exige rodar essas suítes (não rodei nada).
* O orçamento de 20 min pode não comportar o par em um dos modelos; a regra é não começar a janela, não encolher o par.
* Não confirmei: o número de camadas e o tamanho do vocabulário do 1B e do 3B; o tempo de carga de cada modelo (entra na conta do roteiro); se `serve_layer` e `end_token` têm trechos fora das fases da seção 1.1 (o resíduo S_fora acusa).

Perguntas ao Breno (cada uma muda o desenho; as decisões do exec estão na seção 6):
1. Variável de ambiente (mesmo binário, par honesto) ou feature de compilação como a `diag`? Recomendo variável de ambiente.
2. O nível GPU entra já nesta rodada, ou primeiro só host e serviço? Recomendo host e serviço primeiro.
3. Posso criar roteiro novo (`roda-instrumento-janela.sh`) em vez de editar o congelado? Recomendo roteiro novo.
4. Há limite de tempo de janela por modelo além dos 20 min?

## 6. Decisões do exec sobre as perguntas da seção 5 (10/11/2026, antes de qualquer código ou dado)

Autorização do Breno (10/11 18:01): instrumentar o motor, com plano e critério antes dos dados, revisão nível 2 e janelas na 3080 Ti sob a autorização permanente. As quatro perguntas abaixo são de desenho e não mudam o que o Breno autorizou, então o exec decide e a revisão confere:

1. Ativação: variável de ambiente `LASTRO_PERFIL_BLOCO`, desligada por padrão. Não usa feature de compilação, para que o mesmo binário rode com e sem instrumento no par.
2. Níveis: primeiro host e serviço, sem sincronização nova. Eventos CUDA só entram em uma segunda unidade, se o primeiro nível não separar as hipóteses.
3. Roteiro: roteiro novo, com FIXO próprio e UUID da 3080 Ti, em vez de editar o roteiro congelado do perfil.
4. Tempo: duas janelas (1B e 3B), até 20 min cada, como no perfil; voxtype parado antes e religado ao fim; sem repetição automática.

O sha do binário muda: o FIXO do novo roteiro congela o binário instrumentado, e o binário atual (`65faf6e7...`, preservado como descrito na seção 3) continua sendo o do perfil e da comparação já medidos e o das repetições das células SEM PROVA.

Decisões acrescentadas na versão 2 (resposta à rodada 134), também de desenho:

5. Escopo do registro: o decode fica fora do perfil; o perfil é armado só no prefill; uma entrada por pedido, com marca de aquecimento (1.4).
6. H2 é reformulada com medida do host (F3 contra o passo de decode); H7 é adiada para a segunda unidade (1.3).
7. O par ligado e desligado é obrigatório, com orçamento reservado antes das células opcionais, e a falta dele marca SEM PROVA o que depende de fidelidade temporal (2.1).
8. As 8 células SEM PROVA são repetidas com o binário `65faf6e7` preservado, por `--bin`; o par usa o binário novo nos dois braços (3).
9. A janela não começa se o bloco obrigatório do par não couber no tempo (2.1), em vez de encolher o par.

## Mudanças desta versão (resposta à rodada 134)

* rev-1 #1 (P2, soma de fases): duas categorias de métrica, partição disjunta do host e intervalos do serviço com sobreposição permitida, seção 1.1; conservação só na partição, com tolerância de relógio declarada, 1.1; regra global "fases somando <= tempo do bloco" removida da validade (seção 4) e dos testes (seção 3, itens a e a2); cobertura trocada por resíduo do host (≤ 3% do `prefill_ms`) e S_fora do serviço (≤ 10%), que podem falhar, 1.1; atribuição de `end_token` e F0 por geração, 1.2; casos sintéticos (serviço totalmente sobreposto, parcialmente sobreposto, publicação depois do `finish_token`), seção 3, item a2.
* rev-2 P2-2 (contradição e previsões sem poder): mesma correção acima; H3 reformulada sem espera, com R3 = (S_compute + S_livro) / B e limiares por série, 1.3; todas as H têm uma saída "fora da previsão" e uma "sem leitura", 1.3.
* rev-1 #2 (P2, controle do custo): par obrigatório, ordem D L L D D L, 3 processos por braço no mínimo, orçamento reservado antes das células opcionais, 2.1; escopo por (modelo, série) com justificativa da extrapolação entre textos e proibição entre séries, 2.1; controle ausente ou incompleto marca SEM PROVA a avaliação do custo e H1 a H6, com dados brutos preservados, 2.1 e seção 4 (validade temporal); nível GPU retirado do par, 2.1 e seção 4; teste da classificação do controle, seção 3, item f.
* rev-2 P2-1 (escopo do registro): armar só no prefill e decode fora, 1.4; entrada por pedido com índice e marca de aquecimento, 1.4; validade confere `plano_blocos(n, 8)` por pedido medido, seção 4; F0 do bloco 0 reutilizado à parte e fora da mediana, 1.2; teste (c) cobre decode de k tokens e `--repete 2` com 3 entradas, seção 3.
* rev-2 P2-3 (H2, H7, terceiro braço): H2 reformulada com medida do host (F3 contra `passo_ms`), H7 declarada não testável e adiada, 1.3; terceiro braço do par removido, 2.1; cada H aponta para campos que o binário grava (1.3 e 1.4).
* rev-2 P2-4 (H6): limiares escalados pelo resto r, comparação com o último bloco cheio anterior, faixa intermediária "sem leitura", tabela de faixas por texto e regra de texto descritivo, divergência sobre o resto 4 resolvida, 1.3.
* rev-2 P3 (binário): `65faf6e7` preservado em `.herdr/binarios/primeira-palavra-65faf6e7`, usado por `--bin` nas repetições das 8 células, instrumentado em `CARGO_TARGET_DIR` próprio, item (1) da seção 3 corrigido (o "sem" do par é o binário novo desligado), seções 3, 5 e 6 (decisão 8).
* Também: nota sobre a célula 1B L10 médio SEM PROVA na janela de 17:05 (base empírica); respostas da revisão às pendências (`event()` com `DISABLE_TIMING`, `f32_dev` síncrono) incorporadas em 1 e 2; estimativa de duração substituída pela regra do orçamento (seção 4).
* Não coube resolver só com texto: (1) a lista final de fases do serviço e a tolerância real de relógio dependem de ler `serve_layer` e `end_token` na implementação; o critério fixa o teste (resíduo) e deixa a lista para congelar com o código; (2) o orçamento do par depende do tempo de carga de cada modelo, que não foi medido para este plano; o critério fixa a regra (não começar a janela) e o roteiro faz a conta; (3) os limiares numéricos de H1 a H6 são previsões do exec, sem dado que os calibre; só os dados dirão se estavam bem postos; (4) o mecanismo exato pelo qual o serviço sabe a geração e o armamento fica para a implementação, com o critério restrito ao resultado (sem registro nem alocação de decode).

## Mudanças da rodada 135 (antes de qualquer código ou dado)

* rev-1 #1 e rev-2 P2: o controle usa `--repete 3` (1 aquecimento descartado e 3 medidas, 4 pedidos por processo, 9 medidas L por célula, 12 processos e 48 pedidos por modelo no bloco obrigatório); seções 2.1, 4 e Janela e duração alinhadas.
* rev-1 #2: a fidelidade temporal é classificada pela magnitude |desvio|; o teste puro (f) cobre os dois sinais e os limites exatos.
* rev-2 P3: todo braço D roda com `env -u LASTRO_PERFIL_BLOCO`, não produz `perfil-bloco.json`, e o binário imprime o estado lido da variável; a validade confere por processo.
* rev-2 (observação): o 3B é o candidato a não começar se a carga do modelo for a maior parte dos 518 s da janela de 17:05; a regra de não iniciar se o bloco do par não couber continua valendo.
