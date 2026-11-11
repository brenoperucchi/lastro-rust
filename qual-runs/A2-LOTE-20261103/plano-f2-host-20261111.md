# Plano: F2 do host por operação em L100, prefill do Lastro (11/11/2026, versão 3, antes de qualquer código ou dado)

Este arquivo é só o plano e o rascunho do critério. Nada foi editado, compilado ou rodado para escrevê-lo. Tudo abaixo vem da leitura do código em `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src` (`motor/leitura.rs`, `motor/engine.rs`, `motor/line.rs`, `cuda_backend.rs`, `cuda_dense.rs`, `perfil_bloco.rs`) e dos resultados `s3080-instr-{1b,3b}-real-20261110-2104/instrumento-resultado.md`. Todos os números e carimbos de partida citados vêm desses arquivos.

## Mudanças desta versão (versão 3, revisão 146, rev-1 e rev-2)

- rev-1 P2-1 (`tb_l` não é o instante da GPU): a grandeza passa a ser a defasagem assinada entre o fim do enfileiramento no host (fim de `H_SEQ_l`) e o carimbo pós-leitura do serviço `tb_l`. Valores negativos são conservados com sinal. Saem "quanto a GPU está atrás do host", "passo da GPU" (agora "intervalo entre carimbos do serviço") e a latência pequena deduzida de S_espera. Entra o carimbo `tb_obs_l` (primeira observação de `seq[l]`, antes das cópias). Tratado em 3.2, 3.3, 3.4, 4 (I8), 5 (HI), 8.4 e 9 (item 8).
- rev-1 P2-2 e rev-2 P3-1 (classes da HI): HI fica somente descritiva, como a HF, sem classes e sem o rótulo "GPU espera o host". Se o Breno quiser classes, regras numéricas completas antes do congelamento (9, item 9). Teste do `Fake` troca "regimes" por teste de que o relatório preserva os números sem escolher regime (8.4).
- rev-1 P3 (contagem): totais brutos (55, 189, 225) separados das amostras elegíveis depois de excluir o primeiro bloco de cada pedido (50, 180, 220; 30 e 132 com corte para 3 pedidos no curto e no longo). Tratado em 5 e 7.
- rev-2 P3-2 (I8 e junção): a conferência `tb_l >= fim(H_SEQ_l) - TOL_RELOGIO_NS` não foi adotada como invalidação, porque a decisão do exec conserva deltas negativos com sinal. A I8 confere monotonia dos `tb_l` e dos `tb_obs_l`, e a geração `t` gravada em cada registro de serviço tem de casar com a `gen` do bloco (I4). Delta negativo é reportado, não invalida.
- rev-2 P3-3 (cópia dentro do `tb`): declarado em 3.2 que `tb_l` inclui leitura, cópia e escalonamento do serviço, sem separá-los.
- rev-2 P3-4 (restos de texto): "H5" trocado por "HF"; a parte do serviço fora da fase passa a "já medida, exceto os carimbos `tb_obs_l` e `tb_l`"; I7 antes de I8.

### Versão 2 (revisão 145, rev-1 e rev-2)

- rev-1 P2-1 e rev-2 P2-1 (contagem): `H_WAIT_EVENT` vira uma entrada por bloco (soma), sempre presente, com `n_ready` em campo à parte. Tratado em 3.2, 3.3, I3 e seção 8 (teste com `n_ready` 0, 1 e maior que 16). I3 = 6 x layers + 7; capacidade layers x 6 + 8.
- rev-2 P3-1 (camadas e blocos): 1B = 24 e 3B = 32 camadas (`config.json`); números derivados corrigidos em 3.3 e no risco 4; blocos por texto em L100: curto 55, médio 189, longo 225.
- rev-1 P2-2 e rev-2 P3-2 (bordas): cadeia interna com `agora_ns()` próprio na entrada e antes do `Ok(())`; `R_borda` explícito. Tratado em 3.2, seção 4 (I2, I4, I6) e testes.
- rev-1 P2-3 (HD/HE e linha 45): o nível A localiza chamadas demoradas e mede uniformidade; sem leitura causal de bloqueio. Tratado em 2, 5 (HD, HE) e 6.
- rev-2 P1-1 (`tb` por camada): adotado como instrumento de serviço, sem mudar o motor. Tratado em 1 (ponto de partida), 3.2, 3.4, 5 (nova HI), 8 e 9 (item novo).
- rev-1 P2-4 e rev-2 P3-4 (HF): HF fica só descritiva; se o Breno quiser classes, tabela numérica antes do congelamento (9, item 7).
- rev-2 P3-3 (HB): renomeada para o pré-roteador e com a comparação explicitada (5).

## 1. Objetivo

Decisão do Breno: abrir a F2 por operação em L100. Pergunta: em que parte do host o tempo vai dentro da F2 (preparação, lançamento, sincronização, publicação ao serviço).

Ponto de partida medido (instrumento por bloco, 3080 Ti, L100, blocos cheios de 8 posições):
- F2 do host é ~75% do bloco no 1B e ~81% no 3B (3B L10, para referência: F2 80,5% a 81,0%).
- Em L100 o serviço espera ~97% do tempo e computa 0,2 a 0,3%. Ou seja, o serviço não é o gargalo; o tempo de parede está em F2.
- F3 (sincronização da stream depois de F2) é grande, estável e maior que um passo de decode: 1B 4,59 a 4,64 ms (23,7% de B), F3/passo(D) de 1,64 a 1,66; 3B 5,90 a 5,93 ms (18,0% de B), F3/passo(D) de 1,29. Isso é compatível com a GPU ter ainda mais de um passo de trabalho na fila quando o host termina de enfileirar, mas não prova causa. Por isso a pergunta "o tempo de F2 é custo do host ou espera da GPU" fica em aberto; os carimbos de serviço de 3.2 (HI) dão só números descritivos e não a respondem.

Regras desta fase:
- Descritivo. Sem meta de aprovação, sem PASSA, sem veredito geral. Nada prova causa.
- Instrumento ligado por variável de ambiente e desligado por padrão, com par de controle do custo, como no instrumento anterior.
- Janelas só na RTX 3080 Ti (UUID GPU-b22d9465-8552-9f97-812e-380b590645f8). 4090 e 5090 nunca entram.
- Nenhuma mudança de desempenho no motor nesta fase. Só medição. Qualquer mudança de desempenho exige nova decisão do Breno depois do resultado.

## 2. Mapa real de F2 (lido do código, nada inventado)

Os cinco carimbos do bloco em `motor/leitura.rs` (laço de `le_ids_com_pendente`):
- t0 a t1: `F0` = `pendente.take()` ou `eng.next_published()` (publicação ao host; ~0 em L100, pois o `Published` já está no canal).
- t1 a t2: `F1` = `prep.prepara(&ids[..])` = `CudaBackend::begin_step_tokens` (embedding de cada posição, `cos/sin`, `lens`; usa `copy_host_vec_to_device(...).sync_on` por posição).
- t2 a t3: **`F2` = `eng.enqueue_step(p, m)`**. É o que este plano abre.
- t3 a t4: `F3` = `eng.finish_token()` (`sync` da stream de cálculo, teste de `poison`, `returned.send(side)`, contadores).

`enqueue_step` (`motor/engine.rs:285`) faz, nesta ordem:
1. Validações: `closed`, faixa de `m`, `p.origin`, `p.gen`.
2. `enqueue_inner(&p, m)` (`motor/engine.rs:194`):
   - `b.begin_step(c, t, pos, m)` (só com `max_positions > 1`; confere `prepared`, grava `step_m`).
   - `b.upload_table(c, table, side)` (`cuMemcpyHtoDAsync`).
   - `b.wait_event(c, ready[s])` por `s` em `p.ready` (`cuStreamWaitEvent`; número variável por bloco).
   - Para cada camada `l` em `0..layers`, em ordem:
     - `pre_layer` (`attn_block_at` por posição `p` de 0 a m: rmsnorm, 3 gemv, 2 rope, vários `dtod`, `attention`, gemv de saída, axpby, rmsnorm; troca `cs`/`lens` por posição);
     - `route` (por posição: gemv do roteador e `route_topk`; depois `cuStreamWriteValue32` da palavra `m`);
     - `write_value(seq[l], t)` (`cuStreamWriteValue32`);
     - `launch_hits` (por posição: `hit_select`, `moe_gate_up`, `moe_down`);
     - `wait_value(flag[l], t)` (`cuStreamWaitValue32`, só enfileira na stream, não bloqueia o host);
     - `add_y` (por posição: `add_y` ou `add_y_scaled`).
   - `b.post_layers(c)` (`head`: rmsnorm e gemv do `lm_head` da última posição).
   - `b.write_value(c, seq[layers], t)`.
3. Depois de `enqueue_inner`: `cur_m = m`, `current = Some((side_idx, side))`, erro vira `shutdown`.

Fatos do código que moldam o desenho (a conferir na leitura, não para concluir):
- Todas as operações de F2 só enfileiram trabalho na stream de cálculo; o host não espera a GPU nelas por construção. Mas o lançamento de kernel pode bloquear o host quando a fila de lançamentos do driver enche. Se isso ocorrer, parte da F2 é espera da GPU disfarçada de lançamento (a GPU espera `flag[l]`, que só o serviço solta depois de ver `seq[l]`). O nível A, sozinho, não distingue lançamento barato de lançamento bloqueado: ele localiza chamadas demoradas e mede a uniformidade do tempo observado, e o mecanismo fica não identificado. Os carimbos do serviço (3.2, HI) acrescentam números descritivos e também não identificam o mecanismo de espera.
- Cada chamada do backend pega o mutex `st` (`self.st.lock()`), monta `generics` (`Vec<String>` com `to_string`) e chama `async_on` (cutile). Esse custo de preparo por lançamento fica dentro de cada operação nomeada no nível A (abaixo); só o nível B o separa.
- O número de lançamentos por bloco cresce com `m` e com `layers`. A contagem exata de kernels por camada (inclui `attention` e `dtod`) não foi levantada aqui: fica como tarefa da fase de código (item 7.3), e até lá a métrica "ms por lançamento" fica "sem dado".

## 3. O que se mede e como

### 3.1 Dois níveis, e só o A é aprovado nesta fase

- **Nível A (proposto):** carimbo em volta de cada chamada ao `Backend` dentro de `enqueue_inner`, mais a entrada e a saída de `enqueue_step`. Aparece no motor genérico (`motor/engine.rs`), então roda com o `Fake` nos testes sem GPU.
- **Nível B (não faz parte desta fase):** separar dentro de `CudaBackend` o lock de `st`, a montagem de `generics`, o `async_on` e o `cuLaunchKernel`. Só se o nível A mostrar que as operações de lançamento dominam F2, e só com nova decisão do Breno.

### 3.2 Operações nomeadas do nível A (e a que linha do código cada uma corresponde)

Fixas por bloco:
- `H_ENTRADA`: do `agora_ns()` próprio tomado na entrada de `enqueue_step` até o início de `enqueue_inner` (validações). O t2 da `leitura.rs` é lido fora e não é o início da cadeia.
- `H_BEGIN_STEP`: `b.begin_step`.
- `H_UPLOAD_TABELA`: `b.upload_table`.
- `H_WAIT_EVENT`: uma única entrada por bloco, com a soma dos `b.wait_event`; sempre presente, mesmo com `n_ready = 0` (duração perto de zero). `n_ready` fica em campo à parte.
- `H_POST_LAYERS`: `b.post_layers`.
- `H_SEQ_FIM`: o último `b.write_value(seq[layers])`.
- `H_SAIDA`: do fim de `H_SEQ_FIM` até um `agora_ns()` próprio tomado antes do `Ok(())` (atribuição de `cur_m`, `current`).

Total fixo: sete operações (`H_ENTRADA`, `H_BEGIN_STEP`, `H_UPLOAD_TABELA`, `H_WAIT_EVENT`, `H_POST_LAYERS`, `H_SEQ_FIM`, `H_SAIDA`). `H_BEGIN_STEP` é obrigatório em L100, que roda com `max_positions = 8`.

Bordas: `R_borda = (entrada - t2) + (t3 - saída)`, em que entrada e saída são os dois `agora_ns()` próprios; é o tempo entre os carimbos externos de `leitura.rs` e a cadeia interna.

Por camada `l` (seis por camada):
- `H_PRE_LAYER`, `H_ROUTE`, `H_SEQ` (`write_value(seq[l])`), `H_LAUNCH_HITS`, `H_WAIT_FLAG` (`wait_value(flag[l])`), `H_ADD_Y`.

Agrupamentos para a leitura (derivados, não carimbos novos):
- Preparação: `H_ENTRADA` + `H_BEGIN_STEP` + `H_UPLOAD_TABELA` + `H_WAIT_EVENT` (e, fora de F2, `F1`).
- Lançamento por camada: `H_PRE_LAYER` + `H_ROUTE` + `H_LAUNCH_HITS` + `H_ADD_Y` (+ `H_POST_LAYERS`).
- Sinalização à GPU/serviço: `H_SEQ` + `H_WAIT_FLAG` + `H_SEQ_FIM`.
- Fecho: `H_SAIDA`.
- Sincronização e publicação ao serviço dentro de F3 e F0 ficam fora desta fase (ver 6, opção F3).

Instrumento de serviço (alargamento, a confirmar pelo Breno, seção 9): o serviço (`motor/line.rs`, com o mesmo relógio `agora_ns()`) guarda, por camada e por bloco, no `perfil-f2.json`, dois carimbos: `tb_l`, tomado em `apos_espera` (`line.rs:480`), depois de `wait_layer`/`read_step`; e `tb_obs_l`, o instante da primeira observação de `seq[l]`, antes das cópias. Isso não muda o motor nem o protocolo entre as linhas; só acrescenta gravações ao acumulador que o serviço já tem. Como `tb_l` vem depois da leitura, ele inclui a leitura de ids, pesos e `x`, as cópias e o escalonamento do serviço, sem separá-los. Nem `tb_obs_l` nem `tb_l` medem a GPU; a latência de polling e de escalonamento do serviço não é separada.

Grandezas descritivas (diferenças assinadas, em ns):
- `defasagem_l = tb_l - fim(H_SEQ_l)`: defasagem entre o fim do enfileiramento no host e o carimbo pós-leitura do serviço.
- `defasagem_obs_l = tb_obs_l - fim(H_SEQ_l)`: o mesmo, com a primeira observação.
- `intervalo_l = tb_l - tb_(l-1)` e `intervalo_obs_l = tb_obs_l - tb_obs_(l-1)`: intervalos entre carimbos do serviço.
- Valores negativos (observação anterior ao retorno da chamada no host) são conservados com sinal: nunca viram zero e não invalidam. Não há ordem de sincronização que force o carimbo do serviço depois do carimbo do host.
- A fase de código confere se `apos_espera` também cobre `seq[layers]`.

### 3.3 Carimbos e custo do próprio instrumento

- Carimbos encadeados: o fim de uma operação é o início da seguinte, com `agora_ns()` (o `Instant` já usado em `perfil_bloco.rs`). Um carimbo por chamada: 6 por camada + 8 fixos (7 operações e a borda). O 1B tem 24 camadas e o 3B 32 (`config.json`; a fase de código confere em `cfg.layers`): ~152 carimbos por bloco no 1B (151 operações) e ~200 no 3B (199 operações), mais os carimbos do serviço `tb_obs_l` e `tb_l` (duas leituras de relógio por camada).
- Ordem de grandeza do custo: `Instant::now` na faixa de dezenas de ns; 152 a 200 carimbos somam alguns µs por bloco, contra blocos de ~33 ms (3B L100) e menos no 1B. Ordem de grandeza, a medir, não a assumir.
- Armazenamento sem alocar no caminho quente: vetor de capacidade fixa (`layers * 6 + 8` entradas, independente de `n_ready`) reservado na criação do motor quando a variável liga; cada bloco empurra os carimbos num acumulador local e, no fim de `enqueue_step`, entrega ao registro do bloco. Desligado, nada é alocado nem cronometrado (um branch sobre um `Option`).
- Observação importante: o carimbo entre chamadas pode mudar o ritmo do lançamento (efeito do observador). Isso só se mede pelo par de controle, não se presume zero.

### 3.4 Variável e saída

- Nova variável `LASTRO_PERFIL_F2`, só aceita o valor `1` (mesmo contrato de `interpreta` da variável anterior). Só vale com `LASTRO_PERFIL_BLOCO=1` ligado: sem ele a linha de estado diz `perfil-f2: desligado (exige perfil-bloco)` e nada muda.
- Linha de estado impressa pelo binário, no padrão atual: `perfil-f2: ligado|desligado`.
- Saída em arquivo separado `perfil-f2.json` (esquema próprio, com versão), para não mexer no `perfil-bloco.json` versão 2 nem no extrator existente. Extrator novo próprio, sem alterar o `instrumento-extrai.py`.
- O JSON traz, por bloco: `gen`, `m`, `pos_ini`, `n_ready`, lista de `(op, camada, t_ini, t_fim)` em ns, os carimbos t2/t3 do bloco (os mesmos do registro por bloco, para a junção), os dois carimbos de borda (entrada e saída) e, do lado do serviço, `tb_obs_l` e `tb_l` por camada, cada registro com a geração `t` do bloco.

### 3.5 Par de controle (custo do instrumento novo)

Três braços por modelo, só na série L100, texto médio (170 posições, resto 2): D (as duas variáveis desligadas), A (só `LASTRO_PERFIL_BLOCO=1`, o instrumento anterior, já com custo medido em 0,0% a 0,4%) e B (`LASTRO_PERFIL_BLOCO=1` e `LASTRO_PERFIL_F2=1`).
- Medida principal: desvio do `prefill_ms` de B contra A (custo do instrumento novo isolado). Medida de apoio: B contra D (custo total).
- Ordem proposta, quadrado latino de 9 processos por modelo: D A B, B D A, A B D, cada processo com `--repete 3` (1 aquecimento descartado + 3 medidas). Mínimo inviolável: 3 processos válidos por braço.
- Classes pela magnitude |desvio|, como no instrumento anterior: <= 3% comprovada; <= 10% com ressalva; > 10% todas as H daquele modelo SEM PROVA, dados brutos preservados. Reportar também pior caso, desvio em ms por bloco e desvio do `passo_ms` do decode.
- Os processos B valem como dados principais da série (9 pedidos medidos por modelo).
- A variável é passada por processo (`env VAR=1` nos que ligam, `env -u VAR` nos que desligam), com `prova-variavel.txt` por processo e recusa se o ambiente de quem chama já trouxer qualquer das duas variáveis, como no critério anterior.

## 4. Fases, intervalos e invariantes de soma

Por bloco cheio (m = 8), com os carimbos t2 e t3 do registro atual como fronteiras de F2:

- Partição de F2: F2 = t3 - t2 = soma das durações das operações do nível A + `R_borda`, em que `R_borda` é o tempo que o encadeamento deixa de fora entre t2 e o primeiro carimbo interno e entre o último carimbo interno e t3 (`R_borda = (entrada - t2) + (t3 - saída)`, explícito; a cadeia interna começa e termina em carimbos próprios dentro de `enqueue_step`, e t2 e t3 vêm do registro externo). Os carimbos das operações nunca são ajustados para fechar a soma.
- I1 (monotonia e contenção): todos os carimbos internos são não decrescentes e ficam em [t2, t3] do mesmo bloco, com tolerância de relógio de 1 µs (mesmo `TOL_RELOGIO_NS`).
- I2 (soma): soma(operações) + `R_borda` = t3 - t2, dentro de 1 µs por carimbo de borda; a conferência é contra t2 e t3 lidos fora do encadeamento, e não contra a própria soma.
- I3 (estrutura): número de operações do bloco = 6 x layers + 7 (1B: 151; 3B: 199), independente de `n_ready`; ordem das camadas 0 a layers-1 sem repetir nem faltar. Divergência invalida o pedido (não só o bloco).
- I4 (junção): `gen` e `m` de cada bloco do `perfil-f2.json` iguais aos do `perfil-bloco.json` do mesmo pedido; a geração gravada em cada registro de serviço (`tb_obs_l`, `tb_l`) igual à `gen` do bloco; soma(operações) + `R_borda` igual ao F2 do registro por bloco dentro de 1 µs.
- I5 (por camada): soma por camada de `H_PRE_LAYER + H_ROUTE + H_SEQ + H_LAUNCH_HITS + H_WAIT_FLAG + H_ADD_Y` mais as operações fixas = soma de todas as operações.
- I6 (cobertura, ressalva, não invalida): `R_borda`/F2 <= 3%; se exceder, ressalva sobre `R_borda`. Resíduo acima de 3% nunca invalida o pedido por cobertura; carimbo fora do intervalo ou operações sobrepostas continuam inválidos (I1).
- I7 (saída intacta): ids gerados do braço B iguais aos do braço D e à referência 5c (igualdade estrita, como no critério anterior). O instrumento não pode mudar o resultado.
- I8 (carimbos do serviço): `tb_obs_l <= tb_l` na mesma camada; `tb_l` e `tb_obs_l` de um bloco não decrescentes em `l` e não anteriores ao início do bloco; faltar carimbo em camada observada ou geração que não case com o bloco invalida o pedido só para a coleta de carimbos do serviço (HI); as demais hipóteses seguem. A defasagem assinada negativa não é violação: é conservada e reportada com seu sinal. A I8 só trata da coleta; ela não identifica o mecanismo de espera.
- Blocos tratados à parte, fora das medianas: o primeiro bloco de cada pedido (pode incluir primeiro uso de kernel e carga de cache de kernel; o aquecimento descartado cobre só parte disso) e o bloco final com resto (m < 8). Os dois ficam reportados, e o final serve à HF abaixo; o primeiro bloco fica à parte, para a HH.
- Medianas: razão de medianas (mediana da operação sobre mediana de F2, no mesmo conjunto de blocos cheios), não mediana de razões por bloco; mesmo contrato do critério anterior.

## 5. Hipóteses descritivas e classes

Todas as hipóteses são do L100 (único escopo), por modelo (1B e 3B), sobre blocos cheios, medianas por pedido e depois do braço B. Classes: dentro, fora, sem leitura (faixa intermediária), sem dado (ausente ou sem previsão), SEM PROVA (controle >10%, integridade quebrada, referência ausente), mais descritivo. Limiares abaixo são rascunho e só valem depois do Breno aprovar e o critério ser congelado.

- **HA, lançamento domina F2:** dentro se (`H_PRE_LAYER` + `H_ROUTE` + `H_LAUNCH_HITS` + `H_ADD_Y`)/F2 >= 0,80; fora se < 0,50.
- **HB, o pré-roteador (atenção e projeções) é a maior operação:** dentro se `H_PRE_LAYER`/F2 >= 0,40 e a soma de `H_PRE_LAYER` no bloco é maior que a soma de cada uma das outras operações por camada no mesmo bloco (comparação de somas no bloco, não de medianas por camada); fora se `H_PRE_LAYER`/F2 < 0,25.
- **HC, a sinalização é barata:** (`H_SEQ` + `H_WAIT_FLAG` + `H_SEQ_FIM` + `H_UPLOAD_TABELA` + `H_WAIT_EVENT` + `H_BEGIN_STEP`)/F2 <= 0,05 dentro; > 0,15 fora.
- **HD, uniformidade do tempo observado por camada:** razão entre a maior e a menor mediana por camada (soma das seis operações da camada) <= 1,5 dentro; > 3,0 fora; entre os dois, sem leitura. Mede só a uniformidade. Um vetor uniforme recebe a classe de uniformidade e nada mais; nenhum resultado afirma ou exclui bloqueio de fila a partir de HD.
- **HE, caudas por chamada:** parcela de F2 gasta em chamadas cuja duração passa de 10 vezes a mediana da mesma operação na mesma camada: <= 0,05 dentro; > 0,20 fora. Localiza chamadas demoradas; o mecanismo fica não identificado, e chamadas uniformemente demoradas também ficam "dentro".
- **HF, escala com m (resto), somente descritiva:** reportar a razão observada F2(bloco final com resto r) / F2(bloco cheio) e a faixa prevista (r/8 com 25% de margem; resto 2: 0,25, faixa de 0,1875 a 0,3125). Sem classe "fora" nem "dentro". Se o Breno quiser classes, tabela numérica antes do congelamento (seção 9, item 7). Só vale quando o resto for 1, 2 ou 4 (texto curto, médio e longo, como no critério anterior); r = 6 ou 7 fica descritivo.
- **HI, defasagem do host ao serviço (somente descritiva, como a HF):** publica, por modelo e sobre as amostras elegíveis, as distribuições (mínimo, mediana, p90, máximo, com sinal) e as medianas por camada de `defasagem_obs_l` e `defasagem_l`, e dos intervalos entre carimbos do serviço (`intervalo_obs_l`, `intervalo_l`), junto com a mediana das operações do host na mesma camada. Sem classes, sem limiar, sem o rótulo "GPU espera o host" e sem afirmar causa; nem a GPU nem o mecanismo de espera são identificados. Se o Breno quiser classes, regras numéricas completas antes do congelamento (seção 9, item 9).
- **HG, ms por lançamento:** só reportado depois que a fase de código fixar a contagem de kernels por bloco (7.3); sem a contagem, "sem dado". Sem limiar.
- **HH, primeiro bloco do pedido:** F2 do bloco 0 / F2 mediano dos cheios >= 1,5 dentro; reportado, sem limiar de fora (descritivo).

Contagem de blocos cheios em L100 (89, 170 e 364 posições dão 11, 21 e 45 blocos cheios por pedido). Totais brutos: curto 55 (5 pedidos), médio 189 (9 pedidos), longo 225 (5 pedidos). Amostras elegíveis para as medianas, depois de excluir o primeiro bloco de cada pedido (10, 20 e 44 por pedido): curto 50, médio 180, longo 220; com corte de 5 para 3 pedidos nas células curto e longo, 30 e 132. Pedidos inválidos reduzem mais. O primeiro bloco fica à parte, para a HH. Para o curto (poucas amostras por camada), HD, HE e HI ficam descritivas.

Controle do custo: também sai com classe (comprovada, com ressalva, SEM PROVA), como em 3.5. Sem veredito geral. A leitura final usa as classes e as razões, sem frase causal.

## 6. O que fica de fora

- Mudança de desempenho no motor (fila de lançamentos, grafo CUDA, lotes de kernels, qualquer otimização): só depois, com nova decisão do Breno.
- Nível B (dentro de `CudaBackend`: lock, `generics`, `async_on`, `cuLaunchKernel`): fora desta fase.
- F1 (`begin_step_tokens`), F0 e F3 por operação: fora. Opção a decidir pelo Breno (seção 9): partir F3 em `sync`, `poison`, `returned.send` com 3 carimbos extras; o plano padrão não inclui.
- L10, LC (camadas na CPU), decode, a parte do serviço já medida (exceto os carimbos `tb_obs_l` e `tb_l` da HI), o nível GPU (Nsight, eventos CUDA): fora.
- Comparação com o llama.cpp e com o Strata: fora.
- Qualquer 4090 ou 5090: fora.
- Hipótese causal: o plano não afirma causa de nada; classes só dizem se uma previsão pré-registrada ficou dentro ou fora.

## 7. Riscos

1. Efeito do observador: carimbos entre lançamentos podem mudar o ritmo do host e o enchimento da fila de lançamentos. Mitigação: par de controle com três braços; B contra A isola o custo novo; > 10% = SEM PROVA.
2. Soma que fecha por construção e engana: encadear carimbos faz a soma bater sempre. Mitigação: I2 confere contra t2 e t3 lidos fora do encadeamento; I1 e I3 pegam carimbo fora de ordem e operação faltando.
3. Contagem de kernels por camada desconhecida (inclui `attention` e `dtod`): HG fica sem dado até levantar por leitura de código; não inventar.
4. Variação por camada: 24 ou 32 camadas x 6 operações = muitos intervalos curtos (µs) perto da resolução do relógio e da jitter do escalonador. Mitigação: medianas sobre as amostras elegíveis, 180 (médio) e 220 (longo) por texto, depois de excluir o primeiro bloco de cada pedido (totais brutos 189 e 225); o curto (50 elegíveis, 55 brutos) fica descritivo; reportar mínimo, mediana e p90 por operação.
5. Primeiro uso de kernel (compilação/cache de cutile) no bloco 0: fora das medianas dos cheios.
6. Preempção do thread do host pelo sistema: já houve variação de F2 entre 97 e 135 ms por bloco no 3B L10; esperar caudas em L100 também e reportá-las.
7. Mudança de código no caminho quente de lançamento: precisa de teste sem GPU mostrando que desligado não aloca, não lê relógio e não muda o que o `Fake` registra (como `desligado_nao_aloca_nem_grava_nada` e os testes de `leitura_tests.rs`).
8. O `loom` e os testes de modelo do motor (`loom_tests.rs`) não podem ser afetados; o código novo não toca o protocolo de sincronização entre as linhas.
9. Integridade da janela: a 3080 Ti é a única placa; voxtype tem de estar parado; potência 370 W; sem `cargo test`/`bench` rodando; binário congelado por sha. Tudo igual ao critério anterior.

## 8. Ordem de trabalho

1. **Este plano**, revisão do Breno (aprovações da seção 9).
2. **Critério congelado** (`criterio-f2-host-20261111.md`), derivado da seção 5 e do controle de 3.5, com limiares aprovados, ordem das células, orçamento de tempo, FIXO do pacote (sha do critério, plano, binário instrumentado, extrator novo, roteiro). Congelado antes de qualquer dado.
3. **Código** só de instrumento, em `perfil_bloco.rs` (acumulador de operações do host e de `tb` por camada, classificadores das H, esquema do `perfil-f2.json`), `motor/line.rs` (gravar `tb_l` em `apos_espera`, sem mudar o protocolo) e `motor/engine.rs` (carimbos em `enqueue_inner`; `enqueue_inner` recebe um acumulador opcional e fica idêntico quando ausente) e o ponto de ativação em `bin/primeira-palavra.rs`. Também levanta a contagem de kernels por camada (`attention`, `dtod`) para a HG. Sem alterar o que os backends fazem.
4. **Testes sem GPU**, sempre com `CUDA_VISIBLE_DEVICES=` (vazio) e filtro por módulo (por exemplo `perfil_bloco::` e `motor::leitura_tests`), sem rodar o `cargo test` inteiro: variável só liga com `1`; desligado não aloca nem grava; `Fake` produz a estrutura esperada de 6 x layers + 7 operações por bloco, com `n_ready` 0, 1 e maior que 16 dando a mesma contagem e nenhuma realocação do acumulador (capacidade igual antes e depois), com 24 e 32 camadas, e I3 recusando uma operação a mais ou a menos; I1 a I5 e I8 em registros sintéticos (incluindo carimbo fora de ordem, operação faltando, camada repetida, `tb_obs_l` maior que `tb_l`, geração de serviço trocada que não se junta ao bloco); carimbos do serviço com a mesma publicação e custos de cópia diferentes, publicações distintas com o mesmo `tb_l`, e observação anterior ao retorno no host (delta negativo conservado com sinal, sem virar zero nem invalidar), o relatório preservando os deltas e seus limites de interpretação; casos de resíduo de borda 0, 0,5%, exatamente 3% e acima de 3% (os dois primeiros válidos, o último com ressalva, nenhum invalidando por cobertura); teste de que o relatório da HI preserva os números (distribuições e medianas com sinal) sem escolher regime nem emitir classe; classificadores das H nos limites exatos (>=, <=, <, >); o par de controle classifica 3%, 10% e >10%; ids idênticos com o instrumento ligado e desligado. Compilar só com autorização e fora de janela (nunca com `cargo` rodando durante uma janela).
5. **Revisão nível 2** (dois revisores cegos, até 6 rodadas, como no instrumento anterior), sobre o código, os testes e o critério congelado; terceiro parecer do scout se o exec julgar necessário após seis rodadas.
6. **Cópia congelada do binário** (sha registrado no FIXO) e verificação dos fixos (`--verifica-fixos`) antes da janela.
7. **Janela**, só na 3080 Ti, uma por modelo (1B e 3B), em invocações separadas, no máximo 20 min cada, sem repetição automática após SEM PROVA. Dentro de cada janela: preflight e smoke L100; controle de 9 processos (D A B, B D A, A B D, `--repete 3`) no texto médio; depois células curto e longo em B (1 processo, 1 aquecimento + 5 medidas; corte de 5 para 3 só nas ainda não iniciadas). Voxtype parado antes e religado depois, limite de 370 W conferido no começo e no fim, portão de sha do binário antes de tocar em timers, recusa se houver `cargo test`/`bench` ou GPU em uso. Roteiro novo a partir de `roda-instrumento-janela.sh`, com testes do roteiro sem GPU.
8. **Resultado** (`f2-host-resultado.md` por modelo), descritivo, com as classes e sem veredito causal; relatório curto a ser mostrado ao Breno. Qualquer mudança de desempenho no motor volta como pergunta nova, depois.

Commit por passo aprovado, como no ritmo do A1; nada de push sem pedido.

## 9. O que depende do Breno

1. Aprovar este plano e os limiares de 5 (HA a HH são rascunho).
2. Autorizar código só de instrumento no motor (o escopo do 8.3) e a compilação para o teste sem GPU.
3. Confirmar o controle de três braços (D, A, B; 9 processos por modelo) ou pedir só o par A e B (6 processos), com B contra A como medida única e sem o B contra D.
4. Decidir se o plano inclui a partição de F3 (sync, poison, send) como parte da mesma fase; padrão: não.
5. Autorizar as janelas na 3080 Ti depois do critério congelado e da revisão nível 2 (voxtype parado e religado fica como passo manual dele, como antes).
6. Decisão futura, depois do resultado: abrir o nível B ou qualquer mudança de desempenho.
7. Se o Breno quiser classes para HF, aprovar uma tabela numérica completa por resto (1, 2 e 4), com desigualdades inclusivas e exclusivas e tratamento da faixa intermediária, antes do congelamento; senão HF fica descritiva.
8. Confirmar que incluir os carimbos `tb_obs_l` e `tb_l` por camada do serviço (HI) é um alargamento do instrumento para o lado do serviço, ainda sem mudar o motor. A aprovação é só da coleta de carimbos do serviço; ela não identifica o mecanismo de espera, e nenhuma instrumentação de GPU está incluída. A alternativa é declarar que o nível A não distingue lançamento barato de bloqueado e registrar isso como ressalva obrigatória do resultado.
9. Se o Breno quiser classes para a HI, aprovar antes do congelamento regras numéricas completas (referência, estatísticas, tolerâncias, desigualdades inclusivas e exclusivas, faixa intermediária, nomes restritos ao que os carimbos medem); senão a HI fica somente descritiva.
