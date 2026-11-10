# Instrumento de prefill: implementação (10/11/2026)

Implementa o plano `plano-instrumento-prefill-20261110.md` (aprovado na rodada 136). Nada aqui foi rodado em GPU. Verificação sem GPU: `CUDA_VISIBLE_DEVICES=""` e sempre com filtro por módulo (`--lib perfil`, `--lib motor::`, `--lib geracao_regra`, `--lib leitura_regra`, `--lib primeira_palavra_regra`, `--bin primeira-palavra gera_so_texto`). Atenção: `cargo test` SEM filtro NÃO roda sem GPU, porque inclui testes de `cuda_backend` e `gpu` que abrem CUDA (na rodada 137 isso ocupou a 3080 Ti por cerca de 10 min); o plano, seção 3, diz "`cargo test` normal, que já roda sem GPU" e essa frase vale só para os testes novos deste módulo, filtrados. Os testes deste instrumento não usam GPU; a suíte completa usa. Código: `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/perfil_bloco.rs` (módulo novo, com o mesmo bloco de decisões no topo), mais ganchos em `motor/engine.rs`, `motor/leitura.rs`, `motor/line.rs`, `geracao_regra.rs` e `bin/primeira-palavra.rs`.

## O que foi decidido (o plano deixou para a implementação)

1. Ativação. `LASTRO_PERFIL_BLOCO` liga só com o valor exato `1`. Ausente, vazia, `0`, `true`, ` 1`, `1 ` ou não UTF-8 = desligado. A leitura é feita em `Engine::new` (uma vez por motor). `Engine::new_pf(.., bool)` existe para os testes não dependerem do ambiente. Desligado, `Shared::pf` é `None`: nenhum `Instant`, `Vec` ou `Mutex`; o caminho quente só testa um `Option` (serviço) ou um `bool` local por chamada de `le_ids_com_pendente` (host).
2. Lista exata de fases do serviço. Cada operação nomeada é medida com carimbo de INÍCIO e de FIM, e só a soma desses intervalos entra nas fases; o que a thread faz entre operações é `S_fora`:
   - `s_espera`: da entrada de `serve_layer` (na camada 0, no máximo de I_g.ini) até ver `seq[l]`. Inclui a leitura de ids, pesos e `x`, que `read_step` faz no mesmo laço e não se separa sem mexer nele.
   - `s_monta`: união de experts e faltas em `serve_layer` (depois de `release_held`), mais `cpu_step` e a alocação de `y` em `run`.
   - `s_compute`: `compute_step`.
   - `s_answer`: `answer`.
   - `s_livro`: `release_held` (dentro de `serve_layer`), `pin_layer`, `victim`, `copy_after_reader`. O diagnóstico/injeção de `diag` que vem entre `answer` e `pin_layer` (`line.rs` ~l.761) fica FORA: o carimbo de `s_livro` começa depois dele, e esse tempo vira `S_fora`.
   - `s_end_espera`: espera de `seq[L]` em `end_token`.
   - `s_end_trab`: o resto de `end_token`. Inclui `publish`, que espera o host devolver o lado do buffer em `finish_token`: é espera dependente do host dentro de `s_end_trab`, declarada e não separada.
   Os limites de I_g são carimbos próprios: `ini` = max(fim do I_g anterior, instante de armar), `fim` = carimbo lido logo depois de `end_token`. Não dependem das somas. Pequenas lacunas de nanossegundos entre operações (um carimbo e o seguinte) aparecem em `S_fora`, de propósito. Validade: cada operação com fim >= início, dentro de I_g, em ordem e sem sobreposição entre si (thread única), e as somas por fase iguais às das operações.
3. Tolerância de relógio: 1 µs (`TOL_RELOGIO_NS`) para a conservação por bloco (as fases são diferenças dos mesmos carimbos em ns inteiros, então é exata; a tolerância só cobre erro de leitura) e para soma das fases do serviço contra I_g. Por pedido: |armar..desarmar - `prefill_ms`| <= max(0,2 ms, 0,5% do `prefill_ms`), como no plano. Cobertura: resíduo do host / `prefill_ms` <= 3%; mediana de `S_fora / I_g` <= 10% (a mediana por (modelo, série) é conta da análise; o arquivo traz a fração por bloco e a mediana por pedido).
4. Mecanismo de geração e armamento. A chave de junção é o `gen` do `Published`/`TokenEpoch` (token do bloco), igual no host e no serviço. O armamento é um `AtomicBool` compartilhado (Release ao armar e desarmar, Acquire no serviço, lido DEPOIS de ver `seq[0]`) mais o instante de armar. Sem canal novo, sem tocar no protocolo do motor. I_g do bloco = [max(fim do bloco de serviço anterior, instante de armar), fim de `end_token`]. O serviço decide "armado" na camada 0; o decode (desarmado) não registra nada. Com o instrumento ligado a thread ainda carimba o decode em acumuladores locais (sem alocar) e descarta; o custo disso aparece no desvio do `passo_ms` do par.
5. Merge. `Armado::fecha`, chamado em `gera_corpo` depois de `total_ms` (fora de `t_pre..prefill_ms` e de `total_ms`), espera até 500 ms pelos registros do serviço (o do último bloco chega depois de `end_token`). Registro que não chegou = pedido "serviço aberto" (`servico_aberto`); registro sem bloco do host (decode ou outro pedido) = `servico_extra > 0`, que invalida o pedido.
6. Armar e desarmar: `perfil_armar` logo depois de `t_pre`; `logits_ini()` antes de `le_logits`; `desarma()` logo depois. A guarda `Armado` desarma e descarta no `Drop` (erro no meio do prefill). O plano de blocos só é calculado com o instrumento ligado.
7. Aquecimento e índice do pedido: `MotorGeracao::perfil` (uma entrada por pedido concluído) e `MotorGeracao::aquecimento` (marcado por `gera_serie` no pedido k = 0). O binário junta as entradas de `guardado.perfil` e grava `perfil-bloco.json` no fim de `gera_prova`, só se ligado.
8. Binário: `gera_prova` imprime em stdout `perfil-bloco: ligado` ou `perfil-bloco: desligado` (estado lido da variável) logo depois da linha de integridade; `gera_livre` imprime a mesma linha em stderr (stdout é o texto gerado). O portão de UUID não foi tocado.

## Esquema final de `perfil-bloco.json` (versao 2)

Raiz: `tipo` ("perfil-bloco"), `versao` (2), `variavel`, `bloco` (8), `unidades`, `tolerancias` {relogio_ns, pedido_ms, pedido_frac, cobertura_host, cobertura_servico}, `pedidos` (lista).

Cada pedido:
- `modelo` (nome da célula; o binário o preenche, o motor não o conhece), `indice` (reinicia em 0 a cada motor: a chave num arquivo com várias células é o par `(modelo, indice)`), `aquecimento`, `n`, `plano_m` (m de cada bloco), `prefill_ms`, `decode_passo_ms_mediana` (mediana dos `passo_ms` do binário, entra só em H2).
- `blocos[]`: `indice`, `gen`, `m`, `pos_ini`, `f0_pendente`, `f0_ms`, `f1_ms`, `f2_ms`, `f3_ms`, `b_ms`, `t_ns` ([t0..t4] desde a origem do processo), `conserva`.
- `servico[]` (um por bloco, na mesma ordem): `indice`, `gen`, `aberto`; se fechado: `ini_ns`, `fim_ns`, `i_ms`, as sete somas integrais `s_espera_ms`, `s_monta_ms`, `s_compute_ms`, `s_answer_ms`, `s_livro_ms`, `s_end_espera_ms`, `s_end_trab_ms`, as sete durações RECORTADAS em [t0, t4] do bloco do host de mesma geração (`s_espera_recorte_ms`, `s_monta_recorte_ms`, `s_compute_recorte_ms`, `s_answer_recorte_ms`, `s_livro_recorte_ms`, `s_end_espera_recorte_ms`, `s_end_trab_recorte_ms`; calculadas no merge a partir das operações, à parte das somas integrais, que alimentam o H3 principal), `s_fora_ms`, `s_fora_frac`, `inter_bloco_ms` (interseção de I_g com [t0, t4]), `n_ops` (operações guardadas), `coerente`.
- Host por pedido: `f5_ms` (`le_logits`), `r_ini_ms`, `r_entre_ms[]`, `r_fim_ms`, `residuo_host_ms`, `cobertura_host_frac`, `cobertura_host_ressalva`, `armar_ate_desarmar_ms`.
- `r6` (B do bloco final parcial / B do último bloco cheio anterior; `null` sem os dois).
- `servico_aberto`, `servico_extra`, `s_fora_frac_mediana` (`null` sem serviço fechado).
- `valido` e `problemas[]` (plano diferente de `plano_blocos(n, 8)`, bloco que não conserva, blocos sobrepostos, identidade soma(B)+F5+resíduos = armar..desarmar, contraste com `prefill_ms`, serviço incoerente (inclui operações fora de ordem ou sobrepostas), I_g consecutivos sobrepostos, `servico_extra`). Cobertura é ressalva, não invalida.

Campos de H1 a H6: H1 `f2_ms`, `b_ms`, `f3_ms`; H2 `f3_ms`, `b_ms`, `decode_passo_ms_mediana`; H3 `s_compute_ms`, `s_livro_ms`, `b_ms` (versão recortada: `s_compute_recorte_ms`, `s_livro_recorte_ms`); H4 `f1_ms`, `b_ms`; H5 `s_livro_ms`, `s_end_trab_ms`, `b_ms`; H6 `m`, `b_ms`, `r6`.

## Funções puras no módulo (testadas em CPU)

`interpreta`, `classifica_controle` (ordem D L L D D L, magnitude, limites 3% e 10% inclusivos, par truncado, ordem trocada, D que produziu perfil ou imprimiu "ligado", L sem perfil, processo abortado, orçamento), `faixas_h6`/`h6_descritivo`/`classifica_h6`, `RecBloco::conserva`, `RecServico::{coerente,inter_ns,recorta}`, `servicos_sem_sobreposicao`, `AccServ` (operações com início e fim, lacuna em `S_fora`, corte da camada 0 no armar, decode sem registro), `RecPedido::{identidade_host, bate_com_prefill, cobertura_host, r6, problemas}`.

Classificação pré-registrada de H1 a H6 (seção 1.3 do plano), funções puras sobre as medianas: `classifica_h1`, `classifica_h2`, `classifica_h3`, `classifica_h4`, `classifica_h5`, `classifica_h6_medida` e o despachante `classifica_h(Hip, Serie, &MedidasH, &ClasseControle)`, com `ClasseH` = Dentro, Fora, SemLeitura, SemDado, SemProva, Descritivo. Os limiares são os do plano, com `>=`/`<=` e `<`/`>` como escritos. Interpretações onde o plano deixa margem:
- H1, H2 e H4 só têm previsão no L100; H3 e H5 têm uma por série (L10 e L100). Em série sem previsão (LC, ou L10 em H1/H2/H4) o resultado é `SemDado`, não "sem leitura".
- "Sem leitura" é só a faixa intermediária. Dado ausente ou não finito é `SemDado`. Controle `SemProva` ou `Incompleto` dá `SemProva` antes de olhar qualquer número; `Comprovada` e `ComRessalva` classificam.
- Previsões compostas (H2; H3 no L10): "fora" decide com UMA medida que viole, mesmo faltando a outra; "dentro" exige as duas; sem decidir e faltando uma delas é `SemDado`.
- H1: "dentro" usa F2/B e "fora" usa F3/B. Se as duas valerem juntas (soma > B), o dado é incoerente: `SemDado`.
- H6: resto fora de 1..=7, ou faixas que se tocam (restos 6 e 7), é `Descritivo`; sem R6 é `SemDado`.
- R3 de H3 usa a soma integral; a razão com o recorte fica à parte e quem chama escolhe qual número passar.

Testes (g): um por H com valores nos limites exatos e ao redor ("dentro", "fora", "sem leitura"), dado ausente, par de controle incompleto (par truncado e |desvio| de 12%) e resto que torna H6 descritivo.

## O que mudou na rodada 137 (resposta aos pareceres rev-1 e rev-2)

1. rev-1 #1 (P2): a cobertura do serviço era satisfeita por construção. Agora cada operação tem início e fim medidos, só esses intervalos somam, I_g tem limites independentes e o diagnóstico/injeção de `line.rs` não é mais absorvido. Testes pelo acumulador real: 85 ms de 100 ms com lacuna de 15 ms dá `S_fora/I_g` = 0,15, ressalva e durações sem absorver a lacuna; caso sem lacuna dá resíduo zero; operações sobrepostas, fora de ordem ou fora de I_g invalidam.
2. rev-1 #2 (P2): as operações ficam guardadas (capacidade reservada na criação do serviço) e o merge calcula o recorte em [t0, t4] de cada bloco, em campos `*_recorte_ms` separados. Testes: I_g=[0,10] ms, host [5,10] ms, compute em [0,5] ou [5,10] (mesma soma integral, recortes 0 e 5 ms); serviço parcialmente sobreposto; publicação depois de t4; merge e JSON.
3. rev-2 P3-a: classificadores de H1 a H6 e teste (g), acima.
4. rev-2 P3-b: campo `modelo` em cada entrada; teste com dois modelos no mesmo arquivo.
5. rev-2 P3-c: a frase sobre `cargo test` foi corrigida no topo desta nota; o plano não foi editado.
6. `versao` do arquivo passou a 2.

## O que não coube

- Custo de alocação de `fecha`: as operações do bloco são copiadas para o registro (uma alocação por bloco armado) depois do carimbo de fim de I_g; o instrumento ligado paga isso, e o par do controle mede. Fora das operações de cada camada nada aloca.
- Suíte `loom`: `RUSTFLAGS="--cfg motor_loom" cargo test --lib motor` não compila neste repositório nem no HEAD limpo (erros de `SchedCfg`/`new_sched` em `geracao_regra`, `lastro_cli`, `lastro_servidor`, `primeira_palavra_regra`, `reset_regra`, módulos que já existiam), então não foi possível rodá-la. Os ganchos de `line.rs`/`engine.rs` só usam `std` (nada do `sync` do loom) e não mexem na ordem de memória do protocolo.
- Roteiro `roda-instrumento-janela.sh`, parser de controle e relatório: fora desta unidade.
- Medidas na GPU: custo real do instrumento, ids iguais à 5c com a variável ligada e desligada, e as hipóteses.
