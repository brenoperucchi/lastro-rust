# Instrumento de prefill: implementação (10/11/2026)

Implementa o plano `plano-instrumento-prefill-20261110.md` (aprovado na rodada 136). Nada aqui foi rodado em GPU. Código: `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/perfil_bloco.rs` (módulo novo, com o mesmo bloco de decisões no topo), mais ganchos em `motor/engine.rs`, `motor/leitura.rs`, `motor/line.rs`, `geracao_regra.rs` e `bin/primeira-palavra.rs`.

## O que foi decidido (o plano deixou para a implementação)

1. Ativação. `LASTRO_PERFIL_BLOCO` liga só com o valor exato `1`. Ausente, vazia, `0`, `true`, ` 1`, `1 ` ou não UTF-8 = desligado. A leitura é feita em `Engine::new` (uma vez por motor). `Engine::new_pf(.., bool)` existe para os testes não dependerem do ambiente. Desligado, `Shared::pf` é `None`: nenhum `Instant`, `Vec` ou `Mutex`; o caminho quente só testa um `Option` (serviço) ou um `bool` local por chamada de `le_ids_com_pendente` (host).
2. Lista exata de fases do serviço, na ordem em que a thread as executa (ladrilham o bloco; o que sobra é `S_fora`):
   - `s_espera`: do fim da camada anterior (na camada 0, do início do intervalo I_g) até ver `seq[l]`. Inclui a leitura de ids, pesos e `x`, que `read_step` faz no mesmo laço e não se separa sem mexer nele.
   - `s_monta`: união de experts e faltas em `serve_layer`, mais `cpu_step` e a alocação de `y` em `run`.
   - `s_compute`: `compute_step`.
   - `s_answer`: `answer`.
   - `s_livro`: `release_held` (dentro de `serve_layer`), `pin_layer`, `victim`, `copy_after_reader`.
   - `s_end_espera`: espera de `seq[L]` em `end_token`.
   - `s_end_trab`: o resto de `end_token`. Inclui `publish`, que espera o host devolver o lado do buffer em `finish_token`: é espera dependente do host dentro de `s_end_trab`, declarada e não separada.
3. Tolerância de relógio: 1 µs (`TOL_RELOGIO_NS`) para a conservação por bloco (as fases são diferenças dos mesmos carimbos em ns inteiros, então é exata; a tolerância só cobre erro de leitura) e para soma das fases do serviço contra I_g. Por pedido: |armar..desarmar - `prefill_ms`| <= max(0,2 ms, 0,5% do `prefill_ms`), como no plano. Cobertura: resíduo do host / `prefill_ms` <= 3%; mediana de `S_fora / I_g` <= 10% (a mediana por (modelo, série) é conta da análise; o arquivo traz a fração por bloco e a mediana por pedido).
4. Mecanismo de geração e armamento. A chave de junção é o `gen` do `Published`/`TokenEpoch` (token do bloco), igual no host e no serviço. O armamento é um `AtomicBool` compartilhado (Release ao armar e desarmar, Acquire no serviço, lido DEPOIS de ver `seq[0]`) mais o instante de armar. Sem canal novo, sem tocar no protocolo do motor. I_g do bloco = [max(fim do bloco de serviço anterior, instante de armar), fim de `end_token`]. O serviço decide "armado" na camada 0; o decode (desarmado) não registra nada. Com o instrumento ligado a thread ainda carimba o decode em acumuladores locais (sem alocar) e descarta; o custo disso aparece no desvio do `passo_ms` do par.
5. Merge. `Armado::fecha`, chamado em `gera_corpo` depois de `total_ms` (fora de `t_pre..prefill_ms` e de `total_ms`), espera até 500 ms pelos registros do serviço (o do último bloco chega depois de `end_token`). Registro que não chegou = pedido "serviço aberto" (`servico_aberto`); registro sem bloco do host (decode ou outro pedido) = `servico_extra > 0`, que invalida o pedido.
6. Armar e desarmar: `perfil_armar` logo depois de `t_pre`; `logits_ini()` antes de `le_logits`; `desarma()` logo depois. A guarda `Armado` desarma e descarta no `Drop` (erro no meio do prefill). O plano de blocos só é calculado com o instrumento ligado.
7. Aquecimento e índice do pedido: `MotorGeracao::perfil` (uma entrada por pedido concluído) e `MotorGeracao::aquecimento` (marcado por `gera_serie` no pedido k = 0). O binário junta as entradas de `guardado.perfil` e grava `perfil-bloco.json` no fim de `gera_prova`, só se ligado.
8. Binário: `gera_prova` imprime em stdout `perfil-bloco: ligado` ou `perfil-bloco: desligado` (estado lido da variável) logo depois da linha de integridade; `gera_livre` imprime a mesma linha em stderr (stdout é o texto gerado). O portão de UUID não foi tocado.

## Esquema final de `perfil-bloco.json`

Raiz: `tipo` ("perfil-bloco"), `versao` (1), `variavel`, `bloco` (8), `unidades`, `tolerancias` {relogio_ns, pedido_ms, pedido_frac, cobertura_host, cobertura_servico}, `pedidos` (lista).

Cada pedido:
- `indice`, `aquecimento`, `n`, `plano_m` (m de cada bloco), `prefill_ms`, `decode_passo_ms_mediana` (mediana dos `passo_ms` do binário, entra só em H2).
- `blocos[]`: `indice`, `gen`, `m`, `pos_ini`, `f0_pendente`, `f0_ms`, `f1_ms`, `f2_ms`, `f3_ms`, `b_ms`, `t_ns` ([t0..t4] desde a origem do processo), `conserva`.
- `servico[]` (um por bloco, na mesma ordem): `indice`, `gen`, `aberto`; se fechado: `ini_ns`, `fim_ns`, `i_ms`, `s_espera_ms`, `s_monta_ms`, `s_compute_ms`, `s_answer_ms`, `s_livro_ms`, `s_end_espera_ms`, `s_end_trab_ms`, `s_fora_ms`, `s_fora_frac`, `inter_bloco_ms` (interseção de I_g com [t0, t4]), `coerente`.
- Host por pedido: `f5_ms` (`le_logits`), `r_ini_ms`, `r_entre_ms[]`, `r_fim_ms`, `residuo_host_ms`, `cobertura_host_frac`, `cobertura_host_ressalva`, `armar_ate_desarmar_ms`.
- `r6` (B do bloco final parcial / B do último bloco cheio anterior; `null` sem os dois).
- `servico_aberto`, `servico_extra`, `s_fora_frac_mediana` (`null` sem serviço fechado).
- `valido` e `problemas[]` (plano diferente de `plano_blocos(n, 8)`, bloco que não conserva, blocos sobrepostos, identidade soma(B)+F5+resíduos = armar..desarmar, contraste com `prefill_ms`, serviço incoerente, I_g consecutivos sobrepostos, `servico_extra`). Cobertura é ressalva, não invalida.

Campos de H1 a H6: H1 `f2_ms`, `b_ms`, `f3_ms`; H2 `f3_ms`, `b_ms`, `decode_passo_ms_mediana`; H3 `s_compute_ms`, `s_livro_ms`, `b_ms`; H4 `f1_ms`, `b_ms`; H5 `s_livro_ms`, `s_end_trab_ms`, `b_ms`; H6 `m`, `b_ms`, `r6`.

## Funções puras no módulo (testadas em CPU)

`interpreta`, `classifica_controle` (ordem D L L D D L, magnitude, limites 3% e 10% inclusivos, par truncado, ordem trocada, D que produziu perfil ou imprimiu "ligado", L sem perfil, processo abortado, orçamento), `faixas_h6`/`h6_descritivo`/`classifica_h6`, `RecBloco::conserva`, `RecServico::{coerente,inter_ns}`, `servicos_sem_sobreposicao`, `AccServ` (ladrilhamento e corte da camada 0 no armar), `RecPedido::{identidade_host, bate_com_prefill, cobertura_host, r6, problemas}`.

## O que não coube

- Classificadores de H1 a H5 (dentro/fora/sem leitura): só H6 e o controle viraram função pura. Os campos que H1 a H5 usam estão no arquivo; a conta fica para a análise.
- Versão de H3 recortada na interseção com [t0, t4]: o serviço acumula somas por fase, não intervalos por camada, então só `inter_bloco_ms` (I_g inteiro) existe. Recortar por fase pede guardar intervalos por camada.
- Suíte `loom`: `RUSTFLAGS="--cfg motor_loom" cargo test --lib motor` não compila neste repositório nem no HEAD limpo (erros de `SchedCfg`/`new_sched` em `geracao_regra`, `lastro_cli`, `lastro_servidor`, `primeira_palavra_regra`, `reset_regra`, módulos que já existiam), então não foi possível rodá-la. Os ganchos de `line.rs`/`engine.rs` só usam `std` (nada do `sync` do loom) e não mexem na ordem de memória do protocolo.
- Roteiro `roda-instrumento-janela.sh`, parser de controle e relatório: fora desta unidade.
- Medidas na GPU: custo real do instrumento, ids iguais à 5c com a variável ligada e desligada, e as hipóteses.
