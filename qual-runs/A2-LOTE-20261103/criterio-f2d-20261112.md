# Critério congelado da F2D, plano da O3 (escrever direto no cache K/V, 12/11/2026)

Fontes: `plano-o3-cache-direto-20261112.md` (versão 2, aprovada na rodada 173) e a decisão do Breno de 12/11 às 00:37 (`DECISAO-BRENO-O3-OITO-PONTOS-20261112-0037.txt`): O3 com dois kernels novos aceitando ganho pequeno ou nulo, base comum C0 contra C1, dois passos (K e depois V) com uma só janela de medida ao fim, `read_kv` como pendência separada, sem tolerância de bits (kernel que não der os mesmos bits do cache: desistir daquele passo), limites de prefill +1,5% e decode +2% com efeito distinguível só abaixo de -1%, regravação da F2C com a junção da coleta, e a autorização das janelas pelo nome (coleta de bits nos dois modelos, teste de erro em GPU nos dois modelos, medição 1B e medição 3B) pela autorização permanente da 3080 Ti, com voxtype parado e religado e teto de 1200 s por janela e modelo. Revisão nível 2, sem push.

Este arquivo ainda NÃO está congelado: os shas e os hashes de commit abaixo são `PREENCHER NO CONGELAMENTO` e só se preenchem depois da revisão nível 2 do código (C0 e C1) e do pacote.

## 1. Braços, binários e sua fonte

- D_O1: binário de C0, nada ligado.
- D_O3: binário de C1, nada ligado.
- B2_O3: binário de C1 com perfil-bloco, perfil-f2 e perfil-f2b ligados (instrumento v3, 15 registros por posição, nenhum B_DTOD).
- Contexto apenas: binário 22d0bbe0 (instrumento v1, dados de 14:10) e o binário `504eca75` da F2C (O1 sem a infraestrutura V3), nunca como referência de decisão.

Fonte reproduzível, dois commits locais (sem push), cada binário construído do seu commit:

- C0 (base comum): o HEAD da O1 mais a infraestrutura comum. Traz o contrato V3 do instrumento do nível B (`ContratoB::V3`, `ORDEM_POS_B_V3` de 15 registros, capacidade `layers x (15 x max_m + 1)`, texto da contagem, IB3 por contrato) e a constante `CONTRATO_B_PRODUCAO`, ainda no V2 (17 registros), de modo que o motor emite exatamente o que emitia em 71dfc8e. Mantém os kernels e as 2 cópias do cache K/V. Traz também o pacote F2D derivado do da F2C (este critério, `roda-f2d-janela.sh`, `f2d-extrai.py`, `f2b-extrai-v3.py` e os testes). O gancho de coleta já existia. Hash: PREENCHER NO CONGELAMENTO
- C1 (candidato): C0 mais a O3. O diff `git diff <C0> <C1> --stat` se restringe a estes arquivos, e a nada mais:
  - `rust/moe-rt/src/gpu.rs`: os dois kernels novos, `rope_kv` e `gemv_row` (os kernels antigos ficam intactos);
  - `rust/moe-rt/src/cache_direto.rs` (novo) e `rust/moe-rt/src/lib.rs` (só a linha do módulo): `inicio_da_linha`, a conferência `cur < max_len` e que `cur` cabe em `i32`, com testes de CPU;
  - `rust/moe-rt/src/cuda_dense.rs`: `attn_block_b` (V e K gravam direto na linha `cur`; as duas cópias saem), a remoção dos buffers `v` e `kr` e da função `dtod`, e o injetor `FALHA_NO_LANCAMENTO_O3`;
  - `rust/moe-rt/src/cuda_backend.rs`: somente o wrapper `injeta_falha_no_lancamento_o3`;
  - `rust/moe-rt/src/perfil_bloco.rs`: somente a troca de `CONTRATO_B_PRODUCAO` para V3 (a emissão dos registros B_DTOD removidos);
  - testes: `tests/motor_passo_gpu.rs` e `tests/motor_reforma_gpu.rs` (estendidos), `tests/o3_contratos_compila.rs`, `tests/o3_kernels_compila.rs` e `tests/o3_kernels_gpu.rs` (novos).
  Custo medido que a O3 acrescenta além dos kernels e da cópia que sai, todo ele no D_O3 e no B2_O3 (não no D_O1): três leituras relaxadas de `AtomicUsize` por chamada de `attn_block_b` (`injeta_o3` 3, 1 e 2); a conferência `inicio_da_linha`; dois vetores de generics por chamada (`gen_rope_kv`, `gen_gemv_row`, um clone no K) e o cálculo automático do `DivHint` do escalar `cur` no lançador, que entra na chave do cache de JIT. Nada disso tem registro do instrumento próprio: no B2_O3 cai no prelúdio (`B_GENERICS`, os vetores) ou no resíduo do nível B (HC5). C1 também deixa de alocar dois buffers pequenos (`v`, `kvd` floats, e `kr`, `kv_heads·d` floats) e passa a não ter `dead_code`.
  Hash: PREENCHER NO CONGELAMENTO
- Antes do congelamento, conferir commits, o diff C0 a C1 e os caminhos com o instrumento desligado, e registrar os hashes e os papéis dos dois binários aqui e na coleta.
- Build de cada binário (worktree fora da árvore principal e fora do /tmp, alvo de compilação próprio, sem GPU):
  `git worktree add <pasta-c0> <hash-c0>` e `git worktree add <pasta-c1> <hash-c1>`; em cada uma, `cd rust/moe-rt && CUDA_VISIBLE_DEVICES= CARGO_TARGET_DIR=<alvo-proprio> cargo build --release --bin primeira-palavra`. O binário vai para `.herdr/binarios/primeira-palavra-f2d-c0` e `.herdr/binarios/primeira-palavra-f2d-c1`, sem permissão de escrita.
- Onde entram os shas (todos preenchidos no congelamento): `[binario-base]` (C0), `[binario-cand]` (C1), `[script-f2b-extrai-v3]`, `[script-f2d-extrai]`, `[criterio-f2d]`, `[plano-o3]` e `[decisao-o3]` no FIXO de `roda-f2d-janela.sh`, e as linhas abaixo.

Shas congelados (o roteiro recusa se qualquer um divergir):

- hash do commit base: PREENCHER NO CONGELAMENTO
- hash do commit candidato: PREENCHER NO CONGELAMENTO
- binário C0 (D_O1): PREENCHER NO CONGELAMENTO
- binário C1 (D_O3 e B2_O3): PREENCHER NO CONGELAMENTO
- f2b-extrai-v3.py: PREENCHER NO CONGELAMENTO; f2d-extrai.py: PREENCHER NO CONGELAMENTO (o f2b-extrai-v2.py e o f2c-extrai.py da F2C, e o f2b-extrai.py de 14:10, ficam intocados com os shas que o critério da F2C cita)

## 2. Desenho

Por modelo (1B e 3B), texto médio, 9 processos em quadrado latino 3x3 (D_O1 D_O3 B2_O3, B2_O3 D_O1 D_O3, D_O3 B2_O3 D_O1), cada um com `--repete 3` (1 aquecimento e 3 medidas). Teto de 1200 s por janela e modelo. Portões herdados da F2C: UUID da 3080 Ti (GPU-b22d9465-8552-9f97-812e-380b590645f8), 370 W, `--verifica-fixos` sem pipe mascarando o código, binário congelado por sha sem escrita, nenhum `cargo test` nem outro processo de GPU em execução.

## 3. Estatística

Mediana das medianas dos 3 processos de cada braço, célula do médio. Referência única: D_O1 da mesma janela. Razão = D_O3 sobre D_O1 menos 1 (e B2_O3 sobre D_O3 para o efeito do instrumento). Queda nunca é regressão.

Rótulos por métrica (prefill e passo de decode), todos os casos:

- regressão: razão acima de +1,5% no prefill ou acima de +2% no decode;
- efeito distinguível: razão abaixo de -1% E as 3 medianas por processo de D_O3 todas abaixo das 3 de D_O1;
- sem efeito distinguível: qualquer outro caso fora da regressão (razão entre -1% e o limite, razão abaixo de -1% com medianas sobrepostas, razão positiva abaixo do limite, uma métrica abaixo de -1% e a outra não), relatado por métrica com a razão e o motivo;
- sem dado ou SEM PROVA: dado ausente, controle incompleto ou integridade quebrada.

O resultado pode ser nulo, e nulo não é falha do plano.

## 4. Hipóteses

- HC1: ids gerados idênticos entre os três braços e iguais às referências 5c. Qualquer id diferente é regressão.
- HC1b: hash sha256 dos bits f32 dos logits (último token do prefill e cada passo do decode) e do KV por camada ao fim do prefill, C0 contra C1. Vem de uma EXECUÇÃO DE COLETA PRÓPRIA (`roda-f2d-janela.sh --coleta`, fora da janela de medida, que não mede desempenho): dois processos, um por binário, mesmas entradas e `--repete 3`, só `LASTRO_GANCHO_LOGITS=1`. O extrator valida envelope e cobertura (modelo, 4 pedidos com índice 0 de aquecimento, ids do prompt e gerados iguais à referência 5c, um hash de logits por token, KV de todas as camadas, hashes de 64 dígitos) e a procedência (shas dos binários, dos dois arquivos e das entradas contra o manifesto de integridade), e recusa o mesmo arquivo nos dois papéis. O par de binários da coleta (C0 e C1) tem de ser, por papel, o da janela de medida; entradas e critério iguais; a integridade antes x depois também vale; o contrato é `--repete 3`, exatamente 4 pedidos. O resultado da janela sai com `--coleta`, para o HC1b aparecer no próprio `f2d-resultado`. Qualquer bit diferente é regressão e, nesse caso, o passo da O3 que o causou é desistido (decisão 5 do Breno); coleção ausente, incompleta, incoerente ou sem procedência é SEM PROVA.
- HC2a (descritiva): H_PRE_LAYER por camada de B2_O3, nas camadas baixas (cortes refixados com os dados da F2C de 11/11 20:36: 1B camadas 0 a 11 baixas, camada 12 intermediária, patamar de 13 em diante; 3B camadas 0 a 10 baixas, patamar de 11 em diante, sem camada de transição). Referência descritiva do B_DTOD por camada da F2C (B2_novo, mediana entre pedidos e processos, blocos cheios `m = 8` sem o primeiro): 13,69 a 14,21% nas camadas 0 a 11 do 1B e 13,89 a 14,14% nas 0 a 10 do 3B; no B2_O3 o B_DTOD é zero por construção. As parcelas agregadas de 11,4% (1B) e 10,0% (3B) ficam só como descrição da F2C. Não é teto da queda nem critério de aceitação; a tabela por camada entra inteira no resultado.
- HC2b (descritiva): no patamar, a queda do H_PRE_LAYER depende do tempo de GPU das cópias e pode migrar para o B_LANCA; relatar B_LANCA + B_DTOD (agora só B_LANCA).
- HC3: prefill de D_O3 contra D_O1, limite +1,5%. Acima: regressão.
- HC4: passo de decode de D_O3 contra D_O1, limite +2%. Acima: regressão.
- HC5: resíduo do nível B em B2_O3 até 3%.
- HC6: B2_O3 contra D_O3 (prefill): até 3% comprovada, até 10% com ressalva, acima de 10% SEM PROVA e conta como regressão.
- HC1 a HC6 só valem com integridade, ordem do quadrado latino, binário por braço, referências 5c e controle completo (9 processos válidos). Qualquer falha: SEM PROVA.
- Regressão: qualquer id ou bit de logit ou de KV diferente; HC3 ou HC4 acima do limite; HC6 SEM PROVA; erro de processo. Sem ganho em `prefill_ms` mas com H_PRE_LAYER menor não é regressão.

## 5. Execução de coleta, teste de erro em GPU e quem cuida do voxtype

A coleta é execução própria, autorizada pelo nome (decisão do Breno, ponto 8), nos dois modelos, e leva no mesmo envelope o teste de erro em GPU nos DOIS modelos (`--teste-gpu --worktree-cand DIR`): `cargo test --release --test motor_reforma_gpu --test motor_passo_gpu --test o3_kernels_gpu -- --ignored --nocapture` com `E4B_GPU_AUTORIZADO=1` (o `o3_kernels_gpu` confere, isolado, `rope_kv` e `gemv_row` bit a bit contra `rope` e `gemv` e que nenhuma outra linha do cache é escrita, nas duas formas reais `(hidden, kv_heads, head_dim)` = (1024, 8, 64) do 1B e (1536, 8, 64) do 3B, com cache de 16 linhas e `cur` em 0, 1, 2, 4, 8 e 15, que cobrem as cinco classes de divisibilidade do JIT, o meio e a última linha; o teste imprime formas e posições que passaram). O `cargo test` roda com `TMPDIR` numa pasta própria dentro de `target-f2d` do worktree candidato (fora do `/tmp`, fora da conferência de árvore limpa), criada por `mktemp -d` e removida ao fim só pelo caminho literal que o `mktemp` imprimiu; o caminho fica no manifesto como `teste_gpu_tmpdir`, saída e código gravados. Mesmos portões da janela: UUID da 3080 Ti, 370 W, processos proibidos, FIXO e shas dos binários. O `--worktree-cand` só é aceito (antes de parar os timers) se `git -C DIR rev-parse HEAD` for o hash do commit candidato (C1) desta seção e `git status --porcelain` estiver vazio (o alvo `target-f2d` fica de fora); o manifesto grava o commit e o sha dos três executáveis (`motor_passo_gpu-*`, `motor_reforma_gpu-*` e `o3_kernels_gpu-*`), e `f2d-coleta-resultado.json/.md` trazem `teste_gpu` (executado, rc, passou, procedência completa nos três executáveis). O roteiro não para nem religa o voxtype: quem para e religa é o script de abertura em `.herdr`, tanto na janela de medida quanto na coleta.

## 5b. Risco de JIT por divisibilidade de `cur` e como o aquecimento o cobre

Verificado no código do cutile (rev 5759cb3): o lançador gerado pela macro calcula, para todo parâmetro escalar inteiro de um kernel, `DivHint::from_value(cur as i32)` a cada lançamento (`cutile-macro/src/kernel_launcher_generator.rs`), o gerador de entrada emite `assume_div_by` quando o divisor é maior que 1 (`kernel_entry_generator.rs`) e esse hint entra na chave do cache de JIT (`tile_kernel.rs`, `scalar_hints`). Logo `rope_kv` e `gemv_row` especializam por classe de divisibilidade de `cur`: o divisor é a maior potência de 2 que divide `cur`, limitada a 16 (`cur` 0 conta como 16). São 5 classes (1, 2, 4, 8, 16), até 5 compilações por kernel e por conjunto de generics (um modelo por processo). `assume_div_by` é só uma dica do compilador, não muda a ordem de soma; os bits são conferidos pela coleta (HC1b).

Cobertura: o pedido 0 (aquecimento, `--repete 3`) roda o prefill do texto médio L100 (cerca de 100 posições, de `cur` 0 em diante, por `attn_block_at_b`, que fixa `cur` em cada posição do passo) e o decode que o segue. Os 16 primeiros valores de `cur` cobrem as 5 classes, então todas as compilações acontecem no aquecimento, antes de qualquer medida, e os pedidos medidos só acertam o cache. Se um pedido medido mostrar tempo de JIT (um pedido medido muito acima dos demais do mesmo processo), é sinal de classe não coberta: o exec relata o achado junto do resultado, sem reclassificar sozinho. O escalar não vai por outra via: passar `cur` por um tensor de 1 elemento (como `lens`) evitaria a especialização, mas mudaria os kernels e o custo de lançamento além do que o plano aprovado (versão 2) especifica; fica como alternativa, só com nova decisão.

## 6. Fora do critério

Mudança de desempenho no motor além da O3 exige nova decisão do Breno. `read_kv` (robustez de `Arc::try_unwrap`) é pendência separada. Comparação com o B2 de 14:10 ou com o binário `504eca75` é só contexto.
