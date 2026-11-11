# Plano: F2 nível B, só o pré-roteador (11/11/2026, versão 2, antes de qualquer código ou dado)

Só plano e rascunho de critério. Nada foi editado, compilado ou rodado. Base: decisão do Breno `.herdr/DECISAO-BRENO-F2-NIVEL-B-PRE-ROTEADOR-20261111-1123.txt`, decisão de alcance `.herdr/DECISAO-BRENO-ALCANCE-INSTRUMENTO-NIVEL-B-20261111-1139.txt` e resultado `s3080-f2-{1b,3b}-real-20261111-1100/f2-host-resultado.md` (HB: pré-roteador 78,6 a 81,3% de F2; HA 99%; HE 0; HD sem leitura). Molde: `plano-f2-host-20261111.md` e `criterio-f2-host-20261111.md`. Código lido em `rust/moe-rt/src`.

## Mudanças desta versão (rodada 157, rev-1 e rev-2)
- rev-1 P2-1 e rev-2 P2-1 (IB1 sem dado): o armazenamento passa a ser de registros brutos (op, camada, posição, início, fim), não somas. Seções 4 e 6. IB2 descrita como identidade com teste de cobertura, não como prova independente.
- rev-1 P2-2 e rev-2 P2-3 (attention): `gpu_model.rs` entra na lista; variante de `attention` com `Option` mede os dois lançamentos internos e separa o preparo do helper em `B_ATTN_PREP`. Seções 3 e 4; caminho de `gpu_model.rs:345` conserva o comportamento.
- rev-2 P2-2 (desligado e teste): texto corrigido para o desenho com `Option`; o que cada teste prova está declarado. Seções 4, 6 e 10.
- rev-2 P3-1: HB1 descritiva, com a frase obrigatória (seção 7). P3-2: mediana de `H_PRE_LAYER` ao lado da do B de 11:00, entre janelas (seção 5). P3-3: entrega fora de F2, zerar por passo (seção 4). P3-4: caso do `Fake` com F2B ligada (seção 10.4).
- Nova seção 12 com a decisão de alcance de 11:39; o item correspondente saiu da seção 11.

## 1. Objetivo
Abrir por dentro `H_PRE_LAYER` (engine.rs 231-232) e dizer, em ns e em parcelas, onde o host gasta tempo ali. Descritivo, sem veredito causal. Mudança no motor fora do instrumento e push seguem pedindo o Breno.

## 2. O que o código real faz em `pre_layer` (nada inventado)
- `CudaBackend::pre_layer` (cuda_backend.rs 733-747): `self.st.lock()` (734), depois `for p in 0..cur_m()` chama `Dense::attn_block_at` (739-742). Sem modelo denso, retorna Ok sem fazer nada.
- `attn_block_at` (cuda_dense.rs 206-223): dois `mem::swap` (`cs`, `lens`), troca de `cur`, chama `attn_block`, desfaz tudo.
- `attn_block` (cuda_dense.rs 235-275), por posição: prelúdio (`cfg.clone()`, `gen_norm`, `gemv_generics` x2, `gen_rope`, com `to_string`); 9 expressões `.generics(..).async_on(s)` no corpo (rmsnorm, gemv q/k/v, rope q/k, gemv o, axpby, rmsnorm); a chamada `attention(..)` (264) que, em `gpu_model.rs` 43-58, faz `assert` de forma (cinco `dim`, cada um um `Vec`), monta `gen_split` e lança `attn_split` e `attn_combine`; 6 `dtod` (`memcpy_dtod_async`, 78-82).
- Total: 11 lançamentos e 6 cópias por posição; em bloco cheio (m = 8), 88 e 48 por camada, na stream de cálculo, só enfileirando.
- Trava `st`: só a thread do motor a pega em `pre_layer`; a thread de serviço (`line.rs`, `leitura.rs`) chama `copy_expert`, `record`, `query`, `relax`, `sync`, que não tomam `st` (cuda_backend.rs 662-705, 889-896). Sem disputa.

## 3. O que se mede dentro do pré-roteador
Registros por camada e bloco, por posição `p` quando couber:
- `B_LOCK` (1 por camada): do início de `pre_layer` até o `lock()` concedido. Mede o custo de pegar uma trava livre.
- `B_SETUP` (2 por posição): swaps e troca de `cur` antes de `attn_block`, e a restauração depois (cuda_dense.rs 206-223).
- `B_GENERICS` (1 por posição): prelúdio de `attn_block` (`cfg.clone` e os `Vec<String>`).
- `B_ATTN_PREP` (1 por posição): `assert`, `dim` e `gen_split` dentro de `attention` (gpu_model.rs 43-53), antes do primeiro lançamento.
- `B_LANCA` (11 por posição): cada expressão `.generics(..).async_on(s)`, as 9 de `attn_block` e as 2 de `attention` (inclui a montagem de argumentos do lançador do cutile e o que ele chama).
- `B_DTOD` (6 por posição): cada `dtod`.
- `B_RESIDUO` = `H_PRE_LAYER` menos a soma dos intervalos medidos (chamada do trait, laço, carimbo do motor, lacunas entre registros). Calculado depois, nunca ajustado.
Não se mede:
- `cuLaunchKernel` isolado: o símbolo só é chamado em `bin/a1-doorbell.rs`; no modelo está dentro de `async_on` (dependência externa). O relatório fala só em "tempo da expressão `async_on`".
- Execução na GPU, espera da GPU, fila do driver, tempo por kernel na placa, CPU fora da thread do host.
- `route`, `launch_hits`, `add_y`, `post_layers`: continuam só no nível A.

## 4. Instrumento
- Variável `LASTRO_PERFIL_F2B`, só o valor exato `1`; exige `LASTRO_PERFIL_BLOCO=1` e `LASTRO_PERFIL_F2=1`. Sem as duas, a linha de estado diz `perfil-f2b: desligado (exige perfil-bloco e perfil-f2)` e nada muda.
- Desenho: método novo `Backend::pre_layer_f2b(&self, s, layer, acc: &mut AccB)` com padrão que chama `pre_layer` (o `Fake` e o desligado ficam idênticos). O motor só chama a variante com F2B ligada. `CudaBackend` sobrescreve e chama `attn_block_at` e `attention` nas variantes com `Option<&mut AccB>`; o caminho antigo (`pre_layer`, `gpu_model.rs:345`) passa `None`.
- Desligado: sem acumulador, sem leitura de relógio, sem alocação. Há, sim, um teste de `Option` vazio por ponto de medida nas variantes internas (mesmo precedente do nível A, `enqueue_inner`); o plano não promete "nenhum branch", promete "um teste de `Option` vazio, sem relógio e sem alocação".
- Armazenamento: registros brutos de 24 bytes (op, camada, posição, início, fim) em vetor de capacidade fixa `layers x (21 x max_m + 1)`, reservado na criação do motor com F2B ligada (3.4 KB por camada com m = 8). Nenhuma alocação entre t2 e t3. O acumulador é zerado a cada `enqueue_step`; a entrega por bloco copia fora de F2, depois de t4, como `perfil_f2_entrega` do nível A (custo em R_entre do bloco seguinte, medido pelo controle).
- Saída: `perfil-f2b.json` (esquema próprio, versão 1) com os registros brutos, `gen`, `layer`, `m`, `pos_ini`, junção com `perfil-f2.json`. Soma e contagem por operação são derivadas pelo extrator. Extrator novo próprio.
- Carimbos: 2 por registro, encadeados por `agora_ns()` onde possível; cerca de 170 por camada, 4000 por bloco no 1B (24 camadas) e 5400 no 3B (32). Dezenas de ns cada: centenas de µs contra dezenas de ms por bloco. Estimativa, a medir.

## 5. Custo do próprio instrumento e controle
- Mesma janela e classes do nível A. Braços D (sem variáveis), A (`BLOCO=1`) e B2 (`BLOCO=1`, `F2=1`, `F2B=1`), L100 médio, quadrado latino de 9 processos por modelo, `--repete 3`, mínimo 3 válidos por braço.
- Medida principal: |desvio| de `prefill_ms` de B2 contra A: <= 3% comprovada; <= 10% com ressalva; > 10% a janela daquele modelo vira SEM PROVA (dados brutos preservados, sem repetição automática). Apoio: B2 contra D.
- Condição de integridade, citando a I7 do nível A: os ids gerados em B2 são iguais aos de D e A.
- Reportar pior caso, ms por bloco e desvio do `passo_ms` do decode.
- Denominador: as parcelas dividem por `H_PRE_LAYER` do B2, já inflado pelo instrumento. O relatório traz a mediana de `H_PRE_LAYER` por camada do B2 ao lado da do B de 11:00 (F2 sem F2B), com a ressalva "entre janelas". Alternativa no item 11.3.

## 6. Invariantes (rascunho)
- IB1: registros em ordem, sem sobreposição dentro de uma posição, e o primeiro início e o último fim dentro de `[ini, fim]` de `H_PRE_LAYER` da mesma camada no `perfil-f2.json` (tolerância 1 µs). Conferido pelo extrator sobre os registros brutos. Falha invalida o pedido.
- IB2: soma dos intervalos + `B_RESIDUO` = `H_PRE_LAYER`. É identidade por construção; vale como teste de cobertura e excesso: resíduo < -1 µs invalida o pedido. A conservação independente vem da IB4.
- IB3: contagens por camada: `B_LOCK` 1, `B_SETUP` 2m, `B_GENERICS` m, `B_ATTN_PREP` m, `B_LANCA` 11m, `B_DTOD` 6m; camadas 0 a layers-1 sem repetir nem faltar. Divergência invalida o pedido.
- IB4: a soma de `H_PRE_LAYER` do `perfil-f2.json` mais as demais operações mais `R_borda` = t3 - t2 (I2 do nível A, 2 µs), e `gen`, `m`, `pos_ini` iguais aos do bloco.
- IB5 (ressalva): `B_RESIDUO` / `H_PRE_LAYER` <= 3%; acima, ressalva de cobertura.

## 7. Hipóteses descritivas (limiares a aprovar; sem causa)
Blocos cheios elegíveis (sem o primeiro de cada pedido), mediana por camada somada, razão a `H_PRE_LAYER`:
- HB1: parcela de `B_LOCK`. Descritiva. Texto obrigatório: "mede o custo de pegar uma trava livre", na resolução do relógio; a trava não tem disputa (seção 2).
- HB2: parcela de `B_SETUP` + `B_GENERICS` + `B_ATTN_PREP`. Descritiva.
- HB3: parcela de `B_LANCA`. Dentro se >= 0,60, fora se < 0,40, sem leitura no meio (rascunho).
- HB4: parcela de `B_DTOD`. Descritiva.
- HB5 uniformidade: p90/mediana de `H_PRE_LAYER` por camada e dispersão entre camadas. Descritiva.
- HB6 escala: `H_PRE_LAYER` contra índice da camada e contra `m`/`pos_ini`. Descritiva.
Classes: dentro, fora, sem leitura, sem dado, SEM PROVA, descritivo. Nenhuma frase causal.

## 8. Riscos
1. Efeito do observador: carimbo por lançamento pode mudar o ritmo do host e a fila do driver; só o controle mede. > 10% = SEM PROVA.
2. Caminho quente em `cuda_backend.rs`, `cuda_dense.rs` e `gpu_model.rs`: um `Option` por ponto de medida e as variantes não podem mudar ordem, argumentos, stream, `fail_k`/`sync_on_err`, tempo de vida dos tensores nem a restauração dos swaps em caso de erro.
3. Resíduo fecha por construção: por isso IB1 e IB4 usam dados externos aos intervalos.
4. Alocações do prelúdio e de `attention` variam com o alocador; entram em `B_GENERICS` e `B_ATTN_PREP`.
5. Durações de µs perto da resolução do relógio: mediana, p90 e mínimo.
6. `loom` e `loom_tests.rs` não podem ser afetados.

## 9. Fica de fora
Mudança de desempenho no motor; `route`, `launch_hits`, `add_y`, `post_layers` por dentro; tempo na GPU, Nsight, eventos CUDA; `cuLaunchKernel` isolado; F0, F1, F3, L10, LC, decode; 4090 e 5090; llama.cpp e Strata; push.

## 10. Ordem de trabalho
1. Este plano; 2. critério congelado `criterio-f2-nivel-b-pre-roteador-20261111.md` (seções 5 a 7, FIXO com sha de plano, critério, binário, extrator, roteiro); 3. código só de instrumento em `perfil_bloco.rs`, `backend.rs`, `engine.rs`, `cuda_backend.rs`, `cuda_dense.rs` e `gpu_model.rs`.
4. Testes sem GPU, sempre com `CUDA_VISIBLE_DEVICES=` vazio e filtro por módulo (`perfil_bloco::`, `motor::leitura_tests`):
   - só `1` liga; desligado não aloca nem lê relógio (acumulador e sonda testados em si);
   - o `Fake` prova só a escolha da variante e a ordem das chamadas do motor; com F2B ligada o padrão do trait deixa o acumulador vazio e o pedido sai inválido pela IB3 (o teste espera isso e não se relaxa a IB3);
   - o caminho CUDA desligado idêntico se prova por leitura na revisão de código, não por teste;
   - extrator com fixtures de soma e contagem iguais mas ordem, sobreposição ou contenção diferentes (só as válidas passam), junção com outra camada ou geração, resíduo negativo, capacidade máxima sem realocação.
5. Revisão nível 2, limite de 6 rodadas, com o pedido da seção 12. 6. Cópia congelada do binário, sha no FIXO, `--verifica-fixos`.
7. Janela na 3080 Ti, uma por modelo (1B e 3B), teto de 1200 s por janela/modelo (global, corte só em células não iniciadas, controle nunca cortado), voxtype parado e religado, 370 W conferido, sem repetição após SEM PROVA.
8. Resultado `f2b-resultado.md` por modelo, descritivo; commit por passo aprovado; sem push.

## 11. Depende do Breno
1. Aprovar este plano e os limiares de HB3 (rascunho).
2. Confirmar o controle de três braços (9 processos) ou só A e B2; e se quer trocar D por um braço B (F2=1, F2B=0) na mesma janela, para medir o inchaço do pré-roteador dentro dela (D continua conferível contra A).
3. Aceitar que o relatório não fale em `cuLaunchKernel`, só em tempo da expressão `async_on`.
4. Autorizar as janelas depois do critério congelado e da revisão.
5. Qualquer mudança de desempenho no motor volta como pergunta nova depois do resultado.

## 12. Decisão do Breno sobre o alcance (11/11 11:39)
O método novo no trait `Backend`, com implementação padrão idêntica ao comportamento atual, e os pontos de medida em `cuda_dense.rs` e `gpu_model.rs` contam como instrumento. Condições: desligado por padrão, nenhum comportamento do motor muda, e os revisores conferem isso explicitamente.
Pedido explícito aos revisores: conferir, no código e nos testes, que (a) F2B é desligada por padrão e só liga com as três variáveis; (b) o padrão do trait é idêntico ao `pre_layer` atual; (c) as variantes de `attn_block_at` e `attention` mantêm ordem de kernels, argumentos, stream, caminhos de erro, swaps e restauração, e o chamador de `gpu_model.rs:345` não muda; (d) o desligado não aloca nem lê relógio; (e) nada fora do instrumento foi alterado no motor.
