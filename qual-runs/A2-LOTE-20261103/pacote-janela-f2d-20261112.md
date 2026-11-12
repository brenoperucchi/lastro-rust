# Pacote das execuções da F2D (plano da O3, escrever direto no cache K/V): coletas e janelas, 12/11/2026

## O que é
Uma janela por modelo (1B primeiro, depois 3B, nunca as duas juntas), de até 1200 s por janela e modelo na RTX 3080 Ti (UUID GPU-b22d9465-8552-9f97-812e-380b590645f8; a 4090 e a 5090 nunca entram), só o Lastro, descritiva. Mede o efeito da O3 (o gemv de V e o RoPE de K gravam direto na linha `cur` do cache, sem as duas cópias device-para-device) no prefill e no decode. Regras em `criterio-f2d-20261112.md`; plano v2 em `plano-o3-cache-direto-20261112.md` (aprovado na rodada 173); decisão do Breno de 12/11 às 00:37 em `.herdr/DECISAO-BRENO-O3-OITO-PONTOS-20261112-0037.txt`. Código revisado na rodada 174 e 175.

## Fonte e binários congelados
- Commit base C0 (base comum: contrato V3 do instrumento, produção ainda no V2, pacote F2D, sem a O3): `7c25e1d126c0e36219e5f325131eb38040a645c3`.
- Commit candidato C1 (C0 mais a O3; código em `6f7d049`, o candidato é `1fa8abb`, que traz por cima só a ampliação do teste `o3_kernels_gpu`, a regravação da F2C e ajustes do roteiro): `1fa8abb77fef9359391fc9bc323602512984094e`. O commit do pacote congelado vem por cima dele e não altera arquivo de motor. `git diff 7c25e1d 1fa8abb --stat -- rust` traz os 11 arquivos do critério e nada mais.
- Binário base (D_O1): sha256 `4a6f26c7eaf9898cc625ce16e617cc0c68983670abcfec9502f09fd6aab9579a`, em `.herdr/binarios/primeira-palavra-f2d-base-4a6f26c7`.
- Binário candidato (D_O3 e B2_O3): sha256 `f295e90a6a2d3b90332ba5bf11f2a0e337aa646806d2a5e308c6f5c2f00c9cea`, em `.herdr/binarios/primeira-palavra-f2d-cand-f295e90a`.
- Contexto apenas: `primeira-palavra-f2b-22d0bbe0` e o binário `504eca75` da F2C, nunca rodam nesta janela.
- Os dois binários foram construídos em worktrees separados (fora do /tmp, em `~/.cache/lastro-testes/f2d/wt-c0` e `wt-c1`), com alvo de compilação próprio (`target-f2d` de cada worktree), `--release --bin primeira-palavra` e `CUDA_VISIBLE_DEVICES` vazio, sem rodar nada na GPU. Cópias sem permissão de escrita (modo 555).
- Extratores: `f2b-extrai-v3.py` sha `2bcd8761...`, `f2d-extrai.py` sha `098963c9...`; critério `criterio-f2d-20261112.md` sha `9f595163...`; plano sha `e4edd1d0...`; decisão sha `12082796...`. Todos no FIXO de `roda-f2d-janela.sh`. O `f2b-extrai-v2.py` e o `f2c-extrai.py` da F2C ficam intocados.

## FIXO
Conferido por `roda-f2d-janela.sh --verifica-fixos` (sem GPU, só lê; código 0 só se todos OK; resultado no momento do congelamento: código 0, todos OK) e de novo no modo real, antes de envelope, monitor, timers ou GPU, sem pipe que mascare o código. O sha do próprio `roda-f2d-janela.sh` não pode estar nele: o verificador o imprime à parte (informativo) e `decisao.txt` o registra. Sha do roteiro no congelamento: `ee3d55cbe52c9b42abc5f59b7c4fb7a87b713ef88ee305b5dad104036fc89672`.

## Ordem das execuções
1. Coleta 1B e depois coleta 3B, com o teste de erro em GPU nos dois modelos (`abre-f2d-coleta-20261112.sh`).
2. Janela de medida 1B e depois 3B (`abre-f2d-20261112.sh`).
Sem a coleta, a condição de saída bit a bit fica SEM PROVA. A medição só deve abrir depois de a coleta dar HC1b dentro e o teste de GPU passar nos dois modelos; se algum kernel não der os mesmos bits, a janela de medida não se abre para esse passo (ver "Regra de desistir").

## Execução de coleta (`abre-f2d-coleta-20261112.sh <worktree-cand>`)
Execução própria, fora da janela de medida (não mede desempenho), com os mesmos portões. Usa `roda-f2d-janela.sh --modelos <m> --saida <pasta> --coleta --teste-gpu --worktree-cand <wt>`: dois processos por modelo (base e candidato), mesmas entradas, `--repete 3`, só `LASTRO_GANCHO_LOGITS=1`, que grava o hash dos logits e do KV por camada (HC1b). Nos DOIS modelos roda também o teste de erro em GPU: `cargo test --release --test motor_reforma_gpu --test motor_passo_gpu --test o3_kernels_gpu -- --ignored --nocapture` com `E4B_GPU_AUTORIZADO=1` (o roteiro define a variável; teto 300 s), com `TMPDIR` numa pasta própria dentro de `target-f2d` do worktree (fora do /tmp, criada por `mktemp -d`, removida ao fim só pelo caminho literal, registrada no manifesto como `teste_gpu_tmpdir`). Grava `teste-gpu.txt`, o rc, o commit e os shas dos três executáveis. O `--worktree-cand` é canonicalizado com `realpath`, recusado se estiver sob /tmp, e só vale se o HEAD é o commit candidato do critério e a árvore está limpa (exceto `target-f2d`).

Pré-compilação do teste de GPU (já feita, sem GPU e sem executar nada), no worktree do commit candidato `~/.cache/lastro-testes/f2d/wt-c1` (HEAD `1fa8abb`, limpo):
```
cd /home/brenoperucchi/.cache/lastro-testes/f2d/wt-c1/rust/moe-rt
CUDA_VISIBLE_DEVICES= CARGO_TARGET_DIR=/home/brenoperucchi/.cache/lastro-testes/f2d/wt-c1/target-f2d \
  cargo test --release --test motor_reforma_gpu --test motor_passo_gpu --test o3_kernels_gpu --no-run
```
Executáveis gerados em `wt-c1/target-f2d/release/deps/`: `motor_reforma_gpu-3d53de78cf94e50f` (sha `a757aad4...`), `motor_passo_gpu-e36750539425749e` (sha `d26b93bd...`), `o3_kernels_gpu-b69107fba3a751db` (sha `4a5f31d2...`). Comando de abertura:
```
bash /home/brenoperucchi/Devs/lastro-rust/.herdr/abre-f2d-coleta-20261112.sh /home/brenoperucchi/.cache/lastro-testes/f2d/wt-c1
```
O script confere que os três executáveis existem ANTES de parar o voxtype; sem eles, recusa com código 2.

## Janela de medida (`abre-f2d-20261112.sh`)
Comando: `bash /home/brenoperucchi/Devs/lastro-rust/.herdr/abre-f2d-20261112.sh`.
- Por modelo, 9 processos no texto médio, série L100, `--repete 3` (1 aquecimento e 3 medidas), em quadrado latino 3x3: D_O1 D_O3 B2_O3, B2_O3 D_O1 D_O3, D_O3 B2_O3 D_O1.
- Braços: D_O1 = binário base, nada ligado; D_O3 = binário candidato, nada ligado; B2_O3 = candidato com `LASTRO_PERFIL_BLOCO`, `LASTRO_PERFIL_F2` e `LASTRO_PERFIL_F2B` ligados (instrumento v3, 15 registros por posição, sem B_DTOD). As variáveis só valem por processo; no ambiente de quem chama o roteiro recusa (código 2).
- Teto de processo 60 s (1B) e 90 s (3B); teto global 1200 s por janela e modelo. O controle cabe inteiro ou os nove viram NÃO RODOU; sem repetição automática. Sem smoke, curto ou longo.
- Limites: prefill de D_O3 contra D_O1 +1,5% (HC3), decode +2% (HC4); resíduo do nível B em B2_O3 até 3% (HC5); B2_O3 contra D_O3 até 3% comprovada, até 10% ressalva, acima SEM PROVA (HC6). Saída idêntica bit a bit (HC1 na janela por ids, HC1b na coleta).
- Aquecimento e JIT: o pedido 0 cobre as cinco classes de divisibilidade de `cur` (critério, seção 5b).

## Scripts de abertura (`.herdr`, fora do git) e o voxtype
Ordem: FIXOS, estado inicial do voxtype (ilegível = recusa), logs de saída livres (id com segundos; log existente nunca é sobrescrito, recusa antes de qualquer parada), na coleta os três executáveis do teste, e só então parar o voxtype e CONFIRMAR `inactive` (falha de stop ou permanência em `active` = roteiro não é chamado). O voxtype é parado e religado pelo script de abertura, com `trap` no EXIT: só é religado se estava ativo e foi parado pelo script, após sucesso ou falha; falha ao religar é avisada e vira código 3 só se a execução principal tinha dado 0. Erro no 1B impede o 3B e conserva o código. O roteiro `roda-f2d-janela.sh` não mexe no voxtype. Testes sem GPU: `.herdr/test-abre-f2d.sh`.

## Autorizações
As quatro execuções nomeadas (coleta de bits nos dois modelos com o teste de erro em GPU nos dois modelos, medição 1B e medição 3B) estão autorizadas pela decisão do Breno de 12/11 às 00:37 (ponto 8) e pela autorização permanente da 3080 Ti, com teto de 1200 s por janela e modelo, voxtype parado e religado, revisão nível 2 e sem push. A abertura é do coordenador, depois da revisão deste pacote.

## O que cada resultado significa
- HC1b dentro e teste de GPU passou nos dois modelos: a O3 é bit a bit igual à base (logits, KV, ids), a execução do caminho de erro deixa o motor utilizável e igual bit a bit, e a medição pode abrir.
- HC1b REGRESSÃO (qualquer bit de logit ou de KV diferente) ou teste de GPU falhou: a O3 não passa; ver a regra abaixo.
- HC1b SEM PROVA (coleção ausente, incompleta, incoerente ou sem procedência, ou teste de GPU com procedência incompleta): nada se conclui, a coleta se repete antes de medir.
- Medição, por métrica (prefill e passo de decode), referência D_O1: "efeito distinguível" (razão abaixo de -1% e as 3 medianas de D_O3 todas abaixo das 3 de D_O1); "sem efeito distinguível" (qualquer outro caso fora da regressão, relatado com a razão e o motivo; o resultado pode ser nulo e nulo não é falha do plano); "REGRESSÃO" (acima de +1,5% no prefill ou +2% no decode, HC6 SEM PROVA, erro de processo); "sem dado" ou SEM PROVA (controle incompleto ou integridade quebrada).
- HC2a e HC2b são descritivas (tabela por camada do H_PRE_LAYER, B_LANCA).

## Regra de desistir do passo
A O3 tem dois passos (K e depois V) com uma só janela de medida ao fim. Sem tolerância de bits (decisão 5 do Breno): se um kernel novo não der os mesmos bits do cache (HC1b com bit diferente, ou o `o3_kernels_gpu` falhando), desiste-se daquele passo e se reporta, sem afrouxar o critério e sem medir a versão com o passo defeituoso. O passo que o causou é identificado pelo teste isolado `o3_kernels_gpu` (K e V são conferidos em separado) e pela ordem K, V do commit.

## O que cada janela grava
- Medida: `decisao.txt`, `ordem.txt`, `prova-variavel.txt`, `plano.txt`, `orcamento.json`, `teto-excedido.txt` (sempre criado; vazio se nenhum teto estourou), `integridade-antes.txt` e `integridade-depois.txt`, `celulas/ctrl-<modelo>-L100-medio-<NN>-<DO1|DO3|B2O3>`, mais `.carga-antes/depois.txt`, `.aviso.txt` e o `.log` ao lado, e os resultados do extrator (`f2d-extrai.out`, `f2d-resultado.json` e `.md`).
- Coleta: `coleta-manifesto.txt` (com `teste_gpu_commit`, `teste_gpu_tmpdir` e os três shas de executável), `integridade-*.txt`, células `coleta-<modelo>-medio-<papel>`, `f2d-coleta-resultado.json` e `.md` (com `teste_gpu`: executado, rc, passou, procedência), `teste-gpu.txt`.

## Testes sem GPU
`test-roda-f2d-janela.sh`, `test-f2d-extrai.py`, `test-f2d-extrai-e2e.py`, `test-f2b-extrai-v3.py`, `.herdr/test-abre-f2d.sh`, e `cargo test --release --lib` filtrado (`perfil_bloco`, `cache_direto`, `leitura_tests`, `gancho_coleta`, `reforma`, `geracao_regra`) com `TMPDIR` em `~/.cache/lastro-testes` e `CUDA_VISIBLE_DEVICES` vazio.
