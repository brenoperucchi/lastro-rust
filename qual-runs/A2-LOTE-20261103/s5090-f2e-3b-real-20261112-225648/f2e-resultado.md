# F2E, plano da O3 (cache K/V direto), janela do modelo 3B (criterio-f2e-20261112.md)

Resultado conforme os limites congelados: prefill +1,5%, decode +2%, efeito distinguível só abaixo de -1% com as 3 medianas abaixo, B2_O3 contra D_O3 até 3% comprovada, até 10% com ressalva, acima SEM PROVA. Sem tolerância numérica: qualquer id ou bit diferente é regressão.

## Braços (mediana das medianas dos 3 processos, texto médio)

| braço | prefill (ms) | passo de decode (ms) |
|---|---|---|
| DO1 | 523.772 | 4.0645 |
| DO3 | 525.302 | 4.0619 |
| B2O3 | 525.399 | 4.0624 |

- ordem efetiva: confere com ordem.txt (quadrado latino, rc 0, n 3)
  - 01 DO1 ctrl-3b-L100-medio-01-DO1: válido True, prefill 523.829 ms, passo 4.0645 ms
  - 02 DO3 ctrl-3b-L100-medio-02-DO3: válido True, prefill 525.315 ms, passo 4.0597 ms
  - 03 B2O3 ctrl-3b-L100-medio-03-B2O3: válido True, prefill 525.399 ms, passo 4.0624 ms
  - 04 B2O3 ctrl-3b-L100-medio-04-B2O3: válido True, prefill 525.388 ms, passo 4.0617 ms
  - 05 DO1 ctrl-3b-L100-medio-05-DO1: válido True, prefill 523.756 ms, passo 4.0629 ms
  - 06 DO3 ctrl-3b-L100-medio-06-DO3: válido True, prefill 525.291 ms, passo 4.0619 ms
  - 07 DO3 ctrl-3b-L100-medio-07-DO3: válido True, prefill 525.302 ms, passo 4.0664 ms
  - 08 B2O3 ctrl-3b-L100-medio-08-B2O3: válido True, prefill 525.420 ms, passo 4.0644 ms
  - 09 DO1 ctrl-3b-L100-medio-09-DO1: válido True, prefill 523.772 ms, passo 4.0657 ms

## Hipóteses

- HC1 (ids iguais aos de D_O1 e às referências 5c): **dentro**
- HC1b (hash dos bits dos logits e do KV): **dentro**
- HC3 (prefill D_O3 contra D_O1, razão 0.3%; limite +1,5%): **dentro do limite**
- HC4 (decode D_O3 contra D_O1, razão -0.1%; limite +2%): **dentro do limite**
- HC6 (B2_O3 contra D_O3, prefill, razão 0.0%; decode 0.0%): **comprovada**
- efeito no prefill (D_O3 contra D_O1): **sem efeito distinguível** (razão +0.29% dentro de -1% e do limite de regressão)
- efeito no decode (D_O3 contra D_O1): **sem efeito distinguível** (razão -0.06% dentro de -1% e do limite de regressão)
- HC5 (resíduo do nível B em B2_O3, até 3%): **ressalva**
- HC2a e HC2b (H_PRE_LAYER por camada de B2_O3, descritivas): **descritivo**
  - soma antes do degrau (camadas 0 a 10): 1902674 ns; soma do patamar (camada 11 em diante): 12166912 ns; sem camada de transição; parcelas B_LANCA 96.6%, B_DTOD 0.0% (zero por construção), B_LANCA + B_DTOD 96.6%

| camada | H_PRE_LAYER mediana (ns) |
|---|---|
| 0 | 178360 |
| 1 | 173440 |
| 2 | 170815 |
| 3 | 173136 |
| 4 | 172772 |
| 5 | 170511 |
| 6 | 173277 |
| 7 | 173450 |
| 8 | 170760 |
| 9 | 172974 |
| 10 | 173180 |
| 11 | 434614 |
| 12 | 596008 |
| 13 | 586580 |
| 14 | 596318 |
| 15 | 596992 |
| 16 | 597184 |
| 17 | 586448 |
| 18 | 584059 |
| 19 | 584564 |
| 20 | 588778 |
| 21 | 585302 |
| 22 | 580476 |
| 23 | 578680 |
| 24 | 579900 |
| 25 | 582251 |
| 26 | 581398 |
| 27 | 583250 |
| 28 | 584235 |
| 29 | 587390 |
| 30 | 582034 |
| 31 | 590454 |

## Regressão

**nenhuma**

Sem ganho em prefill_ms mas com H_PRE_LAYER menor não é regressão. A comparação com o B2 de 14:10 é só contexto (outra janela, outro instrumento). Qualquer mudança de desempenho no motor além desta exige nova decisão do Breno.

Procedência da execução: clone no commit 0a7d121d261521c66e5a08bf25097ef29c05d08f (árvore limpa: sim); roteiro sha256 cbc1456f597a9d7a5c3940d0011b4697e2fda189ea919a3ebd0a44e720793acd; envelope janela-gpu-strata sha256 3e85f052327e20b78c0897f3b9bad46d358c87d8d788dd186a8502eb6e5a96d4
