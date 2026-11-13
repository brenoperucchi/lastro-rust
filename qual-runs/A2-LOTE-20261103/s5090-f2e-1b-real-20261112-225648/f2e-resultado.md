# F2E, plano da O3 (cache K/V direto), janela do modelo 1B (criterio-f2e-20261112.md)

Resultado conforme os limites congelados: prefill +1,5%, decode +2%, efeito distinguível só abaixo de -1% com as 3 medianas abaixo, B2_O3 contra D_O3 até 3% comprovada, até 10% com ressalva, acima SEM PROVA. Sem tolerância numérica: qualquer id ou bit diferente é regressão.

## Braços (mediana das medianas dos 3 processos, texto médio)

| braço | prefill (ms) | passo de decode (ms) |
|---|---|---|
| DO1 | 311.876 | 2.5782 |
| DO3 | 312.282 | 2.5691 |
| B2O3 | 312.418 | 2.5630 |

- ordem efetiva: confere com ordem.txt (quadrado latino, rc 0, n 3)
  - 01 DO1 ctrl-1b-L100-medio-01-DO1: válido True, prefill 311.876 ms, passo 2.5786 ms
  - 02 DO3 ctrl-1b-L100-medio-02-DO3: válido True, prefill 312.282 ms, passo 2.5667 ms
  - 03 B2O3 ctrl-1b-L100-medio-03-B2O3: válido True, prefill 312.508 ms, passo 2.5662 ms
  - 04 B2O3 ctrl-1b-L100-medio-04-B2O3: válido True, prefill 312.417 ms, passo 2.5630 ms
  - 05 DO1 ctrl-1b-L100-medio-05-DO1: válido True, prefill 311.990 ms, passo 2.5769 ms
  - 06 DO3 ctrl-1b-L100-medio-06-DO3: válido True, prefill 312.394 ms, passo 2.5691 ms
  - 07 DO3 ctrl-1b-L100-medio-07-DO3: válido True, prefill 312.265 ms, passo 2.5700 ms
  - 08 B2O3 ctrl-1b-L100-medio-08-B2O3: válido True, prefill 312.418 ms, passo 2.5619 ms
  - 09 DO1 ctrl-1b-L100-medio-09-DO1: válido True, prefill 311.876 ms, passo 2.5782 ms

## Hipóteses

- HC1 (ids iguais aos de D_O1 e às referências 5c): **dentro**
- HC1b (hash dos bits dos logits e do KV): **dentro**
- HC3 (prefill D_O3 contra D_O1, razão 0.1%; limite +1,5%): **dentro do limite**
- HC4 (decode D_O3 contra D_O1, razão -0.4%; limite +2%): **dentro do limite**
- HC6 (B2_O3 contra D_O3, prefill, razão 0.0%; decode -0.2%): **comprovada**
- efeito no prefill (D_O3 contra D_O1): **sem efeito distinguível** (razão +0.13% dentro de -1% e do limite de regressão)
- efeito no decode (D_O3 contra D_O1): **sem efeito distinguível** (razão -0.35% dentro de -1% e do limite de regressão)
- HC5 (resíduo do nível B em B2_O3, até 3%): **ressalva**
- HC2a e HC2b (H_PRE_LAYER por camada de B2_O3, descritivas): **descritivo**
  - soma antes do degrau (camadas 0 a 11): 2044264 ns; soma do patamar (camada 13 em diante): 5060998 ns; transição na camada 12 (fora das duas somas); parcelas B_LANCA 95.2%, B_DTOD 0.0% (zero por construção), B_LANCA + B_DTOD 95.2%

| camada | H_PRE_LAYER mediana (ns) |
|---|---|
| 0 | 178654 |
| 1 | 169100 |
| 2 | 169371 |
| 3 | 170884 |
| 4 | 167940 |
| 5 | 171166 |
| 6 | 169010 |
| 7 | 169520 |
| 8 | 171051 |
| 9 | 168013 |
| 10 | 171111 |
| 11 | 168442 |
| 12 | 200614 |
| 13 | 462733 |
| 14 | 461672 |
| 15 | 461026 |
| 16 | 457374 |
| 17 | 460718 |
| 18 | 459240 |
| 19 | 458988 |
| 20 | 459416 |
| 21 | 457638 |
| 22 | 460545 |
| 23 | 461648 |

## Regressão

**nenhuma**

Sem ganho em prefill_ms mas com H_PRE_LAYER menor não é regressão. A comparação com o B2 de 14:10 é só contexto (outra janela, outro instrumento). Qualquer mudança de desempenho no motor além desta exige nova decisão do Breno.

Procedência da execução: clone no commit 0a7d121d261521c66e5a08bf25097ef29c05d08f (árvore limpa: sim); roteiro sha256 cbc1456f597a9d7a5c3940d0011b4697e2fda189ea919a3ebd0a44e720793acd; envelope janela-gpu-strata sha256 3e85f052327e20b78c0897f3b9bad46d358c87d8d788dd186a8502eb6e5a96d4
