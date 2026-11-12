# F2C, plano de cópias, janela do modelo 1B (criterio-f2c-20261111.md)

Resultado conforme os limites congelados: prefill +1,5%, decode +2%, B2_novo contra D_novo até 3% comprovada, até 10% com ressalva, acima SEM PROVA. Sem tolerância numérica: qualquer id ou bit diferente é regressão.

## Braços (mediana das medianas dos 3 processos, texto médio)

| braço | prefill (ms) | passo de decode (ms) |
|---|---|---|
| Dvelho | 412.840 | 2.7948 |
| Dnovo | 388.369 | 2.6428 |
| B2novo | 388.052 | 2.6412 |

- ordem efetiva: confere com ordem.txt (quadrado latino, rc 0, n 3)
  - 01 Dvelho ctrl-1b-L100-medio-01-Dvelho: válido True, prefill 412.305 ms, passo 2.7948 ms
  - 02 Dnovo ctrl-1b-L100-medio-02-Dnovo: válido True, prefill 387.491 ms, passo 2.6479 ms
  - 03 B2novo ctrl-1b-L100-medio-03-B2novo: válido True, prefill 387.867 ms, passo 2.6412 ms
  - 04 B2novo ctrl-1b-L100-medio-04-B2novo: válido True, prefill 388.085 ms, passo 2.6485 ms
  - 05 Dvelho ctrl-1b-L100-medio-05-Dvelho: válido True, prefill 413.402 ms, passo 2.7969 ms
  - 06 Dnovo ctrl-1b-L100-medio-06-Dnovo: válido True, prefill 389.029 ms, passo 2.6428 ms
  - 07 Dnovo ctrl-1b-L100-medio-07-Dnovo: válido True, prefill 388.369 ms, passo 2.6409 ms
  - 08 B2novo ctrl-1b-L100-medio-08-B2novo: válido True, prefill 388.052 ms, passo 2.6407 ms
  - 09 Dvelho ctrl-1b-L100-medio-09-Dvelho: válido True, prefill 412.840 ms, passo 2.7890 ms

## Hipóteses

- HC1 (ids iguais aos de D_velho e às referências 5c): **dentro**
- HC1b (hash dos bits dos logits e do KV): **dentro**
- HC3 (prefill D_novo contra D_velho, razão -5.9%; limite +1,5%): **dentro do limite**
- HC4 (decode D_novo contra D_velho, razão -5.4%; limite +2%): **dentro do limite**
- HC6 (B2_novo contra D_novo, prefill, razão -0.1%; decode -0.1%): **comprovada**
- HC5 (resíduo do nível B em B2_novo, até 3%): **ressalva**
- HC2a e HC2b (H_PRE_LAYER por camada de B2_novo, descritivas): **descritivo**
  - soma antes do degrau (camadas 0 a 10): 2662004 ns; soma do patamar (camada 12 em diante): 6782798 ns; transição na camada 11 (fora das duas somas); parcelas B_LANCA 83.3%, B_DTOD 11.4%, B_LANCA + B_DTOD 94.7%

| camada | H_PRE_LAYER mediana (ns) |
|---|---|
| 0 | 258252 |
| 1 | 239382 |
| 2 | 244117 |
| 3 | 237822 |
| 4 | 241276 |
| 5 | 239227 |
| 6 | 241066 |
| 7 | 239252 |
| 8 | 241646 |
| 9 | 240722 |
| 10 | 239242 |
| 11 | 241337 |
| 12 | 287317 |
| 13 | 590654 |
| 14 | 591079 |
| 15 | 590744 |
| 16 | 590959 |
| 17 | 589619 |
| 18 | 590429 |
| 19 | 590714 |
| 20 | 590120 |
| 21 | 590394 |
| 22 | 590160 |
| 23 | 590609 |

## Regressão

**nenhuma**

Sem ganho em prefill_ms mas com H_PRE_LAYER menor não é regressão. A comparação com o B2 de 14:10 é só contexto (outra janela, outro instrumento). Qualquer mudança de desempenho no motor além desta exige nova decisão do Breno.
