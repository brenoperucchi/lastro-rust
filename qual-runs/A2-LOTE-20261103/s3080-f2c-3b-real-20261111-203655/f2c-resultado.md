# F2C, plano de cópias, janela do modelo 3B (criterio-f2c-20261111.md)

Resultado conforme os limites congelados: prefill +1,5%, decode +2%, B2_novo contra D_novo até 3% comprovada, até 10% com ressalva, acima SEM PROVA. Sem tolerância numérica: qualquer id ou bit diferente é regressão.

## Braços (mediana das medianas dos 3 processos, texto médio)

| braço | prefill (ms) | passo de decode (ms) |
|---|---|---|
| Dvelho | 696.226 | 4.5854 |
| Dnovo | 663.081 | 4.3867 |
| B2novo | 663.103 | 4.3830 |

- ordem efetiva: confere com ordem.txt (quadrado latino, rc 0, n 3)
  - 01 Dvelho ctrl-3b-L100-medio-01-Dvelho: válido True, prefill 696.627 ms, passo 4.5907 ms
  - 02 Dnovo ctrl-3b-L100-medio-02-Dnovo: válido True, prefill 663.081 ms, passo 4.3844 ms
  - 03 B2novo ctrl-3b-L100-medio-03-B2novo: válido True, prefill 663.136 ms, passo 4.3903 ms
  - 04 B2novo ctrl-3b-L100-medio-04-B2novo: válido True, prefill 663.103 ms, passo 4.3830 ms
  - 05 Dvelho ctrl-3b-L100-medio-05-Dvelho: válido True, prefill 696.226 ms, passo 4.5854 ms
  - 06 Dnovo ctrl-3b-L100-medio-06-Dnovo: válido True, prefill 665.109 ms, passo 4.3992 ms
  - 07 Dnovo ctrl-3b-L100-medio-07-Dnovo: válido True, prefill 663.070 ms, passo 4.3867 ms
  - 08 B2novo ctrl-3b-L100-medio-08-B2novo: válido True, prefill 662.976 ms, passo 4.3795 ms
  - 09 Dvelho ctrl-3b-L100-medio-09-Dvelho: válido True, prefill 696.184 ms, passo 4.5802 ms

## Hipóteses

- HC1 (ids iguais aos de D_velho e às referências 5c): **dentro**
- HC1b (hash dos bits dos logits e do KV): **SEM PROVA** (pasta da coleta não fornecida (--coleta): a coleta é uma execução própria)
- HC3 (prefill D_novo contra D_velho, razão -4.8%; limite +1,5%): **dentro do limite**
- HC4 (decode D_novo contra D_velho, razão -4.3%; limite +2%): **dentro do limite**
- HC6 (B2_novo contra D_novo, prefill, razão 0.0%; decode -0.1%): **comprovada**
- HC5 (resíduo do nível B em B2_novo, até 3%): **ressalva**
- HC2a e HC2b (H_PRE_LAYER por camada de B2_novo, descritivas): **descritivo**
  - soma antes do degrau (camadas 0 a 8): 2234415 ns; soma do patamar (camada 10 em diante): 16368123 ns; transição na camada 9 (fora das duas somas); parcelas B_LANCA 86.2%, B_DTOD 10.0%, B_LANCA + B_DTOD 96.2%

| camada | H_PRE_LAYER mediana (ns) |
|---|---|
| 0 | 259262 |
| 1 | 248546 |
| 2 | 245707 |
| 3 | 249216 |
| 4 | 244432 |
| 5 | 246966 |
| 6 | 247876 |
| 7 | 244202 |
| 8 | 248206 |
| 9 | 243637 |
| 10 | 246482 |
| 11 | 766520 |
| 12 | 767830 |
| 13 | 767296 |
| 14 | 767866 |
| 15 | 767036 |
| 16 | 767316 |
| 17 | 767456 |
| 18 | 766850 |
| 19 | 767370 |
| 20 | 768520 |
| 21 | 768280 |
| 22 | 768901 |
| 23 | 768341 |
| 24 | 768176 |
| 25 | 768386 |
| 26 | 767440 |
| 27 | 767370 |
| 28 | 767360 |
| 29 | 767796 |
| 30 | 767646 |
| 31 | 767886 |

## Regressão

**nenhuma**

Sem ganho em prefill_ms mas com H_PRE_LAYER menor não é regressão. A comparação com o B2 de 14:10 é só contexto (outra janela, outro instrumento). Qualquer mudança de desempenho no motor além desta exige nova decisão do Breno.
