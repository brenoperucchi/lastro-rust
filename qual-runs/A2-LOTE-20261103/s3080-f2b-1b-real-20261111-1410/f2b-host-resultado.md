# F2 nível B, pré-roteador, janela do modelo 1B (criterio-f2-nivel-b-20261111.md)

Descritivo: sem veredito geral e sem PASSA. HB1 a HB6 são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA', 'descritivo'); B_LANCA é o tempo da chamada de envio de trabalho; o tempo com instrumento não é o tempo de produção.

Smoke (1 processo no curto, em B2): **OK**

## Controle do custo do instrumento (D, A, B2)

Classe (|B2 contra A|): **comprovada**
- ordem efetiva (ordem.txt): confere com ordem.txt (D A B2, B2 D A, A B2 D, rc 0)
- B2 contra A (principal): desvio de prefill_ms 0.3%; pior caso 0.6%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.065 ms (22 blocos); desvio do passo de decode 0.2%
- B2 contra D (apoio): desvio de prefill_ms 0.2%; pior caso 0.7%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.030 ms (22 blocos); desvio do passo de decode 0.2%
- classe descritiva de B2 contra D: comprovada (não decide)
  - 01 D ctrl-1b-L100-medio-01-D: temporal OK, bloco desligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b não/não/não, válido True, prefill mediano 412.59 ms, passo 2.78 ms
  - 02 A ctrl-1b-L100-medio-02-A: temporal OK, bloco ligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b sim/não/não, válido True, prefill mediano 414.47 ms, passo 2.79 ms
  - 03 B2 ctrl-1b-L100-medio-03-B2: temporal OK, bloco ligado, f2 ligado, f2b ligado, arquivos bloco/f2/f2b sim/sim/sim, válido True, prefill mediano 412.86 ms, passo 2.80 ms
  - 04 B2 ctrl-1b-L100-medio-04-B2: temporal OK, bloco ligado, f2 ligado, f2b ligado, arquivos bloco/f2/f2b sim/sim/sim, válido True, prefill mediano 412.96 ms, passo 2.79 ms
  - 05 D ctrl-1b-L100-medio-05-D: temporal OK, bloco desligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b não/não/não, válido True, prefill mediano 410.23 ms, passo 2.78 ms
  - 06 A ctrl-1b-L100-medio-06-A: temporal OK, bloco ligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b sim/não/não, válido True, prefill mediano 411.44 ms, passo 2.78 ms
  - 07 A ctrl-1b-L100-medio-07-A: temporal OK, bloco ligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b sim/não/não, válido True, prefill mediano 410.61 ms, passo 2.78 ms
  - 08 B2 ctrl-1b-L100-medio-08-B2: temporal OK, bloco ligado, f2 ligado, f2b ligado, arquivos bloco/f2/f2b sim/sim/sim, válido True, prefill mediano 411.94 ms, passo 2.78 ms
  - 09 D ctrl-1b-L100-medio-09-D: temporal OK, bloco desligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b não/não/não, válido True, prefill mediano 412.21 ms, passo 2.78 ms
- ids de B2 iguais aos de D e de A (I7, processos do controle): sim

## Texto curto (resto 1)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 50; células: lastro-1b-L100-curto-quente
- parcelas de H_PRE_LAYER (mediana por camada somada): B_LOCK 0.0% (mede o custo de pegar uma trava livre); B_SETUP + B_GENERICS + B_ATTN_PREP 0.7%; B_LANCA 58.5% (tempo da chamada de envio de trabalho (a expressão async_on)); B_DTOD 39.5%
- uniformidade: p90/mediana por camada 1.033; maior/menor mediana entre camadas 2.287

| HB1 | HB2 | HB3 | HB4 | HB5 | HB6 |
|---|---|---|---|---|---|
| descritivo | descritivo | sem leitura | descritivo | descritivo | descritivo |

## Texto medio (resto 2)

status **OK**; pedidos válidos 9 (mínimo 9); amostras elegíveis 180; células: ctrl-1b-L100-medio-03-B2, ctrl-1b-L100-medio-04-B2, ctrl-1b-L100-medio-08-B2
- parcelas de H_PRE_LAYER (mediana por camada somada): B_LOCK 0.0% (mede o custo de pegar uma trava livre); B_SETUP + B_GENERICS + B_ATTN_PREP 0.7%; B_LANCA 58.5% (tempo da chamada de envio de trabalho (a expressão async_on)); B_DTOD 39.5%
- uniformidade: p90/mediana por camada 1.014; maior/menor mediana entre camadas 2.323

| HB1 | HB2 | HB3 | HB4 | HB5 | HB6 |
|---|---|---|---|---|---|
| descritivo | descritivo | sem leitura | descritivo | descritivo | descritivo |

## Texto longo (resto 4)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 220; células: lastro-1b-L100-longo-quente
- parcelas de H_PRE_LAYER (mediana por camada somada): B_LOCK 0.0% (mede o custo de pegar uma trava livre); B_SETUP + B_GENERICS + B_ATTN_PREP 0.7%; B_LANCA 58.7% (tempo da chamada de envio de trabalho (a expressão async_on)); B_DTOD 39.3%
- uniformidade: p90/mediana por camada 1.017; maior/menor mediana entre camadas 2.345

| HB1 | HB2 | HB3 | HB4 | HB5 | HB6 |
|---|---|---|---|---|---|
| descritivo | descritivo | sem leitura | descritivo | descritivo | descritivo |

## H_PRE_LAYER por camada: B2 ao lado do B de 11:00 (entre janelas)

razão mediana B2/B entre as camadas: 0.999 (janelas diferentes; o B2 já inclui o custo do instrumento novo)

| camada | B2 (ns) | B de 11:00 (ns) |
|---|---|---|
| 0 | 315917 | 308652 |
| 1 | 286387 | 276937 |
| 2 | 291972 | 282247 |
| 3 | 285188 | 275647 |
| 4 | 291012 | 281232 |
| 5 | 285067 | 276322 |
| 6 | 290188 | 280102 |
| 7 | 285472 | 275432 |
| 8 | 290952 | 282822 |
| 9 | 285812 | 276137 |
| 10 | 291202 | 281072 |
| 11 | 475394 | 554629 |
| 12 | 657850 | 658510 |
| 13 | 657025 | 658770 |
| 14 | 657030 | 658210 |
| 15 | 657380 | 657895 |
| 16 | 656940 | 657855 |
| 17 | 656585 | 658044 |
| 18 | 656410 | 656875 |
| 19 | 657370 | 658320 |
| 20 | 658050 | 659200 |
| 21 | 656975 | 658655 |
| 22 | 656580 | 658140 |
| 23 | 657640 | 658670 |

## Orçamento

decorrido 204 s de 1200 s; cortes: nenhum; não rodou: nada; teto de processo excedido: nenhum

## Conclusão

Este relatório é descritivo e não declara veredito; qualquer mudança de desempenho no motor exige nova decisão do Breno.
