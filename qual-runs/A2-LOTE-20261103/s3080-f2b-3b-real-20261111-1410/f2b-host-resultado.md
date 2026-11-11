# F2 nível B, pré-roteador, janela do modelo 3B (criterio-f2-nivel-b-20261111.md)

Descritivo: sem veredito geral e sem PASSA. HB1 a HB6 são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA', 'descritivo'); B_LANCA é o tempo da chamada de envio de trabalho; o tempo com instrumento não é o tempo de produção.

Smoke (1 processo no curto, em B2): **OK**

## Controle do custo do instrumento (D, A, B2)

Classe (|B2 contra A|): **comprovada**
- ordem efetiva (ordem.txt): confere com ordem.txt (D A B2, B2 D A, A B2 D, rc 0)
- B2 contra A (principal): desvio de prefill_ms 0.3%; pior caso 1.1%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.100 ms (22 blocos); desvio do passo de decode 0.1%
- B2 contra D (apoio): desvio de prefill_ms 0.4%; pior caso 1.1%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.113 ms (22 blocos); desvio do passo de decode -0.1%
- classe descritiva de B2 contra D: comprovada (não decide)
  - 01 D ctrl-3b-L100-medio-01-D: temporal OK, bloco desligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b não/não/não, válido True, prefill mediano 696.07 ms, passo 4.58 ms
  - 02 A ctrl-3b-L100-medio-02-A: temporal OK, bloco ligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b sim/não/não, válido True, prefill mediano 695.73 ms, passo 4.58 ms
  - 03 B2 ctrl-3b-L100-medio-03-B2: temporal OK, bloco ligado, f2 ligado, f2b ligado, arquivos bloco/f2/f2b sim/sim/sim, válido True, prefill mediano 698.55 ms, passo 4.59 ms
  - 04 B2 ctrl-3b-L100-medio-04-B2: temporal OK, bloco ligado, f2 ligado, f2b ligado, arquivos bloco/f2/f2b sim/sim/sim, válido True, prefill mediano 703.40 ms, passo 4.71 ms
  - 05 D ctrl-3b-L100-medio-05-D: temporal OK, bloco desligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b não/não/não, válido True, prefill mediano 697.19 ms, passo 4.60 ms
  - 06 A ctrl-3b-L100-medio-06-A: temporal OK, bloco ligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b sim/não/não, válido True, prefill mediano 696.36 ms, passo 4.59 ms
  - 07 A ctrl-3b-L100-medio-07-A: temporal OK, bloco ligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b sim/não/não, válido True, prefill mediano 697.89 ms, passo 4.60 ms
  - 08 B2 ctrl-3b-L100-medio-08-B2: temporal OK, bloco ligado, f2 ligado, f2b ligado, arquivos bloco/f2/f2b sim/sim/sim, válido True, prefill mediano 696.08 ms, passo 4.58 ms
  - 09 D ctrl-3b-L100-medio-09-D: temporal OK, bloco desligado, f2 desligado, f2b desligado, arquivos bloco/f2/f2b não/não/não, válido True, prefill mediano 695.96 ms, passo 4.63 ms
- ids de B2 iguais aos de D e de A (I7, processos do controle): sim

## Texto curto (resto 1)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 50; células: lastro-3b-L100-curto-quente
- parcelas de H_PRE_LAYER (mediana por camada somada): B_LOCK 0.0% (mede o custo de pegar uma trava livre); B_SETUP + B_GENERICS + B_ATTN_PREP 0.5%; B_LANCA 58.1% (tempo da chamada de envio de trabalho (a expressão async_on)); B_DTOD 40.4%
- uniformidade: p90/mediana por camada 1.003; maior/menor mediana entre camadas 2.863

| HB1 | HB2 | HB3 | HB4 | HB5 | HB6 |
|---|---|---|---|---|---|
| descritivo | descritivo | sem leitura | descritivo | descritivo | descritivo |

## Texto medio (resto 2)

status **OK**; pedidos válidos 9 (mínimo 9); amostras elegíveis 180; células: ctrl-3b-L100-medio-03-B2, ctrl-3b-L100-medio-04-B2, ctrl-3b-L100-medio-08-B2
- parcelas de H_PRE_LAYER (mediana por camada somada): B_LOCK 0.0% (mede o custo de pegar uma trava livre); B_SETUP + B_GENERICS + B_ATTN_PREP 0.6%; B_LANCA 58.2% (tempo da chamada de envio de trabalho (a expressão async_on)); B_DTOD 40.3%
- uniformidade: p90/mediana por camada 1.005; maior/menor mediana entre camadas 2.861

| HB1 | HB2 | HB3 | HB4 | HB5 | HB6 |
|---|---|---|---|---|---|
| descritivo | descritivo | sem leitura | descritivo | descritivo | descritivo |

## Texto longo (resto 4)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 220; células: lastro-3b-L100-longo-quente
- parcelas de H_PRE_LAYER (mediana por camada somada): B_LOCK 0.0% (mede o custo de pegar uma trava livre); B_SETUP + B_GENERICS + B_ATTN_PREP 0.5%; B_LANCA 58.2% (tempo da chamada de envio de trabalho (a expressão async_on)); B_DTOD 40.4%
- uniformidade: p90/mediana por camada 1.007; maior/menor mediana entre camadas 2.871

| HB1 | HB2 | HB3 | HB4 | HB5 | HB6 |
|---|---|---|---|---|---|
| descritivo | descritivo | sem leitura | descritivo | descritivo | descritivo |

## H_PRE_LAYER por camada: B2 ao lado do B de 11:00 (entre janelas)

razão mediana B2/B entre as camadas: 0.998 (janelas diferentes; o B2 já inclui o custo do instrumento novo)

| camada | B2 (ns) | B de 11:00 (ns) |
|---|---|---|
| 0 | 324442 | 308362 |
| 1 | 303812 | 291348 |
| 2 | 304287 | 289982 |
| 3 | 303177 | 291178 |
| 4 | 300162 | 285377 |
| 5 | 303182 | 291402 |
| 6 | 299172 | 286447 |
| 7 | 304262 | 288552 |
| 8 | 301532 | 289947 |
| 9 | 377383 | 427684 |
| 10 | 840626 | 845612 |
| 11 | 843836 | 846286 |
| 12 | 844252 | 846646 |
| 13 | 844027 | 846192 |
| 14 | 844307 | 846607 |
| 15 | 844446 | 846832 |
| 16 | 844897 | 846486 |
| 17 | 845731 | 846717 |
| 18 | 845346 | 846797 |
| 19 | 844326 | 846236 |
| 20 | 843756 | 846126 |
| 21 | 844376 | 845646 |
| 22 | 844722 | 845837 |
| 23 | 844672 | 845821 |
| 24 | 844967 | 846341 |
| 25 | 843522 | 845847 |
| 26 | 844377 | 846222 |
| 27 | 844117 | 846166 |
| 28 | 844452 | 845562 |
| 29 | 843806 | 845736 |
| 30 | 844001 | 845886 |
| 31 | 843916 | 845437 |

## Orçamento

decorrido 322 s de 1200 s; cortes: nenhum; não rodou: nada; teto de processo excedido: nenhum

## Conclusão

Este relatório é descritivo e não declara veredito; qualquer mudança de desempenho no motor exige nova decisão do Breno.
