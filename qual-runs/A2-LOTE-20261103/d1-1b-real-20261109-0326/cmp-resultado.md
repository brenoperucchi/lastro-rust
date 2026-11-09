# Degrau 1: Lastro com 100% dos experts na GPU contra llama.cpp S1 (resultado descritivo)

Critério: criterio-degrau1-100pct-20261109.md (herda criterio-comparacao-llamacpp-20261108.md, emendas 1 a 9). Sem meta e sem PASSA/REPROVA de velocidade. **Todo rótulo: Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp.** Razão = mediana L100 / mediana S1 (menor que 1 = Lastro mais rápido); prefill e decode ficam SEPARADOS e não são juntados. Dispersão > 15% = ruidosa; SEM PROVA e NÃO RODOU como no critério base (nunca viram razão).

A coluna 'Lastro 10%' da janela anterior (09/11) só entra como referência declarada; nenhuma razão nova a mistura. O S1 foi remedido NESTA janela.

## Smoke (seção 3)
- veredito: OK
- gpu_pico_pct: 20.4
- gpu_leituras_no_intervalo: 18
- cpu_needed: 0.0
- ids: IGUAIS
- slots_impressos: slots 768 de 768
- cmdline: /home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/target/release/primeira-palavra --gera --celula 1b:/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct --so-texto curto --fracao-slots 1.0 --saida /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/d1-1b-real-20261109-0326/smoke/p1/saida

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 1b (uma janela por modelo).

## Orçamento e cortes
Cortes aplicados: nenhum
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    smoke inicio=4s fim=13s rc=0
    01 lastro-1b-L100-curto-frio n=3 inicio=13s fim=54s rc=0
    02 llamacpp-1b-S1-curto-frio n=3 inicio=54s fim=65s rc=0
    03 lastro-1b-L100-curto-quente n=5 inicio=65s fim=81s rc=0
    04 llamacpp-1b-S1-curto-quente n=5 inicio=81s fim=88s rc=0
    05 llamacpp-1b-S1-medio-frio n=3 inicio=88s fim=99s rc=0
    06 lastro-1b-L100-medio-frio n=3 inicio=99s fim=140s rc=0
    07 llamacpp-1b-S1-medio-quente n=5 inicio=140s fim=148s rc=0
    08 lastro-1b-L100-medio-quente n=5 inicio=148s fim=164s rc=0
    09 lastro-1b-L100-longo-frio n=3 inicio=164s fim=207s rc=0
    10 llamacpp-1b-S1-longo-frio n=3 inicio=207s fim=218s rc=0
    11 lastro-1b-L100-longo-quente n=5 inicio=218s fim=236s rc=0
    12 llamacpp-1b-S1-longo-quente n=5 inicio=236s fim=244s rc=0

## Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (L100) | L100 prefill ms | S1 prefill ms | L100 decode ms/passo | S1 decode ms/passo | faixa (seção 5, decode) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 27.122 | 1.333 | mesma sequência | 0.000 (0.000 a 0.000) | 2676.44 (2675.22 a 2684.66) | 98.68 (98.24 a 101.54) | 2.393 (2.366 a 2.422) | 1.795 (1.791 a 1.813) | decode L100/S1 <= 2 |
| 1b | curto | quente | 5 | 20.339 | 1.353 | mesma sequência | 0.000 (0.000 a 0.000) | 140.71 (139.28 a 142.15) | 6.92 (6.85 a 6.94) | 2.360 (2.336 a 2.364) | 1.745 (1.742 a 1.787) | decode L100/S1 <= 2 |
| 1b | medio | frio | 3 | 27.867 | 1.314 | mesma sequência | 0.000 (0.000 a 0.000) | 2800.64 (2798.97 a 2830.81) | 100.50 (99.14 a 101.49) | 2.363 (2.345 a 2.392) | 1.797 (1.794 a 1.837) | decode L100/S1 <= 2 |
| 1b | medio | quente | 5 | 31.046 | 1.360 | mesma sequência | 0.000 (0.000 a 0.000) | 268.95 (267.70 a 271.77) | 8.66 (8.64 a 8.82) | 2.378 (2.350 a 2.455) | 1.749 (1.743 a 1.822) | decode L100/S1 <= 2 |
| 1b | longo | frio | 3 | 29.435 | 1.303 (sequências diferentes) | sequências diferentes | 0.000 (0.000 a 0.000) | 3104.93 (3094.73 a 3115.13) | 105.49 (105.09 a 105.71) | 2.371 (2.347 a 2.440) | 1.820 (1.797 a 1.823) | decode L100/S1 <= 2 |
| 1b | longo | quente | 5 | 44.831 | 1.340 (sequências diferentes) | sequências diferentes | 0.000 (0.000 a 0.000) | 574.69 (570.09 a 577.86) | 12.82 (12.70 a 13.01) | 2.362 (2.353 a 2.396) | 1.764 (1.754 a 1.814) | decode L100/S1 <= 2 |

## Leitura (seção 5: palpites pré-registrados, orientação e NÃO decisão)
- decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2
- decode L100/S1 entre 2 e 6: há overhead de motor moderado (KV f32, sincronização, lançamentos); investigar por perfil
- decode L100/S1 > 6: o motor tem custo próprio grande além do posicionamento; investigar antes de qualquer outro degrau
- Prefill: a razão será provavelmente maior que a do decode; o relatório separa as duas e não as junta.

- 1b curto frio: razão de decode 1.333 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 1b curto quente: razão de decode 1.353 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 1b medio frio: razão de decode 1.314 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 1b medio quente: razão de decode 1.360 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 1b longo frio: razão de decode 1.303 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 1b longo quente: razão de decode 1.340 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'

Limites declarados (seção 6): KV f32 contra f16; fronteira do prefill diferente (base); desktop gráfico como condição comum; memória do host declarada à parte; um Granite BF16 só; sem Strata; não responde sobre outros modelos ou quantizados.

## Razões de prefill e decode que diferem em mais de 1,5x (hipótese, nunca conclusão)
- 1b curto frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 47.5%; soma_passos_ms 2.7%; T_sequencia_ms 49.9% CANDIDATA A GARGALO
- 1b curto quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 48.6%; soma_passos_ms 51.4%; T_sequencia_ms 0.0%
- 1b medio frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 48.7%; soma_passos_ms 2.6%; T_sequencia_ms 48.7% CANDIDATA A GARGALO
- 1b medio quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 64.2%; soma_passos_ms 35.8%; T_sequencia_ms 0.0%
- 1b longo frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 51.2%; soma_passos_ms 2.5%; T_sequencia_ms 46.3% CANDIDATA A GARGALO
- 1b longo quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 79.5%; soma_passos_ms 20.5%; T_sequencia_ms 0.0%

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-1b-L100-curto-frio | 6089 (6088 a 6141) (LIMITE SUPERIOR) | 417.9 |
| llamacpp-1b-S1-curto-frio | 843 (841 a 844) (frio sem aquecimento) | 557.1 |
| lastro-1b-L100-curto-quente | 6121 (6121 a 6121) (LIMITE SUPERIOR) | 423.7 |
| llamacpp-1b-S1-curto-quente | 842 (842 a 842) (inclui aquecimento padrão) | 573.2 |
| lastro-1b-L100-medio-frio | 6078 (5608 a 6089) (LIMITE SUPERIOR) | 423.3 |
| llamacpp-1b-S1-medio-frio | 841 (840 a 846) (frio sem aquecimento) | 556.3 |
| lastro-1b-L100-medio-quente | 5861 (5861 a 5861) (LIMITE SUPERIOR) | 420.4 |
| llamacpp-1b-S1-medio-quente | 842 (842 a 842) (inclui aquecimento padrão) | 571.9 |
| lastro-1b-L100-longo-frio | 6060 (6024 a 6163) (LIMITE SUPERIOR) | 421.8 |
| llamacpp-1b-S1-longo-frio | 840 (839 a 845) (frio sem aquecimento) | 549.5 |
| lastro-1b-L100-longo-quente | 6108 (6108 a 6108) (LIMITE SUPERIOR) | 423.3 |
| llamacpp-1b-S1-longo-quente | 845 (845 a 845) (inclui aquecimento padrão) | 567.0 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs G) é condição comum dos dois lados; processo de compute (C ou C+G) fora da árvore medida invalida a célula. Pico do pmon é amostra esparsa, não fração da placa.

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-1b-L100-curto-frio | 0 | 8 | 0 |
| llamacpp-1b-S1-curto-frio | 0 | 15 | 13 |
| lastro-1b-L100-curto-quente | 0 | 7 | 19 |
| llamacpp-1b-S1-curto-quente | 0 | 43 | 14 |
| lastro-1b-L100-medio-frio | 0 | 0 | 11 |
| llamacpp-1b-S1-medio-frio | 0 | 31 | 4 |
| lastro-1b-L100-medio-quente | 0 | 7 | 4 |
| llamacpp-1b-S1-medio-quente | 0 | 0 | 24 |
| lastro-1b-L100-longo-frio | 0 | 8 | 15 |
| llamacpp-1b-S1-longo-frio | 0 | 24 | 0 |
| lastro-1b-L100-longo-quente | 0 | 7 | 1 |
| llamacpp-1b-S1-longo-quente | 1 | 4 | 27 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## `slots` impressos pelo binário (L100)
- lastro-1b-L100-curto-frio: p0: slots 768 de 768 | p1: slots 768 de 768 | p2: slots 768 de 768
- lastro-1b-L100-curto-quente: p1: slots 768 de 768
- lastro-1b-L100-medio-frio: p0: slots 768 de 768 | p1: slots 768 de 768 | p2: slots 768 de 768
- lastro-1b-L100-medio-quente: p1: slots 768 de 768
- lastro-1b-L100-longo-frio: p0: slots 768 de 768 | p1: slots 768 de 768 | p2: slots 768 de 768
- lastro-1b-L100-longo-quente: p1: slots 768 de 768

## NÃO RODOU (nunca vira razão nem estimativa)
Nenhuma célula.

