# Degrau 2: Lastro com experts por camada inteira contra llama.cpp S2 (resultado descritivo)

Critério: criterio-degrau2-camadas-20261109.md (herda criterio-degrau1-100pct-20261109.md com a Emenda 1 e criterio-comparacao-llamacpp-20261108.md, emendas 1 a 9). Sem meta e sem PASSA/REPROVA de velocidade. **Todo rótulo: Lastro camadas inteiras (29 na CPU) contra llama.cpp S2 (--n-cpu-moe 29); KV f32 no Lastro contra f16 no llama.cpp; op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU.** Razão = mediana LC / mediana S2 (menor que 1 = Lastro mais rápido); prefill e decode ficam SEPARADOS e não são juntados. Dispersão > 15% = ruidosa; SEM PROVA e NÃO RODOU como no critério base (nunca viram razão).

A S2 foi remedida NESTA janela (nenhum número de janela anterior entra em razão nova). S1 e S3 não rodam. cpu/needed do Lastro é ALTO por desenho (os experts das N primeiras camadas só existem na CPU): reportado, nunca exigido como 0.

## Smoke (seção 3)
- veredito: OK
- camadas_cpu: 29
- gpu_pico_pct: 14.7
- gpu_leituras_no_intervalo: 37
- cpu_needed: 0.90625
- ids: IGUAIS
- slots_impressos: camadas-cpu 29 de 32; slots 120 de 1280
- cmdline: /home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/target/release/primeira-palavra --gera --celula 3b:/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct --so-texto curto --camadas-cpu 29 --saida /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/d2-3b-real-20261109-0409/smoke/p1/saida

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 3b (uma janela por modelo).

## Orçamento e cortes
Cortes aplicados: nenhum
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    smoke inicio=3s fim=23s rc=0
    01 lastro-3b-LC-curto-frio n=3 inicio=23s fim=104s rc=0
    02 llamacpp-3b-S2-curto-frio n=3 inicio=104s fim=128s rc=0
    03 lastro-3b-LC-curto-quente n=5 inicio=128s fim=173s rc=0
    04 llamacpp-3b-S2-curto-quente n=5 inicio=173s fim=199s rc=0
    05 llamacpp-3b-S2-medio-frio n=3 inicio=199s fim=223s rc=0
    06 lastro-3b-LC-medio-frio n=3 inicio=223s fim=310s rc=0
    07 llamacpp-3b-S2-medio-quente n=5 inicio=310s fim=336s rc=0
    08 lastro-3b-LC-medio-quente n=5 inicio=336s fim=389s rc=0
    09 lastro-3b-LC-longo-frio n=3 inicio=389s fim=490s rc=0
    10 llamacpp-3b-S2-longo-frio n=3 inicio=490s fim=514s rc=0
    11 lastro-3b-LC-longo-quente n=5 inicio=514s fim=590s rc=0
    12 llamacpp-3b-S2-longo-quente n=5 inicio=590s fim=616s rc=0

## Lastro camadas inteiras (29 na CPU) contra llama.cpp S2 (--n-cpu-moe 29); KV f32 no Lastro contra f16 no llama.cpp; op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (LC) | LC prefill ms | S2 prefill ms | LC decode ms/passo | S2 decode ms/passo | faixa (seção 5, decode) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3b | curto | frio | 3 | 6.560 | 0.991 | mesma sequência | 0.906 (0.906 a 0.906) | 4283.73 (4279.70 a 4350.23) | 653.03 (623.62 a 653.69) | 40.969 (40.900 a 41.101) | 41.329 (40.924 a 41.677) | decode LC/S2 <= 1,25 |
| 3b | curto | quente | 5 | 2.820 | 0.986 | mesma sequência | 0.906 (0.906 a 0.906) | 1558.65 (1551.56 a 1570.65) | 552.74 (543.75 a 566.11) | 40.529 (40.377 a 41.312) | 41.110 (41.037 a 41.442) | decode LC/S2 <= 1,25 |
| 3b | medio | frio | 3 | 8.592 | 0.986 (sequências diferentes) | sequências diferentes | 0.906 (0.906 a 0.906) | 5742.29 (5728.09 a 5806.61) | 668.29 (653.20 a 669.45) | 40.523 (40.502 a 41.054) | 41.101 (40.903 a 41.124) | decode LC/S2 <= 1,25 |
| 3b | medio | quente | 5 | 5.316 | 0.986 (sequências diferentes) | sequências diferentes | 0.906 (0.906 a 0.906) | 2953.89 (2939.43 a 2977.99) | 555.67 (546.25 a 597.75) | 40.907 (40.833 a 41.097) | 41.486 (41.202 a 41.934) | decode LC/S2 <= 1,25 |
| 3b | longo | frio | 3 | 13.343 | 0.994 (sequências diferentes) | sequências diferentes | 0.906 (0.906 a 0.906) | 9015.02 (9008.41 a 9065.93) | 675.64 (673.92 a 720.18) | 40.772 (40.371 a 41.421) | 41.032 (40.941 a 41.170) | decode LC/S2 <= 1,25 |
| 3b | longo | quente | 5 | 10.853 (ruidosa) | 0.980 (sequências diferentes) | sequências diferentes; ruidosa | 0.906 (0.906 a 0.906) | 6257.42 (6244.13 a 8969.76) ruidosa | 576.56 (555.32 a 583.66) | 40.645 (40.413 a 40.916) | 41.480 (41.066 a 41.700) | decode LC/S2 <= 1,25 |

## Leitura (seção 5: palpites pré-registrados, orientação e NÃO decisão)
- decode LC/S2 <= 1,25: equivale
- decode LC/S2 entre 1,25 e 2: diferença moderada
- decode LC/S2 > 2: diferença grande

- 3b curto frio: razão de decode 0.991 cai em 'decode LC/S2 <= 1,25: equivale'
- 3b curto quente: razão de decode 0.986 cai em 'decode LC/S2 <= 1,25: equivale'
- 3b medio frio: razão de decode 0.986 cai em 'decode LC/S2 <= 1,25: equivale'
- 3b medio quente: razão de decode 0.986 cai em 'decode LC/S2 <= 1,25: equivale'
- 3b longo frio: razão de decode 0.994 cai em 'decode LC/S2 <= 1,25: equivale'
- 3b longo quente: razão de decode 0.980 cai em 'decode LC/S2 <= 1,25: equivale'

Limites declarados (seção 6): KV f32 contra f16; fronteira do prefill diferente (base); desktop gráfico como condição comum; memória do host declarada à parte; um Granite BF16 só; sem Strata; não responde sobre outros modelos ou quantizados.

## Razões de prefill e decode que diferem em mais de 1,5x (hipótese, nunca conclusão)
- 3b curto frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 33.8%; soma_passos_ms 20.3%; T_sequencia_ms 45.9% CANDIDATA A GARGALO
- 3b curto quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.8%; soma_passos_ms 62.2%; T_sequencia_ms 0.0%
- 3b medio frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 40.5%; soma_passos_ms 18.0%; T_sequencia_ms 41.5% CANDIDATA A GARGALO
- 3b medio quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 53.4%; soma_passos_ms 46.6%; T_sequencia_ms 0.0%
- 3b longo frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 51.9%; soma_passos_ms 14.7%; T_sequencia_ms 33.4% CANDIDATA A GARGALO
- 3b longo quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 71.1%; soma_passos_ms 28.9%; T_sequencia_ms 0.0%

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-3b-LC-curto-frio | 12160 (12144 a 12182) (LIMITE SUPERIOR) | 24.4 |
| llamacpp-3b-S2-curto-frio | 938 (936 a 943) (frio sem aquecimento) | 24.2 |
| lastro-3b-LC-curto-quente | 12278 (12278 a 12278) (LIMITE SUPERIOR) | 24.7 |
| llamacpp-3b-S2-curto-quente | 986 (986 a 986) (inclui aquecimento padrão) | 24.3 |
| lastro-3b-LC-medio-frio | 12072 (11829 a 12375) (LIMITE SUPERIOR) | 24.7 |
| llamacpp-3b-S2-medio-frio | 929 (925 a 932) (frio sem aquecimento) | 24.3 |
| lastro-3b-LC-medio-quente | 12087 (12087 a 12087) (LIMITE SUPERIOR) | 24.4 |
| llamacpp-3b-S2-medio-quente | 984 (984 a 984) (inclui aquecimento padrão) | 24.1 |
| lastro-3b-LC-longo-frio | 12120 (12099 a 12183) (LIMITE SUPERIOR) | 24.5 |
| llamacpp-3b-S2-longo-frio | 927 (924 a 944) (frio sem aquecimento) | 24.4 |
| lastro-3b-LC-longo-quente | 12161 (12161 a 12161) (LIMITE SUPERIOR) | 24.6 |
| llamacpp-3b-S2-longo-quente | 972 (972 a 972) (inclui aquecimento padrão) | 24.1 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs G) é condição comum dos dois lados; processo de compute (C ou C+G) fora da árvore medida invalida a célula. Pico do pmon é amostra esparsa, não fração da placa.

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-3b-LC-curto-frio | 0 | 2 | 27 |
| llamacpp-3b-S2-curto-frio | 3 | 4 | 0 |
| lastro-3b-LC-curto-quente | 9 | 0 | 24 |
| llamacpp-3b-S2-curto-quente | 2 | 2 | 12 |
| lastro-3b-LC-medio-frio | 14 | 2 | 17 |
| llamacpp-3b-S2-medio-frio | 8 | 5 | 0 |
| lastro-3b-LC-medio-quente | 3 | 1 | 19 |
| llamacpp-3b-S2-medio-quente | 0 | 11 | 16 |
| lastro-3b-LC-longo-frio | 1 | 2 | 0 |
| llamacpp-3b-S2-longo-frio | 1 | 4 | 12 |
| lastro-3b-LC-longo-quente | 0 | 1 | 20 |
| llamacpp-3b-S2-longo-quente | 0 | 13 | 16 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## Linhas `camadas-cpu`/`slots` impressos pelo binário (LC)
- lastro-3b-LC-curto-frio: p0: camadas-cpu 29 de 32; slots 120 de 1280 | p1: camadas-cpu 29 de 32; slots 120 de 1280 | p2: camadas-cpu 29 de 32; slots 120 de 1280
- lastro-3b-LC-curto-quente: p1: camadas-cpu 29 de 32; slots 120 de 1280
- lastro-3b-LC-medio-frio: p0: camadas-cpu 29 de 32; slots 120 de 1280 | p1: camadas-cpu 29 de 32; slots 120 de 1280 | p2: camadas-cpu 29 de 32; slots 120 de 1280
- lastro-3b-LC-medio-quente: p1: camadas-cpu 29 de 32; slots 120 de 1280
- lastro-3b-LC-longo-frio: p0: camadas-cpu 29 de 32; slots 120 de 1280 | p1: camadas-cpu 29 de 32; slots 120 de 1280 | p2: camadas-cpu 29 de 32; slots 120 de 1280
- lastro-3b-LC-longo-quente: p1: camadas-cpu 29 de 32; slots 120 de 1280

## NÃO RODOU (nunca vira razão nem estimativa)
Nenhuma célula.

