# Degrau 2: Lastro com experts por camada inteira contra llama.cpp S2 (resultado descritivo)

Critério: criterio-degrau2-camadas-20261109.md (herda criterio-degrau1-100pct-20261109.md com a Emenda 1 e criterio-comparacao-llamacpp-20261108.md, emendas 1 a 9). Sem meta e sem PASSA/REPROVA de velocidade. **Todo rótulo: Lastro camadas inteiras (22 na CPU) contra llama.cpp S2 (--n-cpu-moe 22); KV f32 no Lastro contra f16 no llama.cpp; op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU.** Razão = mediana LC / mediana S2 (menor que 1 = Lastro mais rápido); prefill e decode ficam SEPARADOS e não são juntados. Dispersão > 15% = ruidosa; SEM PROVA e NÃO RODOU como no critério base (nunca viram razão).

A S2 foi remedida NESTA janela (nenhum número de janela anterior entra em razão nova). S1 e S3 não rodam. cpu/needed do Lastro é ALTO por desenho (os experts das N primeiras camadas só existem na CPU): reportado, nunca exigido como 0.

## Smoke (seção 3)
- veredito: OK
- camadas_cpu: 22
- gpu_pico_pct: 11.8
- gpu_leituras_no_intervalo: 21
- cpu_needed: 0.9166666666666666
- ids: IGUAIS
- slots_impressos: camadas-cpu 22 de 24; slots 64 de 768
- cmdline: /home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/target/release/primeira-palavra --gera --celula 1b:/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct --so-texto curto --camadas-cpu 22 --saida /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/d2-1b-real-20261109-0402/smoke/p1/saida

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 1b (uma janela por modelo).

## Orçamento e cortes
Cortes aplicados: nenhum
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    smoke inicio=3s fim=14s rc=0
    01 lastro-1b-LC-curto-frio n=3 inicio=14s fim=63s rc=0
    02 llamacpp-1b-S2-curto-frio n=3 inicio=63s fim=79s rc=0
    03 lastro-1b-LC-curto-quente n=5 inicio=79s fim=106s rc=0
    04 llamacpp-1b-S2-curto-quente n=5 inicio=106s fim=122s rc=0
    05 llamacpp-1b-S2-medio-frio n=3 inicio=122s fim=138s rc=0
    06 lastro-1b-LC-medio-frio n=3 inicio=138s fim=188s rc=0
    07 llamacpp-1b-S2-medio-quente n=5 inicio=188s fim=204s rc=0
    08 lastro-1b-LC-medio-quente n=5 inicio=204s fim=235s rc=0
    09 lastro-1b-LC-longo-frio n=3 inicio=235s fim=291s rc=0
    10 llamacpp-1b-S2-longo-frio n=3 inicio=291s fim=307s rc=0
    11 lastro-1b-LC-longo-quente n=5 inicio=307s fim=348s rc=0
    12 llamacpp-1b-S2-longo-quente n=5 inicio=348s fim=364s rc=0

## Lastro camadas inteiras (22 na CPU) contra llama.cpp S2 (--n-cpu-moe 22); KV f32 no Lastro contra f16 no llama.cpp; op offload ligado no llama.cpp, prefill da S2 não é leitura de experts na CPU

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (LC) | LC prefill ms | S2 prefill ms | LC decode ms/passo | S2 decode ms/passo | faixa (seção 5, decode) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 10.070 (ruidosa) | 1.037 | mesma sequência; ruidosa | 0.917 (0.917 a 0.917) | 3244.90 (3191.25 a 4971.17) ruidosa | 322.23 (321.38 a 327.06) | 22.162 (21.546 a 22.388) | 21.381 (20.978 a 21.749) | decode LC/S2 <= 1,25 |
| 1b | curto | quente | 5 | 3.311 | 1.013 | mesma sequência | 0.917 (0.917 a 0.917) | 779.56 (766.43 a 788.52) | 235.41 (219.38 a 251.41) | 21.435 (21.194 a 21.562) | 21.151 (20.894 a 21.277) | decode LC/S2 <= 1,25 |
| 1b | medio | frio | 3 | 12.216 | 1.048 | mesma sequência | 0.917 (0.917 a 0.917) | 3923.00 (3907.16 a 3924.54) | 321.13 (317.05 a 332.35) | 21.867 (21.456 a 21.939) | 20.869 (20.700 a 21.326) | decode LC/S2 <= 1,25 |
| 1b | medio | quente | 5 | 6.669 | 1.009 | mesma sequência | 0.917 (0.917 a 0.917) | 1498.48 (1487.81 a 1517.00) | 224.69 (222.98 a 238.99) | 21.280 (21.233 a 21.762) | 21.100 (20.927 a 21.502) | decode LC/S2 <= 1,25 |
| 1b | longo | frio | 3 | 17.388 | 1.024 (sequências diferentes) | sequências diferentes | 0.917 (0.917 a 0.917) | 5599.52 (5586.50 a 5601.75) | 322.04 (319.74 a 332.24) | 21.744 (21.247 a 22.019) | 21.228 (21.209 a 21.262) | decode LC/S2 <= 1,25 |
| 1b | longo | quente | 5 | 13.444 | 1.008 (sequências diferentes) | sequências diferentes | 0.917 (0.917 a 0.917) | 3147.81 (3108.03 a 3198.88) | 234.15 (226.66 a 245.43) | 21.334 (21.269 a 21.759) | 21.174 (20.879 a 21.844) | decode LC/S2 <= 1,25 |

## Leitura (seção 5: palpites pré-registrados, orientação e NÃO decisão)
- decode LC/S2 <= 1,25: equivale
- decode LC/S2 entre 1,25 e 2: diferença moderada
- decode LC/S2 > 2: diferença grande

- 1b curto frio: razão de decode 1.037 cai em 'decode LC/S2 <= 1,25: equivale'
- 1b curto quente: razão de decode 1.013 cai em 'decode LC/S2 <= 1,25: equivale'
- 1b medio frio: razão de decode 1.048 cai em 'decode LC/S2 <= 1,25: equivale'
- 1b medio quente: razão de decode 1.009 cai em 'decode LC/S2 <= 1,25: equivale'
- 1b longo frio: razão de decode 1.024 cai em 'decode LC/S2 <= 1,25: equivale'
- 1b longo quente: razão de decode 1.008 cai em 'decode LC/S2 <= 1,25: equivale'

Limites declarados (seção 6): KV f32 contra f16; fronteira do prefill diferente (base); desktop gráfico como condição comum; memória do host declarada à parte; um Granite BF16 só; sem Strata; não responde sobre outros modelos ou quantizados.

## Razões de prefill e decode que diferem em mais de 1,5x (hipótese, nunca conclusão)
- 1b curto frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 43.5%; soma_passos_ms 18.9%; T_sequencia_ms 37.6% CANDIDATA A GARGALO
- 1b curto quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 36.8%; soma_passos_ms 63.2%; T_sequencia_ms 0.0%
- 1b medio frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 48.6%; soma_passos_ms 17.1%; T_sequencia_ms 34.6% CANDIDATA A GARGALO
- 1b medio quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 52.6%; soma_passos_ms 47.4%; T_sequencia_ms 0.0%
- 1b longo frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 57.3%; soma_passos_ms 14.0%; T_sequencia_ms 28.7%
- 1b longo quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 70.0%; soma_passos_ms 30.0%; T_sequencia_ms 0.0%

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-1b-LC-curto-frio | 5917 (5887 a 5918) (LIMITE SUPERIOR) | 45.1 |
| llamacpp-1b-S2-curto-frio | 634 (633 a 634) (frio sem aquecimento) | 46.8 |
| lastro-1b-LC-curto-quente | 5944 (5944 a 5944) (LIMITE SUPERIOR) | 46.7 |
| llamacpp-1b-S2-curto-quente | 644 (644 a 644) (inclui aquecimento padrão) | 47.3 |
| lastro-1b-LC-medio-frio | 5884 (5430 a 5923) (LIMITE SUPERIOR) | 45.7 |
| llamacpp-1b-S2-medio-frio | 634 (633 a 635) (frio sem aquecimento) | 47.9 |
| lastro-1b-LC-medio-quente | 6018 (6018 a 6018) (LIMITE SUPERIOR) | 47.0 |
| llamacpp-1b-S2-medio-quente | 635 (635 a 635) (inclui aquecimento padrão) | 47.4 |
| lastro-1b-LC-longo-frio | 5906 (5392 a 5916) (LIMITE SUPERIOR) | 46.0 |
| llamacpp-1b-S2-longo-frio | 634 (633 a 636) (frio sem aquecimento) | 47.1 |
| lastro-1b-LC-longo-quente | 5978 (5978 a 5978) (LIMITE SUPERIOR) | 46.9 |
| llamacpp-1b-S2-longo-quente | 659 (659 a 659) (inclui aquecimento padrão) | 47.2 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs G) é condição comum dos dois lados; processo de compute (C ou C+G) fora da árvore medida invalida a célula. Pico do pmon é amostra esparsa, não fração da placa.

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-1b-LC-curto-frio | 0 | 0 | 0 |
| llamacpp-1b-S2-curto-frio | 0 | 0 | 1 |
| lastro-1b-LC-curto-quente | 4 | 3 | 8 |
| llamacpp-1b-S2-curto-quente | 0 | 9 | 51 |
| lastro-1b-LC-medio-frio | 17 | 2 | 35 |
| llamacpp-1b-S2-medio-frio | 3 | 4 | 0 |
| lastro-1b-LC-medio-quente | 0 | 3 | 16 |
| llamacpp-1b-S2-medio-quente | 0 | 10 | 17 |
| lastro-1b-LC-longo-frio | 0 | 2 | 13 |
| llamacpp-1b-S2-longo-frio | 3 | 8 | 0 |
| lastro-1b-LC-longo-quente | 3 | 3 | 0 |
| llamacpp-1b-S2-longo-quente | 0 | 8 | 30 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## Linhas `camadas-cpu`/`slots` impressos pelo binário (LC)
- lastro-1b-LC-curto-frio: p0: camadas-cpu 22 de 24; slots 64 de 768 | p1: camadas-cpu 22 de 24; slots 64 de 768 | p2: camadas-cpu 22 de 24; slots 64 de 768
- lastro-1b-LC-curto-quente: p1: camadas-cpu 22 de 24; slots 64 de 768
- lastro-1b-LC-medio-frio: p0: camadas-cpu 22 de 24; slots 64 de 768 | p1: camadas-cpu 22 de 24; slots 64 de 768 | p2: camadas-cpu 22 de 24; slots 64 de 768
- lastro-1b-LC-medio-quente: p1: camadas-cpu 22 de 24; slots 64 de 768
- lastro-1b-LC-longo-frio: p0: camadas-cpu 22 de 24; slots 64 de 768 | p1: camadas-cpu 22 de 24; slots 64 de 768 | p2: camadas-cpu 22 de 24; slots 64 de 768
- lastro-1b-LC-longo-quente: p1: camadas-cpu 22 de 24; slots 64 de 768

## NÃO RODOU (nunca vira razão nem estimativa)
Nenhuma célula.

