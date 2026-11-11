# Comparação Lastro x llama.cpp (opção d): resultado descritivo

Critério: criterio-comparacao-llamacpp-20261108.md (emendas 1 a 6). Sem meta e sem PASSA/REPROVA de velocidade. **Toda razão é Lastro com 10% dos experts na GPU contra a série indicada do llama.cpp; nenhuma razão isola o motor.** Razão = mediana Lastro / mediana llama.cpp (menor que 1 = Lastro mais rápido), por série, nunca misturadas.

Declarações: KV f16 no llama.cpp contra f32 no Lastro (S1, S2); a ponte de correção é a igualdade dos ids gerados. Fronteira do prefill: `prompt_ms` do llama.cpp termina depois de amostrar o 1º token, `prefill_ms` do Lastro antes do argmax (sem ajuste). Decode: média por passo (63 passos nos dois lados). Quente e frio com o cache de página do GGUF/safetensors quente (o 1º contato com o disco não entra). Carga do Lastro é LIMITE SUPERIOR (inclui a cauda de encerramento).

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 1b (emenda 6: uma janela por modelo; o outro modelo é outra janela e não aparece aqui).

## Orçamento (E8) e cortes
Cortes aplicados: nenhum
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    01 lastro-1b-L-curto-frio n=3 inicio=5s fim=52s rc=0
    02 llamacpp-1b-S1-curto-frio n=3 inicio=52s fim=63s rc=0
    03 llamacpp-1b-S2-curto-frio n=3 inicio=63s fim=79s rc=0
    04 lastro-1b-L-curto-quente n=5 inicio=79s fim=106s rc=0
    05 llamacpp-1b-S1-curto-quente n=5 inicio=106s fim=114s rc=0
    06 llamacpp-1b-S2-curto-quente n=5 inicio=114s fim=131s rc=0
    07 llamacpp-1b-S1-medio-frio n=3 inicio=131s fim=141s rc=0
    08 llamacpp-1b-S2-medio-frio n=3 inicio=141s fim=157s rc=0
    09 lastro-1b-L-medio-frio n=3 inicio=157s fim=210s rc=0
    10 llamacpp-1b-S1-medio-quente n=5 inicio=210s fim=217s rc=0
    11 llamacpp-1b-S2-medio-quente n=5 inicio=217s fim=234s rc=0
    12 lastro-1b-L-medio-quente n=5 inicio=234s fim=266s rc=0
    13 lastro-1b-L-longo-frio n=3 inicio=266s fim=323s rc=0
    14 llamacpp-1b-S1-longo-frio n=3 inicio=323s fim=336s rc=0
    15 llamacpp-1b-S2-longo-frio n=3 inicio=336s fim=352s rc=0
    16 lastro-1b-L-longo-quente n=5 inicio=352s fim=395s rc=0
    17 llamacpp-1b-S1-longo-quente n=5 inicio=395s fim=402s rc=0
    18 llamacpp-1b-S2-longo-quente n=5 inicio=402s fim=419s rc=0
    19 llamacpp-1b-S3-curto-frio n=3 inicio=419s fim=429s rc=0
    20 llamacpp-1b-S3-curto-quente n=5 inicio=429s fim=437s rc=0
    21 llamacpp-1b-S3-medio-frio n=3 inicio=437s fim=447s rc=0
    22 llamacpp-1b-S3-medio-quente n=5 inicio=447s fim=455s rc=0
    23 llamacpp-1b-S3-longo-frio n=3 inicio=455s fim=466s rc=0
    24 llamacpp-1b-S3-longo-quente n=5 inicio=466s fim=473s rc=0

## Lastro 10% experts GPU contra llama.cpp S1 (tudo na GPU, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro)

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 37.441 | 10.803 | mesma sequência | 0.890 (0.890 a 0.890) | 3356.42 (3311.96 a 3419.77) | 89.64 (88.09 a 89.83) | 23.584 (21.532 a 23.689) | 2.183 (2.181 a 2.184) |
| 1b | curto | quente | 5 | 77.261 | 10.055 | mesma sequência | 0.890 (0.890 a 0.890) | 821.82 (814.34 a 900.67) | 10.64 (10.56 a 10.71) | 21.811 (21.446 a 22.867) | 2.169 (2.161 a 2.186) |
| 1b | medio | frio | 3 | 41.389 | 10.169 | mesma sequência | 0.894 (0.894 a 0.894) | 4102.76 (4097.78 a 4168.33) | 99.13 (96.74 a 101.05) | 22.431 (21.974 a 22.547) | 2.206 (2.190 a 2.212) |
| 1b | medio | quente | 5 | 99.855 | 10.025 | mesma sequência | 0.894 (0.894 a 0.894) | 1590.78 (1555.63 a 1605.47) | 15.93 (15.86 a 16.14) | 21.935 (21.490 a 22.196) | 2.188 (2.173 a 2.197) |
| 1b | longo | frio | 3 | 44.000 (ruidosa) | 9.134 (sequências diferentes) | sequências diferentes; ruidosa | 0.899 (0.899 a 0.899) | 5875.06 (5762.80 a 5910.23) | 133.52 (104.76 a 150.59) ruidosa | 21.862 (21.777 a 21.969) | 2.394 (2.207 a 2.510) |
| 1b | longo | quente | 5 | 119.442 | 9.959 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 3338.75 (3259.56 a 3395.43) | 27.95 (27.85 a 28.02) | 21.856 (21.584 a 23.283) | 2.195 (2.187 a 2.206) |

## Lastro 10% experts GPU contra llama.cpp S2 (--n-cpu-moe 22 no 1B e 29 no 3B, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro; descarregamento de ordem de grandeza parecida, NÃO o mesmo: o Lastro descarrega ~90% dos experts de cada camada, o llama.cpp camadas inteiras; op offload ligado, então o prefill da S2 não deve ser lido como 'experts calculados na CPU')

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 10.285 | 1.023 (ruidosa) | mesma sequência; ruidosa | 0.890 (0.890 a 0.890) | 3356.42 (3311.96 a 3419.77) | 326.33 (317.54 a 349.37) | 23.584 (21.532 a 23.689) | 23.064 (22.230 a 25.785) ruidosa |
| 1b | curto | quente | 5 | 3.367 | 0.947 | mesma sequência | 0.890 (0.890 a 0.890) | 821.82 (814.34 a 900.67) | 244.07 (233.99 a 250.98) | 21.811 (21.446 a 22.867) | 23.039 (22.715 a 24.124) |
| 1b | medio | frio | 3 | 12.000 | 0.947 (sequências diferentes) | sequências diferentes | 0.894 (0.894 a 0.894) | 4102.76 (4097.78 a 4168.33) | 341.89 (333.27 a 369.10) | 22.431 (21.974 a 22.547) | 23.695 (23.112 a 23.921) |
| 1b | medio | quente | 5 | 6.533 | 0.979 (sequências diferentes) | sequências diferentes | 0.894 (0.894 a 0.894) | 1590.78 (1555.63 a 1605.47) | 243.48 (237.32 a 257.70) | 21.935 (21.490 a 22.196) | 22.400 (22.141 a 22.908) |
| 1b | longo | frio | 3 | 17.428 | 0.950 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 5875.06 (5762.80 a 5910.23) | 337.10 (329.27 a 337.88) | 21.862 (21.777 a 21.969) | 23.005 (22.416 a 23.124) |
| 1b | longo | quente | 5 | 12.999 | 0.964 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 3338.75 (3259.56 a 3395.43) | 256.84 (251.24 a 269.58) | 21.856 (21.584 a 23.283) | 22.671 (22.056 a 23.118) |

S2 é aproximação declarada (camadas inteiras no llama.cpp, ~90% dos experts de cada camada no Lastro); op offload ligado: o prefill da S2 não deve ser lido como 'experts calculados na CPU'.

## Lastro 10% experts GPU contra llama.cpp S3 (KV f32 com -fa off, paridade de armazenamento; configuração prática, sem atribuir a razão só ao motor)

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 38.725 | 10.295 | mesma sequência | 0.890 (0.890 a 0.890) | 3356.42 (3311.96 a 3419.77) | 86.67 (85.68 a 89.16) | 23.584 (21.532 a 23.689) | 2.291 (2.289 a 2.293) |
| 1b | curto | quente | 5 | 72.760 | 9.612 | mesma sequência | 0.890 (0.890 a 0.890) | 821.82 (814.34 a 900.67) | 11.29 (11.06 a 11.80) | 21.811 (21.446 a 22.867) | 2.269 (2.268 a 2.296) |
| 1b | medio | frio | 3 | 44.840 | 9.780 | mesma sequência | 0.894 (0.894 a 0.894) | 4102.76 (4097.78 a 4168.33) | 91.50 (91.29 a 93.62) | 22.431 (21.974 a 22.547) | 2.294 (2.287 a 2.305) |
| 1b | medio | quente | 5 | 97.272 | 9.587 | mesma sequência | 0.894 (0.894 a 0.894) | 1590.78 (1555.63 a 1605.47) | 16.35 (16.24 a 16.68) | 21.935 (21.490 a 22.196) | 2.288 (2.273 a 2.293) |
| 1b | longo | frio | 3 | 53.409 | 9.160 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 5875.06 (5762.80 a 5910.23) | 110.00 (105.45 a 114.24) | 21.862 (21.777 a 21.969) | 2.387 (2.386 a 2.398) |
| 1b | longo | quente | 5 | 109.979 | 9.158 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 3338.75 (3259.56 a 3395.43) | 30.36 (30.34 a 30.98) | 21.856 (21.584 a 23.283) | 2.387 (2.375 a 2.404) |

## Leitura da diferença (E7, hipótese e nunca conclusão)
- 1b curto frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 44.0%; soma_passos_ms 19.5%; T_sequencia_ms 36.5% CANDIDATA A GARGALO; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 44.0%; soma_passos_ms 19.5%; T_sequencia_ms 36.5% CANDIDATA A GARGALO; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto frio, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 44.0%; soma_passos_ms 19.5%; T_sequencia_ms 36.5% CANDIDATA A GARGALO; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.5%; soma_passos_ms 62.4%; T_sequencia_ms 0.0%; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.5%; soma_passos_ms 62.4%; T_sequencia_ms 0.0%; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto quente, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.5%; soma_passos_ms 62.4%; T_sequencia_ms 0.0%; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b medio frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 49.5%; soma_passos_ms 16.8%; T_sequencia_ms 33.5% CANDIDATA A GARGALO; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 49.5%; soma_passos_ms 16.8%; T_sequencia_ms 33.5% CANDIDATA A GARGALO; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio frio, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 49.5%; soma_passos_ms 16.8%; T_sequencia_ms 33.5% CANDIDATA A GARGALO; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 53.5%; soma_passos_ms 46.5%; T_sequencia_ms 0.0%; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 53.5%; soma_passos_ms 46.5%; T_sequencia_ms 0.0%; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio quente, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 53.5%; soma_passos_ms 46.5%; T_sequencia_ms 0.0%; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b longo frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 58.2%; soma_passos_ms 13.7%; T_sequencia_ms 28.1%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 58.2%; soma_passos_ms 13.7%; T_sequencia_ms 28.1%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo frio, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 58.2%; soma_passos_ms 13.7%; T_sequencia_ms 28.1%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 70.6%; soma_passos_ms 29.4%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 70.6%; soma_passos_ms 29.4%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo quente, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 70.6%; soma_passos_ms 29.4%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-1b-L-curto-frio | 5777 (5515 a 5794) (LIMITE SUPERIOR) | 42.4 |
| llamacpp-1b-S1-curto-frio | 689 (682 a 722) (frio sem aquecimento) | 458.1 |
| llamacpp-1b-S2-curto-frio | 495 (488 a 501) (frio sem aquecimento) | 43.4 |
| llamacpp-1b-S3-curto-frio | 686 (679 a 687) (frio sem aquecimento) | 436.5 |
| lastro-1b-L-curto-quente | 5922 (5922 a 5922) (LIMITE SUPERIOR) | 45.8 |
| llamacpp-1b-S1-curto-quente | 697 (697 a 697) (inclui aquecimento padrão) | 461.0 |
| llamacpp-1b-S2-curto-quente | 635 (635 a 635) (inclui aquecimento padrão) | 43.4 |
| llamacpp-1b-S3-curto-quente | 706 (706 a 706) (inclui aquecimento padrão) | 440.7 |
| lastro-1b-L-medio-frio | 5780 (5764 a 6093) (LIMITE SUPERIOR) | 44.6 |
| llamacpp-1b-S1-medio-frio | 687 (681 a 695) (frio sem aquecimento) | 453.4 |
| llamacpp-1b-S2-medio-frio | 492 (432 a 500) (frio sem aquecimento) | 42.2 |
| llamacpp-1b-S3-medio-frio | 680 (677 a 682) (frio sem aquecimento) | 436.0 |
| lastro-1b-L-medio-quente | 5876 (5876 a 5876) (LIMITE SUPERIOR) | 45.6 |
| llamacpp-1b-S1-medio-quente | 706 (706 a 706) (inclui aquecimento padrão) | 457.0 |
| llamacpp-1b-S2-medio-quente | 638 (638 a 638) (inclui aquecimento padrão) | 44.6 |
| llamacpp-1b-S3-medio-quente | 698 (698 a 698) (inclui aquecimento padrão) | 437.1 |
| lastro-1b-L-longo-frio | 5772 (5769 a 5871) (LIMITE SUPERIOR) | 45.7 |
| llamacpp-1b-S1-longo-frio | 1071 (712 a 2305) ruidosa (frio sem aquecimento) | 417.8 |
| llamacpp-1b-S2-longo-frio | 492 (491 a 496) (frio sem aquecimento) | 43.5 |
| llamacpp-1b-S3-longo-frio | 688 (685 a 694) (frio sem aquecimento) | 419.0 |
| lastro-1b-L-longo-quente | 5876 (5876 a 5876) (LIMITE SUPERIOR) | 45.8 |
| llamacpp-1b-S1-longo-quente | 696 (696 a 696) (inclui aquecimento padrão) | 455.6 |
| llamacpp-1b-S2-longo-quente | 633 (633 a 633) (inclui aquecimento padrão) | 44.1 |
| llamacpp-1b-S3-longo-quente | 703 (703 a 703) (inclui aquecimento padrão) | 419.0 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs de tipo G: compositor Hyprland, quickshell, Xwayland...) divide a placa de medida nos DOIS lados em todas as células: é condição COMUM dos dois lados, fora do SEM PROVA; o efeito pode não ser igual entre eles (o Lastro serve os experts na CPU, a S1 do llama.cpp roda tudo na GPU, a S2 tem parte na CPU). Processos de compute (tipo C ou C+G, como voxtype e o processo de GPU do Chrome) fora da árvore medida NÃO são isentos: qualquer um, ocioso ou não, é SEM PROVA. A coluna do pico do pmon é a soma de sm% dos PIDs de tipo G entre as 3 capturas de uma volta (`nvidia-smi pmon -c 1 -s um`, antes e depois de cada célula): é PICO DE AMOSTRA ESPARSA, não a fração da placa usada pelo desktop; `-` = sem leitura. A coluna do dmon (`dmon -s u -c 3`, só antes da célula, quando nada medido roda) é só informação. Salvaguarda decisiva: soma G > 60% (palpite declarado antes dos dados).

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-1b-L-curto-frio | 0 | 0 | 0 |
| llamacpp-1b-S1-curto-frio | 0 | 0 | 0 |
| llamacpp-1b-S2-curto-frio | 0 | 0 | 0 |
| llamacpp-1b-S3-curto-frio | 0 | 0 | 0 |
| lastro-1b-L-curto-quente | 0 | 0 | 0 |
| llamacpp-1b-S1-curto-quente | 0 | 0 | 0 |
| llamacpp-1b-S2-curto-quente | 0 | 0 | 0 |
| llamacpp-1b-S3-curto-quente | 0 | 0 | 0 |
| lastro-1b-L-medio-frio | 0 | 0 | 0 |
| llamacpp-1b-S1-medio-frio | 0 | 0 | 0 |
| llamacpp-1b-S2-medio-frio | 0 | 0 | 0 |
| llamacpp-1b-S3-medio-frio | 0 | 0 | 0 |
| lastro-1b-L-medio-quente | 0 | 0 | 0 |
| llamacpp-1b-S1-medio-quente | 0 | 0 | 0 |
| llamacpp-1b-S2-medio-quente | 0 | 0 | 0 |
| llamacpp-1b-S3-medio-quente | 0 | 0 | 0 |
| lastro-1b-L-longo-frio | 0 | 0 | 0 |
| llamacpp-1b-S1-longo-frio | 0 | 0 | 0 |
| llamacpp-1b-S2-longo-frio | 0 | 0 | 0 |
| llamacpp-1b-S3-longo-frio | 0 | 0 | 0 |
| lastro-1b-L-longo-quente | 0 | 0 | 0 |
| llamacpp-1b-S1-longo-quente | 0 | 0 | 0 |
| llamacpp-1b-S2-longo-quente | 0 | 0 | 0 |
| llamacpp-1b-S3-longo-quente | 0 | 0 | 0 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## NÃO RODOU (nunca vira razão, PASSA nem estimativa)
Nenhuma célula.

