# Comparação Lastro x llama.cpp (opção d): resultado descritivo

Critério: criterio-comparacao-llamacpp-20261108.md (emendas 1 a 6). Sem meta e sem PASSA/REPROVA de velocidade. **Toda razão é Lastro com 10% dos experts na GPU contra a série indicada do llama.cpp; nenhuma razão isola o motor.** Razão = mediana Lastro / mediana llama.cpp (menor que 1 = Lastro mais rápido), por série, nunca misturadas.

Declarações: KV f16 no llama.cpp contra f32 no Lastro (S1, S2); a ponte de correção é a igualdade dos ids gerados. Fronteira do prefill: `prompt_ms` do llama.cpp termina depois de amostrar o 1º token, `prefill_ms` do Lastro antes do argmax (sem ajuste). Decode: média por passo (63 passos nos dois lados). Quente e frio com o cache de página do GGUF/safetensors quente (o 1º contato com o disco não entra). Carga do Lastro é LIMITE SUPERIOR (inclui a cauda de encerramento).

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 3b (emenda 6: uma janela por modelo; o outro modelo é outra janela e não aparece aqui).

## Orçamento (E8) e cortes
Cortes aplicados: (1) S3 inteira não roda
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    01 lastro-3b-L-curto-frio n=3 inicio=3s fim=87s rc=0
    02 llamacpp-3b-S1-curto-frio n=3 inicio=87s fim=101s rc=0
    03 llamacpp-3b-S2-curto-frio n=3 inicio=101s fim=124s rc=0
    04 lastro-3b-L-curto-quente n=5 inicio=124s fim=170s rc=0
    05 llamacpp-3b-S1-curto-quente n=5 inicio=170s fim=179s rc=0
    06 llamacpp-3b-S2-curto-quente n=5 inicio=179s fim=205s rc=0
    07 llamacpp-3b-S1-medio-frio n=3 inicio=205s fim=219s rc=0
    08 llamacpp-3b-S2-medio-frio n=3 inicio=219s fim=242s rc=0
    09 lastro-3b-L-medio-frio n=3 inicio=242s fim=330s rc=0
    10 llamacpp-3b-S1-medio-quente n=5 inicio=330s fim=339s rc=0
    11 llamacpp-3b-S2-medio-quente n=5 inicio=339s fim=366s rc=0
    12 lastro-3b-L-medio-quente n=5 inicio=366s fim=421s rc=0
    13 lastro-3b-L-longo-frio n=3 inicio=421s fim=526s rc=0
    14 llamacpp-3b-S1-longo-frio n=3 inicio=526s fim=540s rc=0
    15 llamacpp-3b-S2-longo-frio n=3 inicio=540s fim=563s rc=0
    16 lastro-3b-L-longo-quente n=5 inicio=563s fim=637s rc=0
    17 llamacpp-3b-S1-longo-quente n=5 inicio=637s fim=646s rc=0
    18 llamacpp-3b-S2-longo-quente n=5 inicio=646s fim=672s rc=0

## Lastro 10% experts GPU contra llama.cpp S1 (tudo na GPU, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro)

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3b | curto | frio | 3 | 40.767 | 12.215 | mesma sequência | 0.899 (0.899 a 0.899) | 4323.02 (4281.56 a 4323.58) | 106.04 (105.78 a 110.08) | 40.995 (40.720 a 41.299) | 3.356 (3.334 a 3.414) |
| 3b | curto | quente | 5 | 119.480 | 12.550 | mesma sequência | 0.899 (0.899 a 0.899) | 1561.49 (1534.56 a 1642.45) | 13.07 (12.85 a 13.55) | 41.438 (40.604 a 42.064) | 3.302 (3.264 a 3.310) |
| 3b | medio | frio | 3 | 43.695 | 12.319 | mesma sequência | 0.898 (0.898 a 0.898) | 5737.66 (5701.35 a 5740.63) | 131.31 (128.91 a 132.08) | 40.450 (40.271 a 40.934) | 3.284 (3.278 a 3.294) |
| 3b | medio | quente | 5 | 140.935 | 13.007 | mesma sequência | 0.898 (0.898 a 0.898) | 3006.14 (2963.10 a 3107.93) | 21.33 (21.16 a 21.54) | 41.497 (41.291 a 42.831) | 3.190 (3.188 a 3.295) |
| 3b | longo | frio | 3 | 70.135 | 12.421 | mesma sequência | 0.898 (0.898 a 0.898) | 9202.88 (8968.01 a 9336.05) | 131.22 (129.10 a 131.69) | 41.646 (40.504 a 42.601) | 3.353 (3.326 a 3.357) |
| 3b | longo | quente | 5 | 217.775 | 12.563 | mesma sequência | 0.898 (0.898 a 0.898) | 6346.40 (6245.08 a 6422.46) | 29.14 (29.13 a 29.74) | 40.586 (40.305 a 41.164) | 3.231 (3.223 a 3.249) |

## Lastro 10% experts GPU contra llama.cpp S2 (--n-cpu-moe 22 no 1B e 29 no 3B, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro; descarregamento de ordem de grandeza parecida, NÃO o mesmo: o Lastro descarrega ~90% dos experts de cada camada, o llama.cpp camadas inteiras; op offload ligado, então o prefill da S2 não deve ser lido como 'experts calculados na CPU')

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3b | curto | frio | 3 | 6.681 | 0.984 | mesma sequência | 0.899 (0.899 a 0.899) | 4323.02 (4281.56 a 4323.58) | 647.04 (639.79 a 664.72) | 40.995 (40.720 a 41.299) | 41.649 (41.366 a 41.710) |
| 3b | curto | quente | 5 | 2.764 | 0.991 | mesma sequência | 0.899 (0.899 a 0.899) | 1561.49 (1534.56 a 1642.45) | 564.96 (534.77 a 577.36) | 41.438 (40.604 a 42.064) | 41.807 (41.498 a 42.083) |
| 3b | medio | frio | 3 | 8.602 | 0.979 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 5737.66 (5701.35 a 5740.63) | 667.00 (657.03 a 688.94) | 40.450 (40.271 a 40.934) | 41.328 (40.904 a 41.730) |
| 3b | medio | quente | 5 | 5.123 | 0.955 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 3006.14 (2963.10 a 3107.93) | 586.79 (577.51 a 621.10) | 41.497 (41.291 a 42.831) | 43.468 (42.761 a 43.920) |
| 3b | longo | frio | 3 | 13.512 | 1.002 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 9202.88 (8968.01 a 9336.05) | 681.09 (664.91 a 687.52) | 41.646 (40.504 a 42.601) | 41.546 (41.035 a 42.003) |
| 3b | longo | quente | 5 | 11.092 | 0.976 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 6346.40 (6245.08 a 6422.46) | 572.16 (554.87 a 586.09) | 40.586 (40.305 a 41.164) | 41.582 (41.402 a 41.828) |

S2 é aproximação declarada (camadas inteiras no llama.cpp, ~90% dos experts de cada camada no Lastro); op offload ligado: o prefill da S2 não deve ser lido como 'experts calculados na CPU'.

## Lastro 10% experts GPU contra llama.cpp S3 (KV f32 com -fa off, paridade de armazenamento; configuração prática, sem atribuir a razão só ao motor)

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3b | curto | frio | - | NÃO RODOU | NÃO RODOU | célula do llama.cpp não rodou | - | - | - | - | - |
| 3b | curto | quente | - | NÃO RODOU | NÃO RODOU | célula do llama.cpp não rodou | - | - | - | - | - |
| 3b | medio | frio | - | NÃO RODOU | NÃO RODOU | célula do llama.cpp não rodou | - | - | - | - | - |
| 3b | medio | quente | - | NÃO RODOU | NÃO RODOU | célula do llama.cpp não rodou | - | - | - | - | - |
| 3b | longo | frio | - | NÃO RODOU | NÃO RODOU | célula do llama.cpp não rodou | - | - | - | - | - |
| 3b | longo | quente | - | NÃO RODOU | NÃO RODOU | célula do llama.cpp não rodou | - | - | - | - | - |

## Leitura da diferença (E7, hipótese e nunca conclusão)
- 3b curto frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 33.1%; soma_passos_ms 19.9%; T_sequencia_ms 47.2% CANDIDATA A GARGALO; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b curto frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 33.1%; soma_passos_ms 19.9%; T_sequencia_ms 47.2% CANDIDATA A GARGALO; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b curto quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.4%; soma_passos_ms 62.6%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b curto quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.4%; soma_passos_ms 62.6%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b medio frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 40.6%; soma_passos_ms 18.1%; T_sequencia_ms 41.4% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b medio frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 40.6%; soma_passos_ms 18.1%; T_sequencia_ms 41.4% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b medio quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 53.5%; soma_passos_ms 46.5%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b medio quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 53.5%; soma_passos_ms 46.5%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 51.6%; soma_passos_ms 14.7%; T_sequencia_ms 33.8% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 51.6%; soma_passos_ms 14.7%; T_sequencia_ms 33.8% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 71.1%; soma_passos_ms 28.9%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 71.1%; soma_passos_ms 28.9%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-3b-L-curto-frio | 12566 (12327 a 12640) (LIMITE SUPERIOR) | 24.4 |
| llamacpp-3b-S1-curto-frio | 1303 (1301 a 1306) (frio sem aquecimento) | 298.0 |
| llamacpp-3b-S2-curto-frio | 936 (928 a 945) (frio sem aquecimento) | 24.0 |
| lastro-3b-L-curto-quente | 12644 (12644 a 12644) (LIMITE SUPERIOR) | 24.1 |
| llamacpp-3b-S1-curto-quente | 1400 (1400 a 1400) (inclui aquecimento padrão) | 302.9 |
| llamacpp-3b-S2-curto-quente | 963 (963 a 963) (inclui aquecimento padrão) | 23.9 |
| lastro-3b-L-medio-frio | 12309 (12222 a 12328) (LIMITE SUPERIOR) | 24.7 |
| llamacpp-3b-S1-medio-frio | 1304 (1298 a 1304) (frio sem aquecimento) | 304.6 |
| llamacpp-3b-S2-medio-frio | 932 (930 a 938) (frio sem aquecimento) | 24.2 |
| lastro-3b-L-medio-quente | 12686 (12686 a 12686) (LIMITE SUPERIOR) | 24.1 |
| llamacpp-3b-S1-medio-quente | 1373 (1373 a 1373) (inclui aquecimento padrão) | 313.4 |
| llamacpp-3b-S2-medio-quente | 959 (959 a 959) (inclui aquecimento padrão) | 23.0 |
| lastro-3b-L-longo-frio | 12504 (12306 a 12695) (LIMITE SUPERIOR) | 24.0 |
| llamacpp-3b-S1-longo-frio | 1306 (1303 a 1318) (frio sem aquecimento) | 298.3 |
| llamacpp-3b-S2-longo-frio | 933 (931 a 943) (frio sem aquecimento) | 24.1 |
| lastro-3b-L-longo-quente | 12380 (12380 a 12380) (LIMITE SUPERIOR) | 24.6 |
| llamacpp-3b-S1-longo-quente | 1392 (1392 a 1392) (inclui aquecimento padrão) | 309.5 |
| llamacpp-3b-S2-longo-quente | 941 (941 a 941) (inclui aquecimento padrão) | 24.0 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs de tipo G: compositor Hyprland, quickshell, Xwayland...) divide a 4090 nos DOIS lados em todas as células: é condição COMUM dos dois lados, fora do SEM PROVA; o efeito pode não ser igual entre eles (o Lastro serve os experts na CPU, a S1 do llama.cpp roda tudo na GPU, a S2 tem parte na CPU). Processos de compute (tipo C ou C+G, como voxtype e o processo de GPU do Chrome) fora da árvore medida NÃO são isentos: qualquer um, ocioso ou não, é SEM PROVA. A coluna do pico do pmon é a soma de sm% dos PIDs de tipo G entre as 3 capturas de uma volta (`nvidia-smi pmon -c 1 -s um`, antes e depois de cada célula): é PICO DE AMOSTRA ESPARSA, não a fração da placa usada pelo desktop; `-` = sem leitura. A coluna do dmon (`dmon -s u -c 3`, só antes da célula, quando nada medido roda) é só informação. Salvaguarda decisiva: soma G > 60% (palpite declarado antes dos dados).

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-3b-L-curto-frio | 23 | 1 | 12 |
| llamacpp-3b-S1-curto-frio | 3 | 36 | 6 |
| llamacpp-3b-S2-curto-frio | 0 | 7 | 6 |
| lastro-3b-L-curto-quente | 1 | 2 | 35 |
| llamacpp-3b-S1-curto-quente | 5 | 8 | 14 |
| llamacpp-3b-S2-curto-quente | 3 | 5 | 7 |
| lastro-3b-L-medio-frio | 10 | 1 | 13 |
| llamacpp-3b-S1-medio-frio | 0 | 40 | 25 |
| llamacpp-3b-S2-medio-frio | 2 | 6 | 5 |
| lastro-3b-L-medio-quente | 9 | 3 | 27 |
| llamacpp-3b-S1-medio-quente | 1 | 73 | 0 |
| llamacpp-3b-S2-medio-quente | 1 | 8 | 1 |
| lastro-3b-L-longo-frio | 5 | 2 | 26 |
| llamacpp-3b-S1-longo-frio | 8 | 37 | 24 |
| llamacpp-3b-S2-longo-frio | 3 | 2 | 5 |
| lastro-3b-L-longo-quente | 3 | 1 | 2 |
| llamacpp-3b-S1-longo-quente | 0 | 80 | 3 |
| llamacpp-3b-S2-longo-quente | 0 | 6 | 1 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## NÃO RODOU (nunca vira razão, PASSA nem estimativa)
llamacpp-3b-S3-curto-frio, llamacpp-3b-S3-curto-quente, llamacpp-3b-S3-medio-frio, llamacpp-3b-S3-medio-quente, llamacpp-3b-S3-longo-frio, llamacpp-3b-S3-longo-quente

