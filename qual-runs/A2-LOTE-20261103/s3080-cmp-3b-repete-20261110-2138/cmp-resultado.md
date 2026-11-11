# Comparação Lastro x llama.cpp (opção d): resultado descritivo

Critério: criterio-comparacao-llamacpp-20261108.md (emendas 1 a 6). Sem meta e sem PASSA/REPROVA de velocidade. **Toda razão é Lastro com 10% dos experts na GPU contra a série indicada do llama.cpp; nenhuma razão isola o motor.** Razão = mediana Lastro / mediana llama.cpp (menor que 1 = Lastro mais rápido), por série, nunca misturadas.

Declarações: KV f16 no llama.cpp contra f32 no Lastro (S1, S2); a ponte de correção é a igualdade dos ids gerados. Fronteira do prefill: `prompt_ms` do llama.cpp termina depois de amostrar o 1º token, `prefill_ms` do Lastro antes do argmax (sem ajuste). Decode: média por passo (63 passos nos dois lados). Quente e frio com o cache de página do GGUF/safetensors quente (o 1º contato com o disco não entra). Carga do Lastro é LIMITE SUPERIOR (inclui a cauda de encerramento).

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 3b (emenda 6: uma janela por modelo; o outro modelo é outra janela e não aparece aqui).

## Orçamento (E8) e cortes
Cortes aplicados: (1) S3 inteira não roda
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    01 lastro-3b-L-curto-frio n=3 inicio=3s fim=94s rc=0
    02 llamacpp-3b-S1-curto-frio n=3 inicio=94s fim=109s rc=0
    03 llamacpp-3b-S2-curto-frio n=3 inicio=109s fim=132s rc=0
    04 lastro-3b-L-curto-quente n=5 inicio=132s fim=178s rc=0
    05 llamacpp-3b-S1-curto-quente n=5 inicio=178s fim=187s rc=0
    06 llamacpp-3b-S2-curto-quente n=5 inicio=187s fim=213s rc=0
    07 llamacpp-3b-S1-medio-frio n=3 inicio=213s fim=227s rc=0
    08 llamacpp-3b-S2-medio-frio n=3 inicio=227s fim=251s rc=0
    09 lastro-3b-L-medio-frio n=3 inicio=251s fim=342s rc=0
    10 llamacpp-3b-S1-medio-quente n=5 inicio=342s fim=351s rc=0
    11 llamacpp-3b-S2-medio-quente n=5 inicio=351s fim=378s rc=0
    12 lastro-3b-L-medio-quente n=5 inicio=378s fim=434s rc=0
    13 lastro-3b-L-longo-frio n=3 inicio=434s fim=538s rc=0
    14 llamacpp-3b-S1-longo-frio n=3 inicio=538s fim=552s rc=0
    15 llamacpp-3b-S2-longo-frio n=3 inicio=552s fim=576s rc=0
    16 lastro-3b-L-longo-quente n=5 inicio=576s fim=652s rc=0
    17 llamacpp-3b-S1-longo-quente n=5 inicio=652s fim=661s rc=0
    18 llamacpp-3b-S2-longo-quente n=5 inicio=661s fim=688s rc=0

## Lastro 10% experts GPU contra llama.cpp S1 (tudo na GPU, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro)

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3b | curto | frio | 3 | 46.928 | 10.638 | mesma sequência | 0.899 (0.899 a 0.899) | 4545.64 (4435.00 a 4562.01) | 96.86 (96.36 a 97.03) | 41.802 (41.498 a 42.389) | 3.930 (3.929 a 3.932) |
| 3b | curto | quente | 5 | 84.488 | 10.616 | mesma sequência | 0.899 (0.899 a 0.899) | 1634.42 (1618.39 a 1656.59) | 19.34 (19.18 a 20.06) | 41.400 (41.149 a 42.268) | 3.900 (3.895 a 3.939) |
| 3b | medio | frio | 3 | 52.167 | 10.556 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 6401.62 (5964.37 a 6617.70) | 122.71 (122.22 a 124.77) | 41.471 (41.466 a 42.163) | 3.928 (3.915 a 3.944) |
| 3b | medio | quente | 5 | 83.206 | 10.733 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 3104.01 (3095.45 a 3214.02) | 37.30 (37.16 a 37.52) | 41.633 (41.218 a 42.381) | 3.879 (3.877 a 3.885) |
| 3b | longo | frio | 3 | 67.499 | 10.744 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 9557.67 (9533.26 a 9558.50) | 141.60 (141.10 a 142.60) | 42.499 (41.967 a 42.671) | 3.956 (3.950 a 3.961) |
| 3b | longo | quente | 5 | 119.235 | 10.561 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 6607.19 (6550.99 a 6612.66) | 55.41 (55.27 a 55.53) | 41.375 (41.125 a 42.002) | 3.918 (3.916 a 3.924) |

## Lastro 10% experts GPU contra llama.cpp S2 (--n-cpu-moe 22 no 1B e 29 no 3B, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro; descarregamento de ordem de grandeza parecida, NÃO o mesmo: o Lastro descarrega ~90% dos experts de cada camada, o llama.cpp camadas inteiras; op offload ligado, então o prefill da S2 não deve ser lido como 'experts calculados na CPU')

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 3b | curto | frio | 3 | 7.116 | 0.979 | mesma sequência | 0.899 (0.899 a 0.899) | 4545.64 (4435.00 a 4562.01) | 638.79 (636.95 a 642.98) | 41.802 (41.498 a 42.389) | 42.686 (42.156 a 43.398) |
| 3b | curto | quente | 5 | 2.912 | 0.979 | mesma sequência | 0.899 (0.899 a 0.899) | 1634.42 (1618.39 a 1656.59) | 561.28 (545.58 a 575.25) | 41.400 (41.149 a 42.268) | 42.292 (42.164 a 42.701) |
| 3b | medio | frio | 3 | 9.444 | 0.966 | mesma sequência | 0.898 (0.898 a 0.898) | 6401.62 (5964.37 a 6617.70) | 677.82 (668.65 a 679.12) | 41.471 (41.466 a 42.163) | 42.940 (42.631 a 43.176) |
| 3b | medio | quente | 5 | 5.229 | 0.976 | mesma sequência | 0.898 (0.898 a 0.898) | 3104.01 (3095.45 a 3214.02) | 593.67 (588.46 a 606.80) | 41.633 (41.218 a 42.381) | 42.675 (42.366 a 43.822) |
| 3b | longo | frio | 3 | 13.782 | 1.001 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 9557.67 (9533.26 a 9558.50) | 693.49 (691.29 a 703.96) | 42.499 (41.967 a 42.671) | 42.458 (42.385 a 43.401) |
| 3b | longo | quente | 5 | 10.841 | 0.968 (sequências diferentes) | sequências diferentes | 0.898 (0.898 a 0.898) | 6607.19 (6550.99 a 6612.66) | 609.47 (593.30 a 628.27) | 41.375 (41.125 a 42.002) | 42.762 (42.468 a 43.506) |

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
- 3b curto frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 34.1%; soma_passos_ms 20.2%; T_sequencia_ms 45.6% CANDIDATA A GARGALO; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b curto frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 34.1%; soma_passos_ms 20.2%; T_sequencia_ms 45.6% CANDIDATA A GARGALO; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b curto quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 38.4%; soma_passos_ms 61.6%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b curto quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 38.4%; soma_passos_ms 61.6%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 3b medio frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 41.5%; soma_passos_ms 17.1%; T_sequencia_ms 41.0% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b medio frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 41.5%; soma_passos_ms 17.1%; T_sequencia_ms 41.0% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b medio quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 54.4%; soma_passos_ms 45.6%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b medio quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 54.4%; soma_passos_ms 45.6%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 52.4%; soma_passos_ms 14.7%; T_sequencia_ms 33.0% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 52.4%; soma_passos_ms 14.7%; T_sequencia_ms 33.0% CANDIDATA A GARGALO; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 71.7%; soma_passos_ms 28.3%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)
- 3b longo quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 71.7%; soma_passos_ms 28.3%; T_sequencia_ms 0.0%; cpu/needed 0.898 (0.898 a 0.898); hits/needed 0.102 (0.102 a 0.102)

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-3b-L-curto-frio | 12493 (12428 a 13013) (LIMITE SUPERIOR) | 23.9 |
| llamacpp-3b-S1-curto-frio | 1275 (1272 a 1278) (frio sem aquecimento) | 254.5 |
| llamacpp-3b-S2-curto-frio | 884 (884 a 886) (frio sem aquecimento) | 23.4 |
| lastro-3b-L-curto-quente | 12391 (12391 a 12391) (LIMITE SUPERIOR) | 24.2 |
| llamacpp-3b-S1-curto-quente | 1290 (1290 a 1290) (inclui aquecimento padrão) | 256.4 |
| llamacpp-3b-S2-curto-quente | 899 (899 a 899) (inclui aquecimento padrão) | 23.6 |
| lastro-3b-L-medio-frio | 12452 (12420 a 12469) (LIMITE SUPERIOR) | 24.1 |
| llamacpp-3b-S1-medio-frio | 1272 (1270 a 1286) (frio sem aquecimento) | 254.6 |
| llamacpp-3b-S2-medio-frio | 892 (889 a 898) (frio sem aquecimento) | 23.3 |
| lastro-3b-L-medio-quente | 12583 (12583 a 12583) (LIMITE SUPERIOR) | 24.0 |
| llamacpp-3b-S1-medio-quente | 1297 (1297 a 1297) (inclui aquecimento padrão) | 257.8 |
| llamacpp-3b-S2-medio-quente | 908 (908 a 908) (inclui aquecimento padrão) | 23.4 |
| lastro-3b-L-longo-frio | 12510 (12476 a 12643) (LIMITE SUPERIOR) | 23.5 |
| llamacpp-3b-S1-longo-frio | 1485 (1281 a 1492) (frio sem aquecimento) | 252.8 |
| llamacpp-3b-S2-longo-frio | 900 (892 a 907) (frio sem aquecimento) | 23.6 |
| lastro-3b-L-longo-quente | 12400 (12400 a 12400) (LIMITE SUPERIOR) | 24.2 |
| llamacpp-3b-S1-longo-quente | 1301 (1301 a 1301) (inclui aquecimento padrão) | 255.2 |
| llamacpp-3b-S2-longo-quente | 908 (908 a 908) (inclui aquecimento padrão) | 23.4 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs de tipo G: compositor Hyprland, quickshell, Xwayland...) divide a placa de medida nos DOIS lados em todas as células: é condição COMUM dos dois lados, fora do SEM PROVA; o efeito pode não ser igual entre eles (o Lastro serve os experts na CPU, a S1 do llama.cpp roda tudo na GPU, a S2 tem parte na CPU). Processos de compute (tipo C ou C+G, como voxtype e o processo de GPU do Chrome) fora da árvore medida NÃO são isentos: qualquer um, ocioso ou não, é SEM PROVA. A coluna do pico do pmon é a soma de sm% dos PIDs de tipo G entre as 3 capturas de uma volta (`nvidia-smi pmon -c 1 -s um`, antes e depois de cada célula): é PICO DE AMOSTRA ESPARSA, não a fração da placa usada pelo desktop; `-` = sem leitura. A coluna do dmon (`dmon -s u -c 3`, só antes da célula, quando nada medido roda) é só informação. Salvaguarda decisiva: soma G > 60% (palpite declarado antes dos dados).

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-3b-L-curto-frio | 0 | 0 | 0 |
| llamacpp-3b-S1-curto-frio | 0 | 0 | 0 |
| llamacpp-3b-S2-curto-frio | 0 | 0 | 0 |
| lastro-3b-L-curto-quente | 0 | 0 | 0 |
| llamacpp-3b-S1-curto-quente | 0 | 0 | 0 |
| llamacpp-3b-S2-curto-quente | 0 | 0 | 0 |
| lastro-3b-L-medio-frio | 0 | 0 | 0 |
| llamacpp-3b-S1-medio-frio | 0 | 0 | 0 |
| llamacpp-3b-S2-medio-frio | 0 | 0 | 0 |
| lastro-3b-L-medio-quente | 0 | 0 | 0 |
| llamacpp-3b-S1-medio-quente | 0 | 0 | 0 |
| llamacpp-3b-S2-medio-quente | 0 | 0 | 0 |
| lastro-3b-L-longo-frio | 0 | 0 | 0 |
| llamacpp-3b-S1-longo-frio | 0 | 0 | 0 |
| llamacpp-3b-S2-longo-frio | 0 | 0 | 0 |
| lastro-3b-L-longo-quente | 0 | 0 | 0 |
| llamacpp-3b-S1-longo-quente | 0 | 0 | 0 |
| llamacpp-3b-S2-longo-quente | 0 | 0 | 0 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## NÃO RODOU (nunca vira razão, PASSA nem estimativa)
llamacpp-3b-S3-curto-frio, llamacpp-3b-S3-curto-quente, llamacpp-3b-S3-medio-frio, llamacpp-3b-S3-medio-quente, llamacpp-3b-S3-longo-frio, llamacpp-3b-S3-longo-quente

