# Comparação Lastro x llama.cpp (opção d): resultado descritivo

Critério: criterio-comparacao-llamacpp-20261108.md (emendas 1 a 6). Sem meta e sem PASSA/REPROVA de velocidade. **Toda razão é Lastro com 10% dos experts na GPU contra a série indicada do llama.cpp; nenhuma razão isola o motor.** Razão = mediana Lastro / mediana llama.cpp (menor que 1 = Lastro mais rápido), por série, nunca misturadas.

Declarações: KV f16 no llama.cpp contra f32 no Lastro (S1, S2); a ponte de correção é a igualdade dos ids gerados. Fronteira do prefill: `prompt_ms` do llama.cpp termina depois de amostrar o 1º token, `prefill_ms` do Lastro antes do argmax (sem ajuste). Decode: média por passo (63 passos nos dois lados). Quente e frio com o cache de página do GGUF/safetensors quente (o 1º contato com o disco não entra). Carga do Lastro é LIMITE SUPERIOR (inclui a cauda de encerramento).

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 1b (emenda 6: uma janela por modelo; o outro modelo é outra janela e não aparece aqui).

## Orçamento (E8) e cortes
Cortes aplicados: nenhum
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    01 lastro-1b-L-curto-frio n=3 inicio=4s fim=54s rc=0
    02 llamacpp-1b-S1-curto-frio n=3 inicio=54s fim=65s rc=0
    03 llamacpp-1b-S2-curto-frio n=3 inicio=65s fim=81s rc=0
    04 lastro-1b-L-curto-quente n=5 inicio=81s fim=108s rc=0
    05 llamacpp-1b-S1-curto-quente n=5 inicio=108s fim=115s rc=0
    06 llamacpp-1b-S2-curto-quente n=5 inicio=115s fim=132s rc=0
    07 llamacpp-1b-S1-medio-frio n=3 inicio=132s fim=143s rc=0
    08 llamacpp-1b-S2-medio-frio n=3 inicio=143s fim=159s rc=0
    09 lastro-1b-L-medio-frio n=3 inicio=159s fim=209s rc=0
    10 llamacpp-1b-S1-medio-quente n=5 inicio=209s fim=217s rc=0
    11 llamacpp-1b-S2-medio-quente n=5 inicio=217s fim=233s rc=0
    12 lastro-1b-L-medio-quente n=5 inicio=233s fim=264s rc=0
    13 lastro-1b-L-longo-frio n=3 inicio=264s fim=321s rc=0
    14 llamacpp-1b-S1-longo-frio n=3 inicio=321s fim=332s rc=0
    15 llamacpp-1b-S2-longo-frio n=3 inicio=332s fim=348s rc=0
    16 lastro-1b-L-longo-quente n=5 inicio=348s fim=389s rc=0
    17 llamacpp-1b-S1-longo-quente n=5 inicio=389s fim=396s rc=0
    18 llamacpp-1b-S2-longo-quente n=5 inicio=396s fim=413s rc=0
    19 llamacpp-1b-S3-curto-frio n=3 inicio=413s fim=424s rc=0
    20 llamacpp-1b-S3-curto-quente n=5 inicio=424s fim=431s rc=0
    21 llamacpp-1b-S3-medio-frio n=3 inicio=431s fim=442s rc=0
    22 llamacpp-1b-S3-medio-quente n=5 inicio=442s fim=450s rc=0
    23 llamacpp-1b-S3-longo-frio n=3 inicio=450s fim=461s rc=0
    24 llamacpp-1b-S3-longo-quente n=5 inicio=461s fim=469s rc=0

## Lastro 10% experts GPU contra llama.cpp S1 (tudo na GPU, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro)

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 32.959 (ruidosa) | 12.508 | mesma sequência; ruidosa | 0.890 (0.890 a 0.890) | 3280.73 (3232.06 a 5403.28) ruidosa | 99.54 (98.22 a 101.00) | 22.801 (21.955 a 23.208) | 1.823 (1.808 a 1.865) |
| 1b | curto | quente | 5 | 109.019 | 11.758 | mesma sequência | 0.890 (0.890 a 0.890) | 794.64 (764.74 a 810.48) | 7.29 (7.02 a 8.11) | 21.262 (20.955 a 23.090) | 1.808 (1.776 a 1.849) |
| 1b | medio | frio | 3 | 38.791 | 11.964 | mesma sequência | 0.894 (0.894 a 0.894) | 3967.38 (3946.00 a 4052.68) | 102.28 (100.51 a 103.50) | 21.531 (21.519 a 24.084) | 1.800 (1.795 a 1.824) |
| 1b | medio | quente | 5 | 173.644 | 12.326 | mesma sequência | 0.894 (0.894 a 0.894) | 1523.90 (1505.88 a 1546.13) | 8.78 (8.75 a 9.63) | 21.649 (21.251 a 22.428) | 1.756 (1.750 a 1.773) |
| 1b | longo | frio | 3 | 53.158 | 11.773 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 5608.85 (5565.65 a 5650.29) | 105.51 (103.41 a 108.99) | 21.262 (21.073 a 21.644) | 1.806 (1.797 a 1.812) |
| 1b | longo | quente | 5 | 239.218 (ruidosa) | 12.107 (sequências diferentes) | sequências diferentes; ruidosa | 0.899 (0.899 a 0.899) | 3109.12 (3081.55 a 3140.08) | 13.00 (12.74 a 14.94) ruidosa | 21.540 (20.976 a 22.679) | 1.779 (1.750 a 1.882) |

## Lastro 10% experts GPU contra llama.cpp S2 (--n-cpu-moe 22 no 1B e 29 no 3B, -ngl 99, -fa on, KV f16 no llama.cpp contra f32 no Lastro; descarregamento de ordem de grandeza parecida, NÃO o mesmo: o Lastro descarrega ~90% dos experts de cada camada, o llama.cpp camadas inteiras; op offload ligado, então o prefill da S2 não deve ser lido como 'experts calculados na CPU')

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 10.179 (ruidosa) | 1.056 | mesma sequência; ruidosa | 0.890 (0.890 a 0.890) | 3280.73 (3232.06 a 5403.28) ruidosa | 322.29 (319.88 a 333.80) | 22.801 (21.955 a 23.208) | 21.587 (21.555 a 22.040) |
| 1b | curto | quente | 5 | 3.524 (ruidosa) | 0.988 | mesma sequência; ruidosa | 0.890 (0.890 a 0.890) | 794.64 (764.74 a 810.48) | 225.48 (222.98 a 269.36) ruidosa | 21.262 (20.955 a 23.090) | 21.526 (21.397 a 21.702) |
| 1b | medio | frio | 3 | 12.248 | 0.997 | mesma sequência | 0.894 (0.894 a 0.894) | 3967.38 (3946.00 a 4052.68) | 323.93 (320.50 a 329.79) | 21.531 (21.519 a 24.084) | 21.586 (21.533 a 21.678) |
| 1b | medio | quente | 5 | 6.393 | 1.006 | mesma sequência | 0.894 (0.894 a 0.894) | 1523.90 (1505.88 a 1546.13) | 238.37 (227.33 a 244.94) | 21.649 (21.251 a 22.428) | 21.514 (21.423 a 22.480) |
| 1b | longo | frio | 3 | 17.239 | 1.002 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 5608.85 (5565.65 a 5650.29) | 325.35 (322.71 a 338.71) | 21.262 (21.073 a 21.644) | 21.228 (21.100 a 21.507) |
| 1b | longo | quente | 5 | 12.796 | 1.015 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 3109.12 (3081.55 a 3140.08) | 242.98 (227.97 a 254.40) | 21.540 (20.976 a 22.679) | 21.216 (21.147 a 21.492) |

S2 é aproximação declarada (camadas inteiras no llama.cpp, ~90% dos experts de cada camada no Lastro); op offload ligado: o prefill da S2 não deve ser lido como 'experts calculados na CPU'.

## Lastro 10% experts GPU contra llama.cpp S3 (KV f32 com -fa off, paridade de armazenamento; configuração prática, sem atribuir a razão só ao motor)

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (Lastro) | Lastro prefill ms | llama.cpp prefill ms | Lastro decode ms/passo | llama.cpp decode ms/passo |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1b | curto | frio | 3 | 33.900 (ruidosa) | 11.994 | mesma sequência; ruidosa | 0.890 (0.890 a 0.890) | 3280.73 (3232.06 a 5403.28) ruidosa | 96.78 (95.06 a 98.55) | 22.801 (21.955 a 23.208) | 1.901 (1.900 a 1.915) |
| 1b | curto | quente | 5 | 108.262 | 11.263 | mesma sequência | 0.890 (0.890 a 0.890) | 794.64 (764.74 a 810.48) | 7.34 (7.31 a 7.63) | 21.262 (20.955 a 23.090) | 1.888 (1.856 a 1.923) |
| 1b | medio | frio | 3 | 39.640 | 11.329 | mesma sequência | 0.894 (0.894 a 0.894) | 3967.38 (3946.00 a 4052.68) | 100.08 (98.70 a 102.36) | 21.531 (21.519 a 24.084) | 1.900 (1.881 a 1.905) |
| 1b | medio | quente | 5 | 165.912 (ruidosa) | 11.575 | mesma sequência; ruidosa | 0.894 (0.894 a 0.894) | 1523.90 (1505.88 a 1546.13) | 9.19 (9.05 a 11.97) ruidosa | 21.649 (21.251 a 22.428) | 1.870 (1.855 a 1.888) |
| 1b | longo | frio | 3 | 53.234 | 10.825 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 5608.85 (5565.65 a 5650.29) | 105.36 (102.00 a 110.69) | 21.262 (21.073 a 21.644) | 1.964 (1.940 a 2.006) |
| 1b | longo | quente | 5 | 222.972 | 11.199 (sequências diferentes) | sequências diferentes | 0.899 (0.899 a 0.899) | 3109.12 (3081.55 a 3140.08) | 13.94 (13.87 a 14.39) | 21.540 (20.976 a 22.679) | 1.923 (1.917 a 1.936) |

## Leitura da diferença (E7, hipótese e nunca conclusão)
- 1b curto frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 43.4%; soma_passos_ms 18.6%; T_sequencia_ms 38.0% CANDIDATA A GARGALO; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 43.4%; soma_passos_ms 18.6%; T_sequencia_ms 38.0% CANDIDATA A GARGALO; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto frio, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 43.4%; soma_passos_ms 18.6%; T_sequencia_ms 38.0% CANDIDATA A GARGALO; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.1%; soma_passos_ms 62.9%; T_sequencia_ms 0.0%; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.1%; soma_passos_ms 62.9%; T_sequencia_ms 0.0%; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b curto quente, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 37.1%; soma_passos_ms 62.9%; T_sequencia_ms 0.0%; cpu/needed 0.890 (0.890 a 0.890); hits/needed 0.110 (0.110 a 0.110)
- 1b medio frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 48.5%; soma_passos_ms 16.7%; T_sequencia_ms 34.5% CANDIDATA A GARGALO; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 48.5%; soma_passos_ms 16.7%; T_sequencia_ms 34.5% CANDIDATA A GARGALO; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio frio, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 48.5%; soma_passos_ms 16.7%; T_sequencia_ms 34.5% CANDIDATA A GARGALO; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 52.8%; soma_passos_ms 47.2%; T_sequencia_ms 0.0%; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 52.8%; soma_passos_ms 47.2%; T_sequencia_ms 0.0%; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b medio quente, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 52.8%; soma_passos_ms 47.2%; T_sequencia_ms 0.0%; cpu/needed 0.894 (0.894 a 0.894); hits/needed 0.106 (0.106 a 0.106)
- 1b longo frio, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 57.4%; soma_passos_ms 13.7%; T_sequencia_ms 28.7%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo frio, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 57.4%; soma_passos_ms 13.7%; T_sequencia_ms 28.7%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo frio, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 57.4%; soma_passos_ms 13.7%; T_sequencia_ms 28.7%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo quente, S1: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 69.7%; soma_passos_ms 30.3%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo quente, S2: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 69.7%; soma_passos_ms 30.3%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)
- 1b longo quente, S3: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 69.7%; soma_passos_ms 30.3%; T_sequencia_ms 0.0%; cpu/needed 0.899 (0.899 a 0.899); hits/needed 0.101 (0.101 a 0.101)

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-1b-L-curto-frio | 6000 (5590 a 6091) (LIMITE SUPERIOR) | 43.9 |
| llamacpp-1b-S1-curto-frio | 845 (842 a 846) (frio sem aquecimento) | 548.6 |
| llamacpp-1b-S2-curto-frio | 635 (632 a 636) (frio sem aquecimento) | 46.3 |
| llamacpp-1b-S3-curto-frio | 844 (844 a 848) (frio sem aquecimento) | 526.0 |
| lastro-1b-L-curto-quente | 6014 (6014 a 6014) (LIMITE SUPERIOR) | 47.0 |
| llamacpp-1b-S1-curto-quente | 846 (846 a 846) (inclui aquecimento padrão) | 553.0 |
| llamacpp-1b-S2-curto-quente | 636 (636 a 636) (inclui aquecimento padrão) | 46.5 |
| llamacpp-1b-S3-curto-quente | 845 (845 a 845) (inclui aquecimento padrão) | 529.7 |
| lastro-1b-L-medio-frio | 5978 (5459 a 6064) (LIMITE SUPERIOR) | 46.4 |
| llamacpp-1b-S1-medio-frio | 849 (845 a 851) (frio sem aquecimento) | 555.7 |
| llamacpp-1b-S2-medio-frio | 632 (632 a 633) (frio sem aquecimento) | 46.3 |
| llamacpp-1b-S3-medio-frio | 842 (840 a 844) (frio sem aquecimento) | 526.2 |
| lastro-1b-L-medio-quente | 5991 (5991 a 5991) (LIMITE SUPERIOR) | 46.2 |
| llamacpp-1b-S1-medio-quente | 862 (862 a 862) (inclui aquecimento padrão) | 569.4 |
| llamacpp-1b-S2-medio-quente | 632 (632 a 632) (inclui aquecimento padrão) | 46.5 |
| llamacpp-1b-S3-medio-quente | 847 (847 a 847) (inclui aquecimento padrão) | 534.7 |
| lastro-1b-L-longo-frio | 5953 (5916 a 5985) (LIMITE SUPERIOR) | 47.0 |
| llamacpp-1b-S1-longo-frio | 843 (843 a 844) (frio sem aquecimento) | 553.7 |
| llamacpp-1b-S2-longo-frio | 635 (630 a 636) (frio sem aquecimento) | 47.1 |
| llamacpp-1b-S3-longo-frio | 848 (841 a 849) (frio sem aquecimento) | 509.1 |
| lastro-1b-L-longo-quente | 5821 (5821 a 5821) (LIMITE SUPERIOR) | 46.4 |
| llamacpp-1b-S1-longo-quente | 847 (847 a 847) (inclui aquecimento padrão) | 562.0 |
| llamacpp-1b-S2-longo-quente | 635 (635 a 635) (inclui aquecimento padrão) | 47.1 |
| llamacpp-1b-S3-longo-quente | 847 (847 a 847) (inclui aquecimento padrão) | 519.9 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs de tipo G: compositor Hyprland, quickshell, Xwayland...) divide a 4090 nos DOIS lados em todas as células: é condição COMUM dos dois lados, fora do SEM PROVA; o efeito pode não ser igual entre eles (o Lastro serve os experts na CPU, a S1 do llama.cpp roda tudo na GPU, a S2 tem parte na CPU). Processos de compute (tipo C ou C+G, como voxtype e o processo de GPU do Chrome) fora da árvore medida NÃO são isentos: qualquer um, ocioso ou não, é SEM PROVA. A coluna do pico do pmon é a soma de sm% dos PIDs de tipo G entre as 3 capturas de uma volta (`nvidia-smi pmon -c 1 -s um`, antes e depois de cada célula): é PICO DE AMOSTRA ESPARSA, não a fração da placa usada pelo desktop; `-` = sem leitura. A coluna do dmon (`dmon -s u -c 3`, só antes da célula, quando nada medido roda) é só informação. Salvaguarda decisiva: soma G > 60% (palpite declarado antes dos dados).

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-1b-L-curto-frio | 18 | 2 | 45 |
| llamacpp-1b-S1-curto-frio | 5 | 8 | 23 |
| llamacpp-1b-S2-curto-frio | 1 | 7 | 5 |
| llamacpp-1b-S3-curto-frio | 6 | 29 | 35 |
| lastro-1b-L-curto-quente | 11 | 1 | 69 |
| llamacpp-1b-S1-curto-quente | 3 | 62 | 64 |
| llamacpp-1b-S2-curto-quente | 1 | 9 | 0 |
| llamacpp-1b-S3-curto-quente | 2 | 11 | 15 |
| lastro-1b-L-medio-frio | 0 | 3 | 1 |
| llamacpp-1b-S1-medio-frio | 7 | 18 | 0 |
| llamacpp-1b-S2-medio-frio | 0 | 4 | 1 |
| llamacpp-1b-S3-medio-frio | 1 | 20 | 7 |
| lastro-1b-L-medio-quente | 5 | 3 | 2 |
| llamacpp-1b-S1-medio-quente | 5 | 25 | 15 |
| llamacpp-1b-S2-medio-quente | 2 | 4 | 5 |
| llamacpp-1b-S3-medio-quente | 1 | 48 | 12 |
| lastro-1b-L-longo-frio | 10 | 0 | 40 |
| llamacpp-1b-S1-longo-frio | 3 | 7 | 3 |
| llamacpp-1b-S2-longo-frio | 1 | 3 | 6 |
| llamacpp-1b-S3-longo-frio | 0 | 32 | 14 |
| lastro-1b-L-longo-quente | 2 | 2 | 6 |
| llamacpp-1b-S1-longo-quente | 1 | 33 | 14 |
| llamacpp-1b-S2-longo-quente | 2 | 7 | 8 |
| llamacpp-1b-S3-longo-quente | 1 | 3 | 23 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## NÃO RODOU (nunca vira razão, PASSA nem estimativa)
Nenhuma célula.

