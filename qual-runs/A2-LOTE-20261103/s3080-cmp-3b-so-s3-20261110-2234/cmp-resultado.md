# S3 do llama.cpp sozinha (modo --so-s3, Emenda 12): medidas descritivas, SEM razão

Critério: criterio-comparacao-llamacpp-20261108.md, Emenda 12 (emendas 1 a 12). Janela --so-s3 (Emenda 12): só a S3 do llama.cpp (KV f32, -fa off); sem Lastro na janela, logo SEM razão; medidas descritivas do llama.cpp, sem meta e sem PASSA/REPROVA.

Esta janela NÃO tem células do Lastro: nenhuma razão Lastro/llama.cpp é calculada ou publicada, nem como NÃO RODOU, porque o critério exige o Lastro na MESMA janela. Os números abaixo são só do llama.cpp S3 (KV f32, -fa off), descritivos; não são comparáveis com razões de outras janelas nem entram em razão nova. Prefill = `prompt_ms` (termina depois de amostrar o 1º token); decode = média por passo (63 passos).

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte

## Orçamento e cortes
Cortes aplicados: nenhum
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    01 llamacpp-3b-S3-curto-frio n=3 inicio=6s fim=19s rc=0
    02 llamacpp-3b-S3-curto-quente n=5 inicio=19s fim=28s rc=0
    03 llamacpp-3b-S3-medio-frio n=3 inicio=28s fim=41s rc=0
    04 llamacpp-3b-S3-medio-quente n=5 inicio=41s fim=50s rc=0
    05 llamacpp-3b-S3-longo-frio n=3 inicio=50s fim=64s rc=0
    06 llamacpp-3b-S3-longo-quente n=5 inicio=64s fim=73s rc=0

## llama.cpp S3, medidas descritivas

| célula | status | n | prefill ms (mediana, min a max) | decode ms/passo | decode tokens/s | carga ms |
|---|---|---|---|---|---|---|
| llamacpp-3b-S3-curto-frio | OK | 3 | 90.8 (90.4 a 93.2) | 4.10 (4.10 a 4.12) | 244.1 | 1262 (1258 a 1267) |
| llamacpp-3b-S3-curto-quente | OK | 5 | 19.9 (19.8 a 20.1) | 4.08 (4.08 a 4.09) | 245.0 | 1294 (1294 a 1294) |
| llamacpp-3b-S3-medio-frio | OK | 3 | 116.3 (116.0 a 118.1) | 4.10 (4.10 a 4.11) | 243.7 | 1274 (1259 a 1284) |
| llamacpp-3b-S3-medio-quente | OK | 5 | 38.2 (38.0 a 39.9) | 4.07 (4.07 a 4.09) | 245.6 | 1270 (1270 a 1270) |
| llamacpp-3b-S3-longo-frio | OK | 3 | 137.6 (136.3 a 139.1) | 4.33 (4.32 a 4.34) | 231.0 | 1259 (1254 a 1259) |
| llamacpp-3b-S3-longo-quente | OK | 5 | 59.5 (59.3 a 59.8) | 4.29 (4.29 a 4.30) | 233.0 | 1304 (1304 a 1304) |

## SEM PROVA (sem número publicado)
Nenhuma célula.

## NÃO RODOU (nunca vira número nem estimativa)
Nenhuma célula.

