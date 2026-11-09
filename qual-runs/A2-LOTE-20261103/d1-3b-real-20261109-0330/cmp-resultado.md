# Degrau 1: Lastro com 100% dos experts na GPU contra llama.cpp S1 (resultado descritivo)

Critério: criterio-degrau1-100pct-20261109.md (herda criterio-comparacao-llamacpp-20261108.md, emendas 1 a 9). Sem meta e sem PASSA/REPROVA de velocidade. **Todo rótulo: Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp.** Razão = mediana L100 / mediana S1 (menor que 1 = Lastro mais rápido); prefill e decode ficam SEPARADOS e não são juntados. Dispersão > 15% = ruidosa; SEM PROVA e NÃO RODOU como no critério base (nunca viram razão).

A coluna 'Lastro 10%' da janela anterior (09/11) só entra como referência declarada; nenhuma razão nova a mistura. O S1 foi remedido NESTA janela.

## Smoke (seção 3)
- veredito: OK
- gpu_pico_pct: 35.9
- gpu_leituras_no_intervalo: 32
- cpu_needed: 0.0
- ids: IGUAIS
- slots_impressos: slots 1280 de 1280
- cmdline: /home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/target/release/primeira-palavra --gera --celula 3b:/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct --so-texto curto --fracao-slots 1.0 --saida /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/d1-3b-real-20261109-0330/smoke/p1/saida

Observação: memória do host (MT/s) não conferida pelo roteiro: declarar à parte
Janela do(s) modelo(s): 3b (uma janela por modelo).

## Orçamento e cortes
Cortes aplicados: nenhum
Repetições finais: {"quente": 5, "frio": 3}

## Ordem de execução gravada
    smoke inicio=3s fim=19s rc=0
    01 lastro-3b-L100-curto-frio n=3 inicio=19s fim=89s rc=0
    02 llamacpp-3b-S1-curto-frio n=3 inicio=89s fim=103s rc=0
    03 lastro-3b-L100-curto-quente n=5 inicio=103s fim=127s rc=0
    04 llamacpp-3b-S1-curto-quente n=5 inicio=127s fim=135s rc=0
    05 llamacpp-3b-S1-medio-frio n=3 inicio=135s fim=149s rc=0
    06 lastro-3b-L100-medio-frio n=3 inicio=149s fim=222s rc=0
    07 llamacpp-3b-S1-medio-quente n=5 inicio=222s fim=231s rc=0
    08 lastro-3b-L100-medio-quente n=5 inicio=231s fim=256s rc=0
    09 lastro-3b-L100-longo-frio n=3 inicio=256s fim=329s rc=0
    10 llamacpp-3b-S1-longo-frio n=3 inicio=329s fim=342s rc=0
    11 lastro-3b-L100-longo-quente n=5 inicio=342s fim=370s rc=0
    12 llamacpp-3b-S1-longo-quente n=5 inicio=370s fim=379s rc=0

## Lastro 100% experts GPU contra llama.cpp S1; KV f32 no Lastro contra f16 no llama.cpp

| modelo | texto | estado | n | razão prefill | razão decode | marca | cpu/needed (L100) | L100 prefill ms | S1 prefill ms | L100 decode ms/passo | S1 decode ms/passo | faixa (seção 5, decode) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3b | curto | frio | 3 | 29.611 | 1.161 | mesma sequência | 0.000 (0.000 a 0.000) | 3207.68 (3192.00 a 3227.98) | 108.33 (106.17 a 109.64) | 3.811 (3.806 a 3.826) | 3.282 (3.254 a 3.285) | decode L100/S1 <= 2 |
| 3b | curto | quente | 5 | 17.317 | 1.181 | mesma sequência | 0.000 (0.000 a 0.000) | 224.05 (223.03 a 225.24) | 12.94 (12.89 a 12.96) | 3.795 (3.776 a 3.830) | 3.213 (3.198 a 3.291) | decode L100/S1 <= 2 |
| 3b | medio | frio | 3 | 28.138 | 1.170 | mesma sequência | 0.000 (0.000 a 0.000) | 3438.72 (3428.38 a 3444.41) | 122.21 (121.31 a 125.77) | 3.823 (3.812 a 3.838) | 3.268 (3.255 a 3.341) | decode L100/S1 <= 2 |
| 3b | medio | quente | 5 | 20.261 | 1.188 | mesma sequência | 0.000 (0.000 a 0.000) | 431.98 (430.31 a 433.16) | 21.32 (21.19 a 21.45) | 3.808 (3.796 a 3.812) | 3.205 (3.194 a 3.276) | decode L100/S1 <= 2 |
| 3b | longo | frio | 3 | 29.824 | 1.144 | mesma sequência | 0.000 (0.000 a 0.000) | 3900.54 (3897.64 a 3955.21) | 130.79 (128.87 a 130.83) | 3.799 (3.794 a 3.850) | 3.320 (3.310 a 3.332) | decode L100/S1 <= 2 |
| 3b | longo | quente | 5 | 31.669 | 1.171 | mesma sequência | 0.000 (0.000 a 0.000) | 928.49 (923.24 a 930.01) | 29.32 (29.18 a 30.08) | 3.805 (3.788 a 3.829) | 3.248 (3.232 a 3.277) | decode L100/S1 <= 2 |

## Leitura (seção 5: palpites pré-registrados, orientação e NÃO decisão)
- decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2
- decode L100/S1 entre 2 e 6: há overhead de motor moderado (KV f32, sincronização, lançamentos); investigar por perfil
- decode L100/S1 > 6: o motor tem custo próprio grande além do posicionamento; investigar antes de qualquer outro degrau
- Prefill: a razão será provavelmente maior que a do decode; o relatório separa as duas e não as junta.

- 3b curto frio: razão de decode 1.161 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 3b curto quente: razão de decode 1.181 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 3b medio frio: razão de decode 1.170 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 3b medio quente: razão de decode 1.188 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 3b longo frio: razão de decode 1.144 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'
- 3b longo quente: razão de decode 1.171 cai em 'decode L100/S1 <= 2: o motor é competitivo com a GPU a 100%; o próximo passo é o degrau 2 (camadas inteiras) só se ainda se quiser igualar S2'

Limites declarados (seção 6): KV f32 contra f16; fronteira do prefill diferente (base); desktop gráfico como condição comum; memória do host declarada à parte; um Granite BF16 só; sem Strata; não responde sobre outros modelos ou quantizados.

## Razões de prefill e decode que diferem em mais de 1,5x (hipótese, nunca conclusão)
- 3b curto frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 34.5%; soma_passos_ms 2.6%; T_sequencia_ms 63.0% CANDIDATA A GARGALO
- 3b curto quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 48.4%; soma_passos_ms 51.6%; T_sequencia_ms 0.0%
- 3b medio frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 35.9%; soma_passos_ms 2.5%; T_sequencia_ms 61.6% CANDIDATA A GARGALO
- 3b medio quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 64.3%; soma_passos_ms 35.7%; T_sequencia_ms 0.0%
- 3b longo frio: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 38.9%; soma_passos_ms 2.4%; T_sequencia_ms 58.7% CANDIDATA A GARGALO
- 3b longo quente: fase de menor desempenho relativo = prefill; fatias do Lastro sobre total_ms: prefill_ms 79.4%; soma_passos_ms 20.5%; T_sequencia_ms 0.0%

## Carga do modelo (descritiva, sem razão) e tokens/s de decode

| célula | carga ms (mediana, min a max) | decode tokens/s (mediana) |
|---|---|---|
| lastro-3b-L100-curto-frio | 12702 (12690 a 12783) (LIMITE SUPERIOR) | 262.4 |
| llamacpp-3b-S1-curto-frio | 1299 (1293 a 1299) (frio sem aquecimento) | 304.7 |
| lastro-3b-L100-curto-quente | 12748 (12748 a 12748) (LIMITE SUPERIOR) | 263.5 |
| llamacpp-3b-S1-curto-quente | 1335 (1335 a 1335) (inclui aquecimento padrão) | 311.3 |
| lastro-3b-L100-medio-frio | 12768 (12681 a 14973) ruidosa (LIMITE SUPERIOR) | 261.6 |
| llamacpp-3b-S1-medio-frio | 1301 (1295 a 1312) (frio sem aquecimento) | 306.0 |
| lastro-3b-L100-medio-quente | 12751 (12751 a 12751) (LIMITE SUPERIOR) | 262.6 |
| llamacpp-3b-S1-medio-quente | 1332 (1332 a 1332) (inclui aquecimento padrão) | 312.0 |
| lastro-3b-L100-longo-frio | 12692 (12514 a 12695) (LIMITE SUPERIOR) | 263.3 |
| llamacpp-3b-S1-longo-frio | 1311 (1307 a 1313) (frio sem aquecimento) | 301.2 |
| lastro-3b-L100-longo-quente | 12711 (12711 a 12711) (LIMITE SUPERIOR) | 262.8 |
| llamacpp-3b-S1-longo-quente | 1336 (1336 a 1336) (inclui aquecimento padrão) | 307.9 |

## Condição do desktop (emendas 8 e 9)
O desktop gráfico (PIDs G) é condição comum dos dois lados; processo de compute (C ou C+G) fora da árvore medida invalida a célula. Pico do pmon é amostra esparsa, não fração da placa.

| célula | pico do pmon (soma G) antes | pico do pmon (soma G) depois | dmon sm% máx antes (informação) |
|---|---|---|---|
| lastro-3b-L100-curto-frio | 1 | 9 | 4 |
| llamacpp-3b-S1-curto-frio | 1 | 75 | 4 |
| lastro-3b-L100-curto-quente | 0 | 12 | 2 |
| llamacpp-3b-S1-curto-quente | 0 | 28 | 0 |
| lastro-3b-L100-medio-frio | 0 | 7 | 0 |
| llamacpp-3b-S1-medio-frio | 0 | 2 | 1 |
| lastro-3b-L100-medio-quente | 1 | 7 | 23 |
| llamacpp-3b-S1-medio-quente | 0 | 20 | 18 |
| lastro-3b-L100-longo-frio | 0 | 11 | 15 |
| llamacpp-3b-S1-longo-frio | 0 | 47 | 0 |
| lastro-3b-L100-longo-quente | 0 | 7 | 20 |
| llamacpp-3b-S1-longo-quente | 0 | 79 | 4 |

## SEM PROVA (sem razão publicada)
Nenhuma célula.

## `slots` impressos pelo binário (L100)
- lastro-3b-L100-curto-frio: p0: slots 1280 de 1280 | p1: slots 1280 de 1280 | p2: slots 1280 de 1280
- lastro-3b-L100-curto-quente: p1: slots 1280 de 1280
- lastro-3b-L100-medio-frio: p0: slots 1280 de 1280 | p1: slots 1280 de 1280 | p2: slots 1280 de 1280
- lastro-3b-L100-medio-quente: p1: slots 1280 de 1280
- lastro-3b-L100-longo-frio: p0: slots 1280 de 1280 | p1: slots 1280 de 1280 | p2: slots 1280 de 1280
- lastro-3b-L100-longo-quente: p1: slots 1280 de 1280

## NÃO RODOU (nunca vira razão nem estimativa)
Nenhuma célula.

