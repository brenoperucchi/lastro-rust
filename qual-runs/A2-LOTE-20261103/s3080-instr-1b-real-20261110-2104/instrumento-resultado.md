# Instrumento de prefill do Lastro, janela do modelo 1B (criterio-instrumento-prefill-20261110.md)

Descritivo: sem veredito geral e sem PASSA. H1 a H6 são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA'); nada prova causa; o tempo com instrumento não é o tempo de produção; H7 não é avaliada nesta unidade.

## Série L10

Controle do custo do instrumento: **comprovada**
- ordem efetiva do controle (ordem.txt): confere com ordem.txt (D L L D D L, rc 0)
- desvio de prefill_ms (L menos D, sobre D, com sinal): 0.6%; pior caso (maior L menos menor D, sobre menor D): 2.0%; desvio do passo de decode: 0.2%; mediana D 1537.53 ms, mediana L 1547.19 ms
- desvio em ms por bloco: 0.439 ms (diferença das medianas / 22 blocos do texto médio)
  - 01 D ctrl-1b-L10-medio-01-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 1537.53 ms, passo 20.89 ms
  - 02 L ctrl-1b-L10-medio-02-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 1539.30 ms, passo 20.78 ms
  - 03 L ctrl-1b-L10-medio-03-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 1561.14 ms, passo 20.94 ms
  - 04 D ctrl-1b-L10-medio-04-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 1548.79 ms, passo 20.81 ms
  - 05 D ctrl-1b-L10-medio-05-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 1530.90 ms, passo 20.98 ms
  - 06 L ctrl-1b-L10-medio-06-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 1547.19 ms, passo 21.40 ms
- extrapolação do médio (checagem interna: B mediano observado em L, não estimativa de L menos D; |B_texto - B_medio| / B_medio <= 10%; B médio 72.09 ms): curto B 70.59 ms, desvio 2.1%, sustenta; longo B 71.51 ms, desvio 0.8%, sustenta
- ids do braço L iguais aos do D (processos do controle): sim

| texto | status | pedidos | blocos cheios | F0 ms (mediana; faixa; % de B) | F1 | F2 | F3 | B ms | F0 do bloco 0 reutilizado (ms) |
|---|---|---|---|---|---|---|---|---|---|
| curto | OK | 5 | 55 | 0.000 (0.000 a 0.001; 0.0%) | 0.260 (0.208 a 0.938; 0.4%) | 52.484 (46.956 a 62.363; 74.3%) | 17.883 (14.979 a 21.731; 25.3%) | 70.593 | 0.000 |
| medio | OK | 9 | 189 | 0.000 (0.000 a 0.026; 0.0%) | 0.249 (0.191 a 0.412; 0.3%) | 54.014 (43.869 a 71.006; 74.9%) | 17.969 (14.542 a 23.430; 24.9%) | 72.086 | 0.000 |
| longo | OK | 5 | 225 | 0.000 (0.000 a 0.001; 0.0%) | 0.268 (0.193 a 0.439; 0.4%) | 53.588 (46.502 a 67.058; 74.9%) | 17.789 (14.560 a 23.121; 24.9%) | 71.514 | 0.000 |

Serviço (fora da partição do host, com sobreposição; ms por bloco e razão de B; interseção com [t0, t4] à parte):

| texto | S_espera | S_monta | S_compute | S_answer | S_livro | S_end_espera | S_end_trab | S_fora | inter. [t0,t4] | R3 integral | R3 recortado | blocos com serviço | pedidos com serviço aberto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curto | 12.315 (17.4%) | 0.192 (0.3%) | 57.595 (81.6%) | 0.084 (0.1%) | 0.038 (0.1%) | 0.157 (0.2%) | 0.002 (0.0%) | 0.058 | 70.590 | 81.6% | 81.6% | 55 | 0 |
| medio | 12.412 (17.2%) | 0.200 (0.3%) | 59.135 (82.0%) | 0.095 (0.1%) | 0.039 (0.1%) | 0.156 (0.2%) | 0.002 (0.0%) | 0.062 | 72.083 | 82.1% | 82.1% | 189 | 0 |
| longo | 12.609 (17.6%) | 0.192 (0.3%) | 58.355 (81.6%) | 0.092 (0.1%) | 0.037 (0.1%) | 0.157 (0.2%) | 0.003 (0.0%) | 0.054 | 71.510 | 81.7% | 81.7% | 225 | 0 |

R6 e resto por texto: curto: resto 1, R6 0.293 (faixas: custo fixo se >= 0.5625, custo por posição se <= 0.1562); medio: resto 2, R6 0.510 (faixas: custo fixo se >= 0.6250, custo por posição se <= 0.3125); longo: resto 4, R6 0.583 (faixas: custo fixo se >= 0.7500, custo por posição se <= 0.6250)

| texto | H1 | H2 | H3 | H4 | H5 | H6 |
|---|---|---|---|---|---|---|
| curto | sem dado | sem dado | dentro | sem dado | fora | sem leitura |
| medio | sem dado | sem dado | dentro | sem dado | fora | sem leitura |
| longo | sem dado | sem dado | dentro | sem dado | fora | fora |

Medidas usadas nas classes (razões de medianas): curto: F1/B 0.4%, F2/B 74.3%, F3/B 25.3%, F3/passo(D) 0.86, R3 81.6%, S_compute/B 81.6%, R5 0.1%; medio: F1/B 0.3%, F2/B 74.9%, F3/B 24.9%, F3/passo(D) 0.86, R3 82.1%, S_compute/B 82.0%, R5 0.1%; longo: F1/B 0.4%, F2/B 74.9%, F3/B 24.9%, F3/passo(D) 0.85, R3 81.7%, S_compute/B 81.6%, R5 0.1%


## Série L100

Controle do custo do instrumento: **comprovada**
- ordem efetiva do controle (ordem.txt): confere com ordem.txt (D L L D D L, rc 0)
- desvio de prefill_ms (L menos D, sobre D, com sinal): -0.1%; pior caso (maior L menos menor D, sobre menor D): 0.1%; desvio do passo de decode: 0.1%; mediana D 413.31 ms, mediana L 413.09 ms
- desvio em ms por bloco: -0.010 ms (diferença das medianas / 22 blocos do texto médio)
  - 01 D ctrl-1b-L100-medio-01-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 414.12 ms, passo 2.80 ms
  - 02 L ctrl-1b-L100-medio-02-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 412.96 ms, passo 2.80 ms
  - 03 L ctrl-1b-L100-medio-03-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 413.09 ms, passo 2.80 ms
  - 04 D ctrl-1b-L100-medio-04-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 413.31 ms, passo 2.79 ms
  - 05 D ctrl-1b-L100-medio-05-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 412.85 ms, passo 2.79 ms
  - 06 L ctrl-1b-L100-medio-06-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 413.42 ms, passo 2.80 ms
- extrapolação do médio (checagem interna: B mediano observado em L, não estimativa de L menos D; |B_texto - B_medio| / B_medio <= 10%; B médio 19.43 ms): curto B 19.32 ms, desvio 0.6%, sustenta; longo B 19.58 ms, desvio 0.8%, sustenta
- ids do braço L iguais aos do D (processos do controle): sim

| texto | status | pedidos | blocos cheios | F0 ms (mediana; faixa; % de B) | F1 | F2 | F3 | B ms | F0 do bloco 0 reutilizado (ms) |
|---|---|---|---|---|---|---|---|---|---|
| curto | OK | 5 | 55 | 0.000 (0.000 a 0.000; 0.0%) | 0.172 (0.162 a 0.222; 0.9%) | 14.561 (14.530 a 14.681; 75.4%) | 4.587 (4.564 a 4.622; 23.7%) | 19.321 | 0.000 |
| medio | OK | 9 | 189 | 0.000 (0.000 a 0.001; 0.0%) | 0.173 (0.159 a 0.340; 0.9%) | 14.649 (14.470 a 15.498; 75.4%) | 4.612 (4.502 a 4.651; 23.7%) | 19.434 | 0.000 |
| longo | OK | 5 | 225 | 0.000 (0.000 a 0.003; 0.0%) | 0.172 (0.158 a 0.216; 0.9%) | 14.764 (14.528 a 14.918; 75.4%) | 4.639 (4.580 a 4.691; 23.7%) | 19.581 | 0.000 |

Serviço (fora da partição do host, com sobreposição; ms por bloco e razão de B; interseção com [t0, t4] à parte):

| texto | S_espera | S_monta | S_compute | S_answer | S_livro | S_end_espera | S_end_trab | S_fora | inter. [t0,t4] | R3 integral | R3 recortado | blocos com serviço | pedidos com serviço aberto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curto | 18.663 (96.6%) | 0.060 (0.3%) | 0.046 (0.2%) | 0.041 (0.2%) | 0.002 (0.0%) | 0.496 (2.6%) | 0.001 (0.0%) | 0.005 | 19.321 | 0.2% | 0.2% | 55 | 0 |
| medio | 18.781 (96.6%) | 0.060 (0.3%) | 0.048 (0.2%) | 0.043 (0.2%) | 0.002 (0.0%) | 0.496 (2.5%) | 0.001 (0.0%) | 0.004 | 19.433 | 0.3% | 0.3% | 189 | 0 |
| longo | 18.918 (96.6%) | 0.060 (0.3%) | 0.047 (0.2%) | 0.047 (0.2%) | 0.002 (0.0%) | 0.496 (2.5%) | 0.001 (0.0%) | 0.005 | 19.581 | 0.2% | 0.2% | 225 | 0 |

R6 e resto por texto: curto: resto 1, R6 0.138 (faixas: custo fixo se >= 0.5625, custo por posição se <= 0.1562); medio: resto 2, R6 0.260 (faixas: custo fixo se >= 0.6250, custo por posição se <= 0.3125); longo: resto 4, R6 0.507 (faixas: custo fixo se >= 0.7500, custo por posição se <= 0.6250)

| texto | H1 | H2 | H3 | H4 | H5 | H6 |
|---|---|---|---|---|---|---|
| curto | dentro | fora | dentro | fora | dentro | fora |
| medio | dentro | fora | dentro | fora | dentro | fora |
| longo | dentro | fora | dentro | fora | dentro | fora |

Medidas usadas nas classes (razões de medianas): curto: F1/B 0.9%, F2/B 75.4%, F3/B 23.7%, F3/passo(D) 1.64, R3 0.2%, S_compute/B 0.2%, R5 0.0%; medio: F1/B 0.9%, F2/B 75.4%, F3/B 23.7%, F3/passo(D) 1.65, R3 0.3%, S_compute/B 0.2%, R5 0.0%; longo: F1/B 0.9%, F2/B 75.4%, F3/B 23.7%, F3/passo(D) 1.66, R3 0.2%, S_compute/B 0.2%, R5 0.0%


## Série LC

Controle do custo do instrumento: **sem controle** (a série LC não tem par de controle: descritiva, sempre com ressalva)
- extrapolação do médio (checagem interna: B mediano observado em L, não estimativa de L menos D; |B_texto - B_medio| / B_medio <= 10%; B médio 73.93 ms): curto B 72.96 ms, desvio 1.3%, sustenta; longo B 73.13 ms, desvio 1.1%, sustenta
- ids do braço L iguais aos do D (processos do controle): sem dado
- ressalva: LC sem controle: descritiva, sempre com ressalva (divergência contra a 5c, se houver, só reportada)

| texto | status | pedidos | blocos cheios | F0 ms (mediana; faixa; % de B) | F1 | F2 | F3 | B ms | F0 do bloco 0 reutilizado (ms) |
|---|---|---|---|---|---|---|---|---|---|
| curto | OK | 5 | 55 | 0.000 (0.000 a 0.606; 0.0%) | 0.251 (0.196 a 0.299; 0.3%) | 58.044 (50.494 a 66.359; 79.6%) | 15.042 (12.222 a 18.539; 20.6%) | 72.963 | 0.000 |
| medio | OK | 5 | 105 | 0.000 (0.000 a 0.000; 0.0%) | 0.240 (0.202 a 0.292; 0.3%) | 59.049 (48.286 a 71.656; 79.9%) | 14.959 (12.726 a 19.244; 20.2%) | 73.933 | 0.000 |
| longo | OK | 5 | 225 | 0.000 (0.000 a 0.457; 0.0%) | 0.240 (0.201 a 0.395; 0.3%) | 57.994 (50.079 a 70.236; 79.3%) | 15.035 (12.473 a 17.528; 20.6%) | 73.133 | 0.000 |

Serviço (fora da partição do host, com sobreposição; ms por bloco e razão de B; interseção com [t0, t4] à parte):

| texto | S_espera | S_monta | S_compute | S_answer | S_livro | S_end_espera | S_end_trab | S_fora | inter. [t0,t4] | R3 integral | R3 recortado | blocos com serviço | pedidos com serviço aberto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curto | 12.681 (17.4%) | 0.185 (0.3%) | 59.292 (81.3%) | 0.092 (0.1%) | 0.035 (0.0%) | 0.518 (0.7%) | 0.002 (0.0%) | 0.059 | 72.961 | 81.3% | 81.3% | 55 | 0 |
| medio | 12.757 (17.3%) | 0.192 (0.3%) | 60.170 (81.4%) | 0.092 (0.1%) | 0.037 (0.0%) | 0.518 (0.7%) | 0.002 (0.0%) | 0.060 | 73.931 | 81.4% | 81.4% | 105 | 0 |
| longo | 12.917 (17.7%) | 0.183 (0.3%) | 59.305 (81.1%) | 0.092 (0.1%) | 0.036 (0.0%) | 0.518 (0.7%) | 0.002 (0.0%) | 0.057 | 73.132 | 81.1% | 81.1% | 225 | 0 |

R6 e resto por texto: curto: resto 1, R6 0.293 (faixas: custo fixo se >= 0.5625, custo por posição se <= 0.1562); medio: resto 2, R6 0.491 (faixas: custo fixo se >= 0.6250, custo por posição se <= 0.3125); longo: resto 4, R6 0.583 (faixas: custo fixo se >= 0.7500, custo por posição se <= 0.6250)

| texto | H1 | H2 | H3 | H4 | H5 | H6 |
|---|---|---|---|---|---|---|
| curto | descritivo | descritivo | descritivo | descritivo | descritivo | descritivo |
| medio | descritivo | descritivo | descritivo | descritivo | descritivo | descritivo |
| longo | descritivo | descritivo | descritivo | descritivo | descritivo | descritivo |

Medidas usadas nas classes (razões de medianas): curto: F1/B 0.3%, F2/B 79.6%, F3/B 20.6%, F3/passo(D) -, R3 81.3%, S_compute/B 81.3%, R5 0.1%; medio: F1/B 0.3%, F2/B 79.9%, F3/B 20.2%, F3/passo(D) -, R3 81.4%, S_compute/B 81.4%, R5 0.1%; longo: F1/B 0.3%, F2/B 79.3%, F3/B 20.6%, F3/passo(D) -, R3 81.1%, S_compute/B 81.1%, R5 0.1%


## Orçamento

decorrido 472 s de 1200 s; cortes: nenhum; não rodou: nada

## Conclusão

Este relatório não declara PASSA nem conclui causa; H7 e o nível GPU ficam para a segunda unidade.
