# Instrumento de prefill do Lastro, janela do modelo 3B (criterio-instrumento-prefill-20261110.md)

Descritivo: sem veredito geral e sem PASSA. H1 a H6 são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA'); nada prova causa; o tempo com instrumento não é o tempo de produção; H7 não é avaliada nesta unidade.

## Série L10

Controle do custo do instrumento: **comprovada**
- ordem efetiva do controle (ordem.txt): confere com ordem.txt (D L L D D L, rc 0)
- desvio de prefill_ms (L menos D, sobre D, com sinal): -0.0%; pior caso (maior L menos menor D, sobre menor D): 1.0%; desvio do passo de decode: 0.4%; mediana D 3079.14 ms, mediana L 3078.99 ms
- desvio em ms por bloco: -0.007 ms (diferença das medianas / 22 blocos do texto médio)
  - 01 D ctrl-3b-L10-medio-01-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 3056.02 ms, passo 39.78 ms
  - 02 L ctrl-3b-L10-medio-02-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 3078.99 ms, passo 40.47 ms
  - 03 L ctrl-3b-L10-medio-03-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 3078.45 ms, passo 40.67 ms
  - 04 D ctrl-3b-L10-medio-04-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 3079.14 ms, passo 40.33 ms
  - 05 D ctrl-3b-L10-medio-05-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 3103.03 ms, passo 40.30 ms
  - 06 L ctrl-3b-L10-medio-06-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 3087.99 ms, passo 40.25 ms
- extrapolação do médio (checagem interna: B mediano observado em L, não estimativa de L menos D; |B_texto - B_medio| / B_medio <= 10%; B médio 144.51 ms): curto B 142.63 ms, desvio 1.3%, sustenta; longo B 143.02 ms, desvio 1.0%, sustenta
- ids do braço L iguais aos do D (processos do controle): sim

| texto | status | pedidos | blocos cheios | F0 ms (mediana; faixa; % de B) | F1 | F2 | F3 | B ms | F0 do bloco 0 reutilizado (ms) |
|---|---|---|---|---|---|---|---|---|---|
| curto | OK | 5 | 55 | 0.000 (0.000 a 0.001; 0.0%) | 0.254 (0.196 a 0.294; 0.2%) | 114.850 (97.740 a 133.857; 80.5%) | 27.994 (23.149 a 33.288; 19.6%) | 142.635 | 0.000 |
| medio | OK | 9 | 189 | 0.000 (0.000 a 0.023; 0.0%) | 0.262 (0.195 a 0.422; 0.2%) | 116.629 (96.041 a 135.910; 80.7%) | 27.390 (23.156 a 31.816; 19.0%) | 144.512 | 0.000 |
| longo | OK | 5 | 225 | 0.000 (0.000 a 0.393; 0.0%) | 0.259 (0.199 a 0.624; 0.2%) | 115.813 (101.446 a 130.366; 81.0%) | 27.359 (22.012 a 32.905; 19.1%) | 143.022 | 0.000 |

Serviço (fora da partição do host, com sobreposição; ms por bloco e razão de B; interseção com [t0, t4] à parte):

| texto | S_espera | S_monta | S_compute | S_answer | S_livro | S_end_espera | S_end_trab | S_fora | inter. [t0,t4] | R3 integral | R3 recortado | blocos com serviço | pedidos com serviço aberto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curto | 19.652 (13.8%) | 0.287 (0.2%) | 122.117 (85.6%) | 0.167 (0.1%) | 0.082 (0.1%) | 0.213 (0.1%) | 0.003 (0.0%) | 0.083 | 142.632 | 85.7% | 85.7% | 55 | 0 |
| medio | 19.750 (13.7%) | 0.294 (0.2%) | 123.968 (85.8%) | 0.168 (0.1%) | 0.084 (0.1%) | 0.213 (0.1%) | 0.003 (0.0%) | 0.089 | 144.508 | 85.8% | 85.8% | 189 | 0 |
| longo | 19.939 (13.9%) | 0.277 (0.2%) | 122.304 (85.5%) | 0.169 (0.1%) | 0.081 (0.1%) | 0.213 (0.1%) | 0.003 (0.0%) | 0.082 | 143.020 | 85.6% | 85.6% | 225 | 0 |

R6 e resto por texto: curto: resto 1, R6 0.283 (faixas: custo fixo se >= 0.5625, custo por posição se <= 0.1562); medio: resto 2, R6 0.408 (faixas: custo fixo se >= 0.6250, custo por posição se <= 0.3125); longo: resto 4, R6 0.505 (faixas: custo fixo se >= 0.7500, custo por posição se <= 0.6250)

| texto | H1 | H2 | H3 | H4 | H5 | H6 |
|---|---|---|---|---|---|---|
| curto | sem dado | sem dado | dentro | sem dado | fora | sem leitura |
| medio | sem dado | sem dado | dentro | sem dado | fora | sem leitura |
| longo | sem dado | sem dado | dentro | sem dado | fora | fora |

Medidas usadas nas classes (razões de medianas): curto: F1/B 0.2%, F2/B 80.5%, F3/B 19.6%, F3/passo(D) 0.69, R3 85.7%, S_compute/B 85.6%, R5 0.1%; medio: F1/B 0.2%, F2/B 80.7%, F3/B 19.0%, F3/passo(D) 0.68, R3 85.8%, S_compute/B 85.8%, R5 0.1%; longo: F1/B 0.2%, F2/B 81.0%, F3/B 19.1%, F3/passo(D) 0.68, R3 85.6%, S_compute/B 85.5%, R5 0.1%


## Série L100

Controle do custo do instrumento: **comprovada**
- ordem efetiva do controle (ordem.txt): confere com ordem.txt (D L L D D L, rc 0)
- desvio de prefill_ms (L menos D, sobre D, com sinal): 0.0%; pior caso (maior L menos menor D, sobre menor D): 0.4%; desvio do passo de decode: 0.2%; mediana D 695.72 ms, mediana L 695.95 ms
- desvio em ms por bloco: 0.011 ms (diferença das medianas / 22 blocos do texto médio)
  - 01 D ctrl-3b-L100-medio-01-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 698.43 ms, passo 4.60 ms
  - 02 L ctrl-3b-L100-medio-02-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 698.72 ms, passo 4.60 ms
  - 03 L ctrl-3b-L100-medio-03-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 695.95 ms, passo 4.59 ms
  - 04 D ctrl-3b-L100-medio-04-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 695.72 ms, passo 4.58 ms
  - 05 D ctrl-3b-L100-medio-05-D: temporal OK, perfil OK, linha desligado, perfil-bloco.json não, prefill mediano 695.69 ms, passo 4.59 ms
  - 06 L ctrl-3b-L100-medio-06-L: temporal OK, perfil OK, linha ligado, perfil-bloco.json sim, prefill mediano 695.63 ms, passo 4.59 ms
- extrapolação do médio (checagem interna: B mediano observado em L, não estimativa de L menos D; |B_texto - B_medio| / B_medio <= 10%; B médio 32.75 ms): curto B 32.68 ms, desvio 0.2%, sustenta; longo B 32.88 ms, desvio 0.4%, sustenta
- ids do braço L iguais aos do D (processos do controle): sim

| texto | status | pedidos | blocos cheios | F0 ms (mediana; faixa; % de B) | F1 | F2 | F3 | B ms | F0 do bloco 0 reutilizado (ms) |
|---|---|---|---|---|---|---|---|---|---|
| curto | OK | 5 | 55 | 0.000 (0.000 a 0.003; 0.0%) | 0.177 (0.162 a 0.209; 0.5%) | 26.604 (26.572 a 26.699; 81.4%) | 5.896 (5.889 a 5.917; 18.0%) | 32.677 | 0.000 |
| medio | OK | 9 | 189 | 0.000 (0.000 a 0.004; 0.0%) | 0.176 (0.160 a 0.280; 0.5%) | 26.667 (26.545 a 26.876; 81.4%) | 5.908 (5.828 a 5.954; 18.0%) | 32.753 | 0.000 |
| longo | OK | 5 | 225 | 0.000 (0.000 a 0.002; 0.0%) | 0.173 (0.160 a 0.343; 0.5%) | 26.773 (26.576 a 27.177; 81.4%) | 5.930 (5.886 a 5.981; 18.0%) | 32.877 | 0.000 |

Serviço (fora da partição do host, com sobreposição; ms por bloco e razão de B; interseção com [t0, t4] à parte):

| texto | S_espera | S_monta | S_compute | S_answer | S_livro | S_end_espera | S_end_trab | S_fora | inter. [t0,t4] | R3 integral | R3 recortado | blocos com serviço | pedidos com serviço aberto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curto | 31.713 (97.1%) | 0.090 (0.3%) | 0.088 (0.3%) | 0.110 (0.3%) | 0.002 (0.0%) | 0.671 (2.1%) | 0.002 (0.0%) | 0.006 | 32.676 | 0.3% | 0.3% | 55 | 0 |
| medio | 31.789 (97.1%) | 0.089 (0.3%) | 0.089 (0.3%) | 0.101 (0.3%) | 0.002 (0.0%) | 0.671 (2.0%) | 0.002 (0.0%) | 0.006 | 32.753 | 0.3% | 0.3% | 189 | 0 |
| longo | 31.917 (97.1%) | 0.089 (0.3%) | 0.091 (0.3%) | 0.099 (0.3%) | 0.002 (0.0%) | 0.671 (2.0%) | 0.002 (0.0%) | 0.006 | 32.876 | 0.3% | 0.3% | 225 | 0 |

R6 e resto por texto: curto: resto 1, R6 0.136 (faixas: custo fixo se >= 0.5625, custo por posição se <= 0.1562); medio: resto 2, R6 0.259 (faixas: custo fixo se >= 0.6250, custo por posição se <= 0.3125); longo: resto 4, R6 0.506 (faixas: custo fixo se >= 0.7500, custo por posição se <= 0.6250)

| texto | H1 | H2 | H3 | H4 | H5 | H6 |
|---|---|---|---|---|---|---|
| curto | dentro | fora | dentro | fora | dentro | fora |
| medio | dentro | fora | dentro | fora | dentro | fora |
| longo | dentro | fora | dentro | fora | dentro | fora |

Medidas usadas nas classes (razões de medianas): curto: F1/B 0.5%, F2/B 81.4%, F3/B 18.0%, F3/passo(D) 1.29, R3 0.3%, S_compute/B 0.3%, R5 0.0%; medio: F1/B 0.5%, F2/B 81.4%, F3/B 18.0%, F3/passo(D) 1.29, R3 0.3%, S_compute/B 0.3%, R5 0.0%; longo: F1/B 0.5%, F2/B 81.4%, F3/B 18.0%, F3/passo(D) 1.29, R3 0.3%, S_compute/B 0.3%, R5 0.0%


## Série LC

Controle do custo do instrumento: **sem controle** (a série LC não tem par de controle: descritiva, sempre com ressalva)
- extrapolação do médio (checagem interna: B mediano observado em L, não estimativa de L menos D; |B_texto - B_medio| / B_medio <= 10%; B médio 148.10 ms): curto B 145.84 ms, desvio 1.5%, sustenta; longo B 144.30 ms, desvio 2.6%, sustenta
- ids do braço L iguais aos do D (processos do controle): sem dado
- ressalva: LC sem controle: descritiva, sempre com ressalva (divergência contra a 5c, se houver, só reportada)

| texto | status | pedidos | blocos cheios | F0 ms (mediana; faixa; % de B) | F1 | F2 | F3 | B ms | F0 do bloco 0 reutilizado (ms) |
|---|---|---|---|---|---|---|---|---|---|
| curto | OK | 5 | 55 | 0.000 (0.000 a 0.009; 0.0%) | 0.251 (0.200 a 0.425; 0.2%) | 126.657 (110.228 a 169.077; 86.8%) | 18.759 (16.460 a 24.030; 12.9%) | 145.844 | 0.000 |
| medio | OK | 5 | 105 | 0.000 (0.000 a 0.001; 0.0%) | 0.253 (0.178 a 0.358; 0.2%) | 128.990 (108.457 a 148.400; 87.1%) | 18.518 (15.937 a 22.653; 12.5%) | 148.101 | 0.000 |
| longo | OK | 5 | 225 | 0.000 (0.000 a 0.001; 0.0%) | 0.252 (0.179 a 0.442; 0.2%) | 126.639 (111.442 a 145.261; 87.8%) | 17.856 (15.809 a 22.246; 12.4%) | 144.302 | 0.000 |

Serviço (fora da partição do host, com sobreposição; ms por bloco e razão de B; interseção com [t0, t4] à parte):

| texto | S_espera | S_monta | S_compute | S_answer | S_livro | S_end_espera | S_end_trab | S_fora | inter. [t0,t4] | R3 integral | R3 recortado | blocos com serviço | pedidos com serviço aberto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| curto | 20.574 (14.1%) | 0.283 (0.2%) | 123.940 (85.0%) | 0.162 (0.1%) | 0.081 (0.1%) | 0.696 (0.5%) | 0.003 (0.0%) | 0.086 | 145.842 | 85.0% | 85.0% | 55 | 0 |
| medio | 20.667 (14.0%) | 0.287 (0.2%) | 125.922 (85.0%) | 0.161 (0.1%) | 0.082 (0.1%) | 0.695 (0.5%) | 0.003 (0.0%) | 0.092 | 148.097 | 85.1% | 85.1% | 105 | 0 |
| longo | 20.853 (14.5%) | 0.283 (0.2%) | 122.134 (84.6%) | 0.161 (0.1%) | 0.081 (0.1%) | 0.696 (0.5%) | 0.003 (0.0%) | 0.092 | 144.300 | 84.7% | 84.7% | 225 | 0 |

R6 e resto por texto: curto: resto 1, R6 0.296 (faixas: custo fixo se >= 0.5625, custo por posição se <= 0.1562); medio: resto 2, R6 0.411 (faixas: custo fixo se >= 0.6250, custo por posição se <= 0.3125); longo: resto 4, R6 0.507 (faixas: custo fixo se >= 0.7500, custo por posição se <= 0.6250)

| texto | H1 | H2 | H3 | H4 | H5 | H6 |
|---|---|---|---|---|---|---|
| curto | descritivo | descritivo | descritivo | descritivo | descritivo | descritivo |
| medio | descritivo | descritivo | descritivo | descritivo | descritivo | descritivo |
| longo | descritivo | descritivo | descritivo | descritivo | descritivo | descritivo |

Medidas usadas nas classes (razões de medianas): curto: F1/B 0.2%, F2/B 86.8%, F3/B 12.9%, F3/passo(D) -, R3 85.0%, S_compute/B 85.0%, R5 0.1%; medio: F1/B 0.2%, F2/B 87.1%, F3/B 12.5%, F3/passo(D) -, R3 85.1%, S_compute/B 85.0%, R5 0.1%; longo: F1/B 0.2%, F2/B 87.8%, F3/B 12.4%, F3/passo(D) -, R3 84.7%, S_compute/B 84.6%, R5 0.1%


## Orçamento

decorrido 810 s de 1200 s; cortes: nenhum; não rodou: nada

## Conclusão

Este relatório não declara PASSA nem conclui causa; H7 e o nível GPU ficam para a segunda unidade.
