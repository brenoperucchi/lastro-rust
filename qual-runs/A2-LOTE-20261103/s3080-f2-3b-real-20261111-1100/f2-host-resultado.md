# F2 do host por operação em L100, janela do modelo 3B (criterio-f2-host-20261111.md)

Descritivo: sem veredito geral e sem PASSA. HA a HH são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA', 'descritivo'); HF, HG e HI não têm classe de dentro/fora; nada prova causa nem identifica o mecanismo de espera; o tempo com instrumento não é o tempo de produção.

Smoke (1 processo no curto, em B): **OK**

## Controle do custo do instrumento (D, A, B)

Classe (|B contra A|): **comprovada**
- ordem efetiva (ordem.txt): confere com ordem.txt (D A B, B D A, A B D, rc 0)
- B contra A (principal): desvio de prefill_ms 0.4%; pior caso 0.4%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.130 ms (22 blocos); desvio do passo de decode 0.3%
- B contra D (apoio): desvio de prefill_ms 0.3%; pior caso 0.4%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.104 ms (22 blocos); desvio do passo de decode 0.3%
- classe descritiva de B contra D: comprovada (não decide)
  - 01 D ctrl-3b-L100-medio-01-D: temporal OK, bloco desligado, f2 desligado, perfil-bloco.json não, perfil-f2.json não, válido True, prefill mediano 695.98 ms, passo 4.58 ms
  - 02 A ctrl-3b-L100-medio-02-A: temporal OK, bloco ligado, f2 desligado, perfil-bloco.json sim, perfil-f2.json não, válido True, prefill mediano 698.12 ms, passo 4.60 ms
  - 03 B ctrl-3b-L100-medio-03-B: temporal OK, bloco ligado, f2 ligado, perfil-bloco.json sim, perfil-f2.json sim, válido True, prefill mediano 698.04 ms, passo 4.60 ms
  - 04 B ctrl-3b-L100-medio-04-B: temporal OK, bloco ligado, f2 ligado, perfil-bloco.json sim, perfil-f2.json sim, válido True, prefill mediano 698.32 ms, passo 4.61 ms
  - 05 D ctrl-3b-L100-medio-05-D: temporal OK, bloco desligado, f2 desligado, perfil-bloco.json não, perfil-f2.json não, válido True, prefill mediano 695.74 ms, passo 4.58 ms
  - 06 A ctrl-3b-L100-medio-06-A: temporal OK, bloco ligado, f2 desligado, perfil-bloco.json sim, perfil-f2.json não, válido True, prefill mediano 695.40 ms, passo 4.58 ms
  - 07 A ctrl-3b-L100-medio-07-A: temporal OK, bloco ligado, f2 desligado, perfil-bloco.json sim, perfil-f2.json não, válido True, prefill mediano 695.26 ms, passo 4.58 ms
  - 08 B ctrl-3b-L100-medio-08-B: temporal OK, bloco ligado, f2 ligado, perfil-bloco.json sim, perfil-f2.json sim, válido True, prefill mediano 698.27 ms, passo 4.59 ms
  - 09 D ctrl-3b-L100-medio-09-D: temporal OK, bloco desligado, f2 desligado, perfil-bloco.json não, perfil-f2.json não, válido True, prefill mediano 697.40 ms, passo 4.60 ms
- ids de B iguais aos de D (I7, processos do controle): sim

## Texto curto (resto 1)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 50; células: lastro-3b-L100-curto-quente
- HA 99.4%; HB F2 81.3% (maior soma: True); HC 0.5%; HD 2.473; HE 0.0%; HH 0.999; HF razão 0.091 (faixa 0.0938 a 0.1562)

| HA | HB | HC | HD | HE | HF | HG | HH | HI |
|---|---|---|---|---|---|---|---|---|
| dentro | dentro | dentro | descritivo | descritivo | descritivo | sem dado | descritivo | descritivo |

HI (somente descritiva, ns com sinal, 50 amostras): defasagem_obs min 8471, mediana 5322467, p90 5341413, max 5361452; defasagem min 13481, mediana 5327342, p90 5346623, max 5368152; intervalo_obs min 1003108, mediana 1008473, p90 1011848, max 1025068; intervalo min 986828, mediana 1008478, p90 1012218, max 1030628

defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico

## Texto medio (resto 2)

status **OK**; pedidos válidos 9 (mínimo 9); amostras elegíveis 180; células: ctrl-3b-L100-medio-03-B, ctrl-3b-L100-medio-04-B, ctrl-3b-L100-medio-08-B
- HA 99.4%; HB F2 81.0% (maior soma: True); HC 0.5%; HD 2.458; HE 0.0%; HH 0.996; HF razão 0.155 (faixa 0.1875 a 0.3125)

| HA | HB | HC | HD | HE | HF | HG | HH | HI |
|---|---|---|---|---|---|---|---|---|
| dentro | dentro | dentro | sem leitura | dentro | descritivo | sem dado | descritivo | descritivo |

HI (somente descritiva, ns com sinal, 180 amostras): defasagem_obs min 2450, mediana 5337878, p90 5370713, max 5389453; defasagem min 7750, mediana 5342823, p90 5375773, max 5400773; intervalo_obs min 991288, mediana 1012998, p90 1017219, max 1044598; intervalo min 972448, mediana 1012858, p90 1017418, max 1050439

defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico

## Texto longo (resto 4)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 220; células: lastro-3b-L100-longo-quente
- HA 99.4%; HB F2 81.1% (maior soma: True); HC 0.5%; HD 2.486; HE 0.0%; HH 0.994; HF razão 0.408 (faixa 0.3750 a 0.6250)

| HA | HB | HC | HD | HE | HF | HG | HH | HI |
|---|---|---|---|---|---|---|---|---|
| dentro | dentro | dentro | sem leitura | dentro | descritivo | sem dado | descritivo | descritivo |

HI (somente descritiva, ns com sinal, 220 amostras): defasagem_obs min 2350, mediana 5343137, p90 5392813, max 5435342; defasagem min 8300, mediana 5348212, p90 5398343, max 5468874; intervalo_obs min 770087, mediana 1015468, p90 1021908, max 1262590; intervalo min 757446, mediana 1015358, p90 1022018, max 1274851

defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico

## Orçamento

decorrido 309 s de 1200 s; cortes: nenhum; não rodou: nada; teto de processo excedido: nenhum

## Conclusão

Este relatório não declara PASSA nem conclui causa; nível B e qualquer mudança de desempenho exigem nova decisão do Breno.
