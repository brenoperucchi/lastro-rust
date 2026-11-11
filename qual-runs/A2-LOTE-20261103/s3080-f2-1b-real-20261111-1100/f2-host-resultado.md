# F2 do host por operação em L100, janela do modelo 1B (criterio-f2-host-20261111.md)

Descritivo: sem veredito geral e sem PASSA. HA a HH são previsões pré-registradas, só conferidas ('dentro', 'fora', 'sem leitura', 'sem dado', 'SEM PROVA', 'descritivo'); HF, HG e HI não têm classe de dentro/fora; nada prova causa nem identifica o mecanismo de espera; o tempo com instrumento não é o tempo de produção.

Smoke (1 processo no curto, em B): **OK**

## Controle do custo do instrumento (D, A, B)

Classe (|B contra A|): **comprovada**
- ordem efetiva (ordem.txt): confere com ordem.txt (D A B, B D A, A B D, rc 0)
- B contra A (principal): desvio de prefill_ms 0.8%; pior caso 1.0%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.149 ms (22 blocos); desvio do passo de decode 0.4%
- B contra D (apoio): desvio de prefill_ms 0.1%; pior caso 0.9%; diferença de prefill por bloco executado, incluindo primeiro e resto, 0.020 ms (22 blocos); desvio do passo de decode -0.4%
- classe descritiva de B contra D: comprovada (não decide)
  - 01 D ctrl-1b-L100-medio-01-D: temporal OK, bloco desligado, f2 desligado, perfil-bloco.json não, perfil-f2.json não, válido True, prefill mediano 413.81 ms, passo 2.81 ms
  - 02 A ctrl-1b-L100-medio-02-A: temporal OK, bloco ligado, f2 desligado, perfil-bloco.json sim, perfil-f2.json não, válido True, prefill mediano 414.61 ms, passo 2.82 ms
  - 03 B ctrl-1b-L100-medio-03-B: temporal OK, bloco ligado, f2 ligado, perfil-bloco.json sim, perfil-f2.json sim, válido True, prefill mediano 414.25 ms, passo 2.80 ms
  - 04 B ctrl-1b-L100-medio-04-B: temporal OK, bloco ligado, f2 ligado, perfil-bloco.json sim, perfil-f2.json sim, válido True, prefill mediano 414.39 ms, passo 2.80 ms
  - 05 D ctrl-1b-L100-medio-05-D: temporal OK, bloco desligado, f2 desligado, perfil-bloco.json não, perfil-f2.json não, válido True, prefill mediano 410.80 ms, passo 2.78 ms
  - 06 A ctrl-1b-L100-medio-06-A: temporal OK, bloco ligado, f2 desligado, perfil-bloco.json sim, perfil-f2.json não, válido True, prefill mediano 410.30 ms, passo 2.78 ms
  - 07 A ctrl-1b-L100-medio-07-A: temporal OK, bloco ligado, f2 desligado, perfil-bloco.json sim, perfil-f2.json não, válido True, prefill mediano 410.98 ms, passo 2.79 ms
  - 08 B ctrl-1b-L100-medio-08-B: temporal OK, bloco ligado, f2 ligado, perfil-bloco.json sim, perfil-f2.json sim, válido True, prefill mediano 412.06 ms, passo 2.79 ms
  - 09 D ctrl-1b-L100-medio-09-D: temporal OK, bloco desligado, f2 desligado, perfil-bloco.json não, perfil-f2.json não, válido True, prefill mediano 413.87 ms, passo 2.83 ms
- ids de B iguais aos de D (I7, processos do controle): sim

## Texto curto (resto 1)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 50; células: lastro-1b-L100-curto-quente
- HA 99.0%; HB F2 78.6% (maior soma: True); HC 0.7%; HD 2.001; HE 0.0%; HH 0.998; HF razão 0.104 (faixa 0.0938 a 0.1562)

| HA | HB | HC | HD | HE | HF | HG | HH | HI |
|---|---|---|---|---|---|---|---|---|
| dentro | dentro | dentro | descritivo | descritivo | descritivo | sem dado | descritivo | descritivo |

HI (somente descritiva, ns com sinal, 50 amostras): defasagem_obs min 1900, mediana 4157796, p90 4199571, max 4219991; defasagem min 5430, mediana 4161836, p90 4203452, max 4235941; intervalo_obs min 776455, mediana 790611, p90 795166, max 812816; intervalo min 769066, mediana 790531, p90 795416, max 821726

defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico

## Texto medio (resto 2)

status **OK**; pedidos válidos 9 (mínimo 9); amostras elegíveis 180; células: ctrl-1b-L100-medio-03-B, ctrl-1b-L100-medio-04-B, ctrl-1b-L100-medio-08-B
- HA 99.2%; HB F2 78.8% (maior soma: True); HC 0.6%; HD 2.065; HE 0.0%; HH 0.992; HF razão 0.178 (faixa 0.1875 a 0.3125)

| HA | HB | HC | HD | HE | HF | HG | HH | HI |
|---|---|---|---|---|---|---|---|---|
| dentro | dentro | dentro | sem leitura | dentro | descritivo | sem dado | descritivo | descritivo |

HI (somente descritiva, ns com sinal, 180 amostras): defasagem_obs min 2210, mediana 4168966, p90 4229351, max 4392103; defasagem min 5970, mediana 4172416, p90 4232612, max 4396853; intervalo_obs min 662965, mediana 795946, p90 802466, max 944617; intervalo min 662015, mediana 795946, p90 802496, max 943527

defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico

## Texto longo (resto 4)

status **OK**; pedidos válidos 5 (mínimo 3); amostras elegíveis 220; células: lastro-1b-L100-longo-quente
- HA 99.1%; HB F2 78.7% (maior soma: True); HC 0.7%; HD 2.052; HE 0.0%; HH 0.986; HF razão 0.366 (faixa 0.3750 a 0.6250)

| HA | HB | HC | HD | HE | HF | HG | HH | HI |
|---|---|---|---|---|---|---|---|---|
| dentro | dentro | dentro | sem leitura | dentro | descritivo | sem dado | descritivo | descritivo |

HI (somente descritiva, ns com sinal, 220 amostras): defasagem_obs min 1860, mediana 4185894, p90 4269365, max 4306616; defasagem min 5240, mediana 4189264, p90 4273095, max 4310825; intervalo_obs min 458474, mediana 802977, p90 808566, max 1181300; intervalo min 457944, mediana 803016, p90 808677, max 1181730

defasagem negativa e conservada com sinal; tb_obs e tb nao medem a GPU nem identificam o mecanismo de espera; tb inclui leitura, copia e escalonamento do servico

## Orçamento

decorrido 206 s de 1200 s; cortes: nenhum; não rodou: nada; teto de processo excedido: nenhum

## Conclusão

Este relatório não declara PASSA nem conclui causa; nível B e qualquer mudança de desempenho exigem nova decisão do Breno.
