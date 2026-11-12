# F2D, plano da O3 (cache K/V direto), janela do modelo 1B (criterio-f2d-20261112.md)

Resultado conforme os limites congelados: prefill +1,5%, decode +2%, efeito distinguível só abaixo de -1% com as 3 medianas abaixo, B2_O3 contra D_O3 até 3% comprovada, até 10% com ressalva, acima SEM PROVA. Sem tolerância numérica: qualquer id ou bit diferente é regressão.

## Braços (mediana das medianas dos 3 processos, texto médio)

| braço | prefill (ms) | passo de decode (ms) |
|---|---|---|
| DO1 | - | - |
| DO3 | - | - |
| B2O3 | - | - |

- ordem efetiva: confere com ordem.txt (quadrado latino, rc 0, n 3)
  - 01 DO1 ctrl-1b-L100-medio-01-DO1: válido False, prefill 389.566 ms, passo 2.6426 ms
  - 02 DO3 ctrl-1b-L100-medio-02-DO3: válido False, prefill 390.968 ms, passo 2.6696 ms
  - 03 B2O3 ctrl-1b-L100-medio-03-B2O3: válido False, prefill 390.038 ms, passo 2.6722 ms
  - 04 B2O3 ctrl-1b-L100-medio-04-B2O3: válido False, prefill 391.463 ms, passo 2.6670 ms
  - 05 DO1 ctrl-1b-L100-medio-05-DO1: válido False, prefill 391.107 ms, passo 2.6545 ms
  - 06 DO3 ctrl-1b-L100-medio-06-DO3: válido False, prefill 390.562 ms, passo 2.6714 ms
  - 07 DO3 ctrl-1b-L100-medio-07-DO3: válido False, prefill 390.343 ms, passo 2.6702 ms
  - 08 B2O3 ctrl-1b-L100-medio-08-B2O3: válido False, prefill 391.895 ms, passo 2.6830 ms
  - 09 DO1 ctrl-1b-L100-medio-09-DO1: válido False, prefill 390.053 ms, passo 2.6577 ms
- ctrl-1b-L100-medio-01-DO1: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 17.38%);  
- ctrl-1b-L100-medio-02-DO3: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 19.06%);  
- ctrl-1b-L100-medio-03-B2O3: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 15.91%);  
- ctrl-1b-L100-medio-04-B2O3: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 15.19%);  
- ctrl-1b-L100-medio-05-DO1: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 15.94%);  
- ctrl-1b-L100-medio-06-DO3: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 16.20%);  
- ctrl-1b-L100-medio-07-DO3: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 16.03%);  
- ctrl-1b-L100-medio-08-B2O3: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 16.19%);  
- ctrl-1b-L100-medio-09-DO1: processo inválido: portão antes da célula falhou: CPU da máquina não baixou de 10% em 20 s (última janela: 16.29%);  

## Hipóteses

- HC1 (ids iguais aos de D_O1 e às referências 5c): **SEM PROVA**
- HC1b (hash dos bits dos logits e do KV): **dentro**
- HC3 (prefill D_O3 contra D_O1, razão -; limite +1,5%): **SEM PROVA**
- HC4 (decode D_O3 contra D_O1, razão -; limite +2%): **SEM PROVA**
- HC6 (B2_O3 contra D_O3, prefill, razão -; decode -): **SEM PROVA**
- efeito no prefill (D_O3 contra D_O1): **sem dado** (controle incompleto)
- efeito no decode (D_O3 contra D_O1): **sem dado** (controle incompleto)
- HC5 (resíduo do nível B em B2_O3, até 3%): **SEM PROVA**
- HC2a e HC2b (H_PRE_LAYER por camada de B2_O3, descritivas): **SEM PROVA**

## Regressão

**nenhuma**

Sem ganho em prefill_ms mas com H_PRE_LAYER menor não é regressão. A comparação com o B2 de 14:10 é só contexto (outra janela, outro instrumento). Qualquer mudança de desempenho no motor além desta exige nova decisão do Breno.
