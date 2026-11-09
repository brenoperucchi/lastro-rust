# Lastro contra llama.cpp, degraus 1 e 2: comparação igual por igual (09/11/2026, lastro-rust-exec)

Continuação de `RELATORIO-LASTRO-X-LLAMACPP-20261109.md`. Mesma 4090, mesmo Granite BF16 (1B e 3B), mesmos textos, 64 tokens, frio (3) e quente (5), prefill e decode separados. Descritivo, sem meta. Critérios pré-registrados antes dos dados, cada um com revisão nível 2 (degrau 1: r116 e r117; degrau 2: r118 e r119). Commits locais, sem push: evidência `aa1e3b7` (degrau 1) e `38d1317` (degrau 2).

Pastas: `d1-1b-real-20261109-0326`, `d1-3b-real-20261109-0334`, `d2-1b-real-*`, `d2-3b-real-*` (cada uma com `cmp-resultado.md` e `decisao.txt`). Todas as 48 células saíram válidas (nenhuma SEM PROVA), integridade íntegra, smoke OK.

## Quadro (razão Lastro / llama.cpp, mediana; 1,00 = igual)
| comparação | modelo | decode | prefill |
|---|---|---|---|
| Passado (10% dos experts na GPU) contra S1 (tudo na GPU) | 1B / 3B | ~12 a 13 | 33 a 239 / 41 a 218 |
| **Degrau 1:** Lastro 100% experts na GPU contra S1 | 1B | 1,30 a 1,36 (2,4 contra 1,8 ms) | 20 a 45 |
| | 3B | 1,14 a 1,19 (3,8 contra 3,2 ms) | 17 a 32 |
| **Degrau 2:** Lastro 22 (1B) e 29 (3B) camadas na CPU contra S2 (`--n-cpu-moe` igual) | 1B | 1,01 a 1,05 | 3,3 a 17 |
| | 3B | 0,98 a 0,99 | 2,8 a 13 |

Decode em ms por passo: degrau 2 no 1B ~21,3 a 22,2 contra ~21,0 a 21,4; no 3B ~41 contra ~42.

## Leitura
1. **O decode do motor é competitivo.** Com os experts no mesmo lugar, o Lastro empata com o llama.cpp (degrau 2: 0,98 a 1,05) ou fica 14 a 36% mais lento (degrau 1, tudo na GPU). As faixas pré-registradas (até 1,25 equivale; até 2 competitivo) caíram na primeira em todas as células do degrau 2 e no "<=2" em todas do degrau 1. A diferença de 12x da primeira janela vinha de pôr 90% dos experts de cada camada na CPU, não do motor. Isso confirma o que o Breno viu (GPU em 15 a 30%): o 10% por camada manda quase tudo para a CPU.
2. **Posicionar por camada inteira não custa decode.** O degrau 2 prova (cpu/needed = 22/24 e 29/32 exatos) que o motor serve as camadas iniciais na CPU e as finais na GPU sem perda frente ao llama.cpp.
3. **O prefill é o gargalo real.** Mesmo com tudo na GPU, o prefill do Lastro fica 17 a 45x mais lento (3B longo quente 928 ms contra 29 ms; 1B 575 contra 13 ms). No degrau 2, 3 a 17x (S2 mantém *op offload* ligado; não é leitura de experts na CPU). Hipótese, não medida: o Lastro processa os experts do prefill em blocos de 8 posições com um expert por vez, enquanto o llama.cpp usa matmul em lote. Próximo passo técnico sugerido: perfil do prefill (onde o tempo vai) antes de mexer em código.
4. **Frio:** o primeiro prefill frio do L100 custa ~2,7 a 3,9 s (1B) e ~3,2 a 3,9 s (3B) contra ~100 ms no llama.cpp; inclui copiar todos os experts para a GPU (no `T_sequencia`). O quente não paga isso.

## Limites declarados
- KV f32 no Lastro contra f16 no llama.cpp (a S3 de paridade não rodou nesta rodada; só no 1B da primeira janela).
- Fronteira do prefill diferente (llama.cpp termina após amostrar o 1º token; Lastro antes do argmax).
- Ids divergem do llama.cpp em 1B longo (degraus 1 e 2) e em 3B médio e longo (degrau 2); degrau 1 no 3B todos iguais. Divergência reportada, não ajustada; vale investigar antes de afirmar qualidade equivalente.
- Desktop gráfico dividindo a 4090 nos dois lados (condição comum declarada); memória do host declarada à parte; um Granite BF16 só; sem Strata.

## Pendências
- Decisão do Breno: investigar o prefill (perfil) como próxima frente; investigar a divergência de ids no texto longo.
- Antes de qualquer push: os commits têm o caminho `/home/brenoperucchi/...` em muitos arquivos de evidência; PENDENCIAS.md já tem o plano de preparo.
- Espera de ~6 s entre células (portão do critério): a medir.
