# Proposta: passo com várias posições no motor D1 (A2-PASSO)

Escrita pelo exec em 04/11/2026, depois do passo zero (A2-LOTE-LIMITE). Revisada depois da consulta cega aos dois revisores (rodada `lastro-rust-3`) e do E1; o Breno aprovou a direção (prefill em lote primeiro, decode estacionado, A2 atrás do A1-DIAG10 na 4090). Commits locais, sem push.

## 1. O que já se sabe

- Hoje o motor decodifica uma posição por passo. O A2-MULTIPOS-SIM (só simulador) estimou o custo por posição de um passo com M posições juntas. A regra escrita antes: M até 8, ganho de pelo menos 15% nos dois modelos a 10% de slots.
- O A2-LOTE (microbench) mediu o custo de um expert para 1 a 32 posições. Na CPU passou todos os portões: oito posições custam cerca de 1,2 vezes uma. Na GPU o portão de representatividade reprovou duas vezes no 3B (+47% e +42% contra o limite de 40%); sem decisão, sem terceira tentativa.
- O passo zero (A2-LOTE-LIMITE, critério em `criterio-limite.txt` commitado antes de rodar) usou a CPU como medida, a GPU no pior caso (cada posição paga o expert inteiro) e o custo fixo da camada com M posições crescendo até M vezes o de hoje (g_M = M·g). Resultado em M=8: ganho de 44,6% (1B) e 42,5% (3B). Com g constante: 66,5% e 62,7%. Com 25% de slots no pior g_M: 36,3% e 33,6%. A GPU medida e a GPU no pior caso dão o mesmo ganho, porque a GPU fica atrás da CPU na camada.
- Conclusão da regra: VALE PROTOTIPAR NO MOTOR sob as hipóteses abaixo. Só isso. Não é previsão de velocidade.
- E1 (`criterio-e1.txt`, `resultado-e1.txt`, regra escrita antes): classificação ROBUSTO. Se o kernel de CPU do motor perder metade da economia medida, o ganho em M=8 ainda é 21,4% (1B) e 20,4% (3B); perdendo 75%, cai para cerca de 10%. O custo marginal por posição extra tolerável é 0,45 de um expert, contra 0,03 medido. Dos experts da união em M=8, 81% (1B) e 84% (3B) recebem 4 ou menos posições, onde o custo medido é quase plano. Aviso disparado: a fração do tempo com a CPU dominante em M=8 com g_M = M·g é 47,5% no 1B (abaixo dos 50% da regra), embora a CPU domine em 99,8% a 100% das camadas; a métrica dilui com o g_M. Pela regra, o resultado vale como limite condicionado ao kernel de GPU do motor.

## 2. De que o resultado depende (hipóteses e lacunas)

1. O ganho vem quase todo do custo de lote do expert na CPU (w_C(8) = 1,22). Se o kernel de CPU do motor real escalar pior do que o do microbench, o ganho cai. Não há sensibilidade calculada a w_C.
2. As M posições são da mesma sequência, todas úteis, com os roteamentos já conhecidos, a partir de traços de passada causal sobre texto fixo. O w_G(n) = n é o cenário sem economia no cálculo dos experts, não um teto físico: preparação, agrupamento e lançamentos podem custar mais. O projeto não tem fonte real de M posições além do prefill (hoje token a token); decodificação especulativa não existe aqui.
3. O simulador não tem sincronização do passo, a espera de 0,15 a 0,5 ms do A1-DIAG10, nem atenção entre as M posições.
4. O kernel de GPU do microbench é CUDA C++ dedicado, não o cutile do motor. Nunca se mediu um expert em lote no kernel do motor.
5. A "fração de camadas com GPU ou CPU no caminho crítico" não foi calculada (o simulador não expõe).

## 3. Mapa do que o passo com M posições toca no motor

O motor é por token. Peças e o que muda:

- `engine.rs`: `Published` (tabela de residentes publicada por token), `enqueue_token`, `finish_token`, `run_token`. Passa a ser por passo, e `finish_token` lê logits de M posições.
- `backend.rs`, trait `Backend`: `pre_layer` (atenção e norma de entrada), `route` (roteador e top-k), `launch_hits`, `add_y`, `post_layers`. `LayerIo` tem um vetor de entrada x e um y de uma posição, e os ids do top-k de uma posição. Precisa de M posições e da união dos experts com a contagem n_e por expert.
- `HostExperts::compute(layer, keys, weights, x, y)`: calcula uma posição. Precisa da versão em lote (o kernel de CPU do A2-LOTE é o ponto de partida, não o código final).
- Kernel de GPU em lote para os hits (hoje cutile, uma posição). Exige um kernel novo ou a adaptação do cutile.
- Linha de serviço (`line.rs`): protocolo de flags por camada e tabela por token. Passa a tratar a união por camada.
- Simulador e testes: o motor tem backend falso (`fake.rs`) e testes loom; o passo de M posições entra neles antes de qualquer GPU.

O custo fixo g do simulador é um resíduo: g ≈ 85 us (1B) e ≈ 109 us (3B) é o que sobra do tempo de GPU por camada depois de descontar os experts. Ele reúne atenção, norma, roteador, publicação da tabela, flags e sincronização. Com M posições, a estimativa qualitativa é: o roteador e a norma crescem com M mas são pequenos; a atenção lê os pesos uma vez, e o K/V compartilhado da mesma sequência uma vez, então tende a ficar perto de g para M pequeno; a sincronização por camada é fixa. Isso é hipótese, nunca foi medido. O passo zero mostra que mesmo com g_M = M·g o ganho passa de 15%; o que o g_M decide é a margem (e o ganho real), não a viabilidade.

## 4. Fonte das posições (a decisão mais importante antes de gastar esforço)

- Prefill em lote: o prompt é conhecido, todas as M posições são úteis (a_eq = 1), é causal e não precisa de rascunho. Hoje o prefill é token a token e a linha de servidor mede só a resposta. Limite: o prefill é um regime diferente do decode (a união de experts por camada chega perto de todos), então um ganho no prefill não prova ganho no decode.
- Decodificação especulativa (estacionada por decisão do Breno): só as posições aceitas contam. A fração útil mínima para manter 15% em M=8 é 0,39 (1B) e 0,43 (3B) com g constante, e 0,65 (1B) e 0,68 (3B) com g_M = M·g, sem contar o custo do rascunho. Precisa de um mecanismo que o projeto não tem; a variante por busca no próprio prompt dispensa modelo de rascunho e só valeria depois de medir a aceitação fora da GPU.
- Sequências diferentes em lote: a união de experts é outra e o simulador não cobre.

## 5. Etapas (ordem aprovada pelo Breno em 04/11/2026)

- **E2, decidida:** prefill em lote, M até 8, para o tempo até o primeiro token de prompts longos (agente de código). Não prova ganho no decode.
- **E1, feita:** sensibilidade do passo zero, ROBUSTO, com o aviso do caminho crítico acima.
- **E4a, em curso:** contrato do passo de M posições (documento `contrato-e4a.md`), depois protótipo no backend falso e nos testes de concorrência, sem GPU. Para antes de qualquer código de GPU e volta ao Breno.
- **E4b, precisa de "autorizo":** o mesmo passo no backend de CUDA, repetindo por posição os kernels de GPU que já existem e pondo em lote só o kernel de CPU. Mede o tempo por posição contra o M=1 do próprio motor, na mesma execução, e o g_M por carimbo. Oráculo: KV e logits da última posição contra o token a token.
- **E3, só se o E4b mostrar que o g_M domina:** microbench dos kernels densos em lote.
- **Parada:** menos de 15% por posição em M=8 nos dois modelos no E4b fecha o A2. Prefill sem prompts longos no uso real fecha o A2 por falta de uso.

## 6. O que o exec pede aos revisores

Posição com premissas explícitas, sobre:
1. O passo zero sustenta escrever esta proposta ao Breno? Que hipótese escondida invalida a conclusão (além das da seção 2)?
2. A seção 4: qual fonte de posições primeiro e por quê; o prefill em lote é um teste válido para o ganho que interessa (decode)?
3. A seção 5: a ordem E1, E2, E3, E4 está certa? O que falta fixar antes de E3? O E3 deveria vir antes do E4 ou o protótipo mínimo ponta a ponta mostra mais?
4. O mapa do g (seção 3) tem lacuna? Algo que o passo com M posições toca e a seção não lista?
5. Qual o esforço e o risco do E4 em ordem de grandeza, pelo que se vê no código (`backend.rs`, `engine.rs`)?
