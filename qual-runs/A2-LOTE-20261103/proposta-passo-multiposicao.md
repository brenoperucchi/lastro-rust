# Proposta: passo com várias posições no motor D1 (A2-PASSO)

Escrita pelo exec em 04/11/2026, depois do passo zero (A2-LOTE-LIMITE). É uma proposta para o Breno decidir, não uma autorização. Nada aqui foi executado além do que já está commitado (`9cf1a33`, sem push). Antes de ir ao Breno ela passa pelos dois revisores, cegos um do outro.

## 1. O que já se sabe

- Hoje o motor decodifica uma posição por passo. O A2-MULTIPOS-SIM (só simulador) estimou o custo por posição de um passo com M posições juntas. A regra escrita antes: M até 8, ganho de pelo menos 15% nos dois modelos a 10% de slots.
- O A2-LOTE (microbench) mediu o custo de um expert para 1 a 32 posições. Na CPU passou todos os portões: oito posições custam cerca de 1,2 vezes uma. Na GPU o portão de representatividade reprovou duas vezes no 3B (+47% e +42% contra o limite de 40%); sem decisão, sem terceira tentativa.
- O passo zero (A2-LOTE-LIMITE, critério em `criterio-limite.txt` commitado antes de rodar) usou a CPU como medida, a GPU no pior caso (cada posição paga o expert inteiro) e o custo fixo da camada com M posições crescendo até M vezes o de hoje (g_M = M·g). Resultado em M=8: ganho de 44,6% (1B) e 42,5% (3B). Com g constante: 66,5% e 62,7%. Com 25% de slots no pior g_M: 36,3% e 33,6%. A GPU medida e a GPU no pior caso dão o mesmo ganho, porque a GPU fica atrás da CPU na camada.
- Conclusão da regra: VALE PROTOTIPAR NO MOTOR sob as hipóteses abaixo. Só isso. Não é previsão de velocidade.

## 2. De que o resultado depende (hipóteses e lacunas)

1. O ganho vem quase todo do custo de lote do expert na CPU (w_C(8) = 1,22). Se o kernel de CPU do motor real escalar pior do que o do microbench, o ganho cai. Não há sensibilidade calculada a w_C.
2. As M posições são da mesma sequência, todas úteis, com os roteamentos já conhecidos, a partir de traços de passada causal sobre texto fixo. O projeto não tem fonte real de M posições: decodificação especulativa não existe aqui, e o prefill hoje é token a token.
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
- Decodificação especulativa: só as posições aceitas contam; a fração útil mínima para manter 15% em M=8 é 0,39 (1B) e 0,43 (3B) no cenário com g constante. Precisa de um modelo de rascunho ou outro mecanismo que o projeto não tem.
- Sequências diferentes em lote: a união de experts é outra e o simulador não cobre.

## 5. Etapas propostas (cada uma com critério escrito antes e parada se reprovar)

- **E1, sem GPU:** sensibilidade do passo zero a w_C (quanto o custo de lote do expert na CPU pode piorar antes de o ganho cair a 15%) e fração de camadas por lado no caminho crítico. Só simulador, minutos a dezenas de minutos. Saída: o fator de w_C tolerável e se a CPU é de fato o lado que decide.
- **E2, decisão do Breno:** qual é a fonte de posições que vale testar primeiro (prefill em lote, especulação ou nenhuma). Sem fonte, o caminho se fecha por falta de uso, não por falta de ganho.
- **E3, uma janela de GPU (precisa de "autorizo"):** microbench do g_M no mesmo executável que mede o n=1 do próprio kernel do motor (corrige o que travou o A2-LOTE): atenção com M consultas sobre um K/V real, roteador, top-k e sincronização por camada, M de 1 a 8.
- **E4, protótipo mínimo no motor, primeiro no backend falso e nos testes loom:** passo de M posições até a união de experts, com o kernel de CPU em lote; depois GPU.
- Parada: se E1 mostrar que o ganho cai abaixo de 15% com um w_C só um pouco pior que o medido, ou se E2 não tiver fonte de posições, o A2 se fecha sem GPU.

## 6. O que o exec pede aos revisores

Posição com premissas explícitas, sobre:
1. O passo zero sustenta escrever esta proposta ao Breno? Que hipótese escondida invalida a conclusão (além das da seção 2)?
2. A seção 4: qual fonte de posições primeiro e por quê; o prefill em lote é um teste válido para o ganho que interessa (decode)?
3. A seção 5: a ordem E1, E2, E3, E4 está certa? O que falta fixar antes de E3? O E3 deveria vir antes do E4 ou o protótipo mínimo ponta a ponta mostra mais?
4. O mapa do g (seção 3) tem lacuna? Algo que o passo com M posições toca e a seção não lista?
5. Qual o esforço e o risco do E4 em ordem de grandeza, pelo que se vê no código (`backend.rs`, `engine.rs`)?
