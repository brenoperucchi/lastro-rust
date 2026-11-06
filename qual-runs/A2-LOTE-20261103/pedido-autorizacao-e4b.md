# Pedido de autorização do E4b (passo de várias posições na GPU)

Escrito pelo exec em 05/11/2026. Nada disto rodou na GPU. Este arquivo é para o Breno decidir; não é texto público.

## Em uma frase
O protótipo em CPU já foi aprovado (E4a). O E4b leva o mesmo passo de até 8 posições para a placa. Está pronto para você aprovar o critério e dizer qual janela de GPU autoriza, mas a medição de velocidade em si ainda não tem programa escrito.

## O que está pronto e aprovado pelos dois revisores (rodada 75, commit c6867a7)
1. O critério pré-registrado: `/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-e4b.txt`. Diz o que se mede (tempo por posição, M=8 contra M=1, mesma execução, 15 pares por modelo), quando o resultado vale e o que cada resultado faz com o A2.
2. O código do passo de várias posições no backend de CUDA, o kernel de CPU em lote e o oráculo de correção (logits e KV contra token a token, com um mutante de causalidade que ele tem de acusar): `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/cuda_backend.rs`, `cuda_dense.rs`, `cpu.rs` e `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/tests/motor_passo_gpu.rs`.
3. Testes que rodaram agora, sem GPU: 44 do motor, 22 do loom, 5 novos de CPU, 1 com a feature diag, o portão que recusa rodar sem autorização e o controle que prova que a fixture enxerga o contexto.

## O que NÃO está pronto (leia antes de autorizar)
- Nenhuma linha do código CUDA novo nunca rodou na placa. Os kernels do cutile só mostram erro na execução. O primeiro uso pode falhar por motivo banal.
- O M=1 do backend de CUDA foi reestruturado (um buffer por posição). Isso também nunca rodou na placa.
- Não existe ainda o programa que mede o tempo por posição (o critério o descreve, mas só o oráculo de correção foi escrito). Então, hoje, um "autorizo" só pode cobrir a **janela curta de correção**: rodar os 5 testes de GPU do arquivo acima na 4090, com o modelo pequeno de teste. O programa de medida e a janela de 15 pares por modelo seriam um segundo pedido, depois de escritos e revisados.
- O harness ainda não faz a passada com tabela congelada e divisão mista da opção 1 de tolerância; isso depende da sua escolha abaixo.

## Decisões suas antes de qualquer janela (estão marcadas no critério)
1. **Tolerância do oráculo.** Opção 1 (recomendada): igualdade a 1e-5 só na passada de portão com tabela congelada e divisão mista; nas repetições de tempo, 1e-4 em L2 e argmax igual. Opção 2: 1e-5 como teto, com tolerância calibrada (você fixa k, tau_min e P). Opção 3: 1e-4 em tudo. Motivo: no A1-DIAG10 o próprio M=1 já chegou a 1,1e-5 contra a passada residente.
2. **Cabeça de saída.** No M=8 ela roda uma vez por passo; no M=1, em toda posição. Recomendo (a): contar isso no ganho (é real no prefill) e registrar à parte o tempo da cabeça.
3. **Resultado ruidoso.** Mediana ≥ 15% com poucos pares positivos hoje conta como NAO_PASSA, e duas NAO_PASSA fecham o A2. Recomendo o rótulo RUIDOSO, que dá SEM CONCLUSÃO em vez de fechar.
4. **Controle de regressão.** O motor sem o passo, medido antes e depois, com limite X = 5% (proposto).
5. **Suíte antiga e GPU.** `cargo test` sem filtro roda na GPU os testes antigos, que agora passam pelo backend reestruturado. Até o "autorizo" eles ficam fora. Decida se uma GPU curta de teste unitário do código novo também exige "autorizo" do E4b (recomendo que exija).
6. **Parâmetros marcados [PROPOSTO]:** 15 pares, 2 de aquecimento, bloco de 64 posições, mediana com pelo menos 10 de 13 pares positivos, 5 pares extras com carimbo para o g_M, 5 repetições do controle antes e depois.

## O que eu preciso ouvir de você para abrir a janela curta de correção
Uma frase como: "autorizo a janela de correção do E4b: rodar os testes de GPU de `motor_passo_gpu.rs` uma vez, com os critérios aprovados". Eu a executo assim, uma vez, sem outra carga na placa:

`cd /home/brenoperucchi/Devs/lastro-rust/rust/moe-rt && E4B_GPU_AUTORIZADO=1 cargo test --offline --test motor_passo_gpu -- --ignored --test-threads=1`

Duração esperada: alguns minutos (modelo pequeno); não medi, não tenho número. A 4090 está com uso de outros processos (navegador e afins) e eu avisaria antes de rodar. O resultado volta a você sem nenhuma conclusão sobre velocidade.

## Limites
Um "autorizo" cobre só a janela que ele nomeia. Nem o A1-DIAG10 etapa 2, nem a terceira tentativa do A2-LOTE, nem a janela de medida do E4b estão autorizados. Nada foi enviado a remoto: são 66 commits locais, sem push.
