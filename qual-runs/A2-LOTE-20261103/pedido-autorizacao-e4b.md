# Pedido de autorização do E4b: janela de MEDIDA (passo de várias posições na GPU)

Escrito pelo exec em 06/11/2026, no lugar do pedido de 05/11 (que cobria só a janela curta de correção, já feita; está no histórico do git). Nada disto rodou na GPU de medida. Este arquivo é para o Breno decidir; não é texto público.

## Em uma frase
O critério está aprovado por você, o programa de medida está escrito e revisado pelos dois revisores (rodada 81, dois APPROVE), e falta só o seu "autorizo" para a janela que mede se o passo de 8 posições ganha pelo menos 15% por posição contra o motor de uma posição, no modelo 1B e no 3B.

## O que está pronto
1. **Critério imutável** (você aprovou em 06/11, commit 5c13299): `/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-e4b.txt`, SHA-256 `92133bc81b4695eefc8cc4a8a90c63cab6f0dfdf98e5d3303f0d64f4853627be`. Resumo do que decide: 15 pares por modelo (2 de aquecimento), PASSA se a mediana do ganho for ≥ 15% e houver pelo menos ceil(10n/13) pares positivos; com menos de 11 pares válidos, ou controle fora de 5%, ou carga alheia demais, o resultado é SEM CONCLUSÃO ou INVALIDA, nunca um NAO_PASSA falso. O resultado do A2 só sai com o 1B e o 3B.
2. **Suas decisões de 06/11** (`/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/decisoes-e4b-medida.txt`): ordem dos pares invertendo a cada par (leitura A) e g_M **fora** desta janela. O relatório vai dizer "g_M: não medido".
3. **Programa e regra de decisão** (só testados em CPU): `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/bin/e4b-medida.rs` e `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/e4b_regra.rs`. A regra de decisão é a mesma do critério, com tabela de conferência em teste.
4. **Revisão:** rodadas 78 a 81 (nível 2). A rodada 81 fechou com dois APPROVE: `/home/brenoperucchi/Devs/lastro-rust/.herdr/review/lastro-rust-81/lastro-rust-rev-1/verdict.md` (complemento) e `/home/brenoperucchi/Devs/lastro-rust/.herdr/review/lastro-rust-81/lastro-rust-rev-2/verdict.md`. Commit revisado: 266a8fe.
5. **Testes de CPU verdes no commit 266a8fe:** 85 da biblioteca, 8 do programa, 22 de loom, 44 do motor.
6. **Janela curta de correção já feita** (você autorizou em 06/11): 5 testes de GPU ok em 18,36 s, registro em `/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/janela-correcao-e4b.txt`.

## O que NÃO está pronto ou não foi provado (leia antes de autorizar)
- **O código da medida nunca rodou na GPU.** A janela de correção testou o passo de 8 posições com o modelo pequeno de teste, não este programa nem os modelos reais. A primeira execução pode falhar por motivo banal. O programa falha fechado: se algo der errado, o resultado é INVALIDA e o relatório diz o motivo, sem número de velocidade.
- **A cronometragem da cabeça de saída** (só descritiva) é código CUDA que nunca rodou, inclusive o caso "depois do `close`". O teste de GPU guardado que a cobre (`cabeca_cronometrada_devolve_tempo_valido_e_o_motor_segue`) ainda não foi executado. Se falhar na janela, a cabeça sai como PENDENTE e o rótulo do A2 não muda.
- **g_M não é medido** (sua decisão). Isso não afeta nenhum rótulo do A2.
- **Duração:** não medi e não tenho número. O A1.4 deu blocos de 2 a 5 minutos por configuração; este programa faz, por modelo, a referência, o portão, 10 repetições de controle e 30 repetições de pares. Peço para eu avisar a duração real no início da janela, depois do primeiro modelo, em vez de prometer agora.
- **Carga na máquina:** o programa só enxerga processo novo na GPU entre as consultas. Não vê um programa que já estava aberto e passou a trabalhar. Por isso a orientação abaixo.

## Decisão sua que ainda está aberta: timers de CPU
No A1-DIAG10 a janela suspendeu três timers para a CPU ficar quieta: `omabackup-sync.timer`, `omabackup-push.timer` e `inventory-collect.timer` (lista em `/home/brenoperucchi/Devs/lastro-rust/qual-runs/A1-DIAG10-20261103/timers.txt`). Os dois revisores deixaram isso fora de escopo. O E4b usa CPU de verdade (experts em CPU, divisão mista), então um timer disparando no meio pode inflar um par.
- **Opção 1 (recomendo):** suspender os três durante a janela e restaurar o estado que estavam. Eu escrevo e testo o roteiro (captura de estado, parada, `trap` de restauração, restauração conferida) e peço uma nova revisão nível 2 desse roteiro **antes** de você autorizar, porque é código novo que mexe em serviços.
- **Opção 2:** não suspender. Registro o estado dos timers no relatório e aceito o risco; um par contaminado vira SEM CONCLUSÃO pela regra de repetições perdidas, não NAO_PASSA falso.
Se você escolher a opção 1, este pedido ganha o roteiro e a data da nova revisão antes do "autorizo". Com a opção 2 ele já está completo, abaixo.

## O que eu preciso ouvir de você
Uma frase que nomeie a janela, por exemplo: "autorizo a janela de medida do E4b: 1B e 3B, uma vez, com o programa e o critério aprovados, [com timers suspensos | sem suspender timers]". Eu executo assim, uma vez, sem outra carga na placa:

Passo 0 (só CPU, antes de abrir a GPU): confiro que o binário de medida em release é o mesmo e que o `--ensaio` continua imprimindo o plano sem abrir o device. Já compilei e ensaiei agora, no commit 266a8fe: `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/target/release/e4b-medida`, SHA-256 `4c802c994b51cc4a2fbdc34fb7fca5e2b02f0dbe1ce6c2b0c9237a25ae9f87d8`. O `--ensaio` com as duas células imprimiu "15 pares (2 de aquecimento), bloco de 64, C 5+5, X 5%, slots 10%" e "nada foi aberto na GPU". Se o código mudar antes da janela, o hash muda e eu refaço este passo e aviso.

Passo 1 (a janela):

`cd /home/brenoperucchi/Devs/lastro-rust/rust/moe-rt && E4B_MEDIDA_AUTORIZADA=1 ./target/release/e4b-medida --celula 1b:/home/brenoperucchi/models/granite-3.1-1b-a400m-instruct:/home/brenoperucchi/Devs/lastro-rust/qual-runs/REF-granite-1b-texto-a --celula 3b:/home/brenoperucchi/models/granite-3.1-3b-a800m-instruct:/home/brenoperucchi/Devs/lastro-rust/qual-runs/REF-granite-3b-texto-a --saida /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/e4b-medida-janela --criterio /home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/criterio-e4b.txt --threads 8 --afin a1m`

A pasta de saída tem de ser nova ou vazia; o programa recusa uma pasta com resultado antigo. O resultado fica em `decisao.txt` dentro dela, com o rótulo por modelo (PASSA, NAO_PASSA, SEM_CONCLUSAO ou INVALIDA) e o resultado do A2 só se o 1B e o 3B rodaram.

Antes de rodar eu aviso e confiro que a 4090 está sem processo de computação alheio.

## Durante a janela (peço a você)
Não use a máquina para nada pesado: sem ditado (voxtype), vídeo, compilação nem outro trabalho na GPU ou na CPU. A janela é uma medida de velocidade, e qualquer carga pode virar par perdido.

## O que cada resultado faz
- **PASSA nos dois modelos:** o passo de várias posições vale o esforço; o A2 avança para o desenho da integração (decisão sua).
- **NAO_PASSA nos dois:** fecha o A2, como o critério já diz.
- **SEM_CONCLUSAO ou INVALIDA:** nada se conclui; o relatório diz por quê e eu proponho (a você) o que fazer.

## Limites
Um "autorizo" cobre só a janela que ele nomeia. Não estão autorizados: o teste de GPU guardado da cabeça de saída em separado (se você quiser uma janela curta extra para ele, é outro "autorizo"), o A1-DIAG10 etapa 2, a terceira tentativa do A2-LOTE e qualquer push. Nada foi enviado a remoto: os commits continuam locais.
