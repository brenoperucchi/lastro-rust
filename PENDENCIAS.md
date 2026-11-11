# Pendências do lastro-rust

- **Religar XMP 3200 e remedir.** Gatilho: o Breno avisar que religou (08/11/2026: com XMP a máquina ficou muito instável, sem data). Ao disparar: repetir a medida da etapa 4 (quente e fria, 1B e 3B) e comparar com as metas antigas (02:00 de 07/11) e com a referência de 2400 MT/s em `/home/brenoperucchi/Devs/lastro-rust/qual-runs/A2-LOTE-20261103/referencia-velocidade-memoria-2400-20261108.md`.

## Preparar o push público (decisão do Breno, 09/11/2026 02:37, via coordenador)
Repositório continua PÚBLICO. Fazer DEPOIS da comparação com o llama.cpp terminar (não misturar com a janela):
1. Criar ramo de backup local antes de qualquer mudança (ex.: backup-pre-push-20261109).
2. Tirar do histórico LOCAL ainda não enviado (origin/main..HEAD) só os 2 arquivos com a lista de serviços do usuário: qual-runs/A2-LOTE-20261103/servidor-5e-janela-20261108-2052.unidades-antes.txt e .unidades-depois.txt (commit c27c93a). Não tocar em commits já enviados. Conferir que nada mais mudou.
3. Gerar a lista das mensagens de commit não enviadas para o coordenador passar pelo humanizer e levar ao Breno.
4. Push SÓ depois do aval do Breno sobre as mensagens.

Feito em 09/11/2026 04:25: passos 1 e 2 (ramo backup-pre-push-20261109 criado; as 2 listas de unidades saíram só dos commits locais, 134 commits, origin/main continua ancestral; a árvore final difere do backup só nesses 2 arquivos, que estão no .gitignore). Lista das mensagens: /home/brenoperucchi/Devs/lastro-rust/.herdr/mensagens-commits-push-20261109.txt (ignorada pelo git). Sem push.
