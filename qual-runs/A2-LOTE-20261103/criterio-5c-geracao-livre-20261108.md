# Etapa 5c, geração livre curta: critério pré-registrado (08/11/2026)
Escrito ANTES do código, das referências e dos dados. Autorização: decisão do Breno de 08/11 17:56 ("siga o plano autorizado"), recomendação A da etapa 5 aprovada às 02:40 de 07/11 (igualdade de ids em guloso contra a referência HF, empate 1e-3, 64 tokens novos, 3 pedidos por modelo, 1B primeiro, 3B nas janelas).

## O que é
Do texto cru à resposta: tokeniza (5a), aplica o template do Granite (data fixa November 07, 2026, como no 5b), lê o prompt com `le_ids` (prefill, dá os logits do primeiro token novo), e continua em laço guloso (argmax) token a token pelo caminho do motor (`begin_token`+`run_token`+`read_logits`, ou o equivalente que o M1b já usa), 64 tokens novos. O motor é reutilizado entre os 3 pedidos (`nova_sequencia`, já provado). A detokenização do texto de saída usa o tokenizador 5a e para no primeiro EOS (`<|end_of_text|>`, id 0) no uso normal; **a prova gera sempre os 64 tokens, sem parar no EOS**, como a referência (`eos_token_id=None`).

## Pedidos
3 textos crus curtos e fixos (cerca de 20, 80 e 250 palavras), em `5c-textos/`, com sha256 gravado antes da janela; prompt com template + 64 tokens novos cabe folgado em 2.048. O cenário A (2.003 ids) NÃO entra: não cabe com 64 tokens novos no contexto de 2.048 e só vale até a primeira palavra.

## Referência (só CPU, sem GPU de produto)
`tools/ref_geracao_5c.py` (derivado de `tools/ref_greedy.py`): para cada pedido e modelo (1B e 3B), tokeniza com o tokenizer HF, aplica o template, gera 64 tokens gulosos no transformers (float32 a partir de BF16, cache KV, eos_token_id=None) e grava ids do prompt, ids gerados, texto e o gap top1-top2 por passo. Os ids do prompt têm de ser IGUAIS aos do tokenizador Rust (prova 5a); se diferirem, SEM PROVA.

## Regra de rótulo (por pedido, depois por modelo e conjunto)
- PASSA no pedido: os 64 ids gerados pelo Rust iguais aos da referência; ou a primeira divergência acontece num passo com gap da referência < 1e-3 (empate documentado) e, a partir daí, não se compara mais (as sequências seguem caminhos diferentes por construção).
- REPROVA: primeira divergência em passo com gap >= 1e-3, ou ids do prompt diferentes quando os dois lados rodaram.
- SEM PROVA: erro, geração incompleta (< 64 tokens), referência ausente, hash de entrada diferente, integridade quebrada (binário, critério, textos, tokenizer, referência). REPROVA vence SEM PROVA; integridade quebrada derruba tudo.
- Modelo PASSA se os 3 pedidos PASSAM; conjunto PASSA se 1B e 3B PASSAM.
- Descritivo (sem rótulo): ms por token (mediana), tempo do prefill, tempo total até o último token, e a lista de empates usados.

## Janela
Uma janela na 4090 (GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd), até 30 min, `janela-e4b.sh` + `roda-com-limite.sh`, pasta de saída nova, sem repetição automática; REPROVA ou SEM PROVA: paro e volto ao coordenador. 5090 fora.

## Não muda
`Engine`, kernels, protocolo, testes e critérios anteriores. Só se adiciona. Revisão nível 2 antes da janela; commit local sem push só com dois APPROVE (máximo 6 rodadas).
