# Divergência de ids contra o llama.cpp: o Lastro bate com a referência (09/11/2026, lastro-rust-exec; só leitura, sem GPU)

Fonte: ids gerados (64 novos) nas células quentes das 6 janelas já feitas (cmp 1B/3B, degrau 1 1B/3B, degrau 2 1B/3B), contra a referência 5c (Transformers, CPU, float32 com pesos BF16 convertidos, `5c-referencia/ref5c-1b.json` e `ref5c-3b.json`). Nenhum dado novo, nenhuma execução.

## Resultado
- **Lastro contra a referência 5c: 0 divergências em todas as células quentes observadas** (L10, L100 e camadas-CPU; curto, médio e longo; 1B e 3B; 6 janelas).
- **llama.cpp contra a referência 5c: diverge em 4 casos**, sempre no mesmo lugar nas séries que repetem o caso:
  | modelo/texto | série | 1º id diferente | gap da referência nessa posição |
  |---|---|---|---|
  | 1B longo | S1, S2 | 24 | 0,108 |
  | 1B longo | S3 (KV f32, sem FA) | 29 | 0,200 |
  | 3B médio | S2 | 10 | 0,156 |
  | 3B longo | S2 | 27 | 0,012 |
  O llama.cpp S1 no 3B (médio e longo) não diverge.
- Portanto a "divergência Lastro × llama.cpp" é o llama.cpp se afastando da referência, não o Lastro. O motor responde certo nesses casos.

## Leitura (hipótese, não conclusão)
- Os pontos de divergência têm gap pequeno (0,01 a 0,2 em logit, na referência float32): empates práticos, onde ordem de soma e precisão (f16/BF16 nos kernels do llama.cpp, KV f16) podem virar o argmax.
- A S3 do 1B (KV f32, `-fa off`) ainda diverge, em posição um pouco mais tarde (29 contra 24) e gap 0,20. Isso enfraquece a hipótese "só o KV f16 explica"; o resto fica por conta da aritmética dos kernels. Não medido.
- Divergir não prova que o llama.cpp está errado: o gap é da referência float32, e o argmax correto é o dela por definição do critério 5c, não por verdade absoluta.

## Limites
- Só ids; não comparei logits do llama.cpp nem do Lastro por posição. Perguntar "por que o llama.cpp vira o argmax em 24" exigiria logits do llama.cpp (API `n_probs`/logprobs do servidor) na 4090, ainda não pedido.
- Ids do Lastro conferidos só nas células quentes p1 (repetição 0); o smoke e o critério de cada janela já checavam o frio contra a referência.

## Proposta de passo seguinte (barato, opcional)
Uma janela curta na 4090 só no llama.cpp (1B longo, S1): pedir `n_probs` nas posições 20 a 30 e ver se o gap do llama.cpp na posição 24 inverte a ordem dos dois candidatos que a referência separa por 0,108. Sem tocar no Lastro, sem critério novo de velocidade. Só vale se o Breno quiser saber a causa; o essencial (motor certo) já está respondido sem GPU.
