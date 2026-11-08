# Etapa 5d, comando de linha: critério pré-registrado (08/11/2026)
Escrito ANTES do código e dos dados. Autorização: decisão do Breno de 08/11 17:56 (seguir o plano autorizado; "CLI antes de serviço" aprovado em 07/11 02:40).

## O que é
Binário novo `lastro` (crate rust/moe-rt, src/bin/lastro.rs): `lastro --modelo <dir do modelo> [--novos N] [TEXTO]` ou texto pela entrada padrão quando não há TEXTO. Um modelo por execução, motor criado uma vez, tokeniza (5a), template do Granite (data do dia, ou `--data` para reprodutibilidade), prefill, laço guloso do 5c, detokeniza (5a) e imprime o texto da resposta na saída padrão, em fluxo simples (texto puro, sem decoração, newline final único), parando no primeiro EOS ou em N novos tokens (padrão 64; máximo aceito 64 nesta etapa). Mensagens e diagnósticos vão para a saída de erro. Reaproveita os módulos e o laço do 5c (sem duplicar lógica nem tocar em Engine, kernels ou protocolo).

## Recusas (sem GPU, antes de abrir o device, mensagem na saída de erro, código de saída diferente de 0, nunca travar)
1. Entrada vazia ou só espaços (argumento ou entrada padrão): código 2.
2. Texto cujo prompt com template + N novos tokens passa de 2.048 posições: código 2 (a conta usa o tokenizador, antes da GPU).
3. Modelo inexistente, ou sem tokenizer.json/pesos: código 2.
4. `--novos` fora de 1..64, opção desconhecida ou argumento duplicado: código 2.
Falha em tempo de execução (GPU/motor): código 1, nada impresso na saída padrão além do que já saiu; nunca código 0.

## Prova (rótulo)
- Recusas (sem GPU, por teste automatizado e por execução real do binário): as 4 acima dão a mensagem e o código esperados, sem abrir a GPU, em menos de 5 s cada. Falha de uma = REPROVA.
- Texto (uma janela na 4090, 1B e 3B, GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd, até 30 min, janela-e4b.sh + roda-com-limite.sh, pasta nova, sem repetição): para cada um dos 3 textos congelados do 5c (5c-textos/), com `--data "November 07, 2026"` e `--novos 64`, a saída padrão deve ser IGUAL byte a byte (diff) ao `texto_gerado` da referência HF do 5c (5c-referencia/ref5c-<modelo>.json) mais um newline final, e o código de saída 0. Um texto pelo argumento e outro pela entrada padrão em cada modelo, o terceiro pelo argumento. 6 execuções, processo novo cada, sem repetição.
- PASSA: recusas OK e 6 de 6 saídas iguais. REPROVA: qualquer diferença de texto, código de saída errado ou recusa que abra a GPU. SEM PROVA: execução sem resultado, integridade quebrada (binário, critérios, textos, referências, tokenizer) ou janela estourada. REPROVA vence SEM PROVA; integridade quebrada derruba tudo.
- Descritivo (sem rótulo): tempo total de cada execução (processo novo, frio) e o tempo até o primeiro caractere.

## Não muda
Engine, kernels, protocolo, testes e critérios anteriores. Só se adiciona. Revisão nível 2 antes da janela; commit local sem push só com dois APPROVE (máximo 6 rodadas).
