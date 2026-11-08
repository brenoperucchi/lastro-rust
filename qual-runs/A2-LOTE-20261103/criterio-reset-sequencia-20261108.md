# Reset de sequência no Engine: critério pré-registrado (08/11/2026)
Escrito ANTES do código e dos dados. Autorização: decisão do Breno de 08/11 17:56 (via coordenador). Unidade: reutilizar o motor entre pedidos do `primeira-palavra`, sem recriá-lo.

## Escopo
- Muda: `Engine` (método novo `nova_sequencia`: recusa se fechado ou com token em andamento; zera `pos`; não zera `t`), chamada de `CudaBackend::reset_sequence` (já existe) com a stream ociosa, `Sequencia` reutilizável no binário e em `primeira_palavra_regra.rs` (guardar o `Published` pendente em vez de consumi-lo; contadores `needed/hits/cpu` por pedido pela diferença; pedido que falha fecha o motor e o próximo recria).
- NÃO muda: kernels, protocolo de geração, regras de rótulo, testes e critérios anteriores, `tests/model.rs`.

## Provas
1. Sem GPU (FakeGpu): dois pedidos no mesmo `Engine` com `nova_sequencia` no meio; `position()` volta a 0, ordem das chamadas ao backend igual à de Engine novo, geração `t` contígua, `nova_sequencia` com token enfileirado ou motor fechado é recusado, falha no meio fecha o motor. Testes vizinhos (`motor::`, `primeira_palavra_regra`, `bin primeira-palavra`) continuam passando.
2. Com GPU (uma janela, 4090, 30 min, 1B e 3B): lista de pedidos do 5b (C e A, textos já congelados). A = Engine novo por pedido; B = um Engine reutilizado, em ordem A longo -> C curto -> A longo -> C curto (expõe KV velho). PASSA se, para cada pedido, `ids_fnv`, `ids_n`, token escolhido e o vetor de logits forem IDÊNTICOS bit a bit entre A e B. Qualquer diferença: REPROVA, sem repetição automática; volto ao coordenador.
3. Velocidade (descritiva, não decide a prova): T_sequencia do segundo pedido em diante e quente por pedido contra as metas novas (6/18/12/36 s). Meta esperada: T_sequencia ~0 depois do primeiro pedido.

## Rótulos
Correção: PASSA / REPROVA / SEM PROVA (REPROVA vence SEM PROVA; integridade quebrada derruba tudo). Velocidade: ATINGE / NÃO ATINGE, só descrição.

## Revisão
Nível 2 (gatilho protocolo, `Engine`), commit local sem push só com dois APPROVE. Máximo 6 rodadas.
