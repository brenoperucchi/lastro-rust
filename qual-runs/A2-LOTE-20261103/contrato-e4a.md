# Contrato E4a: passo de M posições no motor D1, backend falso (sem GPU)

Rascunho do exec, 04/11/2026, para consulta cega aos dois revisores antes de qualquer código. Escopo aprovado pelo Breno: protótipo no backend falso e nos testes de concorrência; para antes de qualquer código de GPU. Fonte das posições: prefill em lote, M até 8. Motor em `/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/motor/` (`engine.rs`, `backend.rs`, `line.rs`, `fake.rs`, `mod.rs`, `tests.rs`, `loom_tests.rs`).

## 1. Princípio

Com M = 1 o motor tem de se comportar bit a bit como hoje (mesmos testes, mesmos valores de `seq`, `flag`, geração). O passo de M posições é uma generalização, não um motor novo. O protocolo por camada continua: um `seq` e uma `flag` por camada e por passo, `Acquire`/`Release` como hoje, mesma ordem de encerramento no `Drop`.

## 2. Mudanças propostas

1. **Dois relógios.** `t` (geração do protocolo, valor de `seq`/`flag`, GEQ cíclico) conta passos, não posições. A posição na sequência (`pos`, avança M por passo) é do consumidor/backend, não do protocolo. `Config.first_token` mantém o nome e o sentido de primeiro passo. Novo `Config.max_positions` (padrão 1) fixa o M máximo e o tamanho das áreas. Atenção: `Config.prefill` já existe e significa "chaves residentes iniciais"; não é o prefill do prompt e não será renomeado nesta etapa.
2. **M efetivo do passo.** O último bloco do prompt pode ser parcial. `Engine::enqueue_step(p, m)` com `1 <= m <= max_positions`. O backend recebe `begin_step(&self, s, t, m)` (método novo, padrão sem efeito; o padrão atende o M=1 sem tocar nos backends existentes). A linha de serviço aprende o `m` do passo por uma palavra nova em `LayerIo` (`m`), escrita pelo `route` da camada e lida sob o `Acquire` de `seq[l]`, sem canal novo.
3. **`LayerIo`.** `ids`, `w` com `max_positions·topk` palavras (posição por posição, na ordem do top-k de cada uma); `x`, `y` com `max_positions·width`; mais a palavra `m`. `from_raw` passa a `1 + 2·M·topk + 2·M·width` palavras. Com `max_positions = 1` o layout só ganha a palavra `m` e o resto fica igual.
4. **Linha de serviço, `serve_layer`.** Lê as M posições e monta a união dos experts em ordem da primeira aparição (posição por posição, top-k de cada uma), com `n_e` e a lista (posição, peso) por expert. `LayerView` ganha `groups()` (`Vec<Group { key, positions, weights }>`) e mantém `keys()`, `cpu_keys()`, `cpu_weights()` para M = 1. Contagens do `Stats` (`needed`, `hits`, `cpu`) por pedido (n_e por expert), como o `SimM`; as faltas a admitir (`misses`) por chave da união, uma vez. LFU: `touch` n_e vezes por expert e o relógio anda `m` por camada, como no simulador. A admissão fica visível só no passo seguinte (como hoje).
5. **`HostExperts`.** Novo `compute_step(layer, groups, x, y)` com `y` de `m·width` palavras; cada grupo calcula só as suas posições e soma o peso na posição certa. Implementação padrão: laço por posição chamando `compute` (é o oráculo "token a token" e mantém o M = 1 igual). O kernel de CPU em lote (`cpu.rs`, portado do `bin/lote-cpu.rs`) sobrescreve o método no E4b e fica fora desta etapa; no backend falso o `FakeCpu` implementa um `compute_step` próprio.
6. **Backend falso.** `route` escreve M posições (ids e pesos determinísticos por posição), `launch_hits`/`add_y` somam por posição, e o falso conta quantas sincronizações do host acontecem por passo (tem de ser uma, o que o rev-2 chamou de P3). `Violation` ganha a checagem "M posições no `y` de cada camada, nenhuma escrita fora do intervalo da camada".
7. **Saída.** Prefill: só a última posição do bloco precisa de logits (`post_layers` não muda de forma no falso). A leitura de logits real e o KV ficam para o E4b.

## 3. O que NÃO entra

CUDA, cutile, atenção causal dentro do bloco, KV, cabeça de saída, kernel de CPU em lote, política de admissão nova, amostragem. Tudo isso é E4b, com "autorizo".

## 4. Oráculo e testes (critério de conclusão)

- **Equivalência M = 1:** todos os testes e modelos `loom` atuais passam sem alteração de asserção.
- **Passo contra token a token:** para M em {1, 2, 4, 8} e blocos parciais (resto 1 a M-1), com as mesmas escolhas de roteamento por posição, o `y` de cada posição do passo é igual ao do motor rodando as posições uma a uma (valores exatos no falso). Mesmo conjunto de faltas e de acertos por posição.
- **Protocolo:** modelos `loom` e semeaduras do escalonador determinístico rodam com M > 1 sem violação: publicação, geração, proteção de slots, `ready`, cancelamento e drenagem. Se o `loom` precisar de interleavings novos (não só formas novas), isso é parada (seção 5).
- **Contadores:** `needed` = soma dos pedidos por posição; uma sincronização do host por passo; `misses` sem duplicata; LFU conta por pedido.
- **Mutantes:** pelo menos um mutante por regra nova (união fora da ordem da primeira aparição; `m` lido antes do `Acquire`; peso somado na posição errada; bloco parcial lendo posições velhas) com teste que o pega.

## 5. Parada e limites de tamanho

Volta ao Breno, sem seguir, se: o `loom` exigir interleavings novos; mais de 6 unidades revisáveis ou cerca de 2.000 linhas tocadas; qualquer mudança que quebre o M = 1; ou se a palavra `m` em `LayerIo` não puder ser publicada sob o `Acquire` existente. Unidades previstas, cada uma com revisão nível 2 (gatilho de protocolo): U1 formas (`Config`, `LayerIo`, `begin_step`, `enqueue_step`) e falso; U2 linha de serviço (união, LFU, `Stats`); U3 `HostExperts::compute_step` e `FakeCpu`; U4 testes de equivalência, oráculo, `loom` e mutantes.

## 6. Perguntas aos revisores

1. A palavra `m` dentro de `LayerIo` (publicada pela `route` sob `seq[l]`) é a forma certa de a linha de serviço saber o tamanho do passo, ou `m` deve viajar em `Published` / em um campo do motor? Que risco de corrida ou de ordem de memória você vê?
2. `t` por passo e `pos` fora do protocolo mantém intactos o `GEQ` cíclico, `release_value` e a drenagem? Algo no `Drop` ou na guarda da linha de serviço supõe "um `t` = um token" que o contrato não lista?
3. `compute_step` com implementação padrão por posição é um oráculo honesto, ou esconde diferença de ordem de soma que importa mesmo no falso?
4. Contagem do `Stats`, LFU por pedido e relógio de M por camada: coerentes com o `SimM`? O que mais o `SimM` assume que o motor real fará diferente?
5. A lista de testes e as paradas da seção 5 bastam? Que mutante ou invariante falta? O tamanho previsto é realista?

Responda com posição, premissas explícitas, recomendação, validação e próximo passo.
