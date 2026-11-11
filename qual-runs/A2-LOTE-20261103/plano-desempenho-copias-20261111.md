# Plano: menos cópias device-para-device no pré-roteador do prefill (11/11/2026, versão 3)

Escopo da decisão do Breno (14:22): só o plano agora; revisão nível 2, limite 6; mudança no motor só depois da aprovação dele; agrupar envios de trabalho fica fora; sem push. Nada neste plano altera código.

## Mudanças desta versão
Versão 3 (rodada 164):
- rev-1 P2: corrigido o p90 atribuído ao B_LANCA (os 22 a 31 µs são do B_DTOD do 3B); degrau descrito como observação do host e a explicação por fila como hipótese compatível, não demonstrada; retirada a exclusividade sobre o B_LANCA. Recorte e estatística declarados.
- rev-1 P3: números de ruído recalculados com valores completos (decode 1,96%, entre janelas 0,88%).
- rev-2 P2-1: três binários com papéis fixos; D_velho passa a ser base+gancho, para o par diferir só pela O1; a prova de logits usa o mesmo par.
- rev-2 P2-2: teste do caminho de erro em duas camadas (auxiliar testado em CPU e teste de GPU `#[ignore]` com injetor), o de GPU listado na autorização da janela.
- rev-2 P3: HC2a só nas camadas claramente antes do degrau, com tabela por camada completa; hash do KV por camada junto dos logits; HC6 com régua B2 contra D explicitada.

Versão 2 (rodada 163, os dois REQUEST_CHANGES):
- P1 rev-2 e P2-1 rev-1: o instrumento do nível B exige 21 registros por posição (`ORDEM_POS_B` em `perfil_bloco.rs:1932`, `ORDEM_POS` e `ib3` em `f2b-extrai.py:59,119`); a O1 deixa 17. Passa a existir o instrumento v2 (seção 5), sem registros fictícios.
- rev-2 P2-1: dados de 14:10 mostram degrau de fila cheia; premissa "o que custa é a chamada" corrigida e HC2 dividida em HC2a e HC2b.
- rev-2 P2-2: a O1 muda também o decode do motor medido (mesmo `attn_block_b`); `GpuModel` é só precedente. Seção 2 e 6 corrigidas.
- rev-2 P2-3 e rev-1 P2-4: o "antes" misturava binários e janelas; agora a comparação de produção é intercalada na mesma janela, com limites derivados do ruído medido, uma referência única.
- rev-1 P2-2: ids iguais não provam logits iguais; as referências 5c não têm logits. Prova de logits por bits adicionada (seção 4).
- rev-1 P2-3 e rev-2 P3-1: caminho de erro reescrito com requisito de vida útil e alternativas; vira pergunta ao Breno.
- P3-2: HC6 medida de novo. P3-3: O2 fora.

## 1. Objetivo e linha de base
Objetivo: reduzir o custo das cópias `memcpy_dtod_async` de `Dense::attn_block_b` (`/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/cuda_dense.rs`), chamadas por camada e por posição em `pre_layer` (`/home/brenoperucchi/Devs/lastro-rust/rust/moe-rt/src/cuda_backend.rs`, laço `for p in 0..cur_m()`), sem mudar o que o modelo emite.

Linha de base (3080 Ti, 11/11, só o Lastro, série L100):
- F2 (`s3080-f2-{1b,3b}-real-20261111-1100`): pré-roteador 79 a 81% da F2 do prefill.
- F2B (`s3080-f2b-{1b,3b}-real-20261111-1410/f2b-host-resultado.md`), parcelas de `H_PRE_LAYER`: B_DTOD 39,5% (1B) e 40,3 a 40,4% (3B); B_LANCA 58,1 a 58,5%; trava e montagem desprezíveis.
- Braço D do médio, 1B (medianas por processo): prefill 413,81 / 410,80 / 413,87 ms (11:00) e 412,59 / 410,23 / 412,21 ms (14:10); decode 2,808871 / 2,777300 / 2,831641 ms (11:00) e 2,78 ms nos três (14:10; mediana 2,784051). 3B: prefill cerca de 696 ms nas duas janelas.
- Binários diferentes: 11:00 usou `primeira-palavra-f2-e182736f`; 14:10 usou `primeira-palavra-f2b-22d0bbe0`. A base do "antes" é só o de 14:10; o de 11:00 é contexto da F2.

Leitura dos 40%: B_DTOD e B_LANCA são tempo do host (relógio em volta da chamada), e o `H_PRE_LAYER` por camada tem degrau. 3B: camadas 0 a 8 perto de 300 µs, patamar de 838 a 846 µs nas camadas 10 a 31 (22 de 32). 1B: camadas 0 a 10 de 285 a 316 µs, patamar perto de 657 µs nas camadas 12 a 23 (12 de 24). Antes do degrau cada chamada custa cerca de 2 µs de mediana (cópias e lançamentos). No patamar, o p90 por posto (`ceil(0,9*n)`) de `fim-início`, sobre pedidos medidos e blocos cheios sem o primeiro, camadas do patamar, nos processos B2 do médio, é: B_LANCA 5,65 / 5,64 / 5,63 µs (1B) e 9,71 / 8,59 / 9,66 µs (3B); B_DTOD do 3B 25,5 / 22,7 / 28,8 µs, com p90 por camada até 30,8 µs. Curto e longo mostram B_LANCA perto de 5 a 6 µs (1B) e 9 a 10 µs (3B). O degrau é compatível com o host sendo freado por fila de trabalho cheia, mas isso é só hipótese: o código mede o host, não a ocupação da fila nem a duração das cópias na GPU. Se a hipótese valer, parte do efeito no `prefill_ms` viria do tempo de GPU das cópias removidas; se não, de outro mecanismo. O plano não escolhe entre elas, e o efeito pode ser pequeno ou nulo.

Meta descritiva: relatar `H_PRE_LAYER` por camada, `prefill_ms` e passo de decode, antes e depois. Não se promete ganho.

## 2. Inventário: as 6 cópias por posição e camada (lidas do código)
Tamanhos em f32: `qd = heads*head_dim`, `kvd = kv_heads*head_dim` (valores exatos vêm de `Config`). Os buffers de trabalho (`q, k, v, q2, qr, qh, k2, kr, att2, att, o`) são campos de `Dense`, comuns a todas as posições; só `h` e `xn` são por posição.

| # | Linha | Origem | Destino | Tamanho | Por que existe |
|---|---|---|---|---|---|
| 1 | 283 | `q` (`[qd]`) | `q2` (`[heads*2, half]`) | `qd` | só reforma: o RoPE pede `[-1, half]` |
| 2 | 284 | `k` (`[kvd]`) | `k2` (`[kv_heads*2, half]`) | `kvd` | idem, K |
| 3 | 290 | `kr` (saída do RoPE de K) | `k_cache[cur*kvd ..]` | `kvd` | dado real: linha da posição no cache K |
| 4 | 291 | `v` | `v_cache[cur*kvd ..]` | `kvd` | dado real: linha no cache V |
| 5 | 292 | `qr` (`[heads*2, half]`) | `qh` (`[heads, d]`) | `qd` | só reforma: a atenção pede `[heads, d]` |
| 6 | 296 | `att2` (`[heads, d]`) | `att` (`[qd]`) | `qd` | só reforma: o gemv de `o` lê `[qd]` |

4 das 6 (1, 2, 5, 6) são reforma, e em todas o buffer reformado é depois entrada `&` de um kernel. Só 3 e 4 movem dado necessário. O cabeçalho do arquivo diz que a cópia foi escolhida no lugar de `reshape` porque `reshape` consome o tensor e o empréstimo `&mut` mantém o dono vivo se um lançamento falha (rodada 10 do A1.2).

Alcance (correção): o binário medido (`primeira-palavra`) usa `CudaBackend`, e o decode dele passa pelo mesmo `pre_layer` com `cur_m() == 1`, logo pelo mesmo `attn_block_b` e pelas mesmas 6 cópias por camada e passo. `GpuModel::step_with` (`gpu_model.rs`, linhas 340 a 380, que já usa `reshape` e copia só as 2 do cache) é outro motor, usado por outros binários; vale como precedente do `reshape`, não como o decode medido. Portanto a O1 muda o prefill e o decode do motor medido.

## 3. Opções
**O1. Trocar as 4 cópias de reforma por `reshape`.**
- Muda: `attn_block_b` tira o tensor do campo (`Option::take` ou `mem::replace`), reforma, empresta ao kernel e o devolve ao campo na forma original. Vale para prefill e decode.
- Não muda: kernels, argumentos numéricos, ordem dos lançamentos de cálculo, as 2 cópias do cache, a trava, `pre_layer`.
- Ganho esperado: 4 de 6 chamadas a menos por posição e camada, mais 8 `reshape` de metadados no host (sem alocação na GPU; custo pequeno, não medido). Qualitativo; ver a leitura do degrau na seção 1.
- Riscos: caminho de erro (abaixo) e o instrumento v2 (seção 5).

Requisito de vida útil, obrigatório em qualquer política de erro: toda alocação com trabalho enfileirado continua com dono até a espera da stream, em todo caminho de erro, inclusive o de reforma. Nunca liberar primeiro e sincronizar depois (`fail_k` espera a stream antes de retornar; `reshape` consome `self` e um erro de validação o derruba). O `reshape` só falha por tamanho incompatível, determinístico com as formas fixas de `new`, o que reduz o risco sem eliminá-lo.

Alternativas para o caminho de erro (pergunta ao Breno, seção 7):
- A. Restaurar e seguir. Em qualquer erro o tensor volta ao campo, na forma original, antes de propagar; se a restauração falhar, o dono é mantido e a stream é esperada antes do descarte. O `Dense` continua utilizável, como hoje. Recomendada: é o que `attn_block_at_b` já faz com `cs`, `lens` e `cur`. Custa mais código e teste.
- B. Descartar e reconstruir. Erro deixa o `Dense` inutilizável; recuperar exige recriar o modelo, perdendo o cache K/V e a sequência em curso (`reset()` só zera `pos`, não reconstrói). Mais simples; mesmo assim vale o requisito de vida útil.
- C. Não fazer a O1 (manter as cópias). Sem risco novo, sem ganho.

**O3. Escrever direto no cache pelo kernel (elimina 3 e 4).** O RoPE de K e o gemv de V gravariam na linha `cur` do cache, o que pede deslocamento nos kernels de `gpu.rs` (ou partição sobre a linha, que o cutile pode não permitir; não verificado). Mexe no motor inteiro, kernel novo com prova própria e recompilação. Só depois da O1, e só se ela medir bem. Plano próprio.

**Fora:** O2 (fundir as cópias 1 e 2 em uma) não tem objeto se a O1 for escolhida e só valeria com mudança de layout e de kernel. O4 (copiar só o prefixo ativo) não se aplica: cada cópia já move exatamente uma linha. O5, agrupar envios de trabalho, está descartada por decisão do Breno; é a via que ataca o número de lançamentos (B_LANCA, cerca de 58%) e fica para plano próprio; a O1 também pode redistribuir a espera entre chamadas.

Recomendação: O1 com a política A.

## 4. Risco de correção
Mesmos kernels, mesmas entradas, mesma ordem das operações de ponto flutuante: a O1 só remove movimento de bytes. Expectativa: saída IGUAL BIT A BIT, ids e logits, sem tolerância. Nenhuma opção reordena soma ou redução; se a implementação final fizer isso, o plano para e volta ao Breno. O3, se vier, é kernel novo e precisa de prova própria.

Como provar (a prova de ids e a de logits são coisas diferentes):
1. Caminho de erro, em duas camadas (o Fake não executa os lançadores de `Dense`, que exigem `Stream` CUDA):
   a. Sem GPU: a lógica "tira do campo, reforma, empresta, devolve e restaura no erro" fica num auxiliar genérico sobre um traço mínimo (forma e reforma), testado em CPU com um tensor falso que falha na reforma ou no lançamento: o campo volta na forma original (A) ou o `Dense` fica inutilizável (B), não há descarte antes da "espera", e no sucesso todos voltam. Um mutante que não restaura tem de falhar o teste.
   b. Com GPU: teste `#[ignore]` (padrão de `tests/motor_passo_gpu.rs`, `E4B_GPU_AUTORIZADO=1`) com um injetor de falha por lançamento reformado (no padrão de `injeta_falha_na_copia_dos_logits`). Roda só em janela autorizada, com o nome na lista de autorizações da seção 7.
   Mais: reforma sem perda de dado e sucesso devolvendo todos os buffers.
2. Ids: contra as referências 5c (`5c-referencia/ref5c-{1b,3b}.json`, têm ids, gerados, textos e gaps) e contra a série D; ids de B2, A e D do controle iguais entre si e à referência (I7).
3. Logits: as 5c não têm logits (vêm de torch em CPU, não servem para bit a bit), e id igual não prova vetor igual (1 ULP em logit não vencedor passa). Prova nova: hash sha256 sobre todos os bits f32 do vetor de logits, nos mesmos pontos (último token do prefill e cada passo do decode dos mesmos pedidos), mais o sha256 dos bits de `read_kv` por camada no fim do prefill (oráculo E4b), para localizar onde nasce uma divergência no cache. Comparação entre os binários base+gancho e candidato (definidos na seção 5), entradas iguais, ambos com sha. O `22d0bbe0` não tem o gancho e não serve de original. Qualquer bit diferente é falha. Exige um gancho de coleta desligado por padrão (instrumento, com a mesma categoria do nível B) e uma execução de coleta em janela autorizada (pergunta ao Breno), pois esses dados do original não estão arquivados. Comparador (há `to_bits()` em `reset_regra.rs:102`) testado sem GPU: original contra original passa; mutante de 1 ULP com ids iguais falha.
4. GPU só em janela autorizada pelo Breno, na 3080 Ti.
5. Oráculo E4b do KV (`read_kv`) lê o mesmo cache; a posição `p` continua gravando a linha `step_base + p`.

## 5. Medição antes e depois
Problema resolvido: o instrumento do nível B exige a sequência de 21 registros com 6 B_DTOD por posição, em Rust (capacidade `layers*(21*m+1)`, IB3) e em Python (`ib3`). Com a O1 sobram 17 registros e todo bloco falharia a IB3. Atualizar só o sha do binário não resolve; o roteiro fixa caminho e sha (`roda-f2b-janela.sh`, `BIN` e FIXO).

Instrumento v2 (derivado, identificável, com novos hashes; mesma categoria do instrumento aprovado):
- Produtor Rust, validador Python, critério e testes passam a exigir a sequência exata da O1, a de `ORDEM_POS_B` sem as 4 posições das cópias removidas (17 registros), sem registros fictícios. O contrato v1 fica preservado para os dados de 14:10 (fixtures antigas só pelo contrato v1, novas só pelo v2; registro ausente ou trocado continua rejeitado).
- Os 8 `reshape` ficam fora de qualquer op e caem no resíduo (IB2/IB5, ressalva acima de 3%); o custo deles é visto pelo `H_PRE_LAYER` total, que é medido por fora dos carimbos.
- Teste sem GPU: bloco sintético com a sequência da O1 é aceito pela IB3 v2 e recusado pela v1; E2E do extrator com smoke na sequência nova; `--verifica-fixos` com o sha novo. Roteiro e extratores passam a ser derivados dos de 14:10, com FIXO novo (não "os mesmos").

Grandeza comparável antes e depois: `H_PRE_LAYER` por camada (soma por camada, mediana), separada em antes e depois do degrau definido pelos dados de 14:10 (3B: camada 10; 1B: camada 12), fixado antes da janela. As parcelas B_DTOD de 6 chamadas e de 2 chamadas são só descritivas, não se comparam como mesma grandeza.

Janela "depois" (3080 Ti, UUID GPU-b22d9465-8552-9f97-812e-380b590645f8, 370 W, 9 processos por modelo, teto de 1200 s por janela e por modelo, 1B e 3B em janelas separadas), braços intercalados na mesma janela e no mesmo binário-base:
- Três binários, cada um com sha no critério: (i) base+gancho: o HEAD com o gancho de logits e o instrumento v2, tudo desligado, sem a O1; (ii) candidato: base+gancho mais a O1 (o diff entre eles toca só `attn_block_b`, os campos de `Dense` e os testes da O1); (iii) `22d0bbe0` (14:10), só contexto.
- D_velho: base+gancho, nada ligado. D_novo: candidato, nada ligado. B2_novo: candidato, instrumentos ligados. Assim o par D_velho/D_novo difere só pela O1 e o código novo desligado entra nos dois. A mesma dupla é a da prova de logits. Se o Breno preferir `22d0bbe0` como D_velho, o par passa a diferir pela O1 mais o código desligado, e o plano exige conferir esse custo nulo por teste textual do caminho desligado.
- 3 processos por braço, ordem balanceada (quadrado latino). Produção: D_novo contra D_velho. Efeito do instrumento: B2_novo contra D_novo (HC6; a régua é B2 contra D e inclui o instrumento do bloco, então não se compara com a classe B2 contra A de 14:10, e a sequência também mudou). Efeito da mudança no `H_PRE_LAYER`: B2_novo contra o B2 de 14:10, só como contexto declarado (outra janela e outro instrumento); a comparação principal de produção não depende dele.
- Estatística única, fixada antes dos dados: mediana das medianas dos 3 processos de cada braço, célula do médio, e razão novo sobre velho menos 1. Referência única: o D_velho da própria janela. 11:00 e 14:10 ficam como contexto.

Critério congelado em arquivo próprio com sha antes de qualquer dado, com limites derivados do ruído medido: dentro de uma janela, prefill das 3 medianas D do 1B varia 0,75% (11:00) e 0,57% (14:10) (`max/min-1`, valores completos); decode varia 1,96% em 11:00 (2,777300 a 2,831641 ms); entre janelas as medianas D do decode diferem 0,88% (2,808871 contra 2,784051 ms). O limite de +2% no decode fica quase igual à variação interna de 1,96%; por isso é escolha a submeter ao Breno, não limite demonstrado. Por isso a comparação é só na mesma janela, e os limites são: prefill, regressão se novo/velho passar de +1,5% (o dobro da variação observada); decode, +2%. Acima disso, relata-se como regressão; abaixo, é resultado descritivo.

Hipóteses pré-registradas (descritivas, sem PASSA geral):
- HC1: ids iguais aos de D_velho e às referências 5c em todos os pedidos (falha = regressão).
- HC1b: hash dos bits dos logits igual nos pontos da seção 4 (falha = regressão).
- HC2a: nas camadas claramente antes do degrau (0 a 8 no 3B, 0 a 10 no 1B, sem a de transição: 9 no 3B, 11 no 1B), `H_PRE_LAYER` por camada cai da ordem de 4 chamadas de cerca de 2 µs por posição (mecanismo do host). O degrau pode mudar de camada com menos operações na stream; por isso a tabela por camada inteira do "depois" é sempre relatada.
- HC2b: no patamar, `H_PRE_LAYER` cai só se o tempo de GPU das 4 cópias for relevante; parte da espera pode migrar para o B_LANCA. Relatar a soma B_LANCA + B_DTOD.
- HC3: prefill de D_novo não passa de +1,5% sobre D_velho; HC4: decode não passa de +2%. O decode pode cair; também se relata.
- HC5: o resíduo de B2_novo (IB2/IB5) fica na classe comprovada (até 3%).
- HC6: B2_novo contra D_novo (inclui o instrumento do bloco) até 3% de prefill, medida de novo; régua diferente da de 14:10.
Regressão: qualquer id ou bit de logit diferente; HC3 ou HC4 acima do limite; B2_novo contra D_novo acima de 10% (SEM PROVA); erro de processo. Sem ganho em `prefill_ms` mas com `H_PRE_LAYER` menor não é regressão.

## 6. O que fica fora
Agrupar ou reduzir envios de trabalho (B_LANCA, grafos CUDA, lote de posições); kernels de atenção, gemv ou MoE; roteador; layout do cache K/V; O3 (plano próprio); tolerância numérica; qualquer outra GPU; push. O decode muda só pelo efeito direto da O1, e isso é dito ao Breno.

## 7. Ordem de trabalho e o que depende do Breno
1. Este plano (versão 2). 2. Revisão nível 2, limite 6. 3. Aprovação do Breno. 4. Mudança no motor, só depois do "autorizo", em ramo sem push. 5. Instrumento v2, gancho de logits e testes sem GPU. 6. Critério congelado com sha antes dos dados. 7. Pacote derivado, com FIXO novo. 8. Janela de coleta de logits (com o teste de erro em GPU do item 4b da seção 4) e janelas de medição, só com o "autorizo" que nomeia cada uma.

Decisões do Breno, em linguagem direta:
1. Aprovar a O1? Ela mexe no prefill e também no decode do motor, e no instrumento de medida (ele passa a ter uma versão 2).
2. Se der erro de GPU no meio da atenção, o motor deve continuar utilizável como hoje, com mais cuidado no código e nos testes (A, recomendada), ou ficar inutilizável até recriar o modelo, perdendo o contexto da conversa em curso (B, mais simples)? Em ambos os casos a memória da placa só é liberada depois que o trabalho enfileirado terminar.
3. Autoriza uma execução extra de coleta de logits, original contra candidato, em janela, para provar bit a bit? (Sem tolerância numérica; recomendada.)
4. Aceita a medição com três braços na mesma janela (D velho = base com o gancho e sem a O1, D novo, B2 novo), mantendo 9 processos e o teto de 1200 s? Alternativa: usar o binário de 14:10 como D velho, com a ressalva de que o par difere também pelo código novo desligado.
5. Aceita os limites derivados do ruído (prefill +1,5%, decode +2%) ou prefere tratar tudo como descritivo?
6. Autoriza a mudança no motor depois da aprovação, e cada janela por nome (inclusive o teste de erro em GPU).
