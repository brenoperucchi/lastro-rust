# Critério congelado da F2C, plano de desempenho das cópias (11/11/2026)

Fontes: `plano-desempenho-copias-20261111.md` (v3, aprovada na rodada 165) e a decisão do Breno de 11/11 às 15:23 (`DECISAO-BRENO-PLANO-COPIAS-SEIS-PONTOS-20261111-1523.txt`): O1 com política de erro A, coleta extra de hashes autorizada, limites de prefill +1,5% e decode +2%, janelas sob a autorização permanente da 3080 Ti, saída bit a bit idêntica, revisão nível 2 (limite 6), sem push, sem agrupar submissões e sem fundir as cópias de q e k.

## 1. Braços, binários e sua fonte

- D_velho: binário base com o gancho, nada ligado.
- D_novo: binário candidato (O1), nada ligado.
- B2_novo: binário candidato com perfil-bloco, perfil-f2 e perfil-f2b ligados (instrumento v2, 17 registros por posição).
- Contexto apenas: binário 22d0bbe0 (instrumento v1, dados de 14:10), nunca como referência de decisão.

Fonte reproduzível, dois commits locais (sem push), cada binário construído do seu commit:

- Commit base: gancho de coleta, instrumento v2 (contrato v1 preservado), pacote F2C (extratores, roteiro, testes, este critério). Não contém a O1 e compila sem ela. Hash: 84877f02b237057ad5f9f7d3ca054437f410eddc.
- Commit candidato: o commit base mais a O1. Hash: 71dfc8e719260e9ccb2862740360b55a38b30d61.
- O diff entre os dois (`git diff <base> <candidato> --stat`) lista exatamente: `rust/moe-rt/src/cuda_dense.rs` (`attn_block_b`, os campos de `Dense` e as estáticas e ganchos dos injetores de falha), `rust/moe-rt/src/reforma.rs` (novo), `rust/moe-rt/src/lib.rs` (`mod reforma`), `rust/moe-rt/src/cuda_backend.rs` (`sync_before_drop` com visibilidade `pub(crate)` e os injetores `injeta_falha_no_lancamento_reformado` e `injeta_falha_na_reforma`) e `rust/moe-rt/tests/motor_reforma_gpu.rs` (novo). Os injetores somam 12 leituras atômicas relaxadas por posição e camada no candidato (4 nos lançamentos reformados e 8 nas reformas); entram na conta do par D_velho/D_novo como parte da O1.
- Build de cada binário (worktree fora da árvore principal, alvo de compilação próprio, sem GPU):
  `git worktree add <pasta-base> <hash-base>` e `git worktree add <pasta-cand> <hash-cand>`; em cada uma, `cd rust/moe-rt && CARGO_TARGET_DIR=<alvo-proprio> cargo build --release --bin primeira-palavra`. O binário vai para `.herdr/binarios/primeira-palavra-f2c-base-a76a9e72` e `.herdr/binarios/primeira-palavra-f2c-cand-504eca75`, sem permissão de escrita.
- Onde entram os shas (todos preenchidos no congelamento): `[binario-base]`, `[binario-cand]`, `[script-f2b-extrai-v2]`, `[script-f2c-extrai]` e `[criterio-f2c]` no FIXO de `roda-f2c-janela.sh`, e as linhas abaixo.

Shas congelados (o roteiro recusa se qualquer um divergir):

- hash do commit base: 84877f02b237057ad5f9f7d3ca054437f410eddc
- hash do commit candidato: 71dfc8e719260e9ccb2862740360b55a38b30d61
- binário base+gancho: a76a9e72b962d0f6ecec7a65a47c8506b1f7f65d0c68d0c2414e03148f6a2540
- binário candidato: 504eca75b908716ee8e75dcdd0c4e859d76207af82071484ed716426d48216f8
- f2b-extrai-v2.py 4d919ed9a5eedc5b2dd24761f7fb5b65c5af6d893ea61ea1a277b7aa7e57eb1f e f2c-extrai.py 607c4fd11c9960d323503e06c3ece0b1663ebd58d885258455e13c17a25ce43a (o f2b-extrai.py de 14:10 fica intocado, sha cd6aa211..., e continua verificável por roda-f2b-janela.sh)

## 2. Desenho

Por modelo (1B e 3B), texto médio, 9 processos em quadrado latino 3x3 (D_velho D_novo B2_novo, B2_novo D_velho D_novo, D_novo B2_novo D_velho), cada um com `--repete 3` (1 aquecimento e 3 medidas). Teto de 1200 s por janela e modelo. Portões herdados: UUID da 3080 Ti, 370 W, `--verifica-fixos` sem pipe mascarando o código, binário congelado por sha sem escrita.

## 3. Estatística

Mediana das medianas dos 3 processos de cada braço. Referência única: D_velho da mesma janela. Razão = novo/velho - 1. Queda nunca é regressão.

## 4. Hipóteses

- HC1: ids gerados idênticos entre os três braços e iguais às referências 5c. Qualquer id diferente é regressão.
- HC1b: hash sha256 dos bits f32 dos logits (último token do prefill e cada passo do decode) e do KV por camada ao fim do prefill, original (base+gancho) contra candidato. Vem de uma EXECUÇÃO DE COLETA PRÓPRIA (`roda-f2c-janela.sh --coleta`, fora da janela de medida, que não mede desempenho): dois processos, um por binário, mesmas entradas e `--repete 3`, só `LASTRO_GANCHO_LOGITS=1`. O extrator valida envelope e cobertura (modelo, 4 pedidos com índice 0 de aquecimento, ids do prompt e gerados iguais à referência 5c, um hash de logits por token, KV de todas as camadas, hashes de 64 dígitos) e a procedência (shas dos binários, dos dois arquivos e das entradas contra o manifesto de integridade), e recusa o mesmo arquivo nos dois papéis. O par de binários da coleta (original e candidato) tem de ser, por papel, o da janela de medida (itens binario-base e binario-cand da integridade validada), e as entradas e o critério são os mesmos; a integridade antes x depois da coleta também vale; o contrato é `--repete 3`, exatamente 4 pedidos. Par diferente, papéis invertidos, integridade quebrada ou `n_repete` diferente de 3 dão SEM PROVA. Qualquer bit diferente é regressão; coleção ausente, incompleta, incoerente ou sem procedência é SEM PROVA. Sem a coleta, a condição de saída idêntica bit a bit do Breno fica SEM PROVA, mesmo com a janela de medida dentro dos limites.
- HC2a, HC2b: descritivas. H_PRE_LAYER por camada de B2_novo, antes e depois do degrau (1B: camadas 0 a 10 e 12 em diante, transição 11; 3B: 0 a 8 e 10 em diante, transição 9), mais B_LANCA + B_DTOD.
- HC3: prefill de D_novo contra D_velho, limite +1,5%. Acima: regressão.
- HC4: passo de decode de D_novo contra D_velho, limite +2%. Acima: regressão.
- HC5: resíduo do nível B em B2_novo até 3%.
- HC6: B2_novo contra D_novo (prefill): até 3% comprovada, até 10% com ressalva, acima de 10% SEM PROVA e conta como regressão.
- HC1 a HC6 só valem com integridade, ordem do quadrado latino, binário por braço, referências 5c e controle completo (9 processos válidos). Qualquer falha: SEM PROVA.

## 5. Execução de coleta e quem cuida do voxtype

A coleta é execução própria, autorizada à parte (itens 2, 3 e 6 da decisão de 15:23), e opcionalmente leva no mesmo envelope o teste de erro em GPU (`--teste-gpu --worktree-cand DIR`, `motor_reforma_gpu -- --ignored` com `E4B_GPU_AUTORIZADO=1`, saída e código gravados). Mesmos portões da janela: UUID da 3080 Ti, 370 W, processos proibidos, FIXO e shas dos binários. O `--worktree-cand` só é aceito (antes de parar os timers) se `git -C DIR rev-parse HEAD` for o hash do commit candidato desta seção e `git status --porcelain` estiver vazio (o alvo `target-f2c` fica de fora); o manifesto grava o commit e o sha do executável `motor_reforma_gpu-*`, e `f2c-coleta-resultado.json/.md` trazem `teste_gpu` (executado, rc, passou). O roteiro não para nem religa o voxtype: quem para e religa é o script de abertura em `.herdr`, tanto na janela de medida quanto na coleta.

## 6. Fora do critério

Mudança de desempenho no motor além da O1 exige nova decisão do Breno. Comparação com o B2 de 14:10 é só contexto.
