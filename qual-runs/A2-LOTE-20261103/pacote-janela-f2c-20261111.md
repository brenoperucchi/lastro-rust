# Pacote da janela F2C, plano de desempenho das cópias (O1), 11/11/2026

## O que é
Uma janela por modelo (1B primeiro, depois 3B, nunca as duas juntas), de até 1200 s por janela e modelo na RTX 3080 Ti (UUID GPU-b22d9465-8552-9f97-812e-380b590645f8; a 4090 e a 5090 nunca entram), só o Lastro, descritiva. Mede o efeito da O1 (reshape no lugar das quatro cópias de forma no pré-roteador) no prefill e no decode. Regras em `criterio-f2c-20261111.md`; plano v3 em `plano-desempenho-copias-20261111.md` (aprovado na rodada 165); decisão do Breno de 15:23 em `.herdr/DECISAO-BRENO-PLANO-COPIAS-SEIS-PONTOS-20261111-1523.txt`. Código aprovado na rodada 169.

## Fonte e binários congelados
- Commit base (gancho de coleta, instrumento v2, pacote F2C, sem a O1): `84877f02b237057ad5f9f7d3ca054437f410eddc`.
- Commit candidato (base mais a O1): `71dfc8e719260e9ccb2862740360b55a38b30d61`. O commit do pacote congelado vem por cima dele e não o altera.
- Binário base+gancho: sha256 `a76a9e72b962d0f6ecec7a65a47c8506b1f7f65d0c68d0c2414e03148f6a2540`, em `.herdr/binarios/primeira-palavra-f2c-base-a76a9e72`.
- Binário candidato: sha256 `504eca75b908716ee8e75dcdd0c4e859d76207af82071484ed716426d48216f8`, em `.herdr/binarios/primeira-palavra-f2c-cand-504eca75`.
- Contexto apenas: `primeira-palavra-f2b-22d0bbe0` (sha 22d0bbe0...), nunca roda nesta janela.
- Os dois binários foram construídos em worktrees separados (fora do /tmp, em `~/.cache/lastro-testes/f2c/`), com alvo de compilação próprio, `--release --bin primeira-palavra` e `CUDA_VISIBLE_DEVICES` vazio, sem rodar nada na GPU. Cópias sem permissão de escrita (modo 555).
- Extratores: `f2b-extrai-v2.py` sha `4d919ed9...`, `f2c-extrai.py` sha `607c4fd1...`; critério `criterio-f2c-20261111.md` sha `f234e002...`. Todos no FIXO de `roda-f2c-janela.sh`.

## FIXO
Conferido por `roda-f2c-janela.sh --verifica-fixos` (sem GPU, só lê; código 0 só se todos OK) e de novo no modo real, antes de envelope, monitor, timers ou GPU, sem pipe que mascare o código. O sha do próprio `roda-f2c-janela.sh` não pode estar nele: o verificador o imprime à parte e `decisao.txt` o registra.

## Janela de medida (`abre-f2c-20261111.sh`)
- Por modelo, 9 processos no texto médio, série L100, `--repete 3` (1 aquecimento e 3 medidas), em quadrado latino 3x3: D_velho D_novo B2_novo, B2_novo D_velho D_novo, D_novo B2_novo D_velho.
- Braços: D_velho = base+gancho, nada ligado; D_novo = candidato, nada ligado; B2_novo = candidato com `LASTRO_PERFIL_BLOCO`, `LASTRO_PERFIL_F2` e `LASTRO_PERFIL_F2B` ligados (instrumento v2). As variáveis só valem por processo; no ambiente de quem chama o roteiro recusa (código 2).
- Teto de processo 60 s (1B) e 90 s (3B); teto global 1200 s por janela e modelo. O controle cabe inteiro ou os nove viram NÃO RODOU; sem repetição automática. Sem smoke, curto ou longo.
- Limites: prefill de D_novo contra D_velho +1,5% (HC3), decode +2% (HC4); resíduo do nível B em B2_novo até 3% (HC5); B2_novo contra D_novo até 3% comprovada, até 10% ressalva, acima SEM PROVA (HC6). Saída idêntica bit a bit (HC1 na janela por ids, HC1b na coleta).

## Execução de coleta (`abre-f2c-coleta-20261111.sh <worktree-cand>`)
Execução própria, fora da janela de medida (não mede desempenho), com os mesmos portões. Usa `roda-f2c-janela.sh --modelos <m> --saida <pasta> --coleta`: dois processos por modelo (base e candidato), mesmas entradas, `--repete 3`, só `LASTRO_GANCHO_LOGITS=1`, que grava o hash dos logits e do KV por camada (HC1b). No 1B, leva também o teste de erro em GPU: `--teste-gpu --worktree-cand DIR`, que roda `cargo test --release --test motor_reforma_gpu -- --ignored --nocapture` com `E4B_GPU_AUTORIZADO=1` (o roteiro define a variável; teto 300 s), e grava `teste-gpu.txt`, o rc, o commit e o sha do executável.
Antes: criar o worktree do commit candidato (`git worktree add --detach <pasta> 71dfc8e719260e9ccb2862740360b55a38b30d61`), limpo, e compilar o teste sem GPU com `CARGO_TARGET_DIR=<pasta>/target-f2c` e `CUDA_VISIBLE_DEVICES` vazio. O roteiro recusa se o HEAD diferir do commit candidato do critério ou se a árvore estiver suja.

## Autorizações (cada uma pelo nome, com o "autorizo" do Breno)
1. Janela F2C 1B (medida).
2. Janela F2C 3B (medida).
3. Coleta F2C 1B (hashes de logits e KV) com o teste de erro em GPU `motor_reforma_gpu --ignored`.
4. Coleta F2C 3B (hashes de logits e KV).
O voxtype é parado e religado pelos scripts de abertura em `.herdr` (trap), não pelo roteiro. A coleta de 15:23 (itens 2, 3 e 6) cobre as autorizações 3 e 4; as janelas seguem a autorização permanente da 3080 Ti. Sem a coleta, a condição de saída bit a bit fica SEM PROVA.

## O que cada janela grava
- Medida: `decisao.txt`, `ordem.txt`, `prova-variavel.txt`, `plano.txt`, `orcamento.json`, `teto-excedido.txt` (se houver), `integridade-antes.txt` e `integridade-depois.txt`, `celulas/ctrl-<modelo>-L100-medio-<NN>-<Dvelho|Dnovo|B2novo>`, mais `.carga-antes/depois.txt`, `.aviso.txt` e o `.log` ao lado, e os resultados do extrator (`f2c-extrai.out`, JSON e Markdown).
- Coleta: `coleta-manifesto.txt`, `integridade-*.txt`, células `coleta-<modelo>-medio-<papel>`, `f2c-coleta-resultado.json` e `.md` (com `teste_gpu`: executado, rc, passou), `teste-gpu.txt` no 1B.

## Testes sem GPU
`test-roda-f2c-janela.sh`, `test-f2c-extrai.py`, `test-f2c-extrai-e2e.py`, `test-f2b-extrai-v2.py`, `roda-f2b-janela.sh --verifica-fixos` (com `TMPDIR` em `~/.cache/lastro-testes`).
