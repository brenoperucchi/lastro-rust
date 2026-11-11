# Migração dos portões de UUID para a 3080 Ti (10/11/2026)

Decisão do Breno de 10/11 14:54: só a 3080 Ti (GPU-b22d9465-8552-9f97-812e-380b590645f8) passa; a 4090 e a 5090 seguem recusadas.

## O que mudou
- `rust/moe-rt/src/lastro_cli.rs` (fonte única de `lastro` e `lastro-servidor`): `UUID_4090` virou `UUID_3080TI`, mensagem "a 3080 Ti combinada"; contrato de `exige_uuid`/`portoes` igual.
- `rust/moe-rt/src/bin/leitura-correcao.rs` e `leitura-velocidade.rs`: mesma troca (constante, mensagem, docs), testes renomeados para `so_o_uuid_da_3080ti_combinada_passa`.
- Motor e instrumento intocados.

## Testes (CUDA_VISIBLE_DEVICES vazio, filtrados)
- `lastro_cli`: 13 passam (portoes_de_gpu agora cobre ausência, vazio, índice, 5090, 4090 antiga, lista, e a composição autorização+UUID; `portoes_reais_recusam_sem_autorizacao` usa o `portoes()` real).
- `leitura-correcao` e `leitura-velocidade` (filtro `uuid`): 1 passa em cada.

## Binários novos (CARGO_TARGET_DIR=/home/brenoperucchi/Devs/lastro-rust-build-instr2/release, só compilados)
- lastro cb9398827d1915075a74e3b6f35d846a03aa468293021d66a3402288610e69cc
- lastro-servidor c502d606639c0607add8cb7cffa7fed218e9fafe3ffe3db7328a4c09916690f5
- leitura-correcao 8ea27b1b519f74d22a928fbfbbb649767345632b7eb8bcfd6340f0259e9c426e
- leitura-velocidade ef701f975035ee5d178c3dd921ac0454e15c522705c53be3e4dd6da08c6c4e01

Referência: rust/moe-rt/target/release/primeira-palavra continua 65faf6e7…; não toquei em target nem em lastro-rust-build-instr.

## O que não coube
- Cada `leitura-*` mantém seu próprio `exige_uuid` (não usam `lastro_cli`); não centralizei para não mudar o contrato.
- Comentários de `a1-cutile-leitor.rs`, `a1-pool-y.rs` e `motor/ESTADO-A1.*.md` ainda citam o UUID da 4090 (exemplos de execução históricos, sem portão).
- Os binários em `rust/moe-rt/target` não foram recompilados (shas congelados).
