# Estado do A1.1 (o protocolo D1 sem GPU), 31/10, depois da rodada 5

Nada commitado. Nada de GPU rodou. A seção "### A1.1" do plano (`docs/PLANO-runtime-rust-moe-20261029.md`) tem o
resultado, a tabela de mutantes e os limites; este arquivo só diz onde retomar.

## Onde está

- Suíte normal: `taskset -c 16-27 cargo test --lib motor` (20 testes, cerca de 3 s).
- Suíte do `loom`: `RUSTFLAGS="--cfg motor_loom" CARGO_TARGET_DIR=target/loom taskset -c 16-27 cargo test --lib motor`
  (18 testes, cerca de 8 s). O cfg é `motor_loom` porque `--cfg loom` quebra o `cuda-async`.
- `trybuild`: `taskset -c 16-27 cargo test --test tipos protocolo_do_cache` (cerca de 1 min; 16 casos
  `tests/ui/motor_*.rs` com gêmeos em `tests/ui-pass/`).
- Revisão: rodadas `lastro-rust-1` (parcial) e `lastro-rust-2` (os dois, REQUEST CHANGES) corrigidas; a rodada
  `lastro-rust-3` (rev-2 aprova, rev-1 pede loom de cancelamento e 2 pares) está corrigida; a `lastro-rust-4` (rev-2 aprova, rev-1 pede o loom da falha sozinha) está corrigida; a `lastro-rust-5` (rev-2 aprova, rev-1 acha o P1 da ordem poison/soltura) está corrigida; a `lastro-rust-6` foi aprovada pelos dois (rev-1 e rev-2), sem achados. A1.1 fechado. Pareceres em `.herdr/review/lastro-rust-N/`.

## Para lembrar

- O escalonador do backend falso é reproduzível (a linha de serviço entra no sorteio desde o `spawn`); os números da
  tabela de mutantes do plano se repetem.
- `Victim` tem `#[must_use]`, mas `victim()?;` descarta sem aviso; o estado `Evicted` no lugar de `Free` não foi feito.
- O backend de CUDA existe só como interface em `backend.rs`; é o A1.2.
