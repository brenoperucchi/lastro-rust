# lastro-rust — handoff

Runtime MoE em Rust (Granite 1B/3B BF16, experts entre VRAM e RAM, CPU calculando as faltas). Nasceu no
strata-opencode e mudou para cá em 31/10/2026 (seção "Mudança para o lastro-rust" do plano).

## Onde está cada coisa

- Plano e histórico de decisões: `docs/PLANO-runtime-rust-moe-20261029.md` (a fonte da verdade; leia a seção A1 e as
  decisões do Breno de 31/10).
- Código: `rust/moe-rt` (crate principal), mais `rust/m0-gemv`, `rust/moe-monitor` e `rust/p1-posse`.
  Dependências cutile/cuda-core/cuda-async do clone `/home/brenoperucchi/Devs/GitHub/cuda-rust` (commit 5759cb3).
- Simulador e referências: `tools/moe_sim.py`, `tools/ref_*.py`, `tools/llama-forcado/`; testes em `tests/`.
- Medidas: `qual-runs/<MARCO>-<data>/`, cada uma com `criterio.txt` gravado antes de rodar. Os registros antigos citam
  `/home/brenoperucchi/Devs/strata-opencode`; para executar de novo, troque pelo caminho deste repositório.
- Rodadas de revisão 25 a 55 e consultas 5 a 8 do runtime: copiadas em `.herdr/` (fora do git; o original fica em
  `/home/brenoperucchi/Devs/strata-opencode/.herdr/`). As novas: `lastro-rust-N`. Plano de origem (24/10):
  `docs/historico/PLANO-runtime-hybrid-strata-2026-10-24.md`.
- Página no Claude Docs: https://claude.ai/artifact/MNTT17taf1C7RC4VeHD4mt (etapa 4 e riscos atualizados a cada passo).

## Estado (31/10, 17h)

- Feitos e aprovados: M0 a M2, B0, C, comparação com o llama.cpp, A0 (A0a/A0b/A0c) e A1.0 (doorbell; commit ff639fd).
- A1.1 (protocolo sem GPU) aprovado pelos dois revisores na rodada `lastro-rust-6`. Estado em
  `rust/moe-rt/src/motor/ESTADO-A1.1.md`.
- A1.2 (peças na GPU) aprovado pelos dois revisores (checkpoint nas rodadas 7 e 8, final na rodada 12). Estado em
  `rust/moe-rt/src/motor/ESTADO-A1.2.md`. **Próximo: A1.3 (motor D1 completo com o modelo inteiro).**
- Depois: A1.2 (peças de GPU), A1.3 (motor D1), A1.4 (medida final contra lfu_cpu e llama.cpp, mesmas 6 células; a
  janela longa só com o aval do Breno). O plano termina no A1.4.
- Fora do plano, feitos em 31/10 (commit a4faca6): `qual-runs/PCIE-X8-20261031/` (análise x8/x8 contra x16/x4: ficar
  em x8/x8) e `qual-runs/FONTE-2GPU-20261031/` (teste da fonte com as duas GPUs: seguro com a 3080 Ti em 300 W).

## Máquina (depois do reboot de 31/10)

- 4090 no PCIE2 e 3080 Ti no PCIE1, as duas Gen4 x8. Fixe sempre a 4090 pelo UUID:
  `CUDA_VISIBLE_DEVICES=GPU-8f424b03-d413-23cd-ae38-a423d13bf7bd` (o índice 0 do nvidia-smi é a 3080 Ti).
- 3080 Ti limitada a 300 W no boot (`/etc/systemd/system/gpu-power-limit.service`, com o nvidia-persistenced).
- Banda H2D pinned em x8: ~12,4 GB/s; efetiva com a CPU calculando: 12,2 GB/s (era 14,1 em x16; o simulador ainda usa 14,1).
- A CPU bate 100 °C com o m2a-cache em 8 threads.

## Regras de trabalho

- Python: `taskset -c 16-27 tools/.venv-ref/bin/python -I -B`. Cargo: `taskset -c 16-27`. Medidas: `taskset -c 0-15`.
- `cargo test --release` troca `target/release/m2a-cache`: teste no perfil debug, ou `--release --bin <x>` para um só.
  Binários de medida com `cargo build --release --bin <nome>`, e sha256 gravado na pasta da medida.
- Commit local por passo do A1 aprovado pelos dois revisores, sem perguntar; avisar o Breno só quando a revisão fecha ou
  há decisão dele. Teste curto de GPU livre depois de conferir que a 4090 está ociosa.
- Revisão pela skill herdr-review, nível 2 (gatilho `protocolo` no A1). Remoto público: https://github.com/brenoperucchi/lastro-rust; push só com o aval do Breno sobre o texto.
- Bug: teste de regressão permanente que falha antes da correção.
- Comparação com o Strata adiada até o fim do plano (só llama.cpp como base).
