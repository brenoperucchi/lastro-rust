# lastro-rust — handoff

Runtime MoE em Rust (Granite 1B/3B BF16, experts entre VRAM e RAM, CPU calculando as faltas). Nasceu no
strata-opencode e mudou para cá em 31/10/2026 (seção "Mudança para o lastro-rust" do plano).

## Onde está cada coisa

- Plano e histórico de decisões: `docs/PLANO-runtime-rust-moe-20261029.md` (a fonte da verdade; leia a seção A1 e as
  decisões do Breno de 31/10).
- Código: `rust/moe-rt` (crate principal), mais `rust/m0-gemv`, `rust/moe-monitor` e `rust/p1-posse`.
  Dependências cutile/cuda-core/cuda-async fixadas por git no commit 5759cb3 do NVIDIA/cuda-rust (desde 3/11; antes eram `path` para o clone `/home/brenoperucchi/Devs/GitHub/cuda-rust`, ainda no mesmo commit). Atualizar é um passo à parte, depois do diagnóstico dos 10%.
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
  `rust/moe-rt/src/motor/ESTADO-A1.2.md`. A1.3 (motor D1 completo, estado em `rust/moe-rt/src/motor/ESTADO-A1.3.md`) feito e aprovado pelos dois revisores (rodadas 14, 17 e 18); commits locais 3cce95d, cdd7567 e 55f9ded, ainda sem push. A1.4 (medida final) com a janela de 2/11 feita e analisada: o motor passa em 1,05 x o `lfu_cpu` nas 6 células (o ganho a 25% e 50% é da base do motor, não do protocolo, que empata a 25% e 50% e perde a 10%), ganha de 4,7 a 9,3 x do cache do llama.cpp e de 1,4 a 2,2 x do `-ncmoe`; a busca antecipada continua fora (portão da A0a não passa). Resultados em `qual-runs/A1-MEDIDA-20261101/resultado.txt`; estado em `rust/moe-rt/src/motor/ESTADO-A1.4.md`. **O plano do A1 termina aqui; falta a confirmação do Breno sobre o limite de 1,05 x e o push.**
- Depois: A1.2 (peças de GPU), A1.3 (motor D1), A1.4 (medida final contra lfu_cpu e llama.cpp, mesmas 6 células; a
  janela longa só com o aval do Breno). O plano termina no A1.4.
- Fora do plano, feitos em 31/10 (commit a4faca6): `qual-runs/PCIE-X8-20261031/` (análise x8/x8 contra x16/x4: ficar
  em x8/x8) e `qual-runs/FONTE-2GPU-20261031/` (teste da fonte com as duas GPUs: seguro com a 3080 Ti em 300 W).

- A1-DIAG10 (3/11): harness e análise aprovados nível 2; etapa 1 medida e analisada (15:22 a 19:00): passo 0 NAO_REPRODUZIDO nas duas células (1B: d(C) +0,266, d(N0) +0,490; 3B: d(C) +0,290, d(N0) +0,150), orçamento da instrumentação VIOLA; etapa 2 não roda; sem conclusão sobre H1. Próximo passo: critério novo, decisão do Breno. Estado em `rust/moe-rt/src/motor/ESTADO-A1.5.md`; resultado em `qual-runs/A1-DIAG10-20261103/analise-e1.txt`; reanálise: `taskset -c 16-27 tools/.venv-ref/bin/python -I -B qual-runs/A1-DIAG10-20261103/analise.py` (sem argumento). Janela ratificada pelo Breno em 3/11 (20:31); critério v6, timers e teto só registro aprovados; nenhuma janela nova autorizada.

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
