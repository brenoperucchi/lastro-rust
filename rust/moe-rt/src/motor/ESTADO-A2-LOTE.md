# Estado do A2-LOTE (microbench de GEMM em lote de um expert), 3/11

Pergunta: quanto custa calcular um expert do Granite para n = 1 a 32 posições juntas, na GPU (4090) e na CPU de 8 threads, para trocar os dois cenários do A2-MULTIPOS-SIM por um custo medido.

Feito: critério v1.1 (`qual-runs/A2-LOTE-20261103/criterio.txt`), `lote_gpu.cu` (CUDA C++), `rust/moe-rt/src/bin/lote-cpu.rs`, `lote.py` e `test_lote.py` (27 testes), revisados nível 2 nas rodadas 53 a 55 antes de qualquer GPU (rev-1 aprovou na 55; o único ponto de teste do rev-2 foi fechado com mutante). Uma janela de GPU e CPU em 3/11.

Resultado: **portão de representatividade reprovado** (GPU 3B em n=1: 7,65 us contra 5,19 do motor, +47% com limite ±40%), então a decisão é "O MICROBENCH NÃO REPRESENTA O MOTOR" e a tabela não entra no simulador. Razões informativas: w(8) ≈ 1,7 (GPU) e 1,2 (CPU). Detalhe em `qual-runs/A2-LOTE-20261103/resultado.txt`.

Próximo: critério v1.2 (kernel de GPU mais próximo do cutile do motor no 3B, sem mudar limiares nem a regra) e segunda janela; precisa de aval do Breno. Alternativa: medir o kernel cutile do próprio motor em lote.
