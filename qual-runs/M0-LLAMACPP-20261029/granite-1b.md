ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24092 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24092 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| granitemoe 1B.A400M BF16       |   2.49 GiB |     1.33 B | CUDA       |  99 |          0 |           pp512 |   31779.61 ± 2181.91 |
| granitemoe 1B.A400M BF16       |   2.49 GiB |     1.33 B | CUDA       |  99 |          0 |           tg128 |        567.34 ± 1.05 |
| granitemoe 1B.A400M BF16       |   2.49 GiB |     1.33 B | CUDA       |  99 |         12 |           pp512 |      5594.16 ± 92.77 |
| granitemoe 1B.A400M BF16       |   2.49 GiB |     1.33 B | CUDA       |  99 |         12 |           tg128 |         89.27 ± 1.23 |
| granitemoe 1B.A400M BF16       |   2.49 GiB |     1.33 B | CUDA       |  99 |         24 |           pp512 |      3190.48 ± 40.08 |
| granitemoe 1B.A400M BF16       |   2.49 GiB |     1.33 B | CUDA       |  99 |         24 |           tg128 |         49.98 ± 1.81 |

build: 0893ec9 (1)
