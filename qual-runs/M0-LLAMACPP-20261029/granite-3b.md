ggml_cuda_init: found 1 CUDA devices (Total VRAM: 24092 MiB):
  Device 0: NVIDIA GeForce RTX 4090, compute capability 8.9, VMM: yes, VRAM: 24092 MiB
| model                          |       size |     params | backend    | ngl |  n_cpu_moe |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | --: | ---------: | --------------: | -------------------: |
| granitemoe 3B.A800M BF16       |   6.15 GiB |     3.30 B | CUDA       |  99 |          0 |           pp512 |    14825.25 ± 468.72 |
| granitemoe 3B.A800M BF16       |   6.15 GiB |     3.30 B | CUDA       |  99 |          0 |           tg128 |        290.47 ± 3.31 |
| granitemoe 3B.A800M BF16       |   6.15 GiB |     3.30 B | CUDA       |  99 |         16 |           pp512 |      2317.62 ± 43.24 |
| granitemoe 3B.A800M BF16       |   6.15 GiB |     3.30 B | CUDA       |  99 |         16 |           tg128 |         46.40 ± 1.66 |
| granitemoe 3B.A800M BF16       |   6.15 GiB |     3.30 B | CUDA       |  99 |         32 |           pp512 |      1300.52 ± 61.08 |
| granitemoe 3B.A800M BF16       |   6.15 GiB |     3.30 B | CUDA       |  99 |         32 |           tg128 |         27.08 ± 0.46 |

build: 0893ec9 (1)
