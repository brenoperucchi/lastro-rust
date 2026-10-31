// Banda H2D de uma cópia síncrona do tamanho de um expert, a partir de memória pageable (como o cache do llama.cpp, que copia
// do mmap do GGUF) e de memória pinned (como o nosso cache). Mediana de 200 cópias depois de 20 de aquecimento.
//   banda <MiB> [<MiB> ...]
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static double median_ms(void * src, void * dst, size_t n) {
    std::vector<double> t;
    for (int i = 0; i < 220; i++) {
        auto t0 = std::chrono::steady_clock::now();
        cudaMemcpy(dst, src, n, cudaMemcpyHostToDevice);
        double dt = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i >= 20) t.push_back(dt);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

int main(int argc, char ** argv) {
    for (int a = 1; a < argc; a++) {
        size_t n = (size_t) (atof(argv[a]) * 1024 * 1024);
        void * dst;
        void * pinned;
        if (cudaMalloc(&dst, n) != cudaSuccess || cudaMallocHost(&pinned, n) != cudaSuccess) { fprintf(stderr, "alocação falhou\n"); return 1; }
        void * pageable = malloc(n);
        memset(pageable, 1, n);
        memset(pinned, 1, n);
        double pg = median_ms(pageable, dst, n), pn = median_ms(pinned, dst, n);
        printf("%.2f MiB: pageable %.4f ms (%.1f GB/s), pinned %.4f ms (%.1f GB/s)\n", n / 1048576.0, pg, n / pg / 1e6, pn, n / pn / 1e6);
        free(pageable);
        cudaFreeHost(pinned);
        cudaFree(dst);
    }
    return 0;
}
