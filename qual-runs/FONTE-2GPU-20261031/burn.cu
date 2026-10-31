// Carga máxima numa GPU: HGEMM 8192 em tensor cores (cuBLAS), em laço, por <segundos>. Usa o device 0 visível.
//   burn <segundos>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
int main(int argc, char ** argv) {
    double dur = argc > 1 ? atof(argv[1]) : 10;
    const int n = 8192;
    __half *a, *b, *c;
    if (cudaMalloc(&a, sizeof(__half) * n * n) || cudaMalloc(&b, sizeof(__half) * n * n) || cudaMalloc(&c, sizeof(__half) * n * n)) { fprintf(stderr, "alocação falhou\n"); return 1; }
    cudaMemset(a, 0x3c, sizeof(__half) * n * n); cudaMemset(b, 0x3c, sizeof(__half) * n * n);
    cublasHandle_t h; if (cublasCreate(&h)) { fprintf(stderr, "cublasCreate falhou\n"); return 1; }
    __half al = __float2half(1e-4f), be = __float2half(0.f);
    auto t0 = std::chrono::steady_clock::now(); long it = 0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < dur) {
        for (int i = 0; i < 10; i++) cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, a, n, b, n, &be, c, n);
        cudaDeviceSynchronize(); it += 10;
    }
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("%ld gemm em %.1f s: %.1f TFLOPS\n", it, s, 2.0 * n * n * n * it / s / 1e12);
    return 0;
}
