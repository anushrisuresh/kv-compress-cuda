#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>

#define CUDA_CHECK(x) do { \
    cudaError_t err = x;   \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

// ===================== CPU dense =====================
void cpu_dense(
    const float* Q, const float* K, const float* V,
    float* O, int L, int d)
{
    float scale = 1.f / sqrtf((float)d);

    for (int i = 0; i < L; i++) {
        // max logit
        float maxv = -1e30f;
        for (int j = 0; j < L; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            dot *= scale;
            if (dot > maxv) maxv = dot;
        }

        // sum
        float sum = 0.f;
        for (int j = 0; j < L; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            dot *= scale;
            sum += expf(dot - maxv);
        }

        // output
        for (int k = 0; k < d; k++)
            O[i*d + k] = 0;

        for (int j = 0; j < L; j++) {
            float dot = 0.f;
            for (int kk = 0; kk < d; kk++)
                dot += Q[i*d + kk] * K[j*d + kk];
            dot *= scale;
            float w = expf(dot - maxv) / sum;
            for (int k = 0; k < d; k++)
                O[i*d + k] += w * V[j*d + k];
        }
    }
}

// ===================== CPU sparse blockdiag =====================
void cpu_blockdiag(
    const float* Q, const float* K, const float* V,
    float* O, int L, int d, int B)
{
    int NB = L / B;
    float scale = 1.f / sqrtf((float)d);

    for (int i = 0; i < L; i++) {
        int b = i / B;
        int start = b * B;
        int end = start + B;

        float maxv = -1e30f;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            dot *= scale;
            if (dot > maxv) maxv = dot;
        }

        float sum = 0.f;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            dot *= scale;
            sum += expf(dot - maxv);
        }

        for (int k = 0; k < d; k++)
            O[i*d + k] = 0;

        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int kk = 0; kk < d; kk++)
                dot += Q[i*d + kk] * K[j*d + kk];
            dot *= scale;
            float w = expf(dot - maxv) / sum;
            for (int k = 0; k < d; k++)
                O[i*d + k] += w * V[j*d + k];
        }
    }
}

// ===================== GPU dense (1 thread / row) =====================
__global__ void dense_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int L, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;

    float scale = 1.f / sqrtf((float)d);

    float maxv = -1e30f;
    for (int j = 0; j < L; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }

    float sum = 0.f;
    for (int j = 0; j < L; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        dot *= scale;
        sum += expf(dot - maxv);
    }

    for (int k = 0; k < d; k++) O[i*d + k] = 0;

    for (int j = 0; j < L; j++) {
        float dot = 0.f;
        for (int kk = 0; kk < d; kk++)
            dot += Q[i*d + kk] * K[j*d + kk];
        dot *= scale;
        float w = expf(dot - maxv) / sum;
        for (int k = 0; k < d; k++)
            O[i*d + k] += w * V[j*d + k];
    }
}

// ===================== GPU sparse blockdiag =====================
__global__ void blockdiag_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int L, int d, int B)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;

    float scale = 1.f / sqrtf((float)d);

    int b = i / B;
    int start = b * B;
    int end = start + B;

    float maxv = -1e30f;
    for (int j = start; j < end; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        dot *= scale;
        if (dot > maxv) maxv = dot;
    }

    float sum = 0.f;
    for (int j = start; j < end; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        dot *= scale;
        sum += expf(dot - maxv);
    }

    for (int k = 0; k < d; k++) O[i*d + k] = 0;

    for (int j = start; j < end; j++) {
        float dot = 0.f;
        for (int kk = 0; kk < d; kk++)
            dot += Q[i*d + kk] * K[j*d + kk];
        dot *= scale;
        float w = expf(dot - maxv) / sum;
        for (int k = 0; k < d; k++)
            O[i*d + k] += w * V[j*d + k];
    }
}

// ===================== Utility =====================
float max_diff(const float* a, const float* b, int n)
{
    float m = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

// ===================== MAIN =====================
int main(int argc, char** argv)
{
    int L = 512, d = 64, B = 16;
    if (argc > 1) L = atoi(argv[1]);
    if (argc > 2) d = atoi(argv[2]);
    if (argc > 3) B = atoi(argv[3]);

    printf("Config: L=%d, d=%d, B=%d\n", L, d, B);

    std::vector<float> Q(L*d), K(L*d), V(L*d);
    std::vector<float> cpu_dense_out(L*d), cpu_sparse_out(L*d);
    std::vector<float> gpu_dense_out(L*d), gpu_sparse_out(L*d);

    // Random init
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (int i = 0; i < L*d; i++) {
        Q[i] = U(rng);
        K[i] = U(rng);
        V[i] = U(rng);
    }

    // CPU dense
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_dense(Q.data(), K.data(), V.data(), cpu_dense_out.data(), L, d);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu_dense =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU dense (compute only): %.3f ms\n", ms_cpu_dense);

    // CPU sparse
    t0 = std::chrono::high_resolution_clock::now();
    cpu_blockdiag(Q.data(), K.data(), V.data(), cpu_sparse_out.data(), L, d, B);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_cpu_sparse =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("CPU sparse (compute only): %.3f ms\n", ms_cpu_sparse);

    // Allocate GPU
    float *dQ, *dK, *dV, *dO;
    CUDA_CHECK(cudaMalloc(&dQ, L*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dK, L*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dV, L*d*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dO, L*d*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dQ, Q.data(), L*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, K.data(), L*d*sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, V.data(), L*d*sizeof(float), cudaMemcpyHostToDevice));

    int threads = 128;
    int blocks  = (L + threads - 1) / threads;
    int iters   = 50;  // average over many runs for pure compute

    // GPU dense
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++)
        dense_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms_gpu_dense_total;
    cudaEventElapsedTime(&ms_gpu_dense_total, e0, e1);
    float ms_gpu_dense = ms_gpu_dense_total / iters;
    printf("GPU dense (compute only): %.3f ms\n", ms_gpu_dense);

    CUDA_CHECK(cudaMemcpy(gpu_dense_out.data(), dO, L*d*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Dense max diff: %.6f\n", max_diff(cpu_dense_out.data(), gpu_dense_out.data(), L*d));

    // GPU sparse
    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++)
        blockdiag_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d, B);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float ms_gpu_sparse_total;
    cudaEventElapsedTime(&ms_gpu_sparse_total, e0, e1);
    float ms_gpu_sparse = ms_gpu_sparse_total / iters;
    printf("GPU sparse (compute only): %.3f ms\n", ms_gpu_sparse);

    CUDA_CHECK(cudaMemcpy(gpu_sparse_out.data(), dO, L*d*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sparse max diff: %.6f\n", max_diff(cpu_sparse_out.data(), gpu_sparse_out.data(), L*d));

    return 0;
}
