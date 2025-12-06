//added top-k sparse attention implementation

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

/**********************************************************************
 ***********************  CPU DENSE ATTENTION  ************************
 **********************************************************************/
void cpu_dense(const float* Q, const float* K, const float* V,
               float* O, int L, int d)
{
    float scale = 1.f / sqrtf((float)d);

    for (int i = 0; i < L; i++) {

        // --- 1. Compute max logit for numerical stability ---
        float maxv = -1e30f;
        for (int j = 0; j < L; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            float logit = dot * scale;
            if (logit > maxv) maxv = logit;
        }

        // --- 2. Compute sum of exp(logits - max) ---
        float sum = 0.f;
        for (int j = 0; j < L; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            float logit = dot * scale;
            sum += expf(logit - maxv);
        }

        // --- 3. Weighted sum of V ---
        for (int k = 0; k < d; k++) O[i*d + k] = 0.f;

        for (int j = 0; j < L; j++) {
            float dot = 0.f;
            for (int kk = 0; kk < d; kk++)
                dot += Q[i*d + kk] * K[j*d + kk];
            float logit = dot * scale;
            float w = expf(logit - maxv) / sum;

            for (int k = 0; k < d; k++)
                O[i*d + k] += w * V[j*d + k];
        }
    }
}

/**********************************************************************
 *******************  CPU BLOCK-DIAGONAL SPARSE  **********************
 **********************************************************************/
void cpu_blockdiag(const float* Q, const float* K, const float* V,
                   float* O, int L, int d, int B)
{
    assert(L % B == 0);  // NEW: ensure divisibility
    float scale = 1.f / sqrtf((float)d);

    for (int i = 0; i < L; i++) {

        // determine the block this row belongs to
        int b = i / B;
        int start = b * B;
        int end   = start + B;

        // --- 1. Max logit inside the block ---
        float maxv = -1e30f;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            float logit = dot * scale;
            if (logit > maxv) maxv = logit;
        }

        // --- 2. Sum of exp(logits - max) inside block ---
        float sum = 0.f;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            float logit = dot * scale;
            sum += expf(logit - maxv);
        }

        // --- 3. Weighted sum over only B keys ---
        for (int k = 0; k < d; k++) O[i*d + k] = 0.f;

        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int kk = 0; kk < d; kk++)
                dot += Q[i*d + kk] * K[j*d + kk];
            float logit = dot * scale;
            float w = expf(logit - maxv) / sum;

            for (int k = 0; k < d; k++)
                O[i*d + k] += w * V[j*d + k];
        }
    }
}

/**********************************************************************
 *************  CPU BLOCK TOP-K (DYNAMIC SPARSE) ATTENTION  ***********
 **********************************************************************/
// NEW: block-level top-k CPU reference
void cpu_block_topk(const float* Q, const float* K, const float* V,
                    float* O, int L, int d, int B, int Kblocks)
{
    assert(L % B == 0);
    int num_blocks = L / B;
    assert(Kblocks <= num_blocks);

    float scale = 1.f / sqrtf((float)d);

    // temporary arrays for scores and top-k indices
    std::vector<float> block_score(num_blocks);

    for (int i = 0; i < L; i++) {

        // 1) Compute a score per block (max logit over that block)
        for (int b = 0; b < num_blocks; b++) {
            int start = b * B;
            int end   = start + B;

            float block_max = -1e30f;
            for (int j = start; j < end; j++) {
                float dot = 0.f;
                for (int k = 0; k < d; k++)
                    dot += Q[i*d + k] * K[j*d + k];
                float logit = dot * scale;
                if (logit > block_max) block_max = logit;
            }
            block_score[b] = block_max;
        }

        // 2) Select top-Kblocks blocks by score (simple streaming top-k)
        std::vector<int>   top_idx(Kblocks, -1);
        std::vector<float> top_val(Kblocks, -1e30f);

        for (int b = 0; b < num_blocks; b++) {
            float score = block_score[b];

            // find current smallest in top-k
            int min_pos = 0;
            float min_val = top_val[0];
            for (int t = 1; t < Kblocks; t++) {
                if (top_val[t] < min_val) {
                    min_val = top_val[t];
                    min_pos = t;
                }
            }

            if (score > min_val) {
                top_val[min_pos] = score;
                top_idx[min_pos] = b;
            }
        }

        // 3) Softmax over *only* tokens in the selected blocks

        // 3a) find max over all selected blocks
        float maxv = -1e30f;
        for (int t = 0; t < Kblocks; t++) {
            int b = top_idx[t];
            if (b < 0) continue;
            int start = b * B;
            int end   = start + B;
            for (int j = start; j < end; j++) {
                float dot = 0.f;
                for (int k = 0; k < d; k++)
                    dot += Q[i*d + k] * K[j*d + k];
                float logit = dot * scale;
                if (logit > maxv) maxv = logit;
            }
        }

        // 3b) compute normalization sum over selected blocks
        float sum = 0.f;
        for (int t = 0; t < Kblocks; t++) {
            int b = top_idx[t];
            if (b < 0) continue;
            int start = b * B;
            int end   = start + B;
            for (int j = start; j < end; j++) {
                float dot = 0.f;
                for (int k = 0; k < d; k++)
                    dot += Q[i*d + k] * K[j*d + k];
                float logit = dot * scale;
                sum += expf(logit - maxv);
            }
        }

        // 3c) weighted sum of V over selected blocks
        for (int k = 0; k < d; k++) O[i*d + k] = 0.f;

        for (int t = 0; t < Kblocks; t++) {
            int b = top_idx[t];
            if (b < 0) continue;
            int start = b * B;
            int end   = start + B;
            for (int j = start; j < end; j++) {
                float dot = 0.f;
                for (int kk = 0; kk < d; kk++)
                    dot += Q[i*d + kk] * K[j*d + kk];
                float logit = dot * scale;
                float w = expf(logit - maxv) / sum;

                for (int k = 0; k < d; k++)
                    O[i*d + k] += w * V[j*d + k];
            }
        }
    }
}

/**********************************************************************
 **********************  GPU DENSE ATTENTION  *************************
 **********************************************************************/
__global__ void dense_kernel(const float* __restrict__ Q,
                             const float* __restrict__ K,
                             const float* __restrict__ V,
                             float* __restrict__ O,
                             int L, int d)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;

    float scale = 1.f / sqrtf((float)d);

    // --- 1. Max logit ---
    float maxv = -1e30f;
    for (int j = 0; j < L; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        float logit = dot * scale;
        if (logit > maxv) maxv = logit;
    }

    // --- 2. Sum of exp(logits - max) ---
    float sum = 0.f;
    for (int j = 0; j < L; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        float logit = dot * scale;
        sum += expf(logit - maxv);
    }

    // --- 3. Output ---
    for (int k = 0; k < d; k++) O[i*d + k] = 0.f;

    for (int j = 0; j < L; j++) {
        float dot = 0.f;
        for (int kk = 0; kk < d; kk++)
            dot += Q[i*d + kk] * K[j*d + kk];
        float logit = dot * scale;
        float w = expf(logit - maxv) / sum;

        for (int k = 0; k < d; k++)
            O[i*d + k] += w * V[j*d + k];
    }
}

/**********************************************************************
 ****************  GPU BLOCK-DIAGONAL SPARSE ATTENTION  **************
 **********************************************************************/
__global__ void blockdiag_kernel(const float* __restrict__ Q,
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
    int end   = start + B;

    // --- 1. Max inside block ---
    float maxv = -1e30f;
    for (int j = start; j < end; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        float logit = dot * scale;
        if (logit > maxv) maxv = logit;
    }

    // --- 2. Sum inside block ---
    float sum = 0.f;
    for (int j = start; j < end; j++) {
        float dot = 0.f;
        for (int k = 0; k < d; k++)
            dot += Q[i*d + k] * K[j*d + k];
        float logit = dot * scale;
        sum += expf(logit - maxv);
    }

    // --- 3. Weighted sum ---
    for (int k = 0; k < d; k++) O[i*d + k] = 0.f;

    for (int j = start; j < end; j++) {
        float dot = 0.f;
        for (int kk = 0; kk < d; kk++)
            dot += Q[i*d + kk] * K[j*d + kk];
        float logit = dot * scale;
        float w = expf(logit - maxv) / sum;

        for (int k = 0; k < d; k++)
            O[i*d + k] += w * V[j*d + k];
    }
}

/**********************************************************************
 *************  GPU BLOCK TOP-K (DYNAMIC SPARSE) ATTENTION  ***********
 **********************************************************************/
// NEW: simple per-row block top-k kernel
__global__ void block_topk_kernel(const float* __restrict__ Q,
                                  const float* __restrict__ K,
                                  const float* __restrict__ V,
                                  float* __restrict__ O,
                                  int L, int d, int B, int Kblocks)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L) return;

    float scale = 1.f / sqrtf((float)d);
    int num_blocks = L / B;

    // For simplicity, we cap Kblocks at 8 in this demo
    const int KMAX = 8;
    if (Kblocks > KMAX) Kblocks = KMAX;

    // top-k arrays kept in registers
    float top_val[KMAX];
    int   top_idx[KMAX];
    for (int t = 0; t < Kblocks; t++) {
        top_val[t] = -1e30f;
        top_idx[t] = -1;
    }

    // 1) streaming top-k over blocks using block max score
    for (int b = 0; b < num_blocks; b++) {
        int start = b * B;
        int end   = start + B;

        float block_max = -1e30f;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            float logit = dot * scale;
            if (logit > block_max) block_max = logit;
        }

        // insert into top-k
        int min_pos = 0;
        float min_val = top_val[0];
        for (int t = 1; t < Kblocks; t++) {
            if (top_val[t] < min_val) {
                min_val = top_val[t];
                min_pos = t;
            }
        }
        if (block_max > min_val) {
            top_val[min_pos] = block_max;
            top_idx[min_pos] = b;
        }
    }

    // 2) softmax over selected blocks only

    // 2a) max over selected blocks
    float maxv = -1e30f;
    for (int t = 0; t < Kblocks; t++) {
        int b = top_idx[t];
        if (b < 0) continue;
        int start = b * B;
        int end   = start + B;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            float logit = dot * scale;
            if (logit > maxv) maxv = logit;
        }
    }

    // 2b) sum over selected blocks
    float sum = 0.f;
    for (int t = 0; t < Kblocks; t++) {
        int b = top_idx[t];
        if (b < 0) continue;
        int start = b * B;
        int end   = start + B;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d + k] * K[j*d + k];
            float logit = dot * scale;
            sum += expf(logit - maxv);
        }
    }

    // 2c) weighted sum
    for (int k = 0; k < d; k++) O[i*d + k] = 0.f;

    for (int t = 0; t < Kblocks; t++) {
        int b = top_idx[t];
        if (b < 0) continue;
        int start = b * B;
        int end   = start + B;
        for (int j = start; j < end; j++) {
            float dot = 0.f;
            for (int kk = 0; kk < d; kk++)
                dot += Q[i*d + kk] * K[j*d + kk];
            float logit = dot * scale;
            float w = expf(logit - maxv) / sum;

            for (int k = 0; k < d; k++)
                O[i*d + k] += w * V[j*d + k];
        }
    }
}

/**********************************************************************
 ***************************  UTILITIES  ******************************
 **********************************************************************/
float max_diff(const float* a, const float* b, int n)
{
    float m = 0.f;
    for (int i = 0; i < n; i++)
        m = fmaxf(m, fabsf(a[i] - b[i]));
    return m;
}

/**********************************************************************
 ******************************* MAIN ********************************
 **********************************************************************/
int main(int argc, char** argv)
{
    int L = 512, d = 64, B = 16;
    int Kblocks = 2;               // NEW: number of key blocks per query

    if (argc > 1) L = atoi(argv[1]);
    if (argc > 2) d = atoi(argv[2]);
    if (argc > 3) B = atoi(argv[3]);
    if (argc > 4) Kblocks = atoi(argv[4]);

    assert(L % B == 0);
    int num_blocks = L / B;
    if (Kblocks > num_blocks) Kblocks = num_blocks;

    printf("\n=== Sparse Attention Benchmark ===\n");
    printf("Sequence Length L = %d\nEmbedding Dim d = %d\nBlock Size B = %d\nTop-k Blocks = %d\n\n",
           L, d, B, Kblocks);

    // Allocate host memory
    std::vector<float> Q(L*d), K(L*d), V(L*d);
    std::vector<float> cpu_dense_out(L*d), cpu_sparse_out(L*d);
    std::vector<float> gpu_dense_out(L*d), gpu_sparse_out(L*d);

    // NEW: outputs for block top-k
    std::vector<float> cpu_topk_out(L*d), gpu_topk_out(L*d);

    // Initialize random tensors
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    for (int i = 0; i < L*d; i++) {
        Q[i] = U(rng);
        K[i] = U(rng);
        V[i] = U(rng);
    }

    /*********************** CPU Dense ***********************/
    auto t0 = std::chrono::high_resolution_clock::now();
    cpu_dense(Q.data(), K.data(), V.data(), cpu_dense_out.data(), L, d);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_dense_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("CPU Dense  Time: %.3f ms\n", cpu_dense_ms);

    /*********************** CPU Sparse (Block-diag) *********/
    t0 = std::chrono::high_resolution_clock::now();
    cpu_blockdiag(Q.data(), K.data(), V.data(), cpu_sparse_out.data(), L, d, B);
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_sparse_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("CPU Sparse (Block-diag) Time: %.3f ms\n", cpu_sparse_ms);

    /*********************** CPU Block Top-k *****************/
    t0 = std::chrono::high_resolution_clock::now();
    cpu_block_topk(Q.data(), K.data(), V.data(), cpu_topk_out.data(),
                   L, d, B, Kblocks);
    t1 = std::chrono::high_resolution_clock::now();
    double cpu_topk_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("CPU Block Top-k Time: %.3f ms\n", cpu_topk_ms);

    /*********************** GPU Setup ************************/
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
    int iters   = 30;        // average to isolate pure compute

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    /*********************** GPU Dense ************************/
    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++)
        dense_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float total_dense_ms;
    cudaEventElapsedTime(&total_dense_ms, e0, e1);
    float gpu_dense_ms = total_dense_ms / iters;

    CUDA_CHECK(cudaMemcpy(gpu_dense_out.data(), dO, L*d*sizeof(float), cudaMemcpyDeviceToHost));
    float diff_dense = max_diff(cpu_dense_out.data(), gpu_dense_out.data(), L*d);

    printf("\nGPU Dense  Time: %.3f ms\n", gpu_dense_ms);
    printf("Dense Max Diff: %.6f\n", diff_dense);

    /*********************** GPU Sparse (Block-diag) *********/
    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++)
        blockdiag_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d, B);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float total_sparse_ms;
    cudaEventElapsedTime(&total_sparse_ms, e0, e1);
    float gpu_sparse_ms = total_sparse_ms / iters;

    CUDA_CHECK(cudaMemcpy(gpu_sparse_out.data(), dO, L*d*sizeof(float), cudaMemcpyDeviceToHost));
    float diff_sparse = max_diff(cpu_sparse_out.data(), gpu_sparse_out.data(), L*d);

    printf("\nGPU Sparse (Block-diag) Time: %.3f ms\n", gpu_sparse_ms);
    printf("Sparse (Block-diag) Max Diff: %.6f\n", diff_sparse);

    /*********************** GPU Block Top-k *****************/
    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++)
        block_topk_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d, B, Kblocks);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);

    float total_topk_ms;
    cudaEventElapsedTime(&total_topk_ms, e0, e1);
    float gpu_topk_ms = total_topk_ms / iters;

    CUDA_CHECK(cudaMemcpy(gpu_topk_out.data(), dO, L*d*sizeof(float), cudaMemcpyDeviceToHost));
    float diff_topk = max_diff(cpu_topk_out.data(), gpu_topk_out.data(), L*d);

    printf("\nGPU Block Top-k Time: %.3f ms\n", gpu_topk_ms);
    printf("Block Top-k Max Diff: %.6f\n\n", diff_topk);

    /****************************************************************
     *********************** SPEEDUP REPORT *************************
     ****************************************************************/
    printf("========= SPEEDUP REPORT =========\n");
    printf("GPU Dense Speedup          : %.2f×\n", cpu_dense_ms / gpu_dense_ms);
    printf("GPU Block-diag Speedup     : %.2f×\n", cpu_sparse_ms / gpu_sparse_ms);
    printf("GPU Block Top-k Speedup    : %.2f×\n", cpu_topk_ms / gpu_topk_ms);
    printf("----------------------------------\n");
    printf("Sparse vs Dense (CPU)      : block-diag %.2f× faster, top-k %.2f× faster\n",
           cpu_dense_ms / cpu_sparse_ms, cpu_dense_ms / cpu_topk_ms);
    printf("Sparse vs Dense (GPU)      : block-diag %.2f× faster, top-k %.2f× faster\n",
           gpu_dense_ms / gpu_sparse_ms, gpu_dense_ms / gpu_topk_ms);
    printf("==================================\n\n");

    return 0;
}
