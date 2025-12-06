/**
 * block_topk_sparse.cu
 * Block Top-K sparse attention implementation in CUDA
 * Dynamically selects top-K blocks per query for attention
 */

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <random>
#include <vector>

// Error checking macro
#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cudaError_t err = x;                                                       \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,                     \
             cudaGetErrorString(err));                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// ==================== CPU Helper Functions ====================

// Compute dot product
inline float dot_product(const float *Q, const float *K, int i, int j, int d) {
  float dot = 0.f;
  for (int k = 0; k < d; k++) {
    dot += Q[i * d + k] * K[j * d + k];
  }
  return dot;
}

// Find max logit in a range
float find_max_in_range(const float *Q, const float *K, int i, int start,
                        int end, int d, float scale) {
  float maxv = -1e30f;
  for (int j = start; j < end; j++) {
    float logit = dot_product(Q, K, i, j, d) * scale;
    if (logit > maxv)
      maxv = logit;
  }
  return maxv;
}

// Softmax sum in range
float softmax_sum_range(const float *Q, const float *K, int i, int start,
                        int end, int d, float scale, float maxv) {
  float sum = 0.f;
  for (int j = start; j < end; j++) {
    float logit = dot_product(Q, K, i, j, d) * scale;
    sum += expf(logit - maxv);
  }
  return sum;
}

// Weighted sum of values
void weighted_sum_range(const float *Q, const float *K, const float *V,
                        float *O, int i, int start, int end, int d, float scale,
                        float maxv, float sum) {
  for (int j = start; j < end; j++) {
    float logit = dot_product(Q, K, i, j, d) * scale;
    float w = expf(logit - maxv) / sum;
    for (int k = 0; k < d; k++) {
      O[i * d + k] += w * V[j * d + k];
    }
  }
}

// ==================== CPU Attention Functions ====================

// CPU Dense attention
void cpu_dense(const float *Q, const float *K, const float *V, float *O, int L,
               int d) {
  float scale = 1.f / sqrtf((float)d);
  for (int i = 0; i < L; i++) {
    float maxv = find_max_in_range(Q, K, i, 0, L, d, scale);
    float sum = softmax_sum_range(Q, K, i, 0, L, d, scale, maxv);
    for (int k = 0; k < d; k++)
      O[i * d + k] = 0.f;
    weighted_sum_range(Q, K, V, O, i, 0, L, d, scale, maxv, sum);
  }
}

// CPU Block-diagonal
void cpu_blockdiag(const float *Q, const float *K, const float *V, float *O,
                   int L, int d, int B) {
  assert(L % B == 0);
  float scale = 1.f / sqrtf((float)d);
  for (int i = 0; i < L; i++) {
    int b = i / B;
    int start = b * B, end = start + B;
    float maxv = find_max_in_range(Q, K, i, start, end, d, scale);
    float sum = softmax_sum_range(Q, K, i, start, end, d, scale, maxv);
    for (int k = 0; k < d; k++)
      O[i * d + k] = 0.f;
    weighted_sum_range(Q, K, V, O, i, start, end, d, scale, maxv, sum);
  }
}

// ==================== CPU Top-K Helpers ====================

// Compute block scores
void compute_block_scores(const float *Q, const float *K, int i, int d, int B,
                          int num_blocks, float scale,
                          std::vector<float> &scores) {
  for (int b = 0; b < num_blocks; b++) {
    int start = b * B, end = start + B;
    scores[b] = find_max_in_range(Q, K, i, start, end, d, scale);
  }
}

// Select top-K blocks
void select_topk_blocks(const std::vector<float> &scores, int num_blocks,
                        int Kblocks, std::vector<int> &top_idx,
                        std::vector<float> &top_val) {
  for (int t = 0; t < Kblocks; t++) {
    top_idx[t] = -1;
    top_val[t] = -1e30f;
  }
  for (int b = 0; b < num_blocks; b++) {
    float score = scores[b];
    int min_pos = 0;
    for (int t = 1; t < Kblocks; t++) {
      if (top_val[t] < top_val[min_pos])
        min_pos = t;
    }
    if (score > top_val[min_pos]) {
      top_val[min_pos] = score;
      top_idx[min_pos] = b;
    }
  }
}

// Find max over selected blocks
float find_max_selected(const float *Q, const float *K, int i, int d, int B,
                        float scale, const std::vector<int> &top_idx, int K_) {
  float maxv = -1e30f;
  for (int t = 0; t < K_; t++) {
    int b = top_idx[t];
    if (b < 0)
      continue;
    float m = find_max_in_range(Q, K, i, b * B, b * B + B, d, scale);
    if (m > maxv)
      maxv = m;
  }
  return maxv;
}

// Sum over selected blocks
float sum_selected(const float *Q, const float *K, int i, int d, int B,
                   float scale, float maxv, const std::vector<int> &top_idx,
                   int K_) {
  float sum = 0.f;
  for (int t = 0; t < K_; t++) {
    int b = top_idx[t];
    if (b < 0)
      continue;
    sum += softmax_sum_range(Q, K, i, b * B, b * B + B, d, scale, maxv);
  }
  return sum;
}

// CPU Block Top-K
void cpu_block_topk(const float *Q, const float *K, const float *V, float *O,
                    int L, int d, int B, int Kblocks) {
  assert(L % B == 0);
  int num_blocks = L / B;
  assert(Kblocks <= num_blocks);
  float scale = 1.f / sqrtf((float)d);
  std::vector<float> block_score(num_blocks);
  std::vector<int> top_idx(Kblocks);
  std::vector<float> top_val(Kblocks);

  for (int i = 0; i < L; i++) {
    compute_block_scores(Q, K, i, d, B, num_blocks, scale, block_score);
    select_topk_blocks(block_score, num_blocks, Kblocks, top_idx, top_val);
    float maxv = find_max_selected(Q, K, i, d, B, scale, top_idx, Kblocks);
    float sum = sum_selected(Q, K, i, d, B, scale, maxv, top_idx, Kblocks);
    for (int k = 0; k < d; k++)
      O[i * d + k] = 0.f;
    for (int t = 0; t < Kblocks; t++) {
      int b = top_idx[t];
      if (b < 0)
        continue;
      weighted_sum_range(Q, K, V, O, i, b * B, b * B + B, d, scale, maxv, sum);
    }
  }
}

// ==================== GPU Device Helpers ====================

// Device: block max
__device__ float dev_block_max(const float *Q, const float *K, int i, int start,
                               int end, int d, float scale) {
  float maxv = -1e30f;
  for (int j = start; j < end; j++) {
    float dot = 0.f;
    for (int k = 0; k < d; k++)
      dot += Q[i * d + k] * K[j * d + k];
    if (dot * scale > maxv)
      maxv = dot * scale;
  }
  return maxv;
}

// Device: block sum
__device__ float dev_block_sum(const float *Q, const float *K, int i, int start,
                               int end, int d, float scale, float maxv) {
  float sum = 0.f;
  for (int j = start; j < end; j++) {
    float dot = 0.f;
    for (int k = 0; k < d; k++)
      dot += Q[i * d + k] * K[j * d + k];
    sum += expf(dot * scale - maxv);
  }
  return sum;
}

// Device: weighted sum
__device__ void dev_weighted_sum(const float *Q, const float *K, const float *V,
                                 float *O, int i, int start, int end, int d,
                                 float scale, float maxv, float sum) {
  for (int j = start; j < end; j++) {
    float dot = 0.f;
    for (int kk = 0; kk < d; kk++)
      dot += Q[i * d + kk] * K[j * d + kk];
    float w = expf(dot * scale - maxv) / sum;
    for (int k = 0; k < d; k++)
      O[i * d + k] += w * V[j * d + k];
  }
}

// ==================== GPU Kernels ====================

// Dense kernel
__global__ void dense_kernel(const float *__restrict__ Q,
                             const float *__restrict__ K,
                             const float *__restrict__ V, float *__restrict__ O,
                             int L, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L)
    return;
  float scale = 1.f / sqrtf((float)d);
  float maxv = dev_block_max(Q, K, i, 0, L, d, scale);
  float sum = dev_block_sum(Q, K, i, 0, L, d, scale, maxv);
  for (int k = 0; k < d; k++)
    O[i * d + k] = 0.f;
  dev_weighted_sum(Q, K, V, O, i, 0, L, d, scale, maxv, sum);
}

// Block-diagonal kernel
__global__ void blockdiag_kernel(const float *__restrict__ Q,
                                 const float *__restrict__ K,
                                 const float *__restrict__ V,
                                 float *__restrict__ O, int L, int d, int B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L)
    return;
  float scale = 1.f / sqrtf((float)d);
  int b = i / B;
  int start = b * B, end = start + B;
  float maxv = dev_block_max(Q, K, i, start, end, d, scale);
  float sum = dev_block_sum(Q, K, i, start, end, d, scale, maxv);
  for (int k = 0; k < d; k++)
    O[i * d + k] = 0.f;
  dev_weighted_sum(Q, K, V, O, i, start, end, d, scale, maxv, sum);
}

// Top-K: select best blocks
__device__ void dev_select_topk(const float *Q, const float *K, int i, int d,
                                int B, int num_blocks, float scale, int Kblocks,
                                float *top_val, int *top_idx) {
  for (int t = 0; t < Kblocks; t++) {
    top_val[t] = -1e30f;
    top_idx[t] = -1;
  }
  for (int b = 0; b < num_blocks; b++) {
    float bm = dev_block_max(Q, K, i, b * B, b * B + B, d, scale);
    int min_pos = 0;
    for (int t = 1; t < Kblocks; t++) {
      if (top_val[t] < top_val[min_pos])
        min_pos = t;
    }
    if (bm > top_val[min_pos]) {
      top_val[min_pos] = bm;
      top_idx[min_pos] = b;
    }
  }
}

// Top-K: compute over selected
__device__ void dev_topk_attention(const float *Q, const float *K,
                                   const float *V, float *O, int i, int d,
                                   int B, int Kblocks, float scale,
                                   const int *top_idx) {
  float maxv = -1e30f;
  for (int t = 0; t < Kblocks; t++) {
    int b = top_idx[t];
    if (b < 0)
      continue;
    float m = dev_block_max(Q, K, i, b * B, b * B + B, d, scale);
    if (m > maxv)
      maxv = m;
  }
  float sum = 0.f;
  for (int t = 0; t < Kblocks; t++) {
    int b = top_idx[t];
    if (b < 0)
      continue;
    sum += dev_block_sum(Q, K, i, b * B, b * B + B, d, scale, maxv);
  }
  for (int k = 0; k < d; k++)
    O[i * d + k] = 0.f;
  for (int t = 0; t < Kblocks; t++) {
    int b = top_idx[t];
    if (b < 0)
      continue;
    dev_weighted_sum(Q, K, V, O, i, b * B, b * B + B, d, scale, maxv, sum);
  }
}

// Block Top-K kernel
__global__ void block_topk_kernel(const float *__restrict__ Q,
                                  const float *__restrict__ K,
                                  const float *__restrict__ V,
                                  float *__restrict__ O, int L, int d, int B,
                                  int Kblocks) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L)
    return;

  float scale = 1.f / sqrtf((float)d);
  int num_blocks = L / B;
  const int KMAX = 8;
  if (Kblocks > KMAX)
    Kblocks = KMAX;

  float top_val[KMAX];
  int top_idx[KMAX];

  dev_select_topk(Q, K, i, d, B, num_blocks, scale, Kblocks, top_val, top_idx);
  dev_topk_attention(Q, K, V, O, i, d, B, Kblocks, scale, top_idx);
}

// ==================== Utilities ====================

float max_diff(const float *a, const float *b, int n) {
  float m = 0.f;
  for (int i = 0; i < n; i++)
    m = fmaxf(m, fabsf(a[i] - b[i]));
  return m;
}

void init_random(std::vector<float> &Q, std::vector<float> &K,
                 std::vector<float> &V, int size) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (int i = 0; i < size; i++) {
    Q[i] = U(rng);
    K[i] = U(rng);
    V[i] = U(rng);
  }
}

// Run CPU benchmarks
void run_cpu_benchmarks(const std::vector<float> &Q,
                        const std::vector<float> &K,
                        const std::vector<float> &V, std::vector<float> &out_d,
                        std::vector<float> &out_s, std::vector<float> &out_t,
                        int L, int d, int B, int Kb) {
  auto t0 = std::chrono::high_resolution_clock::now();
  cpu_dense(Q.data(), K.data(), V.data(), out_d.data(), L, d);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("CPU Dense: %.3f ms\n", ms);

  t0 = std::chrono::high_resolution_clock::now();
  cpu_blockdiag(Q.data(), K.data(), V.data(), out_s.data(), L, d, B);
  t1 = std::chrono::high_resolution_clock::now();
  ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("CPU Block-diag: %.3f ms\n", ms);

  t0 = std::chrono::high_resolution_clock::now();
  cpu_block_topk(Q.data(), K.data(), V.data(), out_t.data(), L, d, B, Kb);
  t1 = std::chrono::high_resolution_clock::now();
  ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("CPU Block Top-k: %.3f ms\n", ms);
}

// Run single GPU benchmark
float gpu_benchmark(int L, int d, int threads, int blocks, int iters, float *dQ,
                    float *dK, float *dV, float *dO,
                    void (*kern)(const float *, const float *, const float *,
                                 float *, int, int)) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int it = 0; it < iters; it++)
    kern<<<blocks, threads>>>(dQ, dK, dV, dO, L, d);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return ms / iters;
}

// Run GPU benchmarks
void run_gpu_benchmarks(float *dQ, float *dK, float *dV, float *dO,
                        const std::vector<float> &cpu_d,
                        const std::vector<float> &cpu_s,
                        const std::vector<float> &cpu_t, int L, int d, int B,
                        int Kb) {
  int threads = 128, blocks = (L + threads - 1) / threads, iters = 30;
  size_t bytes = L * d * sizeof(float);
  std::vector<float> gpu_out(L * d);

  float ms =
      gpu_benchmark(L, d, threads, blocks, iters, dQ, dK, dV, dO, dense_kernel);
  printf("\nGPU Dense: %.3f ms\n", ms);
  CUDA_CHECK(cudaMemcpy(gpu_out.data(), dO, bytes, cudaMemcpyDeviceToHost));
  printf("Diff: %.6f\n", max_diff(cpu_d.data(), gpu_out.data(), L * d));

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int it = 0; it < iters; it++)
    blockdiag_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d, B);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float total;
  cudaEventElapsedTime(&total, e0, e1);
  printf("\nGPU Block-diag: %.3f ms\n", total / iters);
  CUDA_CHECK(cudaMemcpy(gpu_out.data(), dO, bytes, cudaMemcpyDeviceToHost));
  printf("Diff: %.6f\n", max_diff(cpu_s.data(), gpu_out.data(), L * d));

  cudaEventRecord(e0);
  for (int it = 0; it < iters; it++)
    block_topk_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d, B, Kb);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  cudaEventElapsedTime(&total, e0, e1);
  printf("\nGPU Block Top-k: %.3f ms\n", total / iters);
  CUDA_CHECK(cudaMemcpy(gpu_out.data(), dO, bytes, cudaMemcpyDeviceToHost));
  printf("Diff: %.6f\n\n", max_diff(cpu_t.data(), gpu_out.data(), L * d));
}

// ==================== Main ====================
int main(int argc, char **argv) {
  int L = 512, d = 64, B = 16, Kb = 2;
  if (argc > 1)
    L = atoi(argv[1]);
  if (argc > 2)
    d = atoi(argv[2]);
  if (argc > 3)
    B = atoi(argv[3]);
  if (argc > 4)
    Kb = atoi(argv[4]);

  assert(L % B == 0);
  int nb = L / B;
  if (Kb > nb)
    Kb = nb;

  printf("\n=== Sparse Attention Benchmark ===\n");
  printf("L=%d, d=%d, B=%d, TopK=%d\n\n", L, d, B, Kb);

  std::vector<float> Q(L * d), K(L * d), V(L * d);
  std::vector<float> cpu_d(L * d), cpu_s(L * d), cpu_t(L * d);
  init_random(Q, K, V, L * d);

  run_cpu_benchmarks(Q, K, V, cpu_d, cpu_s, cpu_t, L, d, B, Kb);

  float *dQ, *dK, *dV, *dO;
  size_t bytes = L * d * sizeof(float);
  CUDA_CHECK(cudaMalloc(&dQ, bytes));
  CUDA_CHECK(cudaMalloc(&dK, bytes));
  CUDA_CHECK(cudaMalloc(&dV, bytes));
  CUDA_CHECK(cudaMalloc(&dO, bytes));
  CUDA_CHECK(cudaMemcpy(dQ, Q.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dK, K.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dV, V.data(), bytes, cudaMemcpyHostToDevice));

  run_gpu_benchmarks(dQ, dK, dV, dO, cpu_d, cpu_s, cpu_t, L, d, B, Kb);

  return 0;
}
