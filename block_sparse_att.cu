/**
 * block_sparse_att.cu
 * Block-diagonal sparse attention implementation in CUDA
 * Compares CPU vs GPU performance for dense and sparse attention
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

// Compute dot product between query and key vectors
inline float compute_dot(const float *Q, const float *K, int i, int j, int d) {
  float dot = 0.f;
  for (int k = 0; k < d; k++) {
    dot += Q[i * d + k] * K[j * d + k];
  }
  return dot;
}

// Find max logit over a range of keys
float find_max_logit(const float *Q, const float *K, int i, int start, int end,
                     int d, float scale) {
  float maxv = -1e30f;
  for (int j = start; j < end; j++) {
    float logit = compute_dot(Q, K, i, j, d) * scale;
    if (logit > maxv)
      maxv = logit;
  }
  return maxv;
}

// Compute softmax denominator over a range
float compute_softmax_sum(const float *Q, const float *K, int i, int start,
                          int end, int d, float scale, float maxv) {
  float sum = 0.f;
  for (int j = start; j < end; j++) {
    float logit = compute_dot(Q, K, i, j, d) * scale;
    sum += expf(logit - maxv);
  }
  return sum;
}

// Compute weighted sum of values
void compute_weighted_sum(const float *Q, const float *K, const float *V,
                          float *O, int i, int start, int end, int d,
                          float scale, float maxv, float sum) {
  for (int k = 0; k < d; k++) {
    O[i * d + k] = 0;
  }
  for (int j = start; j < end; j++) {
    float logit = compute_dot(Q, K, i, j, d) * scale;
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
    float maxv = find_max_logit(Q, K, i, 0, L, d, scale);
    float sum = compute_softmax_sum(Q, K, i, 0, L, d, scale, maxv);
    compute_weighted_sum(Q, K, V, O, i, 0, L, d, scale, maxv, sum);
  }
}

// CPU Block-diagonal sparse attention
void cpu_blockdiag(const float *Q, const float *K, const float *V, float *O,
                   int L, int d, int B) {
  float scale = 1.f / sqrtf((float)d);
  for (int i = 0; i < L; i++) {
    int b = i / B;
    int start = b * B;
    int end = start + B;
    float maxv = find_max_logit(Q, K, i, start, end, d, scale);
    float sum = compute_softmax_sum(Q, K, i, start, end, d, scale, maxv);
    compute_weighted_sum(Q, K, V, O, i, start, end, d, scale, maxv, sum);
  }
}

// ==================== GPU Device Helpers ====================

// Find max logit (device)
__device__ float dev_find_max(const float *Q, const float *K, int i, int start,
                              int end, int d, float scale) {
  float maxv = -1e30f;
  for (int j = start; j < end; j++) {
    float dot = 0.f;
    for (int k = 0; k < d; k++) {
      dot += Q[i * d + k] * K[j * d + k];
    }
    if (dot * scale > maxv)
      maxv = dot * scale;
  }
  return maxv;
}

// Compute softmax sum (device)
__device__ float dev_softmax_sum(const float *Q, const float *K, int i,
                                 int start, int end, int d, float scale,
                                 float maxv) {
  float sum = 0.f;
  for (int j = start; j < end; j++) {
    float dot = 0.f;
    for (int k = 0; k < d; k++) {
      dot += Q[i * d + k] * K[j * d + k];
    }
    sum += expf(dot * scale - maxv);
  }
  return sum;
}

// Compute weighted sum (device)
__device__ void dev_weighted_sum(const float *Q, const float *K, const float *V,
                                 float *O, int i, int start, int end, int d,
                                 float scale, float maxv, float sum) {
  for (int k = 0; k < d; k++)
    O[i * d + k] = 0;
  for (int j = start; j < end; j++) {
    float dot = 0.f;
    for (int kk = 0; kk < d; kk++) {
      dot += Q[i * d + kk] * K[j * d + kk];
    }
    float w = expf(dot * scale - maxv) / sum;
    for (int k = 0; k < d; k++) {
      O[i * d + k] += w * V[j * d + k];
    }
  }
}

// ==================== GPU Kernels ====================

// GPU Dense attention kernel
__global__ void dense_kernel(const float *__restrict__ Q,
                             const float *__restrict__ K,
                             const float *__restrict__ V, float *__restrict__ O,
                             int L, int d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L)
    return;

  float scale = 1.f / sqrtf((float)d);
  float maxv = dev_find_max(Q, K, i, 0, L, d, scale);
  float sum = dev_softmax_sum(Q, K, i, 0, L, d, scale, maxv);
  dev_weighted_sum(Q, K, V, O, i, 0, L, d, scale, maxv, sum);
}

// GPU Block-diagonal kernel
__global__ void blockdiag_kernel(const float *__restrict__ Q,
                                 const float *__restrict__ K,
                                 const float *__restrict__ V,
                                 float *__restrict__ O, int L, int d, int B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= L)
    return;

  float scale = 1.f / sqrtf((float)d);
  int b = i / B;
  int start = b * B;
  int end = start + B;

  float maxv = dev_find_max(Q, K, i, start, end, d, scale);
  float sum = dev_softmax_sum(Q, K, i, start, end, d, scale, maxv);
  dev_weighted_sum(Q, K, V, O, i, start, end, d, scale, maxv, sum);
}

// ==================== Utility Functions ====================

float max_diff(const float *a, const float *b, int n) {
  float m = 0;
  for (int i = 0; i < n; i++) {
    float d = fabsf(a[i] - b[i]);
    if (d > m)
      m = d;
  }
  return m;
}

void init_random_tensors(std::vector<float> &Q, std::vector<float> &K,
                         std::vector<float> &V, int size) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> U(-1.f, 1.f);
  for (int i = 0; i < size; i++) {
    Q[i] = U(rng);
    K[i] = U(rng);
    V[i] = U(rng);
  }
}

void setup_gpu_memory(float **dQ, float **dK, float **dV, float **dO,
                      const std::vector<float> &Q, const std::vector<float> &K,
                      const std::vector<float> &V, int L, int d) {
  size_t bytes = L * d * sizeof(float);
  CUDA_CHECK(cudaMalloc(dQ, bytes));
  CUDA_CHECK(cudaMalloc(dK, bytes));
  CUDA_CHECK(cudaMalloc(dV, bytes));
  CUDA_CHECK(cudaMalloc(dO, bytes));
  CUDA_CHECK(cudaMemcpy(*dQ, Q.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*dK, K.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(*dV, V.data(), bytes, cudaMemcpyHostToDevice));
}

float benchmark_kernel(void (*kern)(const float *, const float *, const float *,
                                    float *, int, int),
                       float *dQ, float *dK, float *dV, float *dO, int L, int d,
                       int threads, int blocks, int iters) {
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int it = 0; it < iters; it++) {
    kern<<<blocks, threads>>>(dQ, dK, dV, dO, L, d);
  }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float total_ms;
  cudaEventElapsedTime(&total_ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  return total_ms / iters;
}

// Run CPU benchmarks
void run_cpu_benchmarks(const std::vector<float> &Q,
                        const std::vector<float> &K,
                        const std::vector<float> &V,
                        std::vector<float> &cpu_dense_out,
                        std::vector<float> &cpu_sparse_out, int L, int d,
                        int B) {
  auto t0 = std::chrono::high_resolution_clock::now();
  cpu_dense(Q.data(), K.data(), V.data(), cpu_dense_out.data(), L, d);
  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("CPU dense: %.3f ms\n", ms);

  t0 = std::chrono::high_resolution_clock::now();
  cpu_blockdiag(Q.data(), K.data(), V.data(), cpu_sparse_out.data(), L, d, B);
  t1 = std::chrono::high_resolution_clock::now();
  ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("CPU sparse: %.3f ms\n", ms);
}

// Run GPU benchmarks
void run_gpu_benchmarks(float *dQ, float *dK, float *dV, float *dO,
                        std::vector<float> &gpu_dense_out,
                        std::vector<float> &gpu_sparse_out,
                        const std::vector<float> &cpu_dense_out,
                        const std::vector<float> &cpu_sparse_out, int L, int d,
                        int B) {
  int threads = 128;
  int blocks = (L + threads - 1) / threads;
  int iters = 50;
  size_t bytes = L * d * sizeof(float);

  float ms = benchmark_kernel(dense_kernel, dQ, dK, dV, dO, L, d, threads,
                              blocks, iters);
  printf("GPU dense: %.3f ms\n", ms);
  CUDA_CHECK(
      cudaMemcpy(gpu_dense_out.data(), dO, bytes, cudaMemcpyDeviceToHost));
  printf("Dense diff: %.6f\n",
         max_diff(cpu_dense_out.data(), gpu_dense_out.data(), L * d));

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int it = 0; it < iters; it++) {
    blockdiag_kernel<<<blocks, threads>>>(dQ, dK, dV, dO, L, d, B);
  }
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float total_ms;
  cudaEventElapsedTime(&total_ms, e0, e1);
  printf("GPU sparse: %.3f ms\n", total_ms / iters);

  CUDA_CHECK(
      cudaMemcpy(gpu_sparse_out.data(), dO, bytes, cudaMemcpyDeviceToHost));
  printf("Sparse diff: %.6f\n",
         max_diff(cpu_sparse_out.data(), gpu_sparse_out.data(), L * d));
}

// ==================== Main ====================
int main(int argc, char **argv) {
  int L = 512, d = 64, B = 16;
  if (argc > 1)
    L = atoi(argv[1]);
  if (argc > 2)
    d = atoi(argv[2]);
  if (argc > 3)
    B = atoi(argv[3]);

  printf("Config: L=%d, d=%d, B=%d\n", L, d, B);

  std::vector<float> Q(L * d), K(L * d), V(L * d);
  std::vector<float> cpu_dense_out(L * d), cpu_sparse_out(L * d);
  std::vector<float> gpu_dense_out(L * d), gpu_sparse_out(L * d);

  init_random_tensors(Q, K, V, L * d);
  run_cpu_benchmarks(Q, K, V, cpu_dense_out, cpu_sparse_out, L, d, B);

  float *dQ, *dK, *dV, *dO;
  setup_gpu_memory(&dQ, &dK, &dV, &dO, Q, K, V, L, d);

  run_gpu_benchmarks(dQ, dK, dV, dO, gpu_dense_out, gpu_sparse_out,
                     cpu_dense_out, cpu_sparse_out, L, d, B);

  return 0;
}
