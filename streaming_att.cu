/**
 * streaming_att.cu
 * Streaming attention with sink tokens and sliding window
 * Compares dense vs sparse streaming attention in CUDA
 */

#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>

// Custom min/max for nvcc compatibility
template <typename T> __host__ __device__ inline T min_val(T a, T b) {
  return (a < b) ? a : b;
}

template <typename T> __host__ __device__ inline T max_val(T a, T b) {
  return (a > b) ? a : b;
}

// Error checking macro
#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(err));                                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Global constants
constexpr int B = 1;
constexpr int C = 384;
constexpr int MAX_T = 4096;
constexpr int MAX_KEYS = 512;

// ==================== Device Helpers ====================

// Compute dot product (device)
__device__ float dev_dot(const float *q, const float *k, int C_rt) {
  float dot = 0.0f;
  for (int c = 0; c < C_rt; ++c) {
    dot += q[c] * k[c];
  }
  return dot;
}

// Compute scores and find max (device helper)
__device__ float dev_compute_scores(const float *Q, const float *K, int b,
                                    int t_q, int T_rt, int C_rt, float scale,
                                    float *scores) {
  const float *q_vec = &Q[(b * T_rt + t_q) * C_rt];
  float max_score = -1e30f;
  for (int t_k = 0; t_k < T_rt; ++t_k) {
    const float *k_vec = &K[(b * T_rt + t_k) * C_rt];
    float s = dev_dot(q_vec, k_vec, C_rt) * scale;
    scores[t_k] = s;
    if (s > max_score)
      max_score = s;
  }
  return max_score;
}

// Softmax denominator (device helper)
__device__ float dev_softmax_denom(float *scores, int len, float max_score) {
  float denom = 0.0f;
  for (int i = 0; i < len; ++i) {
    float e = expf(scores[i] - max_score);
    scores[i] = e;
    denom += e;
  }
  return denom;
}

// Weighted sum (device helper)
__device__ void dev_weighted_sum(const float *V, float *out,
                                 const float *scores, float inv_denom, int b,
                                 int T_rt, int C_rt) {
  for (int c = 0; c < C_rt; ++c) {
    float acc = 0.0f;
    for (int t_k = 0; t_k < T_rt; ++t_k) {
      float w = scores[t_k] * inv_denom;
      const float *v_vec = &V[(b * T_rt + t_k) * C_rt];
      acc += w * v_vec[c];
    }
    out[c] = acc;
  }
}

// ==================== Dense Attention Kernel ====================
__global__ void dense_attention_kernel(const float *__restrict__ Q,
                                       const float *__restrict__ K,
                                       const float *__restrict__ V,
                                       float *__restrict__ O, int B_rt,
                                       int T_rt, int C_rt, float scale) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= B_rt * T_rt)
    return;

  int b = row / T_rt;
  int t_q = row % T_rt;
  float *out_vec = &O[(b * T_rt + t_q) * C_rt];
  float scores[MAX_T];

  float max_s = dev_compute_scores(Q, K, b, t_q, T_rt, C_rt, scale, scores);
  float denom = dev_softmax_denom(scores, T_rt, max_s);
  dev_weighted_sum(V, out_vec, scores, 1.0f / denom, b, T_rt, C_rt);
}

// ==================== Streaming Helpers ====================

// Build key list for streaming pattern
__device__ int build_key_list(int t_q, int sink_len, int window_size, int T_rt,
                              int *key_idx) {
  int len = 0;
  if (sink_len > 0) {
    int sink_end = (t_q < sink_len - 1) ? t_q : (sink_len - 1);
    for (int j = 0; j <= sink_end; ++j) {
      key_idx[len++] = j;
    }
  }
  if (t_q >= sink_len) {
    int t_start = t_q - window_size + 1;
    if (t_start < sink_len)
      t_start = sink_len;
    for (int j = t_start; j <= t_q; ++j) {
      if (len < MAX_KEYS)
        key_idx[len++] = j;
    }
  }
  return len;
}

// Compute sparse scores
__device__ float dev_sparse_scores(const float *Q, const float *K, int b,
                                   int t_q, int T_rt, int C_rt, float scale,
                                   float *scores, const int *key_idx, int len) {
  const float *q_vec = &Q[(b * T_rt + t_q) * C_rt];
  float max_s = -1e30f;
  for (int idx = 0; idx < len; ++idx) {
    int t_k = key_idx[idx];
    const float *k_vec = &K[(b * T_rt + t_k) * C_rt];
    float s = dev_dot(q_vec, k_vec, C_rt) * scale;
    scores[idx] = s;
    if (s > max_s)
      max_s = s;
  }
  return max_s;
}

// Sparse weighted sum
__device__ void dev_sparse_weighted_sum(const float *V, float *out,
                                        const float *scores, float inv_denom,
                                        const int *key_idx, int len, int b,
                                        int T_rt, int C_rt) {
  for (int c = 0; c < C_rt; ++c) {
    float acc = 0.0f;
    for (int idx = 0; idx < len; ++idx) {
      int t_k = key_idx[idx];
      float w = scores[idx] * inv_denom;
      const float *v_vec = &V[(b * T_rt + t_k) * C_rt];
      acc += w * v_vec[c];
    }
    out[c] = acc;
  }
}

// ==================== Streaming Attention Kernel ====================
__global__ void streaming_attention_kernel(const float *__restrict__ Q,
                                           const float *__restrict__ K,
                                           const float *__restrict__ V,
                                           float *__restrict__ O, int B_rt,
                                           int T_rt, int C_rt, int window_size,
                                           int sink_size, float scale) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= B_rt * T_rt)
    return;

  int b = row / T_rt;
  int t_q = row % T_rt;
  float *out_vec = &O[(b * T_rt + t_q) * C_rt];

  float scores[MAX_KEYS];
  int key_idx[MAX_KEYS];

  int sink_len = (sink_size < T_rt) ? sink_size : T_rt;
  int len = build_key_list(t_q, sink_len, window_size, T_rt, key_idx);
  if (len > MAX_KEYS)
    len = MAX_KEYS;

  float max_s =
      dev_sparse_scores(Q, K, b, t_q, T_rt, C_rt, scale, scores, key_idx, len);
  float denom = dev_softmax_denom(scores, len, max_s);
  dev_sparse_weighted_sum(V, out_vec, scores, 1.0f / denom, key_idx, len, b,
                          T_rt, C_rt);
}

// ==================== Host Utilities ====================

void init_tensors(std::vector<float> &Q, std::vector<float> &K,
                  std::vector<float> &V, int n) {
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
  for (int i = 0; i < n; ++i) {
    Q[i] = dist(rng);
    K[i] = dist(rng);
    V[i] = dist(rng);
  }
}

void alloc_gpu(float **d_Q, float **d_K, float **d_V, float **d_O_dense,
               float **d_O_stream, size_t bytes) {
  CHECK_CUDA(cudaMalloc(d_Q, bytes));
  CHECK_CUDA(cudaMalloc(d_K, bytes));
  CHECK_CUDA(cudaMalloc(d_V, bytes));
  CHECK_CUDA(cudaMalloc(d_O_dense, bytes));
  CHECK_CUDA(cudaMalloc(d_O_stream, bytes));
}

float run_dense_bench(float *d_Q, float *d_K, float *d_V, float *d_O,
                      int blocks, int threads, int T_rt, float scale, int n) {
  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  for (int i = 0; i < n; ++i) {
    dense_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, B, T_rt, C,
                                                scale);
  }
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  return ms / n;
}

float run_stream_bench(float *d_Q, float *d_K, float *d_V, float *d_O,
                       int blocks, int threads, int T_rt, int ws, int ss,
                       float scale, int n) {
  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  for (int i = 0; i < n; ++i) {
    streaming_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_O, B, T_rt,
                                                    C, ws, ss, scale);
  }
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  return ms / n;
}

void cleanup_gpu(float *d_Q, float *d_K, float *d_V, float *d_O_dense,
                 float *d_O_stream) {
  CHECK_CUDA(cudaFree(d_Q));
  CHECK_CUDA(cudaFree(d_K));
  CHECK_CUDA(cudaFree(d_V));
  CHECK_CUDA(cudaFree(d_O_dense));
  CHECK_CUDA(cudaFree(d_O_stream));
}

// Parse command line arguments
bool parse_args(int argc, char **argv, int &T_rt, int &ws, int &ss) {
  T_rt = 1024;
  ws = 64;
  ss = 16;
  if (argc >= 2)
    T_rt = std::atoi(argv[1]);
  if (argc >= 3)
    ws = std::atoi(argv[2]);
  if (argc >= 4)
    ss = std::atoi(argv[3]);

  if (T_rt > MAX_T) {
    std::cerr << "Error: T=" << T_rt << " > MAX_T=" << MAX_T << "\n";
    return false;
  }
  if (ws + ss > MAX_KEYS) {
    std::cerr << "Error: window+sink > MAX_KEYS\n";
    return false;
  }
  return true;
}

// Run warmup
void warmup(float *d_Q, float *d_K, float *d_V, float *d_O_dense,
            float *d_O_stream, int blocks, int threads, int T_rt, int ws,
            int ss, float scale) {
  dense_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_O_dense, B, T_rt,
                                              C, scale);
  streaming_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_O_stream, B,
                                                  T_rt, C, ws, ss, scale);
  CHECK_CUDA(cudaDeviceSynchronize());
}

// Print results
void print_results(float ms_d, float ms_s, const std::vector<float> &h_d,
                   const std::vector<float> &h_s) {
  std::cout << "Dense avg: " << ms_d << " ms\n";
  std::cout << "Stream avg: " << ms_s << " ms\n";
  std::cout << "Speedup: " << (ms_d / ms_s) << "x\n";
  std::cout << "Output[0]: dense=" << h_d[0] << ", stream=" << h_s[0] << "\n";
}

// Setup GPU and copy data
void setup_gpu_data(float **d_Q, float **d_K, float **d_V, float **d_O_d,
                    float **d_O_s, const std::vector<float> &h_Q,
                    const std::vector<float> &h_K,
                    const std::vector<float> &h_V, size_t bytes) {
  alloc_gpu(d_Q, d_K, d_V, d_O_d, d_O_s, bytes);
  CHECK_CUDA(cudaMemcpy(*d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(*d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(*d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));
}

// Run benchmarks and copy results
void run_benchmarks(float *d_Q, float *d_K, float *d_V, float *d_O_d,
                    float *d_O_s, std::vector<float> &h_O_d,
                    std::vector<float> &h_O_s, int T_rt, int ws, int ss,
                    size_t bytes) {
  int threads = 128;
  int blocks = (B * T_rt + threads - 1) / threads;
  float scale = 1.0f / std::sqrt(static_cast<float>(C));

  warmup(d_Q, d_K, d_V, d_O_d, d_O_s, blocks, threads, T_rt, ws, ss, scale);

  float ms_d =
      run_dense_bench(d_Q, d_K, d_V, d_O_d, blocks, threads, T_rt, scale, 10);
  float ms_s = run_stream_bench(d_Q, d_K, d_V, d_O_s, blocks, threads, T_rt, ws,
                                ss, scale, 10);

  CHECK_CUDA(cudaMemcpy(h_O_d.data(), d_O_d, bytes, cudaMemcpyDeviceToHost));
  CHECK_CUDA(cudaMemcpy(h_O_s.data(), d_O_s, bytes, cudaMemcpyDeviceToHost));

  print_results(ms_d, ms_s, h_O_d, h_O_s);
}

// ==================== Main ====================
int main(int argc, char **argv) {
  int T_rt, ws, ss;
  if (!parse_args(argc, argv, T_rt, ws, ss))
    return 1;

  std::cout << "Dense vs Streaming: T=" << T_rt << ", W=" << ws << ", S=" << ss
            << "\n";

  const int n = B * T_rt * C;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);

  std::vector<float> h_Q(n), h_K(n), h_V(n);
  std::vector<float> h_O_d(n), h_O_s(n);
  init_tensors(h_Q, h_K, h_V, n);

  float *d_Q, *d_K, *d_V, *d_O_d, *d_O_s;
  setup_gpu_data(&d_Q, &d_K, &d_V, &d_O_d, &d_O_s, h_Q, h_K, h_V, bytes);
  run_benchmarks(d_Q, d_K, d_V, d_O_d, d_O_s, h_O_d, h_O_s, T_rt, ws, ss,
                 bytes);
  cleanup_gpu(d_Q, d_K, d_V, d_O_d, d_O_s);

  return 0;
}