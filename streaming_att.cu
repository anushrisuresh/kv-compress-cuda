// attention_compare.cu
// Compare dense vs "streaming" (sinks + sliding-window) attention in CUDA
// Runtime T, WINDOW_SIZE, SINK_SIZE; fixed B=1, C=384

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

// Use our own min/max to avoid std::algorithm issues with nvcc
template<typename T>
__host__ __device__ inline T min_val(T a, T b) { return (a < b) ? a : b; }

template<typename T>
__host__ __device__ inline T max_val(T a, T b) { return (a > b) ? a : b; }

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ---------- Fixed global limits (to keep local arrays compile-time-sized) ----------
constexpr int B = 1;          // batch size fixed to 1
constexpr int C = 384;        // channel / head dimension
constexpr int MAX_T = 4096;   // maximum supported sequence length
constexpr int MAX_KEYS = 512; // max sink+window keys per query (must >= window+sink)

// ---------------------- Dense attention kernel ---------------------- //
// Q, K, V, O are all [B, T, C] flattened as row-major: ((b*T + t)*C + c)

__global__ void dense_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B_runtime, int T_runtime, int C_runtime,
    float scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // each thread = one (b, t_query)
    int total_rows = B_runtime * T_runtime;
    if (row >= total_rows) return;

    int b   = row / T_runtime;
    int t_q = row % T_runtime;

    const float* q_vec = &Q[(b * T_runtime + t_q) * C_runtime];
    float*       out_vec = &O[(b * T_runtime + t_q) * C_runtime];

    // Local buffer for scores over all keys (up to MAX_T)
    float scores_local[MAX_T];

    // 1) Compute raw scores = q · k_j / sqrt(C)
    float max_score = -1e30f;
    for (int t_k = 0; t_k < T_runtime; ++t_k) {
        const float* k_vec = &K[(b * T_runtime + t_k) * C_runtime];
        float dot = 0.0f;
        for (int c = 0; c < C_runtime; ++c) {
            dot += q_vec[c] * k_vec[c];
        }
        float s = dot * scale;
        scores_local[t_k] = s;
        if (s > max_score) max_score = s;
    }

    // 2) Softmax denominator
    float denom = 0.0f;
    for (int t_k = 0; t_k < T_runtime; ++t_k) {
        float e = expf(scores_local[t_k] - max_score);
        scores_local[t_k] = e;  // store exp(score - max) for reuse
        denom += e;
    }
    float inv_denom = 1.0f / denom;

    // 3) Weighted sum over V
    for (int c = 0; c < C_runtime; ++c) {
        float acc = 0.0f;
        for (int t_k = 0; t_k < T_runtime; ++t_k) {
            float w = scores_local[t_k] * inv_denom; // softmax weight
            const float* v_vec = &V[(b * T_runtime + t_k) * C_runtime];
            acc += w * v_vec[c];
        }
        out_vec[c] = acc;
    }
}

// ------------------ Streaming (sparse) attention kernel ------------------ //
// "Streaming" = sink tokens + sliding window:
//
// For a query at position t_q:
//   - If t_q < sink_size: attends to positions [0..t_q] (normal causal).
//   - If t_q >= sink_size:
//       * Global sinks: positions [0..sink_size-1]
//       * Local window: positions [max(sink_size, t_q - window_size + 1) .. t_q]

__global__ void streaming_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B_runtime, int T_runtime, int C_runtime,
    int window_size,
    int sink_size,
    float scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // each thread = one (b, t_query)
    int total_rows = B_runtime * T_runtime;
    if (row >= total_rows) return;

    int b   = row / T_runtime;
    int t_q = row % T_runtime;

    const float* q_vec = &Q[(b * T_runtime + t_q) * C_runtime];
    float*       out_vec = &O[(b * T_runtime + t_q) * C_runtime];

    // We will build a compact list of key indices this query can attend to.
    float scores_local[MAX_KEYS];
    int   key_index_local[MAX_KEYS];

    int len = 0; // number of active keys for this query

    // ----- 1) Sink tokens -----
    int sink_len = (sink_size < T_runtime ? sink_size : T_runtime); // clamp if T small
    if (sink_len > 0) {
        // For early tokens, don't go past t_q.
        int sink_attend_end = (t_q < (sink_len - 1) ? t_q : (sink_len - 1));
        for (int j = 0; j <= sink_attend_end; ++j) {
            key_index_local[len++] = j;
        }
    }

    // ----- 2) Local sliding window (for non-sink tokens) -----
    if (t_q >= sink_len) {
        int t_start = t_q - window_size + 1;
        if (t_start < sink_len) t_start = sink_len; // don't re-include sinks
        int t_end = t_q;

        for (int j = t_start; j <= t_end; ++j) {
            if (len < MAX_KEYS) {
                key_index_local[len++] = j;
            }
        }
    }

    // Just a safety guard (should not trigger for our chosen bounds)
    if (len > MAX_KEYS) len = MAX_KEYS;

    // ----- 3) Compute scores over the selected keys -----
    float max_score = -1e30f;
    for (int idx = 0; idx < len; ++idx) {
        int t_k = key_index_local[idx];
        const float* k_vec = &K[(b * T_runtime + t_k) * C_runtime];

        float dot = 0.0f;
        for (int c = 0; c < C_runtime; ++c) {
            dot += q_vec[c] * k_vec[c];
        }

        float s = dot * scale;
        scores_local[idx] = s;
        if (s > max_score) max_score = s;
    }

    // ----- 4) Softmax over selected keys -----
    float denom = 0.0f;
    for (int idx = 0; idx < len; ++idx) {
        float e = expf(scores_local[idx] - max_score);
        scores_local[idx] = e;
        denom += e;
    }
    float inv_denom = 1.0f / denom;

    // ----- 5) Weighted sum over V -----
    for (int c = 0; c < C_runtime; ++c) {
        float acc = 0.0f;
        for (int idx = 0; idx < len; ++idx) {
            int t_k = key_index_local[idx];
            float w = scores_local[idx] * inv_denom;
            const float* v_vec = &V[(b * T_runtime + t_k) * C_runtime];
            acc += w * v_vec[c];
        }
        out_vec[c] = acc;
    }
}

// ---------------------- Host utility: build mask ---------------------- //
// Build a T x T attention mask (row-major) for "streaming" pattern
// with sinks + sliding window.
//
// For each query i:
//   - if i < sink_size : attends [0..i] (normal causal prefix)
//   - else:
//        * attends to all sink tokens [0..sink_size-1]
//        * plus local window [max(sink_size, i - window_size + 1) .. i]

void build_streaming_mask(std::vector<unsigned char>& mask,
                          int T_runtime,
                          int window_size,
                          int sink_size)
{
    mask.assign(T_runtime * T_runtime, 0);

    int sink_len = min_val(sink_size, T_runtime);

    for (int i = 0; i < T_runtime; ++i) {
        if (i < sink_len) {
            // Early tokens (including sinks themselves): simple causal mask
            for (int j = 0; j <= i; ++j) {
                mask[i * T_runtime + j] = 1;
            }
        } else {
            // 1) Sinks: always visible
            for (int j = 0; j < sink_len; ++j) {
                mask[i * T_runtime + j] = 1;
            }
            // 2) Local window over non-sink positions
            int j_start = i - window_size + 1;
            if (j_start < sink_len) j_start = sink_len;
            int j_end = i;
            for (int j = j_start; j <= j_end; ++j) {
                if (j >= 0 && j < T_runtime) {
                    mask[i * T_runtime + j] = 1;
                }
            }
        }
    }
}

// ------------------------------ Main ------------------------------ //

int main(int argc, char** argv) {
    // Runtime parameters with defaults
    int T_runtime       = 1024;
    int window_size     = 64;
    int sink_size       = 16;

    if (argc >= 2) T_runtime   = std::atoi(argv[1]);   // ./attention_compare T
    if (argc >= 3) window_size = std::atoi(argv[2]);   // ./attention_compare T window
    if (argc >= 4) sink_size   = std::atoi(argv[3]);   // ./attention_compare T window sink

    if (T_runtime > MAX_T) {
        std::cerr << "Error: TRuntime=" << T_runtime
                  << " > MAX_T=" << MAX_T << "\n";
        return 1;
    }

    if (window_size + sink_size > MAX_KEYS) {
        std::cerr << "Error: window_size + sink_size = "
                  << (window_size + sink_size)
                  << " > MAX_KEYS=" << MAX_KEYS << "\n";
        return 1;
    }

    std::cout << "Comparing dense vs streaming sparse attention\n";
    std::cout << "B=" << B << ", T=" << T_runtime << ", C=" << C
              << ", WINDOW_SIZE=" << window_size
              << ", SINK_SIZE=" << sink_size << "\n";

    const int num_elements = B * T_runtime * C;
    const size_t bytes = static_cast<size_t>(num_elements) * sizeof(float);

    // Host buffers
    std::vector<float> h_Q(num_elements);
    std::vector<float> h_K(num_elements);
    std::vector<float> h_V(num_elements);
    std::vector<float> h_O_dense(num_elements);
    std::vector<float> h_O_stream(num_elements);

    // Random initialize Q, K, V
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    for (int i = 0; i < num_elements; ++i) {
        h_Q[i] = dist(rng);
        h_K[i] = dist(rng);
        h_V[i] = dist(rng);
    }

    // Build the T x T attention mask on host (mainly for debugging / visualization)
    std::vector<unsigned char> h_mask;
    build_streaming_mask(h_mask, T_runtime, window_size, sink_size);

    // Device buffers
    float *d_Q, *d_K, *d_V, *d_O_dense, *d_O_stream;
    unsigned char* d_mask; // not used directly in this simple kernel, but allocated for completeness

    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_dense, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_stream, bytes));
    CHECK_CUDA(cudaMalloc(&d_mask, static_cast<size_t>(T_runtime) * T_runtime * sizeof(unsigned char)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask.data(),
                          static_cast<size_t>(T_runtime) * T_runtime * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Launch configuration: one thread per (b, t)
    int total_rows = B * T_runtime;
    int threads = 128;
    int blocks  = (total_rows + threads - 1) / threads;

    float scale = 1.0f / std::sqrt(static_cast<float>(C));

    // CUDA events for timing
    cudaEvent_t start_dense, stop_dense;
    cudaEvent_t start_stream, stop_stream;
    CHECK_CUDA(cudaEventCreate(&start_dense));
    CHECK_CUDA(cudaEventCreate(&stop_dense));
    CHECK_CUDA(cudaEventCreate(&start_stream));
    CHECK_CUDA(cudaEventCreate(&stop_stream));

    // Warm-up runs (to avoid cold-start bias)
    dense_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V,
                                                d_O_dense,
                                                B, T_runtime, C, scale);
    streaming_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V,
                                                    d_O_stream,
                                                    B, T_runtime, C,
                                                    window_size,
                                                    sink_size,
                                                    scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int iters = 10;

    // -------- Dense timing --------
    CHECK_CUDA(cudaEventRecord(start_dense));
    for (int it = 0; it < iters; ++it) {
        dense_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V,
                                                    d_O_dense,
                                                    B, T_runtime, C, scale);
    }
    CHECK_CUDA(cudaEventRecord(stop_dense));
    CHECK_CUDA(cudaEventSynchronize(stop_dense));
    float ms_dense = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_dense, start_dense, stop_dense));
    ms_dense /= iters; // average per run

    // -------- Streaming timing --------
    CHECK_CUDA(cudaEventRecord(start_stream));
    for (int it = 0; it < iters; ++it) {
        streaming_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V,
                                                        d_O_stream,
                                                        B, T_runtime, C,
                                                        window_size,
                                                        sink_size,
                                                        scale);
    }
    CHECK_CUDA(cudaEventRecord(stop_stream));
    CHECK_CUDA(cudaEventSynchronize(stop_stream));
    float ms_stream = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms_stream, start_stream, stop_stream));
    ms_stream /= iters; // average per run

    std::cout << "Dense attention average time   : " << ms_dense  << " ms\n";
    std::cout << "Streaming attention avg time   : " << ms_stream << " ms\n";
    std::cout << "Speedup (dense / streaming)    : " << (ms_dense / ms_stream) << "x\n";

    // Optional: copy back one element to show it's not NaN
    CHECK_CUDA(cudaMemcpy(h_O_dense.data(), d_O_dense, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_O_stream.data(), d_O_stream, bytes, cudaMemcpyDeviceToHost));

    std::cout << "Example outputs (b=0, t=0, c=0):\n";
    std::cout << "  dense    : " << h_O_dense[0]   << "\n";
    std::cout << "  streaming: " << h_O_stream[0]  << "\n";

    // Cleanup
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O_dense));
    CHECK_CUDA(cudaFree(d_O_stream));
    CHECK_CUDA(cudaFree(d_mask));
    CHECK_CUDA(cudaEventDestroy(start_dense));
    CHECK_CUDA(cudaEventDestroy(stop_dense));
    CHECK_CUDA(cudaEventDestroy(start_stream));
    CHECK_CUDA(cudaEventDestroy(stop_stream));

    return 0;
}