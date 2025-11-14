// attention_compare.cu
// Compare dense vs "streaming" (sparse sliding-window) attention in CUDA
// Fixed sizes: B=64, T=256, C=384

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n",                    \
                         __FILE__, __LINE__, cudaGetErrorString(err));        \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Fixed problem sizes (you can change these if you want)
constexpr int B = 64;    // batch
constexpr int T = 256;   // sequence length
constexpr int C = 384;   // channel / head dim

// For streaming attention: each timestep attends to at most this many past tokens
constexpr int WINDOW_SIZE = 64;

// ---------------------- Dense attention kernel ---------------------- //
// Q, K, V, O are all [B, T, C] flattened as row-major: ((b*T + t)*C + c)

__global__ void dense_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int T, int C,
    float scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // each thread = one (b, t_query)
    int total_rows = B * T;
    if (row >= total_rows) return;

    int b = row / T;
    int t_q = row % T;

    const float* q_vec = &Q[(b * T + t_q) * C];
    float* out_vec     = &O[(b * T + t_q) * C];

    // FIX: T is compile-time constant from host, but kernel doesn't treat it as such.
    // So we use a hardcoded max size.
    float scores_local[256];   // MAX_T = 256

    // 1) Compute raw scores = q · k_j / sqrt(C)
    float max_score = -1e30f;
    for (int t_k = 0; t_k < T; ++t_k) {
        const float* k_vec = &K[(b * T + t_k) * C];
        float dot = 0.0f;
        for (int c = 0; c < C; ++c) {
            dot += q_vec[c] * k_vec[c];
        }
        float s = dot * scale;
        scores_local[t_k] = s;
        if (s > max_score) max_score = s;
    }

    // 2) Softmax denominator
    float denom = 0.0f;
    for (int t_k = 0; t_k < T; ++t_k) {
        float e = expf(scores_local[t_k] - max_score);
        scores_local[t_k] = e;  // store exp(score - max) for reuse
        denom += e;
    }
    float inv_denom = 1.0f / denom;

    // 3) Weighted sum over V
    for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        for (int t_k = 0; t_k < T; ++t_k) {
            float w = scores_local[t_k] * inv_denom; // softmax weight
            const float* v_vec = &V[(b * T + t_k) * C];
            acc += w * v_vec[c];
        }
        out_vec[c] = acc;
    }
}

// ------------------ Streaming (sparse) attention kernel ------------------ //
// "Streaming" here = sliding-window: each token attends only to last WINDOW_SIZE
// tokens (and itself). This is *much* cheaper than full T for large T.
//
// We still create a 256x256 attention mask on the host to match this pattern,
// but the kernel uses the window math directly (faster than reading the mask).

__global__ void streaming_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int T, int C,
    int window_size,
    float scale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x; // each thread = one (b, t_query)
    int total_rows = B * T;
    if (row >= total_rows) return;

    int b = row / T;
    int t_q = row % T;

    const float* q_vec = &Q[(b * T + t_q) * C];
    float* out_vec     = &O[(b * T + t_q) * C];

    // Streaming: attend only to [max(0, t_q - window_size + 1), t_q]
    int t_start = t_q - window_size + 1;
    if (t_start < 0) t_start = 0;
    int t_end = t_q; // inclusive

    int window_len = t_end - t_start + 1;

    // FIX: WINDOW_SIZE is compile-time constant from host, but kernel doesn't treat it as such.
    // So we use a hardcoded max size.
    float scores_local[64];    // WINDOW_SIZE=64 is constant

    // 1) Compute raw scores within the window
    float max_score = -1e30f;
    for (int idx = 0; idx < window_len; ++idx) {
        int t_k = t_start + idx;
        const float* k_vec = &K[(b * T + t_k) * C];
        float dot = 0.0f;
        for (int c = 0; c < C; ++c) {
            dot += q_vec[c] * k_vec[c];
        }
        float s = dot * scale;
        scores_local[idx] = s;
        if (s > max_score) max_score = s;
    }

    // 2) Softmax denominator within the window
    float denom = 0.0f;
    for (int idx = 0; idx < window_len; ++idx) {
        float e = expf(scores_local[idx] - max_score);
        scores_local[idx] = e;
        denom += e;
    }
    float inv_denom = 1.0f / denom;

    // 3) Weighted sum over V in the window
    for (int c = 0; c < C; ++c) {
        float acc = 0.0f;
        for (int idx = 0; idx < window_len; ++idx) {
            int t_k = t_start + idx;
            float w = scores_local[idx] * inv_denom;
            const float* v_vec = &V[(b * T + t_k) * C];
            acc += w * v_vec[c];
        }
        out_vec[c] = acc;
    }
}

// ---------------------- Host utility: build mask ---------------------- //
// Build a T x T attention mask (row-major) for the same sliding window pattern.
// mask[i*T + j] = 1 if token i can attend to token j, else 0.

void build_streaming_mask(std::vector<unsigned char>& mask, int T, int window_size) {
    mask.assign(T * T, 0);
    for (int i = 0; i < T; ++i) {
        int j_start = i - window_size + 1;
        if (j_start < 0) j_start = 0;
        int j_end = i; // causal
        for (int j = j_start; j <= j_end; ++j) {
            mask[i * T + j] = 1;
        }
    }
}

// ------------------------------ Main ------------------------------ //

int main() {
    std::cout << "Comparing dense vs streaming sparse attention\n";
    std::cout << "B=" << B << ", T=" << T << ", C=" << C
              << ", WINDOW_SIZE=" << WINDOW_SIZE << "\n";

    const int num_elements = B * T * C;
    const size_t bytes = num_elements * sizeof(float);

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

    // Build the 256x256 attention mask on host (mainly for debugging / visualization)
    std::vector<unsigned char> h_mask;
    build_streaming_mask(h_mask, T, WINDOW_SIZE);

    // Device buffers
    float *d_Q, *d_K, *d_V, *d_O_dense, *d_O_stream;
    unsigned char* d_mask; // not used directly in this simple kernel, but allocated for completeness

    CHECK_CUDA(cudaMalloc(&d_Q, bytes));
    CHECK_CUDA(cudaMalloc(&d_K, bytes));
    CHECK_CUDA(cudaMalloc(&d_V, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_dense, bytes));
    CHECK_CUDA(cudaMalloc(&d_O_stream, bytes));
    CHECK_CUDA(cudaMalloc(&d_mask, T * T * sizeof(unsigned char)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask.data(), T * T * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));

    // Launch configuration: one thread per (b, t)
    int total_rows = B * T;
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
    dense_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_O_dense, B, T, C, scale);
    streaming_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V, d_O_stream,
                                                    B, T, C, WINDOW_SIZE, scale);
    CHECK_CUDA(cudaDeviceSynchronize());

    const int iters = 10;

    // -------- Dense timing --------
    CHECK_CUDA(cudaEventRecord(start_dense));
    for (int it = 0; it < iters; ++it) {
        dense_attention_kernel<<<blocks, threads>>>(d_Q, d_K, d_V,
                                                    d_O_dense, B, T, C, scale);
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
                                                        B, T, C,
                                                        WINDOW_SIZE, scale);
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

    std::cout << "Example outputs (first element of batch 0, token 0, channel 0):\n";
    std::cout << "  dense   : " << h_O_dense[0] << "\n";
    std::cout << "  streaming: " << h_O_stream[0] << "\n";

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