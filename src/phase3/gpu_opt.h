#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstddef>

// =====================
// CUDA ERROR CHECK
// =====================
#define CUDA_CHECK(call)                                                \
    do                                                                  \
    {                                                                   \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess)                                         \
        {                                                               \
            std::cerr << "CUDA error " << __FILE__ << ":" << __LINE__   \
                      << " : " << cudaGetErrorString(err) << std::endl; \
            std::exit(1);                                               \
        }                                                               \
    } while (0)

// =====================
// Hyper-parameters
// =====================
struct AEParams
{
    int batch = 64;
    float lr = 1e-3f;
};

// =====================
// Weights on Device
// =====================
struct AEWeightsDev
{
    // weights
    float *w1 = nullptr, *b1 = nullptr;
    float *w2 = nullptr, *b2 = nullptr;
    float *w3 = nullptr, *b3 = nullptr;
    float *w4 = nullptr, *b4 = nullptr;
    float *w5 = nullptr, *b5 = nullptr;

    // gradients
    float *gw1 = nullptr, *gb1 = nullptr;
    float *gw2 = nullptr, *gb2 = nullptr;
    float *gw3 = nullptr, *gb3 = nullptr;
    float *gw4 = nullptr, *gb4 = nullptr;
    float *gw5 = nullptr, *gb5 = nullptr;

    void alloc(cudaStream_t s);
    void free(cudaStream_t s);
};

// =====================
// Buffers on Device
// =====================
struct AEBuffersDev
{
    // forward activations
    float *x = nullptr;   // [N,3,32,32]
    float *c1 = nullptr;  // [N,256,32,32]
    float *p1 = nullptr;  // [N,256,16,16]
    float *c2 = nullptr;  // [N,128,16,16]
    float *lat = nullptr; // [N,128,8,8]
    float *c3 = nullptr;  // [N,128,8,8]
    float *u1 = nullptr;  // [N,128,16,16]
    float *c4 = nullptr;  // [N,256,16,16]
    float *u2 = nullptr;  // [N,256,32,32]
    float *out = nullptr; // [N,3,32,32]

    // backward gradients
    float *g_out = nullptr;
    float *g_u2 = nullptr;
    float *g_c4 = nullptr;
    float *g_u1 = nullptr;
    float *g_c3 = nullptr;
    float *g_lat = nullptr;
    float *g_c2 = nullptr;
    float *g_p1 = nullptr;
    float *g_c1 = nullptr;
    float *g_x = nullptr;

    // loss buffers
    float *d_loss = nullptr;       // loss của 1 batch
    float *d_epoch_loss = nullptr; // accumulate loss cả epoch

    void alloc(int N, cudaStream_t s);
    void free(cudaStream_t s);
};

// =====================
// Phase 3 Engine
// =====================
struct Phase3Engine
{
    // params
    AEParams p;
    size_t batchBytes = 0;

    // weights & buffers
    AEWeightsDev w;
    AEBuffersDev b;

    // streams & events
    cudaStream_t sH2D[2]{};
    cudaStream_t sCompute[2]{};
    cudaEvent_t eH2D[2]{};

    // pinned host buffers (double buffer)
    float *h_batchPinned[2]{nullptr, nullptr};

    // lifecycle
    void init(const AEParams &params);
    void shutdown();
    void sync_all();

    // -------- training --------
    // NOTE: Phase 3 KHÔNG return loss mỗi batch
    void train_step_async(const float *x_batch_host, int bufIdx);

    // reset loss accumulator (call 1 lần/epoch)
    void reset_epoch_loss();

    // đọc avg loss của epoch (sync 1 lần)
    float get_epoch_loss_avg_sync(int steps_per_epoch);

    // -------- inference / feature extraction --------
    void extract_features(const float *x_host_contig,
                          float *feat_host_contig,
                          int Ntotal);
    
    void save_to_file(const std::string& filename, cudaStream_t s);

    void load_from_file(const std::string& filename, cudaStream_t s);

    void forward_only(int N, cudaStream_t s);
};
