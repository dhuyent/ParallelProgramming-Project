#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cstddef>

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

struct AEParams
{
    int batch = 64;
    float lr = 1e-3f;
};

struct AEWeightsDev
{
    float *w1 = nullptr, *b1 = nullptr;
    float *w2 = nullptr, *b2 = nullptr;
    float *w3 = nullptr, *b3 = nullptr;
    float *w4 = nullptr, *b4 = nullptr;
    float *w5 = nullptr, *b5 = nullptr;

    float *gw1 = nullptr, *gb1 = nullptr;
    float *gw2 = nullptr, *gb2 = nullptr;
    float *gw3 = nullptr, *gb3 = nullptr;
    float *gw4 = nullptr, *gb4 = nullptr;
    float *gw5 = nullptr, *gb5 = nullptr;

    void alloc(cudaStream_t s);
    void free(cudaStream_t s);
};

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

struct Phase3Engine
{
    AEParams p;
    size_t batchBytes = 0;

    AEWeightsDev w;
    AEBuffersDev b;

    cudaStream_t sH2D[2]{};
    cudaStream_t sCompute[2]{};
    cudaEvent_t eH2D[2]{};

    float *h_batchPinned[2]{nullptr, nullptr};
    void init(const AEParams &params);
    void shutdown();
    void sync_all();

    void train_step_async(const float *x_batch_host, int bufIdx);
    void reset_epoch_loss();
    float get_epoch_loss_avg_sync(int steps_per_epoch);
    void extract_features(const float *x_host_contig,
                          float *feat_host_contig,
                          int Ntotal);

    void save_to_file(const std::string &filename, cudaStream_t s);

    void load_from_file(const std::string &filename, cudaStream_t s);

    void forward_only(int N, cudaStream_t s);
};
