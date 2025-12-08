#pragma once
// gpu_autoencoder.cuh
// Header (CUDA-style) for GPUAutoencoder memory management
//
// Provide a class that holds host weight vectors and device pointers,
// and methods to allocate/copy/free them.
//
// - This header contains only declarations. Definitions in gpu_autoencoder.cu
// - Include this file from your .cu/.cpp code before using GPUAutoencoder.

#include <cuda_runtime.h>
#include <vector>
#include <string>

#ifndef GPU_AUTOENCODER_CUH
#define GPU_AUTOENCODER_CUH

// simple CUDA check macro
#define CHECK(call)                                                            \
    {                                                                          \
        const cudaError_t error = call;                                        \
        if (error != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA Error at %s:%d: code %d, reason: %s\n",      \
                    __FILE__, __LINE__, error, cudaGetErrorString(error));     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

class GPUAutoencoder {
public:
    GPUAutoencoder();
    ~GPUAutoencoder();

    // Initialize host weights randomly and allocate device buffers + copy weights
    // stddev: gaussian stddev for weight initialization
    void init_random_weights(float stddev = 0.05f);

    // Copy host weights -> device (upload)
    void copy_weights_to_device();

    // Copy device weights -> host (download)
    void copy_weights_to_host();

    // Save host weights to binary files with given prefix (prefix_w_conv1.bin etc.)
    void save_weights(const std::string& prefix);

    // Load host weights from binary files and upload to device.
    // returns true if successful (all files present), false otherwise
    bool load_weights(const std::string& prefix);

    // Allocate device memory for weights (expects host vectors sized)
    void alloc_weights_on_device();

    // Allocate activation buffers (per-sample)
    void alloc_activations_and_buffers();

    // Allocate gradient accumulator buffers
    void alloc_grads_on_device();

    // Free all device memory
    void free_all();

    // --- Host side weight containers (accessible if you want to read/modify) ---
    std::vector<float> h_w_conv1, h_b_conv1; // conv1: out=256, in=3, k=3
    std::vector<float> h_w_conv2, h_b_conv2; // conv2: out=128, in=256, k=3
    std::vector<float> h_w_dec1,  h_b_dec1;  // dec1: 128->128
    std::vector<float> h_w_dec2,  h_b_dec2;  // dec2: 256<-128
    std::vector<float> h_w_dec3,  h_b_dec3;  // dec3: 3<-256

    // --- Device pointers for weights (after alloc_weights_on_device / copy) ---
    float *d_w_conv1, *d_b_conv1;
    float *d_w_conv2, *d_b_conv2;
    float *d_w_dec1,  *d_b_dec1;
    float *d_w_dec2,  *d_b_dec2;
    float *d_w_dec3,  *d_b_dec3;

    // --- Activation buffers (per-sample) on device ---
    float *d_input;
    float *d_act1;
    float *d_pool1;
    float *d_act2;
    float *d_pool2;
    float *d_latent;
    float *d_dec_act1;
    float *d_up1;
    float *d_dec_act2;
    float *d_up2;
    float *d_out;

    // pool indices for unpooling
    int *d_pool1_idx;
    int *d_pool2_idx;

    // gradient accumulators for weights (device)
    float *d_g_w_conv1, *d_g_b_conv1;
    float *d_g_w_conv2, *d_g_b_conv2;
    float *d_g_w_dec1,  *d_g_b_dec1;
    float *d_g_w_dec2,  *d_g_b_dec2;
    float *d_g_w_dec3,  *d_g_b_dec3;

    // activation gradients (temporary buffers)
    float *d_grad_out;
    float *d_grad_up2;
    float *d_grad_dec_act2;
    float *d_grad_up1;
    float *d_grad_dec_act1;
    float *d_grad_latent;
    float *d_grad_pool2;
    float *d_grad_act2;
    float *d_grad_pool1;
    float *d_grad_act1;

    // constant: convolution kernel size
    static constexpr int KERNEL = 3;

private:
    // helper: allocate device memory and optionally zero it
    void malloc_device_ptr(float** dptr, size_t bytes, bool zero = true);
    void malloc_device_ptr_int(int** dptr, size_t bytes, bool zero = true);

    // indicator whether device buffers allocated
    bool device_allocated_;
};

#endif // GPU_AUTOENCODER_CUH
