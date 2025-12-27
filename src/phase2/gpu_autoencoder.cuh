#ifndef GPU_AUTOENCODER_CUH
#define GPU_AUTOENCODER_CUH

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h> 
#include "kernels.cuh"

class GPUAutoencoder {
public:
    static const int KERNEL = 3; 
    GPUAutoencoder();
    ~GPUAutoencoder();

    void init_random_weights(float stddev = 0.02f);
    
    void alloc_weights_on_device();
    void alloc_activations_and_buffers();
    void alloc_grads_on_device();
    void free_all();

    void copy_weights_to_device();
    void copy_weights_to_host();

    void save_weights(const std::string& filename);
    bool load_weights(const std::string& filename);

    float forward(float* d_input_sample, float* d_target_sample);
    void backward();
    void update_weights(int batch_size, float lr);

    void extract_features(float* d_in, float* h_out);


    // Host Weights (CPU)
    std::vector<float> h_w_conv1, h_b_conv1;
    std::vector<float> h_w_conv2, h_b_conv2;
    std::vector<float> h_w_dec1,  h_b_dec1;
    std::vector<float> h_w_dec2,  h_b_dec2;
    std::vector<float> h_w_dec3,  h_b_dec3;

    // Device Weights (GPU)
    float *d_w_conv1, *d_b_conv1;
    float *d_w_conv2, *d_b_conv2;
    float *d_w_dec1,  *d_b_dec1;
    float *d_w_dec2,  *d_b_dec2;
    float *d_w_dec3,  *d_b_dec3;

    // Device Activations & Buffers (GPU)
    float *d_input;
    float *d_act1, *d_pool1;
    float *d_act2, *d_pool2, *d_latent; 
    float *d_dec_act1, *d_up1;
    float *d_dec_act2, *d_up2;
    float *d_out;

    // Pooling Indices
    int *d_pool1_idx;
    int *d_pool2_idx;

    // Device Gradients - Weights (GPU)
    float *d_g_w_conv1, *d_g_b_conv1;
    float *d_g_w_conv2, *d_g_b_conv2;
    float *d_g_w_dec1,  *d_g_b_dec1;
    float *d_g_w_dec2,  *d_g_b_dec2;
    float *d_g_w_dec3,  *d_g_b_dec3;

    // Device Gradients - Flow (GPU)
    float *d_g_out; 
    float *d_grad_up2, *d_grad_dec_act2;
    float *d_grad_up1, *d_grad_dec_act1;
    float *d_grad_latent;
    float *d_grad_pool2, *d_grad_act2;
    float *d_grad_pool1, *d_grad_act1;

    // Optimization Buffers
    float *d_loss_accum; 

private:
    bool device_allocated_;

    void malloc_device_ptr(float** dptr, size_t bytes, bool zero = false);
    void malloc_device_ptr_int(int** dptr, size_t bytes, bool zero = false);
};

#endif // GPU_AUTOENCODER_CUH