// gpu_autoencoder.cu
// Compile with nvcc: nvcc -O2 -arch=sm_70 train.cu gpu_autoencoder.cu kernels.cu data_loader.cpp -o train_ae

#include "gpu_autoencoder.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm> // std::max

// --------------------------------------------------------------------------
// CONSTRUCTOR & DESTRUCTOR
// --------------------------------------------------------------------------

GPUAutoencoder::GPUAutoencoder()
: d_w_conv1(nullptr), d_b_conv1(nullptr),
  d_w_conv2(nullptr), d_b_conv2(nullptr),
  d_w_dec1(nullptr), d_b_dec1(nullptr),
  d_w_dec2(nullptr), d_b_dec2(nullptr),
  d_w_dec3(nullptr), d_b_dec3(nullptr),
  d_input(nullptr), d_act1(nullptr), d_pool1(nullptr), d_act2(nullptr),
  d_pool2(nullptr), d_latent(nullptr), d_dec_act1(nullptr), d_up1(nullptr),
  d_dec_act2(nullptr), d_up2(nullptr), d_out(nullptr),
  d_pool1_idx(nullptr), d_pool2_idx(nullptr),
  d_g_w_conv1(nullptr), d_g_b_conv1(nullptr),
  d_g_w_conv2(nullptr), d_g_b_conv2(nullptr),
  d_g_w_dec1(nullptr), d_g_b_dec1(nullptr),
  d_g_w_dec2(nullptr), d_g_b_dec2(nullptr),
  d_g_w_dec3(nullptr), d_g_b_dec3(nullptr),
  d_g_out(nullptr),
  d_grad_up2(nullptr), d_grad_dec_act2(nullptr),
  d_grad_up1(nullptr), d_grad_dec_act1(nullptr), d_grad_latent(nullptr),
  d_grad_pool2(nullptr), d_grad_act2(nullptr), d_grad_pool1(nullptr),
  d_grad_act1(nullptr),
  d_loss_accum(nullptr), // <--- UPDATE: Khởi tạo biến thành viên
  device_allocated_(false)
{
}

GPUAutoencoder::~GPUAutoencoder() {
    free_all();
}

// --------------------------------------------------------------------------
// MEMORY MANAGEMENT HELPERS
// --------------------------------------------------------------------------

void GPUAutoencoder::malloc_device_ptr(float** dptr, size_t bytes, bool zero) {
    CHECK(cudaMalloc((void**)dptr, bytes));
    if (zero) CHECK(cudaMemset(*dptr, 0, bytes));
}
void GPUAutoencoder::malloc_device_ptr_int(int** dptr, size_t bytes, bool zero) {
    CHECK(cudaMalloc((void**)dptr, bytes));
    if (zero) CHECK(cudaMemset(*dptr, 0, bytes));
}

// --------------------------------------------------------------------------
// INITIALIZATION (He Init + Allocation)
// --------------------------------------------------------------------------

void GPUAutoencoder::init_random_weights(float stddev) {
    // Kích thước Kernel
    const int k = KERNEL; 

    // Helper: He Initialization (Fan-in)
    auto init_vec = [&](std::vector<float>& v, int fan_in, int size) {
        v.resize(size);
        float limit = sqrt(2.0f / (float)fan_in); // He normal
        std::mt19937 gen(1234);
        std::normal_distribution<float> dist(0.0f, limit);
        for(int i=0; i<size; ++i) v[i] = dist(gen);
    };

    // Encoder
    init_vec(h_w_conv1, 3 * k * k, 256 * 3 * k * k);
    h_b_conv1.assign(256, 0.0f);

    init_vec(h_w_conv2, 256 * k * k, 128 * 256 * k * k);
    h_b_conv2.assign(128, 0.0f);

    // Decoder
    init_vec(h_w_dec1, 128 * k * k, 128 * 128 * k * k);
    h_b_dec1.assign(128, 0.0f);

    init_vec(h_w_dec2, 128 * k * k, 256 * 128 * k * k);
    h_b_dec2.assign(256, 0.0f);

    init_vec(h_w_dec3, 256 * k * k, 3 * 256 * k * k);
    h_b_dec3.assign(3, 0.0f);

    // Cấp phát bộ nhớ GPU
    alloc_weights_on_device();
    alloc_activations_and_buffers();
    alloc_grads_on_device();
    
    device_allocated_ = true;
    printf("[GPUAutoencoder] Initialized weights and allocated device memory.\n");
}

void GPUAutoencoder::alloc_weights_on_device() {
    auto alloc_copy = [&](float** d_ptr, std::vector<float>& h_vec) {
        CHECK(cudaMalloc(d_ptr, h_vec.size() * sizeof(float)));
        CHECK(cudaMemcpy(*d_ptr, h_vec.data(), h_vec.size() * sizeof(float), cudaMemcpyHostToDevice));
    };

    alloc_copy(&d_w_conv1, h_w_conv1); alloc_copy(&d_b_conv1, h_b_conv1);
    alloc_copy(&d_w_conv2, h_w_conv2); alloc_copy(&d_b_conv2, h_b_conv2);
    alloc_copy(&d_w_dec1, h_w_dec1);   alloc_copy(&d_b_dec1, h_b_dec1);
    alloc_copy(&d_w_dec2, h_w_dec2);   alloc_copy(&d_b_dec2, h_b_dec2);
    alloc_copy(&d_w_dec3, h_w_dec3);   alloc_copy(&d_b_dec3, h_b_dec3);
}

void GPUAutoencoder::alloc_activations_and_buffers() {
    size_t s_input = (size_t)3*32*32 * sizeof(float);
    size_t s_act1  = (size_t)256*32*32 * sizeof(float);
    size_t s_pool1 = (size_t)256*16*16 * sizeof(float);
    size_t s_act2  = (size_t)128*16*16 * sizeof(float);
    size_t s_pool2 = (size_t)128*8*8   * sizeof(float);
    size_t s_latent= s_pool2;
    size_t s_dec1  = (size_t)128*8*8   * sizeof(float);
    size_t s_up1   = (size_t)128*16*16 * sizeof(float);
    size_t s_dec2  = (size_t)256*16*16 * sizeof(float);
    size_t s_up2   = (size_t)256*32*32 * sizeof(float);
    size_t s_out   = (size_t)3*32*32   * sizeof(float);

    malloc_device_ptr(&d_input, s_input);
    malloc_device_ptr(&d_act1, s_act1);
    malloc_device_ptr(&d_pool1, s_pool1);
    malloc_device_ptr(&d_act2, s_act2);
    malloc_device_ptr(&d_pool2, s_pool2);
    malloc_device_ptr(&d_latent, s_latent);
    malloc_device_ptr(&d_dec_act1, s_dec1);
    malloc_device_ptr(&d_up1, s_up1);
    malloc_device_ptr(&d_dec_act2, s_dec2);
    malloc_device_ptr(&d_up2, s_up2);
    malloc_device_ptr(&d_out, s_out);

    // <--- UPDATE: Cấp phát bộ nhớ cho d_loss_accum một lần duy nhất ở đây
    malloc_device_ptr(&d_loss_accum, sizeof(float), true);

    malloc_device_ptr_int(&d_pool1_idx, (256*16*16)*sizeof(int));
    malloc_device_ptr_int(&d_pool2_idx, (128*8*8)*sizeof(int));
}

void GPUAutoencoder::alloc_grads_on_device() {
    auto alloc_zero = [&](float** d_ptr, size_t size) {
        CHECK(cudaMalloc(d_ptr, size * sizeof(float)));
        CHECK(cudaMemset(*d_ptr, 0, size * sizeof(float)));
    };

    alloc_zero(&d_g_w_conv1, h_w_conv1.size()); alloc_zero(&d_g_b_conv1, h_b_conv1.size());
    alloc_zero(&d_g_w_conv2, h_w_conv2.size()); alloc_zero(&d_g_b_conv2, h_b_conv2.size());
    alloc_zero(&d_g_w_dec1, h_w_dec1.size());   alloc_zero(&d_g_b_dec1, h_b_dec1.size());
    alloc_zero(&d_g_w_dec2, h_w_dec2.size());   alloc_zero(&d_g_b_dec2, h_b_dec2.size());
    alloc_zero(&d_g_w_dec3, h_w_dec3.size());   alloc_zero(&d_g_b_dec3, h_b_dec3.size());

    // Alloc gradients buffers
    malloc_device_ptr(&d_g_out, 3*32*32*sizeof(float), true);
    
    malloc_device_ptr(&d_grad_up2, 256*32*32*sizeof(float));
    malloc_device_ptr(&d_grad_dec_act2, 256*16*16*sizeof(float));
    malloc_device_ptr(&d_grad_up1, 128*16*16*sizeof(float));
    malloc_device_ptr(&d_grad_dec_act1, 128*8*8*sizeof(float));
    malloc_device_ptr(&d_grad_latent, 128*8*8*sizeof(float));
    malloc_device_ptr(&d_grad_pool2, 128*8*8*sizeof(float));
    malloc_device_ptr(&d_grad_act2, 128*16*16*sizeof(float));
    malloc_device_ptr(&d_grad_pool1, 256*16*16*sizeof(float));
    malloc_device_ptr(&d_grad_act1, 256*32*32*sizeof(float));
}

// --------------------------------------------------------------------------
// DATA TRANSFER (Host <-> Device)
// --------------------------------------------------------------------------

void GPUAutoencoder::copy_weights_to_device() {
    auto copy = [&](float* d_ptr, std::vector<float>& h_vec) {
        CHECK(cudaMemcpy(d_ptr, h_vec.data(), h_vec.size()*sizeof(float), cudaMemcpyHostToDevice));
    };
    copy(d_w_conv1, h_w_conv1); copy(d_b_conv1, h_b_conv1);
    copy(d_w_conv2, h_w_conv2); copy(d_b_conv2, h_b_conv2);
    copy(d_w_dec1, h_w_dec1);   copy(d_b_dec1, h_b_dec1);
    copy(d_w_dec2, h_w_dec2);   copy(d_b_dec2, h_b_dec2);
    copy(d_w_dec3, h_w_dec3);   copy(d_b_dec3, h_b_dec3);
}

void GPUAutoencoder::copy_weights_to_host() {
    auto copy = [&](std::vector<float>& h_vec, float* d_ptr) {
        CHECK(cudaMemcpy(h_vec.data(), d_ptr, h_vec.size()*sizeof(float), cudaMemcpyDeviceToHost));
    };
    copy(h_w_conv1, d_w_conv1); copy(h_b_conv1, d_b_conv1);
    copy(h_w_conv2, d_w_conv2); copy(h_b_conv2, d_b_conv2);
    copy(h_w_dec1, d_w_dec1);   copy(h_b_dec1, d_b_dec1);
    copy(h_w_dec2, d_w_dec2);   copy(h_b_dec2, d_b_dec2);
    copy(h_w_dec3, d_w_dec3);   copy(h_b_dec3, d_b_dec3);
}

// --------------------------------------------------------------------------
// SAVE & LOAD (Single Binary File)
// --------------------------------------------------------------------------

void GPUAutoencoder::save_weights(const std::string& filename) {
    copy_weights_to_host();
    std::ofstream f(filename, std::ios::binary);
    if (!f) { std::cerr << "[Error] Cannot open " << filename << "\n"; return; }

    auto write_vec = [&](const std::vector<float>& v) {
        f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    };

    write_vec(h_w_conv1); write_vec(h_b_conv1);
    write_vec(h_w_conv2); write_vec(h_b_conv2);
    write_vec(h_w_dec1);  write_vec(h_b_dec1);
    write_vec(h_w_dec2);  write_vec(h_b_dec2);
    write_vec(h_w_dec3);  write_vec(h_b_dec3);

    f.close();
    std::cout << "[GPUAutoencoder] Saved weights to '" << filename << "'\n";
}

bool GPUAutoencoder::load_weights(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) { std::cerr << "[Error] Cannot open " << filename << "\n"; return false; }

    // Resize host vectors based on architecture constants (HARDCODED ARCHITECTURE)
    const int k = KERNEL; 
    h_w_conv1.resize(256*3*k*k);   h_b_conv1.resize(256);
    h_w_conv2.resize(128*256*k*k); h_b_conv2.resize(128);
    h_w_dec1.resize(128*128*k*k);  h_b_dec1.resize(128);
    h_w_dec2.resize(256*128*k*k);  h_b_dec2.resize(256);
    h_w_dec3.resize(3*256*k*k);    h_b_dec3.resize(3);

    auto read_vec = [&](std::vector<float>& v) {
        f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
    };

    read_vec(h_w_conv1); read_vec(h_b_conv1);
    read_vec(h_w_conv2); read_vec(h_b_conv2);
    read_vec(h_w_dec1);  read_vec(h_b_dec1);
    read_vec(h_w_dec2);  read_vec(h_b_dec2);
    read_vec(h_w_dec3);  read_vec(h_b_dec3);

    if (!f) return false;
    f.close();
    
    // Nếu chưa cấp phát GPU thì cấp phát luôn
    if (!device_allocated_) {
        alloc_weights_on_device();
        alloc_activations_and_buffers();
        alloc_grads_on_device();
        device_allocated_ = true;
    } else {
        copy_weights_to_device();
    }
    
    std::cout << "[GPUAutoencoder] Loaded weights from '" << filename << "'\n";
    return true;
}

// --------------------------------------------------------------------------
// CORE OPERATIONS
// --------------------------------------------------------------------------

// 1. FORWARD PASS (Train)
float GPUAutoencoder::forward(float* d_input_sample, float* d_target_sample) {
    const int k = KERNEL;
    // Encoder
    launch_conv2d_forward_naive(d_input_sample, d_w_conv1, d_b_conv1, d_act1, 3, 32, 32, 256, k);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(d_act1, 256*32*32);
    CHECK(cudaDeviceSynchronize());
    launch_maxpool2x2_forward(d_act1, d_pool1, d_pool1_idx, 256, 32, 32);
    CHECK(cudaDeviceSynchronize());

    launch_conv2d_forward_naive(d_pool1, d_w_conv2, d_b_conv2, d_act2, 256, 16, 16, 128, k);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(d_act2, 128*16*16);
    CHECK(cudaDeviceSynchronize());
    launch_maxpool2x2_forward(d_act2, d_latent, d_pool2_idx, 128, 16, 16);
    CHECK(cudaDeviceSynchronize());

    // Decoder
    launch_conv2d_forward_naive(d_latent, d_w_dec1, d_b_dec1, d_dec_act1, 128, 8, 8, 128, k);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(d_dec_act1, 128*8*8);
    CHECK(cudaDeviceSynchronize());

    launch_upsample_nn_forward(d_dec_act1, d_up1, 128, 8, 8);
    CHECK(cudaDeviceSynchronize());

    launch_conv2d_forward_naive(d_up1, d_w_dec2, d_b_dec2, d_dec_act2, 128, 16, 16, 256, k);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(d_dec_act2, 256*16*16);
    CHECK(cudaDeviceSynchronize());

    launch_upsample_nn_forward(d_dec_act2, d_up2, 256, 16, 16);
    CHECK(cudaDeviceSynchronize());

    launch_conv2d_forward_naive(d_up2, d_w_dec3, d_b_dec3, d_out, 256, 32, 32, 3, k);
    CHECK(cudaDeviceSynchronize());

    // Compute Loss
    int total_out = 3 * 32 * 32;
    // Reset gradient đầu ra
    CHECK(cudaMemset(d_g_out, 0, total_out * sizeof(float))); 
    
    // <--- UPDATE: Bỏ cudaMalloc ở đây, sử dụng d_loss_accum đã cấp phát sẵn
    // Rất quan trọng: Phải reset giá trị cũ về 0
    CHECK(cudaMemset(d_loss_accum, 0, sizeof(float))); 
    
    launch_mse_loss_and_grad(d_out, d_target_sample, d_g_out, d_loss_accum, total_out);
    CHECK(cudaDeviceSynchronize());
    
    float h_loss = 0.0f;
    CHECK(cudaMemcpy(&h_loss, d_loss_accum, sizeof(float), cudaMemcpyDeviceToHost));
    // <--- UPDATE: Không gọi cudaFree ở đây nữa

    return h_loss;
}

// 2. BACKWARD PASS (Train)
void GPUAutoencoder::backward() {
    const int k = KERNEL;
    // Dec3
    launch_conv2d_weight_grad_naive(d_up2, d_g_out, d_g_w_dec3, d_g_b_dec3, 256, 32, 32, 3, k);
    launch_conv2d_input_grad_naive(d_g_out, d_w_dec3, d_grad_up2, 256, 32, 32, 3, k);

    // Dec2
    launch_upsample_backward(d_grad_up2, d_grad_dec_act2, 256, 16, 16);
    launch_relu_backward(d_grad_dec_act2, d_dec_act2, d_grad_dec_act2, 256*16*16);
    launch_conv2d_weight_grad_naive(d_up1, d_grad_dec_act2, d_g_w_dec2, d_g_b_dec2, 128, 16, 16, 256, k);
    launch_conv2d_input_grad_naive(d_grad_dec_act2, d_w_dec2, d_grad_up1, 128, 16, 16, 256, k);

    // Dec1
    launch_upsample_backward(d_grad_up1, d_grad_dec_act1, 128, 8, 8);
    launch_relu_backward(d_grad_dec_act1, d_dec_act1, d_grad_dec_act1, 128*8*8);
    launch_conv2d_weight_grad_naive(d_latent, d_grad_dec_act1, d_g_w_dec1, d_g_b_dec1, 128, 8, 8, 128, k);
    launch_conv2d_input_grad_naive(d_grad_dec_act1, d_w_dec1, d_grad_latent, 128, 8, 8, 128, k);

    // Conv2
    CHECK(cudaMemset(d_grad_act2, 0, (size_t)128*16*16*sizeof(float)));
    launch_maxpool2x2_backward(d_grad_latent, d_grad_act2, d_pool2_idx, 128, 16, 16);
    launch_relu_backward(d_grad_act2, d_act2, d_grad_act2, 128*16*16);
    launch_conv2d_weight_grad_naive(d_pool1, d_grad_act2, d_g_w_conv2, d_g_b_conv2, 256, 16, 16, 128, k);
    launch_conv2d_input_grad_naive(d_grad_act2, d_w_conv2, d_grad_pool1, 256, 16, 16, 128, k);

    // Conv1
    CHECK(cudaMemset(d_grad_act1, 0, (size_t)256*32*32*sizeof(float)));
    launch_maxpool2x2_backward(d_grad_pool1, d_grad_act1, d_pool1_idx, 256, 32, 32);
    launch_relu_backward(d_grad_act1, d_act1, d_grad_act1, 256*32*32);
    launch_conv2d_weight_grad_naive(d_input, d_grad_act1, d_g_w_conv1, d_g_b_conv1, 3, 32, 32, 256, k);
}

// 3. UPDATE WEIGHTS (Optimizer)
void GPUAutoencoder::update_weights(int batch_size, float lr) {
    auto update = [&](float* w, float* g, size_t size) {
        launch_update_weights_on_device(w, g, size, lr, batch_size);
    };
    auto update_b = [&](float* b, float* g, size_t size) {
        launch_update_bias_on_device(b, g, size, lr, batch_size);
    };

    update(d_w_conv1, d_g_w_conv1, h_w_conv1.size()); update_b(d_b_conv1, d_g_b_conv1, h_b_conv1.size());
    update(d_w_conv2, d_g_w_conv2, h_w_conv2.size()); update_b(d_b_conv2, d_g_b_conv2, h_b_conv2.size());
    update(d_w_dec1, d_g_w_dec1, h_w_dec1.size());    update_b(d_b_dec1, d_g_b_dec1, h_b_dec1.size());
    update(d_w_dec2, d_g_w_dec2, h_w_dec2.size());    update_b(d_b_dec2, d_g_b_dec2, h_b_dec2.size());
    update(d_w_dec3, d_g_w_dec3, h_w_dec3.size());    update_b(d_b_dec3, d_g_b_dec3, h_b_dec3.size());
    
    CHECK(cudaDeviceSynchronize());
}

// 4. EXTRACT FEATURES (Inference for SVM)
void GPUAutoencoder::extract_features(float* d_in, float* h_out) {
    const int k = KERNEL;
    // Conv1
    launch_conv2d_forward_naive(d_in, d_w_conv1, d_b_conv1, d_act1, 3, 32, 32, 256, k);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(d_act1, 256*32*32);
    CHECK(cudaDeviceSynchronize());
    // Pool1
    launch_maxpool2x2_forward(d_act1, d_pool1, d_pool1_idx, 256, 32, 32);
    CHECK(cudaDeviceSynchronize());

    // Conv2
    launch_conv2d_forward_naive(d_pool1, d_w_conv2, d_b_conv2, d_act2, 256, 16, 16, 128, k);
    CHECK(cudaDeviceSynchronize());
    launch_relu_inplace(d_act2, 128*16*16);
    CHECK(cudaDeviceSynchronize());
    // Pool2 -> Latent
    launch_maxpool2x2_forward(d_act2, d_latent, d_pool2_idx, 128, 16, 16);
    CHECK(cudaDeviceSynchronize());

    // Copy to CPU
    size_t latent_size = 128 * 8 * 8 * sizeof(float);
    CHECK(cudaMemcpy(h_out, d_latent, latent_size, cudaMemcpyDeviceToHost));
}

// --------------------------------------------------------------------------
// CLEANUP
// --------------------------------------------------------------------------

void GPUAutoencoder::free_all() {
    auto free_ptr = [](float*& ptr) { if(ptr) { cudaFree(ptr); ptr = nullptr; } };
    auto free_int = [](int*& ptr)   { if(ptr) { cudaFree(ptr); ptr = nullptr; } };

    free_ptr(d_w_conv1); free_ptr(d_b_conv1);
    free_ptr(d_w_conv2); free_ptr(d_b_conv2);
    free_ptr(d_w_dec1);  free_ptr(d_b_dec1);
    free_ptr(d_w_dec2);  free_ptr(d_b_dec2);
    free_ptr(d_w_dec3);  free_ptr(d_b_dec3);

    free_ptr(d_input); free_ptr(d_act1); free_ptr(d_pool1);
    free_ptr(d_act2);  free_ptr(d_pool2); free_ptr(d_latent);
    free_ptr(d_dec_act1); free_ptr(d_up1); free_ptr(d_dec_act2);
    free_ptr(d_up2);   free_ptr(d_out);

    // <--- UPDATE: Giải phóng d_loss_accum
    free_ptr(d_loss_accum);

    free_int(d_pool1_idx); free_int(d_pool2_idx);

    free_ptr(d_g_w_conv1); free_ptr(d_g_b_conv1);
    free_ptr(d_g_w_conv2); free_ptr(d_g_b_conv2);
    free_ptr(d_g_w_dec1);  free_ptr(d_g_b_dec1);
    free_ptr(d_g_w_dec2);  free_ptr(d_g_b_dec2);
    free_ptr(d_g_w_dec3);  free_ptr(d_g_b_dec3);

    free_ptr(d_g_out); 
    free_ptr(d_grad_up2); free_ptr(d_grad_dec_act2);
    free_ptr(d_grad_up1); free_ptr(d_grad_dec_act1); free_ptr(d_grad_latent);
    free_ptr(d_grad_pool2); free_ptr(d_grad_act2); free_ptr(d_grad_pool1);
    free_ptr(d_grad_act1);

    device_allocated_ = false;
}