// gpu_autoencoder.cuh
#ifndef GPU_AUTOENCODER_CUH
#define GPU_AUTOENCODER_CUH

#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h> // Bắt buộc để dùng cudaError_t trong macro

// --- MACRO CHECK ERROR ---
// Định nghĩa tại đây để tất cả các file include header này đều dùng được
#ifndef CHECK
#define CHECK(call) \
    { \
        const cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    }
#endif

class GPUAutoencoder {
public:
    static const int KERNEL = 3; // Kernel size 3x3 cố định

    // --- Constructor & Destructor ---
    GPUAutoencoder();
    ~GPUAutoencoder();

    // --- Initialization ---
    // Khởi tạo trọng số ngẫu nhiên (He Initialization) và cấp phát bộ nhớ GPU
    void init_random_weights(float stddev = 0.02f);
    
    // --- Memory Management ---
    void alloc_weights_on_device();
    void alloc_activations_and_buffers();
    void alloc_grads_on_device();
    void free_all();

    // --- Data Transfer ---
    void copy_weights_to_device();
    void copy_weights_to_host();

    // --- File I/O ---
    void save_weights(const std::string& filename);
    bool load_weights(const std::string& filename);

    // --- CORE OPERATIONS (Encapsulated Logic) ---
    
    // 1. Forward Pass (Training): Input -> Output, trả về Loss
    float forward(float* d_input_sample, float* d_target_sample);

    // 2. Backward Pass (Training): Tính Gradient ngược từ Output -> Input
    void backward();

    // 3. Update Weights (Optimizer): Cập nhật trọng số theo SGD
    void update_weights(int batch_size, float lr);

    // 4. Feature Extraction (SVM Inference): Chỉ chạy Encoder: Input -> Latent
    void extract_features(float* d_in, float* h_out);

    // --- MEMBER VARIABLES (Public để Kernel truy cập dễ dàng) ---

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
    float *d_act2, *d_pool2, *d_latent; // d_latent dùng chung vùng nhớ d_pool2
    float *d_dec_act1, *d_up1;
    float *d_dec_act2, *d_up2;
    float *d_out;

    // Pooling Indices (Cho MaxUnpooling)
    int *d_pool1_idx;
    int *d_pool2_idx;

    // Device Gradients - Weights (GPU)
    float *d_g_w_conv1, *d_g_b_conv1;
    float *d_g_w_conv2, *d_g_b_conv2;
    float *d_g_w_dec1,  *d_g_b_dec1;
    float *d_g_w_dec2,  *d_g_b_dec2;
    float *d_g_w_dec3,  *d_g_b_dec3;

    // Device Gradients - Flow (GPU)
    float *d_grad_out;
    float *d_grad_up2, *d_grad_dec_act2;
    float *d_grad_up1, *d_grad_dec_act1;
    float *d_grad_latent;
    float *d_grad_pool2, *d_grad_act2;
    float *d_grad_pool1, *d_grad_act1;

private:
    bool device_allocated_;

    // Helpers
    void malloc_device_ptr(float** dptr, size_t bytes, bool zero = false);
    void malloc_device_ptr_int(int** dptr, size_t bytes, bool zero = false);
};

#endif // GPU_AUTOENCODER_CUH