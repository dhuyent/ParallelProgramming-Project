// gpu_autoencoder.cu
// Implementation for GPUAutoencoder declared in gpu_autoencoder.cuh
// Compile with nvcc (part of your project): nvcc -O2 -arch=sm_70 ... gpu_autoencoder.cu ...

#include "gpu_autoencoder.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <fstream>
#include <iostream>
#include <cstring>

// ---------------- GPUAutoencoder implementation ----------------

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
  d_grad_out(nullptr), d_grad_up2(nullptr), d_grad_dec_act2(nullptr),
  d_grad_up1(nullptr), d_grad_dec_act1(nullptr), d_grad_latent(nullptr),
  d_grad_pool2(nullptr), d_grad_act2(nullptr), d_grad_pool1(nullptr),
  d_grad_act1(nullptr),
  device_allocated_(false)
{
    // empty
}

GPUAutoencoder::~GPUAutoencoder() {
    free_all();
}

void GPUAutoencoder::malloc_device_ptr(float** dptr, size_t bytes, bool zero) {
    CHECK(cudaMalloc((void**)dptr, bytes));
    if (zero) CHECK(cudaMemset(*dptr, 0, bytes));
}
void GPUAutoencoder::malloc_device_ptr_int(int** dptr, size_t bytes, bool zero) {
    CHECK(cudaMalloc((void**)dptr, bytes));
    if (zero) CHECK(cudaMemset(*dptr, 0, bytes));
}

void GPUAutoencoder::alloc_weights_on_device() {
    // requires host vectors sized correctly
    if (h_w_conv1.empty() || h_w_conv2.empty() || h_w_dec1.empty() || h_w_dec2.empty() || h_w_dec3.empty()) {
        std::cerr << "[alloc_weights_on_device] host weight vectors empty -> call init_random_weights() or set them on host.\n";
        // still attempt to allocate if sizes not set would be unsafe; better return
    }
    CHECK(cudaMalloc(&d_w_conv1, h_w_conv1.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_w_conv1, h_w_conv1.data(), h_w_conv1.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_b_conv1, h_b_conv1.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_b_conv1, h_b_conv1.data(), h_b_conv1.size()*sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_w_conv2, h_w_conv2.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_w_conv2, h_w_conv2.data(), h_w_conv2.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_b_conv2, h_b_conv2.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_b_conv2, h_b_conv2.data(), h_b_conv2.size()*sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_w_dec1, h_w_dec1.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_w_dec1, h_w_dec1.data(), h_w_dec1.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_b_dec1, h_b_dec1.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_b_dec1, h_b_dec1.data(), h_b_dec1.size()*sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_w_dec2, h_w_dec2.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_w_dec2, h_w_dec2.data(), h_w_dec2.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_b_dec2, h_b_dec2.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_b_dec2, h_b_dec2.data(), h_b_dec2.size()*sizeof(float), cudaMemcpyHostToDevice));

    CHECK(cudaMalloc(&d_w_dec3, h_w_dec3.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_w_dec3, h_w_dec3.data(), h_w_dec3.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMalloc(&d_b_dec3, h_b_dec3.size()*sizeof(float)));
    CHECK(cudaMemcpy(d_b_dec3, h_b_dec3.data(), h_b_dec3.size()*sizeof(float), cudaMemcpyHostToDevice));
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

    malloc_device_ptr_int(&d_pool1_idx, (256*16*16)*sizeof(int));
    malloc_device_ptr_int(&d_pool2_idx, (128*8*8)*sizeof(int));
}

void GPUAutoencoder::alloc_grads_on_device() {
    // allocate grads for weights and zero them
    CHECK(cudaMalloc(&d_g_w_conv1, h_w_conv1.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_b_conv1, h_b_conv1.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_w_conv2, h_w_conv2.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_b_conv2, h_b_conv2.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_w_dec1, h_w_dec1.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_b_dec1, h_b_dec1.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_w_dec2, h_w_dec2.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_b_dec2, h_b_dec2.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_w_dec3, h_w_dec3.size()*sizeof(float)));
    CHECK(cudaMalloc(&d_g_b_dec3, h_b_dec3.size()*sizeof(float)));

    CHECK(cudaMemset(d_g_w_conv1, 0, h_w_conv1.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_b_conv1, 0, h_b_conv1.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_w_conv2, 0, h_w_conv2.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_b_conv2, 0, h_b_conv2.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_w_dec1, 0, h_w_dec1.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_b_dec1, 0, h_b_dec1.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_w_dec2, 0, h_w_dec2.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_b_dec2, 0, h_b_dec2.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_w_dec3, 0, h_w_dec3.size()*sizeof(float)));
    CHECK(cudaMemset(d_g_b_dec3, 0, h_b_dec3.size()*sizeof(float)));

    // temp grads
    size_t s_out   = (size_t)3*32*32   * sizeof(float);
    size_t s_up2   = (size_t)256*32*32 * sizeof(float);
    size_t s_dec2  = (size_t)256*16*16 * sizeof(float);
    size_t s_up1   = (size_t)128*16*16 * sizeof(float);
    size_t s_dec1  = (size_t)128*8*8   * sizeof(float);
    size_t s_latent= (size_t)128*8*8   * sizeof(float);
    size_t s_pool2 = (size_t)128*8*8   * sizeof(float);
    size_t s_act2  = (size_t)128*16*16 * sizeof(float);
    size_t s_pool1 = (size_t)256*16*16 * sizeof(float);
    size_t s_act1  = (size_t)256*32*32 * sizeof(float);

    malloc_device_ptr(&d_grad_out, s_out);
    malloc_device_ptr(&d_grad_up2, s_up2);
    malloc_device_ptr(&d_grad_dec_act2, s_dec2);
    malloc_device_ptr(&d_grad_up1, s_up1);
    malloc_device_ptr(&d_grad_dec_act1, s_dec1);
    malloc_device_ptr(&d_grad_latent, s_latent);
    malloc_device_ptr(&d_grad_pool2, s_pool2);
    malloc_device_ptr(&d_grad_act2, s_act2);
    malloc_device_ptr(&d_grad_pool1, s_pool1);
    malloc_device_ptr(&d_grad_act1, s_act1);
}

void GPUAutoencoder::copy_weights_to_device() {
    // allocate device weights if not yet allocated
    if (!d_w_conv1) {
        alloc_weights_on_device();
        return;
    }
    CHECK(cudaMemcpy(d_w_conv1, h_w_conv1.data(), h_w_conv1.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_conv1, h_b_conv1.data(), h_b_conv1.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w_conv2, h_w_conv2.data(), h_w_conv2.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_conv2, h_b_conv2.data(), h_b_conv2.size()*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w_dec1,  h_w_dec1.data(),  h_w_dec1.size()*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_dec1,  h_b_dec1.data(),  h_b_dec1.size()*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w_dec2,  h_w_dec2.data(),  h_w_dec2.size()*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_dec2,  h_b_dec2.data(),  h_b_dec2.size()*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w_dec3,  h_w_dec3.data(),  h_w_dec3.size()*sizeof(float),  cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b_dec3,  h_b_dec3.data(),  h_b_dec3.size()*sizeof(float),  cudaMemcpyHostToDevice));
}

void GPUAutoencoder::copy_weights_to_host() {
    // ensure host vectors sized
    const int k = KERNEL;
    if (h_w_conv1.empty()) h_w_conv1.resize(256*3*k*k);
    if (h_b_conv1.empty()) h_b_conv1.resize(256);
    if (h_w_conv2.empty()) h_w_conv2.resize(128*256*k*k);
    if (h_b_conv2.empty()) h_b_conv2.resize(128);
    if (h_w_dec1.empty())  h_w_dec1.resize(128*128*k*k);
    if (h_b_dec1.empty())  h_b_dec1.resize(128);
    if (h_w_dec2.empty())  h_w_dec2.resize(256*128*k*k);
    if (h_b_dec2.empty())  h_b_dec2.resize(256);
    if (h_w_dec3.empty())  h_w_dec3.resize(3*256*k*k);
    if (h_b_dec3.empty())  h_b_dec3.resize(3);

    CHECK(cudaMemcpy(h_w_conv1.data(), d_w_conv1, h_w_conv1.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b_conv1.data(), d_b_conv1, h_b_conv1.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_w_conv2.data(), d_w_conv2, h_w_conv2.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b_conv2.data(), d_b_conv2, h_b_conv2.size()*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_w_dec1.data(),  d_w_dec1,  h_w_dec1.size()*sizeof(float),  cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b_dec1.data(),  d_b_dec1,  h_b_dec1.size()*sizeof(float),  cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_w_dec2.data(),  d_w_dec2,  h_w_dec2.size()*sizeof(float),  cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b_dec2.data(),  d_b_dec2,  h_b_dec2.size()*sizeof(float),  cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_w_dec3.data(),  d_w_dec3,  h_w_dec3.size()*sizeof(float),  cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_b_dec3.data(),  d_b_dec3,  h_b_dec3.size()*sizeof(float),  cudaMemcpyDeviceToHost));
}

void GPUAutoencoder::init_random_weights(float ignored_val) {
    const int k = KERNEL; // Giả sử KERNEL = 3 trong .cuh
    std::mt19937 gen(1234);

    // Lambda khởi tạo trọng số theo He Initialization (cho ReLU)
    auto init_layer = [&](std::vector<float>& w, std::vector<float>& b, int in_c, int out_c) {
        // 1. Resize vectors đúng kích thước kiến trúc
        w.resize(in_c * out_c * k * k);
        b.resize(out_c);

        // 2. Tính toán chuẩn He Init
        float fan_in = (float)(in_c * k * k);
        float stddev = std::sqrt(2.0f / fan_in); 
        
        // 3. Fill Weight bằng Random Normal
        std::normal_distribution<float> d(0.0f, stddev);
        for (auto &x : w) x = d(gen);

        // 4. Fill Bias bằng 0
        std::fill(b.begin(), b.end(), 0.0f);
    };

    // --- KHỚP VỚI KIẾN TRÚC KERAS CỦA BẠN ---
    
    // Encoder
    // conv2d_1: Input 3 -> Output 256 (Params: 7,168)
    init_layer(h_w_conv1, h_b_conv1, 3, 256);
    
    // conv2d_2: Input 256 (từ pool1) -> Output 128 (Params: 295,040)
    init_layer(h_w_conv2, h_b_conv2, 256, 128);

    // Decoder
    // conv2d_3: Input 128 (từ latent) -> Output 128 (Params: 147,584)
    init_layer(h_w_dec1, h_b_dec1, 128, 128);

    // conv2d_4: Input 128 (từ upsample1) -> Output 256 (Params: 295,168)
    init_layer(h_w_dec2, h_b_dec2, 128, 256);

    // conv2d_5: Input 256 (từ upsample2) -> Output 3 (Params: 6,915)
    // Lưu ý: Layer cuối có thể dùng Xavier thay vì He, nhưng dùng He cũng không lỗi
    init_layer(h_w_dec3, h_b_dec3, 256, 3);

    // --- CẤP PHÁT GPU ---
    alloc_weights_on_device();
    alloc_activations_and_buffers();
    alloc_grads_on_device();
    device_allocated_ = true;
}

void GPUAutoencoder::free_all() {
    auto Ff = [](void*& p){ if (p) { cudaFree(p); p = nullptr; } };

    Ff((void*&)d_w_conv1); Ff((void*&)d_b_conv1);
    Ff((void*&)d_w_conv2); Ff((void*&)d_b_conv2);
    Ff((void*&)d_w_dec1);  Ff((void*&)d_b_dec1);
    Ff((void*&)d_w_dec2);  Ff((void*&)d_b_dec2);
    Ff((void*&)d_w_dec3);  Ff((void*&)d_b_dec3);

    Ff((void*&)d_input); Ff((void*&)d_act1); Ff((void*&)d_pool1); Ff((void*&)d_act2);
    Ff((void*&)d_pool2); Ff((void*&)d_latent); Ff((void*&)d_dec_act1); Ff((void*&)d_up1);
    Ff((void*&)d_dec_act2); Ff((void*&)d_up2); Ff((void*&)d_out);

    Ff((void*&)d_pool1_idx); Ff((void*&)d_pool2_idx);

    Ff((void*&)d_g_w_conv1); Ff((void*&)d_g_b_conv1);
    Ff((void*&)d_g_w_conv2); Ff((void*&)d_g_b_conv2);
    Ff((void*&)d_g_w_dec1);  Ff((void*&)d_g_b_dec1);
    Ff((void*&)d_g_w_dec2);  Ff((void*&)d_g_b_dec2);
    Ff((void*&)d_g_w_dec3);  Ff((void*&)d_g_b_dec3);

    Ff((void*&)d_grad_out); Ff((void*&)d_grad_up2); Ff((void*&)d_grad_dec_act2);
    Ff((void*&)d_grad_up1); Ff((void*&)d_grad_dec_act1); Ff((void*&)d_grad_latent);
    Ff((void*&)d_grad_pool2); Ff((void*&)d_grad_act2); Ff((void*&)d_grad_pool1); Ff((void*&)d_grad_act1);

    device_allocated_ = false;
}

// gpu_autoencoder.cu

void GPUAutoencoder::save_weights(const std::string& filename) {
    // 1. Đảm bảo dữ liệu trên Host là mới nhất
    copy_weights_to_host();

    // 2. Mở 1 file duy nhất để ghi (Binary mode)
    // Lưu ý: filename nên là "model.bin" thay vì chỉ là prefix
    std::ofstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "[save_weights] Cannot open " << filename << " for writing.\n";
        return;
    }

    // 3. Helper lambda để ghi 1 vector
    auto write_vec = [&](const std::vector<float>& v) {
        // Tùy chọn: Có thể ghi kích thước vector trước nếu muốn format linh hoạt
        // f.write((char*)&size, sizeof(int)); 
        
        // Ghi dữ liệu thô (raw bytes)
        f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
    };

    // 4. Ghi tuần tự (Thứ tự này RẤT QUAN TRỌNG, phải khớp với lúc Load)
    write_vec(h_w_conv1);
    write_vec(h_b_conv1);
    write_vec(h_w_conv2);
    write_vec(h_b_conv2);
    write_vec(h_w_dec1);
    write_vec(h_b_dec1);
    write_vec(h_w_dec2);
    write_vec(h_b_dec2);
    write_vec(h_w_dec3);
    write_vec(h_b_dec3);

    f.close();
    std::cout << "[GPUAutoencoder] All weights saved to '" << filename << "'\n";
}

// gpu_autoencoder.cu

bool GPUAutoencoder::load_weights(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) {
        std::cerr << "[GPUAutoencoder] load_weights: Cannot open '" << filename << "'\n";
        return false;
    }

    // 1. Resize các vector trên Host trước để đảm bảo đủ chỗ chứa
    const int k = KERNEL; 
    // Nếu vector chưa được resize (do chưa gọi init), ta resize ở đây
    if (h_w_conv1.empty()) h_w_conv1.resize(256 * 3 * k * k);
    if (h_b_conv1.empty()) h_b_conv1.resize(256);
    if (h_w_conv2.empty()) h_w_conv2.resize(128 * 256 * k * k);
    if (h_b_conv2.empty()) h_b_conv2.resize(128);
    if (h_w_dec1.empty())  h_w_dec1.resize(128 * 128 * k * k);
    if (h_b_dec1.empty())  h_b_dec1.resize(128);
    if (h_w_dec2.empty())  h_w_dec2.resize(256 * 128 * k * k);
    if (h_b_dec2.empty())  h_b_dec2.resize(256);
    if (h_w_dec3.empty())  h_w_dec3.resize(3 * 256 * k * k);
    if (h_b_dec3.empty())  h_b_dec3.resize(3);

    // 2. Helper lambda để đọc 1 vector
    auto read_vec = [&](std::vector<float>& v) {
        f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
    };

    // 3. Đọc tuần tự (THỨ TỰ PHẢI Y HỆT LÚC SAVE)
    read_vec(h_w_conv1);
    read_vec(h_b_conv1);
    read_vec(h_w_conv2);
    read_vec(h_b_conv2);
    read_vec(h_w_dec1);
    read_vec(h_b_dec1);
    read_vec(h_w_dec2);
    read_vec(h_b_dec2);
    read_vec(h_w_dec3);
    read_vec(h_b_dec3);

    if (!f) {
        std::cerr << "[Error] File ended prematurely or read error.\n";
        return false;
    }
    
    f.close();

    // 4. Copy trọng số từ Host lên GPU
    copy_weights_to_device();
    
    std::cout << "[GPUAutoencoder] Successfully loaded weights from '" << filename << "'\n";
    return true;
}
