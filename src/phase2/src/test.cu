// test.cu

#include <cstdio>
#include <vector>
#include <chrono>
#include <string>
#include <iostream>
#include <fstream>

#include "gpu_autoencoder.cuh"   // GPUAutoencoder class + device pointers
#include "forward.cuh"          // forward_single_device, forward_batch helpers
#include "backward.cuh" 
#include "data_loader.h"

// Khai báo prototype hàm save (hoặc include file chứa nó)
void save_reconstruction_samples(GPUAutoencoder& model, CIFAR10Dataset& dataset, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[Error] Cannot open " << filename << " for writing.\n";
        return;
    }

    // Lấy 10 ảnh test đầu tiên (hoặc ít hơn nếu dataset nhỏ)
    int num_samples = 10;
    if (dataset.test_images.size() < num_samples) num_samples = (int)dataset.test_images.size();
    
    // Ghi header: số lượng mẫu
    out.write(reinterpret_cast<const char*>(&num_samples), sizeof(int));

    size_t img_size = 3 * 32 * 32;
    size_t img_bytes = img_size * sizeof(float);
    
    // Buffer trên Host để hứng kết quả từ GPU
    std::vector<float> h_recon(img_size);

    for (int i = 0; i < num_samples; ++i) {
        // 1. Lấy con trỏ dữ liệu ảnh gốc (Host)
        float* h_in = dataset.test_images[i].data();

        // 2. Copy ảnh gốc từ Host -> Device (model.d_input)
        CHECK(cudaMemcpy(model.d_input, h_in, img_bytes, cudaMemcpyHostToDevice));

        // 3. Chạy Forward Pass
        // Lưu ý: forward_single_device cần target để tính loss. 
        // Ta truyền model.d_input vào vị trí target luôn (để tính loss giả), 
        // mục đích chính là để nó điền dữ liệu vào model.d_out.
        forward_single_device(model, model.d_input, model.d_input);

        // 4. Copy ảnh tái tạo từ Device (model.d_out) -> Host (h_recon)
        CHECK(cudaMemcpy(h_recon.data(), model.d_out, img_bytes, cudaMemcpyDeviceToHost));

        // 5. Ghi vào file: [Ảnh Gốc] rồi đến [Ảnh Tái Tạo]
        out.write(reinterpret_cast<const char*>(h_in), img_bytes);
        out.write(reinterpret_cast<const char*>(h_recon.data()), img_bytes);
    }

    out.close();
    std::cout << "[Report] Saved " << num_samples << " reconstruction pairs to " << filename << std::endl;
}

int main() {
    // 1. Load Data (Cần data test)
    std::string cifar_dir = "./cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data(); // Load data để lấy ảnh test

    GPUAutoencoder model;
    // Cấp phát bộ nhớ GPU (giá trị weight ban đầu là random, sẽ bị load đè lên)
    model.init_random_weights(0.0f); 

    // 3. LOAD WEIGHTS (Đọc từ file đơn .bin)
    // Bạn nên ưu tiên load 'ae_best_model.bin' nếu có, nếu không thì 'ae_final.bin'
    std::string model_path = "ae_final.bin"; 

    printf("Loading weights from '%s'...\n", model_path.c_str());
    if (!model.load_weights(model_path)) {
        fprintf(stderr, "Failed to load weights! Please check if file exists.\n");
        return -1;
    }

    // 4. Chạy kiểm tra và lưu kết quả
    save_reconstruction_samples(model, ds, "reconstruction_results.bin");

    return 0;
}