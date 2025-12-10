// test.cu
// Compile: nvcc -O2 -arch=sm_70 test.cu gpu_autoencoder.cu kernels.cu data_loader.cpp -o run_test

#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "gpu_autoencoder.cuh"
#include "data_loader.h"

// Hàm này chỉ lưu file binary (.bin) để chấm điểm hoặc dùng Python đọc sau
void save_binary_results(GPUAutoencoder& model, CIFAR10Dataset& dataset, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[Error] Cannot open " << filename << " for writing.\n";
        return;
    }

    // Lấy 10 ảnh test đầu tiên
    int num_samples = 10;
    if (dataset.test_images.size() < num_samples) num_samples = (int)dataset.test_images.size();
    
    // Ghi header: số lượng mẫu
    out.write(reinterpret_cast<const char*>(&num_samples), sizeof(int));

    size_t img_bytes = 3 * 32 * 32 * sizeof(float);
    std::vector<float> h_recon(3 * 32 * 32);

    std::cout << "[Test] Running inference on " << num_samples << " images...\n";

    for (int i = 0; i < num_samples; ++i) {
        float* h_in = dataset.test_images[i].data();

        // 1. Copy Input -> GPU
        CHECK(cudaMemcpy(model.d_input, h_in, img_bytes, cudaMemcpyHostToDevice));

        // 2. Forward (Dùng method của class)
        // Truyền d_input vào vị trí target vì ở bước test ta không quan tâm loss
        model.forward(model.d_input, model.d_input);

        // 3. Copy Output -> CPU
        CHECK(cudaMemcpy(h_recon.data(), model.d_out, img_bytes, cudaMemcpyDeviceToHost));

        // 4. Ghi vào file Binary: [Input Raw Bytes] [Output Raw Bytes]
        out.write(reinterpret_cast<const char*>(h_in), img_bytes);
        out.write(reinterpret_cast<const char*>(h_recon.data()), img_bytes);
    }

    out.close();
    std::cout << "[Report] Saved binary results to '" << filename << "'\n";
}

int main() {
    // 1. Load Data
    std::string cifar_dir = "./cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data(); 

    if (ds.test_images.empty()) {
        std::cerr << "Error: No test images found.\n";
        return -1;
    }
    
    // 2. Setup Model
    GPUAutoencoder model;
    model.init_random_weights(0.0f); // Cấp phát bộ nhớ

    // 3. Load Weights
    std::string model_path = "ae_final.bin"; 
    // Kiểm tra file tồn tại, nếu không có final thì tìm best
    std::ifstream fcheck(model_path);
    if (!fcheck.good()) model_path = "ae_best_model.bin"; 
    fcheck.close();

    printf("Loading weights from '%s'...\n", model_path.c_str());
    if (!model.load_weights(model_path)) {
        fprintf(stderr, "Failed to load weights!\n");
        return -1;
    }

    // 4. Run Test & Save Binary
    save_binary_results(model, ds, "reconstruction_results.bin");

    return 0;
}