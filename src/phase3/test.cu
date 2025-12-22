#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

#include "gpu_opt.h"     // Chứa định nghĩa Phase3Engine
#include "data_loader.h" // Chứa định nghĩa CIFAR10Dataset

// Hằng số CIFAR-10
static constexpr int H = 32;
static constexpr int W = 32;
static constexpr int C = 3;
static constexpr int IMG_SIZE = C * H * W;

/**
 * Hàm lưu kết quả tái cấu trúc (Reconstruction)
 */
void run_and_save_reconstruction(Phase3Engine& eng, CIFAR10Dataset& ds, const std::string& filename) {
    // 1. Chọn mẫu thử (Ví dụ lấy 10 ảnh đầu tiên từ tập Test)
    const int num_samples = 10;
    const int batch_size = eng.p.batch; // Mặc định là 128
    
    printf("[Test] Preparing batch for %d samples...\n", num_samples);

    // Chuẩn bị mảng Host (Padding 0 cho các phần dư của batch để tránh dữ liệu rác)
    std::vector<float> h_input_batch(batch_size * IMG_SIZE, 0.0f);
    
    for (int i = 0; i < num_samples; ++i) {
        std::memcpy(h_input_batch.data() + i * IMG_SIZE, 
                    ds.test_images[i].data.data(), 
                    IMG_SIZE * sizeof(float));
    }

    // 2. Thực hiện Inference
    printf("[Test] Forwarding through GPU using forward_only...\n");
    
    cudaStream_t s = eng.sCompute[0];

    // Upload dữ liệu từ Host lên Device buffer (b.x)
    CUDA_CHECK(cudaMemcpyAsync(eng.b.x, h_input_batch.data(), 
                               batch_size * IMG_SIZE * sizeof(float), 
                               cudaMemcpyHostToDevice, s));

    // Gọi hàm forward_only đã bổ sung vào engine
    eng.forward_only(batch_size, s);

    // Chờ GPU hoàn thành tính toán
    eng.sync_all();

    // 3. Lấy kết quả về Host
    std::vector<float> h_output_batch(batch_size * IMG_SIZE);
    CUDA_CHECK(cudaMemcpy(h_output_batch.data(), eng.b.out, 
                          batch_size * IMG_SIZE * sizeof(float), 
                          cudaMemcpyDeviceToHost));

    // 4. Ghi file nhị phân cho Python visualize
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "[Error] Failed to open " << filename << " for writing.\n";
        return;
    }

    // Header: Số lượng mẫu thực tế
    out.write(reinterpret_cast<const char*>(&num_samples), sizeof(int));

    // Dữ liệu gốc (chỉ lấy num_samples ảnh đầu)
    out.write(reinterpret_cast<const char*>(h_input_batch.data()), 
              num_samples * IMG_SIZE * sizeof(float));

    // Dữ liệu tái cấu trúc (chỉ lấy num_samples ảnh đầu)
    out.write(reinterpret_cast<const char*>(h_output_batch.data()), 
              num_samples * IMG_SIZE * sizeof(float));

    out.close();
    printf("[Success] Saved %d reconstructed samples to '%s'.\n", num_samples, filename.c_str());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <cifar_dir> [model_path]\n", argv[0]);
        return 1;
    }

    std::string cifar_dir = argv[1];
    std::string model_path = (argc >= 3) ? argv[2] : "trained_ae_weights.bin";

    // 1. Load Dataset
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data();
    if (ds.test_images.empty()) {
        std::cerr << "[Error] No test images loaded.\n";
        return -1;
    }

    // 2. Initialize Engine
    AEParams p;
    p.batch = 128; 
    Phase3Engine eng;
    eng.init(p);

    // 3. Load Weights
    std::ifstream f(model_path);
    if (!f.good()) {
        std::cerr << "[Error] Model file not found: " << model_path << "\n";
        return -1;
    }
    f.close();

    printf("[System] Loading weights from %s...\n", model_path.c_str());
    eng.load_from_file(model_path, eng.sCompute[0]);
    eng.sync_all();

    // 4. Run Test
    run_and_save_reconstruction(eng, ds, "reconstruction.bin");

    eng.shutdown();
    return 0;
}