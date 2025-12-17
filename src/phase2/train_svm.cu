// train_svm.cu
// Compile (ThunderSVM GPU): 
// nvcc -O2 -arch=sm_75 train_svm.cu gpu_autoencoder.cu kernels.cu data_loader.cpp svm_wrapper.cpp -I./thundersvm/include -L./thundersvm/build/lib -lthundersvm -Wl,-rpath,./thundersvm/build/lib -o train_svm

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream> // Cần để check file tồn tại

#include "gpu_autoencoder.cuh"
#include "data_loader.h"
#include "svm_wrapper.h"

// Hàm helper trích xuất đặc trưng
// Chuyển đổi từ Ảnh (3x32x32) -> Latent Vector (8192)
double extract_features_wrapper(GPUAutoencoder& ae, 
                                const std::vector<std::vector<float>>& images,
                                std::vector<std::vector<float>>& out_features) 
{
    // Kích thước latent: 128 channels * 8 height * 8 width
    int latent_dim = 128 * 8 * 8;
    size_t input_bytes = 3 * 32 * 32 * sizeof(float);
    
    // Buffer tạm trên Host
    std::vector<float> h_latent(latent_dim);
    
    // Resize vector đích để tránh cấp phát lại nhiều lần
    out_features.resize(images.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); ++i) {
        // 1. Copy ảnh từ CPU -> GPU (vào buffer ae.d_input)
        cudaMemcpy(ae.d_input, images[i].data(), input_bytes, cudaMemcpyHostToDevice);
        
        // 2. Chạy Encoder (Input -> Latent)
        // Kết quả sẽ được copy từ GPU ra h_latent.data()
        ae.extract_features(ae.d_input, h_latent.data());
        
        // 3. Lưu vào danh sách feature
        out_features[i] = h_latent;

        // In dấu chấm tiến độ mỗi 2000 ảnh
        if ((i + 1) % 2000 == 0) std::cout << "." << std::flush;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    // 1. Load Data
    std::string cifar_dir = "./cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data(); // Data Loader mới đã tự động chỉ lấy 10k ảnh và normalize

    if (ds.train_images.empty()) {
        std::cerr << "Error: No training data found!\n";
        return -1;
    }

    // 2. Load Autoencoder
    GPUAutoencoder ae;
    ae.init_random_weights(0.0f); // Cấp phát bộ nhớ GPU
    
    // Logic kiểm tra file model thông minh hơn
    std::string model_path = "ae_final.bin";
    std::ifstream fcheck(model_path);
    if (!fcheck.good()) {
        fcheck.close();
        model_path = "ae_best_model.bin"; // Thử tìm file backup
    } else {
        fcheck.close();
    }
    
    if (!ae.load_weights(model_path)) {
        std::cerr << "Failed to load Autoencoder weights from " << model_path << "!\n"; 
        return -1;
    }
    std::cout << "Loaded Autoencoder: " << model_path << "\n";

    // 3. Extract Train Features
    std::cout << "Extracting TRAIN features (" << ds.train_images.size() << " samples)...";
    
    std::vector<std::vector<float>> train_features;
    double time_extract = extract_features_wrapper(ae, ds.train_images, train_features);
    
    std::cout << " Done in " << time_extract / 1000.0 << "s\n";

    // 4. Train SVM (ThunderSVM)
    SVMClassifier svm;
    
    // C=10, Gamma=0 (nghĩa là Auto: 1/num_features)
    // Bạn có thể chỉnh C=100 hoặc C=1 để thử nghiệm độ chính xác
    svm.set_parameters(10.0, 0); 
    
    std::cout << "Training SVM (RBF) on GPU... \n";
    auto start_svm = std::chrono::high_resolution_clock::now();
    
    // Gọi hàm train (Wrapper sẽ tự chuyển đổi data sang định dạng ThunderSVM)
    svm.train(train_features, ds.train_labels);
    
    auto end_svm = std::chrono::high_resolution_clock::now();
    double time_train = std::chrono::duration<double, std::milli>(end_svm - start_svm).count();
    
    std::cout << "SVM Training Done in " << time_train / 1000.0 << "s\n";

    // 5. Save Model
    svm.save_model("svm_cifar10.model");

    // Report Time
    std::cout << "\n--- TIME REPORT ---\n";
    std::cout << "Feature Extraction: " << time_extract / 1000.0 << " sec\n";
    std::cout << "SVM Training:       " << time_train / 1000.0 << " sec\n";
    std::cout << "Total Time:         " << (time_extract + time_train) / 1000.0 << " sec\n";

    return 0;
}