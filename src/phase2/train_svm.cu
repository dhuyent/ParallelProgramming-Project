// train_svm.cu
// Compile: nvcc -O2 -arch=sm_70 train_svm.cu gpu_autoencoder.cu kernels.cu forward.cu backward.cu data_loader.cpp svm.cpp svm_wrapper.cpp -o train_svm

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include "gpu_autoencoder.cuh"
#include "data_loader.h"
#include "svm_wrapper.h"

// Hàm helper trích xuất đặc trưng (Copy vào đây để dùng)
double extract_features_wrapper(GPUAutoencoder& ae, 
                                const std::vector<std::vector<float>>& images,
                                std::vector<std::vector<float>>& out_features) 
{
    int latent_dim = 128 * 8 * 8;
    size_t input_bytes = 3 * 32 * 32 * sizeof(float);
    std::vector<float> h_latent(latent_dim);
    out_features.reserve(images.size());

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); ++i) {
        cudaMemcpy(ae.d_input, images[i].data(), input_bytes, cudaMemcpyHostToDevice);
        ae.extract_features(ae.d_input, h_latent.data());
        out_features.push_back(h_latent);

        if ((i + 1) % 5000 == 0) std::cout << "." << std::flush;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {

    // 1. Load Data
    std::string cifar_dir = "./cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data();

    // 2. Load Autoencoder
    GPUAutoencoder ae;
    ae.init_random_weights(0.0f);
    std::string model_path = "ae_final.bin";
    FILE* f = fopen(model_path.c_str(), "rb");
    if (!f) model_path = "ae_final.bin"; else fclose(f);
    
    if (!ae.load_weights(model_path)) {
        std::cerr << "Failed to load Autoencoder!\n"; return -1;
    }
    std::cout << "Loaded Autoencoder: " << model_path << "\n";

    // 3. Extract Train Features
    std::cout << "Extracting TRAIN features (" << ds.train_images.size() << " samples)...";
    std::vector<std::vector<float>> train_features;
    double time_extract = extract_features_wrapper(ae, ds.train_images, train_features);
    std::cout << " Done in " << time_extract / 1000.0 << "s\n";

    // 4. Train SVM
    SVMClassifier svm;
    svm.set_parameters(10.0, 0); // C=10, Gamma=Auto
    
    std::cout << "Training SVM (RBF)... ";
    auto start_svm = std::chrono::high_resolution_clock::now();
    svm.train(train_features, ds.train_labels);
    auto end_svm = std::chrono::high_resolution_clock::now();
    double time_train = std::chrono::duration<double, std::milli>(end_svm - start_svm).count();
    
    std::cout << "Done in " << time_train / 1000.0 << "s\n";

    // 5. Save Model
    svm.save_model("svm_cifar10.model");

    // Report Time
    std::cout << "\n--- TIME REPORT ---\n";
    std::cout << "Feature Extraction (Train): " << time_extract / 1000.0 << " sec\n";
    std::cout << "SVM Training Time:          " << time_train / 1000.0 << " sec\n";

    return 0;
}