// eval_svm.cu
// Compile: nvcc -O2 -arch=sm_75 eval_svm.cu gpu_autoencoder.cu kernels.cu data_loader.cpp svm_wrapper.cpp -I./thundersvm/include -L./thundersvm/build/lib -lthundersvm -Xlinker -rpath -Xlinker ./thundersvm/build/lib -o eval_svm

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>

#include "gpu_autoencoder.cuh"
#include "data_loader.h"
#include "svm_wrapper.h"

const std::string CLASS_NAMES[10] = {
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
};

// Hàm trích xuất đặc trưng
double extract_features_wrapper(GPUAutoencoder& ae, 
                                const std::vector<std::vector<float>>& images,
                                std::vector<std::vector<float>>& out_features) 
{
    int latent_dim = 128 * 8 * 8;
    size_t input_bytes = 3 * 32 * 32 * sizeof(float);
    std::vector<float> h_latent(latent_dim);
    out_features.resize(images.size()); 

    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < images.size(); ++i) {
        cudaMemcpy(ae.d_input, images[i].data(), input_bytes, cudaMemcpyHostToDevice);
        ae.extract_features(ae.d_input, h_latent.data());
        out_features[i] = h_latent;
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== PHASE 2: SVM EVALUATION (BATCH MODE) ===\n";

    // 1. Load Data
    std::string cifar_dir = "./cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data(); 
    std::cout << "Loaded " << ds.test_images.size() << " test images.\n";

    // 2. Load Models
    GPUAutoencoder ae;
    ae.init_random_weights(0.0f);
    
    std::string ae_path = "ae_final.bin";
    std::ifstream fcheck(ae_path);
    if (!fcheck.good()) {
        std::cerr << "Error: File " << ae_path << " not found!\n";
        return -1;
    }
    fcheck.close();

    if (!ae.load_weights(ae_path)) return -1;
    
    SVMClassifier svm;
    if (!svm.load_model("svm_cifar10.model")) {
        std::cerr << "Error: Could not load svm_cifar10.model\n";
        return -1;
    }
    std::cout << "Models loaded.\n";

    // 3. Extract Features
    std::cout << "Extracting features... ";
    std::vector<std::vector<float>> test_features;
    double time_extract = extract_features_wrapper(ae, ds.test_images, test_features);
    std::cout << "Done (" << time_extract / 1000.0 << "s)\n";

    // 4. Batch Prediction (QUAN TRỌNG: CHẠY SONG SONG)
    std::cout << "Predicting all samples on GPU (Parallel)... \n";
    auto start_pred = std::chrono::high_resolution_clock::now();
    
    // Gọi hàm batch
    std::vector<double> all_preds = svm.predict_batch(test_features);
    
    auto end_pred = std::chrono::high_resolution_clock::now();
    double time_pred = std::chrono::duration<double, std::milli>(end_pred - start_pred).count();
    std::cout << "Prediction Done (" << time_pred / 1000.0 << "s)\n";

    // 5. Tính toán kết quả
    if (all_preds.size() != ds.test_labels.size()) {
        std::cerr << "Error: Size mismatch! Preds: " << all_preds.size() << ", Labels: " << ds.test_labels.size() << "\n";
        return -1;
    }

    int correct_total = 0;
    int class_correct[10] = {0};
    int class_total[10] = {0};

    for (size_t i = 0; i < all_preds.size(); ++i) {
        int pred_label = (int)all_preds[i];
        int true_label = ds.test_labels[i];
        
        if (pred_label == true_label) {
            correct_total++;
            class_correct[true_label]++;
        }
        class_total[true_label]++;
    }

    double overall_acc = (double)correct_total / ds.test_labels.size() * 100.0;

    // 6. Report
    std::cout << "\n==================================================\n";
    std::cout << "             EVALUATION REPORT                    \n";
    std::cout << "==================================================\n";
    std::cout << "Feature Extraction:        " << time_extract / 1000.0 << " s\n";
    std::cout << "SVM Batch Prediction:      " << time_pred / 1000.0 << " s\n";
    std::cout << "Overall Accuracy:          " << std::fixed << std::setprecision(2) << overall_acc << "%\n\n";

    std::cout << "Per-class Accuracy Breakdown:\n";
    std::cout << "--------------------------------------------------\n";
    std::cout << std::left << std::setw(15) << "Class Name" << std::setw(10) << "Total" << std::setw(10) << "Correct" << "Accuracy\n";
    std::cout << "--------------------------------------------------\n";
    for (int i = 0; i < 10; ++i) {
        double cls_acc = 0.0;
        if (class_total[i] > 0) cls_acc = (double)class_correct[i] / class_total[i] * 100.0;
        
        std::cout << std::left << std::setw(15) << CLASS_NAMES[i]
                  << std::setw(10) << class_total[i]
                  << std::setw(10) << class_correct[i]
                  << std::fixed << std::setprecision(2) << cls_acc << "%\n";
    }
    std::cout << "--------------------------------------------------\n";

    return 0;
}