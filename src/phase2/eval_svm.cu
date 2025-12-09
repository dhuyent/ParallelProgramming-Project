// eval_svm.cu
// Compile: nvcc -O2 -arch=sm_70 eval_svm.cu gpu_autoencoder.cu kernels.cu forward.cu backward.cu data_loader.cpp svm.cpp svm_wrapper.cpp -o eval_svm

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>

#include "gpu_autoencoder.cuh"
#include "data_loader.h"
#include "svm_wrapper.h"

const std::string CLASS_NAMES[10] = {
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
};

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
    }
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== PHASE 2: SVM EVALUATION ===\n";

    // 1. Load Data (Chỉ cần Test set)
    std::string cifar_dir = "./cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data();
    
    // Normalize Test images
    for(auto& img : ds.test_images) for(float& p : img) p /= 255.0f;

    // 2. Load Models
    GPUAutoencoder ae;
    ae.init_random_weights(0.0f);
    // Load AE
    std::string ae_path = "ae_best_model.bin";
    FILE* f = fopen(ae_path.c_str(), "rb");
    if (!f) ae_path = "ae_final.bin"; else fclose(f);
    if (!ae.load_weights(ae_path)) { std::cerr << "Error loading AE.\n"; return -1; }

    // Load SVM
    SVMClassifier svm;
    if (!svm.load_model("svm_cifar10.model")) {
        std::cerr << "Error: Could not load 'svm_cifar10.model'. Did you run train_svm first?\n";
        return -1;
    }
    std::cout << "Models loaded successfully.\n";

    // 3. Extract Test Features
    std::cout << "Extracting TEST features (" << ds.test_images.size() << " samples)...";
    std::vector<std::vector<float>> test_features;
    double time_extract = extract_features_wrapper(ae, ds.test_images, test_features);
    std::cout << " Done in " << time_extract / 1000.0 << "s\n";

    // 4. Evaluation Loop
    std::cout << "Evaluating accuracy...\n";
    int correct_total = 0;
    int class_correct[10] = {0};
    int class_total[10] = {0};

    for (size_t i = 0; i < test_features.size(); ++i) {
        double pred = svm.predict(test_features[i]);
        int true_label = ds.test_labels[i];
        
        if ((int)pred == true_label) {
            correct_total++;
            class_correct[true_label]++;
        }
        class_total[true_label]++;
    }

    double overall_acc = (double)correct_total / test_features.size() * 100.0;

    // 5. Print Report
    std::cout << "\n==================================================\n";
    std::cout << "             EVALUATION REPORT                    \n";
    std::cout << "==================================================\n";
    std::cout << "Feature Extraction (Test): " << time_extract / 1000.0 << " sec\n";
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