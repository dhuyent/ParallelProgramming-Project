#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>

#include <cuda_runtime.h>
#include "gpu_autoencoder.cuh"
#include "data_loader.h"

double extract_and_save_to_bin(GPUAutoencoder& ae, 
                               const std::vector<std::vector<float>>& images, // Sửa ở đây
                               const std::vector<int>& labels,
                               const std::string& feat_file,
                               const std::string& label_file) 
{
    std::ofstream f_feat(feat_file, std::ios::binary);
    std::ofstream f_label(label_file, std::ios::binary);
    
    if (!f_feat || !f_label) {
        std::cerr << "Error: Cannot open output files for writing!\n";
        return 0;
    }

    const int latent_dim = 128 * 8 * 8;
    size_t input_bytes = 3 * 32 * 32 * sizeof(float);
    std::vector<float> h_feat_buffer(latent_dim);

    auto start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < images.size(); ++i) {
        cudaMemcpy(ae.d_input, images[i].data(), input_bytes, cudaMemcpyHostToDevice);
        
        ae.extract_features(ae.d_input, h_feat_buffer.data());
        
        f_feat.write(reinterpret_cast<const char*>(h_feat_buffer.data()), latent_dim * sizeof(float));
        
        int lbl = labels[i];
        f_label.write(reinterpret_cast<const char*>(&lbl), sizeof(int));

        if ((i + 1) % 2000 == 0) std::cout << "." << std::flush;
    }

    auto end = std::chrono::high_resolution_clock::now();
    f_feat.close();
    f_label.close();
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}
int main() {
    std::string cifar_dir = "./data/cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data(); 

    GPUAutoencoder ae; 
    if (!ae.load_weights("output/gpu_basic_model.bin")) {
        std::cerr << "Error: Could not load weights!\n";
        return -1;
    }

    std::cout << "Extracting & Saving TRAIN features (" << ds.train_images.size() << " samples)...";
    double time_train = extract_and_save_to_bin(ae, ds.train_images, ds.train_labels, "train_features.bin", "train_labels.bin");
    std::cout << " Done in " << time_train / 1000.0 << "s\n";

    std::cout << "Extracting & Saving TEST features (" << ds.test_images.size() << " samples)...";
    double time_test = extract_and_save_to_bin(ae, ds.test_images, ds.test_labels, "test_features.bin", "test_labels.bin");
    std::cout << " Done in " << time_test / 1000.0 << "s\n";

    std::cout << "\n--- EXTRACTION COMPLETE ---\n";
    std::cout << "Total Time: " << (time_train + time_test) / 1000.0 << " sec\n";
    std::cout << "Files generated: train_features.bin, train_labels.bin, test_features.bin, test_labels.bin\n";

    return 0;
}