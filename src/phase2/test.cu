#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

#include "gpu_autoencoder.cuh"
#include "data_loader.h"

void save_binary_results(GPUAutoencoder& model, CIFAR10Dataset& dataset, const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "[Error] Cannot open " << filename << " for writing.\n";
        return;
    }

    int num_samples = 10;
    if (dataset.test_images.size() < num_samples) num_samples = (int)dataset.test_images.size();
    
    out.write(reinterpret_cast<const char*>(&num_samples), sizeof(int));

    size_t img_bytes = 3 * 32 * 32 * sizeof(float);
    std::vector<float> h_recon(3 * 32 * 32);

    std::cout << "[Test] Running inference on " << num_samples << " images...\n";

    for (int i = 0; i < num_samples; ++i) {
        float* h_in = dataset.test_images[i].data();

        CHECK(cudaMemcpy(model.d_input, h_in, img_bytes, cudaMemcpyHostToDevice));

        
        model.forward(model.d_input, model.d_input);
        CHECK(cudaMemcpy(h_recon.data(), model.d_out, img_bytes, cudaMemcpyDeviceToHost));
        out.write(reinterpret_cast<const char*>(h_in), img_bytes);
        out.write(reinterpret_cast<const char*>(h_recon.data()), img_bytes);
    }

    out.close();
    std::cout << "[Report] Saved binary results to '" << filename << "'\n";
}

int main() {
    std::string cifar_dir = "./data/cifar-10-batches-bin";
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data(); 

    if (ds.test_images.empty()) {
        std::cerr << "Error: No test images found.\n";
        return -1;
    }
    
    GPUAutoencoder model;
    model.init_random_weights(0.0f); 
    std::string model_path = "output/gpu_basic_model.bin"; 


    printf("Loading weights from '%s'...\n", model_path.c_str());
    if (!model.load_weights(model_path)) {
        fprintf(stderr, "Failed to load weights!\n");
        return -1;
    }

    save_binary_results(model, ds, "output/gpu_basic_reconstruction.bin");

    return 0;
}