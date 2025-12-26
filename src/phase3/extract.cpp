#include "../data_loader.h"
#include "../utils.h"
#include "gpu_opt.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>

static constexpr int IMG_FLOATS = 32 * 32 * 3; 
static constexpr int FEAT_DIM = 8192;           

static void pack_images(const std::vector<Tensor> &images, std::vector<float> &out) {
    out.resize(images.size() * IMG_FLOATS);
    for (size_t i = 0; i < images.size(); ++i) {
        std::memcpy(out.data() + i * IMG_FLOATS, images[i].data.data(), IMG_FLOATS * sizeof(float));
    }
}

template <typename T>
static void save_bin(const std::string &path, const std::vector<T> &data) {
    std::ofstream f(path, std::ios::binary);
    if (f) {
        f.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
    }
}

int main(int argc, char **argv) {
    std::string cifar_dir = "./data/cifar-10-batches-bin"; 
    
    std::string weights_path = "./output/gpu_opt_model.bin";

    // 1. Load Data
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data();
    
    std::vector<float> train_x_contig, test_x_contig;
    pack_images(ds.train_images, train_x_contig);
    pack_images(ds.test_images, test_x_contig);

    // 2. Init Engine
    AEParams p;
    p.batch = 128;
    Phase3Engine eng;
    eng.init(p);

    // 3. Load Weights
    eng.load_from_file(weights_path, eng.sCompute[0]);
    eng.sync_all();

    // 4. Feature Extraction & Timing
    std::vector<float> train_feat(ds.train_images.size() * FEAT_DIM);
    std::vector<float> test_feat(ds.test_images.size() * FEAT_DIM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::cout << "--- Feature Extraction Timing ---" << std::endl;

    // Time for Train Set (50K images)
    CUDA_CHECK(cudaEventRecord(start, eng.sCompute[0]));
    eng.extract_features(train_x_contig.data(), train_feat.data(), (int)ds.train_images.size());
    CUDA_CHECK(cudaEventRecord(stop, eng.sCompute[0]));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float train_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&train_ms, start, stop));
    std::cout << "Train set (50K images): " << std::fixed << std::setprecision(3) << train_ms / 1000.0f << "s" << std::endl;

    // Time for Test Set (10K images)
    CUDA_CHECK(cudaEventRecord(start, eng.sCompute[0]));
    eng.extract_features(test_x_contig.data(), test_feat.data(), (int)ds.test_images.size());
    CUDA_CHECK(cudaEventRecord(stop, eng.sCompute[0]));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float test_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&test_ms, start, stop));
    std::cout << "Test set (10K images):  " << std::fixed << std::setprecision(3) << test_ms / 1000.0f << "s" << std::endl;

    float total_s = (train_ms + test_ms) / 1000.0f;
    std::cout << "Total Extraction Time:  " << total_s << "s" << std::endl;
    std::cout << "---------------------------------" << std::endl;

    // 5. Save results
    save_bin("train_feat.bin", train_feat);
    save_bin("test_feat.bin", test_feat);
    save_bin("train_label.bin", ds.train_labels);
    save_bin("test_label.bin", ds.test_labels);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    eng.shutdown();

    return 0;
}