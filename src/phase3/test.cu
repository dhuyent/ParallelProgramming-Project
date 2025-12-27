#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <algorithm>

#include "gpu_opt.h"
#include "../data_loader.h"

static constexpr int H = 32;
static constexpr int W = 32;
static constexpr int C = 3;
static constexpr int IMG_SIZE = C * H * W;

void run_and_save_reconstruction(Phase3Engine &eng, CIFAR10Dataset &ds, const std::string &filename)
{
    const int num_samples = 10;
    const int batch_size = eng.p.batch;
    printf("[Test] Preparing batch for %d samples...\n", num_samples);
    std::vector<float> h_input_batch(batch_size * IMG_SIZE, 0.0f);

    for (int i = 0; i < num_samples; ++i)
    {
        std::memcpy(h_input_batch.data() + i * IMG_SIZE,
                    ds.test_images[i].data.data(),
                    IMG_SIZE * sizeof(float));
    }

    printf("[Test] Forwarding through GPU using forward_only...\n");

    cudaStream_t s = eng.sCompute[0];
    CUDA_CHECK(cudaMemcpyAsync(eng.b.x, h_input_batch.data(),
                               batch_size * IMG_SIZE * sizeof(float),
                               cudaMemcpyHostToDevice, s));

    eng.forward_only(batch_size, s);
    eng.sync_all();

    std::vector<float> h_output_batch(batch_size * IMG_SIZE);
    CUDA_CHECK(cudaMemcpy(h_output_batch.data(), eng.b.out,
                          batch_size * IMG_SIZE * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::ofstream out(filename, std::ios::binary);
    if (!out)
    {
        std::cerr << "[Error] Failed to open " << filename << " for writing.\n";
        return;
    }

    out.write(reinterpret_cast<const char *>(&num_samples), sizeof(int));
    out.write(reinterpret_cast<const char *>(h_input_batch.data()),
              num_samples * IMG_SIZE * sizeof(float));
    out.write(reinterpret_cast<const char *>(h_output_batch.data()),
              num_samples * IMG_SIZE * sizeof(float));

    out.close();
    printf("[Success] Saved %d reconstructed samples to '%s'.\n", num_samples, filename.c_str());
}

int main(int argc, char **argv)
{
    std::string cifar_dir = "./data/cifar-10-batches-bin";
    std::string model_path = "output/gpu_opt_model.bin";

    CIFAR10Dataset ds(cifar_dir);
    ds.load_data();
    if (ds.test_images.empty())
    {
        std::cerr << "[Error] No test images loaded.\n";
        return -1;
    }

    AEParams p;
    p.batch = 128;
    Phase3Engine eng;
    eng.init(p);

    std::ifstream f(model_path);
    if (!f.good())
    {
        std::cerr << "[Error] Model file not found: " << model_path << "\n";
        return -1;
    }
    f.close();

    printf("[System] Loading weights from %s...\n", model_path.c_str());
    eng.load_from_file(model_path, eng.sCompute[0]);
    eng.sync_all();

    run_and_save_reconstruction(eng, ds, "output/gpu_opt_reconstruction.bin");
    eng.shutdown();
    return 0;
}