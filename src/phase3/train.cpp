#include "../data_loader.h"
#include "../utils.h"
#include "gpu_opt.h"

#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

static constexpr int H0 = 32, W0 = 32, Cin = 3;
static constexpr int IMG_FLOATS = H0 * W0 * Cin; 

static inline const float *tensor_ptr(const Tensor &t) {
    return t.data.data(); 
}

void print_gpu_memory_usage() {
    size_t free_byte, total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));
    double total_db = (double)total_byte / (1024.0 * 1024.0);
    double used_db = total_db - (double)free_byte / (1024.0 * 1024.0);
    std::cout << "[Memory] GPU Usage: " << std::fixed << std::setprecision(2) 
              << used_db << " MB / " << total_db << " MB" << std::endl;
}

static void pack_all_contiguous(const std::vector<Tensor> &images, std::vector<float> &out_contig) {
    const int N = (int)images.size();
    out_contig.resize((size_t)N * IMG_FLOATS);
    for (int i = 0; i < N; ++i) {
        std::memcpy(out_contig.data() + (size_t)i * IMG_FLOATS, tensor_ptr(images[i]), IMG_FLOATS * sizeof(float));
    }
}

static bool read_f32_file(const std::string &path, std::vector<float> &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    size_t bytes = (size_t)f.tellg();
    f.seekg(0, std::ios::beg);
    if (bytes == 0 || (bytes % sizeof(float)) != 0) return false;
    out.resize(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(out.data()), (std::streamsize)bytes);
    return (bool)f;
}

static void init_weights_host(std::vector<float> &w, float scale = 0.01f) {
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, scale);
    for (auto &v : w) v = nd(rng);
}

int main(int argc, char **argv) {
    std::string cifar_dir = "./data/cifar-10-batches-bin";
    

    // 1) Load & Pack Data
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data();
    std::vector<float> train_x_contig;
    pack_all_contiguous(ds.train_images, train_x_contig);
    std::cout << "[Data] Loaded " << ds.train_images.size() << " train images.\n";
    

    // 2) Init Engine
    AEParams p;
    p.batch = 128;
    p.lr = 0.005f;
    Phase3Engine eng;
    eng.init(p);

    std::string weights_path = "./output/gpu_basic_model.bin";
    // 3) Weights Initialization (Giữ nguyên logic của bạn)
    if (!weights_path.empty()) {
        std::vector<float> W;
        if (!read_f32_file(weights_path, W)) {
            std::cerr << "[Weights] Cannot read weights file: " << weights_path << "\n";
            return 1;
        }
        const size_t EXPECT = 751875;
        if (W.size() != EXPECT) {
            std::cerr << "[Weights] Size mismatch. Expect " << EXPECT << " floats, got " << W.size() << "\n";
            return 1;
        }
        size_t off = 0;
        auto copyW = [&](float *dst, size_t n) {
            CUDA_CHECK(cudaMemcpy(dst, W.data() + off, n * sizeof(float), cudaMemcpyHostToDevice));
            off += n;
        };
        copyW(eng.w.w1, 256 * 3 * 9); copyW(eng.w.b1, 256);
        copyW(eng.w.w2, 128 * 256 * 9); copyW(eng.w.b2, 128);
        copyW(eng.w.w3, 128 * 128 * 9); copyW(eng.w.b3, 128);
        copyW(eng.w.w4, 256 * 128 * 9); copyW(eng.w.b4, 256);
        copyW(eng.w.w5, 3 * 256 * 9); copyW(eng.w.b5, 3);
    } else {
        std::vector<float> hw1((size_t)256 * 3 * 9), hb1(256, 0.f);
        std::vector<float> hw2((size_t)128 * 256 * 9), hb2(128, 0.f);
        std::vector<float> hw3((size_t)128 * 128 * 9), hb3(128, 0.f);
        std::vector<float> hw4((size_t)256 * 128 * 9), hb4(256, 0.f);
        std::vector<float> hw5((size_t)3 * 256 * 9), hb5(3, 0.f);

        init_weights_host(hw1); init_weights_host(hw2); init_weights_host(hw3);
        init_weights_host(hw4); init_weights_host(hw5);

        CUDA_CHECK(cudaMemcpy(eng.w.w1, hw1.data(), hw1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.b1, hb1.data(), hb1.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.w2, hw2.data(), hw2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.b2, hb2.data(), hb2.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.w3, hw3.data(), hw3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.b3, hb3.data(), hb3.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.w4, hw4.data(), hw4.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.b4, hb4.data(), hb4.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.w5, hw5.data(), hw5.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(eng.w.b5, hb5.data(), hb5.size() * sizeof(float), cudaMemcpyHostToDevice));
    }

    // 4) Training with GPU Timers
    cudaEvent_t start_ev, stop_ev, total_start, total_stop;
    CUDA_CHECK(cudaEventCreate(&start_ev)); CUDA_CHECK(cudaEventCreate(&stop_ev));
    CUDA_CHECK(cudaEventCreate(&total_start)); CUDA_CHECK(cudaEventCreate(&total_stop));

    const int max_epochs = 20;
    const int min_epochs = 3;
    const int patience = 2;
    const float rel_tol = 0.005f;
    const int num_batches = (int)ds.train_images.size() / p.batch;
    printf("[Config] epochs=%d, batch_size=%d, lr=%.6f\n", max_epochs, p.batch, p.lr);
    print_gpu_memory_usage();
    std::cout << "-------------------------------------------------" << std::endl;

    float best_loss = 1e30f;
    float final_loss = 0.0f;
    int no_improve = 0;

    std::cout << "[Train] Starting training..." << std::endl;
    CUDA_CHECK(cudaEventRecord(total_start));

    for (int ep = 0; ep < max_epochs; ++ep) {
        eng.reset_epoch_loss();
        CUDA_CHECK(cudaEventRecord(start_ev, eng.sCompute[0]));

        for (int step = 0; step < num_batches; ++step) {
            const float *batch_ptr = train_x_contig.data() + (size_t)step * p.batch * IMG_FLOATS;
            eng.train_step_async(batch_ptr, step & 1);

            if ((step + 1) % 50 == 0 || step == num_batches - 1) {
                float cur_l = eng.get_epoch_loss_avg_sync(step + 1);
                printf("\rEp %02d/%d | Step %03d/%d | Loss: %.5f", ep + 1, max_epochs, step + 1, num_batches, cur_l);
                fflush(stdout);
            }
        }

        CUDA_CHECK(cudaEventRecord(stop_ev, eng.sCompute[0]));
        CUDA_CHECK(cudaEventSynchronize(stop_ev));
        float ms = 0; CUDA_CHECK(cudaEventElapsedTime(&ms, start_ev, stop_ev));

        final_loss = eng.get_epoch_loss_avg_sync(num_batches);
        printf(" | Epoch Time: %.3fs\n", ms / 1000.0f);

        if (ep >= min_epochs) {
            float rel_improve = (best_loss - final_loss) / (best_loss + 1e-12f);
            if (final_loss < best_loss) best_loss = final_loss;
            if (rel_improve < rel_tol) no_improve++; else no_improve = 0;
            if (no_improve >= patience) {
                std::cout << "[Train] Early stopping triggered. No improvement for " << patience << " epochs.\n";
                break;
            }
        } else {
            if (final_loss < best_loss) best_loss = final_loss;
        }
    }

    CUDA_CHECK(cudaEventRecord(total_stop));
    CUDA_CHECK(cudaEventSynchronize(total_stop));
    float total_ms = 0; CUDA_CHECK(cudaEventElapsedTime(&total_ms, total_start, total_stop));

    std::cout << "\n-------------------------------------------" << std::endl;
    std::cout << "FINAL TRAINING SUMMARY" << std::endl;
    std::cout << "Total Training Time: " << total_ms / 1000.0f << " seconds" << std::endl;
    std::cout << "Final Reconstruction Loss: " << std::fixed << std::setprecision(6) << final_loss << std::endl;
    print_gpu_memory_usage();
    std::cout << "-------------------------------------------" << std::endl;

    // 5) Save final weights
    std::cout << "[Save] Saving weights..." << std::endl;
    eng.save_to_file("output/gpu_opt_model.bin", eng.sCompute[0]);

    CUDA_CHECK(cudaEventDestroy(start_ev)); CUDA_CHECK(cudaEventDestroy(stop_ev));
    CUDA_CHECK(cudaEventDestroy(total_start)); CUDA_CHECK(cudaEventDestroy(total_stop));
    eng.shutdown();
    
    return 0;
}