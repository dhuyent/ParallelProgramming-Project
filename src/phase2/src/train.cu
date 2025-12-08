// train.cu
// Compile: nvcc -O2 -arch=sm_70 train.cu gpu_autoencoder.cu kernels.cu forward.cu backward.cu data_loader.cpp -o train_ae

#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm> 
#include <random>    
#include <limits>   

#include "gpu_autoencoder.cuh"
#include "forward.cuh"
#include "backward.cuh"
#include "data_loader.h" 

int main(int argc, char** argv) {
    std::string cifar_dir = "./cifar-10-batches-bin";
    int epochs = 20;           
    int batch_size = 64;       
    float learning_rate = 0.001f;     

    printf("[Config] cifar_dir='%s' epochs=%d batch_size=%d lr_init=%.6f\n",
        cifar_dir.c_str(), epochs, batch_size, learning_rate);

    // 1. Load Data
    CIFAR10Dataset ds(cifar_dir);
    ds.load_data();
    if (ds.train_images.empty()) return 1;

    // 3. Setup Model
    GPUAutoencoder model;
    model.init_random_weights(); 

    size_t s_input = (size_t)3 * 32 * 32 * sizeof(float);
    float* d_target = nullptr;
    CHECK(cudaMalloc((void**)&d_target, s_input));

    // 4. Shuffle Indices Setup
    int n_samples = (int)ds.train_images.size();
    std::vector<int> indices(n_samples);
    for(int i=0; i<n_samples; ++i) indices[i] = i;
    std::mt19937 rng(1234);

    int num_batches = (n_samples + batch_size - 1) / batch_size;
    
    // 5. GPU TIMERS SETUP
    cudaEvent_t epoch_start, epoch_end;
    CHECK(cudaEventCreate(&epoch_start));
    CHECK(cudaEventCreate(&epoch_end));

    // --- MỚI: Timer cho tổng thời gian huấn luyện ---
    cudaEvent_t total_start, total_end;
    CHECK(cudaEventCreate(&total_start));
    CHECK(cudaEventCreate(&total_end));

    printf("[Train] Starting loop. Total steps: %d per epoch.\n\n", num_batches);

    // BẮT ĐẦU TÍNH GIỜ TỔNG
    CHECK(cudaEventRecord(total_start, 0));

    // ================= TRAINING LOOP =================
    for (int ep = 0; ep < epochs; ++ep) {
        
        // Bắt đầu tính giờ Epoch
        CHECK(cudaEventRecord(epoch_start, 0));

        std::shuffle(indices.begin(), indices.end(), rng);
        double epoch_loss = 0.0;
        
        for (int step = 0; step < num_batches; ++step) {
            int start_idx = step * batch_size;
            int end_idx = std::min(start_idx + batch_size, n_samples);
            int cur_bs = end_idx - start_idx;
            if (cur_bs <= 0) break;

            // Zero gradients
            CHECK(cudaMemset(model.d_g_w_conv1, 0, model.h_w_conv1.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_b_conv1, 0, model.h_b_conv1.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_w_conv2, 0, model.h_w_conv2.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_b_conv2, 0, model.h_b_conv2.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_w_dec1,  0, model.h_w_dec1.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_b_dec1,  0, model.h_b_dec1.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_w_dec2,  0, model.h_w_dec2.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_b_dec2,  0, model.h_b_dec2.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_w_dec3,  0, model.h_w_dec3.size()*sizeof(float)));
            CHECK(cudaMemset(model.d_g_b_dec3,  0, model.h_b_dec3.size()*sizeof(float)));

            double batch_loss = 0.0;
            
            for (int b = 0; b < cur_bs; ++b) {
                int data_idx = indices[start_idx + b];
                float* img_ptr = ds.train_images[data_idx].data();
                CHECK(cudaMemcpy(model.d_input, img_ptr, s_input, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(d_target, img_ptr, s_input, cudaMemcpyHostToDevice));
                float sample_loss = forward_single_device(model, model.d_input, d_target);
                batch_loss += sample_loss;
                backward_single_device(model);
            }

            sgd_update_on_device(model, cur_bs, learning_rate);

            epoch_loss += batch_loss;

            // LOGGING
            if ((step+1) % 100 == 0 || step == num_batches-1) {
                printf("Epoch %02d/%d | Step %03d/%d | Batch Size: %d | Loss: %.6f | LR: %.6f\n",
                       ep+1, epochs, 
                       step+1, num_batches, 
                       cur_bs, 
                       batch_loss/cur_bs, 
                       learning_rate);
                fflush(stdout);
            }
        }

        // Dừng tính giờ Epoch
        CHECK(cudaEventRecord(epoch_end, 0));
        CHECK(cudaEventSynchronize(epoch_end));
        float epoch_ms = 0.0f;
        CHECK(cudaEventElapsedTime(&epoch_ms, epoch_start, epoch_end));

        double avg_loss = epoch_loss / n_samples;
        
        printf("=== Epoch %02d Done. Avg Loss: %.6f | Epoch Time: %.2f sec ===\n", 
               ep+1, avg_loss, epoch_ms / 1000.0f);
        printf("------------------------------------------------------------\n"); 
    } 

    // DỪNG TÍNH GIỜ TỔNG
    CHECK(cudaEventRecord(total_end, 0));
    CHECK(cudaEventSynchronize(total_end));
    float total_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&total_ms, total_start, total_end));

    // Tính toán hiển thị phút/giây
    float total_seconds = total_ms / 1000.0f;
    int minutes = (int)total_seconds / 60;
    float seconds = total_seconds - (minutes * 60);

    printf("\n[Done] Training finished.\n");
    printf("[Time] Total Training Time: %.2f seconds (%d min %.2f sec)\n", 
            total_seconds, minutes, seconds);

    CHECK(cudaFree(d_target));
    CHECK(cudaEventDestroy(epoch_start)); CHECK(cudaEventDestroy(epoch_end));
    CHECK(cudaEventDestroy(total_start)); CHECK(cudaEventDestroy(total_end)); // Cleanup

    model.save_weights("ae_final.bin");
    
    return 0;
}