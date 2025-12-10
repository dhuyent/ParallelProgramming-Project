// train.cu
// Compile: nvcc -O2 -arch=sm_70 train.cu gpu_autoencoder.cu kernels.cu data_loader.cpp -o train_ae

#include <cstdio>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm> 
#include <random>    
#include <limits>   

#include "gpu_autoencoder.cuh"
#include "data_loader.h" 

// Helper: Kiểm tra bộ nhớ GPU
void print_gpu_memory_usage() {
    size_t free_byte, total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

    if (cudaSuccess != cuda_status){
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        return;
    }

    double free_db = (double)free_byte;
    double total_db = (double)total_byte;
    double used_db = total_db - free_db;

    printf("[GPU Mem] Used: %.2f MB | Free: %.2f MB | Total: %.2f MB\n", 
           used_db / 1024.0 / 1024.0, 
           free_db / 1024.0 / 1024.0, 
           total_db / 1024.0 / 1024.0);
}

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

    // 2. Setup Model
    GPUAutoencoder model;
    model.init_random_weights(); 

    // Kiểm tra bộ nhớ sau khi init model
    printf("--- Memory after Model Init ---\n");
    print_gpu_memory_usage();

    // Buffer tạm cho target
    size_t s_input = (size_t)3 * 32 * 32 * sizeof(float);
    float* d_target = nullptr;
    CHECK(cudaMalloc((void**)&d_target, s_input));

    int n_samples = (int)ds.train_images.size();
    int num_batches = (n_samples + batch_size - 1) / batch_size;
    
    // 3. Timers Setup
    cudaEvent_t epoch_start, epoch_end;
    CHECK(cudaEventCreate(&epoch_start));
    CHECK(cudaEventCreate(&epoch_end));

    cudaEvent_t total_start, total_end;
    CHECK(cudaEventCreate(&total_start));
    CHECK(cudaEventCreate(&total_end));

    printf("[Train] Starting loop. Total steps: %d per epoch.\n\n", num_batches);

    // BẮT ĐẦU TÍNH GIỜ TỔNG
    CHECK(cudaEventRecord(total_start, 0));

    // ================= TRAINING LOOP =================
    for (int ep = 0; ep < epochs; ++ep) {
        
        CHECK(cudaEventRecord(epoch_start, 0));

        // Shuffle Data
        ds.shuffle_train_data();
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
            
            // --- VÒNG LẶP BATCH ---
            for (int b = 0; b < cur_bs; ++b) {
                int current_index = start_idx + b;
                float* img_ptr = ds.train_images[current_index].data();
                
                CHECK(cudaMemcpy(model.d_input, img_ptr, s_input, cudaMemcpyHostToDevice));
                CHECK(cudaMemcpy(d_target, img_ptr, s_input, cudaMemcpyHostToDevice));
                
                // --- CẬP NHẬT GỌI HÀM FORWARD (Dùng method của class) ---
                float sample_loss = model.forward(model.d_input, d_target);
                
                batch_loss += sample_loss;
                
                // --- CẬP NHẬT GỌI HÀM BACKWARD (Dùng method của class) ---
                model.backward();
            }

            // --- CẬP NHẬT GỌI HÀM UPDATE (Dùng method của class) ---
            model.update_weights(cur_bs, learning_rate);

            epoch_loss += batch_loss;

            // LOGGING
            if ((step+1) % 100 == 0 || step == num_batches-1) {
                size_t free, total;
                cudaMemGetInfo(&free, &total);
                double used_mb = (double)(total - free) / 1024.0 / 1024.0;
                
                printf("Epoch %02d/%d | Step %03d/%d | BS: %d | Loss: %.6f | LR: %.6f | Mem: %.0f MB\n",
                       ep+1, epochs, 
                       step+1, num_batches, 
                       cur_bs, 
                       batch_loss/cur_bs, 
                       learning_rate,
                       used_mb);
                fflush(stdout);
            }
        }

        CHECK(cudaEventRecord(epoch_end, 0));
        CHECK(cudaEventSynchronize(epoch_end));
        float epoch_ms = 0.0f;
        CHECK(cudaEventElapsedTime(&epoch_ms, epoch_start, epoch_end));

        double avg_loss = epoch_loss / n_samples;
        
        printf("=== Epoch %02d Done. Avg Loss: %.6f | Epoch Time: %.2f sec ===\n", 
               ep+1, avg_loss, epoch_ms / 1000.0f);
        printf("------------------------------------------------------------\n"); 
    } 

    CHECK(cudaEventRecord(total_end, 0));
    CHECK(cudaEventSynchronize(total_end));
    float total_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&total_ms, total_start, total_end));

    float total_seconds = total_ms / 1000.0f;
    int minutes = (int)total_seconds / 60;
    float seconds = total_seconds - (minutes * 60);

    printf("\n[Done] Training finished.\n");
    printf("[Time] Total Training Time: %.2f seconds (%d min %.2f sec)\n", 
            total_seconds, minutes, seconds);

    print_gpu_memory_usage();

    CHECK(cudaFree(d_target));
    CHECK(cudaEventDestroy(epoch_start)); CHECK(cudaEventDestroy(epoch_end));
    CHECK(cudaEventDestroy(total_start)); CHECK(cudaEventDestroy(total_end));

    model.save_weights("ae_final.bin");
    
    return 0;
}