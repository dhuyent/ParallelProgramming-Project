#include "../data_loader.h"      
#include "autoencoder.h" 
#include <iostream>
#include <fstream> 
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;

// Helper: Lưu ảnh gốc và ảnh tái tạo cho báo cáo 
void save_reconstruction_samples(Autoencoder& model, CIFAR10Dataset& dataset, const string& filename) {
    ofstream out(filename, ios::binary);
    if (!out.is_open()) return;

    // Lấy 10 ảnh test đầu tiên
    int num_samples = 10;
    out.write((char*)&num_samples, sizeof(int));

    for (int i = 0; i < num_samples; ++i) {
        Tensor input(3, 32, 32);
        input.data = dataset.test_images[i]; 

        // Forward để lấy ảnh tái tạo
        Tensor output = model.forward(input);

        // Lưu ảnh 
        out.write((char*)input.data.data(), input.data.size() * sizeof(float));
        out.write((char*)output.data.data(), output.data.size() * sizeof(float));
    }
    out.close();
    cout << "[Report] Saved reconstruction samples to " << filename << endl;
}

int main(int argc, char** argv) {
    cout << "--- Phase 1: CPU Baseline ---\n";
    
    // 1. Load Data & Preprocessing 
    CIFAR10Dataset dataset("./data/cifar-10-batches-bin");
    dataset.load_data();

    // 2. Hyperparameters Setup 
    int BATCH_SIZE = 32;
    int EPOCHS = 20;     
    float LR = 0.001f;

    if (argc > 1) EPOCHS = atoi(argv[1]);

    cout << "Training Settings:\n";
    cout << "- Batch Size: " << BATCH_SIZE << "\n";
    cout << "- Epochs: " << EPOCHS << "\n";
    cout << "- Learning Rate: " << LR << "\n";

    // 3. Initialize Model
    Autoencoder model;
    CpuTimer epoch_timer, total_timer;
    
    cout << "\n[Start Training]...\n";
    total_timer.Start();

    // 4. Training Loop 
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        epoch_timer.Start();
        float total_loss = 0.0f;
        int num_batches = dataset.train_images.size() / BATCH_SIZE;

        // Shuffle data đầu mỗi epoch 
        dataset.shuffle_train_data();

        // Loop over batches 
        for (int b = 0; b < num_batches; ++b) {
            float batch_loss = 0.0f;

            // Xử lý từng ảnh trong batch 
            for (int i = 0; i < BATCH_SIZE; ++i) {
                Tensor img(3, 32, 32);
                img.data = dataset.train_images[b * BATCH_SIZE + i];
                
                // Forward Pass 
                Tensor output = model.forward(img);
                
                // Compute Loss 
                float loss = mse_loss(output, img);
                batch_loss += loss;
                
                // Backward Pass 
                Tensor grad = mse_loss_grad(output, img);
                model.backward(grad, LR);
            }
            
            // Average loss của batch
            float avg_batch_loss = batch_loss / BATCH_SIZE;
            total_loss += avg_batch_loss;

            // Track training loss 
            if ((b + 1) % 10 == 0 || b == 0) {
                cout << "\rEpoch " << epoch + 1 << "/" << EPOCHS 
                     << " [Batch " << b + 1 << "/" << num_batches << "] "
                     << "Loss: " << fixed << setprecision(5) << avg_batch_loss << flush;
            }
        }
        
        epoch_timer.Stop();
        
        // Measure time per epoch 
        cout << "\nEpoch " << epoch + 1 << " Completed. "
             << "Avg Loss: " << (total_loss / num_batches) 
             << " | Time: " << epoch_timer.ElapsedSeconds() << "s" << endl;
    }

    total_timer.Stop();
    cout << "Total Time: " << total_timer.ElapsedSeconds() << "s\n";

    // 5. Save Trained Model Weights 
    model.save_weights("cpu_model.bin");

    // 6. Save Sample Reconstructions (Cho phần báo cáo Visual)
    save_reconstruction_samples(model, dataset, "reconstruction_samples.bin");
    
    return 0;
}