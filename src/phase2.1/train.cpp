#include "../data_loader.h"
#include "autoencoder.h"
#include <chrono>
#include <iomanip>

int main() {
    // 1. Load Data 
    cout << "--- Phase 1: CPU Baseline ---\n";
    CIFAR10Dataset dataset("./data/cifar-10-batches-bin");
    dataset.load_data();

    // 2. Hyperparameters [cite: 262]
    int BATCH_SIZE = 32;
    int EPOCHS = 5; // Để demo, PDF yêu cầu 20 nhưng CPU rất chậm
    float LR = 0.001f;

    Autoencoder model;
    
    // 3. Training Loop [cite: 263]
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto t_start = chrono::high_resolution_clock::now();
        float total_loss = 0.0f;
        int num_batches = dataset.train_images.size() / BATCH_SIZE;

        dataset.shuffle_train_data();

        // Loop over batches [cite: 264]
        for (int b = 0; b < num_batches; ++b) {
            float batch_loss = 0.0f;

            // Xử lý từng ảnh trong batch (Giả lập batch training trên CPU)
            for (int i = 0; i < BATCH_SIZE; ++i) {
                int idx = b * BATCH_SIZE + i;
                
                // Chuẩn bị input (cần convert vector -> Tensor)
                Tensor img(3, 32, 32);
                img.data = dataset.train_images[idx];

                // Forward -> Loss -> Backward -> Update [cite: 265]
                Tensor output = model.forward(img);
                float loss = mse_loss(output, img);
                
                Tensor grad = mse_loss_grad(output, img);
                model.backward(grad, LR);

                batch_loss += loss;
            }
            
            total_loss += batch_loss / BATCH_SIZE;

            // In tiến độ (mỗi 10 batch)
            if (b % 10 == 0) {
                cout << "\rEpoch " << epoch+1 << "/" << EPOCHS 
                          << " [Batch " << b << "/" << num_batches << "]"
                          << " Loss: " << fixed << setprecision(4) 
                          << (batch_loss / BATCH_SIZE) << flush;
            }
        }

        auto t_end = chrono::high_resolution_clock::now();
        double duration = chrono::duration<double>(t_end - t_start).count();
        
        cout << "\nEpoch " << epoch+1 << " Done. Avg Loss: " << (total_loss/num_batches)
                  << " Time: " << duration << "s\n";
    }

    // 4. Save Model [cite: 268]
    model.save_weights("cpu_model.bin");
    return 0;
}