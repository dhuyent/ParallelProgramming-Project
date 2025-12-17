#include "../data_loader.h"      
#include "autoencoder.h" 
#include <iostream>
#include <fstream> 
#include <vector>
#include <cstdlib>
#include <chrono>
#include <iomanip>

using namespace std;

// Hàm trích xuất đặc trưng
void extract_and_save_features(Autoencoder& model, CIFAR10Dataset& dataset, const string& prefix) {
    string train_file = "output/" + prefix + "_train_features.bin";
    string test_file  = "output/" + prefix + "_test_features.bin";
    
    auto save_set = [&](const vector<Tensor>& images, const vector<int>& labels, const string& filename) {
        ofstream out(filename, ios::binary);
        if (!out.is_open()) return;

        int num_samples = static_cast<int>(images.size());
        int feature_dim = 8 * 8 * 128; 

        out.write((char*)&num_samples, sizeof(int));
        out.write((char*)&feature_dim, sizeof(int));

        cout << "Extracting features to " << filename << " (" << num_samples << " samples)..." << endl;

        for (int i = 0; i < num_samples; ++i) {
            const Tensor& input = images[i];
            
            vector<float> features = model.extract_features(input);
            out.write((char*)features.data(), features.size() * sizeof(float));
            
            int label = labels[i];
            out.write((char*)&label, sizeof(int));
        }
        out.close();
    };

    save_set(dataset.train_images, dataset.train_labels, train_file);
    save_set(dataset.test_images, dataset.test_labels, test_file);
}

// Hàm lưu ảnh tái tạo
void save_reconstruction_samples(Autoencoder& model, CIFAR10Dataset& dataset, const string& filename) {
    ofstream out(filename, ios::binary);
    if (!out.is_open()) return;

    // Lấy 10 ảnh test đầu tiên
    int num_samples = 10;
    out.write((char*)&num_samples, sizeof(int));

    for (int i = 0; i < num_samples; ++i) {
        const Tensor& input = dataset.test_images[i];

        // Forward để lấy ảnh tái tạo
        Tensor output = model.forward(input);

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

        // Shuffle data đầu mỗi epoch 
        dataset.shuffle_train_data();

        // Tính số batch
        int num_batches = (dataset.train_images.size() + BATCH_SIZE - 1) / BATCH_SIZE;

        for (int b = 0; b < num_batches; ++b) {
            Batch batch = dataset.get_batch(b * BATCH_SIZE, BATCH_SIZE);
            float batch_loss = 0.0f;

            // Xử lý từng ảnh trong batch
            for (const Tensor& img : batch.inputs) {
                // Forward
                Tensor output = model.forward(img);

                // Loss
                float loss = mse_loss(output, img);
                batch_loss += loss;
                
                // Backward
                Tensor grad = mse_loss_grad(output, img);
                model.backward(grad);
            }
            
            // Cập nhật weights sau mỗi batch
            float effective_lr = LR / batch.inputs.size(); 
            model.update(effective_lr);

            // Tính loss trung bình cho batch
            float avg_batch_loss = batch_loss / batch.inputs.size();
            total_loss += avg_batch_loss;

            // Tracking
            if ((b + 1) % 10 == 0 || b == 0) {
                cout << "\rEpoch " << epoch + 1 << "/" << EPOCHS 
                     << " [Batch " << b + 1 << "/" << num_batches << "] "
                     << "Loss: " << fixed << setprecision(5) << avg_batch_loss << flush;
            }
        }
        epoch_timer.Stop();
        
        long mem_usage = get_memory_usage() / 1024;
        
        cout << "\nEpoch " << epoch + 1 << " Done. "
             << "Avg Loss: " << (total_loss / num_batches) 
             << " | Time: " << epoch_timer.ElapsedSeconds() << "s" 
             << " | Mem: " << mem_usage << " MB" << endl;
    }

    total_timer.Stop();
    cout << "Total Time: " << total_timer.ElapsedSeconds() << "s\n";

    // 5. Save & Extract
    model.save_weights("output/cpu_model.bin");
    save_reconstruction_samples(model, dataset, "output/cpu_reconstruction.bin");
    extract_and_save_features(model, dataset, "cpu");
    
    return 0;
}