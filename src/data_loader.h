#pragma once
#include "utils.h"
#include <vector>

struct Batch {
    vector<Tensor> inputs;
    vector<int> labels;
};

class CIFAR10Dataset {
public:
    vector<Tensor> train_images; // 50,000 images
    vector<int> train_labels;
    vector<Tensor> test_images;  // 10,000 images
    vector<int> test_labels;

    CIFAR10Dataset(const string& data_dir);
    void load_data();        // Đọc binary files
    void shuffle_train_data(); // Trộn dữ liệu 

    Batch get_batch(size_t start_idx, size_t batch_size); // Lấy batch dữ liệu

private:
    string data_dir;
    void read_batch(const string& filename, vector<Tensor>& images, vector<int>& labels);
};