#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>

CIFAR10Dataset::CIFAR10Dataset(const string& path) : data_dir(path) {}

void CIFAR10Dataset::load_data() {
    for (int i = 1; i <= 1; ++i) { // Cut down on training data 
        read_batch(data_dir + "/data_batch_" + to_string(i) + ".bin", train_images, train_labels);
    }
    read_batch(data_dir + "/test_batch.bin", test_images, test_labels);
    cout << "[Data] Loaded " << train_images.size() << " train, " << test_images.size() << " test images.\n";
}

void CIFAR10Dataset::read_batch(const string& filename, vector<Tensor>& images, vector<int>& labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) { cerr << "Cannot open " << filename << endl; exit(1); }

    int record_size = 3073; // 1 byte label + 3072 bytes pixel
    vector<uint8_t> buffer(record_size);

    while (file.read(reinterpret_cast<char*>(buffer.data()), record_size)) {
        labels.push_back(buffer[0]);
        Tensor img(3, 32, 32);
        for (int i = 0; i < 3072; ++i) {
            img.data[i] = static_cast<float>(buffer[i + 1]) / 255.0f;
        }
        images.push_back(img);
    }
    file.close();
}

void CIFAR10Dataset::shuffle_train_data() {
    if (train_images.empty()) return;
    static random_device rd;
    static mt19937 g(rd());

    for (size_t i = train_images.size() - 1; i > 0; --i) {
        uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(g);
        swap(train_images[i], train_images[j]);
        swap(train_labels[i], train_labels[j]);
    }
}

Batch CIFAR10Dataset::get_batch(size_t start_idx, size_t batch_size) {
    Batch batch;
    size_t end_idx = min(start_idx + batch_size, train_images.size());
    
    batch.inputs.reserve(end_idx - start_idx);
    batch.labels.reserve(end_idx - start_idx);

    for (size_t i = start_idx; i < end_idx; ++i) {
        batch.inputs.push_back(train_images[i]);
        batch.labels.push_back(train_labels[i]);
    }
    return batch;
}