#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>

CIFAR10Dataset::CIFAR10Dataset(const string& path) : data_dir(path) {}

void CIFAR10Dataset::load_data() {
    for (int i = 1; i <= 5; ++i) {
        read_batch(data_dir + "/data_batch_" + to_string(i) + ".bin", train_images, train_labels);
    }
    read_batch(data_dir + "/test_batch.bin", test_images, test_labels);
    cout << "[Data] Loaded " << train_images.size() << " train, " << test_images.size() << " test images.\n";
}

void CIFAR10Dataset::read_batch(const string& filename, vector<vector<float>>& images, vector<int>& labels) {
    ifstream file(filename, ios::binary);
    if (!file.is_open()) { cerr << "Cannot open " << filename << endl; exit(1); }

    int record_size = 3073; // 1 byte label + 3072 bytes pixel
    vector<uint8_t> buffer(record_size);

    while (file.read(reinterpret_cast<char*>(buffer.data()), record_size)) {
        labels.push_back(buffer[0]);
        vector<float> img(3072);
        for (int i = 0; i < 3072; ++i) img[i] = buffer[i + 1] / 255.0f; // Normalize [0,1] [cite: 240]
        images.push_back(img);
    }
    file.close();
}

void CIFAR10Dataset::shuffle_train_data() {
    random_device rd; mt19937 g(rd());
    vector<size_t> p(train_images.size());
    for(size_t i=0; i<p.size(); i++) p[i] = i;
    shuffle(p.begin(), p.end(), g);
    
    vector<vector<float>> img_new; img_new.reserve(train_images.size());
    vector<int> lbl_new; lbl_new.reserve(train_labels.size());
    
    for(size_t i : p) {
        img_new.push_back(train_images[i]);
        lbl_new.push_back(train_labels[i]);
    }
    train_images = move(img_new);
    train_labels = move(lbl_new);
}