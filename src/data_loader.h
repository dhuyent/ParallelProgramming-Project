#include "utils.h"
#include <vector>

class CIFAR10Dataset {
public:
    vector<vector<float>> train_images; // 50,000 images
    vector<int> train_labels;
    vector<vector<float>> test_images;  // 10,000 images
    vector<int> test_labels;

    CIFAR10Dataset(const string& data_dir);
    void load_data();        // Đọc binary files
    void shuffle_train_data(); // Trộn dữ liệu 

private:
    string data_dir;
    void read_batch(const string& filename, vector<vector<float>>& images, vector<int>& labels);
};