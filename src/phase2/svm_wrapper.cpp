#include "svm_wrapper.h"
#include <iostream>
#include <cmath>

SVMClassifier::SVMClassifier() : model(nullptr) {
    // 1. Cấu hình mặc định theo yêu cầu của bạn
    param.svm_type = C_SVC;     // Classification
    param.kernel_type = RBF;    // Kernel RBF
    param.degree = 3;
    param.gamma = 0;            // Sẽ tự tính nếu người dùng không set (1/num_features)
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 10.0;             // Yêu cầu: C = 10
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
}

SVMClassifier::~SVMClassifier() {
    if(model) svm_free_and_destroy_model(&model);
    // data_pool và x_space tự giải phóng nhờ std::vector
}

void SVMClassifier::set_parameters(double C, double gamma) {
    param.C = C;
    param.gamma = gamma;
}

void SVMClassifier::train(const std::vector<std::vector<float>>& features, const std::vector<int>& labels) {
    if (features.empty()) return;

    int num_samples = (int)features.size();
    int num_dims = (int)features[0].size();

    // Setup Gamma = auto (1 / num_features) nếu chưa set
    if (param.gamma == 0) {
        param.gamma = 1.0 / num_dims;
        std::cout << "[SVM] Gamma set to auto: " << param.gamma << "\n";
    }

    // 2. Chuyển đổi dữ liệu sang định dạng LIBSVM (Sparse format)
    // LIBSVM cần mảng các svm_node. Mỗi vector kết thúc bằng index = -1
    
    prob.l = num_samples;
    prob.y = new double[num_samples];
    prob.x = new svm_node*[num_samples];

    // Cấp phát vùng nhớ cho toàn bộ node
    // Mỗi sample cần (num_dims + 1) node (cộng 1 cho node kết thúc -1)
    data_pool.resize(num_samples * (num_dims + 1));
    
    for (int i = 0; i < num_samples; ++i) {
        prob.y[i] = (double)labels[i];
        prob.x[i] = &data_pool[i * (num_dims + 1)];

        for (int j = 0; j < num_dims; ++j) {
            data_pool[i * (num_dims + 1) + j].index = j + 1; // Index bắt đầu từ 1
            data_pool[i * (num_dims + 1) + j].value = (double)features[i][j];
        }
        // Node kết thúc
        data_pool[i * (num_dims + 1) + num_dims].index = -1;
        data_pool[i * (num_dims + 1) + num_dims].value = 0;
    }

    // 3. Huấn luyện
    std::cout << "[SVM] Training started with C=" << param.C << ", Gamma=" << param.gamma << "...\n";
    const char* error_msg = svm_check_parameter(&prob, &param);
    if (error_msg) {
        std::cerr << "[SVM Error] Parameters: " << error_msg << "\n";
        return;
    }

    model = svm_train(&prob, &param);
    std::cout << "[SVM] Training finished.\n";

    // Cleanup pointer tạm (data_pool vẫn giữ dữ liệu thật)
    delete[] prob.y;
    delete[] prob.x;
}

double SVMClassifier::predict(const std::vector<float>& feature) {
    if (!model) return -1;
    
    // Convert single vector to svm_node array
    std::vector<svm_node> nodes(feature.size() + 1);
    for (size_t i = 0; i < feature.size(); ++i) {
        nodes[i].index = (int)i + 1;
        nodes[i].value = (double)feature[i];
    }
    nodes[feature.size()].index = -1;

    return svm_predict(model, nodes.data());
}

void SVMClassifier::save_model(const std::string& filename) {
    if (svm_save_model(filename.c_str(), model)) {
        std::cerr << "[SVM Error] Could not save model to " << filename << "\n";
    } else {
        std::cout << "[SVM] Model saved to " << filename << "\n";
    }
}

bool SVMClassifier::load_model(const std::string& filename) {
    if(model) svm_free_and_destroy_model(&model);
    model = svm_load_model(filename.c_str());
    if (!model) {
        std::cerr << "[SVM Error] Could not load model " << filename << "\n";
        return false;
    }
    return true;
}

double SVMClassifier::evaluate(const std::vector<std::vector<float>>& features, const std::vector<int>& labels) {
    int correct = 0;
    for (size_t i = 0; i < features.size(); ++i) {
        double pred = predict(features[i]);
        if ((int)pred == labels[i]) correct++;
    }
    return (double)correct / features.size() * 100.0;
}