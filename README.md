# Đồ án Cuối kỳ - Lập trình song song

**Đề tài: Autoencoder & SVM for CIFAR-10 Classification**

**Lớp:** 22KHMT

**Sinh viên thực hiện:** Nhóm 6

| STT | MSSV | Họ và tên |
| --- | --- | --- |
| 1 | 22127008 | Đặng Châu Anh |
| 2 | 22127170 | Trần Dịu Huyền |
| 3 | 22127359 | Chu Thúy Quỳnh |

## Kế hoạch thực hiện & phân công công việc

Chi tiết về kế hoạch và phân chia công việc được lưu trong file Excel `Team-Plan-and-Work-Distribution.xlsx`.

## Tổ chức mã nguồn

Cấu trúc thư mục dự án bao gồm:

```text
├── data/                       # Chứa dữ liệu CIFAR-10 (bin files)
├── output/                     # Chứa file weights (.bin) và logs sau khi train
│   ├── cpu_model.bin
│   ├── gpu_basic_model.bin
|   ├── gpu_opt_model.bin       # Trọng số huấn luyện cuối cùng
│   └── ...
├── src/                        # Mã nguồn chính
│   ├── phase1/                 # Phase 1: CPU Implementation (Baseline)
│   ├── phase2/                 # Phase 2: GPU Basic Implementation
│   ├── phase3/                 # Phase 3: GPU Optimization (Shared Memory)
│   ├── data_loader.cpp         # Xử lý đọc dữ liệu CIFAR-10
│   └── utils.h                 # Các hàm tiện ích chung
├── Report.ipynb                # Chạy chương trình và báo cáo kết quả
└── src.zip                     # File zip của src/ để chạy trên Report
```

## Hướng dẫn chạy chương trình

### 1. Yêu cầu phần cứng & môi trường
- **Hardware (GPU):**
    - GPU NVIDIA có hỗ trợ CUDA (Compute Capability >= 7.5)
    - VRAM: Tối thiểu 4GB (Khuyến nghị 8GB+ để chạy batch size lớn)
    - RAM hệ thống: Tối thiểu 8GB

- **Software & Libraries:**
    - OS: Linux (Ubuntu 20.04/22.04) hoặc Windows (với WSL2)
    - Compiler: `g++` (Standard C++11 trở lên)
    - CUDA Toolkit: Phiên bản 11.0 trở lên
    - Python 3.x

### 2. Cài đặt 

**Bước 1: Giải nén mã nguồn**

```bash
unzip src.zip
```

**Bước 2: Chuẩn bị dữ liệu**
Tạo thư mục `data` và tải bộ dữ liệu CIFAR-10 vào thư mục này hoặc chạy cell trên `Report.ipynb`. Đảm bảo cấu trúc như sau:

```text
data/
  cifar-10-batches-bin/
    data_batch_1.bin
    ...
    test_batch.bin
```

**Bước 3: Tạo thư mục output**

```bash
mkdir -p output
```

### 3. Biên dịch và thực thi

Project được chia thành 3 phases. Dưới đây là lệnh biên dịch cho từng phase từ thư mục gốc của project.

#### Phase 1: CPU Baseline
- **Biên dịch:**
```bash
!g++ -O3 -I./src src/phase1/*.cpp src/*.cpp -o run_phase1
```

- **Thực thi:**
```bash
./run_phase1
```

#### Phase 2: Naive GPU Implementation
- **Biên dịch:**
```bash
# Lưu ý: Thay arch=sm_XX bằng kiến trúc GPU cụ thể (VD: sm_75 cho T4, sm_80 cho A100)
!nvcc -O2 -arch=sm_XX src/phase2/train.cu src/phase2/gpu_autoencoder.cu src/phase2/kernels.cu src/phase2/data_loader.cpp -o run_phase2
```

- **Thực thi:**
```bash
./run_phase2
```

#### Phase 3: Optimized GPU Implementation (Shared Memory)
- **Biên dịch:**
```bash
!nvcc -O2 -arch=sm_XX src/phase3/train.cpp src/data_loader.cpp src/phase3/gpu_opt.cu -o run_phase3
```

- **Thực thi:**
```bash
./run_phase3 ./data/cifar-10-batches-bin
```

### 4. Kết quả mong đợi
Sau khi chạy các lệnh trên, chương trình sẽ in ra màn hình thông tin log quá trình training và lưu các file vào thư mục `output/`.

**Console output:**

```text
[Data] Loaded 50000 train images.
[Config] Epochs: 20, Batch Size: 64, LR: 0.001
-------------------------------------------------
[Train] Starting training...
Ep 01/20 | Loss: 0.0452 | Time: 105.4s
...
Ep 20/20 | Loss: 0.0143 | Time: 105.1s
[Summary] Total Time: 2120.5s
```

**Output files:**
- `output/*_model.bin`: Trọng số mô hình sau khi train
- `output/*_reconstruction.bin`: Ảnh tái tạo để kiểm tra trực quan
- `output/*_features.bin`: Vector đặc trưng trích xuất (dùng cho SVM)

---

## Video trình bày