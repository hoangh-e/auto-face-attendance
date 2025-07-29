# 🎥 AI Attendance System - Local Camera Demo (Fixed Version)

Phiên bản đã sửa lỗi của hệ thống điểm danh AI với camera local, khắc phục tất cả các lỗi import và dependency.

## 🚀 Cách sử dụng

### 1. Kiểm tra hệ thống trước khi chạy

```bash
python test_demo.py
```

Script này sẽ kiểm tra:
- ✅ Tất cả các thư viện cần thiết đã được cài đặt
- ✅ Thư mục làm việc được tạo đúng
- ✅ Camera có hoạt động không
- ✅ AI model có load được không

### 2. Chạy demo chính

```bash
python local_camera_demo_fixed.py
```

## 📁 Cấu trúc thư mục cần thiết

```
notebooks/
├── employees/                 # Thư mục chứa ảnh nhân viên
│   ├── John_Doe/             # Thư mục cho từng nhân viên
│   │   ├── photo1.jpg
│   │   ├── photo2.jpg
│   │   └── ...
│   ├── Jane_Smith/
│   │   ├── portrait1.png
│   │   └── portrait2.png
│   └── ...
├── data/                     # Database và dữ liệu
└── snapshots/               # Ảnh chụp từ camera
```

## 🔧 Các lỗi đã được sửa

### 1. **Lỗi import cv2**
```python
# Trước (lỗi):
import cv2

# Sau (đã sửa):
try:
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print("❌ OpenCV import failed. Installing...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'opencv-python', '--upgrade'])
    import cv2
```

### 2. **Lỗi biến không được định nghĩa**
```python
# Đã thêm định nghĩa các biến toàn cục:
base_dir = Path.cwd()
employees_dir = base_dir / 'employees'
data_dir = base_dir / 'data'
snapshots_dir = base_dir / 'snapshots'
```

### 3. **Lỗi dependencies**
- Tự động cài đặt các thư viện thiếu
- Error handling cho việc load AI models
- Fallback về CPU mode nếu GPU không có

### 4. **Lỗi camera**
- Improved camera detection
- Better error messages
- Thread-safe camera operations

## 📊 Các tính năng chính

### 1. **Quản lý nhân viên tự động**
```python
employee_manager = LocalEmployeeManager(ai_system, db)
employee_manager.scan_employee_folders()  # Tự động scan và đăng ký
```

### 2. **Phát hiện camera**
```python
camera_manager = LocalCameraManager()
cameras = camera_manager.detect_cameras()  # Tìm camera có sẵn
camera_manager.initialize_camera(0)        # Khởi tạo camera
```

### 3. **Database local**
```python
db = LocalDatabase()
stats = db.get_statistics()  # Thống kê
```

### 4. **AI Face Recognition**
```python
ai_system = LocalAISystem()
faces = ai_system.detect_and_recognize(image)  # Nhận diện khuôn mặt
```

## 🛠️ Troubleshooting

### Lỗi thường gặp:

#### 1. **"No module named 'cv2'"**
```bash
pip install opencv-python
```

#### 2. **"No module named 'insightface'"**
```bash
pip install insightface
```

#### 3. **"Camera not detected"**
- Kiểm tra camera có được kết nối không
- Kiểm tra quyền truy cập camera
- Thử camera index khác (0, 1, 2...)

#### 4. **AI model load failed**
```bash
# Cài đặt PyTorch cho CPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 5. **Insufficient memory**
- Giảm det_size trong AI model
- Sử dụng model nhẹ hơn (buffalo_s thay vì buffalo_l)

## 📋 System Requirements

### Minimum:
- Python 3.8+
- 4GB RAM
- Webcam (USB hoặc built-in)
- CPU: Intel i5 hoặc tương đương

### Recommended:
- Python 3.9+
- 8GB RAM
- GPU với CUDA support
- CPU: Intel i7 hoặc AMD Ryzen 7+

## 🚀 Hướng dẫn sử dụng từng bước

### Bước 1: Chuẩn bị ảnh nhân viên
1. Tạo thư mục cho mỗi nhân viên trong `employees/`
2. Đặt tên thư mục theo format: `Ten_Nhan_Vien` (VD: `Nguyen_Van_A`)
3. Thêm 2-5 ảnh rõ mặt cho mỗi nhân viên
4. Format ảnh: JPG, PNG, JPEG

### Bước 2: Chạy test
```bash
python test_demo.py
```

### Bước 3: Chạy demo
```bash
python local_camera_demo_fixed.py
```

### Bước 4: Xem kết quả
- Nhân viên sẽ được tự động đăng ký từ thư mục
- Camera sẽ được khởi tạo
- Database sẽ lưu trữ thông tin điểm danh

## 📈 Performance Tips

1. **Tối ưu AI model:**
   ```python
   # Sử dụng det_size nhỏ hơn cho tốc độ nhanh hơn
   app.prepare(ctx_id=-1, det_size=(320, 320), det_thresh=0.5)
   ```

2. **Tối ưu camera:**
   ```python
   # Giảm resolution để tăng FPS
   camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

3. **Tối ưu database:**
   - Thường xuyên vacuum database
   - Index các cột quan trọng

## 🔗 Links hữu ích

- [InsightFace Documentation](https://github.com/deepinsight/insightface)
- [OpenCV Documentation](https://docs.opencv.org/)
- [SQLite Documentation](https://sqlite.org/docs.html)

## 💡 Tips & Tricks

1. **Chất lượng ảnh tốt:**
   - Ánh sáng đều
   - Khuôn mặt rõ ràng
   - Không bị che khuất
   - Góc chụp từ phía trước

2. **Performance:**
   - Đóng các ứng dụng khác khi chạy
   - Sử dụng SSD thay vì HDD
   - Đảm bảo đủ RAM

3. **Backup:**
   - Backup database thường xuyên
   - Lưu trữ ảnh nhân viên ở nhiều nơi

## 🆘 Support

Nếu gặp lỗi, vui lòng:
1. Chạy `python test_demo.py` để kiểm tra
2. Xem log chi tiết
3. Kiểm tra system requirements
4. Cài đặt lại dependencies nếu cần

## ✅ Checklist trước khi sử dụng

- [ ] Python 3.8+ đã cài đặt
- [ ] Tất cả dependencies đã cài đặt
- [ ] Camera hoạt động bình thường
- [ ] Thư mục `employees/` đã có ảnh nhân viên
- [ ] Test script chạy thành công
- [ ] Đủ dung lượng ổ cứng (ít nhất 2GB)

🎉 **Chúc bạn sử dụng thành công!** 