Xây dựng UI giao diện cho demo, với các tính năng sau:
- Tất cả được xây dựng trong 1 màn hình
- Xây trong /demo, chia folder, file nhỏ ra theo logic, hàm.
- Thanh giao diện được chia thành 4 phần như sau:
    + Phần 1: Thanh Header chứa tên đề tài, tên nhóm, tên thành viên.:
    Tên đề tài: PHÂN LOẠI MỨC ĐỘ HỨNG THÚ HỌC TẬP CỦA SINH VIÊN TRONG LỚP HỌC BẰNG PHÂN TÍCH KHUÔN MẶT
    Tên nhóm: Nhóm 16
    Tên thành viên - MSSV: Huỳnh Ngọc Thạch - 23133072; Huỳnh Hữu Huy - 23133027; Đỗ Kiến Hưng - 23133030; Nguyễn Tân Thành - 23133068
    + Phần 2: Khu vực hiển thị video từ Webcam (BÊN TRÁI). Được chia làm 2 phần
    - Phần 2.1: Khu vực hiển thị video từ Webcam và kết quả dự đoán (Engaged/Not Engaged) của CNN kết hợp SVM (phần này tôi đã xây dựng rồi)
    - Phần 2.2: Khu vực hiển thị video từ Webcam và kết quả dự đoán (Engaged/Not Engaged) của HOG kết hợp SVM (phần này tôi chưa xây dựng, dùng /demo/models/hog_svm_model.pkl)
    + Phần 3: Khu vực hiển thị thông tin thêm (BÊN PHẢI). Hiển thị (FPS, số lượng khuôn mặt, chỉ số đánh giá, ...).

* Lưu ý: 
- Sử dụng thư viện phổ biến của python.
- Xây dựng wireframe giao diện, styling giao diện trước, sau đó xây dựng giao diện theo wireframe và styling.
- Xây dựng giao diện theo logic, hàm, folder nhỏ.
- Hãy đề xuất kỹ thuật nào đó để việc demo trở nên mượt mà. Vì giới hạn đang là 30 fps, thực tế chạy 15-20 fps. Vì vậy tôi nghĩ có thể cần phải cải thiện thêm để demo chạy mượt mà hơn.