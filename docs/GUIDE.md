LUỒNG STEP-BY-STEP XÂY DỰNG APP DEMO (Code Logic)
Bạn cần xây dựng theo luồng sau:
Bước 1: Khởi tạo (Init)
Load file cnn_model.h5 (Dùng để trích xuất đặc trưng).
Load file svm_model.pkl.
Load haarcascade_frontalface_default.xml.
Khởi tạo Webcam (cv2.VideoCapture(0)).
Bước 2: Vòng lặp chính (Main Loop - While True)
Đọc frame từ Webcam (cap.read()).
Resize frame (ví dụ về Width=640).
Chuyển sang ảnh xám (gray_frame) để phục vụ Haar-like.
Face Detection: Dùng Haar để tìm toạ độ (x, y, w, h).
Nếu có mặt:
Lấy vùng mặt to nhất (nếu có nhiều mặt).
Cắt vùng mặt (ROI): face_img = frame[y:y+h, x:x+w].
Tiền xử lý (QUAN TRỌNG): Gọi hàm custom_preprocess(face_img) (Hàm này phải copy y chang từ lúc train).
Feature Extraction: Đưa ảnh đã xử lý vào CNN → ra vector đặc trưng.
Classification: Đưa vector vào SVM → ra kết quả (ví dụ: 1 là Engaged).
Vẽ: Vẽ hình chữ nhật xanh/đỏ quanh mặt và viết chữ lên frame.
Hiển thị frame lên màn hình (cv2.imshow hoặc Tkinter Canvas).
Bấm 'Q' để thoát.

Lưu ý
Resize khung hình Webcam: Đừng process full HD (1920x1080). Hãy resize khung hình webcam về 640x480 hoặc 800x600 trước khi đưa vào xử lý. Tốc độ sẽ tăng gấp 3-4 lần.
Skip Frames (Nhảy khung hình): Không cần dự đoán ở tất cả các frame.
Webcam chạy 30 FPS (30 hình/giây).
Bạn chỉ cần detect và predict ở mỗi 5 frame (tức là 6 lần/giây) là mắt thường thấy mượt rồi. Các frame ở giữa giữ nguyên kết quả cũ.
Ánh sáng: Đây là kẻ thù của Demo. Hãy đảm bảo vị trí ngồi demo có ánh sáng tốt, chiếu thẳng vào mặt. Tránh ngồi ngược sáng (cửa sổ sau lưng).

- Chỉ code trong ./src/demo
- Sử dụng model trong ./src/demo/models
- Sử dụng lại các hàm xử lý ảnh trong .src/data_processing
- Chia nhỏ code thành các function, folder nhỏ.