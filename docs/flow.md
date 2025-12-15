Ảnh trong ./data đã được label.
 -> CẮT KHUÔN MẶT ROI HAARLIKE (THÀNH) /face_detection
 -> CHUYỂN ẢNH XÁM -> GAUSSIAN -> HISTOGRAM -> RESIZE (128x128) - /data_processing  ĐẢM BẢO VIỆC NHẬN ĐƯỢC INPUT LÀ HÌNH ẢNH TRAIN TỪ DATA VÀ HÌNH ẢNH TỪ WEBCAM. (HUY)
 -> OUTPUT: ẢNH ĐÃ XỬ LÝ 
 -> CNN KHI NÀY ĐỐNG VAI TRÒ TRÍCH XUẤT ĐẶT TRƯNG (TỪ TRONG NOTEBOOK HIỆN TẠI student_engagement_CNNmodel.ipynb ĐANG ÁP DỤNG ẢNH GỐC, XLA CƠ BẢN, 256, FULL BƯỚC HUẤN LUYỆN 
 -> CHUYỂN THÀNH SỬ DỤNG ẢNH ĐÃ XỬ LÝ -> SIZE 128x128 -> BỎ BƯỚC CUỐI HUẤN LUYỆN (SOFTMAX) ) -> THÀNH FILE (.H5) -> SVM TRAIN LẤY PHÂN LOẠI (.PKL). XONG (THẠCH)
Việc xử lý ảnh, demo trong codebase này.
Model sẽ được train trong file ipynb ở colab