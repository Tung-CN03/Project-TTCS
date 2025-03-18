# Project-TTCS
## I.	Tổng quan
### 1.	Giới thiệu:
 Trong những năm gần đây, trí tuệ nhân tạo (AI) và Mạng đối nghịch tạo sinh (GAN) đã phát triển mạnh mẽ, mang lại nhiều đột phá trong nhiều lĩnh vực khác nhau. Anycost GAN là một biến thể tiên tiến của mô hình GAN, có khả năng tối ưu hóa hiệu suất mà không ảnh hưởng đến chất lượng hình ảnh.
###2.	Mục tiêu:
-	Nghiên cứu nguyên lý hoạt động của mô hình Anycost GAN.
-	Xây dựng một hệ thống tổng hợp và chỉnh sửa ảnh dựa trên mô hình này.
-	Ứng dụng mô hình Anycost GAN vào trong việc chỉnh sửa khuôn mặt, tạo hình ảnh giả lập, và hỗ trợ trong các chuyên ngành liên quan.

## II.	Cở sở lý thuyết
### 1.	Nguyên lý Anycost GAN
Anycost GAN dựa trên ý tưởng tương tự như Anycost Neural Networks, trong đó một mô hình có thể chạy ở nhiều cấu hình khác nhau với mức sử dụng tài nguyên khác nhau. Điều này có nghĩa là thay vì huấn luyện nhiều mô hình GAN riêng biệt với kích thước khác nhau, ta chỉ cần huấn luyện một mạng duy nhất có thể thích ứng với nhiều cấu hình.
Anycost GAN đạt được điều này bằng cách áp dụng ba kỹ thuật chính:
-	Sub-network Sampling (Lấy mẫu mạng con): Mạng chính có thể hoạt động với nhiều kích thước khác nhau bằng cách vô hiệu hóa một số phần trong kiến trúc mạng.
-	Weight Sharing (Chia sẻ trọng số): Các mạng con chia sẻ trọng số với mạng lớn nhất, giúp tiết kiệm tài nguyên khi huấn luyện và giảm chi phí bộ nhớ.
-	Progressive Training (Huấn luyện tiến hóa): Mạng được huấn luyện với nhiều cấu hình khác nhau để đảm bảo rằng cả mô hình lớn và nhỏ đều có thể tạo ra kết quả tốt.
