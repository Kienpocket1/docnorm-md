# Regression quality gate - DocNorm MD 1.4

Hai đầu ra thực tế bị lỗi được dùng làm regression đầu vào cho bộ hậu kiểm.
Tài liệu nguồn và nội dung nội bộ không được đóng gói cùng mã nguồn.

| Regression | Biểu hiện | Chính sách 1.4 |
| --- | --- | --- |
| Đề toán | Một câu lặp hơn 100 lần | `runaway_repeated_line` và `runaway_repeated_block` - chặn xuất |
| Đề toán | Khối Câu 9/10 lặp nguyên văn | `duplicate_question_block` - chặn xuất |
| Đề toán | Thiếu lựa chọn D hoặc xuất một lựa chọn lạc | `incomplete_option_set` - chặn xuất |
| Đề toán | Mất câu/lời giải so với text layer | lexical coverage + question coverage - retry một lần rồi chặn |
| Đề toán | Model kéo dài vô hạn | `no_repeat_ngram_size`, repetition penalty và giới hạn 1.800 token |
| PDF scan | Từ trong cùng câu bị đảo | bỏ thứ tự node OCR; sắp box theo hàng và trái-phải |
| PDF scan | Bảng bị làm phẳng | phát hiện đường ngang/dọc, ánh xạ OCR box vào cell và xuất GFM |
| Job dài | Giao diện đứng yên tại một mốc | worker ghi tiến độ; UI cập nhật số trang đã xử lý |

## Điều kiện xuất bản

- Không có ký tự thay thế U+FFFD.
- Math validator không có error.
- Mọi trang của worker fail-closed đều có provenance.
- Không có trang worker ở trạng thái `FAILED_VALIDATION`.
- Source SHA256 không thay đổi trong khi OCR/VLM chạy.

Khi một điều kiện không đạt, job kết thúc với `OUTPUT_QUALITY_FAILED` và không
tạo Markdown tải xuống. Đây là hành vi chủ ý để dữ liệu lỗi không đi vào bước
chunking.
