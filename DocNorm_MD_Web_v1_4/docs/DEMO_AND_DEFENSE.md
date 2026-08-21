# Kịch bản demo và bảo vệ đồ án

## Demo 5 phút

1. Chạy `launch_docnorm.cmd`; trình duyệt mở tại `127.0.0.1:8010`.
2. Chỉ trạng thái `Pipeline sẵn sàng` hoặc `fallback sẵn sàng` trên header.
3. Upload một DOCX tổng hợp có heading, list và table; chọn `Tự động`.
4. Giải thích bốn bước tiến trình: upload, extract, verify, normalize.
5. Mở kết quả, chỉ các metric, engine chain và warning.
6. Sửa một dòng Markdown rồi nhấn `Lưu chỉnh sửa`.
7. Tải ZIP và mở bốn artifact: Markdown, report, manifest, ZIP/assets.
8. Upload PDF scan mẫu; chỉ ra trạng thái `Geometry OCR`, engine chain và
   `ocr_invocations=1` trong report.
9. Chỉ tiến độ OCR theo từng trang và bảng GFM được dựng từ lưới ảnh.
10. Xóa job để chứng minh dữ liệu được quản lý cục bộ.
11. Với PDF đề toán, chọn profile Đề toán và chỉ metric `Trang VLM đọc lại`:
    trang văn xuôi sạch không tốn lượt VLM, trang ma trận được xuất LaTeX.

Không dùng tài liệu nội bộ thật trong video hoặc bản nộp. Tạo một PDF/DOCX mẫu
không chứa tên người, công ty, chữ ký hoặc biểu mẫu bí mật.

## Câu hỏi phản biện thường gặp

### Vì sao không làm chatbot hoàn chỉnh?

Chất lượng RAG phụ thuộc trực tiếp vào dữ liệu đầu vào. Đồ án tập trung giải
quyết ranh giới khó và có thể đo được: chuyển tài liệu bán cấu trúc thành
Markdown sạch, giữ provenance và phát hiện trường hợp cần review trước chunking.

### Vì sao có hai môi trường Python?

OpenDataLoader/Docling/Torch có dependency nặng và dễ xung đột. Web runtime nhẹ
được tách khỏi worker extraction; hai bên giao tiếp qua file manifest và
subprocess có timeout.

### Fallback có làm sai dữ liệu không?

Fallback không được trình bày như kết quả đầy đủ. Report luôn ghi
`BASIC_FALLBACK` và `PASS_WITH_WARNINGS`. PDF scan chỉ được công bố khi Geometry
OCR hoàn tất, đủ provenance và mọi trang vượt kiểm định; worker hỏng thì job
dừng fail-closed.

### Vì sao không dùng structural repair cho mọi file?

Stage đó được sinh từ audit một quy trình quản lý vật tư và chứa quy tắc ngữ
nghĩa cụ thể. Chạy toàn cục có nguy cơ chế thêm nội dung. DocNorm MD 1.4 dùng
fingerprint chặt để chỉ kích hoạt profile khi đúng họ tài liệu; file khác đi qua
normalization tổng quát. Khi người dùng ép profile sai tài liệu, hệ thống dừng
fail-closed thay vì đoán.

### Vì sao không chạy VLM cho toàn bộ PDF?

Chi phí lớn nhất là nạp model và sinh token theo từng trang. Hệ thống dùng
OpenDataLoader làm fast path, chấm điểm rủi ro theo provenance trang và chỉ gọi
Qwen cho tối đa các trang cần thiết. Model được nạp một lần cho cả nhóm trang.
Nếu số trang rủi ro vượt giới hạn, report ghi rõ trang chưa xử lý thay vì âm
thầm công bố kết quả như đã sửa xong.

### Làm sao chứng minh kết quả không bị thay đổi âm thầm?

Mỗi artifact có `size_bytes` và `sha256` trong `manifest.json`; report ghi source
SHA, engine chain, issue và số ký tự hỏng. Khi người dùng sửa Markdown, manifest
và ZIP được tạo lại đồng bộ.
