# DocNorm MD 1.4

DocNorm MD là công cụ web chạy cục bộ để chuyển PDF/DOCX thành Markdown UTF-8,
kiểm tra chất lượng, xem trước, chỉnh sửa và tải gói kết quả. Phiên bản này dừng
trước chunking, embedding và chatbot.

## Chức năng

- Upload PDF hoặc DOCX tối đa 100 MB.
- PDF có text ưu tiên OpenDataLoader Local 2.5.0.
- PDF quét ảnh tự chuyển sang Geometry OCR `EasyOCR vi,en`: box được sắp theo
  trên-xuống/trái-phải và bảng có đường kẻ được dựng lại theo giao điểm lưới.
- Hybrid Docling vẫn là backend tùy chọn, không còn được nạp sẵn và chiếm VRAM.
- DOCX ưu tiên extractor native của `pipeline_v2`.
- Chế độ nhanh bằng PyMuPDF/python-docx được gắn nhãn
  `BASIC_FALLBACK`, không giả vờ là kết quả pipeline đầy đủ.
- Quality report, provenance coverage, engine chain và cảnh báo rõ ràng.
- Preview an toàn: Markdown được hiển thị dưới dạng text, không thực thi HTML.
- Chỉnh sửa Markdown và tự tạo lại manifest/ZIP.
- Lịch sử job lưu cục bộ; không gọi dịch vụ ngoài.
- Tiến trình dài có heartbeat để thời gian xử lý vẫn cập nhật khi OCR nhiều trang.
- Nhánh `Đề toán / ma trận / đáp án hai cột` dùng Qwen3-VL 2B 4-bit cục bộ
  để đọc lại đúng các trang rủi ro, xuất ma trận/vector sang LaTeX và giữ nhãn
  A → B → C → D đúng một lần.
- Tối ưu `fast-first`: OpenDataLoader vẫn xử lý trang sạch; Qwen chỉ nạp một
  lần cho job và mặc định đọc lại tối đa 8 trang có điểm rủi ro cao nhất.
- Validator toán chặn nhãn đáp án kép, thiếu A–D, thứ tự bất thường, ngoặc
  LaTeX lệch, lặp câu/lặp khối, phình output, mất câu so với text layer, code
  fence, ký tự thay thế và dòng rác đại diện ngoặc ma trận.
- Qwen dùng text layer làm từ điển đối chiếu nhưng lấy thứ tự từ ảnh; generation
  có chống lặp và retry ngắn. Trang không vượt kiểm định không được ghi đè lên
  kết quả gốc.
- Profile `Tự nhận diện an toàn` dùng fingerprint chặt để sửa riêng tài liệu
  **Quy trình quản lý vật tư Petrolimex**: thứ tự định nghĩa, số thứ tự kép,
  bảng gãy trang, ký hiệu tự sinh, footer bị gắn heading, boilerplate và hai
  lưu đồ có nhánh Đạt/Không đạt.
- Hậu kiểm fail-closed: nếu profile chuyên biệt đã bật nhưng không sửa đủ cấu
  trúc bắt buộc, job thất bại thay vì công bố Markdown sai.

Profile chuyên biệt không chạy toàn cục. `Auto` chỉ kích hoạt khi các mã tài
liệu, biểu mẫu, heading và thuật ngữ cùng khớp; tài liệu khác giữ nguyên luồng
normalization tổng quát. Người dùng có thể chọn `Chỉ chuẩn hóa tổng quát` để
tắt hoàn toàn profile.

## Cài đặt trên Windows

Giải nén/clone repository, sau đó mở PowerShell tại thư mục
`DocNorm_MD_Web_v1_4`. Lấy thư mục gốc một cách tự động:

```powershell
$ProjectRoot = (Resolve-Path "..").Path
```

Chạy installer:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File ".\install_windows.ps1" `
    -ProjectRoot $ProjectRoot
```

Sau khi thấy `DOCNORM_MD_INSTALL=PASS`:

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File ".\run_windows.ps1" `
    -ProjectRoot $ProjectRoot
```

Trình duyệt tự mở tại `http://127.0.0.1:8010`. Nhấn `Ctrl+C` trong PowerShell
để dừng server. Geometry OCR và Qwen chỉ nạp khi có job cần dùng, nhờ vậy mở
web nhanh và không giữ VRAM khi rảnh. Có thể double-click `launch_docnorm.cmd`
sau khi đã cài. Chỉ đặt `DOCNORM_START_HYBRID=1` trước khi chạy nếu muốn bật
backend Docling tùy chọn.

Nếu installer in `geometry_ocr_ready=false`, chạy đúng lệnh `Fix:` mà installer
hiển thị rồi chạy lại installer. Lệnh này chỉ bổ sung EasyOCR/OpenCV vào `venv`
GPU hiện có; không tạo thêm môi trường và không cài lại model Qwen.

## Đề toán và tối ưu thời gian

- PDF toán có text layer: chạy bộ trích xuất nhanh trước, chấm rủi ro từng
  trang rồi chỉ gọi Qwen cho trang có ma trận, đáp án hai cột, nhãn trùng,
  ký tự ngoặc lỗi hoặc thứ tự đáp án bất thường.
- PDF toán dạng scan: chọn rõ profile `Đề toán / ma trận / đáp án hai cột` để
  định tuyến thẳng Qwen, tránh chạy Docling rồi lại chạy thêm VLM.
- Trang tự luận sạch chỉ có văn xuôi được giữ ở fast path. Giới hạn selective
  fallback mặc định là 8 trang/job và có thể chỉnh bằng
  `DOCNORM_MATH_MAX_VLM_PAGES`.
- Giao diện cập nhật `đã kiểm định x/y trang`, không còn đứng yên ở 64% trong
  toàn bộ thời gian model chạy.
- Model được nạp một lần trong subprocess, xử lý các trang đã chọn rồi thoát
  để giải phóng VRAM. Không chạy song song nhiều job trên GPU 6 GB.
- Worker kiểm tra phải còn ít nhất 2.8 GB VRAM và dừng sớm với hướng dẫn rõ
  ràng thay vì chờ tới lỗi CUDA out-of-memory.
- Cần môi trường `venv` hiện có với CUDA, Transformers, bitsandbytes, PyMuPDF
  và model cache `Qwen/Qwen3-VL-2B-Instruct`. Installer chỉ kiểm tra môi trường,
  không tự tải model và không gửi tài liệu ra Internet.

## Kiểm thử

Các lỗi thực tế và điều kiện chặn tương ứng được tóm tắt tại
`docs/REGRESSION_1_4.md`.

```powershell
powershell.exe `
    -NoProfile `
    -ExecutionPolicy Bypass `
    -File ".\smoke_test_windows.ps1" `
    -ProjectRoot $ProjectRoot
```

Chẩn đoán môi trường:

```powershell
.\diagnose_windows.ps1 `
    -ProjectRoot $ProjectRoot
```

## Đầu ra

Mỗi lần xử lý nằm tại:

```text
<ProjectRoot>\docnorm_runs\<job_id>\output
```

Bao gồm:

```text
document.normalized.md
quality_report.json
manifest.json
docnorm_result.zip
assets\                 # nếu tài liệu có asset được giữ lại
```

## Geometry OCR/Hybrid và fail-closed

- Cả ba chế độ tự nhận diện PDF scan và gọi Geometry OCR. Reader được nạp một
  lần/job, chạy GPU nếu CUDA sẵn sàng, rồi thoát để giải phóng VRAM.
- Báo cáo ghi `EASYOCR_GEOMETRY`, `ocr_invocations=1`, provenance từng trang,
  độ tin cậy OCR và số bảng dựng từ lưới.
- Nếu trang không đủ chữ/độ tin cậy, job dừng `OUTPUT_QUALITY_FAILED`; không có
  file Markdown lỗi mang nhãn hoàn tất.
- Hybrid chỉ là fallback tùy chọn localhost; request vẫn đặt
  `hybrid_fallback=false` khi được bật.
- Geometry OCR dùng `<ProjectRoot>\venv` hiện có với PyTorch CUDA, EasyOCR,
  OpenCV, Pillow và PyMuPDF. Hybrid tùy chọn mới cần `venv_opendataloader`,
  OpenDataLoader PDF 2.5.0, Docling và Java như môi trường Stage 6.1B đã kiểm thử.
- Math VLM mặc định chạy offline từ `<ProjectRoot>\venv`. Nếu môi trường/model
  chưa sẵn sàng, Auto gắn `NEEDS_REVIEW`; profile Đề toán dừng fail-closed.

## Giới hạn có chủ ý

- OCR vẫn có thể nhầm chữ viết tay, ô gộp phức tạp hoặc dấu tiếng Việt; luôn xem
  quality issues và đối chiếu PDF trước khi dùng dữ liệu quan trọng.
- DOCX không có page map đáng tin cậy nên giao diện hiển thị “Không xác định”.
- Chế độ `Giữ cấu trúc` fail-closed nếu `pipeline_v2` hoặc
  `venv_opendataloader` chưa sẵn sàng.
- Không đưa `venv*`, model weights, `data_local`, tài liệu nội bộ,
  `pipeline_v2_output` hoặc `legacy_artifacts` vào bản nộp mã nguồn.
- Profile quản lý vật tư chỉ áp dụng đúng họ tài liệu đã audit. Profile đề toán
  là bộ định tuyến/validator tổng quát nhưng vẫn yêu cầu đối chiếu PDF ở trang
  bị gắn `NEEDS_REVIEW`.

## Kiến trúc

```text
Browser (127.0.0.1)
  -> FastAPI /api/jobs
  -> PipelineBridge
       -> Scan detector
       -> OpenDataLoader Local (PDF có text)
       -> EasyOCR Geometry vi,en + table grid (PDF scan)
       -> OpenDataLoader Hybrid tùy chọn
       -> Selective Qwen3-VL 2B 4-bit (trang toán rủi ro)
       -> DocxNativeExtractor (DOCX)
       -> BASIC_FALLBACK (explicit/labelled)
  -> Quality + generic normalization
  -> Safe fingerprint/math risk score -> selective repair -> fail-closed validation
  -> Markdown + report + manifest + ZIP
```
