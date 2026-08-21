# DocNorm MD

DocNorm MD là công cụ web chạy cục bộ để chuyển PDF/DOCX thành Markdown có
cấu trúc, kiểm định chất lượng và provenance trước khi dữ liệu đi vào bước
chunking của hệ thống RAG.

Repository này là bản mã nguồn sạch để review trên GitHub. Model, virtual
environment, tài liệu nội bộ và kết quả chạy không được commit.

## Thành phần

```text
.
├── DocNorm_MD_Web_v1_4/   # FastAPI web app và quality gates
├── pipeline_v2/           # Extraction/normalization pipeline
├── docs/                  # Báo cáo audit source
├── install_windows.ps1    # Cài đặt từ thư mục gốc
├── run_windows.ps1        # Chạy ứng dụng
├── test_windows.ps1       # Chạy cả hai test suite
└── verify_repository.ps1  # Kiểm tra trước khi git push
```

## Chức năng chính

- PDF có text layer: OpenDataLoader Local và contract-native normalization.
- PDF scan: EasyOCR Geometry sắp box theo tọa độ và tái dựng bảng có đường kẻ.
- Tài liệu toán: selective Qwen3-VL với kiểm định chống lặp, thiếu câu và sai
  cấu trúc đáp án.
- DOCX: native extraction giữ heading, paragraph, table và list.
- Fail-closed quality gate: trang không vượt kiểm định không được công bố như
  kết quả hoàn tất.
- Chạy cục bộ; không gửi tài liệu tới dịch vụ OCR bên ngoài.

## Cài đặt trên Windows

Yêu cầu Python 3.12. Mở PowerShell tại repository rồi chạy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\install_windows.ps1"
```

Installer tạo/reuse `venv_docnorm_web` và cài hai package ở editable mode.
Các backend GPU là tùy chọn và được phát hiện từ môi trường cục bộ hiện có.

## Chạy web

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\run_windows.ps1"
```

Ứng dụng mở tại `http://127.0.0.1:8010`. Kết quả cục bộ được ghi vào
`docnorm_runs/` và đã bị loại khỏi Git.

## Kiểm thử

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\test_windows.ps1"
```

Hoặc chạy kiểm tra an toàn repository trước khi push:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\verify_repository.ps1"
```

## Backend tùy chọn

- Geometry OCR cần môi trường local có PyTorch CUDA, EasyOCR, OpenCV, Pillow và
  PyMuPDF.
- Math VLM cần Qwen3-VL 2B đã cache cục bộ.
- OpenDataLoader Hybrid cần OpenDataLoader PDF 2.5.0, Docling và Java.

Model weights không nằm trong repository. Web vẫn import và các unit test vẫn
chạy khi không có GPU/model.

## Audit pipeline

Chi tiết đối chiếu các stage và kết quả test nằm tại
[`docs/PIPELINE_AUDIT.md`](docs/PIPELINE_AUDIT.md).

## Đưa lên GitHub

Nên tạo repository **Private** nếu tài liệu hoặc quy tắc nghiệp vụ thuộc phạm
vi thực tập của doanh nghiệp. Sau khi giải nén bản bàn giao, mở PowerShell ngay
tại thư mục này và chạy:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\verify_repository.ps1"
git init
git branch -M main
git add .
git commit -m "Initial DocNorm MD submission"
git remote add origin https://github.com/<tai-khoan>/<repository>.git
git push -u origin main
```

Thay `<tai-khoan>/<repository>` bằng repository do bạn tạo trên GitHub. Không
chép thêm `venv*`, model, PDF nội bộ hoặc `docnorm_runs` vào repository; các
thành phần này đã được `.gitignore` chặn.
