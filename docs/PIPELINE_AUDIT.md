# Pipeline source audit

Ngày audit: 2026-08-21.

Nguồn kiểm tra là ZIP `pipeline_v2_current_for_audit_20260821_215627.zip` được
đóng gói trực tiếp từ workspace Windows hiện tại.

## Kết quả apply stage

| Stage | Overlay files | Missing in current pipeline | Kết quả |
| --- | ---: | ---: | --- |
| 5.1A Contracts | 5 | 0 | Applied |
| 5.1B Worker | 3 | 0 | Applied |
| 5.1C Adapter | 3 | 0 | Applied; shared init later updated |
| 5.1D Mapping quality | 5 | 0 | Applied; shared init later updated |
| 5.1E Orchestration commit | 3 | 0 | Applied |
| 5.2A Contract-native normalization | 3 | 0 | Applied; normalization later updated |
| 5.2B Semantic chunking | 3 | 0 | Applied |
| 5.2C Citation validation | 3 | 0 | Applied |
| Stage 5 Extract completion | 6 | 0 | Applied; shared init later updated |
| 6.1A Quality gate/router | 8 | 0 | Exact match |
| 6.1B Hybrid page fallback | 7 | 0 | Exact match |
| 6.1C Hybrid reverify/merge | 2 | 0 | Exact match |
| 7.1A Reverified normalization | 2 | 0 | Exact match |
| 7.1B Structural repair | 4 | 0 | Exact match |
| 7.1B Hotfix 2 | 4 | 0 | Exact match |

Không có file overlay nào bị thiếu. Các hash khác bản stage cũ đều thuộc file
được stage mới hơn cập nhật và đã khớp overlay mới nhất.

## Stage 7.1B

`normalization/structural_repair.py` và test tương ứng đã được apply đầy đủ.
Smoke test tài liệu thí điểm trước đây thất bại vì fragment định nghĩa không
được nhận diện duy nhất, không phải vì code chưa được cài. Web v1.4 xử lý trường
hợp không chắc chắn theo nguyên tắc fail-closed.

## Kiểm thử

- Python compile: PASS.
- Pipeline test collection: 178 tests.
- Pipeline test execution: 178/178 PASS.
- Web v1.4 test collection: 34 tests.
- Web v1.4 test execution trong repository hợp nhất: 34/34 PASS. Integration
  dùng worker giả lập, không gọi mạng và không yêu cầu backend Windows thật.

## Nội dung đã loại khỏi repository

- `rag_pipeline_v2.egg-info` do editable install sinh ra.
- `.pytest_cache`, `__pycache__`, `.pyc`.
- Virtual environments, model weights, local data và generated outputs.
- Stage delivery folders/ZIP vì đã được hợp nhất vào source hiện tại.
