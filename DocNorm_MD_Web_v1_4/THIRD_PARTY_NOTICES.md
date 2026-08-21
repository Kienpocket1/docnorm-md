# Third-party notices

DocNorm MD integrates or optionally calls the following projects. Their own
licenses and notices remain controlling:

- FastAPI — MIT License
- Starlette — BSD 3-Clause License
- Uvicorn — BSD 3-Clause License
- Pydantic — MIT License
- python-docx — MIT License
- PyMuPDF — AGPL/commercial dual licensing; review distribution obligations
  before redistributing binaries in a public submission.
- OpenDataLoader PDF and its transitive Java/Python dependencies — use the
  license metadata supplied by the installed `opendataloader-pdf` distribution.
- Docling and its OCR/model dependencies — used only from the separately
  installed local Hybrid environment; retain the licenses supplied there.
- EasyOCR and OpenCV — geometry-preserving scan OCR loaded from the separately
  installed CUDA environment; retain their distribution and model notices.
- Qwen3-VL, Transformers and bitsandbytes — optional local mathematical-page
  fallback loaded from a separately installed environment/model cache; retain
  the model card, code licenses and notices supplied by their distributions.

This source deliverable does not bundle Python virtual environments, model
weights, Java archives from installed environments, or company documents.
