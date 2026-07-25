## Notes for README (draft as I go)

- Model files (model.onnx, tokenizer.json) are gitignored — download from
  Xenova/all-MiniLM-L6-v2 on HuggingFace, place in models/all-MiniLM-L6-v2/
- pypdf used for PDF extraction (tested vs pdfplumber, pypdf worked fine on these 9 papers)
- gitsource's chunk_documents() used for chunking (size=2000, step=1000) — same as Module 2
- ingest.py runs the full pipeline: extract → chunk → embed → index → save