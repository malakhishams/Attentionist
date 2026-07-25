from pypdf import PdfReader
import logging
from pathlib import Path
from gitsource import chunk_documents
from embedder import Embedder
import minsearch
import pickle


def extract_text_from_pdf(file) -> str:
    """Extracts raw text from an uploaded PDF file."""
    try:
        reader = PdfReader(file)
        text = "".join([page.extract_text() for page in reader.pages])
        logging.info("PDF extraction completed successfully.")
        return text
    except Exception as e:
        logging.error("Failed to read PDF file format.")
        return ""

########################################################################################

def extract_all_papers(raw_dir="data/raw", output_dir="data/processed"):
    """Extracts text from every PDF in raw_dir and saves as .txt files."""
    base_dir = Path(__file__).resolve().parent.parent  # adjust if needed for your structure
    raw_path = base_dir / raw_dir
    output_path = base_dir / output_dir
    output_path.mkdir(parents=True, exist_ok=True)

    for pdf_file in raw_path.glob("*.pdf"):
        text = extract_text_from_pdf(pdf_file)
        out_file = output_path / (pdf_file.stem + ".txt")
        out_file.write_text(text, encoding="utf-8")
        print(f"Extracted: {pdf_file.name} -> {len(text)} chars")
        
#################################################################################################

def build_documents(processed_dir="data/processed"):
    """Turns extracted .txt files into gitsource-compatible document dicts."""
    base_dir = Path(__file__).resolve().parent.parent
    proc_path = base_dir / processed_dir
    documents = []
    for file in proc_path.glob("*.txt"):
        documents.append({
            "filename" : file.stem,
            "content": file.read_text(encoding="utf-8")
        })
    return documents

#############################################################################################

def build_chunks(processed_dir="data/processed", size=2000, step=1000):
    """Builds documents from processed text files and chunks them."""
    documents = build_documents(processed_dir)
    chunks = chunk_documents(documents, size=size, step=step)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
    return chunks

########################################################################################################

def build_index(chunks):
    """Embeds chunks and builds a minsearch VectorSearch index."""
    base_dir = Path(__file__).resolve().parent.parent  # Attentionist/ root
    model_path = base_dir / "models" / "all-MiniLM-L6-v2"
    embedder = Embedder(path=model_path)

    chunk_contents = [c["content"] for c in chunks]
    embedded_chunks = embedder.encode_batch(chunk_contents)

    index = minsearch.VectorSearch()
    index.fit(embedded_chunks, chunks)

    print(f"Indexed {len(chunks)} chunks.")
    return index

##############################################################################3

def save_index(index, path="data/processed/index.pkl"):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / path, "wb") as f:
        pickle.dump(index, f)

    print("index saved")

#########################################################################################33



if __name__ == "__main__":
    extract_all_papers()
    chunks = build_chunks()
    index = build_index(chunks)
    save_index(index)
