import pickle
from pathlib import Path
import sys
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from src.embedder import Embedder


def load_index(index_path="data/processed/index.pkl"):
    """Loads the pickled minsearch index built by ingest.py."""
    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / index_path

    if not full_path.exists():
        raise FileNotFoundError(
            f"Index not found at {full_path}. "
            f"Run 'python src/ingest.py' first to build it."
        )

    with open(full_path, "rb") as f:
        index = pickle.load(f)

    return index

#############################################################################

def load_embedder(model_path="models/all-MiniLM-L6-v2"):
    """Loads the same embedder used during ingestion."""
    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / model_path

    return Embedder(path=full_path)

######################################################################################

def retrieve(query, index, embedder, top_k=5):

    """
    Given a user query, returns the top_k most relevant chunks.
    """

    query_vector = embedder.encode(query)
    result = index.search(query_vector, num_results=top_k)

    return result


if __name__ == "__main__":

    index = load_index()
    lengths = [len(chunk["content"]) for chunk in index.docs]

    print(f"Total chunks: {len(index.docs)}")
    embedder = load_embedder()
    results = retrieve("How does Reformer reduce attention complexity?", index, embedder)
    for r in results:
        print(r["filename"], "-", r["content"][:100])