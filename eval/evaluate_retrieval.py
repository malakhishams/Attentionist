import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval import load_index, load_embedder, retrieve

def load_ground_truth(path="eval/results/ground_truth.json"):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / path, encoding="utf-8") as f:
        return json.load(f)


############################################################################################3333

def evaluate(ground_truth, index, embedder, top_k=5):

    hits = 0
    reciprocal_ranks = []

    for item in ground_truth:
        results = retrieve(item["question"], index, embedder, top_k=top_k)

        rank = None

        for i, result in enumerate(results):
            if result["filename"] == item["filename"] and result["start"] == item["start"]: # same filename and content retrieved
                rank = i + 1    # found
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1/rank)     # used for mrr later
        else:
            reciprocal_ranks.append(0)

    hit_rate = hits / len(ground_truth)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return { "hit_rate":hit_rate , 
                "mrr":mrr,
                "total": len(ground_truth)}

    ###################################################################################

if __name__ == "__main__":
    ground_truth = load_ground_truth()
    index = load_index()
    embedder = load_embedder()

    results_k5 = evaluate(ground_truth, index, embedder, top_k=5)
    results_k10 = evaluate(ground_truth, index, embedder, top_k=10)

    print(f"top_k=5:  Hit Rate {results_k5['hit_rate']:.2%}, MRR {results_k5['mrr']:.3f}")
    print(f"top_k=10: Hit Rate {results_k10['hit_rate']:.2%}, MRR {results_k10['mrr']:.3f}")
