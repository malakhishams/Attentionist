import sys
from pathlib import Path
import time
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag import answer_query

def load_sample(path="eval/results/judge_sample.json"):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / path, encoding="utf-8") as f:
        return json.load(f)


###################################################################################

def generate_rag_answer(sample):

    result = []

    for i, item in enumerate(sample):
        print(f"[{i+1}/{len(sample)}] {item['question'][:60]}...")
        answer, sources = answer_query(item["question"])
        result.append({
            "question": item["question"],
            "filename": item["filename"],
            "rag_answer": answer,
            "sources": sources
        })
        time.sleep(13)
    return result


##############################################################################

def save_results(results, path="eval/results/rag_answers.json"):
    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / path
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} RAG answers to {full_path}")

###############################################################################

if __name__ == "__main__":
    sample = load_sample()
    results = generate_rag_answer(sample)
    save_results(results)
