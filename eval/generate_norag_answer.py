import sys
from pathlib import Path
import time
import json

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.llm import ask_llm, client


def load_sample(path="eval/results/judge_sample.json"):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / path, encoding="utf-8") as f:
        return json.load(f)


###################################################################################################


def load_existing_results(path="eval/results/norag_answers.json"):
    """Resume support."""
    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / path
    if full_path.exists():
        with open(full_path, encoding="utf-8") as f:
            return json.load(f)
    return []

#######################################################################################

def generate_norag_answers(sample, client_to_use):
    """Asks Gemini the same questions with NO retrieved context - baseline for comparison."""
    existing = load_existing_results()
    done_questions = {r["question"] for r in existing}
    results = existing

    for i, item in enumerate(sample):
        if item["question"] in done_questions:
            continue

        print(f"[{i+1}/{len(sample)}] {item['question'][:60]}...")

        # No context, no system prompt about grounding - just ask directly
        answer = ask_llm(item["question"])

        results.append({
            "question": item["question"],
            "filename": item["filename"],
            "norag_answer": answer
        })
        save_results(results)
        time.sleep(13)

    return results


################################################################################################

def save_results(results, path="eval/results/norag_answers.json"):
    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / path
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} no-RAG answers to {full_path}")


###############################################################################################

if __name__ == "__main__":
    sample = load_sample()
    results = generate_norag_answers(sample, client_to_use=client)