import json
import random
from pathlib import Path

def load_ground_truth(path="eval/results/ground_truth.json"):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / path, encoding="utf-8") as f:
        return json.load(f)


def sample_for_judging(ground_truth, n_per_paper=2, seed=42):
    """Samples a subset for LLM-as-judge evaluation, spread across papers."""
    random.seed(seed)  # reproducible sampling

    by_paper = {}
    for item in ground_truth:
        by_paper.setdefault(item["filename"], []).append(item)

    sampled = []
    for filename, items in by_paper.items():
        sampled.extend(random.sample(items, min(n_per_paper, len(items))))

    return sampled


if __name__ == "__main__":
    gt = load_ground_truth()
    sample = sample_for_judging(gt, n_per_paper=2)

    print(f"Selected {len(sample)} questions across {len(set(s['filename'] for s in sample))} papers")
    for s in sample:
        print(f"- [{s['filename']}] {s['question']}")

    # save the sample for the next steps
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / "eval/results/judge_sample.json", "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)