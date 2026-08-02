import sys
import time
from pathlib import Path
import json
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.llm import client, new_client

class JudgmentScore(BaseModel):
    score: int  # 1-5
    reasoning: str

#####################################################################

def load_json(path):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / path, encoding="utf-8") as f:
        return json.load(f)

################################################################

def judge_answer(question, answer, context=None, client_to_use=None):

    """
    Scores an answer's quality/correctness on a 1-5 scale.
    If context is provided, also judges groundedness (did it use the context correctly).
    """

    active_client = new_client

    if context:
        prompt = f"""
    You are evaluating the quality of an AI-generated answer about transformer/attention research,
      based on general knowledge (no external sources provided).

Question: {question}

AI's answer: {answer}

Score this answer from 1-5 on factual correctness:
1 = wrong or nonsensical
3 = partially correct
5 = fully correct and accurate

Provide a score and brief reasoning.
"""

    else:
        prompt = f"""You are evaluating the quality of an AI-generated answer about transformer/attention research,
          based on general knowledge (no external sources provided).

Question: {question}

AI's answer: {answer}

Score this answer from 1-5 on factual correctness:
1 = wrong or nonsensical
3 = partially correct
5 = fully correct and accurate

Provide a score and brief reasoning."""

    response = active_client.chat.completions.parse(
        model="gemini-2.5-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format=JudgmentScore,
    )
    return response.choices[0].message.parsed

#######################################################################################################

def run_judging():
    rag_answers = load_json("eval/results/rag_answers.json")
    norag_answers = load_json("eval/results/norag_answers.json")

    results = []

    print("Judging RAG answers...")
    for i, item in enumerate(rag_answers):
        print(f"[{i+1}/{len(rag_answers)}] RAG: {item['question'][:50]}...")
        context = "\n\n---\n\n".join(item["retrieved_context"])
        judgment = judge_answer(item["question"], item["rag_answer"], context=context)
        results.append({
            "question": item["question"],
            "type": "rag",
            "answer": item["rag_answer"],
            "score": judgment.score,
            "reasoning": judgment.reasoning
        })
        time.sleep(13)

    print("Judging no-RAG answers...")
    for i, item in enumerate(norag_answers):
        print(f"[{i+1}/{len(norag_answers)}] No-RAG: {item['question'][:50]}...")
        judgment = judge_answer(item["question"], item["norag_answer"], context=None)
        results.append({
            "question": item["question"],
            "type": "norag",
            "answer": item["norag_answer"],
            "score": judgment.score,
            "reasoning": judgment.reasoning
        })
        time.sleep(13)

    return results


def save_and_summarize(results, path="eval/results/judge_scores.json"):
    base_dir = Path(__file__).resolve().parent.parent
    with open(base_dir / path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rag_scores = [r["score"] for r in results if r["type"] == "rag"]
    norag_scores = [r["score"] for r in results if r["type"] == "norag"]

    print(f"\nRAG average score: {sum(rag_scores)/len(rag_scores):.2f}")
    print(f"No-RAG average score: {sum(norag_scores)/len(norag_scores):.2f}")


if __name__ == "__main__":
    results = run_judging()
    save_and_summarize(results)