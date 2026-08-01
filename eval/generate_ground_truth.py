import json
from pydantic import BaseModel
from pathlib import Path
import sys
import random
import time
from openai import RateLimitError


sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.retrieval import load_index
from src.llm import client
from src.llm import new_client

class GeneratedQuestion(BaseModel):
    question : str

def group_chunks_by_paper(index, n_per_paper=4):

    """Groups chunks by paper, samples n_per_paper from each."""

    by_paper ={}
    for chunk in index.docs:
        filename = chunk["filename"]
        if filename not in by_paper:
            by_paper[filename] = []
        by_paper[filename].append(chunk)

    # take same number of chunks from all chunks for each paper

    sampled = []
    for filename,chunks in by_paper.items():
        sampled.extend(random.sample(chunks, min(n_per_paper,len(chunks))))

    return sampled

##################################################################################################

def generate_quesion(chunk, max_retries=3):
    """Asks Gemini to generate a question this chunk would answer."""

    prompt = f"""
        Given this excerpt from a research paper,
        generate ONE clear, specific question that this excerpt directly answers.
        The question should be answerable using only this excerpt.

        Excerpt: {chunk}

        Respond ONLY with the question, do NOT return any other text.

    """

    for attempt in range(max_retries):
        try:
            response = new_client.chat.completions.create(

                model="gemini-2.5-flash",
                messages=[{"role":"user", "content":prompt}]
                #response_format=GeneratedQuestion

            )

            return response.choices[0].message.content.strip()

        except RateLimitError as e:
                print("======================================================")
                print(e)
                print("======================================================")
                wait = 60  # or parse e's retryDelay if you want to be precise
                print(f"Rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(60)
    raise RuntimeError("Failed after max retries")


###################################################################################

def build_ground_truth(n_per_paper):

    index = load_index()
    sampled_chunks = group_chunks_by_paper(index, n_per_paper)

    ground_truth = []

    print("Generating Ground Truth...")
    print("="*20)
    for i,chunk in enumerate(sampled_chunks):
        print(f"[{i+1}/{len(sampled_chunks)}] Requesting question for chunk_id={chunk.get('id', i)}, filename={chunk['filename']}...")
        question = generate_quesion(chunk["content"])
        ground_truth.append({
            "question" : question,
            "start": chunk["start"], 
            "filename" : chunk["filename"]
        })
        save_ground_truth(ground_truth)
        time.sleep(13) 

    return ground_truth
################################################################################################

def save_ground_truth(ground_truth, path="eval/results/ground_truth.json"):
    base_dir = Path(__file__).resolve().parent.parent
    full_path = base_dir / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(ground_truth)} ground-truth questions to {full_path}")

#########################################################################################

# Add this before your main loop to check quota
def check_quota():
    try:
        # This might vary based on your client setup
        print("Checking API availability...")
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[{"role":"user", "content":"test"}],
            max_tokens=10
        )
        print("✓ API is accessible")
    except RateLimitError:
        print("✗ Already rate limited. Waiting 5 minutes...")
        time.sleep(300)
    except Exception as e:
        print(f"API check error: {e}")

if __name__ == "__main__":
    #check_quota()
    gt = build_ground_truth(n_per_paper=4)
    save_ground_truth(gt)