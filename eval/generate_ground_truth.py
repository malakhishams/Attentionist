import json
from pydantic import BaseModel
from pathlib import Path
import sys
import random
from src.retrieval import load_index
from src.llm import client

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

def generate_quesion(chunk):
    """Asks Gemini to generate a question this chunk would answer."""

    prompt = """
        Given this excerpt from a research paper,
        generate ONE clear, specific question that this excerpt directly answers.
        The question should be answerable using only this excerpt.

        Excerpt: {chunk}

        Respond ONLY with the question, do NOT return any other text.

    """
    response = client.chat.completions.create(

        model="gemini-2.5-flash",
        messages=[{"role":"user", "content":prompt}],
        response_format=GeneratedQuestion

    )

    return response.choices[0].message.parsed.question

###################################################################################

def build_ground_truth(n_per_paper):
    pass