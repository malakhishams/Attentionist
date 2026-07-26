import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.environ["GEMINI_API_KEY"]

client = OpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

###############################################################################################

def ask_llm(prompt, model="gemini-2.5-flash", system_prompt=None):
    """Sends a prompt to Gemini and returns the text response."""

    messages = []
    if system_prompt:
        messages.append({"role":"system", "content":system_prompt})

    messages.append({"role":"user", "content":prompt})

    response = client.chat.completions.create(model=model,messages=messages)

    return response.choices[0].message.content

###########################################################################################

def build_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(
        f"[Source: {chunk['filename']}]\n{chunk['content']}"
        for chunk in retrieved_chunks
    )

    system_prompt = (
        """
        You are Attentionist, an assistant that answers questions about transformer architecture and attention mechanisms,
        grounded strictly in the provided research paper excerpts. If the answer is not contained in the provided context, say so clearly 
        rather than guessing or using outside knowledge. Also cite which paper(s) your answer draws from.
        """
    )

    user_prompt = f"""Context from research papers:

{context}

Question: {query}

Answer based only on the context above."""

    return system_prompt, user_prompt

#####################################################################################################

