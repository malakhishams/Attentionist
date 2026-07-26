from llm import ask_llm, build_prompt
from retrieval import retrieve, load_index, load_embedder

def answer_query(query, top_k=5):
    """
    Full RAG flow: retrieve relevant chunks, build a prompt, get an answer.
    Returns (answer, sources).
    """

    index = load_index()
    embedder = load_embedder()

    chunks = retrieve(query, index, embedder, top_k=top_k)
    system_prompt, user_prompt = build_prompt(query, chunks)

    answer = ask_llm(user_prompt, system_prompt=system_prompt)

    sources = [ chunk["filename"] for chunk in chunks ]

    return answer, sources

##################################################

if __name__ == "__main__":
    query = "What is locality-sensitive hashing attention in Reformer?"
    answer, sources = answer_query(query)

    print("Question:", query)
    print("\nAnswer:\n", answer)
    print("\nSources:", sources)