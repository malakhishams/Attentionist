import sys
import traceback
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.rag import answer_query
from monitoring import log_interaction, update_feedback

# --- Page config ------------------------------------------------------------

st.set_page_config(
    page_title="Attentionist",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Attentionist")
st.caption(
    "Ask questions about transformer architecture and attention mechanisms — "
    "grounded in 9 foundational papers (Attention Is All You Need, BERT, GPT-3, "
    "ViT, Reformer, Longformer, Performer, Mamba, RoFormer)."
)

# Session state for chat history 

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources", "interaction_id", "feedback"}

# --- Helpers -------------------------------------------------------------------

def render_sources(sources):
    if sources:
        with st.expander("Sources"):
            for s in sources:
                filename = s.get("filename", "unknown")
                content = s.get("content", "")
                st.markdown(f"**{filename}**")
                st.text(content[:300] + ("..." if len(content) > 300 else ""))


def render_feedback_buttons(msg_index):
    msg = st.session_state.messages[msg_index]
    interaction_id = msg.get("interaction_id")
    if interaction_id is None:
        return  # nothing to attach feedback to

    col1, col2, col3 = st.columns([1, 1, 8])
    current = msg.get("feedback")

    with col1:
        if st.button("👍", key=f"up_{interaction_id}", type="primary" if current == "up" else "secondary"):
            update_feedback(interaction_id, "up")
            st.session_state.messages[msg_index]["feedback"] = "up"
            st.rerun()
    with col2:
        if st.button("👎", key=f"down_{interaction_id}", type="primary" if current == "down" else "secondary"):
            update_feedback(interaction_id, "down")
            st.session_state.messages[msg_index]["feedback"] = "down"
            st.rerun()

# --- Render chat history ------------------------------------------------------

for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))
            render_feedback_buttons(i)

# --- Chat input ----------------------------------------------------------------

if question := st.chat_input("Ask about attention mechanisms, transformers, or any of the 9 papers..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                answer, sources, chunks = answer_query(question)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                traceback.print_exc()  # prints full traceback to the terminal for debugging
                answer = "Sorry, something went wrong while generating this answer."
                sources = []
                chunks = []

        interaction_id = log_interaction(question, answer, chunks)

        st.markdown(answer)
        render_sources(chunks)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
        "interaction_id": interaction_id,
        "feedback": None,
    })
    st.rerun()  # rerun so feedback buttons render immediately below the new message

# --- Sidebar: reset + info -----------------------------------------------------

with st.sidebar:
    st.header("About")
    st.markdown(
        "Attentionist is a RAG system built for the LLM Zoomcamp 2026 final project. "
        "It retrieves relevant passages from 9 transformer/attention research papers "
        "and uses Gemini to generate grounded answers."
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()