import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# ---------- SETUP ----------
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("bible.index")
chunks = np.load("chunks.npy", allow_pickle=True)

# ---------- RETRIEVAL ----------
def retrieve(query, k=5):
    q_vec = embedder.encode([query])
    q_vec = np.array(q_vec).astype("float32")
    faiss.normalize_L2(q_vec)

    _, idx = index.search(q_vec, k)
    return [chunks[i] for i in idx[0]]

def ask_llm(question):
    context = "\n".join(retrieve(question))

    prompt = f"""
You are a Bible scholar.

Use ONLY the context below (Bible verses + commentary).
Do not use outside knowledge.

Context:
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text

# ---------- UI ----------
st.set_page_config(page_title="Bible RAG Assistant", layout="centered")
st.title("📖 Bible Study Assistant")
st.caption("RAG-based • Bible + Commentary • Gemini 2.5 Flash")

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.chat_input("Ask a Bible question...")

if question:
    answer = ask_llm(question)

    st.session_state.chat.append(("You", question))
    st.session_state.chat.append(("BibleBot", answer))

for speaker, text in st.session_state.chat:
    with st.chat_message("assistant" if speaker == "BibleBot" else "user"):
        st.write(text)
