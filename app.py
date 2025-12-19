import os
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# ---------------- SETUP ----------------
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("bible.index")
chunks = np.load("chunks.npy", allow_pickle=True)

# ---------------- RETRIEVAL ----------------
def retrieve(query, k=8):
    query = query.strip().lower()
    q_vec = embedder.encode([query])
    q_vec = np.array(q_vec).astype("float32")
    faiss.normalize_L2(q_vec)

    _, idx = index.search(q_vec, k)
    return [chunks[i] for i in idx[0]]

# ---------------- CACHED LLM CALL ----------------
@st.cache_data(show_spinner=False)
def cached_llm_answer(question, context):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
You are a Bible study assistant.

You may:
- Explain Bible verses found in the context
- Summarize biblical narratives when multiple verses are present
- Use commentary for interpretation

Rules:
- Do NOT use modern opinions
- Do NOT use outside sources
- Stay faithful to Scripture

Context:
{context}

Question:
{question}
"""
    )
    return response.text

def ask_llm(question):
    context = "\n".join(retrieve(question))
    try:
        return cached_llm_answer(question, context)
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Daily API quota reached. Please try again later."
        return "An unexpected error occurred."

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Bible Study Assistant", layout="centered")
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
