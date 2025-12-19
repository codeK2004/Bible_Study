import os
import re
import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# ---------------- ENV SETUP ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY) if API_KEY else None

# ---------------- LOAD MODELS & DATA ----------------
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

# ---------------- VERSE EXTRACTION (FIXED & CLEAN) ----------------
def extract_verses(chunks, limit=8):
    verses = []
    current_ref = None
    current_text = ""

    for chunk in chunks:
        for raw_line in chunk.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            # Detect verse start: {106:1} OR 106:1
            match = re.search(r"\{?(\d+:\d+)\}?", line)

            if match:
                # Save previous verse
                if current_ref and current_text:
                    verses.append(f"**{current_ref}** — {current_text.strip()}")
                    if len(verses) >= limit:
                        return verses

                # Start new verse
                current_ref = match.group(1)
                current_text = re.sub(r"\{?\d+:\d+\}?", "", line).strip()

            else:
                # Continue verse text
                if current_ref:
                    current_text += " " + line

        if len(verses) >= limit:
            break

    # Add final verse
    if current_ref and current_text and len(verses) < limit:
        verses.append(f"**{current_ref}** — {current_text.strip()}")

    return verses

# ---------------- GEMINI (CACHED) ----------------
@st.cache_data(show_spinner=False)
def cached_llm_answer(question, context):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"""
You are a Bible study assistant.

You may:
- Explain Bible verses
- Summarize biblical teaching
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

# ---------------- ANSWER LOGIC ----------------
def answer_question(question, mode):
    retrieved_chunks = retrieve(question)
    verses = extract_verses(retrieved_chunks)
    context = "\n".join(retrieved_chunks)

    # Scripture-only mode OR Gemini unavailable
    if mode == "Scripture only (No AI)" or client is None:
        if verses:
            return "📖 **Bible verses related to your question:**\n\n" + "\n\n".join(verses)
        return "📖 Scripture exists on this topic, but verses could not be clearly extracted."

    # Gemini explain mode
    try:
        return cached_llm_answer(question, context)
    except Exception:
        if verses:
            return (
                "⚠️ Gemini unavailable. Showing Scripture instead:\n\n"
                + "\n\n".join(verses)
            )
        return "⚠️ Gemini unavailable and no verses could be extracted."

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Bible Study Assistant", layout="centered")

st.title("📖 Bible Study Assistant")
st.caption("RAG-based • Bible + Commentary • Gemini 2.5 Flash")

mode = st.radio(
    "Answer mode",
    ["Explain (Gemini)", "Scripture only (No AI)"],
    horizontal=True
)

if "chat" not in st.session_state:
    st.session_state.chat = []

question = st.chat_input("Ask a Bible question...")

if question:
    answer = answer_question(question, mode)
    st.session_state.chat.append(("You", question))
    st.session_state.chat.append(("BibleBot", answer))

for speaker, text in st.session_state.chat:
    with st.chat_message("assistant" if speaker == "BibleBot" else "user"):
        st.write(text)
