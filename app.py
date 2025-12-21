import streamlit as st
st.set_page_config(page_title="Bible Study Assistant", layout="centered")

import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

# ---------------- ENV ----------------
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = None
if API_KEY:
    try:
        client = genai.Client(api_key=API_KEY)
    except Exception:
        client = None

# ---------------- LOAD DATA ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_index():
    return faiss.read_index("bible.index")

@st.cache_resource
def load_chunks():
    return np.load("bible_chunks.npy", allow_pickle=True)

embedder = load_embedder()
index = load_index()
bible_chunks = load_chunks()

# ---------------- HELPERS ----------------
def parse_chunk(chunk):
    parts = chunk.split("|", 3)
    if len(parts) != 4:
        return None
    book, chapter, verse, text = parts
    return book, int(chapter), int(verse), text

def normalize_book(book: str) -> str:
    return book.lower().strip()

# ---------------- BOOK DETECTION ----------------
BOOKS = [
    "genesis","exodus","leviticus","numbers","deuteronomy",
    "psalms","matthew","mark","luke","john","acts","romans"
]

def detect_book_chapter(question):
    q = question.lower()
    for book in BOOKS:
        m = re.search(rf"{book}\s+(\d+)", q)
        if m:
            return book, int(m.group(1))
    return None, None

# ---------------- SEARCH ----------------
def semantic_search(query, k=5):
    vec = embedder.encode([query]).astype("float32")
    faiss.normalize_L2(vec)
    _, idx = index.search(vec, k)
    return [bible_chunks[i] for i in idx[0]]

def get_chapter(book, chapter):
    results = []
    for chunk in bible_chunks:
        parsed = parse_chunk(chunk)
        if not parsed:
            continue
        b, c, v, text = parsed
        if normalize_book(b) == book and c == chapter:
            results.append(f"{b.title()} {c}:{v} {text}")
    return results

# ---------------- GEMINI (OPTIONAL) ----------------
def gemini_answer(prompt):
    if not client:
        return None
    try:
        return client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        ).text
    except Exception:
        return None

# ---------------- ANSWER LOGIC ----------------
def answer(question, mode):
    book, chapter = detect_book_chapter(question)

    if book and chapter:
        verses = get_chapter(book, chapter)
        if not verses:
            return f"No verses found for {book.title()} {chapter}"

        scripture = "\n".join(verses)

        if mode == "Scripture only":
            return f"📖 **{book.title()} {chapter}**\n\n{scripture}"

        prompt = f"""
Summarize the following Scripture faithfully:

{scripture}
"""
        ai = gemini_answer(prompt)
        return ai if ai else scripture

    # fallback semantic
    chunks = semantic_search(question)
    readable = []
    for c in chunks:
        parsed = parse_chunk(c)
        if parsed:
            b, ch, v, t = parsed
            readable.append(f"{b.title()} {ch}:{v} {t}")

    scripture = "\n".join(readable)

    if mode == "Scripture only":
        return scripture

    prompt = f"""
Answer the question using Scripture only:

{scripture}

Question: {question}
"""
    ai = gemini_answer(prompt)
    return ai if ai else scripture

# ---------------- UI ----------------
st.title("📖 Bible Study Assistant")

mode = st.radio(
    "Answer mode",
    ["Scripture only", "Scripture + Commentary"],
    horizontal=True
)

question = st.chat_input("Ask a Bible question")

if question:
    with st.spinner("Searching Scripture..."):
        response = answer(question, mode)
    st.chat_message("assistant").write(response)
