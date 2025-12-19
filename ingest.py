import os
import faiss
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

# Load environment
load_dotenv()

# Load local embedding model (NO API)
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- PDF LOADER ----------
def load_pdf(path):
    reader = PdfReader(path)

    if reader.is_encrypted:
        reader.decrypt("")

    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    if not text.strip():
        raise RuntimeError(f"No extractable text in {path}")

    return text


print("Loading Bible PDF...")
bible_text = load_pdf("data/bible.pdf")
print("Bible loaded.")

print("Loading Commentary PDF...")
commentary_text = load_pdf("data/commentary.pdf")
print("Commentary loaded.")

combined_text = bible_text + "\n\n" + commentary_text

# ---------- CHUNK ----------
def chunk_text(text, size=500):
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i + size])
        i += size
    return chunks

chunks = chunk_text(combined_text)
print("Chunks:", len(chunks))

# ---------- EMBEDDINGS (LOCAL) ----------
vectors = model.encode(chunks, show_progress_bar=True)
vectors = np.array(vectors).astype("float32")

# ---------- FAISS ----------
index = faiss.IndexFlatIP(vectors.shape[1])
faiss.normalize_L2(vectors)
index.add(vectors)


faiss.write_index(index, "bible.index")
np.save("chunks.npy", chunks)

print("✅ FAISS index created successfully.")
