import os
import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai

# Load env
load_dotenv()

# Gemini client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Load local embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load FAISS + chunks
index = faiss.read_index("bible.index")
chunks = np.load("chunks.npy", allow_pickle=True)

def retrieve(query, k=8):
    q_vec = embedder.encode([query])
    q_vec = np.array(q_vec).astype("float32")
    _, idx = index.search(q_vec, k)
    return [chunks[i] for i in idx[0]]

def ask(question):
    question = question.strip().lower()
    retrieved_chunks = retrieve(question, k=10)
    context = "\n".join(retrieve(question))

    prompt = f"""
You are a Bible study assistant.

You are allowed to:
1. Explain Bible verses found in the context.
2. Summarize biblical narratives if multiple verses are present.
3. Use commentary ONLY for interpretation and explanation.

Rules:
- Do NOT introduce modern opinions.
- Do NOT use non-biblical sources.
- If Scripture speaks clearly, explain it plainly.
- If the topic is not addressed in the provided Scripture at all, say so honestly.

Context (Bible verses + commentary):
{context}

Question:
{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    print("\nAnswer:\n", response.text)

# -------- INTERACTIVE LOOP --------
while True:
    q = input("\nAsk a Bible question (type 'exit' to quit): ")
    if q.lower() == "exit":
        print("Goodbye 🙏")
        break
    ask(q)
