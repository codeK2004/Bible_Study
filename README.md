# 📖 Bible Study Assistant (RAG)

A **Retrieval-Augmented Generation (RAG)** based Bible study application that answers questions using **Bible text and trusted commentary**, ensuring accurate, non-hallucinated responses.  
The system combines **local embeddings + FAISS** for fast retrieval with **Gemini 2.5 Flash** for answer generation.

---

## ✨ Features
- Bible-specific question answering
- Uses external Bible commentary (not LLM memory)
- Hallucination-controlled responses
- Fast semantic search with FAISS
- Local embeddings (no API quota issues)
- Simple chat UI using Streamlit
- Runs fully on a local machine

---

## 🛠 Tech Stack
- **LLM:** Gemini 2.5 Flash  
- **Embeddings:** SentenceTransformers (all-MiniLM-L6-v2)  
- **Vector DB:** FAISS  
- **Backend:** Python  
- **PDF Parsing:** PyPDF2  
- **UI:** Streamlit  

---

## 📂 Project Structure
```text
Bible_Study/
├── data/
│   ├── bible.pdf
│   └── commentary.pdf
├── ingest.py
├── rag.py
├── app.py
├── .env
├── bible.index
└── chunks.npy ```


⚙️ Setup & Run
1️⃣ Add API key
Create a .env file:
env
Copy code
GEMINI_API_KEY=your_api_key_here
2️⃣ Install dependencies
bash
Copy code
python -m pip install faiss-cpu sentence-transformers torch PyPDF2 streamlit google-genai python-dotenv
3️⃣ Build the index (run once)
bash
Copy code
python ingest.py
4️⃣ Run the application
bash
Copy code
streamlit run app.py

👤 Author
Steff
AI & Generative AI Enthusiast

📜 License
Uses public-domain Bible and commentary texts.
Provided for educational and research purposes.

