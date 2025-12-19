**Bible Study Assistant (RAG)**
A Retrieval-Augmented Generation (RAG) based Bible study application that answers questions using Bible text and trusted commentary, ensuring accurate and non-hallucinated responses. 
The system uses local embeddings + FAISS for fast retrieval and Gemini 2.5 Flash for answer generation.

**Tech Stack**
Gemini 2.5 Flash (LLM)
SentenceTransformers (Local Embeddings)
FAISS (Vector Search)
Python
PyPDF2
Streamlit (UI)

**Setup & Run**
Add API key in .env
GEMINI_API_KEY=your_api_key_here


Install dependencies
python -m pip install faiss-cpu sentence-transformers torch PyPDF2 streamlit google-genai python-dotenv


Build index (run once)
python ingest.py


Run the app
streamlit run app.py

**License**
This project uses public-domain Bible and commentary texts.
Code is provided for educational and research purposes.
