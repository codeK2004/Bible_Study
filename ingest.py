import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from bible_parser import parse_bible

# ---------------- PDF LOADER ----------------
def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

# ---------------- CANONICAL BOOK NORMALIZER ----------------
def canonical_book(book: str) -> str:
    book = book.lower().strip()

    if book in ["psalm", "psalms"]:
        return "psalms"

    # strip common prefixes
    book = book.replace("the gospel according to", "").strip()
    book = book.replace("st.", "").strip()
    book = book.replace("saint", "").strip()

    return book

# ---------------- MAIN ----------------
def main():
    print("📖 Loading Bible PDF...")
    raw = load_pdf("data/bible.pdf")

    print("🧠 Parsing Bible...")
    verses = parse_bible(raw)

    bible_chunks = []
    for v in verses:
        book = canonical_book(v["book"])
        bible_chunks.append(
            f"{book}|{v['chapter']}|{v['verse']}|{v['text']}"
        )

    print(f"✅ Parsed {len(bible_chunks)} verses")

    print("🔢 Creating embeddings...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(
        bible_chunks,
        batch_size=64,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")

    print("📦 Building FAISS index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "bible.index")
    np.save("bible_chunks.npy", np.array(bible_chunks, dtype=object))

    print("🎉 Ingestion complete")

if __name__ == "__main__":
    main()
