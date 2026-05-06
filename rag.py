import requests
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from typing import List, Optional

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_NAME = "local-model"  # Placeholder; LM Studio ignores this
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # words
TEMPERATURE = 0.7
MAX_TOKENS = 500
TOP_K_RESULTS = 3

# ─────────────────────────────────────────────────────────────
# 1. LM STUDIO INTERFACE
# ─────────────────────────────────────────────────────────────
def ask_local_model(prompt: str, context: str = "") -> str:
    import time
    start = time.time()
    print(f"🔗 Connecting to LM Studio at {LM_STUDIO_URL}...")
    
    messages = []
    if context.strip():
        messages.append({
            "role": "system",
            "content": f"Answer using ONLY this context:\n\n{context}"
        })
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            LM_STUDIO_URL,
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": TEMPERATURE,
                "max_tokens": MAX_TOKENS
            },
            timeout=120  # Extended timeout
        )
        elapsed = time.time() - start
        print(f"✅ Response received in {elapsed:.1f}s")
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
        
    except requests.exceptions.Timeout:
        return "⚠️ Timeout: LM Studio took too long. Try a smaller model or increase timeout."
    except requests.exceptions.ConnectionError:
        return "⚠️ Connection refused: Is LM Studio running with server enabled on port 1234?"
    except Exception as e:
        return f"⚠️ Unexpected error: {type(e).__name__}: {e}"

# ─────────────────────────────────────────────────────────────
# 2. DOCUMENT PROCESSING
# ─────────────────────────────────────────────────────────────
def read_pdf(filepath: str) -> str:
    """Extract raw text from a PDF file."""
    doc = fitz.open(filepath)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into overlapping word-based chunks for better context retention."""
    words = text.split()
    chunks = []
    overlap = int(chunk_size * 0.1)  # 10% overlap to preserve context across boundaries
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks

# ─────────────────────────────────────────────────────────────
# 3. VECTOR INDEXING & SEARCH (ChromaDB + Sentence Transformers)
# ─────────────────────────────────────────────────────────────
class DocumentIndex:
    """Simple wrapper around ChromaDB for document embedding and retrieval."""
    
    def __init__(self, collection_name: str = "docs", embedding_model: str = EMBEDDING_MODEL):
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        if self.client.list_collections() and collection_name in [c.name for c in self.client.list_collections()]:
            self.client.delete_collection(collection_name)
        self.collection = self.client.create_collection(collection_name)
    
    def index_chunks(self, chunks: List[str]) -> None:
        """Embed and store document chunks."""
        if not chunks:
            return
        embeddings = self.model.encode(chunks, show_progress_bar=False).tolist()
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    
    def search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[str]:
        """Retrieve most relevant chunks for a query."""
        query_embedding = self.model.encode([query], show_progress_bar=False).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, len(self.collection.get()["ids"]))
        )
        return results["documents"][0] if results["documents"] else []

# ─────────────────────────────────────────────────────────────
# 4. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def answer_question_from_pdf(pdf_path: str, question: str) -> str:
    """
    End-to-end: load PDF → chunk → index → retrieve → generate answer.
    """
    print("📄 Reading PDF...")
    text = read_pdf(pdf_path)
    chunks = chunk_text(text)
    print(f"✂️  Split into {len(chunks)} chunks")
    
    print("🔍 Building vector index...")
    index = DocumentIndex()
    index.index_chunks(chunks)
    
    print("🎯 Searching for relevant context...")
    relevant_chunks = index.search(question)
    context = "\n\n---\n\n".join(relevant_chunks)
    
    print("🤖 Generating answer with local model...")
    answer = ask_local_model(question, context)
    
    return answer

# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 🔧 Change these two lines to use your own PDF and question
    PDF_FILE = r"C:\path\to\your\lecture_notes.pdf"
    QUESTION = "What is Newton's first law of motion?"
    
    print(f"\n🚀 StudyMind Core — Asking: '{QUESTION}'\n")
    result = answer_question_from_pdf(PDF_FILE, QUESTION)
    print(f"\n💡 Answer:\n{result}\n")
