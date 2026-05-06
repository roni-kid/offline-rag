# 🤖 Offline RAG — Local AI Document Q&A

> The companion code for **"Run AI on Your Own PC"** — the ebook by JAR Studios.

Run AI on your own documents, completely offline. No API keys. No cloud. No subscription.

---

## What This Does

Point it at any PDF (lecture notes, textbooks, past questions) and ask questions in plain English. The script finds the most relevant sections and generates a grounded answer using your local LM Studio model.

This is the exact foundation that [StudyMind](https://github.com/roni-kid/StudyMine) is built on.

---

## Quick Start

### 1. Install dependencies
```bash
pip install requests PyMuPDF sentence-transformers chromadb
```

### 2. Start LM Studio
- Open **LM Studio** → load any model → go to **Local Server** → click **Start Server**
- Default port: `1234`

### 3. Run
Edit the two lines at the bottom of `rag.py`:
```python
PDF_FILE = r"C:\path\to\your\notes.pdf"
QUESTION = "What are the key concepts in Chapter 3?"
```

Then run:
```bash
python rag.py
```

---

## Example Output

```
🚀 Asking: 'What is Newton's first law of motion?'

📄 Reading PDF...
✂️  Split into 53 chunks
🔍 Building vector index...
🎯 Searching for relevant context...
🤖 Generating answer with local model...
🔗 Connecting to LM Studio at http://localhost:1234/v1/chat/completions...
✅ Response received in 2.1s

💡 Answer:
Newton's First Law (Law of Inertia) states that an object at rest stays
at rest and an object in motion continues moving at constant velocity
unless acted upon by an external force.
```

---

## Requirements

| Package | Purpose |
|---|---|
| `requests` | HTTP calls to LM Studio |
| `PyMuPDF` | PDF text extraction |
| `sentence-transformers` | Text → vector embeddings |
| `chromadb` | Vector similarity search |

---

## How It Works

1. **Read** — PyMuPDF extracts text from your PDF page by page
2. **Chunk** — Text is split into 500-word pieces with 10% overlap
3. **Index** — `all-MiniLM-L6-v2` encodes each chunk into a vector, stored in ChromaDB
4. **Retrieve** — Your question is encoded and the 3 closest chunks are found
5. **Generate** — Those chunks + your question are sent to your local LM Studio model

---

## Tuning

| Variable | Default | Change if... |
|---|---|---|
| `CHUNK_SIZE` | 500 | Answers are vague → lower to 300 |
| `TOP_K_RESULTS` | 3 | Missing context → raise to 5 |
| `MAX_TOKENS` | 500 | Answers cut off → raise to 800 |
| `TEMPERATURE` | 0.7 | Too creative → lower to 0.3 |

---

## Part of the JAR Studios Ebook Series

**"Run AI on Your Own PC"** — full setup guide, model recommendations, and 5 real offline AI workflows.

Available on [Gumroad](https://gumroad.com)

---

*Built by RoniKid · Computer Engineering · GCTU*
