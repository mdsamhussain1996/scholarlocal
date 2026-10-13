# ScholarLocal 📚🔬

> **A Private, 100% Offline Research-Paper Agent for Mathematics & Control Theory**  
> Covers fractional calculus, neural network synchronization, controllability, and dynamical systems.

ScholarLocal acts as an offline research assistant over your personal PDF library. It indexes local mathematics papers, provides citations with exact paper titles and page numbers (`[AuthorYear, p. X]`), extracts theorems and assumptions into structured JSON schemas, and drafts LaTeX text with authentic BibTeX keys.

**Zero Cloud Leakage:** All embeddings and LLM inference run locally on your machine via Ollama. No document content ever leaves your hardware.

---

## 🚀 Key Features

- **Local Inference via Ollama**: Supports configurable hardware tiers:
  - `small` (8 GB RAM): `llama3.2:1b` or `qwen2.5:3b`
  - `medium` (16 GB RAM / GPU): `qwen2.5:7b` or `llama3.1:8b`
  - `large` (32 GB RAM / 12 GB+ GPU): `qwen2.5:14b`
  - Embeddings: `nomic-embed-text`
- **Math-Specific Ingestion**:
  - Page-by-page text parsing via PyMuPDF.
  - Section-aware chunking (~500–800 tokens) that strictly preserves theorem, lemma, definition, and displayed equation boundaries.
  - Automatic BibTeX matching via DOI and fuzzy title matching (`rapidfuzz`).
- **Structured Theorem Extraction**:
  - Detects `Theorem`, `Lemma`, `Proposition`, `Corollary`, `Definition`, `Assumption`, and `Remark` blocks.
  - Extracts structured mathematical parameters: system type, fractional derivative, order range, control type, delays, result type, key conditions, and method.
- **Dense Vector & Hybrid Retrieval**:
  - Persistent ChromaDB vector store.
  - Exact term keyword matching (BM25) and Reciprocal Rank Fusion (RRF).
- **Interactive CLI & Rich Visual Output**:
  - `scholar status`: Check Ollama connection and database stats.
  - `scholar index`: Ingest, deduplicate, chunk, and embed library PDFs.
  - `scholar search "<query>"`: Fast vector similarity search with page citations and section markers.

---

## 🛠️ Installation & Setup

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/mdsamhussain1996/scholarlocal.git
cd scholarlocal

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Pull Ollama Models (Local)
Ensure [Ollama](https://ollama.ai) is installed and running:
```bash
# Pull local embedding model
ollama pull nomic-embed-text

# Pull LLM based on your hardware tier:
# Small Tier (8 GB RAM):
ollama pull llama3.2:1b

# Medium Tier (16 GB RAM / GPU):
ollama pull qwen2.5:7b
```

---

## 📖 Quickstart CLI

### Check System Status
```bash
python -m src.cli status
```

### Index Your PDF Library
Place your research PDFs and `.bib` files in `library/`:
```bash
python -m src.cli index
```

### Search Passages with Citations
```bash
python -m src.cli search "Prabhakar Mittag-Leffler controllability" --top-k 3
python -m src.cli search "sliding mode finite time synchronization" --top-k 3
```

---

## 📁 Project Structure

```
scholarlocal/
├── config.yaml                    # Hardware tiers, model paths, chunking params
├── generate_sample_library.py     # Sample PDF generator for research papers
├── library/                       # Research PDFs and references.bib
│   ├── references.bib
│   ├── ansari2024prabhakar.pdf
│   ├── ansari2024competitive.pdf
│   ├── ansari2024sliding.pdf
│   ├── ansari2024quasiproj.pdf
│   └── ansari2025multiterm.pdf
├── prompts/                       # External LLM prompt templates
│   ├── metadata_extraction.txt
│   └── theorem_summary.txt
├── src/
│   ├── config.py                  # Pydantic configuration loader
│   ├── llm.py                     # Ollama wrapper with retries and local embeddings
│   ├── storage.py                 # SQLite persistence (papers, chunks, theorems)
│   ├── cli.py                     # Typer CLI (status, index, search)
│   ├── ingest/                    # Parser, chunker, metadata, theorem extraction
│   │   ├── parser.py
│   │   ├── metadata.py
│   │   ├── chunker.py
│   │   └── theorems.py
│   └── index/                     # Vector store and retrieval
│       └── vector_store.py
└── tests/                         # Unit tests
    └── test_step1.py
```

---

## 🧪 Testing

Run automated tests:
```bash
pytest tests/
```

---

## 🔒 Privacy & Grounding Guarantee
ScholarLocal runs completely offline. All document processing, vector embeddings, and mathematical reasoning occur on your local device. The arXiv network feature is disabled by default and strictly isolated.

---

## 👨‍🏫 Author & Academic Affiliation

**Dr. Md Samshad Hussain Ansari**  
Assistant Professor of Mathematics, Newton School of Technology, ADYPU, Pune  
Ph.D. in Mathematics, Indian Institute of Technology Mandi (2025)  
- 🌐 Academic Portfolio: [https://mdsamhussain1996.github.io/](https://mdsamhussain1996.github.io/)  
- 📚 Google Scholar: [Md Samshad Ansari](https://scholar.google.com/citations?user=3b2jd4EAAAAJ)  
- 🔬 ResearchGate: [Md-Samshad-Ansari](https://www.researchgate.net/profile/Md-Samshad-Ansari)  
- 💻 GitHub: [@mdsamhussain1996](https://github.com/mdsamhussain1996)

