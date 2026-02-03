# RAG System for arXiv

Retrieval-Augmented Generation (RAG) system for searching and generating answers based on arXiv scientific papers.

## Results

| Metric | Value |
|--------|-------|
| **MRR@5** | **0.9398** |
| Accuracy @1 | 91.2% |
| Average time | ~50 ms/query |
| Document base | 98,213 papers |

## Architecture

```
Query → Embedder → FAISS Search (top-50) → Reranker (top-5) → LLM → Answer
```

**Components:**
- **Embedder**: `all-MiniLM-L6-v2` (80MB) — semantic representation
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (130MB) — reranking
- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct` (3GB) — answer generation
- **Vector DB**: FAISS IndexFlatIP — cosine similarity

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexandraVanpaga/rag_arxiv.git
cd rag_arXiv

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# 1. Download data from Yandex.Disk
python -m src.get_raw_data

# 2. Data preprocessing
python -m src.preprocessing
```

### Running RAG

```bash
# 3. Building index and evaluation on test data
python -m src.rag_model
python -m src.init_and_eval

# 4. Generating answers with LLM
python -m src.generation
```

## Performance Analysis

### Distribution of Correct Answer Positions

| Position | Count | Percentage |
|----------|-------|------------|
| 1st place | 912 | 91.2% |
| 2nd place | 42 | 4.2% |
| 3rd place | 14 | 1.4% |
| 4th place | 7 | 0.7% |
| 5th place | 2 | 0.2% |
| Not found | 23 | 2.3% |
EOF
Salida

# RAG System for arXiv

Retrieval-Augmented Generation (RAG) system for searching and generating answers based on arXiv scientific papers.

## Results

| Metric | Value |
|--------|-------|
| **MRR@5** | **0.9398** |
| Accuracy @1 | 91.2% |
| Average time | ~50 ms/query |
| Document base | 98,213 papers |

## Architecture

```
Query → Embedder → FAISS Search (top-50) → Reranker (top-5) → LLM → Answer
```

**Components:**
- **Embedder**: `all-MiniLM-L6-v2` (80MB) — semantic representation
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (130MB) — reranking
- **LLM**: `Qwen/Qwen2.5-1.5B-Instruct` (3GB) — answer generation
- **Vector DB**: FAISS IndexFlatIP — cosine similarity

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/AlexandraVanpaga/rag_arxiv.git
cd rag_arXiv

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

```bash
# 1. Download data from Yandex.Disk
python -m src.get_raw_data

# 2. Data preprocessing
python -m src.preprocessing
```

### Running RAG

```bash
# 3. Building index and evaluation on test data
python -m src.rag_model
python -m src.init_and_eval

# 4. Generating answers with LLM
python -m src.generation
```

## Performance Analysis

### Distribution of Correct Answer Positions

| Position | Count | Percentage |
|----------|-------|------------|
| 1st place | 912 | 91.2% |
| 2nd place | 42 | 4.2% |
| 3rd place | 14 | 1.4% |
| 4th place | 7 | 0.7% |
| 5th place | 2 | 0.2% |
| Not found | 23 | 2.3% |
