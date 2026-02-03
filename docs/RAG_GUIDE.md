# RAG-based Snippet Retrieval Guide

This guide explains how the RAG (Retrieval-Augmented Generation) system works in llmMetricsRAG for semantic snippet retrieval.

---

## Table of Contents

- [Overview](#overview)
- [How RAG Works](#how-rag-works)
- [Architecture](#architecture)
- [Setup & Configuration](#setup--configuration)
- [API Reference](#api-reference)
- [Managing Snippets](#managing-snippets)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Overview

### The Problem

When using snippet-grounded mode, sending ALL code snippets with every request wastes tokens:

| Approach | Snippets Sent | Tokens per Request |
|----------|---------------|-------------------|
| All snippets | 50+ | ~35,000 |
| Keyword matching | 3-5 | ~2,000 |
| **RAG (semantic)** | 1-2 | **~500** |

### The Solution

RAG uses **semantic search** with vector embeddings to find only the most relevant snippets for each user query.

**Token reduction: 95%+**

---

## How RAG Works

### Step-by-Step Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  1. INDEXING (One-time)                                         │
│                                                                  │
│  Each snippet → Sentence Transformer → Vector Embedding         │
│                                                                  │
│  add.py      → [0.12, 0.45, -0.23, ...]                        │
│  subtract.py → [0.08, 0.41, -0.19, ...]                        │
│  auth.py     → [-0.34, 0.67, 0.12, ...]                        │
│                                                                  │
│  All embeddings stored in ChromaDB                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. QUERY TIME                                                   │
│                                                                  │
│  User asks: "How do I add two numbers?"                         │
│                    │                                             │
│                    ▼                                             │
│  Query → Sentence Transformer → Query Embedding                  │
│          [0.11, 0.44, -0.21, ...]                               │
│                    │                                             │
│                    ▼                                             │
│  Compare query embedding to all snippet embeddings              │
│  using cosine similarity                                         │
│                    │                                             │
│                    ▼                                             │
│  Results:                                                        │
│    add.py      → similarity: 0.92 ✓ (match!)                   │
│    subtract.py → similarity: 0.45 ✓ (match!)                   │
│    auth.py     → similarity: 0.12 ✗ (not relevant)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. INJECTION                                                    │
│                                                                  │
│  Only top 2 snippets (add.py, subtract.py) are injected        │
│  into the LLM prompt instead of all 50+ snippets                │
│                                                                  │
│  Token savings: 35,000 → 500 (98% reduction!)                   │
└─────────────────────────────────────────────────────────────────┘
```

### Semantic Understanding

Unlike keyword matching, RAG understands **meaning**:

| Query | Keyword Match | RAG Match |
|-------|---------------|-----------|
| "sum two values" | ❌ No match (no "sum" in snippets) | ✅ Finds `add.py` |
| "combine numbers" | ❌ No match | ✅ Finds `add.py` |
| "authentication" | ✅ Finds `auth.py` | ✅ Finds `auth.py` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Cline / User                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Proxy Server (FastAPI)                        │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Request      │───▶│ RAG          │───▶│ Grounding    │      │
│  │ Handler      │    │ Retriever    │    │ Injector     │      │
│  └──────────────┘    └──────┬───────┘    └──────────────┘      │
│                             │                                    │
│                             ▼                                    │
│                    ┌──────────────────┐                         │
│                    │ ChromaDB         │                         │
│                    │ (Vector Store)   │                         │
│                    └──────────────────┘                         │
│                             │                                    │
│                    ┌──────────────────┐                         │
│                    │ Sentence         │                         │
│                    │ Transformer      │                         │
│                    │ (Embeddings)     │                         │
│                    └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                    ┌──────────────────┐
                    │ LLM API          │
                    │ (OpenAI/Claude)  │
                    └──────────────────┘
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Vector Store | ChromaDB | Stores snippet embeddings |
| Embeddings | Sentence Transformers | Converts text to vectors |
| Model | all-MiniLM-L6-v2 | Fast, accurate embeddings |

---

## Setup & Configuration

### 1. Install Dependencies

```bash
cd llmMetricsRAG
pip install -r proxy/requirements.txt

# This installs:
# - chromadb>=0.4.0
# - sentence-transformers>=2.2.0
```

### 2. Configure RAG

In `config.yaml`:

```yaml
snippet_grounded_mode:
  # Enable RAG (semantic search)
  use_rag: true
  
  # Maximum snippets to return (1-2 is usually enough)
  max_snippets: 2
  
  # Minimum similarity score (0.0 to 1.0)
  # Higher = stricter matching
  min_match_score: 0.3
```

### 3. Start Server

```bash
cd /path/to/parent
PYTHONPATH=$(pwd) python3 -m uvicorn llmMetricsRAG.proxy.server:app --port 8000
```

### 4. Index Snippets

```bash
# First-time indexing
curl -X POST http://localhost:8000/api/rag/index

# Expected output:
# {"status":"success","indexed_snippets":42,"message":"Indexed 42 snippets"}
```

---

## API Reference

### Index Snippets

```bash
POST /api/rag/index?force=false
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `force` | `false` | If `true`, rebuilds entire index |

**Response:**
```json
{
  "status": "success",
  "indexed_snippets": 42,
  "message": "Indexed 42 snippets for RAG search"
}
```

### Search Snippets

```bash
GET /api/rag/search?query=add+numbers&top_k=2&min_score=0.3
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `query` | required | Search query |
| `top_k` | `2` | Max results |
| `min_score` | `0.3` | Min similarity |

**Response:**
```json
{
  "query": "add numbers",
  "results": [
    {
      "name": "add",
      "category": "math_operations",
      "score": 0.892,
      "content_preview": "def add(a: int, b: int) -> int:..."
    }
  ],
  "total_results": 1
}
```

### Check Status

```bash
GET /api/rag/status
```

**Response:**
```json
{
  "available": true,
  "enabled": true,
  "indexed_snippets": 42,
  "model": "all-MiniLM-L6-v2",
  "max_snippets": 2,
  "min_score": 0.3
}
```

---

## Managing Snippets

### Adding New Snippets

1. Add file to `snippets/` directory:
   ```
   snippets/
   └── new_category/
       └── new_snippet.py
   ```

2. Re-index:
   ```bash
   curl -X POST "http://localhost:8000/api/rag/index?force=true"
   ```

### Modifying Snippets

1. Edit the snippet file
2. Re-index:
   ```bash
   curl -X POST "http://localhost:8000/api/rag/index?force=true"
   ```

### Removing Snippets

1. Delete the file from `snippets/`
2. Re-index:
   ```bash
   curl -X POST "http://localhost:8000/api/rag/index?force=true"
   ```

### Automation

#### On Server Startup
```bash
#!/bin/bash
python3 -m uvicorn llmMetricsRAG.proxy.server:app --port 8000 &
sleep 5
curl -X POST "http://localhost:8000/api/rag/index?force=true"
```

#### Cron Job (Hourly)
```bash
0 * * * * curl -X POST "http://localhost:8000/api/rag/index?force=true"
```

---

## Testing

### 1. Start the Server

```bash
cd /path/to/parent
PYTHONPATH=$(pwd) python3 -m uvicorn llmMetricsRAG.proxy.server:app --port 8000
```

### 2. Check RAG Status

```bash
curl http://localhost:8000/api/rag/status | python3 -m json.tool
```

Expected output:
```json
{
  "available": true,
  "enabled": true,
  "indexed_snippets": 4
}
```

### 3. Index Snippets

```bash
curl -X POST http://localhost:8000/api/rag/index | python3 -m json.tool
```

### 4. Test Semantic Search

```bash
# Should find add.py
curl "http://localhost:8000/api/rag/search?query=sum+two+values" | python3 -m json.tool

# Should find divide.py
curl "http://localhost:8000/api/rag/search?query=division+operation" | python3 -m json.tool

# Should find nothing (unrelated query)
curl "http://localhost:8000/api/rag/search?query=database+connection" | python3 -m json.tool
```

### 5. Test with Cline

In Cline settings:
```
API Provider: OpenAI Compatible
Base URL: http://localhost:8000/v1
Custom Headers:
  X-Assistance-Mode: snippet
  X-Candidate-ID: test-user
```

Then ask: "How do I add two numbers?"

Check that only relevant snippets are used:
```bash
# Check metrics
curl http://localhost:8000/api/metrics/overview | python3 -m json.tool
```

### 6. Compare Token Usage

```bash
# Generate comparison report
curl -X POST http://localhost:8000/api/metrics/report | python3 -m json.tool
```

---

## Troubleshooting

### RAG Not Available

**Error:** `"available": false, "message": "RAG dependencies not installed"`

**Solution:**
```bash
pip install chromadb sentence-transformers
```

### No Snippets Indexed

**Error:** `"indexed_snippets": 0`

**Solution:**
```bash
curl -X POST "http://localhost:8000/api/rag/index?force=true"
```

### Slow First Request

**Cause:** Embedding model loads on first use (~5-10 seconds)

**Solution:** Pre-warm by calling index endpoint after server start

### Wrong Snippets Returned

**Cause:** Low similarity scores or stale index

**Solution:**
1. Increase `min_match_score` in config
2. Re-index: `curl -X POST "...?force=true"`

### Index Persistence

RAG index is stored at: `llmMetricsRAG/data/rag_index/`

To completely reset:
```bash
rm -rf llmMetricsRAG/data/rag_index/
curl -X POST "http://localhost:8000/api/rag/index"
```

---

## Python API

```python
from llmMetricsRAG.snippets import (
    RAGRetriever,
    index_all_snippets,
    rag_search,
    get_rag_snippets,
)

# Index all snippets
count = index_all_snippets(force_reindex=True)
print(f"Indexed {count} snippets")

# Search
results = rag_search("how to add numbers", top_k=2)
for r in results:
    print(f"{r.name}: {r.score:.2f}")

# Get as dicts (for injection)
snippets = get_rag_snippets("add numbers")
print(snippets)
```

---

## Performance

| Operation | Time |
|-----------|------|
| Index 50 snippets | ~10 seconds |
| Search query | ~50ms |
| Embedding model load | ~5 seconds (first time) |

### Token Savings

| Scenario | Without RAG | With RAG | Savings |
|----------|-------------|----------|---------|
| Single query | 35,000 | 500 | 98% |
| 7-turn conversation | 249,000 | 3,500 | 99% |
| 100 users/day | 3.5M | 50K | 98% |

---

## Summary

1. **Index once:** `curl -X POST .../api/rag/index`
2. **Re-index when snippets change:** `curl -X POST .../api/rag/index?force=true`
3. **Automatic in requests:** RAG finds relevant snippets automatically
4. **Massive token savings:** 95-99% reduction

The user experience stays the same - they just ask questions and get snippet-grounded answers with dramatically fewer tokens!
