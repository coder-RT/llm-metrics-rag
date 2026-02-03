"""
Code Snippets for LLM Metrics

This package contains code snippets that are provided to candidates
during hackathon problems. Snippets are organized into subdirectories
by category.

Directory Structure:
    snippets/
    ├── __init__.py
    ├── math_operations/
    │   ├── add.py
    │   ├── subtract.py
    │   ├── multiply.py
    │   └── divide.py
    └── [other_categories]/
        └── [snippet_files].py
"""

from .snippet_loader import (
    SnippetLoader,
    Snippet,
    load_snippet,
    load_snippets_from_directory,
    list_available_snippets,
    load_all_snippets,
)

from .snippet_matcher import (
    SnippetMatcher,
    MatchResult,
    match_snippets,
    get_relevant_snippets,
)

# RAG-based retrieval (optional - requires chromadb and sentence-transformers)
try:
    from .rag_retriever import (
        RAGRetriever,
        RAGResult,
        get_retriever,
        index_all_snippets,
        rag_search,
        get_rag_snippets,
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    RAGRetriever = None
    RAGResult = None
    get_retriever = None
    index_all_snippets = None
    rag_search = None
    get_rag_snippets = None

__all__ = [
    "SnippetLoader",
    "Snippet",
    "load_snippet",
    "load_snippets_from_directory", 
    "list_available_snippets",
    "load_all_snippets",
    "SnippetMatcher",
    "MatchResult",
    "match_snippets",
    "get_relevant_snippets",
    # RAG exports
    "RAG_AVAILABLE",
    "RAGRetriever",
    "RAGResult",
    "get_retriever",
    "index_all_snippets",
    "rag_search",
    "get_rag_snippets",
]
