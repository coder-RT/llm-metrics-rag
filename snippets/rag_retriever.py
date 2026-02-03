"""
RAG-based Snippet Retriever

Uses ChromaDB and sentence-transformers for semantic search
to find the most relevant code snippets for a user query.

This dramatically reduces token usage by only sending 1-2
relevant snippets instead of all snippets.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

# Lazy imports for optional dependencies
chromadb = None
SentenceTransformer = None


def _ensure_dependencies():
    """Lazily import RAG dependencies."""
    global chromadb, SentenceTransformer
    if chromadb is None:
        try:
            import chromadb as _chromadb
            chromadb = _chromadb
        except ImportError:
            raise ImportError(
                "chromadb is required for RAG. Install with: pip install chromadb"
            )
    if SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _ST
            SentenceTransformer = _ST
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for RAG. "
                "Install with: pip install sentence-transformers"
            )


@dataclass
class RAGResult:
    """Result from RAG retrieval."""
    name: str
    category: str
    content: str
    score: float  # Similarity score (higher = more relevant)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "content": self.content,
            "score": self.score,
            "source_path": f"snippets/{self.category}/{self.name}" if self.category else f"snippets/{self.name}"
        }


class RAGRetriever:
    """
    Retrieval-Augmented Generation for code snippets.
    
    Uses semantic search to find the most relevant snippets
    for a user query, dramatically reducing token usage.
    
    Usage:
        retriever = RAGRetriever()
        retriever.index_snippets(snippets)  # One-time indexing
        
        results = retriever.search("how to add numbers", top_k=2)
        # Returns only the 1-2 most relevant snippets
    """
    
    # Default embedding model (small and fast)
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        persist_directory: Optional[str] = None,
        collection_name: str = "code_snippets"
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            model_name: Sentence transformer model name
            persist_directory: Directory to persist the vector DB (optional)
            collection_name: Name of the ChromaDB collection
        """
        _ensure_dependencies()
        
        self.model_name = model_name
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize embedding model
        self._model = None
        
        # Initialize ChromaDB
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        self._indexed = False
    
    @property
    def model(self):
        """Lazy load the embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def _extract_searchable_content(self, name: str, category: str, content: str) -> str:
        """
        Extract the most meaningful parts of a snippet for embedding.
        
        This improves RAG matching by:
        1. Removing boilerplate headers/comments
        2. Emphasizing function names and docstrings
        3. Including keywords for better semantic matching
        """
        import re
        
        lines = content.split('\n')
        meaningful_parts = []
        
        # Add name and category with high weight (repeated for emphasis)
        meaningful_parts.append(f"{name} {name} {name}")
        if category and category != "root":
            meaningful_parts.append(f"{category}")
        
        # Extract function definitions and their docstrings
        in_docstring = False
        docstring_content = []
        
        for line in lines:
            stripped = line.strip()
            
            # Skip header comments (lines starting with # at the top)
            if stripped.startswith('#') and not meaningful_parts[1:]:
                continue
            
            # Capture function definitions
            if stripped.startswith('def ') or stripped.startswith('async def '):
                # Extract function name and signature
                func_match = re.match(r'(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)', stripped)
                if func_match:
                    func_name = func_match.group(1)
                    # Add function name multiple times for emphasis
                    meaningful_parts.append(f"{func_name} {func_name}")
            
            # Capture class definitions
            if stripped.startswith('class '):
                class_match = re.match(r'class\s+(\w+)', stripped)
                if class_match:
                    class_name = class_match.group(1)
                    meaningful_parts.append(f"{class_name} {class_name}")
            
            # Capture docstrings
            if '"""' in stripped or "'''" in stripped:
                quote = '"""' if '"""' in stripped else "'''"
                if stripped.count(quote) >= 2:
                    # Single line docstring
                    docstring = stripped.strip(quote).strip()
                    if docstring:
                        meaningful_parts.append(docstring)
                elif not in_docstring:
                    in_docstring = True
                    docstring_content = [stripped.replace(quote, '').strip()]
                else:
                    docstring_content.append(stripped.replace(quote, '').strip())
                    meaningful_parts.append(' '.join(docstring_content))
                    in_docstring = False
                    docstring_content = []
            elif in_docstring:
                docstring_content.append(stripped)
        
        result = ' '.join(meaningful_parts)
        return result if result.strip() else f"{name} {category} code snippet"
    
    def index_snippets(self, snippets: List[Any], force_reindex: bool = False) -> int:
        """
        Index snippets for semantic search.
        
        Args:
            snippets: List of Snippet objects (from snippet_loader)
            force_reindex: If True, clear and rebuild the index
            
        Returns:
            Number of snippets indexed
        """
        # Check if already indexed
        if self._collection.count() > 0 and not force_reindex:
            print(f"Collection already has {self._collection.count()} snippets. "
                  f"Use force_reindex=True to rebuild.")
            self._indexed = True
            return self._collection.count()
        
        # Clear existing if force reindex
        if force_reindex and self._collection.count() > 0:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        if not snippets:
            return 0
        
        print(f"Indexing {len(snippets)} snippets...")
        
        # Prepare data for indexing
        documents = []
        metadatas = []
        ids = []
        
        for snippet in snippets:
            # Extract meaningful content for indexing (function names, docstrings)
            # This improves matching by removing boilerplate headers
            searchable_text = self._extract_searchable_content(
                snippet.name, snippet.category, snippet.content
            )
            
            documents.append(searchable_text)
            metadatas.append({
                "name": snippet.name,
                "category": snippet.category or "root",
                "content": snippet.content,
                "language": getattr(snippet, 'language', 'python'),
            })
            # Include language/extension to make IDs unique (handles add.py vs add.js)
            lang = getattr(snippet, 'language', 'unknown')
            ids.append(f"{snippet.category or 'root'}/{snippet.name}.{lang}")
        
        # Generate embeddings
        embeddings = self.model.encode(documents, show_progress_bar=True)
        
        # Add to collection
        self._collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        self._indexed = True
        print(f"Indexed {len(snippets)} snippets successfully")
        
        return len(snippets)
    
    def search(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.3
    ) -> List[RAGResult]:
        """
        Search for relevant snippets using semantic similarity.
        
        Args:
            query: User's question/prompt
            top_k: Maximum number of results to return
            min_score: Minimum similarity score (0-1, higher = stricter)
            
        Returns:
            List of RAGResult objects, sorted by relevance
        """
        if self._collection.count() == 0:
            print("Warning: No snippets indexed. Call index_snippets() first.")
            return []
        
        # Generate query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Search
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["metadatas", "distances"]
        )
        
        # Process results
        rag_results = []
        
        if results and results['metadatas'] and results['metadatas'][0]:
            for i, metadata in enumerate(results['metadatas'][0]):
                # Convert distance to similarity score (ChromaDB returns distance)
                # For cosine distance: similarity = 1 - distance
                distance = results['distances'][0][i] if results['distances'] else 0
                similarity = 1 - distance
                
                if similarity >= min_score:
                    rag_results.append(RAGResult(
                        name=metadata.get('name', ''),
                        category=metadata.get('category', ''),
                        content=metadata.get('content', ''),
                        score=similarity
                    ))
        
        return rag_results
    
    def get_relevant_snippets(
        self,
        query: str,
        top_k: int = 2,
        min_score: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Get relevant snippets as dictionaries (for use in proxy).
        
        Args:
            query: User's question
            top_k: Max snippets to return
            min_score: Minimum similarity score
            
        Returns:
            List of snippet dictionaries with content and metadata
        """
        results = self.search(query, top_k, min_score)
        return [r.to_dict() for r in results]
    
    def count(self) -> int:
        """Return number of indexed snippets."""
        return self._collection.count()
    
    def clear(self) -> None:
        """Clear all indexed snippets."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._indexed = False


# ============================================================================
# Convenience Functions
# ============================================================================

_default_retriever: Optional[RAGRetriever] = None


def get_retriever(persist_dir: Optional[str] = None) -> RAGRetriever:
    """Get or create the default RAG retriever."""
    global _default_retriever
    if _default_retriever is None:
        # Use persistent storage in data directory
        if persist_dir is None:
            persist_dir = str(Path(__file__).parent.parent / "data" / "rag_index")
        _default_retriever = RAGRetriever(persist_directory=persist_dir)
    return _default_retriever


def index_all_snippets(force_reindex: bool = False) -> int:
    """
    Index all snippets from the snippets directory.
    
    Args:
        force_reindex: If True, rebuild the entire index
        
    Returns:
        Number of snippets indexed
    """
    from .snippet_loader import load_all_snippets
    
    retriever = get_retriever()
    snippets = load_all_snippets([".py", ".js", ".jsx", ".ts", ".tsx"])
    
    return retriever.index_snippets(snippets, force_reindex=force_reindex)


def rag_search(query: str, top_k: int = 2, min_score: float = 0.3) -> List[RAGResult]:
    """
    Search for relevant snippets using RAG.
    
    Args:
        query: User's question
        top_k: Max results
        min_score: Minimum similarity
        
    Returns:
        List of RAGResult objects
    """
    retriever = get_retriever()
    
    # Auto-index if needed
    if retriever.count() == 0:
        index_all_snippets()
    
    return retriever.search(query, top_k, min_score)


def get_rag_snippets(query: str, top_k: int = 2, min_score: float = 0.3) -> List[Dict]:
    """
    Get relevant snippets as dictionaries.
    
    Args:
        query: User's question
        top_k: Max results
        min_score: Minimum similarity
        
    Returns:
        List of snippet dictionaries
    """
    retriever = get_retriever()
    
    # Auto-index if needed
    if retriever.count() == 0:
        index_all_snippets()
    
    return retriever.get_relevant_snippets(query, top_k, min_score)
