"""
Snippet Matcher

Provides intelligent snippet selection based on user queries.
Only loads relevant snippets to reduce token usage.
"""

import re
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass
from .snippet_loader import Snippet, load_all_snippets, SnippetLoader


@dataclass
class MatchResult:
    """Result of snippet matching."""
    snippet: Snippet
    score: float  # 0.0 to 1.0
    matched_keywords: List[str]


class SnippetMatcher:
    """
    Matches user queries to relevant snippets.
    
    Uses keyword matching against:
    - Snippet names
    - Category names
    - Function/class names in content
    - Comments in content
    
    Usage:
        matcher = SnippetMatcher()
        relevant = matcher.match("how do I add two numbers?", max_snippets=5)
    """
    
    # Common stop words to ignore
    STOP_WORDS = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "each", "few",
        "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "just", "and", "but",
        "if", "or", "because", "until", "while", "this", "that", "these",
        "those", "what", "which", "who", "whom", "i", "me", "my", "myself",
        "we", "our", "you", "your", "he", "him", "she", "her", "it", "its",
        "they", "them", "their", "please", "help", "want", "like", "using",
        "use", "make", "create", "write", "code", "function", "class",
    }
    
    # Keywords that indicate specific operations
    OPERATION_KEYWORDS = {
        "add": ["add", "addition", "sum", "plus", "combine", "append"],
        "subtract": ["subtract", "subtraction", "minus", "difference", "remove"],
        "multiply": ["multiply", "multiplication", "times", "product"],
        "divide": ["divide", "division", "quotient", "split"],
        "sort": ["sort", "sorting", "order", "arrange"],
        "search": ["search", "find", "lookup", "locate", "query"],
        "filter": ["filter", "filtering", "select", "where"],
        "validate": ["validate", "validation", "check", "verify"],
        "auth": ["auth", "authenticate", "login", "logout", "password", "token"],
        "file": ["file", "read", "write", "open", "save", "load"],
        "api": ["api", "request", "response", "endpoint", "http", "rest"],
        "database": ["database", "db", "sql", "query", "insert", "update", "delete"],
        "ui": ["ui", "button", "form", "input", "modal", "component", "render"],
        "test": ["test", "testing", "assert", "mock", "spec"],
    }
    
    def __init__(self, extensions: Optional[List[str]] = None):
        """
        Initialize the matcher.
        
        Args:
            extensions: File extensions to include when loading snippets
        """
        self.extensions = extensions or [".py", ".js", ".jsx", ".ts", ".tsx"]
        self._snippets: Optional[List[Snippet]] = None
        self._snippet_keywords: Dict[str, Set[str]] = {}
    
    def _load_snippets(self) -> List[Snippet]:
        """Load all snippets (cached)."""
        if self._snippets is None:
            self._snippets = load_all_snippets(self.extensions)
            self._build_keyword_index()
        return self._snippets
    
    def _build_keyword_index(self) -> None:
        """Build keyword index for all snippets."""
        for snippet in self._snippets or []:
            keywords = self._extract_snippet_keywords(snippet)
            key = f"{snippet.category}/{snippet.name}"
            self._snippet_keywords[key] = keywords
    
    def _extract_snippet_keywords(self, snippet: Snippet) -> Set[str]:
        """Extract keywords from a snippet."""
        keywords = set()
        
        # Add name parts (split on underscore, camelCase)
        name_parts = self._split_identifier(snippet.name)
        keywords.update(name_parts)
        
        # Add category parts
        category_parts = self._split_identifier(snippet.category)
        keywords.update(category_parts)
        
        # Extract function/class names from content
        func_pattern = r'(?:def|function|class|const|let|var)\s+(\w+)'
        for match in re.finditer(func_pattern, snippet.content):
            func_name = match.group(1)
            keywords.update(self._split_identifier(func_name))
        
        # Extract from docstrings/comments (first line only for efficiency)
        comment_patterns = [
            r'"""([^"]+)"""',  # Python docstring
            r"'''([^']+)'''",  # Python docstring alt
            r'//\s*(.+)',      # JS single line comment
            r'#\s*(.+)',       # Python comment
            r'/\*\s*(.+?)\*/', # JS block comment
        ]
        for pattern in comment_patterns:
            for match in re.finditer(pattern, snippet.content[:500]):  # First 500 chars
                comment_words = match.group(1).lower().split()
                keywords.update(w for w in comment_words if len(w) > 2)
        
        # Remove stop words
        keywords = {k.lower() for k in keywords if k.lower() not in self.STOP_WORDS}
        
        return keywords
    
    def _split_identifier(self, identifier: str) -> List[str]:
        """Split an identifier into words (handles snake_case and camelCase)."""
        # Split on underscores
        parts = identifier.split("_")
        
        # Split camelCase
        result = []
        for part in parts:
            # Insert space before uppercase letters
            split = re.sub(r'([A-Z])', r' \1', part).split()
            result.extend(s.lower() for s in split if s)
        
        return [r for r in result if len(r) > 1]
    
    def _extract_query_keywords(self, query: str) -> Set[str]:
        """Extract keywords from user query."""
        # Lowercase and split
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove stop words
        keywords = {w for w in words if w not in self.STOP_WORDS and len(w) > 1}
        
        # Expand operation keywords
        expanded = set()
        for keyword in keywords:
            expanded.add(keyword)
            # Check if this keyword maps to an operation
            for op, synonyms in self.OPERATION_KEYWORDS.items():
                if keyword in synonyms:
                    expanded.add(op)
                    expanded.update(synonyms)
        
        return expanded
    
    def match(
        self,
        query: str,
        max_snippets: int = 5,
        min_score: float = 0.1
    ) -> List[MatchResult]:
        """
        Match query to relevant snippets.
        
        Args:
            query: User's question/prompt
            max_snippets: Maximum number of snippets to return
            min_score: Minimum match score (0.0 to 1.0)
            
        Returns:
            List of MatchResult sorted by relevance (highest first)
        """
        snippets = self._load_snippets()
        query_keywords = self._extract_query_keywords(query)
        
        if not query_keywords:
            return []
        
        results = []
        
        for snippet in snippets:
            key = f"{snippet.category}/{snippet.name}"
            snippet_keywords = self._snippet_keywords.get(key, set())
            
            if not snippet_keywords:
                continue
            
            # Calculate match score
            matched = query_keywords & snippet_keywords
            
            if matched:
                # Score is proportion of query keywords matched
                score = len(matched) / len(query_keywords)
                
                # Boost score if snippet name matches
                name_lower = snippet.name.lower()
                for kw in query_keywords:
                    if kw in name_lower:
                        score = min(1.0, score + 0.3)
                        break
                
                # Boost score if category matches
                category_lower = snippet.category.lower()
                for kw in query_keywords:
                    if kw in category_lower:
                        score = min(1.0, score + 0.2)
                        break
                
                if score >= min_score:
                    results.append(MatchResult(
                        snippet=snippet,
                        score=score,
                        matched_keywords=list(matched)
                    ))
        
        # Sort by score (highest first) and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:max_snippets]
    
    def get_relevant_snippets(
        self,
        query: str,
        max_snippets: int = 5,
        min_score: float = 0.1
    ) -> List[Snippet]:
        """
        Get relevant snippets for a query (simplified API).
        
        Args:
            query: User's question/prompt
            max_snippets: Maximum snippets to return
            min_score: Minimum match score
            
        Returns:
            List of relevant Snippet objects
        """
        results = self.match(query, max_snippets, min_score)
        return [r.snippet for r in results]
    
    def reload(self) -> None:
        """Reload snippets from disk."""
        self._snippets = None
        self._snippet_keywords.clear()
        self._load_snippets()


# ============================================================================
# Convenience Functions
# ============================================================================

_default_matcher: Optional[SnippetMatcher] = None


def _get_matcher() -> SnippetMatcher:
    """Get or create the default matcher."""
    global _default_matcher
    if _default_matcher is None:
        _default_matcher = SnippetMatcher()
    return _default_matcher


def match_snippets(
    query: str,
    max_snippets: int = 5,
    min_score: float = 0.1
) -> List[MatchResult]:
    """
    Match query to relevant snippets.
    
    Args:
        query: User's question
        max_snippets: Max snippets to return
        min_score: Minimum match score
        
    Returns:
        List of MatchResult
    """
    return _get_matcher().match(query, max_snippets, min_score)


def get_relevant_snippets(
    query: str,
    max_snippets: int = 5,
    min_score: float = 0.1
) -> List[Snippet]:
    """
    Get relevant snippets for a query.
    
    Args:
        query: User's question
        max_snippets: Max snippets to return
        
    Returns:
        List of Snippet objects
    """
    return _get_matcher().get_relevant_snippets(query, max_snippets, min_score)
