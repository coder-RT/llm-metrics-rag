"""
Snippet Loader

This module provides functionality to load code snippets from files
for use in hackathon problems. Snippets can be loaded from individual
files or entire directories.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, field


# Default snippets directory
SNIPPETS_DIR = Path(__file__).parent


@dataclass
class Snippet:
    """
    Represents a code snippet loaded from a file.
    
    Attributes:
        name: Name of the snippet (filename without extension)
        path: Full path to the snippet file
        content: The actual code content
        category: Category/subdirectory the snippet belongs to
        language: Programming language (inferred from extension)
    """
    name: str
    path: Path
    content: str
    category: str = ""
    language: str = "python"
    
    def __str__(self) -> str:
        return self.content
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "content": self.content,
            "category": self.category,
            "language": self.language,
        }


class SnippetLoader:
    """
    Loads and manages code snippets from the snippets directory.
    
    Usage:
        loader = SnippetLoader()
        
        # Load a specific snippet
        snippet = loader.load("math_operations/add")
        
        # Load all snippets from a category
        snippets = loader.load_category("math_operations")
        
        # List available snippets
        available = loader.list_all()
    """
    
    def __init__(self, snippets_dir: Optional[Path] = None):
        """
        Initialize the snippet loader.
        
        Args:
            snippets_dir: Path to snippets directory.
                         Defaults to the 'snippets' directory in llmMetrics.
        """
        self.snippets_dir = snippets_dir or SNIPPETS_DIR
        self._cache: Dict[str, Snippet] = {}
    
    def load(self, snippet_path: str) -> Optional[Snippet]:
        """
        Load a single snippet by path.
        
        Args:
            snippet_path: Path relative to snippets directory.
                         Can be 'category/name' or just 'name.py'
                         
        Returns:
            Snippet object, or None if not found
            
        Example:
            >>> loader = SnippetLoader()
            >>> snippet = loader.load("math_operations/add")
            >>> print(snippet.content)
        """
        # Check cache first
        if snippet_path in self._cache:
            return self._cache[snippet_path]
        
        # Try different path variations
        possible_paths = [
            self.snippets_dir / f"{snippet_path}.py",
            self.snippets_dir / snippet_path,
            self.snippets_dir / f"{snippet_path}",
        ]
        
        for path in possible_paths:
            if path.exists() and path.is_file():
                snippet = self._load_file(path)
                if snippet:
                    self._cache[snippet_path] = snippet
                    return snippet
        
        return None
    
    def load_category(self, category: str) -> List[Snippet]:
        """
        Load all snippets from a category (subdirectory).
        
        Args:
            category: Name of the category/subdirectory
            
        Returns:
            List of Snippet objects
            
        Example:
            >>> loader = SnippetLoader()
            >>> snippets = loader.load_category("math_operations")
            >>> for s in snippets:
            ...     print(s.name)
        """
        category_dir = self.snippets_dir / category
        
        if not category_dir.exists() or not category_dir.is_dir():
            return []
        
        snippets = []
        for file_path in category_dir.glob("*.py"):
            if file_path.name.startswith("__"):
                continue  # Skip __init__.py etc.
            
            snippet = self._load_file(file_path, category=category)
            if snippet:
                snippets.append(snippet)
        
        return snippets
    
    def load_multiple(self, snippet_paths: List[str]) -> List[Snippet]:
        """
        Load multiple snippets by path.
        
        Args:
            snippet_paths: List of snippet paths
            
        Returns:
            List of successfully loaded Snippet objects
        """
        snippets = []
        for path in snippet_paths:
            snippet = self.load(path)
            if snippet:
                snippets.append(snippet)
        return snippets
    
    def get_content(self, snippet_path: str) -> str:
        """
        Get just the content of a snippet as a string.
        
        Args:
            snippet_path: Path to the snippet
            
        Returns:
            Snippet content, or empty string if not found
        """
        snippet = self.load(snippet_path)
        return snippet.content if snippet else ""
    
    def get_contents(self, snippet_paths: List[str]) -> List[str]:
        """
        Get contents of multiple snippets as strings.
        
        Args:
            snippet_paths: List of snippet paths
            
        Returns:
            List of snippet content strings
        """
        return [s.content for s in self.load_multiple(snippet_paths)]
    
    def list_all(self) -> Dict[str, List[str]]:
        """
        List all available snippets organized by category.
        
        Returns:
            Dictionary mapping category names to list of snippet names
        """
        result = {}
        
        for item in self.snippets_dir.iterdir():
            if item.is_dir() and not item.name.startswith(("_", ".")):
                category_snippets = []
                for file_path in item.glob("*.py"):
                    if not file_path.name.startswith("__"):
                        category_snippets.append(file_path.stem)
                if category_snippets:
                    result[item.name] = sorted(category_snippets)
            elif item.is_file() and item.suffix == ".py" and not item.name.startswith("_"):
                if "root" not in result:
                    result["root"] = []
                result["root"].append(item.stem)
        
        return result
    
    def list_categories(self) -> List[str]:
        """
        List available snippet categories.
        
        Returns:
            List of category names
        """
        categories = []
        for item in self.snippets_dir.iterdir():
            if item.is_dir() and not item.name.startswith(("_", ".")):
                categories.append(item.name)
        return sorted(categories)
    
    def load_all(self, extensions: Optional[List[str]] = None) -> List[Snippet]:
        """
        Load ALL snippets from all directories.
        
        Args:
            extensions: List of file extensions to include (e.g., [".py", ".js"])
                       If None, loads .py files by default
                       
        Returns:
            List of all Snippet objects from all categories
            
        Example:
            >>> loader = SnippetLoader()
            >>> all_snippets = loader.load_all([".py", ".js", ".jsx"])
        """
        if extensions is None:
            extensions = [".py"]
        
        # Normalize extensions (ensure they start with .)
        extensions = [ext if ext.startswith(".") else f".{ext}" for ext in extensions]
        
        all_snippets = []
        
        # Walk through all directories
        for root, dirs, files in os.walk(self.snippets_dir):
            # Skip hidden and private directories
            dirs[:] = [d for d in dirs if not d.startswith(("_", "."))]
            
            root_path = Path(root)
            
            for file_name in files:
                file_path = root_path / file_name
                
                # Skip non-matching extensions
                if file_path.suffix.lower() not in extensions:
                    continue
                
                # Skip private/hidden files
                if file_name.startswith(("_", ".")):
                    continue
                
                # Skip README, internal files, and non-code files
                skip_files = (
                    "readme.md", "readme.txt", "readme",
                    "snippet_loader.py", "snippet_matcher.py", "rag_retriever.py",  # Internal modules
                    "__init__.py",
                )
                if file_name.lower() in skip_files:
                    continue
                
                # Determine category from path
                try:
                    relative = file_path.relative_to(self.snippets_dir)
                    if len(relative.parts) > 1:
                        category = relative.parts[0]
                    else:
                        category = "root"
                except ValueError:
                    category = "root"
                
                # Load the snippet
                snippet = self._load_file(file_path, category=category)
                if snippet:
                    all_snippets.append(snippet)
        
        return all_snippets
    
    def _load_file(self, path: Path, category: str = "") -> Optional[Snippet]:
        """Load a snippet from a file path."""
        try:
            content = path.read_text(encoding="utf-8")
            
            # Infer category from path if not provided
            if not category:
                relative = path.relative_to(self.snippets_dir)
                if len(relative.parts) > 1:
                    category = relative.parts[0]
            
            # Infer language from extension
            ext_to_lang = {
                ".py": "python",
                ".js": "javascript",
                ".ts": "typescript",
                ".java": "java",
                ".go": "go",
                ".rs": "rust",
                ".cpp": "cpp",
                ".c": "c",
            }
            language = ext_to_lang.get(path.suffix, "text")
            
            return Snippet(
                name=path.stem,
                path=path,
                content=content,
                category=category,
                language=language,
            )
        except Exception as e:
            print(f"Error loading snippet {path}: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the snippet cache."""
        self._cache.clear()


# ============================================================================
# Convenience Functions
# ============================================================================

# Global loader instance
_default_loader: Optional[SnippetLoader] = None


def _get_loader() -> SnippetLoader:
    """Get or create the default snippet loader."""
    global _default_loader
    if _default_loader is None:
        _default_loader = SnippetLoader()
    return _default_loader


def load_snippet(path: str) -> Optional[Snippet]:
    """
    Load a single snippet by path.
    
    Args:
        path: Snippet path (e.g., "math_operations/add")
        
    Returns:
        Snippet object or None
    """
    return _get_loader().load(path)


def load_snippets_from_directory(category: str) -> List[Snippet]:
    """
    Load all snippets from a category directory.
    
    Args:
        category: Category name (subdirectory)
        
    Returns:
        List of Snippet objects
    """
    return _get_loader().load_category(category)


def list_available_snippets() -> Dict[str, List[str]]:
    """
    List all available snippets.
    
    Returns:
        Dictionary of category -> snippet names
    """
    return _get_loader().list_all()


def get_snippet_content(path: str) -> str:
    """
    Get snippet content as string.
    
    Args:
        path: Snippet path
        
    Returns:
        Snippet content string
    """
    return _get_loader().get_content(path)


def get_snippet_contents(paths: List[str]) -> List[str]:
    """
    Get multiple snippet contents as strings.
    
    Args:
        paths: List of snippet paths
        
    Returns:
        List of content strings
    """
    return _get_loader().get_contents(paths)


def load_all_snippets(extensions: Optional[List[str]] = None) -> List[Snippet]:
    """
    Load ALL snippets from all directories.
    
    Args:
        extensions: List of file extensions to include
        
    Returns:
        List of all Snippet objects
    """
    return _get_loader().load_all(extensions)
