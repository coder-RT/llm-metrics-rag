"""
Configuration Store for Hackathon Problems

Manages the toggle state (snippet_grounded vs free_form) and
code snippets for each problem. Supports SQLite storage for
persistence across server restarts.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from contextlib import contextmanager


@dataclass
class SnippetInfo:
    """
    Information about a code snippet including its source.
    
    Attributes:
        content: The actual code content
        source_path: Path to the source file (e.g., "snippets/math_operations/add.py")
        name: Name of the snippet (e.g., "add")
        category: Category/subdirectory (e.g., "math_operations")
    """
    content: str
    source_path: str = ""
    name: str = ""
    category: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "source_path": self.source_path,
            "name": self.name,
            "category": self.category,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnippetInfo":
        if isinstance(data, str):
            return cls(content=data)
        return cls(
            content=data.get("content", ""),
            source_path=data.get("source_path", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
        )


@dataclass
class ProblemConfig:
    """
    Configuration for a single hackathon problem.
    
    Attributes:
        problem_id: Unique identifier for the problem
        title: Human-readable problem title
        mode: 'snippet_grounded' or 'free_form'
        snippets: List of code snippets with source information
        snippet_metadata: List of snippet metadata (source paths, names)
        description: Problem statement/description
        created_at: When the problem was created
        updated_at: When the config was last updated
    """
    problem_id: str
    title: str = ""
    mode: str = "snippet_grounded"  # or "free_form"
    snippets: List[str] = field(default_factory=list)  # Raw content for backward compat
    snippet_metadata: List[SnippetInfo] = field(default_factory=list)  # With source info
    description: str = ""
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def is_grounded(self) -> bool:
        """Check if problem is in snippet-grounded mode."""
        return self.mode == "snippet_grounded"
    
    def get_snippets_with_sources(self) -> List[Dict[str, Any]]:
        """Get snippets with their source information."""
        if self.snippet_metadata:
            return [s.to_dict() for s in self.snippet_metadata]
        # Fallback to plain snippets without source info
        return [{"content": s, "source_path": "", "name": ""} for s in self.snippets]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem_id": self.problem_id,
            "title": self.title,
            "mode": self.mode,
            "snippets": self.snippets,
            "snippet_sources": [s.to_dict() for s in self.snippet_metadata] if self.snippet_metadata else [],
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class ConfigStore:
    """
    SQLite-based configuration store for hackathon problems.
    
    Stores problem configurations including:
    - Toggle state (snippet_grounded vs free_form)
    - Code snippets to provide to candidates
    - Problem metadata
    
    Usage:
        store = ConfigStore("./config.db")
        
        # Create a problem
        store.create_problem(
            problem_id="binary_search",
            title="Implement Binary Search",
            mode="snippet_grounded",
            snippets=["def binary_search(arr, target): pass"]
        )
        
        # Get problem config
        config = store.get_problem("binary_search")
        
        # Toggle mode
        store.set_mode("binary_search", "free_form")
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the config store.
        
        Args:
            db_path: Path to SQLite database. Defaults to './hackathon_config.db'
        """
        self.db_path = db_path or Path("./hackathon_config.db")
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Problems table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS problems (
                    problem_id TEXT PRIMARY KEY,
                    title TEXT,
                    mode TEXT DEFAULT 'snippet_grounded',
                    snippets TEXT,
                    snippet_metadata TEXT,
                    description TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Candidate assignments (which problem each candidate is working on)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidate_assignments (
                    candidate_id TEXT PRIMARY KEY,
                    problem_id TEXT,
                    assigned_at TEXT,
                    FOREIGN KEY (problem_id) REFERENCES problems(problem_id)
                )
            """)
            
            # Global settings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
    
    # ====================
    # PROBLEM MANAGEMENT
    # ====================
    
    def create_problem(
        self,
        problem_id: str,
        title: str = "",
        mode: str = "snippet_grounded",
        snippets: Optional[List[str]] = None,
        snippet_metadata: Optional[List[SnippetInfo]] = None,
        description: str = ""
    ) -> ProblemConfig:
        """
        Create a new problem configuration.
        
        Args:
            problem_id: Unique identifier
            title: Problem title
            mode: 'snippet_grounded' or 'free_form'
            snippets: List of code snippets (raw content)
            snippet_metadata: List of SnippetInfo with source paths
            description: Problem description
            
        Returns:
            Created ProblemConfig
        """
        now = datetime.utcnow().isoformat()
        snippets = snippets or []
        snippet_metadata = snippet_metadata or []
        
        # Serialize snippet metadata
        metadata_json = json.dumps([s.to_dict() for s in snippet_metadata])
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO problems 
                (problem_id, title, mode, snippets, snippet_metadata, description, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                problem_id,
                title,
                mode,
                json.dumps(snippets),
                metadata_json,
                description,
                now,
                now,
            ))
        
        return ProblemConfig(
            problem_id=problem_id,
            title=title,
            mode=mode,
            snippets=snippets,
            snippet_metadata=snippet_metadata,
            description=description,
        )
    
    def get_problem(self, problem_id: str) -> Optional[ProblemConfig]:
        """
        Get a problem configuration.
        
        Args:
            problem_id: Problem identifier
            
        Returns:
            ProblemConfig or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM problems WHERE problem_id = ?",
                (problem_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Parse snippet metadata if available
            snippet_metadata = []
            try:
                metadata_raw = row["snippet_metadata"] if "snippet_metadata" in row.keys() else None
                if metadata_raw:
                    metadata_list = json.loads(metadata_raw)
                    snippet_metadata = [SnippetInfo.from_dict(m) for m in metadata_list]
            except (json.JSONDecodeError, KeyError):
                pass
            
            return ProblemConfig(
                problem_id=row["problem_id"],
                title=row["title"] or "",
                mode=row["mode"] or "snippet_grounded",
                snippets=json.loads(row["snippets"] or "[]"),
                snippet_metadata=snippet_metadata,
                description=row["description"] or "",
                created_at=datetime.fromisoformat(row["created_at"]) if row["created_at"] else datetime.utcnow(),
                updated_at=datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else datetime.utcnow(),
            )
    
    def list_problems(self) -> List[ProblemConfig]:
        """Get all problem configurations."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM problems ORDER BY created_at DESC")
            rows = cursor.fetchall()
            
            return [
                ProblemConfig(
                    problem_id=row["problem_id"],
                    title=row["title"] or "",
                    mode=row["mode"] or "snippet_grounded",
                    snippets=json.loads(row["snippets"] or "[]"),
                    description=row["description"] or "",
                )
                for row in rows
            ]
    
    def delete_problem(self, problem_id: str) -> bool:
        """Delete a problem configuration."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM problems WHERE problem_id = ?",
                (problem_id,)
            )
            return cursor.rowcount > 0
    
    # ====================
    # MODE TOGGLE
    # ====================
    
    def set_mode(self, problem_id: str, mode: str) -> bool:
        """
        Set the assistance mode for a problem.
        
        Args:
            problem_id: Problem identifier
            mode: 'snippet_grounded' or 'free_form'
            
        Returns:
            True if updated, False if problem not found
        """
        if mode not in ("snippet_grounded", "free_form"):
            raise ValueError(f"Invalid mode: {mode}")
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE problems 
                SET mode = ?, updated_at = ?
                WHERE problem_id = ?
            """, (mode, datetime.utcnow().isoformat(), problem_id))
            return cursor.rowcount > 0
    
    def toggle_mode(self, problem_id: str) -> Optional[str]:
        """
        Toggle between snippet_grounded and free_form modes.
        
        Args:
            problem_id: Problem identifier
            
        Returns:
            New mode, or None if problem not found
        """
        config = self.get_problem(problem_id)
        if not config:
            return None
        
        new_mode = "free_form" if config.mode == "snippet_grounded" else "snippet_grounded"
        self.set_mode(problem_id, new_mode)
        return new_mode
    
    # ====================
    # SNIPPET MANAGEMENT
    # ====================
    
    def set_snippets(self, problem_id: str, snippets: List[str]) -> bool:
        """
        Set the code snippets for a problem.
        
        Args:
            problem_id: Problem identifier
            snippets: List of code snippet strings
            
        Returns:
            True if updated
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE problems 
                SET snippets = ?, updated_at = ?
                WHERE problem_id = ?
            """, (json.dumps(snippets), datetime.utcnow().isoformat(), problem_id))
            return cursor.rowcount > 0
    
    def add_snippet(self, problem_id: str, snippet: str) -> bool:
        """Add a snippet to a problem's snippet list."""
        config = self.get_problem(problem_id)
        if not config:
            return False
        
        config.snippets.append(snippet)
        return self.set_snippets(problem_id, config.snippets)
    
    # ====================
    # CANDIDATE ASSIGNMENT
    # ====================
    
    def assign_candidate(self, candidate_id: str, problem_id: str) -> None:
        """Assign a candidate to a problem."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO candidate_assignments
                (candidate_id, problem_id, assigned_at)
                VALUES (?, ?, ?)
            """, (candidate_id, problem_id, datetime.utcnow().isoformat()))
    
    def get_candidate_problem(self, candidate_id: str) -> Optional[str]:
        """Get the problem ID assigned to a candidate."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT problem_id FROM candidate_assignments WHERE candidate_id = ?",
                (candidate_id,)
            )
            row = cursor.fetchone()
            return row["problem_id"] if row else None
    
    def get_candidate_config(self, candidate_id: str) -> Optional[ProblemConfig]:
        """Get the problem config for a candidate's assigned problem."""
        problem_id = self.get_candidate_problem(candidate_id)
        if not problem_id:
            return None
        return self.get_problem(problem_id)
    
    # ====================
    # GLOBAL SETTINGS
    # ====================
    
    def set_setting(self, key: str, value: str) -> None:
        """Set a global setting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, value)
            )
    
    def get_setting(self, key: str, default: str = "") -> str:
        """Get a global setting."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else default
    
    def get_default_mode(self) -> str:
        """Get the default mode for new problems."""
        return self.get_setting("default_mode", "snippet_grounded")
    
    def set_default_mode(self, mode: str) -> None:
        """Set the default mode for new problems."""
        self.set_setting("default_mode", mode)
