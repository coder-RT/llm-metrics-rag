"""
LLM Metrics Database Layer

This module provides SQLite database operations for storing and querying
LLM usage metrics. It handles schema creation, CRUD operations, and
aggregation queries.
"""

import sqlite3
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import contextmanager
from datetime import datetime

from .models import (
    AssistanceMode,
    LLMInteraction,
    CandidateSession,
    TokenMetrics,
    EfficiencyMetrics,
    QualityMetrics,
)


# Default database file path
DEFAULT_DB_PATH = Path("llm_metrics.db")


class MetricsDatabase:
    """
    SQLite database manager for LLM metrics storage.
    
    This class handles all database operations including:
    - Schema initialization
    - Saving interactions and sessions
    - Querying metrics by various criteria
    - Aggregating data for analysis
    
    Attributes:
        db_path: Path to the SQLite database file
    """
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database connection.
        
        Args:
            db_path: Path to SQLite database file. 
                    Defaults to 'llm_metrics.db' in current directory.
        """
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        
        Yields a connection with row_factory set to sqlite3.Row
        for dictionary-like access to results.
        """
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
        """
        Initialize database schema.
        
        Creates three main tables if they don't exist:
        - interactions: Individual LLM calls with all token counts
        - sessions: Candidate session summaries
        - problems: Problem-level aggregated metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table for individual LLM interactions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interactions (
                    interaction_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    snippets_provided TEXT,
                    
                    -- Token metrics
                    problem_statement_tokens INTEGER DEFAULT 0,
                    snippet_tokens INTEGER DEFAULT 0,
                    user_prompt_tokens INTEGER DEFAULT 0,
                    model_response_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    
                    -- Quality metrics
                    grounding_compliance_rate REAL DEFAULT 0.0,
                    snippet_reference_rate REAL DEFAULT 0.0,
                    hallucination_indicators INTEGER DEFAULT 0,
                    
                    -- Indexes for common queries
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            
            # Table for session summaries
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    candidate_id TEXT NOT NULL,
                    problem_id TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    
                    -- Aggregated token metrics
                    interaction_count INTEGER DEFAULT 0,
                    total_input_tokens INTEGER DEFAULT 0,
                    total_response_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    
                    -- Efficiency metrics
                    iterations INTEGER DEFAULT 0,
                    time_to_completion_seconds REAL DEFAULT 0.0,
                    tokens_per_submission REAL DEFAULT 0.0,
                    accepted INTEGER DEFAULT 0,
                    
                    -- Quality metrics
                    grounding_compliance_rate REAL DEFAULT 0.0,
                    snippet_reference_rate REAL DEFAULT 0.0,
                    hallucination_indicators INTEGER DEFAULT 0
                )
            """)
            
            # Table for problem-level aggregations
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS problems (
                    problem_id TEXT PRIMARY KEY,
                    total_sessions INTEGER DEFAULT 0,
                    total_interactions INTEGER DEFAULT 0,
                    
                    -- Mode distribution
                    free_form_sessions INTEGER DEFAULT 0,
                    snippet_grounded_sessions INTEGER DEFAULT 0,
                    
                    -- Aggregated metrics
                    avg_tokens_free_form REAL DEFAULT 0.0,
                    avg_tokens_grounded REAL DEFAULT 0.0,
                    avg_iterations_free_form REAL DEFAULT 0.0,
                    avg_iterations_grounded REAL DEFAULT 0.0,
                    
                    -- Success rates
                    acceptance_rate_free_form REAL DEFAULT 0.0,
                    acceptance_rate_grounded REAL DEFAULT 0.0,
                    
                    last_updated TEXT
                )
            """)
            
            # Create indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_session 
                ON interactions(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_candidate 
                ON interactions(candidate_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_interactions_mode 
                ON interactions(mode)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_candidate 
                ON sessions(candidate_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_problem 
                ON sessions(problem_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_sessions_mode 
                ON sessions(mode)
            """)
    
    # =====================
    # SAVE OPERATIONS
    # =====================
    
    def save_interaction(self, interaction: LLMInteraction) -> None:
        """
        Save a single LLM interaction to the database.
        
        Args:
            interaction: LLMInteraction object to persist
        """
        data = interaction.to_dict()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO interactions (
                    interaction_id, session_id, candidate_id, problem_id,
                    mode, timestamp, prompt, response, snippets_provided,
                    problem_statement_tokens, snippet_tokens, user_prompt_tokens,
                    model_response_tokens, total_tokens,
                    grounding_compliance_rate, snippet_reference_rate,
                    hallucination_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["interaction_id"],
                data["session_id"],
                data["candidate_id"],
                data["problem_id"],
                data["mode"],
                data["timestamp"],
                data["prompt"],
                data["response"],
                json.dumps(data["snippets_provided"]),
                data["problem_statement_tokens"],
                data["snippet_tokens"],
                data["user_prompt_tokens"],
                data["model_response_tokens"],
                data["total_tokens"],
                data["grounding_compliance_rate"],
                data["snippet_reference_rate"],
                data["hallucination_indicators"],
            ))
    
    def save_session(self, session: CandidateSession) -> None:
        """
        Save a candidate session to the database.
        
        Args:
            session: CandidateSession object to persist
        """
        data = session.to_dict()
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions (
                    session_id, candidate_id, problem_id, mode,
                    start_time, end_time, interaction_count,
                    total_input_tokens, total_response_tokens, total_tokens,
                    iterations, time_to_completion_seconds, tokens_per_submission,
                    accepted, grounding_compliance_rate, snippet_reference_rate,
                    hallucination_indicators
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data["session_id"],
                data["candidate_id"],
                data["problem_id"],
                data["mode"],
                data["start_time"],
                data["end_time"],
                data["interaction_count"],
                data["total_input_tokens"],
                data["total_response_tokens"],
                data["total_tokens"],
                data["iterations"],
                data["time_to_completion_seconds"],
                data["tokens_per_submission"],
                1 if data["accepted"] else 0,
                data["grounding_compliance_rate"],
                data["snippet_reference_rate"],
                data["hallucination_indicators"],
            ))
    
    def update_problem_stats(self, problem_id: str) -> None:
        """
        Update aggregated statistics for a problem.
        
        Recalculates all problem-level metrics based on sessions.
        
        Args:
            problem_id: Identifier of the problem to update
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get session counts by mode
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN mode = 'free_form' THEN 1 ELSE 0 END) as free_form,
                    SUM(CASE WHEN mode = 'snippet_grounded' THEN 1 ELSE 0 END) as grounded
                FROM sessions
                WHERE problem_id = ?
            """, (problem_id,))
            counts = cursor.fetchone()
            
            # Get average tokens by mode
            cursor.execute("""
                SELECT 
                    AVG(CASE WHEN mode = 'free_form' THEN total_tokens END) as avg_ff,
                    AVG(CASE WHEN mode = 'snippet_grounded' THEN total_tokens END) as avg_gr,
                    AVG(CASE WHEN mode = 'free_form' THEN iterations END) as iter_ff,
                    AVG(CASE WHEN mode = 'snippet_grounded' THEN iterations END) as iter_gr
                FROM sessions
                WHERE problem_id = ?
            """, (problem_id,))
            avgs = cursor.fetchone()
            
            # Get acceptance rates by mode
            cursor.execute("""
                SELECT 
                    AVG(CASE WHEN mode = 'free_form' THEN accepted END) as acc_ff,
                    AVG(CASE WHEN mode = 'snippet_grounded' THEN accepted END) as acc_gr
                FROM sessions
                WHERE problem_id = ?
            """, (problem_id,))
            acc_rates = cursor.fetchone()
            
            # Count total interactions
            cursor.execute("""
                SELECT COUNT(*) FROM interactions WHERE problem_id = ?
            """, (problem_id,))
            interaction_count = cursor.fetchone()[0]
            
            # Upsert problem stats
            cursor.execute("""
                INSERT OR REPLACE INTO problems (
                    problem_id, total_sessions, total_interactions,
                    free_form_sessions, snippet_grounded_sessions,
                    avg_tokens_free_form, avg_tokens_grounded,
                    avg_iterations_free_form, avg_iterations_grounded,
                    acceptance_rate_free_form, acceptance_rate_grounded,
                    last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                problem_id,
                counts["total"] or 0,
                interaction_count or 0,
                counts["free_form"] or 0,
                counts["grounded"] or 0,
                avgs["avg_ff"] or 0.0,
                avgs["avg_gr"] or 0.0,
                avgs["iter_ff"] or 0.0,
                avgs["iter_gr"] or 0.0,
                acc_rates["acc_ff"] or 0.0,
                acc_rates["acc_gr"] or 0.0,
                datetime.utcnow().isoformat(),
            ))
    
    # =====================
    # QUERY OPERATIONS
    # =====================
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a session by its ID.
        
        Args:
            session_id: The session identifier
            
        Returns:
            Dictionary with session data, or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_session_interactions(
        self, 
        session_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all interactions for a session.
        
        Args:
            session_id: The session identifier
            
        Returns:
            List of interaction dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM interactions WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def get_candidate_sessions(
        self, 
        candidate_id: str
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all interactions for a candidate (grouped by session-like periods).
        
        Args:
            candidate_id: The candidate identifier
            
        Returns:
            List of interaction dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                # Query interactions instead of sessions
                cursor.execute(
                    """SELECT 
                        interaction_id, candidate_id, problem_id, mode,
                        timestamp, total_tokens, user_prompt_tokens, 
                        model_response_tokens
                    FROM interactions 
                    WHERE candidate_id = ? 
                    ORDER BY timestamp""",
                    (candidate_id,)
                )
                return [dict(row) for row in cursor.fetchall()]
            except Exception:
                return []
    
    def get_problem_sessions(
        self, 
        problem_id: str,
        mode: Optional[AssistanceMode] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all sessions for a problem, optionally filtered by mode.
        
        Args:
            problem_id: The problem identifier
            mode: Optional filter by assistance mode
            
        Returns:
            List of session dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if mode:
                cursor.execute(
                    "SELECT * FROM sessions WHERE problem_id = ? AND mode = ?",
                    (problem_id, mode.value)
                )
            else:
                cursor.execute(
                    "SELECT * FROM sessions WHERE problem_id = ?",
                    (problem_id,)
                )
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_mode_comparison(
        self, 
        problem_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare metrics between Free-Form and Snippet-Grounded modes.
        
        Args:
            problem_id: Optional filter by specific problem
            
        Returns:
            Dictionary with comparison metrics for both modes
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Build query based on whether problem_id filter is applied
            where_clause = "WHERE problem_id = ?" if problem_id else ""
            params = (problem_id,) if problem_id else ()
            
            # Query from interactions table (more reliable than sessions)
            cursor.execute(f"""
                SELECT 
                    mode,
                    COUNT(*) as interaction_count,
                    SUM(total_tokens) as total_tokens,
                    AVG(total_tokens) as avg_total_tokens,
                    SUM(user_prompt_tokens) as total_input_tokens,
                    AVG(user_prompt_tokens) as avg_input_tokens,
                    SUM(model_response_tokens) as total_response_tokens,
                    AVG(model_response_tokens) as avg_response_tokens,
                    AVG(grounding_compliance_rate) as avg_grounding
                FROM interactions
                {where_clause}
                GROUP BY mode
            """, params)
            
            results = {
                "free_form": {},
                "snippet_grounded": {},
            }
            
            for row in cursor.fetchall():
                mode_key = row["mode"]
                results[mode_key] = {
                    "interaction_count": row["interaction_count"],
                    "total_tokens": row["total_tokens"] or 0,
                    "avg_total_tokens": round(row["avg_total_tokens"] or 0, 1),
                    "total_input_tokens": row["total_input_tokens"] or 0,
                    "avg_input_tokens": round(row["avg_input_tokens"] or 0, 1),
                    "total_response_tokens": row["total_response_tokens"] or 0,
                    "avg_response_tokens": round(row["avg_response_tokens"] or 0, 1),
                    "avg_grounding_score": round(row["avg_grounding"] or 0, 2),
                }
            
            # Calculate token savings
            ff_tokens = results["free_form"].get("avg_total_tokens", 0)
            gr_tokens = results["snippet_grounded"].get("avg_total_tokens", 0)
            
            if ff_tokens > 0:
                token_reduction_pct = ((ff_tokens - gr_tokens) / ff_tokens) * 100
            else:
                token_reduction_pct = 0
            
            results["comparison"] = {
                "token_reduction_percent": round(token_reduction_pct, 1),
                "grounded_saves_tokens": gr_tokens < ff_tokens,
            }
            
            return results
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics from interactions table.
        
        Returns:
            Dictionary with aggregate metrics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Get direct interaction stats (primary source of truth)
            try:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_interactions,
                        SUM(total_tokens) as total_tokens,
                        AVG(total_tokens) as avg_tokens_per_interaction,
                        SUM(user_prompt_tokens) as total_input_tokens,
                        SUM(model_response_tokens) as total_output_tokens,
                        COUNT(DISTINCT candidate_id) as unique_candidates,
                        COUNT(DISTINCT problem_id) as unique_problems
                    FROM interactions
                """)
                interaction_row = cursor.fetchone()
            except Exception:
                interaction_row = None
            
            if interaction_row:
                return {
                    "total_interactions": interaction_row["total_interactions"] or 0,
                    "total_tokens": interaction_row["total_tokens"] or 0,
                    "total_input_tokens": interaction_row["total_input_tokens"] or 0,
                    "total_output_tokens": interaction_row["total_output_tokens"] or 0,
                    "avg_tokens_per_interaction": round(interaction_row["avg_tokens_per_interaction"] or 0, 1),
                    "unique_candidates": interaction_row["unique_candidates"] or 0,
                    "unique_problems": interaction_row["unique_problems"] or 0,
                }
            else:
                return {
                    "total_interactions": 0,
                    "total_tokens": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "avg_tokens_per_interaction": 0,
                    "unique_candidates": 0,
                    "unique_problems": 0,
                }
    
    def get_candidate_total_tokens(self, candidate_id: str) -> int:
        """
        Get total tokens used by a candidate across all interactions.
        
        Args:
            candidate_id: The candidate identifier
            
        Returns:
            Total token count
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    "SELECT SUM(total_tokens) FROM interactions WHERE candidate_id = ?",
                    (candidate_id,)
                )
                result = cursor.fetchone()[0]
                return result or 0
            except Exception:
                return 0

