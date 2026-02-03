"""
LLM Metrics Tracker

This module provides the main tracking class for recording and managing
LLM usage metrics in the hackathon platform. It serves as the primary
interface for integrating metrics collection into applications.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager

from .models import (
    AssistanceMode,
    TokenMetrics,
    QualityMetrics,
    LLMInteraction,
    CandidateSession,
)
from .database import MetricsDatabase
from .utils import (
    count_tokens,
    estimate_snippet_tokens,
    calculate_grounding_score,
    calculate_snippet_reference_rate,
    detect_hallucination_indicators,
)


class LLMMetricsTracker:
    """
    Main class for tracking LLM usage metrics.
    
    This class provides a high-level API for recording LLM interactions,
    managing sessions, and switching between assistance modes. It handles
    all metric calculations internally and persists data to SQLite.
    
    Usage:
        tracker = LLMMetricsTracker()
        
        # Start a session for a candidate
        tracker.start_session("candidate_123", "problem_456")
        tracker.set_mode(AssistanceMode.SNIPPET_GROUNDED)
        
        # Record interactions
        tracker.record_interaction(
            prompt="How do I fix this bug?",
            response="You can modify the function...",
            snippets=["def my_function(): ..."]
        )
        
        # End session and get summary
        summary = tracker.end_session(accepted=True)
    
    Attributes:
        db: MetricsDatabase instance for persistence
        current_session: Active CandidateSession (if any)
        current_mode: Current AssistanceMode setting
        model: LLM model name for token counting
    """
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        model: str = "gpt-4",
        default_mode: AssistanceMode = AssistanceMode.SNIPPET_GROUNDED
    ):
        """
        Initialize the metrics tracker.
        
        Args:
            db_path: Path to SQLite database file.
                    Defaults to 'llm_metrics.db' in current directory.
            model: LLM model name for accurate token counting.
            default_mode: Default assistance mode for new sessions.
        """
        self.db = MetricsDatabase(db_path)
        self.model = model
        self.default_mode = default_mode
        self.current_session: Optional[CandidateSession] = None
        self.current_mode: AssistanceMode = default_mode
        self._problem_statement: str = ""
        self._problem_statement_tokens: int = 0
    
    # =====================
    # SESSION MANAGEMENT
    # =====================
    
    def start_session(
        self,
        candidate_id: str,
        problem_id: str,
        problem_statement: str = "",
        mode: Optional[AssistanceMode] = None
    ) -> str:
        """
        Start a new tracking session for a candidate.
        
        Creates a new CandidateSession and sets it as the active session.
        If a session is already active, it will be finalized first.
        
        Args:
            candidate_id: Unique identifier for the candidate
            problem_id: Unique identifier for the problem
            problem_statement: The problem description text (for token counting)
            mode: Assistance mode to use (defaults to tracker's default_mode)
            
        Returns:
            The session_id of the newly created session
            
        Example:
            >>> tracker = LLMMetricsTracker()
            >>> session_id = tracker.start_session("user_123", "prob_456")
            >>> print(f"Started session: {session_id}")
        """
        # End any existing session
        if self.current_session:
            self.end_session()
        
        # Set mode
        self.current_mode = mode or self.default_mode
        
        # Store problem statement for token calculations
        self._problem_statement = problem_statement
        self._problem_statement_tokens = count_tokens(problem_statement, self.model)
        
        # Create new session
        self.current_session = CandidateSession(
            candidate_id=candidate_id,
            problem_id=problem_id,
            mode=self.current_mode,
        )
        
        return self.current_session.session_id
    
    def end_session(self, accepted: bool = False) -> Optional[Dict[str, Any]]:
        """
        End the current session and persist the data.
        
        Finalizes the session, calculates aggregated metrics, saves to
        database, and updates problem statistics.
        
        Args:
            accepted: Whether the candidate's solution was accepted
            
        Returns:
            Dictionary with session summary, or None if no active session
            
        Example:
            >>> summary = tracker.end_session(accepted=True)
            >>> print(f"Total tokens: {summary['total_tokens']}")
        """
        if not self.current_session:
            return None
        
        # Finalize session
        self.current_session.finalize(accepted=accepted)
        
        # Save to database
        self.db.save_session(self.current_session)
        
        # Update problem-level statistics
        self.db.update_problem_stats(self.current_session.problem_id)
        
        # Get summary before clearing
        summary = self.current_session.to_dict()
        
        # Clear current session
        self.current_session = None
        self._problem_statement = ""
        self._problem_statement_tokens = 0
        
        return summary
    
    def set_mode(self, mode: AssistanceMode) -> None:
        """
        Switch the assistance mode for the current session.
        
        This can be called at any time during a session. The mode
        affects how interactions are categorized and analyzed.
        
        Args:
            mode: The new AssistanceMode to use
            
        Example:
            >>> tracker.set_mode(AssistanceMode.FREE_FORM)
        """
        self.current_mode = mode
        if self.current_session:
            self.current_session.mode = mode
    
    @contextmanager
    def session(
        self,
        candidate_id: str,
        problem_id: str,
        problem_statement: str = "",
        mode: Optional[AssistanceMode] = None
    ):
        """
        Context manager for session handling.
        
        Automatically starts and ends a session, ensuring proper cleanup.
        
        Args:
            candidate_id: Unique identifier for the candidate
            problem_id: Unique identifier for the problem
            problem_statement: The problem description text
            mode: Assistance mode to use
            
        Yields:
            The session_id
            
        Example:
            >>> with tracker.session("user_123", "prob_456") as session_id:
            ...     tracker.record_interaction(prompt="Help", response="...")
        """
        session_id = self.start_session(
            candidate_id, problem_id, problem_statement, mode
        )
        try:
            yield session_id
        finally:
            self.end_session()
    
    # =====================
    # INTERACTION RECORDING
    # =====================
    
    def record_interaction(
        self,
        prompt: str,
        response: str,
        snippets: Optional[List[str]] = None,
        problem_statement: Optional[str] = None
    ) -> Optional[str]:
        """
        Record a single LLM interaction.
        
        This is the primary method for logging LLM usage. It calculates
        all token metrics and quality scores automatically.
        
        Args:
            prompt: The user's prompt/question sent to the LLM
            response: The LLM's response text
            snippets: Optional list of code snippets provided to the LLM
            problem_statement: Optional override for problem statement tokens
            
        Returns:
            The interaction_id, or None if no active session
            
        Example:
            >>> interaction_id = tracker.record_interaction(
            ...     prompt="How do I fix the null check?",
            ...     response="You should add an early return...",
            ...     snippets=["def validate(data):\\n    if data is None:..."]
            ... )
        """
        if not self.current_session:
            raise RuntimeError(
                "No active session. Call start_session() first."
            )
        
        snippets = snippets or []
        
        # Calculate token metrics
        problem_tokens = (
            count_tokens(problem_statement, self.model)
            if problem_statement
            else self._problem_statement_tokens
        )
        
        token_metrics = TokenMetrics(
            problem_statement_tokens=problem_tokens,
            snippet_tokens=estimate_snippet_tokens(snippets, self.model),
            user_prompt_tokens=count_tokens(prompt, self.model),
            model_response_tokens=count_tokens(response, self.model),
        )
        
        # Calculate quality metrics (especially for grounded mode)
        quality_metrics = None
        if snippets:
            grounding_score = calculate_grounding_score(response, snippets)
            snippet_ref_rate = calculate_snippet_reference_rate(response, snippets)
            hallucinations = detect_hallucination_indicators(response, snippets)
            
            quality_metrics = QualityMetrics(
                grounding_compliance_rate=grounding_score,
                snippet_reference_rate=snippet_ref_rate,
                hallucination_indicators=hallucinations,
            )
        
        # Create interaction record
        interaction = LLMInteraction(
            session_id=self.current_session.session_id,
            candidate_id=self.current_session.candidate_id,
            problem_id=self.current_session.problem_id,
            mode=self.current_mode,
            prompt=prompt,
            response=response,
            snippets_provided=snippets,
            token_metrics=token_metrics,
            quality_metrics=quality_metrics,
        )
        
        # Add to session and save
        self.current_session.add_interaction(interaction)
        self.db.save_interaction(interaction)
        
        return interaction.interaction_id
    
    def record_batch(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Record multiple interactions at once.
        
        Useful for bulk importing or replaying recorded sessions.
        
        Args:
            interactions: List of interaction dictionaries with keys:
                         - prompt: str
                         - response: str
                         - snippets: Optional[List[str]]
                         
        Returns:
            List of interaction_ids for recorded interactions
            
        Example:
            >>> ids = tracker.record_batch([
            ...     {"prompt": "Question 1", "response": "Answer 1"},
            ...     {"prompt": "Question 2", "response": "Answer 2"},
            ... ])
        """
        interaction_ids = []
        
        for item in interactions:
            interaction_id = self.record_interaction(
                prompt=item["prompt"],
                response=item["response"],
                snippets=item.get("snippets"),
            )
            if interaction_id:
                interaction_ids.append(interaction_id)
        
        return interaction_ids
    
    # =====================
    # SESSION QUERIES
    # =====================
    
    def get_session_summary(self) -> Optional[Dict[str, Any]]:
        """
        Get summary of the current active session.
        
        Returns aggregated metrics for the ongoing session without
        ending it.
        
        Returns:
            Dictionary with session metrics, or None if no active session
        """
        if not self.current_session:
            return None
        
        return {
            "session_id": self.current_session.session_id,
            "candidate_id": self.current_session.candidate_id,
            "problem_id": self.current_session.problem_id,
            "mode": self.current_mode.value,
            "is_active": True,
            "interaction_count": self.current_session.interaction_count,
            "duration_seconds": self.current_session.duration_seconds,
            "total_tokens": self.current_session.total_token_metrics.total_tokens,
            "total_input_tokens": self.current_session.total_token_metrics.total_input_tokens,
            "total_response_tokens": self.current_session.total_token_metrics.model_response_tokens,
        }
    
    def get_current_token_usage(self) -> int:
        """
        Get total tokens used in the current session.
        
        Returns:
            Total token count, or 0 if no active session
        """
        if not self.current_session:
            return 0
        return self.current_session.total_token_metrics.total_tokens
    
    def get_interaction_count(self) -> int:
        """
        Get number of interactions in the current session.
        
        Returns:
            Interaction count, or 0 if no active session
        """
        if not self.current_session:
            return 0
        return self.current_session.interaction_count
    
    # =====================
    # HISTORICAL QUERIES
    # =====================
    
    def get_candidate_history(self, candidate_id: str) -> List[Dict[str, Any]]:
        """
        Get all sessions for a specific candidate.
        
        Args:
            candidate_id: The candidate identifier
            
        Returns:
            List of session dictionaries
        """
        return self.db.get_candidate_sessions(candidate_id)
    
    def get_candidate_total_tokens(self, candidate_id: str) -> int:
        """
        Get total tokens used by a candidate across all sessions.
        
        Args:
            candidate_id: The candidate identifier
            
        Returns:
            Total token count
        """
        return self.db.get_candidate_total_tokens(candidate_id)
    
    def get_mode_comparison(
        self, 
        problem_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare metrics between Free-Form and Snippet-Grounded modes.
        
        Args:
            problem_id: Optional filter by specific problem
            
        Returns:
            Dictionary with comparison metrics
        """
        return self.db.get_mode_comparison(problem_id)
    
    def get_overall_stats(self) -> Dict[str, Any]:
        """
        Get overall statistics across all sessions.
        
        Returns:
            Dictionary with aggregate metrics
        """
        return self.db.get_overall_stats()


# Convenience function for quick tracking
def create_tracker(
    db_path: Optional[str] = None,
    model: str = "gpt-4",
    default_mode: str = "snippet_grounded"
) -> LLMMetricsTracker:
    """
    Factory function to create a configured tracker.
    
    Args:
        db_path: Path to database file (optional)
        model: LLM model name for token counting
        default_mode: Default mode ('free_form' or 'snippet_grounded')
        
    Returns:
        Configured LLMMetricsTracker instance
        
    Example:
        >>> tracker = create_tracker(db_path="./metrics.db")
    """
    mode = (
        AssistanceMode.FREE_FORM 
        if default_mode == "free_form" 
        else AssistanceMode.SNIPPET_GROUNDED
    )
    
    return LLMMetricsTracker(
        db_path=Path(db_path) if db_path else None,
        model=model,
        default_mode=mode,
    )

