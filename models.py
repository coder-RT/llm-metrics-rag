"""
LLM Metrics Data Models

This module defines all data structures used for tracking LLM usage metrics
in the hackathon platform. It includes enums for assistance modes and 
dataclasses for various metric types.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from uuid import uuid4


class AssistanceMode(Enum):
    """
    Defines the two LLM assistance modes available in the hackathon platform.
    
    FREE_FORM: Unrestricted LLM assistance where the model can generate
               full solutions, propose new designs, and provide detailed
               explanations. Higher token usage, greater variability.
    
    SNIPPET_GROUNDED: Restricted mode where LLM can only work with provided
                      code snippets. Lower token usage, better evaluation
                      fairness, reduced hallucination risk.
    """
    FREE_FORM = "free_form"
    SNIPPET_GROUNDED = "snippet_grounded"


@dataclass
class TokenMetrics:
    """
    Captures token-level metrics for an LLM interaction.
    
    Attributes:
        problem_statement_tokens: Tokens in the original problem description
        snippet_tokens: Tokens in the provided code snippets (if any)
        user_prompt_tokens: Tokens in the user's prompt/question
        model_response_tokens: Tokens in the LLM's response
        total_tokens: Sum of all tokens (input + output)
    """
    problem_statement_tokens: int = 0
    snippet_tokens: int = 0
    user_prompt_tokens: int = 0
    model_response_tokens: int = 0
    
    @property
    def total_input_tokens(self) -> int:
        """Total input tokens (problem + snippets + user prompt)."""
        return (self.problem_statement_tokens + 
                self.snippet_tokens + 
                self.user_prompt_tokens)
    
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.total_input_tokens + self.model_response_tokens
    
    @property
    def model_to_user_ratio(self) -> float:
        """
        Ratio of model response tokens to user prompt tokens.
        Higher ratio indicates more verbose model responses.
        """
        if self.user_prompt_tokens == 0:
            return 0.0
        return self.model_response_tokens / self.user_prompt_tokens


@dataclass
class EfficiencyMetrics:
    """
    Captures efficiency-related metrics for a session or problem.
    
    Attributes:
        iterations: Number of LLM interactions before solution acceptance
        time_to_completion_seconds: Total time from start to accepted submission
        tokens_per_submission: Average tokens used per submission attempt
        accepted: Whether the solution was ultimately accepted
    """
    iterations: int = 0
    time_to_completion_seconds: float = 0.0
    tokens_per_submission: float = 0.0
    accepted: bool = False
    
    @property
    def tokens_per_iteration(self) -> float:
        """Average tokens per LLM interaction."""
        if self.iterations == 0:
            return 0.0
        return self.tokens_per_submission / self.iterations


@dataclass
class QualityMetrics:
    """
    Captures quality and compliance metrics for LLM interactions.
    
    Attributes:
        grounding_compliance_rate: Percentage of responses that properly
                                   reference provided snippets (0.0 to 1.0)
        snippet_reference_rate: How often the model references code snippets
                               in its responses (0.0 to 1.0)
        hallucination_indicators: Count of potential hallucination instances
                                  (references to non-existent code, etc.)
        solution_variance_score: Measure of how different this solution is
                                from other candidates (0.0 = identical)
    """
    grounding_compliance_rate: float = 0.0
    snippet_reference_rate: float = 0.0
    hallucination_indicators: int = 0
    solution_variance_score: float = 0.0
    
    def is_well_grounded(self, threshold: float = 0.7) -> bool:
        """Check if the interaction meets grounding quality threshold."""
        return self.grounding_compliance_rate >= threshold


@dataclass
class LLMInteraction:
    """
    Complete record of a single LLM interaction/call.
    
    This is the primary unit of data collection - each time a candidate
    sends a prompt to the LLM and receives a response, one LLMInteraction
    record is created.
    
    Attributes:
        interaction_id: Unique identifier for this interaction
        session_id: Reference to the parent session
        candidate_id: Identifier for the candidate/user
        problem_id: Identifier for the problem being solved
        mode: The assistance mode in effect during this interaction
        timestamp: When the interaction occurred
        prompt: The user's prompt text
        response: The LLM's response text
        snippets_provided: List of code snippets provided to the LLM
        token_metrics: Token counts for this interaction
        quality_metrics: Quality scores for this interaction
    """
    session_id: str
    candidate_id: str
    problem_id: str
    mode: AssistanceMode
    prompt: str
    response: str
    token_metrics: TokenMetrics
    interaction_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    snippets_provided: List[str] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        # Ensure prompt and response are strings
        prompt_str = self.prompt if isinstance(self.prompt, str) else str(self.prompt)
        response_str = self.response if isinstance(self.response, str) else str(self.response)
        
        return {
            "interaction_id": self.interaction_id,
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "mode": self.mode.value,
            "timestamp": self.timestamp.isoformat(),
            "prompt": prompt_str,
            "response": response_str,
            "snippets_provided": self.snippets_provided,
            "problem_statement_tokens": self.token_metrics.problem_statement_tokens,
            "snippet_tokens": self.token_metrics.snippet_tokens,
            "user_prompt_tokens": self.token_metrics.user_prompt_tokens,
            "model_response_tokens": self.token_metrics.model_response_tokens,
            "total_tokens": self.token_metrics.total_tokens,
            "grounding_compliance_rate": (
                self.quality_metrics.grounding_compliance_rate 
                if self.quality_metrics else 0.0
            ),
            "snippet_reference_rate": (
                self.quality_metrics.snippet_reference_rate 
                if self.quality_metrics else 0.0
            ),
            "hallucination_indicators": (
                self.quality_metrics.hallucination_indicators 
                if self.quality_metrics else 0
            ),
        }


@dataclass
class CandidateSession:
    """
    Aggregated metrics for a single candidate's session on a problem.
    
    A session represents one candidate's complete attempt at solving
    a problem, potentially including multiple LLM interactions.
    
    Attributes:
        session_id: Unique identifier for this session
        candidate_id: Identifier for the candidate
        problem_id: Identifier for the problem
        mode: Primary assistance mode used (may switch during session)
        start_time: When the session began
        end_time: When the session ended (None if still active)
        interactions: List of all LLM interactions in this session
        total_token_metrics: Aggregated token counts
        efficiency_metrics: Efficiency measurements
        quality_metrics: Aggregated quality scores
    """
    candidate_id: str
    problem_id: str
    mode: AssistanceMode
    session_id: str = field(default_factory=lambda: str(uuid4()))
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    interactions: List[LLMInteraction] = field(default_factory=list)
    total_token_metrics: TokenMetrics = field(default_factory=TokenMetrics)
    efficiency_metrics: EfficiencyMetrics = field(default_factory=EfficiencyMetrics)
    quality_metrics: QualityMetrics = field(default_factory=QualityMetrics)
    
    @property
    def is_active(self) -> bool:
        """Check if the session is still ongoing."""
        return self.end_time is None
    
    @property
    def duration_seconds(self) -> float:
        """Calculate session duration in seconds."""
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
    
    @property
    def interaction_count(self) -> int:
        """Number of LLM interactions in this session."""
        return len(self.interactions)
    
    def add_interaction(self, interaction: LLMInteraction) -> None:
        """Add an interaction and update aggregated metrics."""
        self.interactions.append(interaction)
        
        # Update token totals
        self.total_token_metrics.problem_statement_tokens += (
            interaction.token_metrics.problem_statement_tokens
        )
        self.total_token_metrics.snippet_tokens += (
            interaction.token_metrics.snippet_tokens
        )
        self.total_token_metrics.user_prompt_tokens += (
            interaction.token_metrics.user_prompt_tokens
        )
        self.total_token_metrics.model_response_tokens += (
            interaction.token_metrics.model_response_tokens
        )
        
        # Update efficiency metrics
        self.efficiency_metrics.iterations = len(self.interactions)
    
    def finalize(self, accepted: bool = False) -> None:
        """
        Finalize the session, calculating final metrics.
        
        Args:
            accepted: Whether the candidate's solution was accepted
        """
        self.end_time = datetime.utcnow()
        self.efficiency_metrics.accepted = accepted
        self.efficiency_metrics.time_to_completion_seconds = self.duration_seconds
        
        if self.interaction_count > 0:
            self.efficiency_metrics.tokens_per_submission = (
                self.total_token_metrics.total_tokens / self.interaction_count
            )
            
            # Calculate average quality metrics
            total_grounding = sum(
                i.quality_metrics.grounding_compliance_rate 
                for i in self.interactions 
                if i.quality_metrics
            )
            total_snippet_ref = sum(
                i.quality_metrics.snippet_reference_rate 
                for i in self.interactions 
                if i.quality_metrics
            )
            quality_count = sum(1 for i in self.interactions if i.quality_metrics)
            
            if quality_count > 0:
                self.quality_metrics.grounding_compliance_rate = (
                    total_grounding / quality_count
                )
                self.quality_metrics.snippet_reference_rate = (
                    total_snippet_ref / quality_count
                )
                self.quality_metrics.hallucination_indicators = sum(
                    i.quality_metrics.hallucination_indicators 
                    for i in self.interactions 
                    if i.quality_metrics
                )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for database storage."""
        return {
            "session_id": self.session_id,
            "candidate_id": self.candidate_id,
            "problem_id": self.problem_id,
            "mode": self.mode.value,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "interaction_count": self.interaction_count,
            "total_input_tokens": self.total_token_metrics.total_input_tokens,
            "total_response_tokens": self.total_token_metrics.model_response_tokens,
            "total_tokens": self.total_token_metrics.total_tokens,
            "iterations": self.efficiency_metrics.iterations,
            "time_to_completion_seconds": self.efficiency_metrics.time_to_completion_seconds,
            "tokens_per_submission": self.efficiency_metrics.tokens_per_submission,
            "accepted": self.efficiency_metrics.accepted,
            "grounding_compliance_rate": self.quality_metrics.grounding_compliance_rate,
            "snippet_reference_rate": self.quality_metrics.snippet_reference_rate,
            "hallucination_indicators": self.quality_metrics.hallucination_indicators,
        }

