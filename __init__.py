"""
LLM Metrics Tracking System

A Python library for tracking and analyzing LLM usage in hackathon platforms.
Supports two assistance modes (Free-Form and Snippet-Grounded) and provides
comprehensive metrics collection, storage, and analysis capabilities.

Quick Start:
    from llmMetrics import LLMMetricsTracker, AssistanceMode

    # Create a tracker
    tracker = LLMMetricsTracker()

    # Start a session
    tracker.start_session("candidate_123", "problem_456")
    tracker.set_mode(AssistanceMode.SNIPPET_GROUNDED)

    # Record LLM interactions
    tracker.record_interaction(
        prompt="How do I fix this bug?",
        response="You can modify the function...",
        snippets=["def my_function(): ..."]
    )

    # End session and get summary
    summary = tracker.end_session(accepted=True)
    print(f"Total tokens: {summary['total_tokens']}")

Modules:
    - models: Data classes and enums (AssistanceMode, TokenMetrics, etc.)
    - database: SQLite storage layer (MetricsDatabase)
    - tracker: Main tracking class (LLMMetricsTracker)
    - analyzers: Analysis and reporting functions
    - utils: Token counting and grounding utilities

For more details, see the individual module docstrings.
"""

__version__ = "0.1.0"
__author__ = "Hackathon Platform Team"

# =============================================================================
# Core Classes and Enums
# =============================================================================

from .models import (
    # Enums
    AssistanceMode,
    
    # Data classes
    TokenMetrics,
    EfficiencyMetrics,
    QualityMetrics,
    LLMInteraction,
    CandidateSession,
)

# =============================================================================
# Main Tracker
# =============================================================================

from .tracker import (
    LLMMetricsTracker,
    create_tracker,
)

# =============================================================================
# Database
# =============================================================================

from .database import (
    MetricsDatabase,
)

# =============================================================================
# Analysis Functions
# =============================================================================

from .analyzers import (
    # Comparison and analysis
    compare_modes,
    ModeComparisonResult,
    calculate_savings,
    get_variance_stats,
    
    # Reporting
    generate_report,
    get_leaderboard,
    export_metrics_summary,
)

# =============================================================================
# Utility Functions
# =============================================================================

from .utils import (
    count_tokens,
    estimate_snippet_tokens,
    calculate_grounding_score,
    calculate_snippet_reference_rate,
    detect_hallucination_indicators,
    calculate_cost_estimate,
    format_token_count,
)

# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Version
    "__version__",
    
    # Enums
    "AssistanceMode",
    
    # Data Models
    "TokenMetrics",
    "EfficiencyMetrics", 
    "QualityMetrics",
    "LLMInteraction",
    "CandidateSession",
    
    # Main Tracker
    "LLMMetricsTracker",
    "create_tracker",
    
    # Database
    "MetricsDatabase",
    
    # Analysis
    "compare_modes",
    "ModeComparisonResult",
    "calculate_savings",
    "get_variance_stats",
    "generate_report",
    "get_leaderboard",
    "export_metrics_summary",
    
    # Utilities
    "count_tokens",
    "estimate_snippet_tokens",
    "calculate_grounding_score",
    "calculate_snippet_reference_rate",
    "detect_hallucination_indicators",
    "calculate_cost_estimate",
    "format_token_count",
]

