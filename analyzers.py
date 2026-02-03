"""
LLM Metrics Analyzers

This module provides analysis and reporting functions for LLM usage data.
It includes mode comparison, cost estimation, variance analysis, and
report generation capabilities.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .models import AssistanceMode
from .database import MetricsDatabase
from .utils import calculate_cost_estimate, format_token_count


@dataclass
class ModeComparisonResult:
    """
    Results from comparing Free-Form vs Snippet-Grounded modes.
    
    Attributes:
        free_form_avg_tokens: Average tokens per session in free-form mode
        grounded_avg_tokens: Average tokens per session in grounded mode
        token_reduction_percent: Percentage reduction from grounded mode
        free_form_sessions: Number of free-form sessions
        grounded_sessions: Number of grounded sessions
        free_form_acceptance_rate: Solution acceptance rate for free-form
        grounded_acceptance_rate: Solution acceptance rate for grounded
    """
    free_form_avg_tokens: float
    grounded_avg_tokens: float
    token_reduction_percent: float
    free_form_sessions: int
    grounded_sessions: int
    free_form_acceptance_rate: float
    grounded_acceptance_rate: float
    free_form_avg_iterations: float
    grounded_avg_iterations: float


def compare_modes(
    db: MetricsDatabase,
    problem_id: Optional[str] = None
) -> ModeComparisonResult:
    """
    Compare token usage and metrics between assistance modes.
    
    This function analyzes all sessions and provides a detailed comparison
    of Free-Form vs Snippet-Grounded modes, helping quantify the benefits
    of grounded assistance.
    
    Args:
        db: MetricsDatabase instance
        problem_id: Optional filter by specific problem
        
    Returns:
        ModeComparisonResult with detailed comparison metrics
        
    Example:
        >>> db = MetricsDatabase()
        >>> result = compare_modes(db)
        >>> print(f"Token reduction: {result.token_reduction_percent:.1f}%")
    """
    comparison = db.get_mode_comparison(problem_id)
    
    ff_data = comparison.get("free_form", {})
    gr_data = comparison.get("snippet_grounded", {})
    
    ff_tokens = ff_data.get("avg_total_tokens", 0)
    gr_tokens = gr_data.get("avg_total_tokens", 0)
    
    # Calculate reduction percentage
    if ff_tokens > 0:
        reduction = ((ff_tokens - gr_tokens) / ff_tokens) * 100
    else:
        reduction = 0.0
    
    return ModeComparisonResult(
        free_form_avg_tokens=ff_tokens,
        grounded_avg_tokens=gr_tokens,
        token_reduction_percent=reduction,
        free_form_sessions=ff_data.get("session_count", 0),
        grounded_sessions=gr_data.get("session_count", 0),
        free_form_acceptance_rate=ff_data.get("acceptance_rate", 0),
        grounded_acceptance_rate=gr_data.get("acceptance_rate", 0),
        free_form_avg_iterations=ff_data.get("avg_iterations", 0),
        grounded_avg_iterations=gr_data.get("avg_iterations", 0),
    )


def calculate_savings(
    db: MetricsDatabase,
    input_cost_per_1k: float = 0.03,
    output_cost_per_1k: float = 0.06,
    problem_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate cost savings from using Snippet-Grounded mode.
    
    Estimates the dollar amount saved by using grounded mode instead
    of free-form assistance, based on actual token usage data.
    
    Args:
        db: MetricsDatabase instance
        input_cost_per_1k: Cost per 1000 input tokens (default: GPT-4 pricing)
        output_cost_per_1k: Cost per 1000 output tokens
        problem_id: Optional filter by specific problem
        
    Returns:
        Dictionary with savings calculations:
        - actual_cost: Cost with current mode distribution
        - all_free_form_cost: Hypothetical cost if all were free-form
        - all_grounded_cost: Hypothetical cost if all were grounded
        - savings_from_grounded: Amount saved by grounded sessions
        - potential_additional_savings: If remaining free-form switched
        
    Example:
        >>> savings = calculate_savings(db)
        >>> print(f"Saved: ${savings['savings_from_grounded']:.2f}")
    """
    comparison = db.get_mode_comparison(problem_id)
    
    ff_data = comparison.get("free_form", {})
    gr_data = comparison.get("snippet_grounded", {})
    
    ff_sessions = ff_data.get("session_count", 0)
    gr_sessions = gr_data.get("session_count", 0)
    
    ff_avg_input = ff_data.get("avg_input_tokens", 0)
    ff_avg_output = ff_data.get("avg_response_tokens", 0)
    gr_avg_input = gr_data.get("avg_input_tokens", 0)
    gr_avg_output = gr_data.get("avg_response_tokens", 0)
    
    # Calculate costs per session type
    ff_cost_per_session = calculate_cost_estimate(
        int(ff_avg_input), int(ff_avg_output),
        input_cost_per_1k, output_cost_per_1k
    )
    gr_cost_per_session = calculate_cost_estimate(
        int(gr_avg_input), int(gr_avg_output),
        input_cost_per_1k, output_cost_per_1k
    )
    
    # Actual cost
    actual_cost = (ff_sessions * ff_cost_per_session + 
                   gr_sessions * gr_cost_per_session)
    
    # Hypothetical: all free-form
    total_sessions = ff_sessions + gr_sessions
    all_ff_cost = total_sessions * ff_cost_per_session
    
    # Hypothetical: all grounded
    all_gr_cost = total_sessions * gr_cost_per_session
    
    # Savings from grounded sessions
    savings_from_grounded = gr_sessions * (ff_cost_per_session - gr_cost_per_session)
    
    # Potential additional savings
    potential_savings = ff_sessions * (ff_cost_per_session - gr_cost_per_session)
    
    return {
        "total_sessions": total_sessions,
        "free_form_sessions": ff_sessions,
        "grounded_sessions": gr_sessions,
        "cost_per_free_form_session": ff_cost_per_session,
        "cost_per_grounded_session": gr_cost_per_session,
        "actual_cost": round(actual_cost, 2),
        "all_free_form_cost": round(all_ff_cost, 2),
        "all_grounded_cost": round(all_gr_cost, 2),
        "savings_from_grounded": round(savings_from_grounded, 2),
        "potential_additional_savings": round(potential_savings, 2),
        "total_potential_savings": round(savings_from_grounded + potential_savings, 2),
    }


def generate_report(
    db: MetricsDatabase,
    candidate_id: str
) -> Dict[str, Any]:
    """
    Generate a comprehensive report for a candidate.
    
    Creates a detailed summary of a candidate's LLM usage across
    all sessions, including token usage, efficiency metrics, and
    quality scores.
    
    Args:
        db: MetricsDatabase instance
        candidate_id: The candidate identifier
        
    Returns:
        Dictionary with comprehensive candidate report:
        - summary: Overall statistics
        - sessions: List of session details
        - mode_breakdown: Usage by assistance mode
        - recommendations: Suggestions based on usage patterns
        
    Example:
        >>> report = generate_report(db, "candidate_123")
        >>> print(f"Total tokens: {report['summary']['total_tokens']}")
    """
    sessions = db.get_candidate_sessions(candidate_id)
    
    if not sessions:
        return {
            "candidate_id": candidate_id,
            "error": "No sessions found for this candidate",
            "generated_at": datetime.utcnow().isoformat(),
        }
    
    # Calculate aggregate metrics
    total_tokens = sum(s["total_tokens"] for s in sessions)
    total_interactions = sum(s["interaction_count"] for s in sessions)
    total_time = sum(s["time_to_completion_seconds"] for s in sessions)
    accepted_count = sum(1 for s in sessions if s["accepted"])
    
    # Mode breakdown
    ff_sessions = [s for s in sessions if s["mode"] == "free_form"]
    gr_sessions = [s for s in sessions if s["mode"] == "snippet_grounded"]
    
    ff_tokens = sum(s["total_tokens"] for s in ff_sessions)
    gr_tokens = sum(s["total_tokens"] for s in gr_sessions)
    
    # Quality averages
    avg_grounding = (
        sum(s["grounding_compliance_rate"] for s in sessions) / len(sessions)
        if sessions else 0
    )
    avg_snippet_ref = (
        sum(s["snippet_reference_rate"] for s in sessions) / len(sessions)
        if sessions else 0
    )
    
    # Generate recommendations
    recommendations = _generate_recommendations(
        sessions, avg_grounding, ff_sessions, gr_sessions
    )
    
    return {
        "candidate_id": candidate_id,
        "generated_at": datetime.utcnow().isoformat(),
        "summary": {
            "total_sessions": len(sessions),
            "total_interactions": total_interactions,
            "total_tokens": total_tokens,
            "total_tokens_formatted": format_token_count(total_tokens),
            "total_time_seconds": total_time,
            "acceptance_rate": accepted_count / len(sessions) if sessions else 0,
            "avg_tokens_per_session": total_tokens / len(sessions) if sessions else 0,
            "avg_iterations_per_session": (
                total_interactions / len(sessions) if sessions else 0
            ),
        },
        "mode_breakdown": {
            "free_form": {
                "session_count": len(ff_sessions),
                "total_tokens": ff_tokens,
                "avg_tokens": ff_tokens / len(ff_sessions) if ff_sessions else 0,
            },
            "snippet_grounded": {
                "session_count": len(gr_sessions),
                "total_tokens": gr_tokens,
                "avg_tokens": gr_tokens / len(gr_sessions) if gr_sessions else 0,
            },
        },
        "quality": {
            "avg_grounding_compliance": avg_grounding,
            "avg_snippet_reference_rate": avg_snippet_ref,
            "total_hallucination_indicators": sum(
                s["hallucination_indicators"] for s in sessions
            ),
        },
        "sessions": [
            {
                "session_id": s["session_id"],
                "problem_id": s["problem_id"],
                "mode": s["mode"],
                "tokens": s["total_tokens"],
                "iterations": s["iterations"],
                "accepted": bool(s["accepted"]),
                "time_seconds": s["time_to_completion_seconds"],
            }
            for s in sessions
        ],
        "recommendations": recommendations,
    }


def _generate_recommendations(
    sessions: List[Dict],
    avg_grounding: float,
    ff_sessions: List[Dict],
    gr_sessions: List[Dict]
) -> List[str]:
    """Generate recommendations based on usage patterns."""
    recommendations = []
    
    # High free-form usage
    if len(ff_sessions) > len(gr_sessions):
        recommendations.append(
            "Consider using Snippet-Grounded mode more often to reduce token usage"
        )
    
    # Low grounding compliance
    if avg_grounding < 0.5 and gr_sessions:
        recommendations.append(
            "Grounding compliance is low - ensure provided snippets are relevant"
        )
    
    # High iteration count
    avg_iterations = (
        sum(s["iterations"] for s in sessions) / len(sessions)
        if sessions else 0
    )
    if avg_iterations > 5:
        recommendations.append(
            "High iteration count detected - consider more focused prompts"
        )
    
    # Token efficiency
    if ff_sessions and gr_sessions:
        ff_avg = sum(s["total_tokens"] for s in ff_sessions) / len(ff_sessions)
        gr_avg = sum(s["total_tokens"] for s in gr_sessions) / len(gr_sessions)
        if ff_avg > gr_avg * 1.5:
            recommendations.append(
                f"Free-form mode uses {((ff_avg/gr_avg - 1) * 100):.0f}% more tokens"
            )
    
    if not recommendations:
        recommendations.append("Usage patterns look optimal!")
    
    return recommendations


def get_variance_stats(
    db: MetricsDatabase,
    problem_id: str
) -> Dict[str, Any]:
    """
    Calculate solution variance statistics for a problem.
    
    Measures how different candidate solutions are from each other,
    which helps evaluate the fairness and consistency of LLM assistance.
    
    Args:
        db: MetricsDatabase instance
        problem_id: The problem identifier
        
    Returns:
        Dictionary with variance statistics:
        - token_variance: Variance in token usage
        - iteration_variance: Variance in iteration counts
        - time_variance: Variance in completion times
        - mode_impact: How mode affects variance
        
    Example:
        >>> stats = get_variance_stats(db, "problem_123")
        >>> print(f"Token variance: {stats['token_variance']:.2f}")
    """
    sessions = db.get_problem_sessions(problem_id)
    
    if len(sessions) < 2:
        return {
            "problem_id": problem_id,
            "error": "Need at least 2 sessions to calculate variance",
            "session_count": len(sessions),
        }
    
    # Extract metrics
    tokens = [s["total_tokens"] for s in sessions]
    iterations = [s["iterations"] for s in sessions]
    times = [s["time_to_completion_seconds"] for s in sessions]
    
    # Calculate variance (using population variance)
    def variance(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def std_dev(values: List[float]) -> float:
        return variance(values) ** 0.5
    
    # By mode
    ff_sessions = [s for s in sessions if s["mode"] == "free_form"]
    gr_sessions = [s for s in sessions if s["mode"] == "snippet_grounded"]
    
    ff_tokens = [s["total_tokens"] for s in ff_sessions]
    gr_tokens = [s["total_tokens"] for s in gr_sessions]
    
    return {
        "problem_id": problem_id,
        "session_count": len(sessions),
        "overall": {
            "token_mean": sum(tokens) / len(tokens),
            "token_std_dev": std_dev(tokens),
            "token_variance": variance(tokens),
            "iteration_mean": sum(iterations) / len(iterations),
            "iteration_std_dev": std_dev(iterations),
            "time_mean": sum(times) / len(times),
            "time_std_dev": std_dev(times),
        },
        "by_mode": {
            "free_form": {
                "count": len(ff_sessions),
                "token_mean": sum(ff_tokens) / len(ff_tokens) if ff_tokens else 0,
                "token_std_dev": std_dev(ff_tokens) if ff_tokens else 0,
            },
            "snippet_grounded": {
                "count": len(gr_sessions),
                "token_mean": sum(gr_tokens) / len(gr_tokens) if gr_tokens else 0,
                "token_std_dev": std_dev(gr_tokens) if gr_tokens else 0,
            },
        },
        "mode_reduces_variance": (
            std_dev(gr_tokens) < std_dev(ff_tokens)
            if ff_tokens and gr_tokens else None
        ),
    }


def get_leaderboard(
    db: MetricsDatabase,
    problem_id: Optional[str] = None,
    metric: str = "efficiency",
    limit: int = 10
) -> List[Dict[str, Any]]:
    """
    Generate a leaderboard based on specified metric.
    
    Creates a ranked list of candidates based on token efficiency,
    acceptance rate, or speed.
    
    Args:
        db: MetricsDatabase instance
        problem_id: Optional filter by specific problem
        metric: Ranking metric - 'efficiency' (tokens), 'speed', or 'quality'
        limit: Maximum number of entries to return
        
    Returns:
        List of leaderboard entries with rank and metrics
        
    Example:
        >>> leaderboard = get_leaderboard(db, metric="efficiency")
        >>> for entry in leaderboard:
        ...     print(f"{entry['rank']}. {entry['candidate_id']}")
    """
    # Get all sessions
    if problem_id:
        sessions = db.get_problem_sessions(problem_id)
    else:
        stats = db.get_overall_stats()
        # For overall, we need to aggregate by candidate
        # This is a simplified version - in production, add a proper query
        sessions = []
    
    if not sessions:
        return []
    
    # Group by candidate
    candidate_metrics: Dict[str, Dict] = {}
    
    for session in sessions:
        cid = session["candidate_id"]
        if cid not in candidate_metrics:
            candidate_metrics[cid] = {
                "total_tokens": 0,
                "total_time": 0,
                "session_count": 0,
                "accepted_count": 0,
                "grounding_sum": 0,
            }
        
        m = candidate_metrics[cid]
        m["total_tokens"] += session["total_tokens"]
        m["total_time"] += session["time_to_completion_seconds"]
        m["session_count"] += 1
        m["accepted_count"] += 1 if session["accepted"] else 0
        m["grounding_sum"] += session["grounding_compliance_rate"]
    
    # Calculate averages and sort
    entries = []
    for cid, m in candidate_metrics.items():
        entry = {
            "candidate_id": cid,
            "session_count": m["session_count"],
            "total_tokens": m["total_tokens"],
            "avg_tokens": m["total_tokens"] / m["session_count"],
            "avg_time": m["total_time"] / m["session_count"],
            "acceptance_rate": m["accepted_count"] / m["session_count"],
            "avg_grounding": m["grounding_sum"] / m["session_count"],
        }
        entries.append(entry)
    
    # Sort based on metric
    if metric == "efficiency":
        # Lower tokens = better
        entries.sort(key=lambda x: x["avg_tokens"])
    elif metric == "speed":
        # Lower time = better
        entries.sort(key=lambda x: x["avg_time"])
    elif metric == "quality":
        # Higher grounding = better
        entries.sort(key=lambda x: -x["avg_grounding"])
    else:
        # Default: efficiency
        entries.sort(key=lambda x: x["avg_tokens"])
    
    # Add rank and limit
    result = []
    for i, entry in enumerate(entries[:limit]):
        entry["rank"] = i + 1
        result.append(entry)
    
    return result


def export_metrics_summary(
    db: MetricsDatabase,
    format: str = "dict"
) -> Dict[str, Any]:
    """
    Export a complete metrics summary for reporting.
    
    Creates a comprehensive export of all metrics suitable for
    dashboards, reports, or external analysis.
    
    Args:
        db: MetricsDatabase instance
        format: Output format ('dict' for now, extensible to 'csv', 'json')
        
    Returns:
        Complete metrics summary dictionary
    """
    overall = db.get_overall_stats()
    mode_comparison = compare_modes(db)
    savings = calculate_savings(db)
    
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "overall_stats": overall,
        "mode_comparison": {
            "free_form_avg_tokens": mode_comparison.free_form_avg_tokens,
            "grounded_avg_tokens": mode_comparison.grounded_avg_tokens,
            "token_reduction_percent": mode_comparison.token_reduction_percent,
            "free_form_acceptance_rate": mode_comparison.free_form_acceptance_rate,
            "grounded_acceptance_rate": mode_comparison.grounded_acceptance_rate,
        },
        "cost_analysis": savings,
        "key_insights": _generate_insights(overall, mode_comparison, savings),
    }


def _generate_insights(
    overall: Dict[str, Any],
    comparison: ModeComparisonResult,
    savings: Dict[str, Any]
) -> List[str]:
    """Generate key insights from the data."""
    insights = []
    
    if comparison.token_reduction_percent > 25:
        insights.append(
            f"Snippet-Grounded mode reduces tokens by "
            f"{comparison.token_reduction_percent:.1f}%"
        )
    
    if savings["potential_additional_savings"] > 0:
        insights.append(
            f"Switching remaining free-form sessions could save "
            f"${savings['potential_additional_savings']:.2f}"
        )
    
    if comparison.grounded_acceptance_rate >= comparison.free_form_acceptance_rate:
        insights.append(
            "Grounded mode maintains similar or better acceptance rates"
        )
    
    if overall["total_sessions"] > 0:
        insights.append(
            f"Processed {overall['total_sessions']} sessions with "
            f"{format_token_count(overall['total_tokens'])} tokens"
        )
    
    return insights

