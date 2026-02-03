"""
Report Generator for LLM Metrics

Generates markdown reports comparing snippet-grounded vs free-form modes.
These reports are designed to be shared with managers and stakeholders.
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "llm_metrics.db"


def get_interaction_stats(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Get interaction statistics grouped by mode."""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            mode,
            COUNT(*) as interactions,
            SUM(total_tokens) as total_tokens,
            ROUND(AVG(total_tokens), 1) as avg_tokens,
            SUM(user_prompt_tokens) as input_tokens,
            SUM(model_response_tokens) as output_tokens,
            ROUND(AVG(model_response_tokens), 1) as avg_output_tokens,
            ROUND(AVG(grounding_compliance_rate), 3) as avg_grounding
        FROM interactions
        GROUP BY mode
    """)
    
    results = {}
    for row in cursor.fetchall():
        results[row["mode"]] = dict(row)
    
    conn.close()
    return results


def get_overall_stats(db_path: Path = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """Get overall statistics."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT 
            COUNT(*) as total_interactions,
            SUM(total_tokens) as total_tokens,
            COUNT(DISTINCT candidate_id) as unique_candidates,
            COUNT(DISTINCT problem_id) as unique_problems
        FROM interactions
    """)
    
    row = cursor.fetchone()
    conn.close()
    
    return {
        "total_interactions": row[0] or 0,
        "total_tokens": row[1] or 0,
        "unique_candidates": row[2] or 0,
        "unique_problems": row[3] or 0,
    }


def calculate_cost_savings(
    free_form_tokens: int,
    grounded_tokens: int,
    free_form_count: int,
    grounded_count: int,
    input_cost_per_1k: float = 0.03,
    output_cost_per_1k: float = 0.06
) -> Dict[str, float]:
    """Calculate estimated cost savings."""
    if free_form_count == 0 or grounded_count == 0:
        return {
            "estimated_savings_percent": 0,
            "estimated_cost_free_form": 0,
            "estimated_cost_grounded": 0,
        }
    
    avg_ff = free_form_tokens / free_form_count
    avg_gr = grounded_tokens / grounded_count
    
    # Estimate cost per interaction (simplified - assuming 60% input, 40% output)
    cost_per_ff = (avg_ff * 0.6 * input_cost_per_1k + avg_ff * 0.4 * output_cost_per_1k) / 1000
    cost_per_gr = (avg_gr * 0.6 * input_cost_per_1k + avg_gr * 0.4 * output_cost_per_1k) / 1000
    
    savings_percent = ((avg_ff - avg_gr) / avg_ff * 100) if avg_ff > 0 else 0
    
    return {
        "avg_tokens_free_form": avg_ff,
        "avg_tokens_grounded": avg_gr,
        "estimated_savings_percent": round(savings_percent, 1),
        "cost_per_interaction_free_form": round(cost_per_ff, 4),
        "cost_per_interaction_grounded": round(cost_per_gr, 4),
    }


def generate_markdown_report(
    db_path: Path = DEFAULT_DB_PATH,
    report_title: str = "LLM Usage Metrics Report"
) -> str:
    """
    Generate a markdown report comparing snippet-grounded vs free-form modes.
    
    Args:
        db_path: Path to the metrics database
        report_title: Title for the report
        
    Returns:
        Markdown formatted report string
    """
    # Gather data
    mode_stats = get_interaction_stats(db_path)
    overall = get_overall_stats(db_path)
    
    free_form = mode_stats.get("free_form", {})
    grounded = mode_stats.get("snippet_grounded", {})
    
    ff_tokens = free_form.get("total_tokens", 0) or 0
    gr_tokens = grounded.get("total_tokens", 0) or 0
    ff_count = free_form.get("interactions", 0) or 0
    gr_count = grounded.get("interactions", 0) or 0
    
    cost_analysis = calculate_cost_savings(ff_tokens, gr_tokens, ff_count, gr_count)
    
    # Calculate key metrics
    total_interactions = ff_count + gr_count
    grounded_percentage = (gr_count / total_interactions * 100) if total_interactions > 0 else 0
    
    # Token comparison
    ff_avg = cost_analysis.get("avg_tokens_free_form", 0)
    gr_avg = cost_analysis.get("avg_tokens_grounded", 0)
    token_diff = ff_avg - gr_avg
    token_diff_pct = cost_analysis.get("estimated_savings_percent", 0)
    
    # Generate report
    report = f"""# {report_title}

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

---

## Executive Summary

This report compares LLM token usage between **Free-Form** mode (unrestricted AI assistance) and **Snippet-Grounded** mode (AI constrained to provided code snippets).

### Key Findings

| Metric | Value |
|--------|-------|
| Total Interactions | {total_interactions:,} |
| Total Tokens Used | {overall['total_tokens']:,} |
| Unique Candidates | {overall['unique_candidates']} |
| Unique Problems | {overall['unique_problems']} |

---

## Mode Comparison

### Interaction Distribution

| Mode | Interactions | Percentage |
|------|-------------|------------|
| **Free-Form** | {ff_count:,} | {(ff_count/total_interactions*100) if total_interactions > 0 else 0:.1f}% |
| **Snippet-Grounded** | {gr_count:,} | {grounded_percentage:.1f}% |

### Token Usage Comparison

| Metric | Free-Form | Snippet-Grounded | Difference |
|--------|-----------|------------------|------------|
| **Total Tokens** | {ff_tokens:,} | {gr_tokens:,} | - |
| **Avg Tokens/Interaction** | {ff_avg:,.1f} | {gr_avg:,.1f} | {token_diff:+,.1f} ({token_diff_pct:+.1f}%) |
| **Input Tokens** | {free_form.get('input_tokens', 0):,} | {grounded.get('input_tokens', 0):,} | - |
| **Output Tokens** | {free_form.get('output_tokens', 0):,} | {grounded.get('output_tokens', 0):,} | - |
| **Avg Output Tokens** | {free_form.get('avg_output_tokens', 0):,.1f} | {grounded.get('avg_output_tokens', 0):,.1f} | - |

---

## Cost Analysis

Based on typical LLM pricing (GPT-4 rates: $0.03/1K input, $0.06/1K output):

| Metric | Free-Form | Snippet-Grounded |
|--------|-----------|------------------|
| **Est. Cost per Interaction** | ${cost_analysis.get('cost_per_interaction_free_form', 0):.4f} | ${cost_analysis.get('cost_per_interaction_grounded', 0):.4f} |

### Projected Savings

"""
    
    # Add savings projection if we have data
    if ff_count > 0 and gr_count > 0:
        savings_per_100 = (cost_analysis['cost_per_interaction_free_form'] - cost_analysis['cost_per_interaction_grounded']) * 100
        savings_per_1000 = savings_per_100 * 10
        
        report += f"""| Scale | If All Free-Form | If All Snippet-Grounded | Savings |
|-------|-----------------|------------------------|---------|
| 100 interactions | ${cost_analysis['cost_per_interaction_free_form'] * 100:.2f} | ${cost_analysis['cost_per_interaction_grounded'] * 100:.2f} | ${savings_per_100:.2f} |
| 1,000 interactions | ${cost_analysis['cost_per_interaction_free_form'] * 1000:.2f} | ${cost_analysis['cost_per_interaction_grounded'] * 1000:.2f} | ${savings_per_1000:.2f} |
| 10,000 interactions | ${cost_analysis['cost_per_interaction_free_form'] * 10000:.2f} | ${cost_analysis['cost_per_interaction_grounded'] * 10000:.2f} | ${savings_per_1000 * 10:.2f} |

"""
    else:
        report += """*Insufficient data to calculate savings projection. Need interactions in both modes.*

"""
    
    # Quality metrics
    report += f"""---

## Quality Metrics

| Metric | Free-Form | Snippet-Grounded |
|--------|-----------|------------------|
| **Avg Grounding Score** | N/A | {grounded.get('avg_grounding', 0):.1%} |

> **Grounding Score**: Measures how well the LLM response references the provided code snippets. Higher is better.

---

## Benefits of Snippet-Grounded Mode

### Cost Control
- **~{abs(token_diff_pct):.0f}% token reduction** per interaction
- Predictable usage budgets for large events
- Lower infrastructure costs at scale

### Evaluation Fairness
- More comparable outputs across candidates
- Lower variance in AI assistance
- Better signal of real coding skills

### Technical Reliability
- Reduced hallucinations (AI inventing code)
- Context-anchored answers
- Fewer irrelevant generations

---

## Recommendations

"""
    
    if token_diff_pct > 0:
        report += f"""1. **Default to Snippet-Grounded mode** for hiring hackathons to maximize cost savings ({token_diff_pct:.0f}% reduction)
2. **Use Free-Form mode** only for practice/learning sessions
3. **Monitor grounding scores** to ensure snippet quality
"""
    else:
        report += """1. **Collect more data** in both modes for accurate comparison
2. **Ensure snippets are relevant** to reduce input token overhead
3. **Monitor output token counts** as the primary savings indicator
"""
    
    report += f"""
---

## Data Sources

- **Database**: `{db_path}`
- **Report Generated By**: LLM Metrics Proxy System
- **Timestamp**: {datetime.now().isoformat()}

---

*This report was automatically generated. For questions, contact the platform team.*
"""
    
    return report


def save_report(
    output_path: Path,
    db_path: Path = DEFAULT_DB_PATH,
    report_title: str = "LLM Usage Metrics Report"
) -> Path:
    """
    Generate and save a markdown report to a file.
    
    Args:
        output_path: Path where the report will be saved
        db_path: Path to the metrics database
        report_title: Title for the report
        
    Returns:
        Path to the saved report file
    """
    report = generate_markdown_report(db_path, report_title)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    
    return output_path
