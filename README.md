# LLM Metrics RAG - Semantic Snippet Retrieval

A Python library for tracking LLM usage in hackathon platforms with **RAG-based snippet retrieval** (Retrieval-Augmented Generation). Uses semantic search to find the most relevant code snippets, dramatically reducing token usage.

## 🚀 Key Feature: RAG-based Snippet Selection

Instead of sending ALL snippets with every request, RAG uses semantic search to find only the 1-2 most relevant snippets:

| Approach | Snippets Sent | Tokens per Request |
|----------|---------------|-------------------|
| All snippets (baseline) | 50+ | ~35,000 |
| Smart selection (keyword) | 3-5 | ~2,000 |
| **RAG (semantic search)** | 1-2 | **~500** |

**Token reduction: 95%+ compared to baseline!**

---

## Table of Contents

- [Core Concept: The Two Assistance Modes](#1-core-concept-the-two-assistance-modes)
- [Module Breakdown](#2-module-breakdown)
  - [models.py - Data Structures](#21-modelspy---data-structures)
  - [database.py - SQLite Storage](#22-databasepy---sqlite-storage)
  - [utils.py - Token Counting & Grounding Analysis](#23-utilspy---token-counting--grounding-analysis)
  - [tracker.py - The Main Tracking Class](#24-trackerpy---the-main-tracking-class)
  - [analyzers.py - Reporting & Analysis](#25-analyzerspy---reporting--analysis)
- [Complete Data Flow Diagram](#3-complete-data-flow-diagram)
- [Key Metrics Captured](#4-key-metrics-captured)
- [Installation & Usage](#5-installation--usage)
- [Quick Start Example](#6-quick-start-example)

---

## 1. Core Concept: The Two Assistance Modes

The system tracks LLM usage across two modes:

| Mode | Behavior | Token Impact |
|------|----------|--------------|
| **FREE_FORM** | LLM can generate full solutions, propose designs, no restrictions | High tokens (~100% baseline) |
| **SNIPPET_GROUNDED** | LLM must work only with provided code snippets, no full solutions | Lower tokens (~25-40% reduction) |

---

## 2. Module Breakdown

### 2.1 `models.py` - Data Structures

This defines all the data containers:

#### AssistanceMode Enum

```python
class AssistanceMode(Enum):
    FREE_FORM = "free_form"
    SNIPPET_GROUNDED = "snippet_grounded"
```

#### TokenMetrics

Tracks token counts for a single interaction:

```python
@dataclass
class TokenMetrics:
    problem_statement_tokens: int = 0   # Tokens in problem description
    snippet_tokens: int = 0              # Tokens in provided code snippets
    user_prompt_tokens: int = 0          # Tokens in user's question
    model_response_tokens: int = 0       # Tokens in LLM's response
    
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed (input + output)."""
        return self.total_input_tokens + self.model_response_tokens
```

#### LLMInteraction

One complete LLM call (prompt + response):

```python
@dataclass
class LLMInteraction:
    session_id: str          # Which session this belongs to
    candidate_id: str        # Who made this call
    problem_id: str          # Which problem they're solving
    mode: AssistanceMode     # Current mode
    prompt: str              # What user asked
    response: str            # What LLM replied
    token_metrics: TokenMetrics
    quality_metrics: QualityMetrics  # Grounding scores
```

#### CandidateSession

Groups all interactions for one candidate on one problem:

```python
@dataclass
class CandidateSession:
    candidate_id: str
    problem_id: str
    mode: AssistanceMode
    interactions: List[LLMInteraction]  # All LLM calls in this session
    total_token_metrics: TokenMetrics    # Aggregated tokens
    efficiency_metrics: EfficiencyMetrics
```

---

### 2.2 `database.py` - SQLite Storage

Three tables store the data:

#### Table: `interactions`

| Column | Description |
|--------|-------------|
| `interaction_id` (PK) | Individual LLM call |
| `session_id` (FK) | Links to session |
| `candidate_id` | Who made the call |
| `prompt`, `response` | The actual text |
| `*_tokens` columns | All token counts |
| `grounding_*` | Quality metrics |

#### Table: `sessions`

| Column | Description |
|--------|-------------|
| `session_id` (PK) | One candidate + one problem |
| `candidate_id` | The candidate |
| `problem_id` | The problem |
| `mode` | free_form or snippet_grounded |
| `total_tokens` | Sum of all interaction tokens |
| `iterations` | Number of LLM calls |
| `accepted` | Did they solve it? |

#### Table: `problems`

| Column | Description |
|--------|-------------|
| `problem_id` (PK) | Aggregated per-problem stats |
| `avg_tokens_free_form` | Average tokens in free-form mode |
| `avg_tokens_grounded` | Average tokens in grounded mode |
| `acceptance_rate_*` | Success rates per mode |

#### Key Methods

```python
db = MetricsDatabase("./metrics.db")

# Save a single interaction
db.save_interaction(interaction)

# Save session summary
db.save_session(session)

# Query: Compare modes
db.get_mode_comparison()  # Returns avg tokens per mode

# Query: Get candidate history
db.get_candidate_sessions("candidate_123")
```

---

### 2.3 `utils.py` - Token Counting & Grounding Analysis

#### Token Counting

Uses `tiktoken` (OpenAI's library) for accurate counting:

```python
from llmMetrics import count_tokens

count_tokens("Hello, world!")  # Returns: 4
count_tokens("def foo():\n    return 42")  # Returns: ~10
```

#### Grounding Score

Measures how well the LLM response references provided snippets:

```python
from llmMetrics import calculate_grounding_score

snippet = "def calculate_sum(numbers):\n    return sum(numbers)"
response = "You can modify calculate_sum to handle empty lists"

score = calculate_grounding_score(response, [snippet])
# Returns: 0.8 (high - response references the snippet)
```

**The algorithm:**

1. Extract identifiers from snippets (function/variable names)
2. Check how many identifiers appear in the response
3. Bonus points if response code blocks reuse snippet patterns
4. Score from 0.0 (no grounding) to 1.0 (fully grounded)

#### Hallucination Detection

Flags potential invented code:

```python
from llmMetrics import detect_hallucination_indicators

# If response calls functions that don't exist in snippets
indicators = detect_hallucination_indicators(response, snippets)
# Returns: count of suspicious function calls
```

---

### 2.4 `tracker.py` - The Main Tracking Class

This is what you use in your application:

```python
from llmMetrics import LLMMetricsTracker, AssistanceMode

# Initialize (creates DB automatically)
tracker = LLMMetricsTracker(
    db_path="./metrics.db",      # Where to store data
    model="gpt-4",               # For accurate token counting
    default_mode=AssistanceMode.SNIPPET_GROUNDED
)
```

#### Lifecycle of a Session

```
┌──────────────────┐
│ start_session()  │ ─── Creates CandidateSession object
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│   set_mode()     │ ─── Switch between FREE_FORM / SNIPPET_GROUNDED
└────────┬─────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ record_interaction(prompt, response, snippets)           │
│                                                          │
│   1. Count tokens (prompt, response, snippets)           │
│   2. Calculate grounding score (if snippets provided)    │
│   3. Detect hallucination indicators                     │
│   4. Create LLMInteraction object                        │
│   5. Add to session, save to database                    │
└────────┬─────────────────────────────────────────────────┘
         │  (repeat for each LLM call)
         ▼
┌──────────────────┐
│  end_session()   │ ─── Finalize, aggregate metrics, save to DB
└──────────────────┘
```

#### Example Flow

```python
# 1. Start session
session_id = tracker.start_session(
    candidate_id="user_123",
    problem_id="binary_search_problem",
    problem_statement="Implement binary search..."
)

# 2. Set mode (grounded = recommended)
tracker.set_mode(AssistanceMode.SNIPPET_GROUNDED)

# 3. Record each LLM interaction
tracker.record_interaction(
    prompt="How do I handle the edge case when array is empty?",
    response="Add a check at the start: if not arr: return -1",
    snippets=["def binary_search(arr, target):\n    left, right = 0, len(arr)-1"]
)

# 4. More interactions...
tracker.record_interaction(
    prompt="What if target is at index 0?",
    response="Your mid calculation handles this...",
    snippets=["def binary_search(arr, target):..."]
)

# 5. End session
summary = tracker.end_session(accepted=True)
```

#### What `summary` Contains

```python
{
    "session_id": "abc-123-...",
    "candidate_id": "user_123",
    "problem_id": "binary_search_problem",
    "mode": "snippet_grounded",
    "interaction_count": 2,
    "total_tokens": 450,
    "total_input_tokens": 300,
    "total_response_tokens": 150,
    "iterations": 2,
    "time_to_completion_seconds": 120.5,
    "accepted": True,
    "grounding_compliance_rate": 0.85,
}
```

---

### 2.5 `analyzers.py` - Reporting & Analysis

#### Compare Modes

See token savings:

```python
from llmMetrics import compare_modes

result = compare_modes(tracker.db)

print(f"Free-Form avg tokens: {result.free_form_avg_tokens}")
print(f"Grounded avg tokens: {result.grounded_avg_tokens}")
print(f"Token reduction: {result.token_reduction_percent:.1f}%")
# Example: "Token reduction: 32.5%"
```

#### Calculate Cost Savings

```python
from llmMetrics import calculate_savings

savings = calculate_savings(tracker.db)
print(f"Actual cost: ${savings['actual_cost']}")
print(f"If all free-form: ${savings['all_free_form_cost']}")
print(f"Saved by grounded: ${savings['savings_from_grounded']}")
```

#### Generate Candidate Report

```python
from llmMetrics import generate_report

report = generate_report(tracker.db, "candidate_123")

# Contains:
# - summary: total tokens, sessions, acceptance rate
# - mode_breakdown: free-form vs grounded usage
# - quality: grounding compliance, hallucinations
# - recommendations: based on usage patterns
```

#### Variance Analysis

Measure solution consistency:

```python
from llmMetrics import get_variance_stats

stats = get_variance_stats(tracker.db, "problem_456")

# Shows token variance across candidates
# Lower variance in grounded mode = more fair evaluation
```

---

## 3. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      HACKATHON PLATFORM                              │
│                                                                      │
│   Candidate makes LLM request                                        │
│         │                                                            │
│         ▼                                                            │
│   ┌─────────────────────────────────────────────────────────┐       │
│   │  tracker.record_interaction(prompt, response, snippets)  │       │
│   └────────────────────────┬────────────────────────────────┘       │
└────────────────────────────┼────────────────────────────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │            utils.py                    │
         │                                        │
         │  count_tokens(prompt) ──────────┐     │
         │  count_tokens(response) ────────┤     │
         │  estimate_snippet_tokens() ─────┤     │
         │  calculate_grounding_score() ───┤     │
         │  detect_hallucinations() ───────┘     │
         └───────────────────┬───────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │            models.py                   │
         │                                        │
         │  TokenMetrics(...)                     │
         │  QualityMetrics(...)                   │
         │  LLMInteraction(...)                   │
         └───────────────────┬───────────────────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │           database.py                  │
         │                                        │
         │  save_interaction() ──► interactions   │
         │  save_session() ──────► sessions       │
         │  update_problem_stats()► problems      │
         └───────────────────┬───────────────────┘
                             │
                             ▼
                     ┌───────────────┐
                     │  SQLite DB    │
                     │               │
                     │ interactions  │
                     │ sessions      │
                     │ problems      │
                     └───────┬───────┘
                             │
                             ▼
         ┌───────────────────────────────────────┐
         │          analyzers.py                  │
         │                                        │
         │  compare_modes() ────► Mode comparison │
         │  calculate_savings()─► Cost analysis   │
         │  generate_report() ──► Candidate report│
         │  get_variance_stats()► Fairness stats  │
         └───────────────────────────────────────┘
```

---

## 4. Key Metrics Captured

| Category | Metric | Description |
|----------|--------|-------------|
| **Token** | `problem_statement_tokens` | Tokens in problem description |
| **Token** | `snippet_tokens` | Tokens in provided code |
| **Token** | `user_prompt_tokens` | Tokens in user's question |
| **Token** | `model_response_tokens` | Tokens in LLM reply |
| **Token** | `total_tokens` | Sum of all tokens |
| **Efficiency** | `iterations` | Number of LLM calls |
| **Efficiency** | `time_to_completion` | Seconds to solve |
| **Efficiency** | `tokens_per_submission` | Average tokens per attempt |
| **Quality** | `grounding_compliance_rate` | How well response uses snippets (0-1) |
| **Quality** | `snippet_reference_rate` | How often snippets are referenced |
| **Quality** | `hallucination_indicators` | Count of invented code |

---

## 5. Installation & Usage

```bash
cd llmMetricsRAG
pip install -r requirements.txt

# Install RAG dependencies
pip install chromadb sentence-transformers
```

### RAG Setup (One-time)

```bash
# Index all snippets for semantic search
curl -X POST http://localhost:8000/api/rag/index

# Verify RAG is working
curl http://localhost:8000/api/rag/status
```

### RAG API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/rag/index` | POST | Index snippets for RAG (run once) |
| `/api/rag/search?query=...` | GET | Search snippets semantically |
| `/api/rag/status` | GET | Check RAG system status |

### RAG Configuration

In `config.yaml`:
```yaml
snippet_grounded_mode:
  use_rag: true          # Enable RAG (semantic search)
  max_snippets: 2        # Only return top 2 matches
  min_match_score: 0.3   # Minimum similarity score
```

Then in your code:

```python
from llmMetrics import (
    LLMMetricsTracker,
    AssistanceMode,
    compare_modes,
    generate_report,
)

# Create tracker
tracker = LLMMetricsTracker()

# Use context manager for automatic cleanup
with tracker.session("user_123", "problem_456") as session_id:
    tracker.record_interaction(
        prompt="Help me debug",
        response="The issue is...",
        snippets=["def buggy_function(): ..."]
    )

# Analyze results
comparison = compare_modes(tracker.db)
print(f"Grounded mode saves {comparison.token_reduction_percent:.1f}% tokens")
```

---

## 6. Quick Start Example

```python
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
```

---

## License

MIT License
