# LLM Metrics - User Guide

This guide explains how to configure Cline to work with the LLM Metrics proxy and how to switch between assistance modes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Cline Configuration](#cline-configuration)
3. [Key Concepts](#key-concepts)
4. [Switching Modes](#switching-modes)
5. [API Reference](#api-reference)
6. [Testing & Verification](#testing--verification)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Start the Proxy Server

```bash
cd /path/to/llmMetrics
PYTHONPATH=/path/to/parent python3 -m uvicorn llmMetrics.proxy.server:app --host 0.0.0.0 --port 8000
```

### 2. Configure Cline

```
API Provider: OpenAI Compatible
Base URL: http://localhost:8000/v1
API Key: your-api-key
```

### 3. Start Using Cline

That's it! The proxy will track all your LLM interactions.

---

## Cline Configuration

### Step 1: Open Cline Settings

In VS Code/Cursor, open Cline settings (gear icon in Cline panel).

### Step 2: Configure API Settings

| Setting | Value |
|---------|-------|
| **API Provider** | `OpenAI Compatible` |
| **Base URL** | `http://localhost:8000/v1` |
| **API Key** | Your OpenAI/Org API key |
| **Model ID** | `gpt-4o`, `claude-3-5-sonnet`, etc. |

### Step 3: Add Optional Headers (Recommended)

In the "Custom Headers" section, add:

```
X-Candidate-ID: your-username
X-Assistance-Mode: snippet
```

### Header Reference

| Header | Required | Description |
|--------|----------|-------------|
| `X-Candidate-ID` | Optional | Your unique identifier for tracking |
| `X-Problem-ID` | Optional | Which problem you're working on |
| `X-Assistance-Mode` | Optional | `snippet` or `free` to override mode |

---

## Key Concepts

### What is a Candidate?

A **Candidate** is a user of the system. In a hackathon context, this would be a participant. The `X-Candidate-ID` header identifies who is making the request.

**Examples:**
- `X-Candidate-ID: john.doe`
- `X-Candidate-ID: team-alpha-member-1`
- `X-Candidate-ID: participant-42`

**Why it matters:**
- Tracks token usage per user
- Enables per-user mode settings
- Generates per-candidate reports

---

### What is a Problem ID?

A **Problem ID** is a unique identifier for a coding problem or task. Each problem can have:

- A **mode** (snippet_grounded or free_form)
- **Code snippets** that candidates should use
- A **description** of the task

**Examples:**
- `X-Problem-ID: algorithm-challenge-1`
- `X-Problem-ID: math-operations`
- `X-Problem-ID: interview-q3`

**Why it matters:**
- Different problems can have different modes
- Snippets are loaded based on the problem
- Metrics are grouped by problem for analysis

---

### What are the Two Modes?

#### 1. Snippet-Grounded Mode (`snippet`)

The LLM is constrained to work with provided code snippets.

| Feature | Behavior |
|---------|----------|
| Code Snippets | Injected into every prompt |
| Max Tokens | Limited (default: 500) |
| LLM Behavior | References provided code, doesn't create from scratch |
| Best For | Hackathons, interviews, controlled environments |

#### 2. Free-Form Mode (`free`)

The LLM has no restrictions.

| Feature | Behavior |
|---------|----------|
| Code Snippets | Not injected |
| Max Tokens | Unlimited |
| LLM Behavior | Can generate anything |
| Best For | Learning, exploration, practice |

---

## Switching Modes

### Method 1: Via HTTP Header (Per Request)

Add this header in Cline settings:

**For Snippet Mode:**
```
X-Assistance-Mode: snippet
```

**For Free Mode:**
```
X-Assistance-Mode: free
```

### Method 2: Via API (Persistent)

Switch modes without changing Cline settings:

```bash
# Switch to snippet mode
curl -X POST "http://localhost:8000/api/mode/set" \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "your-username", "mode": "snippet"}'

# Switch to free mode
curl -X POST "http://localhost:8000/api/mode/set" \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "your-username", "mode": "free"}'
```

### Method 3: Toggle Problem's Default Mode

```bash
# Toggle between modes for a specific problem
curl -X POST "http://localhost:8000/api/problems/my-problem/toggle-mode"
```

### Mode Priority

When determining which mode to use, the system checks (in order):

1. `X-Assistance-Mode` header (highest priority)
2. API mode override (`/api/mode/set`)
3. Problem's default mode
4. Fallback: `free_form`

---

## API Reference

### Mode Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/mode` | GET | View available modes and settings |
| `/api/mode/set` | POST | Set mode for a candidate |
| `/api/mode/{candidate_id}` | GET | Get current mode for a candidate |
| `/api/mode/{candidate_id}` | DELETE | Clear mode override |

### Problem Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/problems` | GET | List all problems |
| `/api/problems` | POST | Create a new problem |
| `/api/problems/{id}` | GET | Get problem details |
| `/api/problems/{id}/toggle-mode` | POST | Toggle problem's mode |
| `/api/problems/from-snippets` | POST | Create problem from snippet files |

### Metrics & Reports

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/metrics/overview` | GET | Overall metrics summary |
| `/api/metrics/comparison` | GET | Compare modes |
| `/api/reports/comparison` | GET | Get markdown report |
| `/api/reports/comparison/save` | POST | Save report to file |

### Configuration

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/config` | GET | View current configuration |
| `/api/config/reload` | POST | Reload config from file |

---

## Testing & Verification

### 1. Verify Server is Running

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "target_api": "https://api.openai.com/v1", "active_sessions": 0}
```

### 2. Check Your Current Mode

```bash
curl http://localhost:8000/api/mode/your-username
```

### 3. Test Snippet Mode

1. Set mode to snippet:
   ```bash
   curl -X POST "http://localhost:8000/api/mode/set" \
     -H "Content-Type: application/json" \
     -d '{"candidate_id": "your-username", "mode": "snippet"}'
   ```

2. Use Cline to ask a question
3. Notice the response references provided code snippets

### 4. Test Free Mode

1. Set mode to free:
   ```bash
   curl -X POST "http://localhost:8000/api/mode/set" \
     -H "Content-Type: application/json" \
     -d '{"candidate_id": "your-username", "mode": "free"}'
   ```

2. Ask the same question in Cline
3. Notice the response is unrestricted

### 5. Generate Comparison Report

```bash
curl -X POST "http://localhost:8000/api/reports/comparison/save"
```

Check the report in `llmMetrics/reports/` directory.

---

## Troubleshooting

### "Connection Refused" Error

**Problem:** Cline can't connect to the proxy.

**Solution:**
1. Ensure the server is running: `curl http://localhost:8000/health`
2. Check the port is correct (8000)
3. Verify Base URL in Cline: `http://localhost:8000/v1`

### Snippets Not Being Injected

**Problem:** In snippet mode, but LLM isn't referencing the code.

**Solution:**
1. Check mode is set correctly: `curl http://localhost:8000/api/mode/your-username`
2. Verify problem has snippets: `curl http://localhost:8000/api/problems/your-problem`
3. Ensure `X-Problem-ID` header is set in Cline

### High Token Usage in Snippet Mode

**Problem:** Snippet mode is using more tokens than expected.

**Solution:**
1. Use smaller snippets (function signatures only, not full implementations)
2. Reduce the number of snippets per problem
3. Lower `max_tokens` in `config.yaml`

### Mode Not Changing

**Problem:** Changed mode but behavior didn't change.

**Solution:**
1. Header takes priority - check `X-Assistance-Mode` header in Cline
2. Clear API override: `curl -X DELETE http://localhost:8000/api/mode/your-username`
3. Reload config: `curl -X POST http://localhost:8000/api/config/reload`

---

## Configuration File

Settings are stored in `llmMetrics/config.yaml`:

```yaml
# Snippet-grounded mode settings
snippet_grounded_mode:
  max_tokens: 500        # Limit response length
  strictness: "moderate" # strict, moderate, or light

# Free-form mode settings  
free_form_mode:
  max_tokens: null       # No limit

# Proxy settings
proxy:
  target_url: "https://api.openai.com/v1"  # Your org URL
```

After editing, reload without restart:
```bash
curl -X POST http://localhost:8000/api/config/reload
```

---

## Database & Maintenance Commands

### Reset Database (Start Fresh)

Remove all metrics and problem configurations:

```bash
# Stop the server first (Ctrl+C or):
pkill -f uvicorn

# Delete metrics database (all interaction history)
rm -f llmMetrics/data/llm_metrics.db

# Delete problem config database (all problems and assignments)
rm -f llmMetrics/data/hackathon_config.db

# Or delete both at once
rm -f llmMetrics/data/*.db

# Restart server (databases recreated automatically)
python run.py
```

### One-Liner: Full Reset

```bash
pkill -f uvicorn; rm -f llmMetrics/data/*.db; python run.py
```

### After Reset: Re-create Problems

```bash
# Create a problem with snippets
curl -X POST "http://localhost:8000/api/problems/from-snippets" \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "math-test",
    "title": "Math Operations",
    "snippet_paths": ["math_operations/add", "math_operations/subtract"]
  }'

# Set mode for your user
curl -X POST "http://localhost:8000/api/mode/set" \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "your-username", "mode": "snippet"}'
```

### View Database Location

Databases are stored in:
```
llmMetrics/data/
  ├── llm_metrics.db      # Interaction history, token counts
  └── hackathon_config.db # Problems, snippets, assignments
```

### Backup Databases

```bash
# Create backup
cp llmMetrics/data/llm_metrics.db llmMetrics/data/llm_metrics_backup.db
cp llmMetrics/data/hackathon_config.db llmMetrics/data/hackathon_config_backup.db
```

### Server Management

| Command | Description |
|---------|-------------|
| `python run.py` | Start server with config.yaml settings |
| `python run.py --port 3000` | Start on custom port |
| `python run.py --reload` | Start with auto-reload (development) |
| `pkill -f uvicorn` | Stop the server |
| `curl http://localhost:8000/health` | Check if server is running |

---

## Summary

| Task | Command/Action |
|------|----------------|
| Start server | `python3 -m uvicorn llmMetrics.proxy.server:app --port 8000` |
| Switch to snippet mode | `X-Assistance-Mode: snippet` header or `/api/mode/set` |
| Switch to free mode | `X-Assistance-Mode: free` header or `/api/mode/set` |
| Check current mode | `GET /api/mode/{candidate_id}` |
| Generate report | `POST /api/reports/comparison/save` |
| View config | `GET /api/config` |

---

*For technical details, see the main [README.md](../README.md)*
