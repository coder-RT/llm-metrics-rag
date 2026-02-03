# Cline Integration Setup Guide

This guide explains how to configure Cline (and other LLM-based coding tools) to work with the LLM Metrics Proxy for hackathon platforms.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuring Cline](#configuring-cline)
- [Admin API Usage](#admin-api-usage)
- [Environment Variables](#environment-variables)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

---

## Overview

The LLM Metrics Proxy intercepts API calls from Cline to capture usage metrics and optionally enforce snippet-grounded mode.

```
┌─────────────────┐     ┌─────────────────────┐     ┌─────────────────┐
│     Cline       │────►│   Metrics Proxy     │────►│   OpenAI/       │
│  (VS Code)      │◄────│   (localhost:8000)  │◄────│   Anthropic     │
└─────────────────┘     └──────────┬──────────┘     └─────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │   SQLite Database   │
                        │   - Metrics         │
                        │   - Configurations  │
                        └─────────────────────┘
```

---

## Architecture

### Components

| Component | Purpose |
|-----------|---------|
| **Proxy Server** | Intercepts LLM API calls |
| **Config Store** | Manages problem toggles and snippets |
| **Metrics Tracker** | Records token usage and quality metrics |
| **Admin API** | REST API for managing problems and viewing metrics |

### How Interception Works

1. **Cline** is configured to use the proxy as its API endpoint
2. When a candidate asks a question, **Cline sends request to proxy**
3. Proxy checks the **problem configuration** (snippet_grounded vs free_form)
4. If snippet mode: **injects grounding prompt + code snippets**
5. Proxy **forwards to actual LLM** (OpenAI/Anthropic)
6. Proxy **captures response and records metrics**
7. **Returns response to Cline**

---

## Quick Start

### 1. Install Dependencies

```bash
cd llmMetrics
pip install -r requirements.txt
pip install -r proxy/requirements.txt
```

### 2. Set Environment Variables

```bash
# Your actual LLM API key
export OPENAI_API_KEY="sk-..."
# OR for Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
```

### 3. Start the Proxy Server

```bash
# Option 1: Using uvicorn directly
uvicorn llmMetrics.proxy.server:app --host 0.0.0.0 --port 8000

# Option 2: Using Python
python -m llmMetrics.proxy.server
```

### 4. Create a Problem Configuration

```bash
curl -X POST http://localhost:8000/api/problems \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "binary_search",
    "title": "Implement Binary Search",
    "mode": "snippet_grounded",
    "snippets": [
      "def binary_search(arr: list, target: int) -> int:\n    # TODO: Implement binary search\n    pass"
    ],
    "description": "Implement an efficient binary search algorithm"
  }'
```

### 5. Assign Candidate to Problem

```bash
curl -X POST http://localhost:8000/api/candidates/assign \
  -H "Content-Type: application/json" \
  -d '{
    "candidate_id": "candidate_123",
    "problem_id": "binary_search"
  }'
```

### 6. Configure Cline to Use Proxy

See [Configuring Cline](#configuring-cline) section below.

---

## Configuring Cline

### Option 1: VS Code Settings (Recommended)

Open VS Code settings (`Cmd+,` or `Ctrl+,`) and search for "Cline". Set the following:

```json
{
  "cline.apiProvider": "openai",
  "cline.openaiBaseUrl": "http://localhost:8000/v1",
  "cline.openaiApiKey": "your-actual-openai-key"
}
```

### Option 2: Settings JSON File

Add to `.vscode/settings.json`:

```json
{
  "cline.apiProvider": "openai",
  "cline.openaiBaseUrl": "http://localhost:8000/v1",
  "cline.openaiApiKey": "${env:OPENAI_API_KEY}"
}
```

### Option 3: For Anthropic/Claude

```json
{
  "cline.apiProvider": "anthropic",
  "cline.anthropicBaseUrl": "http://localhost:8000/v1",
  "cline.anthropicApiKey": "your-anthropic-key"
}
```

### Adding Candidate/Problem Headers

To automatically include candidate and problem IDs, you can configure custom headers (if Cline supports it) or use environment variables:

```bash
# In terminal before starting VS Code
export CANDIDATE_ID="candidate_123"
export PROBLEM_ID="binary_search"
```

The proxy also supports candidate assignment via the Admin API, so the candidate ID alone is sufficient.

---

## Admin API Usage

### Problem Management

#### Create Problem

```bash
curl -X POST http://localhost:8000/api/problems \
  -H "Content-Type: application/json" \
  -d '{
    "problem_id": "two_sum",
    "title": "Two Sum",
    "mode": "snippet_grounded",
    "snippets": [
      "def two_sum(nums: list, target: int) -> list:\n    pass"
    ]
  }'
```

#### List All Problems

```bash
curl http://localhost:8000/api/problems
```

#### Get Specific Problem

```bash
curl http://localhost:8000/api/problems/two_sum
```

#### Toggle Mode (Switch between snippet_grounded ↔ free_form)

```bash
curl -X POST http://localhost:8000/api/problems/two_sum/toggle
```

#### Set Mode Explicitly

```bash
curl -X PATCH http://localhost:8000/api/problems/two_sum/mode \
  -H "Content-Type: application/json" \
  -d '{"mode": "free_form"}'
```

#### Update Snippets

```bash
curl -X PUT http://localhost:8000/api/problems/two_sum/snippets \
  -H "Content-Type: application/json" \
  -d '{
    "snippets": [
      "def two_sum(nums: list, target: int) -> list:\n    # Hint: use a hash map\n    pass"
    ]
  }'
```

### Candidate Management

#### Assign Candidate to Problem

```bash
curl -X POST http://localhost:8000/api/candidates/assign \
  -H "Content-Type: application/json" \
  -d '{"candidate_id": "user_456", "problem_id": "two_sum"}'
```

#### Get Candidate's Assignment

```bash
curl http://localhost:8000/api/candidates/user_456/problem
```

### Metrics

#### Get Overall Metrics

```bash
curl http://localhost:8000/api/metrics/overview
```

#### Compare Modes (Snippet vs Free-Form)

```bash
curl http://localhost:8000/api/metrics/mode-comparison
```

#### Get Candidate Metrics

```bash
curl http://localhost:8000/api/metrics/candidates/user_456
```

#### End Candidate Session

```bash
curl -X POST "http://localhost:8000/api/sessions/user_456/end?accepted=true"
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | Required if using OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic API key | Required if using Anthropic |
| `LLM_TARGET_API` | Target API provider (`openai`, `anthropic`) | `openai` |
| `LLM_TARGET_URL` | Custom LLM API URL | Auto-detected |
| `METRICS_DB_PATH` | Path to metrics SQLite database | `./llm_metrics.db` |
| `CONFIG_DB_PATH` | Path to config SQLite database | `./hackathon_config.db` |

---

## Docker Deployment

### Dockerfile

Create `Dockerfile` in the `llmMetrics` directory:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
COPY proxy/requirements.txt proxy/
RUN pip install -r requirements.txt -r proxy/requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "llmMetrics.proxy.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml

```yaml
version: '3.8'

services:
  llm-proxy:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Run with Docker

```bash
# Build and run
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Troubleshooting

### Cline Not Connecting to Proxy

1. **Check proxy is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify Cline settings:**
   - Ensure `openaiBaseUrl` points to `http://localhost:8000/v1`
   - Check the API key is correct

3. **Check for port conflicts:**
   ```bash
   lsof -i :8000
   ```

### Grounding Not Working

1. **Verify problem has snippets:**
   ```bash
   curl http://localhost:8000/api/problems/YOUR_PROBLEM_ID
   ```

2. **Check mode is snippet_grounded:**
   ```bash
   curl http://localhost:8000/api/problems/YOUR_PROBLEM_ID | jq '.mode'
   ```

3. **Ensure candidate is assigned:**
   ```bash
   curl http://localhost:8000/api/candidates/YOUR_CANDIDATE_ID/problem
   ```

### API Key Errors

1. **Check environment variable:**
   ```bash
   echo $OPENAI_API_KEY
   ```

2. **Verify key is valid:**
   ```bash
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### Viewing Logs

```bash
# If running with uvicorn
uvicorn llmMetrics.proxy.server:app --log-level debug

# If running with Docker
docker-compose logs -f llm-proxy
```

---

## API Documentation

The proxy includes auto-generated API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Security Considerations

For production deployments:

1. **Use HTTPS** - Put the proxy behind a reverse proxy (nginx) with SSL
2. **Add authentication** - Implement API key validation for admin endpoints
3. **Limit origins** - Restrict CORS to specific domains
4. **Secure database** - Store SQLite files in protected directories
5. **Rate limiting** - Add rate limits to prevent abuse

---

## Next Steps

- See [README.md](README.md) for detailed metrics system documentation
- Use the Admin API to create problems and assign candidates
- Monitor metrics via `/api/metrics/overview`
- Analyze mode comparison via `/api/metrics/mode-comparison`
