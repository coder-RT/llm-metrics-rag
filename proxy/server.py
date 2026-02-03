"""
LLM Metrics Proxy Server

A FastAPI-based proxy server that intercepts LLM API calls from tools
like Cline, captures usage metrics, and optionally enforces snippet-grounded mode.

Run with:
    uvicorn llmMetrics.proxy.server:app --host 0.0.0.0 --port 8000

Or programmatically:
    from llmMetrics.proxy import start_server
    start_server(port=8000)
"""

import os
import re
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our metrics tracking
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from llmMetricsRAG import LLMMetricsTracker, AssistanceMode
from llmMetricsRAG.proxy.config_store import ConfigStore, ProblemConfig, SnippetInfo
from llmMetricsRAG.proxy.grounding import (
    GroundingInjector,
    should_inject_grounding,
    extract_user_message,
)
from llmMetricsRAG.snippets import SnippetLoader, load_snippets_from_directory, load_all_snippets
from llmMetricsRAG.proxy.report_generator import generate_markdown_report, save_report
from llmMetricsRAG.config_loader import get_config, reload_config


# ============================================================================
# CONFIGURATION
# ============================================================================

# Base directory for llmMetrics package
LLM_METRICS_DIR = Path(__file__).parent.parent

# Load target URL from config.yaml (can be overridden by env var)
_startup_config = get_config()
TARGET_BASE_URL = os.getenv("LLM_TARGET_URL") or _startup_config.proxy.target_url

# Database paths - now inside llmMetrics directory
METRICS_DB_PATH = Path(os.getenv("METRICS_DB_PATH", str(LLM_METRICS_DIR / "data" / "llm_metrics.db")))
CONFIG_DB_PATH = Path(os.getenv("CONFIG_DB_PATH", str(LLM_METRICS_DIR / "data" / "hackathon_config.db")))


# ============================================================================
# GLOBAL INSTANCES
# ============================================================================

tracker: Optional[LLMMetricsTracker] = None
config_store: Optional[ConfigStore] = None
grounding_injector: Optional[GroundingInjector] = None
http_client: Optional[httpx.AsyncClient] = None

# Active sessions by candidate ID
active_sessions: Dict[str, str] = {}


# ============================================================================
# LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global tracker, config_store, grounding_injector, http_client
    
    # Startup
    print(f"Starting LLM Metrics Proxy Server")
    print(f"Target API: {TARGET_BASE_URL}")
    print(f"Metrics DB: {METRICS_DB_PATH}")
    print(f"Config DB: {CONFIG_DB_PATH}")
    
    tracker = LLMMetricsTracker(db_path=METRICS_DB_PATH)
    config_store = ConfigStore(db_path=CONFIG_DB_PATH)
    grounding_injector = GroundingInjector(strictness="moderate")
    
    # Load SSL verification setting from config
    verify_ssl: Any = False  # Default to False for internal networks
    try:
        from llmMetricsRAG.config_loader import ConfigLoader
        config = ConfigLoader()
        verify_ssl = config.proxy.verify_ssl
        print(f"SSL Verification: {verify_ssl}")
    except Exception as e:
        print(f"Could not load SSL config, using verify_ssl=False: {e}")
    
    http_client = httpx.AsyncClient(timeout=120.0, verify=verify_ssl)
    
    yield
    
    # Shutdown
    if http_client:
        await http_client.aclose()
    print("Proxy server stopped")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="LLM Metrics Proxy",
    description="Intercepts LLM API calls for metrics tracking and snippet grounding",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for browser-based tools
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ProblemCreate(BaseModel):
    problem_id: str
    title: str = ""
    mode: str = "snippet_grounded"
    snippets: List[str] = []
    description: str = ""


class ModeUpdate(BaseModel):
    mode: str


class SnippetsUpdate(BaseModel):
    snippets: List[str]


class CandidateAssignment(BaseModel):
    candidate_id: str
    problem_id: str


# ============================================================================
# PROXY ENDPOINTS
# ============================================================================

@app.post("/v1/chat/completions")
async def proxy_chat_completions(
    request: Request,
    x_candidate_id: Optional[str] = Header(None, alias="X-Candidate-ID"),
    x_problem_id: Optional[str] = Header(None, alias="X-Problem-ID"),
    x_assistance_mode: Optional[str] = Header(None, alias="X-Assistance-Mode"),
):
    """
    Proxy endpoint for OpenAI-compatible chat completions.
    
    This endpoint:
    1. Receives requests from Cline (or other tools)
    2. Looks up the problem config to check mode
    3. If snippet_grounded: injects grounding prompt + snippets
    4. Forwards to actual LLM API
    5. Captures metrics (tokens, grounding score, etc.)
    6. Returns response to Cline
    
    Headers:
        X-Candidate-ID: Candidate identifier (optional)
        X-Problem-ID: Problem identifier (optional, can use candidate assignment)
        X-Assistance-Mode: Override mode - "snippet" or "free" (optional)
        Authorization: Bearer token for LLM API
        
    Mode Override:
        Users can switch modes within Cline by adding the X-Assistance-Mode header:
        - "snippet" or "snippet_grounded" -> Uses snippets, enforces max_tokens
        - "free" or "free_form" -> No restrictions
    """
    global tracker, config_store, grounding_injector, http_client
    
    # Parse request body
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    # Get candidate and problem info
    candidate_id = x_candidate_id or request.headers.get("x-candidate-id") or "anonymous"
    problem_id = x_problem_id or request.headers.get("x-problem-id")
    
    # If no problem_id, check candidate assignment
    if not problem_id and config_store:
        problem_id = config_store.get_candidate_problem(candidate_id)
    
    problem_id = problem_id or "default"
    
    # Get problem config
    config: Optional[ProblemConfig] = None
    if config_store:
        config = config_store.get_problem(problem_id)
    
    # Determine mode and snippets
    mode = config.mode if config else "free_form"
    snippets = config.snippets if config else []
    
    # Allow mode override via header (for easy switching in Cline)
    mode_override = x_assistance_mode or request.headers.get("x-assistance-mode")
    if mode_override:
        mode_override = mode_override.lower().strip()
        if mode_override in ("snippet", "snippet_grounded", "grounded", "on"):
            mode = "snippet_grounded"
        elif mode_override in ("free", "free_form", "freeform", "off"):
            mode = "free_form"
    
    # Also check persistent mode overrides (set via API)
    if candidate_id in _candidate_mode_overrides:
        mode = _candidate_mode_overrides[candidate_id]
    
    # Get config settings for auto-loading
    app_config = get_config()
    sg_settings = app_config.snippet_grounded
    
    # Get snippet metadata with source paths if available
    snippets_with_sources = []
    
    # Auto-load snippets if enabled and in snippet mode
    if mode == "snippet_grounded" and sg_settings.auto_load_all_snippets:
        # Extract user query for smart matching
        messages = body.get("messages", [])
        user_query = extract_user_message(messages)
        
        # Helper to create relative snippet path (snippets/category/name.ext)
        def get_relative_path(snippet):
            if hasattr(snippet, 'category') and snippet.category and snippet.category != "root":
                if hasattr(snippet, 'path'):
                    return f"snippets/{snippet.category}/{snippet.path.name}"
                return f"snippets/{snippet.category}/{snippet.name}"
            if hasattr(snippet, 'path'):
                return f"snippets/{snippet.path.name}"
            return f"snippets/{snippet.name}"
        
        if sg_settings.smart_selection and user_query:
            # Try RAG first (semantic search), fallback to keyword matching
            rag_used = False
            
            if sg_settings.use_rag:
                try:
                    from llmMetricsRAG.snippets import RAG_AVAILABLE, get_rag_snippets
                    if RAG_AVAILABLE:
                        rag_results = get_rag_snippets(
                            query=user_query,
                            top_k=sg_settings.max_snippets,
                            min_score=sg_settings.min_match_score
                        )
                        if rag_results:
                            snippets_with_sources = rag_results
                            snippets = [r["content"] for r in rag_results]
                            rag_used = True
                            print(f"RAG: Found {len(rag_results)} relevant snippets")
                except Exception as e:
                    print(f"RAG failed, falling back to keyword matching: {e}")
            
            if not rag_used:
                # Fallback to keyword-based smart selection
                from llmMetricsRAG.snippets import get_relevant_snippets
                relevant_snippets = get_relevant_snippets(
                    query=user_query,
                    max_snippets=sg_settings.max_snippets,
                    min_score=sg_settings.min_match_score
                )
                snippets_with_sources = [
                    {
                        "content": s.content,
                        "source_path": get_relative_path(s),
                        "name": s.name,
                        "category": s.category
                    }
                    for s in relevant_snippets
                ]
                snippets = [s.content for s in relevant_snippets]
        else:
            # Load all snippets (original behavior)
            all_snippets = load_all_snippets(sg_settings.snippet_extensions)
            snippets_with_sources = [
                {
                    "content": s.content,
                    "source_path": get_relative_path(s),
                    "name": s.name,
                    "category": s.category
                }
                for s in all_snippets
            ]
            snippets = [s.content for s in all_snippets]
    elif config and config.snippet_metadata:
        snippets_with_sources = config.get_snippets_with_sources()
    elif snippets:
        snippets_with_sources = [{"content": s, "source_path": "", "name": ""} for s in snippets]
    
    # Extract original messages
    messages = body.get("messages", [])
    original_user_message = extract_user_message(messages)
    
    # Inject grounding if needed (with source paths)
    if mode == "snippet_grounded" and snippets_with_sources and grounding_injector:
        if should_inject_grounding(messages, mode, snippets):
            result = grounding_injector.inject(
                messages=messages,
                snippets=snippets_with_sources,  # Now includes source paths
                strictness=sg_settings.strictness,
                add_reminder=len(messages) > sg_settings.reminder_threshold
            )
            body["messages"] = result.modified_messages
        
        # Enforce max_tokens limit for snippet-grounded mode only
        if sg_settings.max_tokens is not None:
            # Only set if not already specified or if our limit is lower
            current_max = body.get("max_tokens")
            if current_max is None or current_max > sg_settings.max_tokens:
                body["max_tokens"] = sg_settings.max_tokens
    
    # Start/resume session
    if tracker:
        if candidate_id not in active_sessions:
            session_id = tracker.start_session(candidate_id, problem_id)
            active_sessions[candidate_id] = session_id
        
        tracker.set_mode(
            AssistanceMode.SNIPPET_GROUNDED 
            if mode == "snippet_grounded" 
            else AssistanceMode.FREE_FORM
        )
    
    # Check if streaming is requested
    is_streaming = body.get("stream", False)
    
    # Get authorization header
    auth_header = request.headers.get("authorization")
    if not auth_header:
        # Try to get from environment
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if api_key:
            auth_header = f"Bearer {api_key}"
    
    # Forward to LLM API
    headers = {
        "Authorization": auth_header,
        "Content-Type": "application/json",
    }
    
    target_url = f"{TARGET_BASE_URL}/chat/completions"
    
    try:
        if is_streaming:
            return await handle_streaming_response(
                body, headers, target_url, 
                candidate_id, problem_id, original_user_message, snippets, mode
            )
        else:
            return await handle_regular_response(
                body, headers, target_url,
                candidate_id, problem_id, original_user_message, snippets, mode
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Error calling LLM API: {e}")


async def handle_regular_response(
    body: Dict[str, Any],
    headers: Dict[str, str],
    target_url: str,
    candidate_id: str,
    problem_id: str,
    user_message: str,
    snippets: List[str],
    mode: str,
) -> JSONResponse:
    """Handle non-streaming LLM response."""
    global tracker, http_client
    
    response = await http_client.post(
        target_url,
        json=body,
        headers=headers,
    )
    
    result = response.json()
    
    # Extract LLM response
    llm_response = ""
    if "choices" in result and result["choices"]:
        llm_response = result["choices"][0].get("message", {}).get("content", "")
    
    # Record metrics
    if tracker:
        try:
            tracker.record_interaction(
                prompt=user_message,
                response=llm_response,
                snippets=snippets if mode == "snippet_grounded" else []
            )
        except RuntimeError:
            # Session might have been ended, start a new one
            tracker.start_session(candidate_id, problem_id)
            tracker.record_interaction(
                prompt=user_message,
                response=llm_response,
                snippets=snippets if mode == "snippet_grounded" else []
            )
    
    return JSONResponse(content=result)


async def handle_streaming_response(
    body: Dict[str, Any],
    headers: Dict[str, str],
    target_url: str,
    candidate_id: str,
    problem_id: str,
    user_message: str,
    snippets: List[str],
    mode: str,
) -> StreamingResponse:
    """Handle streaming LLM response."""
    
    async def stream_and_capture():
        global tracker, http_client
        
        full_response = []
        
        async with http_client.stream(
            "POST",
            target_url,
            json=body,
            headers=headers,
        ) as response:
            async for chunk in response.aiter_bytes():
                # Yield chunk to client
                yield chunk
                
                # Try to capture content from SSE data
                try:
                    chunk_str = chunk.decode("utf-8")
                    for line in chunk_str.split("\n"):
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data = json.loads(line[6:])
                            if "choices" in data and data["choices"]:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_response.append(content)
                except:
                    pass
        
        # Record metrics after stream completes
        if tracker and full_response:
            llm_response = "".join(full_response)
            try:
                tracker.record_interaction(
                    prompt=user_message,
                    response=llm_response,
                    snippets=snippets if mode == "snippet_grounded" else []
                )
            except RuntimeError:
                tracker.start_session(candidate_id, problem_id)
                tracker.record_interaction(
                    prompt=user_message,
                    response=llm_response,
                    snippets=snippets if mode == "snippet_grounded" else []
                )
    
    return StreamingResponse(
        stream_and_capture(),
        media_type="text/event-stream",
    )


# ============================================================================
# ANTHROPIC COMPATIBILITY
# ============================================================================

@app.post("/v1/messages")
async def proxy_anthropic_messages(
    request: Request,
    x_candidate_id: Optional[str] = Header(None, alias="X-Candidate-ID"),
    x_problem_id: Optional[str] = Header(None, alias="X-Problem-ID"),
):
    """
    Proxy endpoint for Anthropic Messages API.
    
    Handles the Anthropic-specific message format while applying
    the same grounding and metrics logic.
    """
    global tracker, config_store, grounding_injector, http_client
    
    body = await request.json()
    
    candidate_id = x_candidate_id or "anonymous"
    problem_id = x_problem_id
    
    if not problem_id and config_store:
        problem_id = config_store.get_candidate_problem(candidate_id)
    problem_id = problem_id or "default"
    
    # Get config
    config = config_store.get_problem(problem_id) if config_store else None
    mode = config.mode if config else "free_form"
    snippets = config.snippets if config else []
    
    # Anthropic uses "messages" array similar to OpenAI
    messages = body.get("messages", [])
    user_message = extract_user_message(messages)
    
    # Inject grounding via system prompt (Anthropic-style)
    if mode == "snippet_grounded" and snippets and grounding_injector:
        grounding_msg = grounding_injector.create_grounding_message(snippets)
        
        # Anthropic uses "system" as a top-level field
        existing_system = body.get("system", "")
        body["system"] = grounding_msg["content"] + "\n\n" + existing_system
    
    # Forward to Anthropic
    anthropic_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1/messages")
    
    headers = {
        "x-api-key": request.headers.get("x-api-key") or os.getenv("ANTHROPIC_API_KEY"),
        "anthropic-version": request.headers.get("anthropic-version", "2024-01-01"),
        "Content-Type": "application/json",
    }
    
    response = await http_client.post(anthropic_url, json=body, headers=headers)
    result = response.json()
    
    # Extract response
    llm_response = ""
    if "content" in result and result["content"]:
        llm_response = result["content"][0].get("text", "")
    
    # Record metrics
    if tracker:
        if candidate_id not in active_sessions:
            tracker.start_session(candidate_id, problem_id)
            active_sessions[candidate_id] = tracker.current_session.session_id
        
        tracker.set_mode(
            AssistanceMode.SNIPPET_GROUNDED if mode == "snippet_grounded" else AssistanceMode.FREE_FORM
        )
        tracker.record_interaction(
            prompt=user_message,
            response=llm_response,
            snippets=snippets if mode == "snippet_grounded" else []
        )
    
    return JSONResponse(content=result)


# ============================================================================
# ADMIN API - PROBLEM MANAGEMENT
# ============================================================================

@app.post("/api/problems")
async def create_problem(problem: ProblemCreate):
    """Create or update a problem configuration."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    config = config_store.create_problem(
        problem_id=problem.problem_id,
        title=problem.title,
        mode=problem.mode,
        snippets=problem.snippets,
        description=problem.description,
    )
    return config.to_dict()


@app.get("/api/problems")
async def list_problems():
    """List all problem configurations."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    problems = config_store.list_problems()
    return [p.to_dict() for p in problems]


@app.get("/api/problems/{problem_id}")
async def get_problem(problem_id: str):
    """Get a specific problem configuration."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    config = config_store.get_problem(problem_id)
    if not config:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    return config.to_dict()


@app.patch("/api/problems/{problem_id}/mode")
async def update_mode(problem_id: str, update: ModeUpdate):
    """Update the mode for a problem (toggle between snippet_grounded and free_form)."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    if update.mode not in ("snippet_grounded", "free_form"):
        raise HTTPException(status_code=400, detail="Invalid mode")
    
    success = config_store.set_mode(problem_id, update.mode)
    if not success:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    return {"problem_id": problem_id, "mode": update.mode}


@app.post("/api/problems/{problem_id}/toggle")
async def toggle_mode(problem_id: str):
    """Toggle the mode for a problem."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    new_mode = config_store.toggle_mode(problem_id)
    if not new_mode:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    return {"problem_id": problem_id, "mode": new_mode}


@app.put("/api/problems/{problem_id}/snippets")
async def update_snippets(problem_id: str, update: SnippetsUpdate):
    """Update the snippets for a problem."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    success = config_store.set_snippets(problem_id, update.snippets)
    if not success:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    return {"problem_id": problem_id, "snippets": update.snippets}


@app.delete("/api/problems/{problem_id}")
async def delete_problem(problem_id: str):
    """Delete a problem configuration."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    success = config_store.delete_problem(problem_id)
    if not success:
        raise HTTPException(status_code=404, detail="Problem not found")
    
    return {"deleted": problem_id}


# ============================================================================
# ADMIN API - CANDIDATE MANAGEMENT
# ============================================================================

@app.post("/api/candidates/assign")
async def assign_candidate(assignment: CandidateAssignment):
    """Assign a candidate to a problem."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    config_store.assign_candidate(assignment.candidate_id, assignment.problem_id)
    return {"candidate_id": assignment.candidate_id, "problem_id": assignment.problem_id}


@app.get("/api/candidates/{candidate_id}/problem")
async def get_candidate_assignment(candidate_id: str):
    """Get the problem assigned to a candidate."""
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    problem_id = config_store.get_candidate_problem(candidate_id)
    if not problem_id:
        raise HTTPException(status_code=404, detail="Candidate not assigned")
    
    return {"candidate_id": candidate_id, "problem_id": problem_id}


# ============================================================================
# MODE TOGGLE API - For easy switching in Cline
# ============================================================================

# In-memory mode overrides (candidate_id -> mode)
_candidate_mode_overrides: Dict[str, str] = {}


@app.get("/api/mode")
async def get_current_mode_info():
    """
    Get current mode configuration and available options.
    
    Returns information about how to switch modes in Cline.
    """
    config = get_config()
    return {
        "available_modes": ["snippet_grounded", "free_form"],
        "how_to_switch": {
            "via_header": "Add header 'X-Assistance-Mode: snippet' or 'X-Assistance-Mode: free'",
            "via_api": "POST /api/mode/set with {candidate_id, mode}",
            "via_problem": "Each problem has a default mode that can be toggled"
        },
        "snippet_grounded_settings": {
            "max_tokens": config.snippet_grounded.max_tokens,
            "strictness": config.snippet_grounded.strictness,
        },
        "free_form_settings": {
            "max_tokens": config.free_form.max_tokens,
        },
        "active_overrides": _candidate_mode_overrides,
    }


class ModeSetRequest(BaseModel):
    candidate_id: str
    mode: str  # "snippet", "snippet_grounded", "free", "free_form"


@app.post("/api/mode/set")
async def set_candidate_mode(req: ModeSetRequest):
    """
    Set the mode for a specific candidate.
    
    This override persists until cleared or the server restarts.
    Useful for testing or when candidates need to switch modes.
    """
    mode = req.mode.lower().strip()
    
    if mode in ("snippet", "snippet_grounded", "grounded", "on"):
        _candidate_mode_overrides[req.candidate_id] = "snippet_grounded"
    elif mode in ("free", "free_form", "freeform", "off"):
        _candidate_mode_overrides[req.candidate_id] = "free_form"
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid mode: {req.mode}. Use 'snippet' or 'free'"
        )
    
    return {
        "candidate_id": req.candidate_id,
        "mode": _candidate_mode_overrides[req.candidate_id],
        "message": f"Mode set to {_candidate_mode_overrides[req.candidate_id]}"
    }


@app.delete("/api/mode/{candidate_id}")
async def clear_candidate_mode(candidate_id: str):
    """Clear the mode override for a candidate (uses problem default)."""
    if candidate_id in _candidate_mode_overrides:
        del _candidate_mode_overrides[candidate_id]
        return {"candidate_id": candidate_id, "status": "cleared", "message": "Will use problem default"}
    
    return {"candidate_id": candidate_id, "status": "not_found", "message": "No override was set"}


@app.get("/api/mode/{candidate_id}")
async def get_candidate_mode(candidate_id: str):
    """Get the current mode for a candidate."""
    override = _candidate_mode_overrides.get(candidate_id)
    
    # Also check what problem they're assigned to
    problem_id = None
    problem_mode = None
    if config_store:
        problem_id = config_store.get_candidate_problem(candidate_id)
        if problem_id:
            config = config_store.get_problem(problem_id)
            problem_mode = config.mode if config else None
    
    effective_mode = override or problem_mode or "free_form"
    
    return {
        "candidate_id": candidate_id,
        "effective_mode": effective_mode,
        "override": override,
        "problem_id": problem_id,
        "problem_default_mode": problem_mode,
    }


# ============================================================================
# ADMIN API - METRICS
# ============================================================================

@app.get("/api/metrics/overview")
async def get_metrics_overview():
    """Get overall metrics summary."""
    if not tracker:
        raise HTTPException(status_code=500, detail="Tracker not initialized")
    
    return tracker.get_overall_stats()


@app.get("/api/metrics/mode-comparison")
async def get_mode_comparison(problem_id: Optional[str] = None):
    """Get comparison between snippet_grounded and free_form modes."""
    if not tracker:
        raise HTTPException(status_code=500, detail="Tracker not initialized")
    
    return tracker.get_mode_comparison(problem_id)


@app.get("/api/metrics/candidates/{candidate_id}")
async def get_candidate_metrics(candidate_id: str):
    """Get metrics for a specific candidate."""
    if not tracker:
        raise HTTPException(status_code=500, detail="Tracker not initialized")
    
    sessions = tracker.get_candidate_history(candidate_id)
    total_tokens = tracker.get_candidate_total_tokens(candidate_id)
    
    return {
        "candidate_id": candidate_id,
        "total_tokens": total_tokens,
        "sessions": sessions,
    }


@app.post("/api/sessions/{candidate_id}/end")
async def end_candidate_session(candidate_id: str, accepted: bool = False):
    """End a candidate's session."""
    if not tracker:
        raise HTTPException(status_code=500, detail="Tracker not initialized")
    
    if candidate_id in active_sessions:
        summary = tracker.end_session(accepted=accepted)
        del active_sessions[candidate_id]
        return summary
    
    raise HTTPException(status_code=404, detail="No active session for candidate")


# ============================================================================
# ADMIN API - REPORTS
# ============================================================================

@app.get("/api/reports/comparison")
async def get_comparison_report():
    """
    Generate a markdown report comparing snippet-grounded vs free-form modes.
    
    Returns the report as JSON with markdown content.
    """
    try:
        report_content = generate_markdown_report(
            db_path=METRICS_DB_PATH,
            report_title="LLM Usage Metrics Report"
        )
        return {
            "report": report_content,
            "generated_at": datetime.utcnow().isoformat(),
            "format": "markdown"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {e}")


@app.get("/api/reports/comparison/download")
async def download_comparison_report():
    """
    Generate and download a markdown report as a file.
    
    Returns the report as a downloadable .md file.
    """
    from fastapi.responses import Response
    
    try:
        report_content = generate_markdown_report(
            db_path=METRICS_DB_PATH,
            report_title="LLM Usage Metrics Report"
        )
        
        filename = f"llm_metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        return Response(
            content=report_content,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {e}")


@app.post("/api/reports/comparison/save")
async def save_comparison_report(filename: Optional[str] = None):
    """
    Generate and save a markdown report to the reports directory.
    
    Args:
        filename: Optional custom filename (without .md extension)
        
    Returns:
        Path to the saved report file
    """
    try:
        # Create reports directory
        reports_dir = LLM_METRICS_DIR / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Generate filename
        if not filename:
            filename = f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        output_path = reports_dir / f"{filename}.md"
        
        # Save report
        saved_path = save_report(
            output_path=output_path,
            db_path=METRICS_DB_PATH,
            report_title="LLM Usage Metrics Report"
        )
        
        return {
            "saved_to": str(saved_path),
            "filename": f"{filename}.md",
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving report: {e}")


@app.get("/api/reports/raw-stats")
async def get_raw_statistics():
    """
    Get raw statistics for custom reporting.
    
    Returns detailed metrics in JSON format.
    """
    import sqlite3
    
    try:
        conn = sqlite3.connect(str(METRICS_DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Mode comparison
        cursor.execute("""
            SELECT 
                mode,
                COUNT(*) as interactions,
                SUM(total_tokens) as total_tokens,
                ROUND(AVG(total_tokens), 1) as avg_tokens,
                SUM(user_prompt_tokens) as input_tokens,
                SUM(model_response_tokens) as output_tokens,
                ROUND(AVG(model_response_tokens), 1) as avg_output_tokens
            FROM interactions
            GROUP BY mode
        """)
        
        mode_stats = {}
        for row in cursor.fetchall():
            mode_stats[row["mode"]] = dict(row)
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_interactions,
                SUM(total_tokens) as total_tokens,
                COUNT(DISTINCT candidate_id) as unique_candidates,
                COUNT(DISTINCT problem_id) as unique_problems
            FROM interactions
        """)
        row = cursor.fetchone()
        overall = dict(row) if row else {}
        
        # Recent interactions
        cursor.execute("""
            SELECT candidate_id, problem_id, mode, total_tokens, timestamp
            FROM interactions
            ORDER BY timestamp DESC
            LIMIT 10
        """)
        recent = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            "mode_comparison": mode_stats,
            "overall": overall,
            "recent_interactions": recent,
            "generated_at": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {e}")


# ============================================================================
# ADMIN API - SNIPPETS
# ============================================================================

# Global snippet loader
snippet_loader: Optional[SnippetLoader] = None


@app.get("/api/snippets")
async def list_snippets():
    """List all available snippets organized by category."""
    global snippet_loader
    if snippet_loader is None:
        snippet_loader = SnippetLoader()
    
    return snippet_loader.list_all()


@app.get("/api/snippets/all")
async def list_all_snippets():
    """
    List ALL snippets that would be auto-loaded in snippet mode.
    
    Uses the configured file extensions from config.yaml.
    """
    config = get_config()
    extensions = config.snippet_grounded.snippet_extensions
    
    all_snippets = load_all_snippets(extensions)
    
    return {
        "auto_load_enabled": config.snippet_grounded.auto_load_all_snippets,
        "extensions": extensions,
        "total_snippets": len(all_snippets),
        "snippets": [
            {
                "name": s.name,
                "category": s.category,
                "path": str(s.path),
                "language": s.language,
                "content_preview": s.content[:200] + "..." if len(s.content) > 200 else s.content
            }
            for s in all_snippets
        ]
    }


# ============================================================================
# RAG ENDPOINTS
# ============================================================================

@app.post("/api/rag/index")
async def index_snippets_for_rag(force: bool = False):
    """
    Index all snippets for RAG (semantic search).
    
    This creates vector embeddings for all snippets, enabling
    semantic search to find relevant snippets based on meaning.
    
    Args:
        force: If True, rebuild the entire index
        
    Returns:
        Status and number of snippets indexed
    """
    try:
        from llmMetricsRAG.snippets import RAG_AVAILABLE, index_all_snippets
        if not RAG_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "RAG dependencies not installed. Run: pip install chromadb sentence-transformers"
            }
        
        count = index_all_snippets(force_reindex=force)
        return {
            "status": "success",
            "indexed_snippets": count,
            "message": f"Indexed {count} snippets for RAG search"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/api/rag/search")
async def search_snippets_rag(query: str, top_k: int = 2, min_score: float = 0.3):
    """
    Search snippets using RAG (semantic search).
    
    Args:
        query: Search query
        top_k: Maximum number of results
        min_score: Minimum similarity score (0-1)
        
    Returns:
        Matching snippets with similarity scores
    """
    try:
        from llmMetricsRAG.snippets import RAG_AVAILABLE, rag_search
        if not RAG_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "RAG dependencies not installed"
            }
        
        results = rag_search(query, top_k=top_k, min_score=min_score)
        return {
            "query": query,
            "results": [
                {
                    "name": r.name,
                    "category": r.category,
                    "score": round(r.score, 3),
                    "content_preview": r.content[:200] + "..." if len(r.content) > 200 else r.content
                }
                for r in results
            ],
            "total_results": len(results)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/api/rag/status")
async def get_rag_status():
    """Get RAG system status."""
    try:
        from llmMetricsRAG.snippets import RAG_AVAILABLE, get_retriever
        if not RAG_AVAILABLE:
            return {
                "available": False,
                "message": "RAG dependencies not installed. Run: pip install chromadb sentence-transformers"
            }
        
        retriever = get_retriever()
        config = get_config()
        
        return {
            "available": True,
            "enabled": config.snippet_grounded.use_rag,
            "indexed_snippets": retriever.count(),
            "model": retriever.model_name,
            "max_snippets": config.snippet_grounded.max_snippets,
            "min_score": config.snippet_grounded.min_match_score
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }


@app.get("/api/snippets/categories")
async def list_snippet_categories():
    """List available snippet categories."""
    global snippet_loader
    if snippet_loader is None:
        snippet_loader = SnippetLoader()
    
    return {"categories": snippet_loader.list_categories()}


@app.get("/api/snippets/{category}")
async def get_category_snippets(category: str):
    """Get all snippets in a category."""
    global snippet_loader
    if snippet_loader is None:
        snippet_loader = SnippetLoader()
    
    snippets = snippet_loader.load_category(category)
    if not snippets:
        raise HTTPException(status_code=404, detail=f"Category '{category}' not found or empty")
    
    return {
        "category": category,
        "snippets": [s.to_dict() for s in snippets]
    }


@app.get("/api/snippets/{category}/{name}")
async def get_snippet(category: str, name: str):
    """Get a specific snippet by category and name."""
    global snippet_loader
    if snippet_loader is None:
        snippet_loader = SnippetLoader()
    
    snippet = snippet_loader.load(f"{category}/{name}")
    if not snippet:
        raise HTTPException(status_code=404, detail=f"Snippet '{category}/{name}' not found")
    
    return snippet.to_dict()


class ProblemFromSnippets(BaseModel):
    """Create a problem using snippets from files."""
    problem_id: str
    title: str = ""
    mode: str = "snippet_grounded"
    snippet_paths: List[str]  # e.g., ["math_operations/add", "math_operations/subtract"]
    description: str = ""


@app.post("/api/problems/from-snippets")
async def create_problem_from_snippets(request: ProblemFromSnippets):
    """
    Create a problem configuration using snippets loaded from files.
    
    Example:
        POST /api/problems/from-snippets
        {
            "problem_id": "math_basics",
            "title": "Basic Math Operations",
            "mode": "snippet_grounded",
            "snippet_paths": ["math_operations/add", "math_operations/subtract"]
        }
    """
    global snippet_loader, config_store
    
    if snippet_loader is None:
        snippet_loader = SnippetLoader()
    
    if not config_store:
        raise HTTPException(status_code=500, detail="Config store not initialized")
    
    # Load snippets from files
    snippets = snippet_loader.load_multiple(request.snippet_paths)
    if not snippets:
        raise HTTPException(
            status_code=404, 
            detail=f"No snippets found for paths: {request.snippet_paths}"
        )
    
    # Extract content as strings (for backward compatibility)
    snippet_contents = [s.content for s in snippets]
    
    # Create snippet metadata with source paths
    snippet_metadata = [
        SnippetInfo(
            content=s.content,
            source_path=str(s.path),
            name=s.name,
            category=s.category,
        )
        for s in snippets
    ]
    
    # Create the problem with metadata
    config = config_store.create_problem(
        problem_id=request.problem_id,
        title=request.title,
        mode=request.mode,
        snippets=snippet_contents,
        snippet_metadata=snippet_metadata,
        description=request.description,
    )
    
    return {
        **config.to_dict(),
        "loaded_snippets": [
            {"name": s.name, "source": str(s.path)} for s in snippets
        ],
    }


# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "target_api": TARGET_BASE_URL,
        "active_sessions": len(active_sessions),
    }


# ============================================================================
# CONFIG API
# ============================================================================

@app.get("/api/config")
async def get_configuration():
    """
    Get current configuration settings.
    
    Returns all configuration values from config.yaml.
    """
    config = get_config()
    return {
        "snippet_grounded_mode": {
            "max_tokens": config.snippet_grounded.max_tokens,
            "strictness": config.snippet_grounded.strictness,
            "add_reminder": config.snippet_grounded.add_reminder,
            "reminder_threshold": config.snippet_grounded.reminder_threshold,
            "show_source_path": config.snippet_grounded.show_source_path,
            "auto_load_all_snippets": config.snippet_grounded.auto_load_all_snippets,
            "smart_selection": config.snippet_grounded.smart_selection,
            "max_snippets": config.snippet_grounded.max_snippets,
            "min_match_score": config.snippet_grounded.min_match_score,
            "snippet_extensions": config.snippet_grounded.snippet_extensions,
        },
        "free_form_mode": {
            "max_tokens": config.free_form.max_tokens,
            "track_metrics": config.free_form.track_metrics,
        },
        "cost_settings": {
            "input_cost_per_1k": config.cost.input_cost_per_1k,
            "output_cost_per_1k": config.cost.output_cost_per_1k,
            "model": config.cost.model,
        },
        "proxy": {
            "host": config.proxy.host,
            "port": config.proxy.port,
            "target_url": config.proxy.target_url,
            "timeout": config.proxy.timeout,
            "debug": config.proxy.debug,
        },
        "config_file": str(config.config_path),
    }


@app.post("/api/config/reload")
async def reload_configuration():
    """
    Reload configuration from config.yaml file.
    
    Use this to apply changes made to the config file without restarting the server.
    """
    config = reload_config()
    return {
        "status": "reloaded",
        "snippet_grounded_mode": {
            "max_tokens": config.snippet_grounded.max_tokens,
            "strictness": config.snippet_grounded.strictness,
        },
        "free_form_mode": {
            "max_tokens": config.free_form.max_tokens,
        },
    }


@app.get("/api/config/mode-settings/{mode}")
async def get_mode_specific_settings(mode: str):
    """
    Get settings for a specific mode.
    
    Args:
        mode: 'snippet_grounded' or 'free_form'
    """
    config = get_config()
    
    if mode not in ["snippet_grounded", "free_form"]:
        raise HTTPException(
            status_code=400, 
            detail="Mode must be 'snippet_grounded' or 'free_form'"
        )
    
    return config.get_mode_settings(mode)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "LLM Metrics Proxy",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


# ============================================================================
# STARTUP HELPER
# ============================================================================

def start_server(host: str = None, port: int = None):
    """
    Start the proxy server.
    
    Args:
        host: Server host (defaults to config.yaml value)
        port: Server port (defaults to config.yaml value)
    """
    import uvicorn
    
    config = get_config()
    server_host = host or config.proxy.host
    server_port = port or config.proxy.port
    
    print(f"Starting LLM Metrics Proxy on {server_host}:{server_port}")
    print(f"Target API: {config.proxy.target_url}")
    print(f"Docs: http://{server_host}:{server_port}/docs")
    
    uvicorn.run(app, host=server_host, port=server_port)


if __name__ == "__main__":
    start_server()
