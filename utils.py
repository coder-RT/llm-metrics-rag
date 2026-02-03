"""
LLM Metrics Utility Functions

This module provides utility functions for token counting, grounding score
calculation, and other helper operations used throughout the metrics system.
"""

import re
from typing import List, Optional, Set
from functools import lru_cache

# Try to import tiktoken for accurate token counting
# Falls back to estimation if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


# Default model for token counting (GPT-4/ChatGPT)
DEFAULT_MODEL = "gpt-4"

# Approximate tokens per character for fallback estimation
CHARS_PER_TOKEN = 4


@lru_cache(maxsize=8)
def _get_encoding(model: str = DEFAULT_MODEL):
    """
    Get the tiktoken encoding for a specific model.
    
    Uses LRU cache to avoid repeated encoding initialization.
    
    Args:
        model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
        
    Returns:
        tiktoken.Encoding object
    """
    if not TIKTOKEN_AVAILABLE:
        return None
    
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base encoding (used by GPT-4)
        return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Count the number of tokens in a text string.
    
    Uses tiktoken for accurate counting if available,
    otherwise falls back to character-based estimation.
    
    Args:
        text: The text to count tokens for
        model: The model to use for encoding (default: gpt-4)
        
    Returns:
        Number of tokens in the text
        
    Example:
        >>> count_tokens("Hello, world!")
        4
        >>> count_tokens("def hello():\n    print('world')")
        12
    """
    if not text:
        return 0
    
    # Ensure text is a string
    if not isinstance(text, str):
        if isinstance(text, list):
            # Handle list of message dicts (Cline format)
            text = " ".join(
                str(item.get("content", item) if isinstance(item, dict) else item)
                for item in text
            )
        else:
            text = str(text)
    
    encoding = _get_encoding(model)
    
    if encoding:
        # Accurate token counting using tiktoken
        return len(encoding.encode(text))
    else:
        # Fallback: estimate based on character count
        # This is a rough approximation (1 token ≈ 4 characters)
        return max(1, len(text) // CHARS_PER_TOKEN)


def estimate_snippet_tokens(snippets: List[str], model: str = DEFAULT_MODEL) -> int:
    """
    Count total tokens across multiple code snippets.
    
    Args:
        snippets: List of code snippet strings
        model: The model to use for encoding
        
    Returns:
        Total token count across all snippets
        
    Example:
        >>> snippets = ["def foo():", "def bar():"]
        >>> estimate_snippet_tokens(snippets)
        10
    """
    if not snippets:
        return 0
    
    return sum(count_tokens(snippet, model) for snippet in snippets)


def extract_code_blocks(text: str) -> List[str]:
    """
    Extract code blocks from markdown-formatted text.
    
    Identifies code blocks wrapped in triple backticks.
    
    Args:
        text: Text containing markdown code blocks
        
    Returns:
        List of code block contents
        
    Example:
        >>> text = "Here's code:\\n```python\\nprint('hi')\\n```"
        >>> extract_code_blocks(text)
        ["print('hi')"]
    """
    # Match triple-backtick code blocks (with or without language)
    pattern = r"```(?:\w+)?\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]


def extract_identifiers(code: str) -> Set[str]:
    """
    Extract function and variable names from code.
    
    Identifies common programming identifiers for grounding analysis.
    
    Args:
        code: Source code string
        
    Returns:
        Set of identifier names found in the code
        
    Example:
        >>> code = "def calculate_total(items):\\n    return sum(items)"
        >>> identifiers = extract_identifiers(code)
        >>> "calculate_total" in identifiers
        True
    """
    # Match function definitions and variable names
    # This is a simplified pattern that works for Python/JS-like languages
    patterns = [
        r'\bdef\s+(\w+)',           # Python function
        r'\bfunction\s+(\w+)',       # JS function
        r'\bclass\s+(\w+)',          # Class definition
        r'\b(\w+)\s*=',              # Variable assignment
        r'\bconst\s+(\w+)',          # JS const
        r'\blet\s+(\w+)',            # JS let
        r'\bvar\s+(\w+)',            # JS var
    ]
    
    identifiers = set()
    for pattern in patterns:
        matches = re.findall(pattern, code)
        identifiers.update(matches)
    
    # Filter out common keywords and short names
    keywords = {
        'if', 'else', 'for', 'while', 'return', 'import', 'from',
        'try', 'except', 'finally', 'with', 'as', 'in', 'is', 'not',
        'and', 'or', 'True', 'False', 'None', 'self', 'cls',
    }
    
    return identifiers - keywords


def calculate_grounding_score(
    response: str, 
    snippets: List[str],
    min_reference_length: int = 3
) -> float:
    """
    Calculate how well the LLM response is grounded in provided snippets.
    
    This measures the degree to which the model's response references
    or builds upon the provided code snippets. Higher scores indicate
    better grounding.
    
    Scoring methodology:
    1. Extract identifiers from snippets (function/variable names)
    2. Check how many identifiers are referenced in the response
    3. Check if response contains code that mirrors snippet patterns
    
    Args:
        response: The LLM's response text
        snippets: List of code snippets that were provided
        min_reference_length: Minimum identifier length to consider
        
    Returns:
        Grounding score from 0.0 (no grounding) to 1.0 (fully grounded)
        
    Example:
        >>> snippet = "def calculate_sum(numbers):\\n    return sum(numbers)"
        >>> response = "You can modify calculate_sum to handle edge cases"
        >>> score = calculate_grounding_score(response, [snippet])
        >>> score > 0.5
        True
    """
    if not snippets or not response:
        return 0.0
    
    # Combine all snippets for analysis
    combined_snippets = "\n".join(snippets)
    
    # Extract identifiers from snippets
    snippet_identifiers = extract_identifiers(combined_snippets)
    
    # Filter to significant identifiers
    significant_identifiers = {
        ident for ident in snippet_identifiers 
        if len(ident) >= min_reference_length
    }
    
    if not significant_identifiers:
        # No identifiers to track, check for direct snippet presence
        snippet_present = any(
            snippet[:50] in response or snippet[-50:] in response
            for snippet in snippets if len(snippet) > 50
        )
        return 1.0 if snippet_present else 0.5
    
    # Count how many snippet identifiers appear in response
    response_lower = response.lower()
    referenced_count = sum(
        1 for ident in significant_identifiers
        if ident.lower() in response_lower
    )
    
    # Calculate base score from identifier references
    identifier_score = referenced_count / len(significant_identifiers)
    
    # Bonus: Check if response contains code blocks with snippet patterns
    response_code_blocks = extract_code_blocks(response)
    code_similarity_bonus = 0.0
    
    if response_code_blocks:
        response_identifiers = set()
        for block in response_code_blocks:
            response_identifiers.update(extract_identifiers(block))
        
        # Check overlap between response code and snippet identifiers
        overlap = response_identifiers & snippet_identifiers
        if overlap:
            code_similarity_bonus = len(overlap) / len(snippet_identifiers) * 0.3
    
    # Final score (capped at 1.0)
    final_score = min(1.0, identifier_score + code_similarity_bonus)
    
    return round(final_score, 3)


def calculate_snippet_reference_rate(
    response: str, 
    snippets: List[str]
) -> float:
    """
    Calculate the rate at which the response explicitly references snippets.
    
    This is a simpler metric than grounding score - it checks for
    direct textual references to snippet content.
    
    Args:
        response: The LLM's response text
        snippets: List of code snippets provided
        
    Returns:
        Reference rate from 0.0 to 1.0
    """
    if not snippets or not response:
        return 0.0
    
    # Count snippets that are referenced
    referenced = 0
    
    for snippet in snippets:
        # Check for partial matches (at least 20 chars or whole snippet)
        check_length = min(20, len(snippet))
        if check_length > 0:
            snippet_start = snippet[:check_length]
            if snippet_start in response:
                referenced += 1
                continue
        
        # Check for identifier references
        identifiers = extract_identifiers(snippet)
        if any(ident in response for ident in identifiers if len(ident) > 3):
            referenced += 1
    
    return referenced / len(snippets)


def detect_hallucination_indicators(
    response: str,
    snippets: List[str],
    known_functions: Optional[List[str]] = None
) -> int:
    """
    Detect potential hallucination indicators in the response.
    
    Hallucination indicators include:
    - References to functions not in snippets
    - Invented API calls
    - Non-existent library references
    
    Args:
        response: The LLM's response text
        snippets: List of provided code snippets
        known_functions: Optional list of valid function names
        
    Returns:
        Count of potential hallucination indicators
    """
    indicators = 0
    
    # Extract code blocks from response
    response_code = extract_code_blocks(response)
    if not response_code:
        return 0
    
    combined_response_code = "\n".join(response_code)
    combined_snippets = "\n".join(snippets) if snippets else ""
    
    # Get identifiers from snippets (known valid)
    valid_identifiers = extract_identifiers(combined_snippets)
    
    # Get identifiers from response code
    response_identifiers = extract_identifiers(combined_response_code)
    
    # Add known functions to valid set
    if known_functions:
        valid_identifiers.update(known_functions)
    
    # Check for function calls in response not in valid set
    # Look for function call patterns: function_name(
    call_pattern = r'(\w+)\s*\('
    calls = re.findall(call_pattern, combined_response_code)
    
    # Filter to significant calls (not common built-ins)
    builtins = {
        'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict',
        'set', 'tuple', 'sum', 'max', 'min', 'sorted', 'enumerate',
        'zip', 'map', 'filter', 'open', 'type', 'isinstance', 'hasattr',
        'getattr', 'setattr', 'input', 'abs', 'round', 'format',
        'console', 'log', 'error', 'warn', 'push', 'pop', 'append',
    }
    
    for call in calls:
        if len(call) > 3 and call not in builtins and call not in valid_identifiers:
            # Potential hallucination - invented function
            indicators += 1
    
    return indicators


def format_token_count(tokens: int) -> str:
    """
    Format a token count for display.
    
    Args:
        tokens: Number of tokens
        
    Returns:
        Formatted string (e.g., "1.2K", "500")
    """
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M"
    elif tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K"
    else:
        return str(tokens)


def calculate_cost_estimate(
    input_tokens: int,
    output_tokens: int,
    input_cost_per_1k: float = 0.03,
    output_cost_per_1k: float = 0.06
) -> float:
    """
    Estimate the cost of token usage.
    
    Default costs are based on GPT-4 pricing (as of 2024).
    
    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        input_cost_per_1k: Cost per 1000 input tokens
        output_cost_per_1k: Cost per 1000 output tokens
        
    Returns:
        Estimated cost in dollars
    """
    input_cost = (input_tokens / 1000) * input_cost_per_1k
    output_cost = (output_tokens / 1000) * output_cost_per_1k
    return round(input_cost + output_cost, 4)

