"""
Grounding Prompts and Injection Logic

This module handles the injection of grounding prompts and code snippets
into LLM requests when snippet-grounded mode is enabled.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass


@dataclass
class SnippetWithSource:
    """
    A code snippet with its source file information.
    
    Attributes:
        content: The actual code content
        source_path: Path to the source file (relative to snippets dir)
        name: Name of the snippet
        category: Category/subdirectory the snippet belongs to
    """
    content: str
    source_path: str = ""
    name: str = ""
    category: str = ""
    
    def __str__(self) -> str:
        return self.content


# ============================================================================
# GROUNDING SYSTEM PROMPTS
# ============================================================================

SNIPPET_GROUNDED_SYSTEM_PROMPT = """⚠️ SNIPPET-GROUNDED MODE ACTIVE ⚠️

You MUST use the code snippets below to answer. You are NOT allowed to generate code from scratch.

## MANDATORY RESPONSE FORMAT:

1. **START** every response by quoting or referencing a specific snippet below
2. Say "Based on the `[snippet_name]` snippet..." or "Looking at the provided code..."
3. Only THEN provide your explanation or suggestion
4. If no snippet is relevant, say "None of the provided snippets cover this topic."

## WHAT YOU CAN DO:
✅ Explain how the provided snippets work
✅ Suggest modifications to the provided code
✅ Debug issues in the provided snippets
✅ Show how to USE the provided functions

## WHAT YOU CANNOT DO:
❌ Write new code from scratch
❌ Generate complete solutions not based on snippets
❌ Use file system tools (read_file, list_dir, etc.)
❌ Browse or search the workspace
❌ Answer without referencing a snippet first

## YOUR AVAILABLE CODE SNIPPETS:

{snippets}

---

REMEMBER: Start EVERY response with "Based on the [snippet] snippet..." or similar. If the question doesn't relate to any snippet, say so and ask the user to rephrase."""


SNIPPET_GROUNDED_REMINDER = """
⚠️ SNIPPET MODE REMINDER: You MUST reference a specific snippet in your response.
Start with "Based on the [snippet_name] snippet..." or "Looking at the provided [snippet]..."
Do NOT generate new code. Do NOT use file system tools. Only use the snippets shown earlier."""


# Alternative prompts for different strictness levels
GROUNDING_PROMPTS = {
    "strict": """⚠️ STRICT SNIPPET-GROUNDED MODE ⚠️

MANDATORY: Start EVERY response with "Based on the [snippet_name] snippet..."

ABSOLUTE RULES:
1. You can ONLY discuss the exact code snippets provided below
2. You CANNOT write new functions or complete implementations
3. You may ONLY explain or suggest small modifications to existing code
4. If the question doesn't relate to a snippet, say "This isn't covered by the provided snippets"
5. Do NOT use file system tools or browse directories

PROVIDED SNIPPETS:
{snippets}

Remember: ALWAYS start with "Based on the [snippet] snippet..." or similar.""",

    "moderate": """⚠️ SNIPPET-GROUNDED MODE ⚠️

MANDATORY: Reference a specific snippet in EVERY response.
Say "Looking at the [snippet_name] code..." or "Based on the provided [snippet]..."

RULES:
1. Focus on the provided code snippets below
2. You may explain, debug, or suggest improvements
3. Do NOT generate complete solutions from scratch
4. Reference specific function/variable names from the snippets
5. Do NOT use file system tools or browse directories

PROVIDED SNIPPETS:
{snippets}

Remember: Always reference which snippet you're discussing.""",

    "light": """You are helping with code based on these snippets:

{snippets}

Please reference the relevant snippet when answering. Try to work within this existing code structure.""",
}


# ============================================================================
# GROUNDING INJECTOR CLASS
# ============================================================================

@dataclass
class InjectionResult:
    """Result of prompt injection."""
    modified_messages: List[Dict[str, Any]]
    injection_applied: bool
    snippets_included: int
    estimated_tokens_added: int


class GroundingInjector:
    """
    Handles injection of grounding prompts into LLM requests.
    
    This class modifies the messages array sent to the LLM to include:
    - A system prompt explaining snippet-grounded rules
    - The actual code snippets candidates should work with
    - Optional reminders in long conversations
    
    Usage:
        injector = GroundingInjector()
        
        modified_body = injector.inject(
            messages=original_messages,
            snippets=["def foo(): pass"],
            strictness="moderate"
        )
    """
    
    def __init__(self, strictness: str = "moderate"):
        """
        Initialize the injector.
        
        Args:
            strictness: Level of grounding enforcement
                       'strict' - Very restrictive, no new code
                       'moderate' - Balanced (default)
                       'light' - Gentle guidance
        """
        self.strictness = strictness
    
    def inject(
        self,
        messages: List[Dict[str, Any]],
        snippets: List[str],
        strictness: Optional[str] = None,
        add_reminder: bool = False
    ) -> InjectionResult:
        """
        Inject grounding prompt and snippets into messages.
        
        Args:
            messages: Original message array from the request
            snippets: Code snippets to include
            strictness: Override default strictness level
            add_reminder: Add reminder at end for long conversations
            
        Returns:
            InjectionResult with modified messages
        """
        if not snippets:
            return InjectionResult(
                modified_messages=messages,
                injection_applied=False,
                snippets_included=0,
                estimated_tokens_added=0,
            )
        
        level = strictness or self.strictness
        
        # Format snippets
        formatted_snippets = self._format_snippets(snippets)
        
        # Get appropriate prompt template
        if level in GROUNDING_PROMPTS:
            prompt_template = GROUNDING_PROMPTS[level]
        else:
            prompt_template = SNIPPET_GROUNDED_SYSTEM_PROMPT
        
        # Create grounding message
        grounding_content = prompt_template.format(snippets=formatted_snippets)
        grounding_message = {
            "role": "system",
            "content": grounding_content
        }
        
        # Build modified messages
        modified = []
        
        # Check if grounding is already present
        grounding_already_present = False
        for msg in messages:
            if msg.get("role") == "system":
                content = msg.get("content", "")
                if "SNIPPET-GROUNDED" in content or "PROVIDED SNIPPETS" in content:
                    grounding_already_present = True
                    break
        
        if grounding_already_present:
            # Grounding already exists - just copy messages and add reminder if needed
            modified = list(messages)
            tokens_added = 0
        else:
            # Check if there's already a system message
            has_system = any(m.get("role") == "system" for m in messages)
            
            if has_system:
                # Prepend our grounding to existing system message
                for msg in messages:
                    if msg.get("role") == "system":
                        modified.append({
                            "role": "system",
                            "content": grounding_content + "\n\n---\n\n" + msg.get("content", "")
                        })
                    else:
                        modified.append(msg)
            else:
                # Add grounding as first message
                modified = [grounding_message] + list(messages)
        
        # Always add reminder for long conversations (even if grounding exists)
        # This helps prevent the model from "drifting" into free-form mode
        reminder_added = False
        if add_reminder and len(messages) > 4:
            modified.append({
                "role": "system",
                "content": SNIPPET_GROUNDED_REMINDER
            })
            reminder_added = True
        
        # Estimate tokens added
        if grounding_already_present:
            # Only reminder tokens added (if any)
            tokens_added = self._estimate_tokens(SNIPPET_GROUNDED_REMINDER) if reminder_added else 0
        else:
            tokens_added = self._estimate_tokens(grounding_content)
            if reminder_added:
                tokens_added += self._estimate_tokens(SNIPPET_GROUNDED_REMINDER)
        
        return InjectionResult(
            modified_messages=modified,
            injection_applied=True,
            snippets_included=0 if grounding_already_present else len(snippets),
            estimated_tokens_added=tokens_added,
        )
    
    def _format_snippets(self, snippets: List[Union[str, SnippetWithSource, Dict]]) -> str:
        """
        Format snippets for inclusion in prompt.
        
        Supports multiple input formats:
        - Plain strings (legacy)
        - SnippetWithSource objects
        - Dicts with 'content' and optional 'source_path' keys
        """
        formatted_parts = []
        
        for i, snippet in enumerate(snippets, 1):
            # Extract content and source info
            if isinstance(snippet, SnippetWithSource):
                content = snippet.content
                source_path = snippet.source_path
                name = snippet.name
            elif isinstance(snippet, dict):
                content = snippet.get("content", str(snippet))
                source_path = snippet.get("source_path", "")
                name = snippet.get("name", "")
            else:
                content = str(snippet)
                source_path = ""
                name = ""
            
            # Detect language
            language = self._detect_language(content)
            
            # Get category from dict if available
            category = ""
            if isinstance(snippet, dict):
                category = snippet.get("category", "")
            elif isinstance(snippet, SnippetWithSource):
                category = snippet.category
            
            # Format WITHOUT file paths to prevent AI from navigating file system
            # Only show snippet name and category (no paths!)
            if category and category != "root" and name:
                header = f"### Snippet {i}: `{category}/{name}` (REFERENCE ONLY - DO NOT EDIT)"
            elif name:
                header = f"### Snippet {i}: `{name}` (REFERENCE ONLY - DO NOT EDIT)"
            else:
                header = f"### Snippet {i}: (REFERENCE ONLY - DO NOT EDIT)"
            
            formatted_parts.append(
                f"{header}\n```{language}\n{content.strip()}\n```"
            )
        
        return "\n\n".join(formatted_parts)
    
    def _detect_language(self, code: str) -> str:
        """Simple language detection based on code patterns."""
        code_lower = code.lower()
        
        if "def " in code or "import " in code or "class " in code_lower:
            if "self" in code or "def __" in code:
                return "python"
        if "function " in code or "const " in code or "let " in code:
            return "javascript"
        if "func " in code and "{" in code:
            return "go"
        if "fn " in code and "->" in code:
            return "rust"
        if "public " in code or "private " in code:
            if "class " in code:
                return "java"
        if "#include" in code:
            return "cpp"
        
        return ""  # No language tag
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars per token)."""
        return len(text) // 4
    
    def create_grounding_message(
        self, 
        snippets: List[str],
        strictness: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a standalone grounding message.
        
        Useful for manual injection or testing.
        
        Args:
            snippets: Code snippets to include
            strictness: Strictness level
            
        Returns:
            Message dict with role and content
        """
        level = strictness or self.strictness
        formatted = self._format_snippets(snippets)
        
        if level in GROUNDING_PROMPTS:
            content = GROUNDING_PROMPTS[level].format(snippets=formatted)
        else:
            content = SNIPPET_GROUNDED_SYSTEM_PROMPT.format(snippets=formatted)
        
        return {
            "role": "system",
            "content": content
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def should_inject_grounding(
    messages: List[Dict[str, Any]],
    mode: str,
    snippets: List[str]
) -> bool:
    """
    Determine if grounding should be injected.
    
    Args:
        messages: The message array
        mode: Current mode ('snippet_grounded' or 'free_form')
        snippets: Available snippets
        
    Returns:
        True if grounding should be injected
    """
    if mode != "snippet_grounded":
        return False
    
    if not snippets:
        return False
    
    # Always inject grounding in snippet mode to ensure context is maintained
    # The inject() function will handle adding reminders for long conversations
    # This ensures the model doesn't "drift" into free-form mode over time
    return True


def extract_user_message(messages: List[Dict[str, Any]]) -> str:
    """Extract the most recent user message."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            # Handle multi-modal content (list of content parts)
            if isinstance(content, list):
                # Extract text parts from content array
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif "text" in part:
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                return " ".join(text_parts)
            elif isinstance(content, str):
                return content
            else:
                return str(content)
    return ""


def count_conversation_turns(messages: List[Dict[str, Any]]) -> int:
    """Count the number of user messages (conversation turns)."""
    return sum(1 for msg in messages if msg.get("role") == "user")
