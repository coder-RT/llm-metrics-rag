"""
Configuration Loader for LLM Metrics

Loads and manages configuration from config.yaml file.
Provides easy access to snippet-grounded mode settings and other parameters.
"""

import os
from pathlib import Path
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass


# Try to import PyYAML, fall back to basic parsing if not available
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# Default configuration values
DEFAULT_CONFIG = {
    "snippet_grounded_mode": {
        "max_tokens": 500,
        "strictness": "moderate",
        "add_reminder": True,
        "reminder_threshold": 8,
        "show_source_path": True,
    },
    "free_form_mode": {
        "max_tokens": None,
        "track_metrics": True,
    },
    "cost_settings": {
        "input_cost_per_1k": 0.03,
        "output_cost_per_1k": 0.06,
        "model": "gpt-4",
    },
    "proxy": {
        "target_api": "openai",
        "timeout": 120,
        "debug": False,
    },
    "database": {
        "metrics_db": "data/llm_metrics.db",
        "config_db": "data/hackathon_config.db",
    },
}


@dataclass
class SnippetGroundedSettings:
    """Settings for snippet-grounded mode."""
    max_tokens: Optional[int] = 500
    strictness: str = "moderate"
    add_reminder: bool = True
    reminder_threshold: int = 8
    show_source_path: bool = True
    auto_load_all_snippets: bool = True
    smart_selection: bool = True  # Use query-based snippet matching
    use_rag: bool = True  # Use RAG (semantic search) for snippet selection
    max_snippets: int = 2  # Max snippets (RAG is accurate, 1-2 is usually enough)
    min_match_score: float = 0.3  # Min score for smart/RAG selection
    snippet_extensions: List[str] = None
    
    def __post_init__(self):
        if self.snippet_extensions is None:
            self.snippet_extensions = [".py", ".js", ".jsx", ".ts", ".tsx"]


@dataclass
class FreeFormSettings:
    """Settings for free-form mode."""
    max_tokens: Optional[int] = None
    track_metrics: bool = True


@dataclass
class CostSettings:
    """Settings for cost estimation."""
    input_cost_per_1k: float = 0.03
    output_cost_per_1k: float = 0.06
    model: str = "gpt-4"


@dataclass
class ProxySettings:
    """Settings for the proxy server."""
    host: str = "0.0.0.0"
    port: int = 8000
    target_url: str = "https://api.openai.com/v1"
    timeout: int = 120
    verify_ssl: Union[bool, str] = True  # True, False, or path to CA cert
    debug: bool = False


class ConfigLoader:
    """
    Loads and manages configuration from config.yaml.
    
    Usage:
        config = ConfigLoader()
        
        # Get snippet-grounded settings
        max_tokens = config.snippet_grounded.max_tokens
        strictness = config.snippet_grounded.strictness
        
        # Get cost settings
        input_cost = config.cost.input_cost_per_1k
        
        # Reload config
        config.reload()
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the config loader.
        
        Args:
            config_path: Path to config.yaml. Defaults to llmMetrics/config.yaml
        """
        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = Path(__file__).parent / "config.yaml"
        
        self._raw_config: Dict[str, Any] = {}
        self._snippet_grounded: Optional[SnippetGroundedSettings] = None
        self._free_form: Optional[FreeFormSettings] = None
        self._cost: Optional[CostSettings] = None
        self._proxy: Optional[ProxySettings] = None
        
        self.reload()
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._raw_config = self._load_config()
        self._parse_settings()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            print(f"Config file not found: {self.config_path}, using defaults")
            return DEFAULT_CONFIG.copy()
        
        if not YAML_AVAILABLE:
            print("PyYAML not installed, using default config")
            return DEFAULT_CONFIG.copy()
        
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config is None:
                    return DEFAULT_CONFIG.copy()
                return config
        except Exception as e:
            print(f"Error loading config: {e}, using defaults")
            return DEFAULT_CONFIG.copy()
    
    def _parse_settings(self) -> None:
        """Parse raw config into dataclasses."""
        # Snippet-grounded settings
        sg_config = self._raw_config.get("snippet_grounded_mode", {})
        self._snippet_grounded = SnippetGroundedSettings(
            max_tokens=sg_config.get("max_tokens", 500),
            strictness=sg_config.get("strictness", "moderate"),
            add_reminder=sg_config.get("add_reminder", True),
            reminder_threshold=sg_config.get("reminder_threshold", 8),
            show_source_path=sg_config.get("show_source_path", True),
            auto_load_all_snippets=sg_config.get("auto_load_all_snippets", True),
            smart_selection=sg_config.get("smart_selection", True),
            use_rag=sg_config.get("use_rag", True),
            max_snippets=sg_config.get("max_snippets", 2),
            min_match_score=sg_config.get("min_match_score", 0.3),
            snippet_extensions=sg_config.get("snippet_extensions", [".py", ".js", ".jsx", ".ts", ".tsx"]),
        )
        
        # Free-form settings
        ff_config = self._raw_config.get("free_form_mode", {})
        self._free_form = FreeFormSettings(
            max_tokens=ff_config.get("max_tokens"),
            track_metrics=ff_config.get("track_metrics", True),
        )
        
        # Cost settings
        cost_config = self._raw_config.get("cost_settings", {})
        self._cost = CostSettings(
            input_cost_per_1k=cost_config.get("input_cost_per_1k", 0.03),
            output_cost_per_1k=cost_config.get("output_cost_per_1k", 0.06),
            model=cost_config.get("model", "gpt-4"),
        )
        
        # Proxy settings
        proxy_config = self._raw_config.get("proxy", {})
        self._proxy = ProxySettings(
            host=proxy_config.get("host", "0.0.0.0"),
            port=proxy_config.get("port", 8000),
            target_url=proxy_config.get("target_url", "https://api.openai.com/v1"),
            timeout=proxy_config.get("timeout", 120),
            verify_ssl=proxy_config.get("verify_ssl", False),  # Default to False for internal networks
            debug=proxy_config.get("debug", False),
        )
    
    @property
    def snippet_grounded(self) -> SnippetGroundedSettings:
        """Get snippet-grounded mode settings."""
        return self._snippet_grounded
    
    @property
    def free_form(self) -> FreeFormSettings:
        """Get free-form mode settings."""
        return self._free_form
    
    @property
    def cost(self) -> CostSettings:
        """Get cost estimation settings."""
        return self._cost
    
    @property
    def proxy(self) -> ProxySettings:
        """Get proxy server settings."""
        return self._proxy
    
    @property
    def raw(self) -> Dict[str, Any]:
        """Get raw configuration dictionary."""
        return self._raw_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation key.
        
        Args:
            key: Dot-notation key (e.g., "snippet_grounded_mode.max_tokens")
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split(".")
        value = self._raw_config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_mode_settings(self, mode: str) -> Dict[str, Any]:
        """
        Get settings for a specific mode.
        
        Args:
            mode: "snippet_grounded" or "free_form"
            
        Returns:
            Dictionary of settings for that mode
        """
        if mode == "snippet_grounded":
            return {
                "max_tokens": self._snippet_grounded.max_tokens,
                "strictness": self._snippet_grounded.strictness,
                "add_reminder": self._snippet_grounded.add_reminder,
                "reminder_threshold": self._snippet_grounded.reminder_threshold,
                "show_source_path": self._snippet_grounded.show_source_path,
            }
        else:
            return {
                "max_tokens": self._free_form.max_tokens,
                "track_metrics": self._free_form.track_metrics,
            }


# Global config instance
_config: Optional[ConfigLoader] = None


def get_config() -> ConfigLoader:
    """Get or create the global config instance."""
    global _config
    if _config is None:
        _config = ConfigLoader()
    return _config


def reload_config() -> ConfigLoader:
    """Reload the global config instance."""
    global _config
    _config = ConfigLoader()
    return _config
