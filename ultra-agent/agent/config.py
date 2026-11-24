"""Configuration management for ireri."""

import json
import os
from pathlib import Path
from typing import Dict, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Global config directory
CONFIG_DIR = Path.home() / ".ireri-code"
CONFIG_FILE = CONFIG_DIR / "config.json"
MODELS_CACHE_FILE = CONFIG_DIR / "models.json"
PROFILES_FILE = CONFIG_DIR / "profiles.json"


class AgentConfig(BaseSettings):
    """Agent configuration loaded from global config file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,
    )

    # API Configuration
    openai_api_key: str = Field(default="", description="API key for OpenAI-compatible endpoint")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for the API endpoint"
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Model identifier to use"
    )

    # Agent Behavior
    max_iterations: int = Field(
        default=1000,
        description="Maximum agent loop iterations per task"
    )
    max_tool_calls: int = Field(
        default=1000,
        description="Maximum tool calls per task"
    )
    command_timeout: int = Field(
        default=60,
        description="Command execution timeout in seconds"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    # Tool Access Policies ("allow", "ask", "deny")
    tool_access: Dict[str, str] = Field(
        default_factory=dict,
        description="Per-tool access policies (allow, ask, deny)"
    )

    # workspace_root is not a pydantic field - managed via property
    _workspace_root: Optional[Path] = None

    def get_tool_access(self, tool_name: str, default: str = "ask") -> str:
        """Get configured access policy for a tool ('allow', 'ask', or 'deny')."""
        policy = self.tool_access.get(tool_name, default)
        if policy in ("allow", "ask", "deny"):
            return policy
        return default

    def set_tool_access(self, tool_name: str, policy: str) -> None:
        """Set access policy for a tool ('allow', 'ask', or 'deny')."""
        normalized = policy.strip().lower()
        if normalized not in ("allow", "ask", "deny"):
            raise ValueError(f"Invalid tool access policy '{policy}'. Must be 'allow', 'ask', or 'deny'.")
        self.tool_access[tool_name] = normalized

    def reset_tool_access(self, tool_name: Optional[str] = None) -> None:
        """Reset tool access policy override for one tool or all tools."""
        if tool_name is None:
            self.tool_access.clear()
        else:
            self.tool_access.pop(tool_name, None)

    @property
    def workspace_root(self) -> Path:
        """Get workspace root - defaults to current working directory at runtime."""
        if self._workspace_root is not None:
            return self._workspace_root.resolve()
        return Path.cwd().resolve()

    @workspace_root.setter
    def workspace_root(self, value: Path) -> None:
        """Setter for workspace_root (allows CLI override but not persisted)."""
        self._workspace_root = value

    def to_dict(self) -> dict:
        """Export config as dictionary for saving."""
        return {
            "openai_api_key": self.openai_api_key,
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "command_timeout": self.command_timeout,
            "log_level": self.log_level,
            "tool_access": dict(self.tool_access),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentConfig":
        """Create config from dictionary."""
        return cls(**data)


# Global config instance
_config: Optional[AgentConfig] = None


def _ensure_config_dir() -> None:
    """Ensure global config directory exists."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_global_config() -> dict:
    """Load configuration from global config file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_global_config(config: AgentConfig) -> None:
    """Save configuration to global config file."""
    _ensure_config_dir()
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def _load_profiles() -> dict:
    """Load API key profiles from file."""
    if PROFILES_FILE.exists():
        try:
            with open(PROFILES_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_profiles(profiles: dict) -> None:
    """Save API key profiles to file."""
    _ensure_config_dir()
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def get_profiles() -> dict:
    """Get all saved API key profiles."""
    return _load_profiles()


def save_profile(name: str, api_key: str, base_url: str = "https://api.openai.com/v1", model: str = "") -> None:
    """Save an API key profile."""
    profiles = _load_profiles()
    profiles[name] = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }
    _save_profiles(profiles)


def delete_profile(name: str) -> bool:
    """Delete an API key profile."""
    profiles = _load_profiles()
    if name in profiles:
        del profiles[name]
        _save_profiles(profiles)
        return True
    return False


def get_profile(name: str) -> Optional[dict]:
    """Get a specific API key profile."""
    profiles = _load_profiles()
    return profiles.get(name)


def load_profile(name: str) -> bool:
    """Load a profile as the active config."""
    profile = get_profile(name)
    if not profile:
        return False
    update_config(
        openai_api_key=profile["api_key"],
        openai_base_url=profile["base_url"],
    )
    if profile.get("model"):
        update_config(openai_model=profile["model"])
    return True


def _load_dotenv_file() -> dict:
    """Load .env file and return as dict."""
    from pathlib import Path
    env_path = Path(".env")
    if not env_path.exists():
        env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return {}

    result = {}
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                result[key.strip()] = value.strip()
    return result


def get_config() -> AgentConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        # Priority: global config file > .env file > defaults
        global_values = _load_global_config()
        dotenv_values = _load_dotenv_file()

        # Merge: global config takes priority, then .env, then defaults
        merged = {}
        merged.update(global_values)  # Global config (highest priority)
        for k, v in dotenv_values.items():
            if k not in merged:  # Only use .env if not in global config
                merged[k] = v

        _config = AgentConfig(**merged)
    return _config


def set_config(config: AgentConfig) -> None:
    """Set the global configuration instance and save to file."""
    global _config
    _config = config
    _save_global_config(config)


def reset_config() -> None:
    """Reset the global configuration instance (useful after config changes)."""
    global _config
    _config = None


def update_config(**kwargs) -> AgentConfig:
    """Update specific config values and save."""
    config = get_config()
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    set_config(config)
    return config


def get_available_models(base_url: str = None, api_key: str = None) -> list[dict]:
    """Fetch available models from OpenAI-compatible API."""
    import httpx
    config = get_config()
    url = (base_url or config.openai_base_url).rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key or config.openai_api_key}"}

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except Exception:
        return []


def cache_models(models: list[dict]) -> None:
    """Cache model list to file."""
    _ensure_config_dir()
    with open(MODELS_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2)


def get_cached_models() -> list[dict]:
    """Get cached model list."""
    if MODELS_CACHE_FILE.exists():
        try:
            with open(MODELS_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []