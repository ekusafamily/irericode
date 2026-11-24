"""Session and chat history management for ireri."""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

from agent.context import ContextManager, ContextEntry

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages workspace session persistence and chat history logging."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = Path(workspace_root or Path.cwd()).resolve()
        self.session_dir = self._resolve_session_dir()
        self.session_file = self.session_dir / "session.json"
        self.history_file = self.session_dir / "chat_history.log"
        self._ensure_dir()

    def _resolve_session_dir(self) -> Path:
        """Resolve session directory inside workspace (.ireri/), or in ~/.ireri-code/sessions/ as fallback."""
        try:
            ws_dir = self.workspace_root / ".ireri"
            ws_dir.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = ws_dir / ".test_write"
            test_file.touch()
            test_file.unlink()
            return ws_dir
        except OSError:
            # Fallback to user home directory
            h = hashlib.sha256(str(self.workspace_root).encode()).hexdigest()[:12]
            name = self.workspace_root.name or "root"
            home_dir = Path.home() / ".ireri-code" / "sessions" / f"{name}_{h}"
            home_dir.mkdir(parents=True, exist_ok=True)
            return home_dir

    def _ensure_dir(self) -> None:
        """Ensure session directory exists."""
        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

    def save_session(self, context: ContextManager, model: str = "") -> bool:
        """Save the full ContextManager state to session.json."""
        try:
            self._ensure_dir()
            data = {
                "workspace_root": str(self.workspace_root),
                "saved_at": datetime.now().isoformat(),
                "model": model,
                "context": context.to_dict(),
            }
            with open(self.session_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False

    def load_session(self) -> Optional[Dict[str, Any]]:
        """Load saved session data if present."""
        if not self.session_file.exists():
            return None
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            logger.warning(f"Could not load session from {self.session_file}: {e}")
            return None

    def has_saved_session(self) -> bool:
        """Check if a valid session with user/assistant messages exists."""
        if not self.session_file.exists():
            return False
        try:
            data = self.load_session()
            if not data:
                return False
            ctx = data.get("context", {})
            msgs = ctx.get("messages", [])
            # Must have at least one user or assistant message
            return any(m.get("role") in ("user", "assistant") for m in msgs)
        except Exception:
            return False

    def clear_session(self) -> bool:
        """Clear the current session.json file while preserving the chat_history.log."""
        try:
            if self.session_file.exists():
                self.session_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Failed to clear session: {e}")
            return False

    def log_message(
        self,
        role: str,
        content: str,
        tool_name: Optional[str] = None,
        tool_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append an entry to the human-readable chat_history.log."""
        try:
            self._ensure_dir()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            lines = []

            if not self.history_file.exists() or self.history_file.stat().st_size == 0:
                header = (
                    f"{'='*80}\n"
                    f"ireri Chat History Log\n"
                    f"Workspace: {self.workspace_root}\n"
                    f"Created: {now_str}\n"
                    f"{'='*80}\n\n"
                )
                lines.append(header)

            role_upper = role.upper()
            if role_upper == "TOOL_CALL":
                lines.append(f"[{now_str}] [TOOL_CALL] {tool_name or ''}\n")
                if tool_args:
                    lines.append(f"Args: {json.dumps(tool_args, ensure_ascii=False)}\n")
                lines.append(f"Result: {content.strip()}\n\n")
            elif role_upper == "USER":
                lines.append(f"[{now_str}] [USER]\n{content.strip()}\n\n")
            elif role_upper in ("AGENT", "ASSISTANT"):
                lines.append(f"[{now_str}] [AGENT]\n{content.strip()}\n\n")
            else:
                lines.append(f"[{now_str}] [{role_upper}]\n{content.strip()}\n\n")

            with open(self.history_file, "a", encoding="utf-8") as f:
                f.writelines(lines)
        except Exception as e:
            logger.error(f"Failed to append to chat_history.log: {e}")

    def get_recent_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get the recent conversation messages from session."""
        data = self.load_session()
        if not data:
            return []
        msgs = data.get("context", {}).get("messages", [])
        # Filter for user and assistant messages for readable history
        readable = [m for m in msgs if m.get("role") in ("user", "assistant") and m.get("content")]
        return readable[-limit:]
