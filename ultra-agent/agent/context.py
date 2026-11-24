"""Context management for ireri."""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

from agent.config import get_config

import logging

logger = logging.getLogger(__name__)


@dataclass
class ContextEntry:
    """A single entry in the conversation context."""
    role: str  # user, assistant, tool
    content: str
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None  # for tool messages


class ContextManager:
    """Manages conversation context and project state."""

    def __init__(self, max_tokens: int = 100_000):
        self.max_tokens = max_tokens
        self.messages: List[ContextEntry] = []
        self.project_summary: str = ""
        self.files_inspected: Dict[str, int] = {}  # path -> line count
        self.tool_calls_count = 0
        self.iteration_count = 0
        self.total_prompt_tokens = 0       # Tokens up (sent to model)
        self.total_completion_tokens = 0   # Tokens down (received from model)

    def add_user_message(self, content: str) -> None:
        """Add a user message."""
        self.messages.append(ContextEntry(role="user", content=content))

    def add_assistant_message(
        self,
        content: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Add an assistant message."""
        self.messages.append(ContextEntry(
            role="assistant",
            content=content or "",
            tool_calls=tool_calls,
        ))

    def add_tool_result(
        self,
        tool_call_id: str,
        name: str,
        content: str,
    ) -> None:
        """Add a tool result message."""
        self.messages.append(ContextEntry(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            name=name,
        ))
        self.tool_calls_count += 1

    def add_system_message(self, content: str) -> None:
        """Add a system message (inserted at beginning)."""
        self.messages.insert(0, ContextEntry(role="system", content=content))

    def get_messages_for_model(self) -> List[Dict[str, Any]]:
        """Get messages formatted for the model API."""
        result = []
        for msg in self.messages:
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                msg_dict = {"role": "assistant"}
                if msg.content:
                    msg_dict["content"] = msg.content
                if msg.tool_calls:
                    msg_dict["tool_calls"] = msg.tool_calls
                result.append(msg_dict)
            elif msg.role == "tool":
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "content": msg.content,
                })
        return result

    def record_file_inspection(self, path: str, line_count: int) -> None:
        """Record that a file was inspected."""
        self.files_inspected[path] = line_count

    def get_project_context(self) -> str:
        """Get a summary of project context for the model."""
        parts = []

        if self.project_summary:
            parts.append(f"Project Summary:\n{self.project_summary}")

        if self.files_inspected:
            parts.append("Files inspected this session:")
            for path, lines in sorted(self.files_inspected.items()):
                parts.append(f"  - {path} ({lines} lines)")

        return "\n\n".join(parts) if parts else "No project context yet."

    def set_project_summary(self, summary: str) -> None:
        """Set the project summary."""
        self.project_summary = summary

    def increment_iteration(self) -> int:
        """Increment and return iteration count."""
        self.iteration_count += 1
        return self.iteration_count

    def reset_iteration_count(self) -> None:
        """Reset iteration count for a new task/message generation."""
        self.iteration_count = 0

    def get_iteration_count(self) -> int:
        """Get current iteration count."""
        return self.iteration_count

    def get_tool_calls_count(self) -> int:
        """Get total tool calls count."""
        return self.tool_calls_count

    def clear(self) -> None:
        """Clear context (keep system messages)."""
        system_msgs = [m for m in self.messages if m.role == "system"]
        self.messages = system_msgs
        self.tool_calls_count = 0
        self.iteration_count = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def add_token_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        """Accumulate token usage for the session."""
        self.total_prompt_tokens += max(0, prompt_tokens)
        self.total_completion_tokens += max(0, completion_tokens)

    def get_token_stats(self) -> Dict[str, Any]:
        """Get session token statistics and active context estimation."""
        return {
            "tokens_up": self.total_prompt_tokens,
            "tokens_down": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "approx_context_tokens": self.estimate_tokens(),
        }

    def estimate_tokens(self) -> int:
        """Rough token estimation."""
        total = 0
        for msg in self.messages:
            total += len(msg.content) // 4  # rough estimate
            if msg.tool_calls:
                total += len(json.dumps(msg.tool_calls)) // 4
        return total

    def trim_if_needed(self, model_max_tokens: int = 128_000) -> None:
        """Trim context if it exceeds limits."""
        # Reserve tokens for response and tools
        available = model_max_tokens - 8000

        while self.estimate_tokens() > available and len(self.messages) > 2:
            # Remove oldest non-system messages
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    self.messages.pop(i)
                    break

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context for persistence."""
        return {
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_calls": m.tool_calls,
                    "tool_call_id": m.tool_call_id,
                    "name": m.name,
                }
                for m in self.messages
            ],
            "project_summary": self.project_summary,
            "files_inspected": self.files_inspected,
            "tool_calls_count": self.tool_calls_count,
            "iteration_count": self.iteration_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContextManager":
        """Deserialize context."""
        ctx = cls()
        ctx.project_summary = data.get("project_summary", "")
        ctx.files_inspected = data.get("files_inspected", {})
        ctx.tool_calls_count = data.get("tool_calls_count", 0)
        ctx.iteration_count = data.get("iteration_count", 0)
        ctx.total_prompt_tokens = data.get("total_prompt_tokens", 0)
        ctx.total_completion_tokens = data.get("total_completion_tokens", 0)

        for msg_data in data.get("messages", []):
            ctx.messages.append(ContextEntry(**msg_data))

        return ctx