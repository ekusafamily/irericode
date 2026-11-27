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

    @staticmethod
    def _compact_old_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Shrink bulky write_file/edit_file payloads in older turns while preserving structure."""
        compacted = []
        for tc in tool_calls:
            tc_copy = dict(tc)
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            raw_args = fn.get("arguments", "")
            if fn_name in ("write_file", "edit_file") and isinstance(raw_args, str) and len(raw_args) > 300:
                try:
                    parsed = json.loads(raw_args)
                    if isinstance(parsed, dict):
                        slim = {"path": parsed.get("path", "")}
                        if "content" in parsed and isinstance(parsed["content"], str):
                            line_count = parsed["content"].count("\n") + 1
                            slim["note"] = f"[omitted {line_count} lines already written]"
                        else:
                            slim["note"] = "[omitted older edit payload]"
                        tc_copy["function"] = {
                            "name": fn_name,
                            "arguments": json.dumps(slim),
                        }
                except Exception:
                    pass
            compacted.append(tc_copy)
        return compacted

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 400) -> str:
        """Truncate long older tool output keeping head and tail."""
        if not text or len(text) <= max_chars:
            return text
        head = max_chars // 2
        tail = max_chars - head - 30
        omitted = len(text) - head - max(0, tail)
        return f"{text[:head]}\n...[omitted {omitted} chars]...\n{text[-tail:]}" if tail > 0 else text[:max_chars]

    def get_messages_for_model(self, keep_recent_messages: int = 12) -> List[Dict[str, Any]]:
        """Get messages formatted for the model API, pruning bulky older tool payloads."""
        total = len(self.messages)
        cutoff = max(0, total - keep_recent_messages)
        # Never split an assistant(tool_calls) from its following tool messages
        while cutoff > 0 and cutoff < total and self.messages[cutoff].role == "tool":
            cutoff -= 1

        result = []
        for idx, msg in enumerate(self.messages):
            is_old = idx < cutoff
            if msg.role == "system":
                result.append({"role": "system", "content": msg.content})
            elif msg.role == "user":
                result.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                msg_dict: Dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    msg_dict["content"] = msg.content
                if msg.tool_calls:
                    msg_dict["tool_calls"] = (
                        self._compact_old_tool_calls(msg.tool_calls)
                        if is_old
                        else msg.tool_calls
                    )
                result.append(msg_dict)
            elif msg.role == "tool":
                content = self._truncate_text(msg.content, max_chars=400) if is_old else msg.content
                result.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id,
                    "name": msg.name,
                    "content": content,
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
        """Rough token estimation of the active model payload."""
        total = 0
        for msg in self.get_messages_for_model():
            content = msg.get("content") or ""
            total += len(content) // 4
            if msg.get("tool_calls"):
                total += len(json.dumps(msg["tool_calls"])) // 4
        return total

    def trim_if_needed(self, model_max_tokens: int = 36_000) -> None:
        """Trim oldest non-system turn groups if context exceeds token budget."""
        available = model_max_tokens - 6000

        while self.estimate_tokens() > available and len(self.messages) > 4:
            # Find first non-system message
            pop_idx = None
            for i, msg in enumerate(self.messages):
                if msg.role != "system":
                    pop_idx = i
                    break
            if pop_idx is None:
                break

            removed = self.messages.pop(pop_idx)
            # If we removed an assistant message with tool_calls, also remove its paired tool results
            if removed.role == "assistant" and removed.tool_calls:
                while pop_idx < len(self.messages) and self.messages[pop_idx].role == "tool":
                    self.messages.pop(pop_idx)

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