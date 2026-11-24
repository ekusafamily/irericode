"""Unit tests for incremental error retry and session context preservation."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from agent.model import ModelClient, RETRY_DELAYS, ModelResponse
from agent.context import ContextManager, ContextEntry
from agent.session import SessionManager
from agent.controller import AgentController


def test_retry_delays_schedule():
    """Verify that incremental delays match the user requirement: 1s, 5s, 15s, 30s, 60s (1m), 180s (3m)."""
    assert RETRY_DELAYS == [1, 5, 15, 30, 60, 180]


@pytest.mark.asyncio
async def test_model_client_retry_success_after_failure():
    """Verify that model_client retries on error and succeeds on subsequent attempt."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "Success after retry"
    mock_message.tool_calls = None
    mock_choice.message = mock_message
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage = None

    client = ModelClient(api_key="test-key", base_url="http://test.local/v1", model="test-model")

    # Create mock create function: fails once with ConnectionError, then returns mock_response
    attempts = 0
    async def mock_create(**kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("Connection refused by host")
        return mock_response

    client.client.chat.completions.create = mock_create

    events = []
    def callback(event, data):
        events.append((event, data))

    # Use fast test delays: [0, 0] instead of sleeping 1s, 5s in unit test
    result = await client.chat(
        messages=[{"role": "user", "content": "hi"}],
        retry_delays=[0, 0],
        progress_callback=callback,
    )

    assert result.content == "Success after retry"
    assert attempts == 2
    assert any(e[0] == "model_retry" for e in events)
    assert any(e[0] == "model_retry_success" for e in events)


@pytest.mark.asyncio
async def test_model_client_retry_exhaustion():
    """Verify that model_client exhausts retries and raises error."""
    client = ModelClient(api_key="test-key", base_url="http://test.local/v1", model="test-model")

    attempts = 0
    async def mock_create(**kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError(f"Persistent failure {attempts}")

    client.client.chat.completions.create = mock_create

    events = []
    def callback(event, data):
        events.append((event, data))

    with pytest.raises(RuntimeError, match="Persistent failure"):
        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            retry_delays=[0, 0, 0],
            progress_callback=callback,
        )

    # 1 initial + 3 retries = 4 attempts total
    assert attempts == 4
    retry_events = [e for e in events if e[0] == "model_retry"]
    assert len(retry_events) == 3


def test_session_manager_persistence(tmp_path: Path):
    """Verify SessionManager saves, loads, logs, and clears session context."""
    sm = SessionManager(workspace_root=tmp_path)
    assert sm.session_file.parent == tmp_path / ".ireri"
    assert not sm.has_saved_session()

    ctx = ContextManager()
    ctx.add_user_message("Hello ireri")
    ctx.add_assistant_message("Hello! How can I help?")
    ctx.record_file_inspection("test.py", 42)

    # Save session
    ok = sm.save_session(ctx, model="test-model")
    assert ok is True
    assert sm.has_saved_session() is True

    # Load session
    loaded_data = sm.load_session()
    assert loaded_data is not None
    assert loaded_data["model"] == "test-model"
    assert len(loaded_data["context"]["messages"]) == 2
    assert loaded_data["context"]["files_inspected"]["test.py"] == 42

    # Log messages
    sm.log_message("USER", "Hello ireri")
    sm.log_message("AGENT", "Hello! How can I help?")
    sm.log_message("TOOL_CALL", "File created", tool_name="write_file", tool_args={"path": "calc.py"})

    assert sm.history_file.exists()
    log_text = sm.history_file.read_text(encoding="utf-8")
    assert "[USER]" in log_text
    assert "Hello ireri" in log_text
    assert "[AGENT]" in log_text
    assert "[TOOL_CALL] write_file" in log_text

    # Recent history
    recent = sm.get_recent_history(limit=5)
    assert len(recent) == 2
    assert recent[0]["content"] == "Hello ireri"

    # Clear session
    sm.clear_session()
    assert not sm.has_saved_session()
    # History log should still be preserved
    assert sm.history_file.exists()


def test_controller_session_integration(tmp_path: Path):
    """Verify AgentController loads and saves session context."""
    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)

    # Add user message and save
    controller.context.add_user_message("Create an app")
    controller.context.add_assistant_message("I will create the app")
    controller.save_session()

    assert sm.has_saved_session()

    # Create new controller instance and load session
    new_controller = AgentController(session_manager=sm)
    restored = new_controller.load_session()
    assert restored is True

    # System message at 0, followed by restored user and assistant
    roles = [m.role for m in new_controller.context.messages]
    assert roles == ["system", "user", "assistant"]
    assert new_controller.context.messages[1].content == "Create an app"

    # Reset should clear session
    new_controller.reset()
    assert not sm.has_saved_session()
