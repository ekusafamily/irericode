"""Unit tests for incremental error retry and session context preservation."""

import pytest
import asyncio
import json
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


def test_session_manager_error_log(tmp_path: Path):
    """Verify that SessionManager logs errors to .ireri/errors.log and retrieves them."""
    sm = SessionManager(workspace_root=tmp_path)
    assert sm.error_log_file == tmp_path / ".ireri" / "errors.log"

    # Initially empty
    assert sm.get_recent_errors() == []

    # Log several errors
    sm.log_error("MODEL_ERROR", "Rate limit exceeded 429", {"model": "nvidia/nemotron"})
    sm.log_error("TOOL_ERROR", "File not found: test.py", {"tool": "read_file"})
    sm.log_error("USER_INTERRUPT", "Task cancelled by Ctrl+C")

    assert sm.error_log_file.exists()
    content = sm.error_log_file.read_text(encoding="utf-8")
    assert "[MODEL_ERROR]" in content
    assert "Rate limit exceeded 429" in content
    assert "[TOOL_ERROR]" in content
    assert "[USER_INTERRUPT]" in content

    recent = sm.get_recent_errors(limit=2)
    assert len(recent) == 2
    assert recent[-1]["type"] == "USER_INTERRUPT"
    assert recent[-1]["message"] == "Task cancelled by Ctrl+C"


def test_controller_message_queue(tmp_path: Path):
    """Verify message queuing functionality in AgentController."""
    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)

    assert not controller.has_queued_messages()
    assert controller.get_queued_messages() == []

    # Queue multiple messages
    assert controller.queue_message("task 1") == 1
    assert controller.queue_message("task 2") == 2
    assert controller.queue_message("task 3") == 3

    assert controller.has_queued_messages()
    assert controller.get_queued_messages() == ["task 1", "task 2", "task 3"]

    # Pop message
    popped = controller.pop_queued_message()
    assert popped == "task 1"
    assert len(controller.get_queued_messages()) == 2

    # Clear queue
    cleared = controller.clear_message_queue()
    assert cleared == 2
    assert not controller.has_queued_messages()


@pytest.mark.asyncio
async def test_controller_btw_direct_chat_and_enlighten(tmp_path: Path):
    """Verify that /btw calls the model directly without tools and enlightens the agent context."""
    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)

    # Mock model client chat response
    mock_response = ModelResponse(content="We are using PostgreSQL with asyncpg.")
    controller.model_client.chat = AsyncMock(return_value=mock_response)

    # Ask direct question via /btw
    question = "What database are we using?"
    answer = await controller.ask_direct(question)

    assert answer == "We are using PostgreSQL with asyncpg."
    # Verify model was called with tools=None (pure chat, no tools)
    call_kwargs = controller.model_client.chat.call_args.kwargs
    assert call_kwargs.get("tools") is None

    # Verify context was enlightened with both question and answer
    user_msgs = [m for m in controller.context.messages if m.role == "user" and "[Side Note / BTW from User]" in m.content]
    assistant_msgs = [m for m in controller.context.messages if m.role == "assistant" and "[Direct Answer / Guidance]" in m.content]
    assert len(user_msgs) == 1
    assert question in user_msgs[0].content
    assert len(assistant_msgs) == 1
    assert "PostgreSQL with asyncpg" in assistant_msgs[0].content

    # Verify history log also captured the enlightenment
    assert sm.history_file.exists()
    log_text = sm.history_file.read_text(encoding="utf-8")
    assert "[BTW_USER]" in log_text
    assert question in log_text
    assert "[BTW_AGENT]" in log_text


def test_model_call_timeout_constant():
    """Verify default model call timeout is 180 seconds (3 minutes)."""
    from agent.model import MODEL_CALL_TIMEOUT
    assert MODEL_CALL_TIMEOUT == 180.0


@pytest.mark.asyncio
async def test_model_client_timeout_diagnosis_and_reload():
    """Verify that when a model call hangs/times out, it diagnoses the issue, reloads the client, and emits model_timeout event."""
    from agent.model import ModelClient

    client = ModelClient(api_key="test-key", base_url="http://test.local/v1", model="test-model", timeout=0.05)
    initial_internal_client = client.client

    # Simulate hanging model call
    async def hanging_create(**kwargs):
        await asyncio.sleep(1.0)  # longer than 0.05s timeout
        return MagicMock()

    client.client.chat.completions.create = hanging_create

    events = []
    def callback(event, data):
        events.append((event, data))

    # Mock diagnose_connection so test doesn't make real network calls
    with patch("agent.model.diagnose_connection", new_callable=AsyncMock) as mock_diag:
        mock_diag.return_value = {
            "reachable": True,
            "latency_ms": 45,
            "status_code": 200,
            "error": None,
            "summary": "Endpoint reachable and healthy (45ms). Provider under heavy load.",
        }

        with pytest.raises(TimeoutError, match="No response from model"):
            await client.chat(
                messages=[{"role": "user", "content": "hi"}],
                retry_delays=[],  # fail immediately on first timeout for test
                request_timeout=0.05,
                progress_callback=callback,
            )

        # Verify diagnosis was called
        mock_diag.assert_called_once_with("http://test.local/v1", "test-key")

    # Verify model_timeout event was emitted
    timeout_events = [e for e in events if e[0] == "model_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0][1]["diagnostic"]["reachable"] is True

    # Verify client was reloaded (re-instantiated)
    assert client.client is not initial_internal_client


def test_context_manager_token_tracking():
    """Verify ContextManager tracks prompt tokens, completion tokens, and serializes them."""
    ctx = ContextManager()
    assert ctx.total_prompt_tokens == 0
    assert ctx.total_completion_tokens == 0

    ctx.add_user_message("Hello, how are you?")
    stats1 = ctx.get_token_stats()
    assert stats1["tokens_up"] == 0
    assert stats1["tokens_down"] == 0
    assert stats1["approx_context_tokens"] > 0

    # Add usage
    ctx.add_token_usage(prompt_tokens=150, completion_tokens=45)
    ctx.add_token_usage(prompt_tokens=200, completion_tokens=60)

    stats2 = ctx.get_token_stats()
    assert stats2["tokens_up"] == 350
    assert stats2["tokens_down"] == 105
    assert stats2["total_tokens"] == 455
    assert stats2["approx_context_tokens"] > 0

    # Serialization to dict & restoration
    d = ctx.to_dict()
    assert d["total_prompt_tokens"] == 350
    assert d["total_completion_tokens"] == 105

    restored = ContextManager.from_dict(d)
    assert restored.total_prompt_tokens == 350
    assert restored.total_completion_tokens == 105
    assert restored.get_token_stats()["total_tokens"] == 455

    # Clear resets tokens
    ctx.clear()
    assert ctx.total_prompt_tokens == 0
    assert ctx.total_completion_tokens == 0
    assert ctx.get_token_stats()["total_tokens"] == 0


@pytest.mark.asyncio
async def test_controller_token_tracking_and_status(tmp_path: Path):
    """Verify AgentController tracks tokens during run_task and /btw, reporting via get_status()."""
    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)

    # Mock response with usage
    mock_resp = ModelResponse(
        content="Task completed successfully.",
        usage={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
    )
    controller.model_client.chat = AsyncMock(return_value=mock_resp)

    events = []
    def callback(event, data):
        events.append((event, data))

    result = await controller.run_task("Count to 10", progress_callback=callback)
    assert "Task completed successfully" in result

    # Check progress callback received token metrics
    resp_events = [e for e in events if e[0] == "model_response"]
    assert len(resp_events) == 1
    event_data = resp_events[0][1]
    assert event_data["tokens_up"] == 120
    assert event_data["tokens_down"] == 30
    assert event_data["total_tokens_up"] == 120
    assert event_data["total_tokens_down"] == 30

    # Verify controller.get_status()
    status = controller.get_status()
    assert status["tokens_up"] == 120
    assert status["tokens_down"] == 30
    assert status["total_tokens"] == 150
    assert status["approx_context_tokens"] > 0

    # Run /btw and verify tokens accumulate
    mock_btw_resp = ModelResponse(
        content="Direct answer.",
        usage={"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}
    )
    controller.model_client.chat = AsyncMock(return_value=mock_btw_resp)
    await controller.ask_direct("Quick question?")

    status2 = controller.get_status()
    assert status2["tokens_up"] == 170
    assert status2["tokens_down"] == 40
    assert status2["total_tokens"] == 210


@pytest.mark.asyncio
async def test_tool_error_sent_to_model_for_automatic_action(tmp_path: Path):
    """Verify that when a tool encounters an error, full diagnostic error details and automatic action instructions are sent to the model."""
    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)

    # First turn: model calls read_file on non-existent file
    tool_call = {
        "id": "call_123",
        "type": "function",
        "function": {
            "name": "read_file",
            "arguments": json.dumps({"path": "missing_file.py"}),
        },
    }
    resp1 = ModelResponse(
        content=None,
        tool_calls=[tool_call],
        usage={"prompt_tokens": 100, "completion_tokens": 20},
    )

    # Second turn: model receives error and takes corrective action
    resp2 = ModelResponse(
        content="The file was missing, so I created it and resolved the issue.",
        usage={"prompt_tokens": 150, "completion_tokens": 30},
    )

    controller.model_client.chat = AsyncMock(side_effect=[resp1, resp2])

    result = await controller.run_task("Inspect missing_file.py")
    assert "The file was missing" in result

    # Verify that the tool result in context contained the error and automatic action directive
    tool_messages = [m for m in controller.context.messages if m.role == "tool"]
    assert len(tool_messages) == 1
    content = tool_messages[0].content
    assert "[TOOL ERROR in read_file]" in content
    assert "File does not exist: missing_file.py" in content
    assert "AUTOMATIC ACTION REQUIRED" in content
    assert "take appropriate corrective actions automatically" in content

    # Verify error was logged in session manager error log
    assert sm.error_log_file.exists()
    error_content = sm.error_log_file.read_text(encoding="utf-8")
    assert "[TOOL_ERROR]" in error_content
    assert "missing_file.py" in error_content


@pytest.mark.asyncio
async def test_controller_recover_from_error_automatically(tmp_path: Path):
    """Verify that controller.recover_from_error sends the error to the model with an automatic action recovery prompt."""
    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)

    mock_resp = ModelResponse(
        content="I analyzed the connection error, started the local fallback service, and the issue is resolved and done.",
        usage={"prompt_tokens": 80, "completion_tokens": 25},
    )
    controller.model_client.chat = AsyncMock(return_value=mock_resp)

    events = []
    def callback(event, data):
        events.append((event, data))

    error_text = "ConnectionRefusedError: [Errno 111] Connection refused on localhost:5432"
    result = await controller.recover_from_error(error_text, progress_callback=callback)

    assert "analyzed the connection error" in result

    # Check that error_encountered was emitted
    error_events = [e for e in events if e[0] == "error_encountered"]
    assert len(error_events) == 1
    assert error_events[0][1]["error"] == error_text

    # Verify error recovery prompt in context
    user_msgs = [m for m in controller.context.messages if m.role == "user"]
    assert any("An error was encountered during execution" in m.content and error_text in m.content for m in user_msgs)
    assert any("take appropriate corrective actions" in m.content for m in user_msgs)

    # Verify logged to error.log
    assert sm.error_log_file.exists()
    log_text = sm.error_log_file.read_text(encoding="utf-8")
    assert "[AUTO_RECOVERY]" in log_text
    assert "ConnectionRefusedError" in log_text


@pytest.mark.asyncio
async def test_run_command_captures_error_on_nonzero_exit():
    """Verify that RunCommandTool captures failure output in error when command returns non-zero exit code."""
    from agent.tools.terminal import RunCommandTool, RunCommandArgs
    tool = RunCommandTool()

    # Run a command that fails
    res = await tool.execute(RunCommandArgs(command="python -c \"import sys; sys.stderr.write('fatal crash'); sys.exit(2)\""))
    assert res.success is False
    assert "fatal crash" in res.error
    assert res.output["returncode"] == 2


def test_config_tool_access_methods():
    """Verify AgentConfig tool_access get/set/reset and serialization."""
    from agent.config import AgentConfig

    cfg = AgentConfig()
    assert cfg.get_tool_access("write_file", default="ask") == "ask"
    assert cfg.get_tool_access("read_file", default="allow") == "allow"

    cfg.set_tool_access("write_file", "ALLOW")
    cfg.set_tool_access("delete_file", "deny")
    assert cfg.get_tool_access("write_file") == "allow"
    assert cfg.get_tool_access("delete_file") == "deny"

    with pytest.raises(ValueError, match="Invalid tool access policy"):
        cfg.set_tool_access("write_file", "invalid_policy")

    d = cfg.to_dict()
    assert d["tool_access"] == {"write_file": "allow", "delete_file": "deny"}

    restored = AgentConfig.from_dict(d)
    assert restored.get_tool_access("write_file") == "allow"
    assert restored.get_tool_access("delete_file") == "deny"

    restored.reset_tool_access("write_file")
    assert restored.get_tool_access("write_file", default="ask") == "ask"
    assert restored.get_tool_access("delete_file") == "deny"

    restored.reset_tool_access()
    assert restored.tool_access == {}


@pytest.mark.asyncio
async def test_controller_enforces_tool_access_policies(tmp_path: Path):
    """Verify AgentController enforces allow, ask, and deny tool_access policies."""
    from agent.config import reset_config
    reset_config()
    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)
    controller.config.workspace_root = tmp_path

    approval_calls = []

    async def mock_approval(tool_name, args, dangerous):
        approval_calls.append(tool_name)
        return True

    controller.approval_callback = mock_approval

    # 1. Deny policy blocks tool without calling approval_callback
    controller.config.set_tool_access("read_file", "deny")
    (tmp_path / "sample.txt").write_text("hello", encoding="utf-8")
    res_deny = await controller._execute_tool_call({
        "id": "c1",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "sample.txt"})},
    })
    assert res_deny.success is False
    assert "denied by tool access settings" in res_deny.error
    assert approval_calls == []

    # 2. Ask policy on read_file triggers approval_callback even though requires_approval=False
    controller.config.set_tool_access("read_file", "ask")
    res_ask = await controller._execute_tool_call({
        "id": "c2",
        "function": {"name": "read_file", "arguments": json.dumps({"path": "sample.txt"})},
    })
    assert res_ask.success is True
    assert approval_calls == ["read_file"]

    # 3. Allow policy on write_file skips approval_callback even though requires_approval=True
    controller.config.set_tool_access("write_file", "allow")
    res_allow = await controller._execute_tool_call({
        "id": "c3",
        "function": {"name": "write_file", "arguments": json.dumps({"path": "created.txt", "content": "ok"})},
    })
    assert res_allow.success is True
    assert approval_calls == ["read_file"]  # write_file did NOT prompt
    assert (tmp_path / "created.txt").read_text(encoding="utf-8") == "ok"


@pytest.mark.asyncio
async def test_handle_tools_settings_command(tmp_path: Path):
    """Verify /settings tools subcommands update and persist tool access policies."""
    from agent.cli import _handle_tools_settings_command

    sm = SessionManager(workspace_root=tmp_path)
    controller = AgentController(session_manager=sm)

    with patch("agent.config._save_global_config") as mock_save:
        await _handle_tools_settings_command(["write_file", "allow"], controller)
        assert controller.config.get_tool_access("write_file") == "allow"
        assert mock_save.call_count == 1

        await _handle_tools_settings_command(["delete_file", "deny"], controller)
        assert controller.config.get_tool_access("delete_file") == "deny"

        await _handle_tools_settings_command(["write_file", "reset"], controller)
        assert "write_file" not in controller.config.tool_access

        await _handle_tools_settings_command(["reset"], controller)
        assert controller.config.tool_access == {}







