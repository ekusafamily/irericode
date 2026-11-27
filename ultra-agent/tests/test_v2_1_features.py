"""Tests for v2.1.0 features: key-free init, NVIDIA NIM preset, API sync, denial feedback, slash auto-resolve, default command."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from typer.testing import CliRunner

from agent import __version__
from agent.cli import _resolve_first_slash_command, app
from agent.config import (
    NVIDIA_NIM_BASE_URL,
    NVIDIA_NIM_DEFAULT_MODEL,
    get_config,
    get_profiles,
    reset_config,
    sync_api_credentials,
)
from agent.controller import AgentController
from agent.model import ModelClient, ModelResponse
from agent.tools import Tool, ToolArgument, ToolRegistry, ToolResult


def test_version_is_2_1_3():
    assert __version__ == "2.1.3"


def test_nvidia_nim_preset_constants():
    assert NVIDIA_NIM_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert NVIDIA_NIM_DEFAULT_MODEL == "nvidia/nemotron-3.5-super-vl-120b-a12b"


@pytest.mark.asyncio
async def test_initialize_without_api_key(monkeypatch, tmp_path):
    """Agent and ModelClient should initialize cleanly without an API key."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr("agent.config.CONFIG_DIR", tmp_path / ".ireri-code")
    monkeypatch.setattr("agent.config.CONFIG_FILE", tmp_path / ".ireri-code" / "config.json")
    monkeypatch.setattr("agent.config.PROFILES_FILE", tmp_path / ".ireri-code" / "profiles.json")
    monkeypatch.chdir(tmp_path)
    reset_config()

    client = ModelClient(api_key="")
    assert client.api_key == ""

    with pytest.raises(ValueError, match="API key is not configured"):
        await client.chat(messages=[{"role": "user", "content": "hi"}], retry_delays=[0])


def test_sync_api_credentials(monkeypatch, tmp_path):
    """sync_api_credentials should save to global config, profiles, and os.environ."""
    monkeypatch.setattr("agent.config.CONFIG_DIR", tmp_path / ".ireri-code")
    monkeypatch.setattr("agent.config.CONFIG_FILE", tmp_path / ".ireri-code" / "config.json")
    monkeypatch.setattr("agent.config.PROFILES_FILE", tmp_path / ".ireri-code" / "profiles.json")
    monkeypatch.chdir(tmp_path)
    reset_config()

    cfg = sync_api_credentials(
        api_key="nvapi-test-1234",
        base_url=NVIDIA_NIM_BASE_URL,
        model=NVIDIA_NIM_DEFAULT_MODEL,
    )
    assert cfg.openai_api_key == "nvapi-test-1234"
    assert cfg.openai_base_url == NVIDIA_NIM_BASE_URL
    assert cfg.openai_model == NVIDIA_NIM_DEFAULT_MODEL
    assert os.environ["OPENAI_API_KEY"] == "nvapi-test-1234"

    profiles = get_profiles()
    assert "nvidia" in profiles
    assert profiles["nvidia"]["model"] == NVIDIA_NIM_DEFAULT_MODEL


def test_slash_command_first_suggestion_resolution():
    assert _resolve_first_slash_command("/se") == "/settings"
    assert _resolve_first_slash_command("/m") == "/model"
    assert _resolve_first_slash_command("/lo") == "/logs"
    assert _resolve_first_slash_command("/help") is None
    assert _resolve_first_slash_command("/q") is None
    assert _resolve_first_slash_command("/quit") is None
    assert _resolve_first_slash_command("hello") is None


class _DummyArgs(ToolArgument):
    value: str


class _DummyTool(Tool):
    name = "dummy_write"
    description = "Dummy tool requiring approval"
    args_schema = _DummyArgs
    requires_approval = True

    async def execute(self, args: _DummyArgs) -> ToolResult:
        return ToolResult(success=True, output="executed")


@pytest.mark.asyncio
async def test_tool_denial_with_optional_feedback(monkeypatch, tmp_path):
    """When user denies tool permission and gives feedback, it is sent to the model."""
    monkeypatch.setattr("agent.config.CONFIG_DIR", tmp_path / ".ireri-code")
    monkeypatch.setattr("agent.config.CONFIG_FILE", tmp_path / ".ireri-code" / "config.json")
    monkeypatch.setattr("agent.config.PROFILES_FILE", tmp_path / ".ireri-code" / "profiles.json")
    monkeypatch.chdir(tmp_path)
    reset_config()

    registry = ToolRegistry()
    registry.register(_DummyTool())

    mock_client = MagicMock(spec=ModelClient)
    mock_client.model = "test-model"
    mock_client.progress_callback = None
    mock_client.chat = AsyncMock(side_effect=[
        ModelResponse(
            content=None,
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "dummy_write", "arguments": '{"value": "x"}'},
            }],
        ),
        ModelResponse(content="Understood, I will use pytest instead. Task complete."),
    ])

    async def approval_cb(tool_name, args, dangerous):
        return False, "Use pytest instead"

    controller = AgentController(
        model_client=mock_client,
        tool_registry=registry,
        approval_callback=approval_cb,
    )
    result = await controller.run_task("do something")
    assert "pytest" in result

    tool_msgs = [m for m in controller.context.messages if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert "User instruction: Use pytest instead" in tool_msgs[0].content


def test_cli_default_command_is_chat():
    runner = CliRunner()
    res = runner.invoke(app, ["--help"])
    assert res.exit_code == 0
    assert "--project" in res.output
    assert "--model" in res.output
    assert "chat" in res.output


def test_progress_callback_handles_none_output_and_metadata():
    """Failed tool_complete events with output=None or metadata=None must not raise AttributeError."""
    from agent.cli import _progress_callback

    for tool_name in ("read_file", "edit_file", "write_file", "list_files", "delete_file"):
        _progress_callback(
            "tool_complete",
            {
                "tool": tool_name,
                "success": False,
                "output": None,
                "metadata": None,
                "error": "Path escapes workspace",
            },
        )


@pytest.mark.asyncio
async def test_get_process_logs_missing_id_includes_hint():
    from agent.tools.terminal import GetProcessLogsArgs, GetProcessLogsTool

    tool = GetProcessLogsTool()
    res = await tool.execute(GetProcessLogsArgs(process_id="proc-999"))
    assert res.success is False
    assert "Process not found: proc-999" in res.error


@pytest.mark.asyncio
async def test_streaming_emits_real_time_token_counts():
    """Streaming chat should emit live token_stream events as chunks arrive."""
    client = ModelClient(api_key="test-key", base_url="http://test.local/v1", model="test-model")

    class _Delta:
        def __init__(self, content=None):
            self.content = content
            self.tool_calls = None

    class _Choice:
        def __init__(self, content=None, finish_reason=None):
            self.delta = _Delta(content)
            self.finish_reason = finish_reason

    class _Chunk:
        def __init__(self, content=None, finish_reason=None, usage=None):
            self.choices = [_Choice(content, finish_reason)]
            self.usage = usage

    async def _fake_stream():
        yield _Chunk("Hello ")
        yield _Chunk("world!")
        yield _Chunk(None, finish_reason="stop")

    async def _fake_create(**kwargs):
        return _fake_stream()

    client.client.chat.completions.create = _fake_create
    events = []
    res = await client.chat(
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        progress_callback=lambda ev, data: events.append((ev, data)),
    )
    assert res.content == "Hello world!"
    stream_events = [d for ev, d in events if ev == "token_stream"]
    assert len(stream_events) >= 2
    assert stream_events[-1]["tokens_down"] >= stream_events[0]["tokens_down"]


def test_conversational_greeting_detection():
    mock_client = MagicMock(spec=ModelClient)
    mock_client.model = "test-model"
    mock_client.progress_callback = None
    ctrl = AgentController(model_client=mock_client)

    assert ctrl._is_conversational("hi!") is True
    assert ctrl._is_conversational("good morning") is True
    assert ctrl._is_conversational("Hello ireri!") is True
    assert ctrl._is_conversational("create a python calculator") is False
    assert ctrl._is_conversational("fix the bug in app.py") is False



