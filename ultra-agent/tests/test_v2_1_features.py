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


def test_version_is_2_1_9():
    assert __version__ == "2.1.9"


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


def test_google_provider_preset_and_sync(monkeypatch, tmp_path):
    from agent.config import (
        GOOGLE_GEMINI_BASE_URL,
        GOOGLE_GEMINI_DEFAULT_MODEL,
        GOOGLE_GEMINI_GEMMA_MODEL,
        MODEL_RATE_LIMITS_RPM,
    )

    assert GOOGLE_GEMINI_BASE_URL == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert GOOGLE_GEMINI_DEFAULT_MODEL == "gemini-3.5-flash-lite"
    assert GOOGLE_GEMINI_GEMMA_MODEL == "gemma-4-31b-it"
    assert MODEL_RATE_LIMITS_RPM["gemini-3.5-flash-lite"] == 15
    assert MODEL_RATE_LIMITS_RPM["gemma-4-31b-it"] == 30

    monkeypatch.setattr("agent.config.CONFIG_DIR", tmp_path / ".ireri-code")
    monkeypatch.setattr("agent.config.CONFIG_FILE", tmp_path / ".ireri-code" / "config.json")
    monkeypatch.setattr("agent.config.PROFILES_FILE", tmp_path / ".ireri-code" / "profiles.json")
    monkeypatch.chdir(tmp_path)
    reset_config()

    cfg = sync_api_credentials(
        api_key="AIza-test-key",
        base_url=GOOGLE_GEMINI_BASE_URL,
        model=GOOGLE_GEMINI_DEFAULT_MODEL,
    )
    assert cfg.openai_model == "gemini-3.5-flash-lite"
    profiles = get_profiles()
    assert "google" in profiles
    assert profiles["google"]["model"] == "gemini-3.5-flash-lite"


@pytest.mark.asyncio
async def test_model_rate_limiter_waits_for_reset(monkeypatch):
    from agent.model import ModelRateLimiter

    limiter = ModelRateLimiter(window_seconds=0.15)
    assert limiter.get_rpm_limit("gemini-3.5-flash-lite") == 15
    assert limiter.get_rpm_limit("models/gemini-3.5-flash-lite") == 15
    assert limiter.get_rpm_limit("gemma-4-31b-it") == 30
    assert limiter.get_rpm_limit("gpt-4o-mini") is None

    # Patch to 2 calls per 0.15s window for fast deterministic test
    monkeypatch.setitem(__import__("agent.config", fromlist=["MODEL_RATE_LIMITS_RPM"]).MODEL_RATE_LIMITS_RPM, "gemini-3.5-flash-lite", 2)

    events = []
    await limiter.acquire("gemini-3.5-flash-lite", progress_callback=lambda ev, d: events.append((ev, d)))
    await limiter.acquire("gemini-3.5-flash-lite", progress_callback=lambda ev, d: events.append((ev, d)))
    assert events == []

    waited = await limiter.acquire("gemini-3.5-flash-lite", progress_callback=lambda ev, d: events.append((ev, d)))
    assert waited > 0
    assert any(ev == "rate_limit_wait" and d["rpm_limit"] == 2 for ev, d in events)


def test_parse_response_preserves_extra_content_thought_signature():
    client = ModelClient(api_key="test-key", base_url="http://test.local/v1", model="gemini-3.5-flash-lite")

    tc = MagicMock()
    tc.id = "call_1"
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = "list_files"
    tc.function.arguments = '{"recursive": true}'
    tc.extra_content = {"google": {"thought_signature": "sig_abc123"}}

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    choice = MagicMock()
    choice.message = msg
    choice.finish_reason = "tool_calls"
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = None

    parsed = client._parse_response(resp)
    assert parsed.tool_calls[0]["extra_content"] == {"google": {"thought_signature": "sig_abc123"}}


@pytest.mark.asyncio
async def test_direct_text_reply_does_not_force_continuation(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mock_client = MagicMock(spec=ModelClient)
    mock_client.model = "gemini-3.5-flash-lite"
    mock_client.progress_callback = None
    mock_client.chat = AsyncMock(return_value=ModelResponse(content="Bonjour! How can I help?"))

    ctrl = AgentController(model_client=mock_client)
    assert ctrl._is_conversational("hiii") is True
    assert ctrl._is_conversational("bonjour") is True
    assert ctrl._is_conversational("mambo") is True

    res = await ctrl.run_task("tell me a joke")
    assert res == "Bonjour! How can I help?"
    assert mock_client.chat.call_count == 1


def test_format_tokens_compact():
    from agent.cli import format_tokens

    assert format_tokens(0) == "0"
    assert format_tokens(420) == "420"
    assert format_tokens(1000) == "1k"
    assert format_tokens(2742) == "2.7k"
    assert format_tokens(10000) == "10k"
    assert format_tokens(10500) == "10.5k"
    assert format_tokens(100000) == "100k"
    assert format_tokens(1500000) == "1.5M"


def test_render_task_with_side_tokens_collapsed_and_expanded():
    from agent.cli import render_task_with_side_tokens

    expanded = render_task_with_side_tokens("fix bug", "Done!", 2700, 350, 3000, collapsed=False)
    collapsed = render_task_with_side_tokens("fix bug", "Done!", 2700, 350, 3000, collapsed=True)
    assert expanded is not None
    assert collapsed is not None


def test_context_prunes_old_write_file_and_trims_safely():
    from agent.context import ContextManager

    ctx = ContextManager()
    ctx.add_system_message("sys")
    ctx.add_user_message("build app")

    big_code = "\n".join(f"print({i})" for i in range(150))
    ctx.add_assistant_message(
        content=None,
        tool_calls=[{
            "id": "call_old",
            "type": "function",
            "function": {"name": "write_file", "arguments": f'{{"path": "app.py", "content": "{big_code}"}}'.replace("\n", "\\n")},
        }],
    )
    ctx.add_tool_result("call_old", "write_file", "x" * 2000)

    # Add 14 recent messages so the old write_file falls outside keep_recent_messages=12
    for i in range(7):
        ctx.add_user_message(f"step {i}")
        ctx.add_assistant_message(f"ok {i}")

    formatted = ctx.get_messages_for_model(keep_recent_messages=12)
    old_assistant = [m for m in formatted if m.get("tool_calls")][0]
    assert "omitted" in old_assistant["tool_calls"][0]["function"]["arguments"]
    old_tool = [m for m in formatted if m["role"] == "tool"][0]
    assert len(old_tool["content"]) < 500

    # Pair-safe trimming: assistant+tool pair is removed together if over budget
    ctx.trim_if_needed(model_max_tokens=7000)
    roles = [m.role for m in ctx.messages]
    assert roles[0] == "system"
    # Never leave an orphan 'tool' message right after 'system' or 'user'
    for idx, r in enumerate(roles):
        if r == "tool":
            assert idx > 0 and (roles[idx - 1] in ("assistant", "tool"))


@pytest.mark.asyncio
async def test_google_tpm_and_rpd_limits(monkeypatch, tmp_path):
    from agent.config import GOOGLE_PEAK_RPD, GOOGLE_PEAK_TPM
    from agent.model import DailyQuotaExceededError, ModelRateLimiter

    assert GOOGLE_PEAK_TPM == 220_000
    assert GOOGLE_PEAK_RPD == 490

    usage_file = tmp_path / "google_usage.json"
    monkeypatch.setattr("agent.model.GOOGLE_USAGE_FILE", usage_file)

    limiter = ModelRateLimiter(window_seconds=0.15)
    events = []

    # 1st call with 150k input tokens succeeds immediately
    waited1 = await limiter.acquire(
        "gemini-3.5-flash-lite",
        est_input_tokens=150_000,
        progress_callback=lambda ev, d: events.append((ev, d)),
    )
    assert waited1 == 0.0

    # 2nd call with 100k input tokens exceeds 220k TPM -> waits for window reset
    waited2 = await limiter.acquire(
        "gemini-3.5-flash-lite",
        est_input_tokens=100_000,
        progress_callback=lambda ev, d: events.append((ev, d)),
    )
    assert waited2 > 0.0
    assert any(ev == "rate_limit_wait" and d.get("reason") == "tpm" for ev, d in events)

    # Daily RPD cap: simulate 490 requests already used today (PT)
    limiter._daily_count = 490
    usage = limiter.get_daily_usage("gemini-3.5-flash-lite")
    assert usage["is_google"] is True
    assert usage["used"] == 490
    assert usage["limit"] == 490
    with pytest.raises(DailyQuotaExceededError, match="490/490"):
        await limiter.acquire("gemini-3.5-flash-lite", est_input_tokens=100)


@pytest.mark.asyncio
async def test_read_file_dedup_default_cap_and_compact(monkeypatch, tmp_path):
    from agent.context import ContextManager
    from agent.tools.files import ReadFileArgs, ReadFileTool

    monkeypatch.chdir(tmp_path)
    reset_config()
    get_config().workspace_root = tmp_path

    # 1. read_file default 200-line cap
    big_file = tmp_path / "big.txt"
    big_file.write_text("\n".join(f"line {i}" for i in range(350)), encoding="utf-8")
    res = await ReadFileTool().execute(ReadFileArgs(path="big.txt"))
    assert res.success is True
    assert res.output["total_lines"] == 350
    assert res.output["end_line"] == 200
    assert res.output["truncated"] is True

    # 2. read_file deduplication + compact_history
    ctx = ContextManager()
    ctx.add_system_message("sys")
    ctx.add_user_message("first")
    ctx.add_assistant_message(
        content=None,
        tool_calls=[{"id": "r1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "big.txt"}'}}],
    )
    ctx.add_tool_result("r1", "read_file", "old file content " * 20)
    ctx.add_assistant_message(
        content=None,
        tool_calls=[{"id": "r2", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "big.txt"}'}}],
    )
    ctx.add_tool_result("r2", "read_file", "latest file content")

    formatted = ctx.get_messages_for_model()
    tool_msgs = [m for m in formatted if m["role"] == "tool"]
    assert "superseded" in tool_msgs[0]["content"]
    assert tool_msgs[1]["content"] == "latest file content"

    before, after = ctx.compact_history(keep_recent_messages=2)
    assert after < before
    assert ctx.messages[0].role == "system"


@pytest.mark.asyncio
async def test_nvidia_40_rpm_rate_limit(monkeypatch):
    from agent.config import NVIDIA_NIM_BASE_URL, NVIDIA_PEAK_RPM
    from agent.model import ModelRateLimiter

    assert NVIDIA_PEAK_RPM == 40
    limiter = ModelRateLimiter(window_seconds=0.15)
    assert limiter.get_rpm_limit("nvidia/nemotron-3.5-super-vl-120b-a12b", NVIDIA_NIM_BASE_URL) == 40
    assert limiter.get_rpm_limit("meta/llama-3.3-70b-instruct", NVIDIA_NIM_BASE_URL) == 40

    monkeypatch.setattr("agent.model.NVIDIA_PEAK_RPM", 2)
    events = []
    await limiter.acquire("meta/llama-3.3-70b-instruct", base_url=NVIDIA_NIM_BASE_URL, progress_callback=lambda e, d: events.append((e, d)))
    await limiter.acquire("nvidia/nemotron-3.5-super-vl-120b-a12b", base_url=NVIDIA_NIM_BASE_URL, progress_callback=lambda e, d: events.append((e, d)))
    waited = await limiter.acquire("meta/llama-3.3-70b-instruct", base_url=NVIDIA_NIM_BASE_URL, progress_callback=lambda e, d: events.append((e, d)))
    assert waited > 0
    assert any(e == "rate_limit_wait" and d.get("rpm_limit") == 2 for e, d in events)










