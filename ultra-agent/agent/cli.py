"""Command-line interface for ireri."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.align import Align
from rich.columns import Columns
from rich.text import Text
from rich.box import ROUNDED

from agent.config import get_config, AgentConfig, reset_config
from agent.controller import AgentController, run_agent_task
from agent.context import ContextManager
from agent.model import get_model_client, reset_model_client
from agent.tools import get_registry
from typing import Dict, Any

console = Console()
app = typer.Typer(
    name="ireri",
    help="IRERI CODE - Terminal AI Coding Agent",
    add_completion=False,
)


def format_tokens(count: int) -> str:
    """Format token counts compactly (e.g. 420 -> '420', 1000 -> '1k', 2742 -> '2.7k', 10000 -> '10k')."""
    n = max(0, int(count or 0))
    if n < 1000:
        return str(n)
    if n < 1_000_000:
        val = n / 1000.0
        if val.is_integer() or n >= 100_000:
            return f"{int(round(val))}k"
        formatted = f"{val:.1f}".rstrip("0").rstrip(".")
        return f"{formatted}k"
    val = n / 1_000_000.0
    if val.is_integer() or n >= 100_000_000:
        return f"{int(round(val))}M"
    formatted = f"{val:.1f}".rstrip("0").rstrip(".")
    return f"{formatted}M"


def render_task_with_side_tokens(
    task: str,
    result: str,
    tokens_up: int,
    tokens_down: int,
    approx_context: int,
    collapsed: bool = False,
    rpd_info: Optional[Dict[str, Any]] = None,
) -> Table:
    """Render a task/response on the left and its token summary at the right side."""
    task_preview = task.strip().replace("\n", " ")
    if len(task_preview) > 64:
        task_preview = task_preview[:61] + "..."

    total = tokens_up + tokens_down
    grid = Table.grid(expand=True)
    grid.add_column(ratio=4)
    grid.add_column(width=24, justify="right")

    rpd_inline = f" [dim]RPD {rpd_info['used']}/{rpd_info['limit']}[/dim]" if rpd_info and rpd_info.get("is_google") else ""

    if collapsed:
        left_panel = Panel(
            f"[bold cyan]▸ {task_preview}[/bold cyan]  [dim](collapsed — /collapse to expand)[/dim]",
            border_style="dim",
            box=ROUNDED,
            padding=(0, 1),
        )
        right_panel = Panel(
            f"[cyan]↑{format_tokens(tokens_up)}[/cyan] [green]↓{format_tokens(tokens_down)}[/green] [magenta]Σ{format_tokens(total)}[/magenta]{rpd_inline}",
            border_style="dim",
            box=ROUNDED,
            padding=(0, 1),
        )
    else:
        left_panel = Panel(
            Markdown(result),
            title=f"[bold bright_green]▾ ireri[/bold bright_green] [dim]— {task_preview}[/dim]",
            border_style="bright_green",
            box=ROUNDED,
            padding=(0, 1),
        )
        rpd_line = f"\n[dim]RPD {rpd_info['used']}/{rpd_info['limit']}[/dim]" if rpd_info and rpd_info.get("is_google") else ""
        right_panel = Panel(
            f"[cyan]↑ {format_tokens(tokens_up)}[/cyan] sent\n"
            f"[green]↓ {format_tokens(tokens_down)}[/green] recv\n"
            f"[magenta]Σ {format_tokens(total)}[/magenta] total\n"
            f"[yellow]~{format_tokens(approx_context)}[/yellow] ctx{rpd_line}",
            title="[bold dim]Tokens[/bold dim]",
            border_style="cyan",
            box=ROUNDED,
            padding=(0, 1),
        )

    grid.add_row(left_panel, right_panel)
    return grid


def render_user_message(content: str) -> Panel:
    """Render a user message on the right side."""
    panel = Panel(
        Markdown(content),
        title="[bold cyan]You[/bold cyan]",
        border_style="cyan",
        box=ROUNDED,
        padding=(0, 1),
        width=min(console.width - 4, 100),
    )
    return Align.right(panel)


def render_agent_message(content: str) -> Panel:
    """Render an agent message on the left side."""
    panel = Panel(
        Markdown(content),
        title="[bold bright_green]ireri[/bold bright_green]",
        border_style="bright_green",
        box=ROUNDED,
        padding=(0, 1),
        width=min(console.width - 4, 100),
    )
    return Align.left(panel)


def render_system_message(content: str) -> Panel:
    """Render a system/info message centered."""
    if isinstance(content, str):
        try:
            text = Text.from_markup(content, justify="center")
        except Exception:
            text = Text(content, justify="center")
    else:
        text = content
    return Panel(
        text,
        border_style="dim",
        box=ROUNDED,
        padding=(0, 1),
        width=console.width - 4,
    )


def _progress_callback(event: str, data: Dict[str, Any], thinking: Optional["ThinkingIndicator"] = None) -> None:
    """Display progress updates from the agent."""
    if event == "iteration_start":
        if thinking:
            thinking.stop()
            ctx_toks = data.get("approx_context_tokens", 0)
            if ctx_toks:
                thinking.start(f"[bold bright_cyan]Thinking...[/bold bright_cyan]  [cyan]↑ {format_tokens(ctx_toks)} tokens[/cyan]")
            else:
                thinking.start("[bold bright_cyan]Thinking...[/bold bright_cyan]")
    elif event == "token_stream":
        if thinking:
            up = data.get("tokens_up", 0)
            down = data.get("tokens_down", 0)
            msg = (
                f"[bold bright_cyan]Counting tokens...[/bold bright_cyan]  "
                f"[cyan]↑ {format_tokens(up)}[/cyan] [dim]|[/dim] [green]↓ {format_tokens(down)}[/green] [dim]tokens[/dim]"
            )
            if thinking._status is None:
                thinking.start(msg)
            else:
                thinking.update(msg)
    elif event == "model_response":
        if thinking:
            thinking.stop()
        up = data.get("tokens_up", 0)
        down = data.get("tokens_down", 0)
        tok_str = ""
        if up or down:
            tok_str = f" [dim cyan](↑ {format_tokens(up)} sent, ↓ {format_tokens(down)} received)[/dim cyan]"

        if data.get("tool_calls_count", 0) > 0:
            console.print(f"\n  [cyan]Model: Planning {data['tool_calls_count']} tool call(s)[/cyan]{tok_str}")
        elif data.get("has_content"):
            console.print(f"\n  [cyan]Model: Responding...[/cyan]{tok_str}")
    elif event == "tool_start":
        if thinking:
            thinking.stop()
        console.print()  # spacing before tool action
        tool = data["tool"]
        args = data.get("args", {})
        if tool == "write_file":
            console.print(f"  [green]>[/green] Creating file: [bold]{args.get('path', 'unknown')}[/bold]")
        elif tool == "edit_file":
            console.print(f"  [yellow]>[/yellow] Editing file: [bold]{args.get('path', 'unknown')}[/bold]")
        elif tool == "read_file":
            console.print(f"  [blue]>[/blue] Reading file: [bold]{args.get('path', 'unknown')}[/bold]")
        elif tool == "list_files":
            console.print(f"  [blue]>[/blue] Listing directory: [bold]{args.get('path', '.')}[/bold]")
        elif tool == "run_command":
            cmd = args.get('command', 'unknown')
            console.print(f"  [magenta]>[/magenta] Running: [dim]{cmd[:60]}[/dim]...")
        elif tool == "delete_file":
            console.print(f"  [red]>[/red] Deleting: [bold]{args.get('path', 'unknown')}[/bold]")
        elif tool == "create_directory":
            console.print(f"  [green]>[/green] Creating directory: [bold]{args.get('path', 'unknown')}[/bold]")
        elif tool == "search_files":
            console.print(f"  [blue]>[/blue] Searching: [bold]{args.get('pattern', 'unknown')}[/bold]")
        elif tool == "grep":
            console.print(f"  [blue]>[/blue] Grepping: [bold]{args.get('pattern', 'unknown')}[/bold]")
        else:
            console.print(f"  [dim]>[/dim] {tool}: {args}")
    elif event == "tool_complete":
        if thinking:
            thinking.stop()
        tool = data["tool"]
        success = data.get("success", False)
        metadata = data.get("metadata") or {}
        output = data.get("output") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        if not isinstance(output, dict):
            output = {}

        if not success and tool != "run_command":
            console.print(f"  [red]FAIL[/red] {tool} failed: {data.get('error', 'unknown')}\n")
        elif tool == "write_file":
            action = metadata.get("action", "written")
            console.print(f"  [green]OK[/green] File {action}\n")
        elif tool == "edit_file":
            replacements = output.get("replacements", 0)
            console.print(f"  [yellow]OK[/yellow] File edited ({replacements} replacement(s))\n")
        elif tool == "read_file":
            lines = output.get("total_lines", 0)
            console.print(f"  [blue]OK[/blue] Read {lines} lines\n")
        elif tool == "list_files":
            count = metadata.get("count", 0)
            console.print(f"  [blue]OK[/blue] Found {count} items\n")
        elif tool == "run_command":
            if success:
                if metadata.get("background") and metadata.get("process_id"):
                    pid_str = metadata.get("pid", "?")
                    proc_id = metadata["process_id"]
                    console.print(
                        f"  [magenta]OK[/magenta] Started in background ([cyan]{proc_id}[/cyan], PID {pid_str}) "
                        f"— use [bold]/logs {proc_id}[/bold] to view logs\n"
                    )
                else:
                    console.print(f"  [magenta]OK[/magenta] Command completed\n")
            else:
                console.print(f"  [red]FAIL[/red] Command failed\n")
        elif tool == "delete_file":
            action = metadata.get("action", "deleted")
            console.print(f"  [red]OK[/red] {action}\n")
        elif tool == "create_directory":
            console.print(f"  [green]OK[/green] Directory created\n")
        elif tool == "search_files" or tool == "grep":
            console.print(f"  [blue]OK[/blue] Search completed\n")
        else:
            console.print(f"  [dim]OK[/dim] {tool} completed\n")
    elif event == "tool_error":
        if thinking:
            thinking.stop()
        console.print(f"  [red]FAIL[/red] {data['tool']} error: {data.get('error', 'unknown')}\n")
    elif event == "tool_denied":
        if thinking:
            thinking.stop()
        feedback = data.get("feedback")
        if feedback:
            console.print(f"  [yellow]DENIED[/yellow] {data['tool']} denied — [cyan]instruction sent: {feedback}[/cyan]\n")
        else:
            console.print(f"  [yellow]DENIED[/yellow] {data['tool']} denied\n")
    elif event == "model_error":
        if thinking:
            thinking.stop()
        console.print(f"\n  [red]Model error:[/red] {data.get('error', 'unknown')}\n")
    elif event == "model_retry":
        if thinking:
            thinking.stop()
        attempt = data.get("attempt", 1)
        max_attempts = data.get("max_attempts", 6)
        delay = data.get("delay", 1)
        err = data.get("error", "Error")
        delay_str = f"{delay // 60}m" if delay >= 60 else f"{delay}s"
        console.print(f"\n  [yellow]⚠ API error: {err}[/yellow]")
        console.print(f"  [yellow]  Retrying in {delay_str} (attempt {attempt}/{max_attempts})...[/yellow]\n")
        if thinking:
            thinking.start(f"[bold bright_cyan]Waiting {delay_str} before retry ({attempt}/{max_attempts})...[/bold bright_cyan]")
    elif event == "model_retry_success":
        if thinking:
            thinking.stop()
        console.print(f"\n  [green]✓[/green] API call recovered on retry attempt {data.get('attempt')}\n")
        if thinking:
            thinking.start("[bold bright_cyan]Thinking...[/bold bright_cyan]")
    elif event == "rate_limit_wait":
        if thinking:
            import math
            model_name = data.get("model", "model")
            reason = data.get("reason", "rpm")
            secs = max(1, math.ceil(data.get("wait_seconds", 1)))
            if reason == "tpm":
                tpm = data.get("tpm_limit", 220_000)
                limit_label = f"{format_tokens(tpm)} tokens/min"
            else:
                rpm = data.get("rpm_limit", "?")
                limit_label = f"{rpm}/min"
            msg = (
                f"[bold yellow]⏳ Rate limit ({model_name}, {limit_label}) — "
                f"resetting in {secs}s...[/bold yellow]"
            )
            if thinking._status is None:
                thinking.start(msg)
            else:
                thinking.update(msg)
    elif event == "model_timeout":
        if thinking:
            thinking.stop()
        diagnostic = data.get("diagnostic", {})
        summary = diagnostic.get("summary", "Could not diagnose connection issue.")
        attempt = data.get("attempt", 1)
        max_attempts = data.get("max_attempts", 6)
        timeout_sec = int(data.get("timeout", 180))
        console.print(f"\n  [bold red]⚠ No response from model within {timeout_sec // 60} minutes ({timeout_sec}s).[/bold red]")
        console.print(f"  [cyan]🔍 Diagnostic:[/cyan] [dim]{summary}[/dim]")
        console.print(f"  [bold green]🔄 Reloaded client connection pool.[/bold green] [yellow]Retrying (attempt {attempt}/{max_attempts})...[/yellow]\n")
        if thinking:
            thinking.start("[bold bright_cyan]Retrying request with reloaded connection...[/bold bright_cyan]")
    elif event == "error_encountered":
        if thinking:
            thinking.stop()
        err = data.get("error", "Unknown error")
        console.print(f"\n  [bold yellow]⚠ Error encountered:[/bold yellow] [yellow]{err}[/yellow]")
        console.print(f"  [cyan]↳ Automatically sent to model for analysis and corrective action...[/cyan]\n")
        if thinking:
            thinking.start("[bold bright_cyan]Model analyzing error and taking corrective action...[/bold bright_cyan]")


class _AwaitableNone:
    def __await__(self):
        async def _noop():
            return None
        return _noop().__await__()


class ThinkingIndicator:
    """Status indicator for when the agent/model is thinking."""

    def __init__(self, console: Console):
        self.console = console
        self._status = None

    def start(self, message: str = "[bold bright_cyan]Thinking...[/bold bright_cyan]"):
        """Start the status spinner."""
        if self._status is None:
            try:
                self._status = self.console.status(message, spinner="dots")
                self._status.start()
            except Exception:
                pass
        return _AwaitableNone()

    def update(self, message: str):
        """Update the spinner message in place (used for live token counting)."""
        if self._status is not None:
            try:
                self._status.update(message)
            except Exception:
                pass
        return _AwaitableNone()

    def stop(self):
        """Stop the status spinner."""
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:
                pass
            self._status = None
        return _AwaitableNone()


async def _prompt_arrow_menu(
    title: str,
    options: List[Dict[str, Any]],
    hotkeys: Optional[Dict[str, str]] = None,
    subtitle: str = "Use ↑/↓ arrows + Enter to select (Esc to cancel)",
    max_visible: int = 16,
    initial_index: int = 0,
) -> Optional[str]:
    """Render an interactive arrow-key selection menu using prompt_toolkit.

    Returns the selected option's 'key', or None if cancelled/unavailable.
    """
    if not sys.stdin.isatty():
        raise RuntimeError("stdin is not a TTY")
    if not options:
        return None

    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, HSplit, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.styles import Style

    selected_idx = max(0, min(initial_index, len(options) - 1))
    kb = KeyBindings()

    @kb.add("up")
    @kb.add("k")
    def _up(event):
        nonlocal selected_idx
        selected_idx = (selected_idx - 1) % len(options)

    @kb.add("down")
    @kb.add("j")
    def _down(event):
        nonlocal selected_idx
        selected_idx = (selected_idx + 1) % len(options)

    @kb.add("enter")
    def _enter(event):
        event.app.exit(result=options[selected_idx]["key"])

    @kb.add("escape")
    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    if hotkeys:
        for hk, target_key in hotkeys.items():
            @kb.add(hk)
            def _hotkey(event, _target=target_key):
                event.app.exit(result=_target)

    def _get_formatted_text():
        lines = []
        if title:
            lines.append(("class:title", f"  {title}\n"))
        if subtitle:
            lines.append(("class:subtitle", f"  {subtitle}\n"))

        total = len(options)
        if total > max_visible:
            half = max_visible // 2
            start = max(0, min(selected_idx - half, total - max_visible))
            end = start + max_visible
        else:
            start = 0
            end = total

        if start > 0:
            lines.append(("class:subtitle", f"    ↑ {start} more above...\n"))

        for idx in range(start, end):
            opt = options[idx]
            is_sel = idx == selected_idx
            pointer = "  ❯ " if is_sel else "    "
            style_cls = "class:selected" if is_sel else opt.get("pt_style", "class:item")
            lines.append((style_cls, f"{pointer}{opt['label']}\n"))

        if end < total:
            lines.append(("class:subtitle", f"    ↓ {total - end} more below...\n"))
        return lines

    style = Style.from_dict({
        "title": "bold ansibrightcyan",
        "subtitle": "ansibrightblack",
        "selected": "bold ansibrightyellow reverse",
        "item": "ansiwhite",
        "allow": "ansigreen",
        "deny": "ansired",
        "accent": "ansicyan",
    })

    app = Application(
        layout=Layout(HSplit([Window(content=FormattedTextControl(_get_formatted_text))])),
        key_bindings=kb,
        style=style,
        full_screen=False,
        erase_when_done=True,
    )
    return await app.run_async()


async def _ask_denial_feedback() -> Optional[str]:
    """Optionally ask the user what the agent should do instead after denying a tool."""
    loop = asyncio.get_event_loop()
    feedback = await loop.run_in_executor(
        None,
        lambda: Prompt.ask(
            "[dim cyan]Tell the agent what to do instead (optional, press Enter to skip)[/dim cyan]",
            default="",
        ),
    )
    cleaned = feedback.strip()
    return cleaned or None


async def _prompt_tool_approval_interactive(
    tool_name: str,
    args: Dict[str, Any],
    dangerous: bool,
    controller: AgentController,
) -> tuple[bool, Optional[str]]:
    """Interactive arrow-key selector for tool approval.

    Returns (approved, optional_feedback), or raises RuntimeError if TTY is unavailable.
    """
    from agent.config import _save_global_config

    while True:
        options = [
            {"key": "approve_once", "label": "[y] ✓ Approve once", "pt_style": "class:allow"},
            {"key": "deny_once", "label": "[n] ✗ Deny once", "pt_style": "class:deny"},
            {"key": "always_allow", "label": f"[p] ★ Always allow '{tool_name}' (save to /settings)", "pt_style": "class:allow"},
            {"key": "always_deny", "label": f"[b] ⊘ Never allow '{tool_name}' (save deny to /settings)", "pt_style": "class:deny"},
            {"key": "approve_all", "label": "[a] ✓✓ Approve all for this session", "pt_style": "class:accent"},
            {"key": "deny_all", "label": "[d] ✗✗ Deny all for this session", "pt_style": "class:deny"},
            {"key": "chat", "label": "[/] 💬 Ask model (/btw) or queue a message", "pt_style": "class:item"},
        ]
        hotkeys = {
            "y": "approve_once",
            "Y": "approve_once",
            "n": "deny_once",
            "N": "deny_once",
            "p": "always_allow",
            "P": "always_allow",
            "b": "always_deny",
            "B": "always_deny",
            "a": "approve_all",
            "A": "approve_all",
            "d": "deny_all",
            "D": "deny_all",
            "/": "chat",
        }

        choice = await _prompt_arrow_menu(
            title=f"Choose action for '{tool_name}':",
            options=options,
            hotkeys=hotkeys,
        )

        if choice is None:
            console.print("  [red]✗ Denied[/red]")
            return False, None
        elif choice == "deny_once":
            console.print("  [red]✗ Denied[/red]")
            feedback = await _ask_denial_feedback()
            return False, feedback
        elif choice == "approve_once":
            console.print("  [green]✓ Approved[/green]")
            return True, None
        elif choice == "always_allow":
            controller.config.set_tool_access(tool_name, "allow")
            _save_global_config(controller.config)
            console.print(render_system_message(f"[green]✓ Always allowing '{tool_name}' (saved to /settings)[/green]"))
            return True, None
        elif choice == "always_deny":
            controller.config.set_tool_access(tool_name, "deny")
            _save_global_config(controller.config)
            console.print(render_system_message(f"[red]⊘ Tool '{tool_name}' set to deny (saved to /settings)[/red]"))
            feedback = await _ask_denial_feedback()
            return False, feedback
        elif choice == "approve_all":
            controller.set_approve_all(True)
            console.print(render_system_message("[green]✓✓ Approved all future tool calls for this session[/green]"))
            return True, None
        elif choice == "deny_all":
            controller.set_deny_all(True)
            console.print(render_system_message("[red]✗✗ Denied all future tool calls for this session[/red]"))
            feedback = await _ask_denial_feedback()
            return False, feedback
        elif choice == "chat":
            loop = asyncio.get_event_loop()
            msg = await loop.run_in_executor(
                None,
                lambda: Prompt.ask("[bold cyan]Enter /btw <question> or message to queue (blank to go back)[/bold cyan]", default="")
            )
            cleaned = msg.strip()
            if not cleaned:
                continue
            if cleaned.lower().startswith("/btw"):
                question = cleaned[4:].strip()
                if question:
                    await _handle_btw_command(question, controller)
                else:
                    console.print("[yellow]Usage: /btw <question>[/yellow]")
            else:
                q_len = controller.queue_message(cleaned)
                console.print(render_system_message(f"[green]✓ Queued message #{q_len}:[/green] [white]{cleaned}[/white]"))
            # Loop back to the approval selector


async def _run_first_time_setup(config: AgentConfig) -> None:
    """Beginner-friendly first-run onboarding when no API key is configured."""
    from agent.config import (
        GOOGLE_GEMINI_BASE_URL,
        GOOGLE_GEMINI_DEFAULT_MODEL,
        NVIDIA_NIM_BASE_URL,
        NVIDIA_NIM_DEFAULT_MODEL,
        sync_api_credentials,
    )

    console.print()
    console.print(Panel(
        "[bold bright_cyan]Welcome to IRERI CODE![/bold bright_cyan]\n"
        "No API key is configured yet. Let's set up your AI provider in seconds\n"
        "(or skip and configure later using [cyan]/provider[/cyan]).\n\n"
        "[bold bright_yellow]★ Free Tier Presets Available ★[/bold bright_yellow]\n"
        f"• [bold]Google Gemini:[/bold] [green]{GOOGLE_GEMINI_DEFAULT_MODEL}[/green] (15 RPM) / [green]gemma-4-31b-it[/green] (30 RPM) — [underline cyan]https://aistudio.google.com/apikey[/underline cyan]\n"
        f"• [bold]NVIDIA NIM:[/bold]    [green]{NVIDIA_NIM_DEFAULT_MODEL}[/green] — [underline cyan]https://build.nvidia.com/[/underline cyan]",
        title="[bold green]Quick Setup[/bold green]",
        border_style="bright_cyan",
        box=ROUNDED,
        padding=(1, 2),
    ))

    choice: Optional[str] = None
    if sys.stdin.isatty():
        try:
            choice = await _prompt_arrow_menu(
                title="Choose how to configure ireri:",
                options=[
                    {
                        "key": "google",
                        "label": f"★ Google Gemini Preset (Free Tier — {GOOGLE_GEMINI_DEFAULT_MODEL}, 15 RPM)",
                        "pt_style": "class:allow",
                    },
                    {
                        "key": "nvidia",
                        "label": "★ NVIDIA NIM Preset (Free Tier — only enter API key)",
                        "pt_style": "class:allow",
                    },
                    {
                        "key": "custom",
                        "label": "⚙ Custom Provider (Base URL, API Key & Default Model)",
                        "pt_style": "class:accent",
                    },
                    {
                        "key": "skip",
                        "label": "→ Skip for now (initialize without API key)",
                        "pt_style": "class:item",
                    },
                ],
                hotkeys={"1": "google", "2": "nvidia", "3": "custom", "4": "skip"},
            )
        except Exception:
            choice = None

    if choice is None:
        return

    loop = asyncio.get_event_loop()
    if choice == "google":
        console.print(f"\n[dim]Get a free Google Gemini API key at: https://aistudio.google.com/apikey[/dim]")
        entered_key = (await loop.run_in_executor(
            None,
            lambda: Prompt.ask("[bold cyan]Enter Google Gemini API Key (AIza... or Enter to skip)[/bold cyan]", default=""),
        )).strip()
        sync_api_credentials(
            api_key=entered_key,
            base_url=GOOGLE_GEMINI_BASE_URL,
            model=GOOGLE_GEMINI_DEFAULT_MODEL,
            profile_name="google",
        )
        if entered_key:
            console.print(render_system_message(
                f"[green]✓ Google Gemini configured and synced![/green]\n"
                f"Default Model: [cyan]{GOOGLE_GEMINI_DEFAULT_MODEL}[/cyan] (15 calls/min limit)  •  Change anytime with [bold]/model[/bold]"
            ))
        else:
            console.print(render_system_message(
                "[yellow]Skipped API key. Google Gemini preset URL & model saved — run /provider google anytime.[/yellow]"
            ))
    elif choice == "nvidia":
        console.print(f"\n[dim]Get a free NVIDIA API key at: https://build.nvidia.com/[/dim]")
        entered_key = (await loop.run_in_executor(
            None,
            lambda: Prompt.ask("[bold cyan]Enter NVIDIA API Key (nvapi-... or Enter to skip)[/bold cyan]", default=""),
        )).strip()
        if entered_key:
            sync_api_credentials(
                api_key=entered_key,
                base_url=NVIDIA_NIM_BASE_URL,
                model=NVIDIA_NIM_DEFAULT_MODEL,
                profile_name="nvidia",
            )
            console.print(render_system_message(
                f"[green]✓ NVIDIA NIM configured and synced![/green]\n"
                f"Model: [cyan]{NVIDIA_NIM_DEFAULT_MODEL}[/cyan]  •  Change anytime with [bold]/model[/bold]"
            ))
        else:
            # Still pre-set NVIDIA base URL & default model so user only needs to add key later
            sync_api_credentials(
                api_key="",
                base_url=NVIDIA_NIM_BASE_URL,
                model=NVIDIA_NIM_DEFAULT_MODEL,
                profile_name="nvidia",
            )
            console.print(render_system_message(
                "[yellow]Skipped API key. NVIDIA NIM preset URL & model saved — run /provider nvidia anytime.[/yellow]"
            ))
    elif choice == "custom":
        entered_url = (await loop.run_in_executor(
            None,
            lambda: Prompt.ask("[bold cyan]Base URL[/bold cyan]", default=config.openai_base_url or "https://api.openai.com/v1"),
        )).strip()
        entered_key = (await loop.run_in_executor(
            None,
            lambda: Prompt.ask("[bold cyan]API Key (Enter to skip)[/bold cyan]", default=""),
        )).strip()
        entered_model = (await loop.run_in_executor(
            None,
            lambda: Prompt.ask("[bold cyan]Default Model[/bold cyan]", default=config.openai_model or "gpt-4o-mini"),
        )).strip()
        sync_api_credentials(
            api_key=entered_key,
            base_url=entered_url,
            model=entered_model,
        )
        if entered_key:
            console.print(render_system_message(
                f"[green]✓ Provider configured and synced![/green] Model: [cyan]{entered_model}[/cyan]"
            ))
        else:
            console.print(render_system_message(
                "[yellow]Saved Base URL & Model without API key — run /provider anytime to add your key.[/yellow]"
            ))
    else:
        console.print(render_system_message(
            "[dim]Starting without API key. Run [bold]/provider[/bold] or [bold]/settings[/bold] when ready.[/dim]"
        ))


async def _chat_async(
    project: Optional[Path],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_iterations: Optional[int] = None,
    auto_approve: bool = False,
    new_session: bool = False,
):
    """Async implementation of chat."""
    from agent.config import sync_api_credentials
    reset_config()
    config = get_config()

    # Override config with CLI args
    # workspace_root always defaults to current directory unless --project specified
    if project:
        config.workspace_root = project.resolve()
    else:
        config.workspace_root = Path.cwd().resolve()
    if model:
        config.openai_model = model
    if base_url:
        config.openai_base_url = base_url
    if api_key:
        config.openai_api_key = api_key
        sync_api_credentials(api_key=api_key, base_url=config.openai_base_url, model=config.openai_model)
    if max_iterations is not None:
        config.max_iterations = max_iterations

    # First-run onboarding if no API key is configured (agent still initializes without it!)
    if not config.openai_api_key and sys.stdin.isatty():
        await _run_first_time_setup(config)
        reset_config()
        config = get_config()
        if project:
            config.workspace_root = project.resolve()
        if model:
            config.openai_model = model
        if base_url:
            config.openai_base_url = base_url
        if max_iterations is not None:
            config.max_iterations = max_iterations

    # Welcome banner - IRERI CODE in large font
    console.print()
    banner_lines = [
        "  ██╗██████╗ ███████╗██████╗ ██╗    ██████╗ ██████╗ ██████╗ ███████╗",
        "  ██║██╔══██╗██╔════╝██╔══██╗██║   ██╔════╝██╔═══██╗██╔══██╗██╔════╝",
        "  ██║██████╔╝█████╗  ██████╔╝██║   ██║     ██║   ██║██║  ██║█████╗  ",
        "  ██║██╔══██╗██╔══╝  ██╔══██╗██║   ██║     ██║   ██║██║  ██║██╔══╝  ",
        "  ██║██║  ██║███████╗██║  ██║██║   ╚██████╗╚██████╔╝██████╔╝███████╗",
        "  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
    ]
    for line in banner_lines:
        console.print(f"[bold bright_cyan]{line}[/bold bright_cyan]")
    console.print()
    console.print("[bold bright_green]  Terminal AI Coding Agent  v2.1.9[/bold bright_green]")
    console.print()
    key_status = "[green]Configured[/green]" if config.openai_api_key else "[yellow]Not set (/provider)[/yellow]"
    console.print(render_system_message(
        f"Model: [green]{config.openai_model}[/green]  •  "
        f"API Key: {key_status}  •  "
        f"Workspace: [green]{config.workspace_root}[/green]"
    ))
    console.print(render_system_message("[dim]Type 'exit' or 'quit' to leave. Type '/help' for commands.[/dim]"))
    console.print()

    # Initialize controller
    controller = AgentController()
    global _current_controller
    _current_controller = controller
    if auto_approve:
        controller.set_approve_all(True)

    # Check for preserved previous session to continue
    if not new_session and controller.has_saved_session():
        restored = controller.load_session()
        if restored:
            user_msg_count = len([m for m in controller.context.messages if m.role == "user"])
            console.print(render_system_message(
                f"[green]Preserved session loaded[/green] ([bold cyan]{user_msg_count}[/bold cyan] previous interaction(s) restored). "
                f"[dim]Type '/history' to view, or '/reset' to start fresh.[/dim]"
            ))
            console.print()

    thinking = ThinkingIndicator(console)
    tasks_collapsed = False
    task_history: List[Dict[str, Any]] = []

    # Initialize prompt_toolkit session with auto-completion, Enter auto-select, and dynamic prompt
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.shortcuts import CompleteStyle
    pt_completer = SlashCommandCompleter()
    pt_kb = KeyBindings()

    @pt_kb.add("enter")
    def _pt_enter(event):
        buf = event.current_buffer
        text = buf.text.strip()
        if buf.complete_state:
            if buf.complete_state.current_completion is not None:
                buf.apply_completion(buf.complete_state.current_completion)
            elif (
                buf.complete_state.completions
                and text.startswith("/")
                and " " not in text
                and text.lower() not in SLASH_COMMANDS
                and text.lower() not in ("/q", "/quit", "/log", "/processes", "/cancel")
            ):
                buf.apply_completion(buf.complete_state.completions[0])
        elif text.startswith("/") and " " not in text:
            matched = _resolve_first_slash_command(text)
            if matched:
                buf.text = matched
                buf.cursor_position = len(matched)
        buf.validate_and_handle()

    def _get_token_toolbar():
        from agent.model import get_rate_limiter
        stats = controller.context.get_token_stats()
        collapse_label = "Collapsed" if tasks_collapsed else "Expanded"
        rpd = get_rate_limiter().get_daily_usage(controller.model_client.model, controller.model_client.base_url)
        rpd_part = (
            f"  |  <ansiyellow>RPD: {rpd['used']}/{rpd['limit']}</ansiyellow>"
            if rpd.get("is_google")
            else ""
        )
        return HTML(
            f" <b>Tokens:</b> "
            f"<ansicyan>↑ {format_tokens(stats['tokens_up'])} sent</ansicyan>  |  "
            f"<ansigreen>↓ {format_tokens(stats['tokens_down'])} recv</ansigreen>  |  "
            f"<ansimagenta>Σ {format_tokens(stats['total_tokens'])}</ansimagenta>  |  "
            f"<ansiyellow>~{format_tokens(stats['approx_context_tokens'])} ctx</ansiyellow>"
            f"{rpd_part}  |  "
            f"<ansibrightblack>Tasks: {collapse_label} (/collapse)</ansibrightblack> "
        )

    pt_session = PromptSession(
        erase_when_done=True,
        completer=pt_completer,
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
        style=SLASH_COMPLETION_STYLE,
        key_bindings=pt_kb,
        bottom_toolbar=_get_token_toolbar,
    )
    approval_handoff_event = asyncio.Event()

    # Define approval callback for interactive mode
    async def _approval_callback(tool_name: str, args: Dict[str, Any], dangerous: bool) -> tuple[bool, Optional[str]]:
        """Prompt user for tool approval with interactive arrow selection and /btw support."""
        if controller._approve_all:
            return True, None
        if controller._deny_all:
            return False, None

        thinking.stop()
        from rich.markup import escape

        # Format args for display
        args_display = ", ".join(f"{k}={v}" for k, v in args.items())
        if len(args_display) > 100:
            args_display = args_display[:97] + "..."

        # Build prompt message
        danger_indicator = " [red]! DANGEROUS[/red]" if dangerous else ""
        console.print()
        approval_text = Text.from_markup(
            f"[yellow]Approval required:[/yellow] {escape(tool_name)}({escape(args_display)}){danger_indicator}",
            justify="center",
        )
        console.print(Panel(approval_text, border_style="yellow" if not dangerous else "red", box=ROUNDED, padding=(0, 1), width=console.width - 4))

        loop = asyncio.get_event_loop()
        future = loop.create_future()
        controller.active_approval = future

        # Hand off stdin from pt_session if it is currently running
        try:
            if pt_session.app.is_running:
                approval_handoff_event.clear()
                pt_session.app.exit(result="__APPROVAL_HANDOFF__")
                try:
                    await asyncio.wait_for(approval_handoff_event.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        except Exception:
            pass

        try:
            # Try interactive arrow-key selector first
            try:
                approved = await _prompt_tool_approval_interactive(tool_name, args, dangerous, controller)
                if not future.done():
                    future.set_result(approved)
                return approved
            except asyncio.CancelledError:
                raise
            except Exception:
                # Fallback to text-based prompt for non-TTY / automated environments
                options_text = Text.from_markup(
                    "Options: [bold]y[/bold]=approve  [bold]n[/bold]=deny  [bold]a[/bold]=approve all  [bold]d[/bold]=deny all  [bold]/btw <q>[/bold]=ask model",
                    justify="center",
                )
                console.print(Panel(options_text, border_style="dim", box=ROUNDED, padding=(0, 1), width=console.width - 4))
                approved = await future
                return approved
        except asyncio.CancelledError:
            return False
        finally:
            controller.active_approval = None

    controller.approval_callback = _approval_callback

    background_worker_task: Optional[asyncio.Task] = None

    async def _run_task_and_drain_queue(task: str):
        """Background worker executing tasks from the queue in sequence while prompt stays active."""
        current_task = task
        while current_task and not controller.cancelled:
            pre_stats = controller.context.get_token_stats()
            try:
                result = await controller.run_task(
                    current_task,
                    progress_callback=lambda event, data: _progress_callback(event, data, thinking),
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                controller.session_manager.log_error("TASK_ERROR", str(e))
                if not controller.cancelled:
                    console.print()
                    console.print(render_system_message(
                        f"[yellow]⚠ Error encountered: {e}[/yellow]\n"
                        f"[cyan]→ Sending error to model for automatic corrective actions...[/cyan]"
                    ))
                    try:
                        thinking.start("[bold bright_cyan]Model taking corrective actions...[/bold bright_cyan]")
                        result = await controller.recover_from_error(
                            error=str(e),
                            progress_callback=lambda event, data: _progress_callback(event, data, thinking),
                        )
                    except Exception as recovery_err:
                        result = f"Error during task: {e}\nAutomatic recovery failed: {recovery_err}"
                else:
                    result = "Task cancelled by user"
            finally:
                thinking.stop()

            stats = controller.context.get_token_stats()
            task_up = max(0, stats["tokens_up"] - pre_stats["tokens_up"])
            task_down = max(0, stats["tokens_down"] - pre_stats["tokens_down"])
            task_history.append({
                "task": current_task,
                "result": result,
                "tokens_up": task_up,
                "tokens_down": task_down,
                "approx_context": stats["approx_context_tokens"],
                "collapsed": tasks_collapsed,
            })

            from agent.model import get_rate_limiter
            rpd_info = get_rate_limiter().get_daily_usage(controller.model_client.model, controller.model_client.base_url)

            # Display task/result on the left with tokens at the side
            console.print()
            console.print(render_task_with_side_tokens(
                task=current_task,
                result=result,
                tokens_up=task_up,
                tokens_down=task_down,
                approx_context=stats["approx_context_tokens"],
                collapsed=tasks_collapsed,
                rpd_info=rpd_info,
            ))

            # Process next queued message if available
            if controller.has_queued_messages() and not controller.cancelled:
                current_task = controller.pop_queued_message()
                remaining = len(controller.message_queue)
                console.print()
                console.print(render_system_message(
                    f"[bold cyan]▶ Running queued task ({remaining} remaining in queue):[/bold cyan]\n"
                    f"[white]{current_task}[/white]"
                ))
                console.print(render_user_message(current_task))
            else:
                current_task = None

        # Ensure prompt_toolkit redraws a clean input prompt after background task finishes
        try:
            if pt_session.app.is_running:
                pt_session.app.invalidate()
        except Exception:
            pass

    # Chat loop
    async def _chat_loop():
        nonlocal background_worker_task
        while True:
            try:
                # If interactive approval is currently active on TTY, wait for it to finish
                if controller.active_approval and not controller.active_approval.done() and sys.stdin.isatty():
                    try:
                        await asyncio.shield(controller.active_approval)
                    except (asyncio.CancelledError, Exception):
                        pass
                    continue

                console.print()
                # Get user input with prompt_toolkit (patch_stdout keeps prompt clean during background output)
                try:
                    from prompt_toolkit.patch_stdout import patch_stdout
                    with patch_stdout(raw=True):
                        user_input = await pt_session.prompt_async(
                            _get_dynamic_prompt,
                            completer=pt_completer,
                            complete_while_typing=True,
                            complete_style=CompleteStyle.COLUMN,
                            style=SLASH_COMPLETION_STYLE,
                        )
                except (EOFError, KeyboardInterrupt):
                    raise
                except Exception:
                    loop = asyncio.get_event_loop()
                    user_input = await loop.run_in_executor(
                        None,
                        lambda: _get_input_with_readline()
                    )

                if user_input == "__APPROVAL_HANDOFF__":
                    approval_handoff_event.set()
                    if controller.active_approval and not controller.active_approval.done():
                        try:
                            await asyncio.shield(controller.active_approval)
                        except (asyncio.CancelledError, Exception):
                            pass
                    continue

                if user_input.lower() in ("exit", "quit", "q"):
                    if background_worker_task and not background_worker_task.done():
                        controller.cancel()
                        background_worker_task.cancel()
                    console.print()
                    console.print(render_system_message("[yellow]Goodbye from ireri![/yellow]"))
                    break

                if not user_input.strip():
                    continue

                # Check if tool approval is currently waiting
                if controller.active_approval and not controller.active_approval.done():
                    cleaned = user_input.strip()
                    if cleaned.lower().startswith("/btw"):
                        question = cleaned[4:].strip()
                        if question:
                            await _handle_btw_command(question, controller)
                        else:
                            console.print("[yellow]Usage: /btw <question>[/yellow]")
                        continue

                    if cleaned.lower().startswith("/queue"):
                        await _handle_queue_command(cleaned.split(), controller)
                        continue

                    choice = cleaned.lower()
                    if choice == "a":
                        controller.set_approve_all(True)
                        console.print(render_system_message("[green]Approved all future tool calls for this session[/green]"))
                        controller.active_approval.set_result(True)
                        continue
                    elif choice == "d":
                        controller.set_deny_all(True)
                        console.print(render_system_message("[red]Denied all future tool calls for this session[/red]"))
                        controller.active_approval.set_result(False)
                        continue
                    elif choice in ("y", "yes"):
                        controller.active_approval.set_result(True)
                        continue
                    elif choice in ("n", "no"):
                        controller.active_approval.set_result(False)
                        continue
                    else:
                        q_len = controller.queue_message(cleaned)
                        console.print(render_system_message(
                            f"[green]✓ Queued message #{q_len}:[/green] [white]{cleaned}[/white]\n"
                            f"[dim]Please approve or deny the current tool call with [bold]y[/bold] or [bold]n[/bold].[/dim]"
                        ))
                        continue

                # Handle /cancel command
                if user_input.lower() in ("/cancel", "cancel"):
                    is_working = controller.running or (background_worker_task and not background_worker_task.done())
                    if is_working:
                        controller.cancel()
                        if background_worker_task:
                            background_worker_task.cancel()
                        if controller.active_approval and not controller.active_approval.done():
                            controller.active_approval.cancel()
                        thinking.stop()
                        controller.cancelled = False
                        console.print(render_system_message("[yellow]⚠ Task cancelled by user.[/yellow]"))
                    else:
                        console.print(render_system_message("[dim]No active task running to cancel.[/dim]"))
                    continue

                # Handle /btw side-chat (works both when working and when idle!)
                if user_input.lower().startswith("/btw"):
                    question = user_input[4:].strip()
                    if question:
                        await _handle_btw_command(question, controller)
                    else:
                        console.print("[yellow]Usage: /btw <your question or side note>[/yellow]")
                        console.print("[dim]Example: /btw what database are we using?[/dim]")
                    continue

                # Handle /queue command
                if user_input.lower().startswith("/queue"):
                    await _handle_queue_command(user_input.split(), controller)
                    continue

                # Handle /collapse and /tasks commands
                if user_input.strip().lower() in ("/collapse", "/expand"):
                    tasks_collapsed = not tasks_collapsed if user_input.strip().lower() == "/collapse" else False
                    for entry in task_history:
                        entry["collapsed"] = tasks_collapsed
                    state_word = "collapsed" if tasks_collapsed else "expanded"
                    console.print(render_system_message(f"[cyan]Tasks {state_word}.[/cyan]"))
                    for entry in task_history:
                        console.print(render_task_with_side_tokens(
                            task=entry["task"],
                            result=entry["result"],
                            tokens_up=entry["tokens_up"],
                            tokens_down=entry["tokens_down"],
                            approx_context=entry["approx_context"],
                            collapsed=entry["collapsed"],
                        ))
                    continue

                if user_input.strip().lower() == "/tasks":
                    if not task_history:
                        console.print(render_system_message("[dim]No completed tasks in this session yet.[/dim]"))
                        continue
                    while True:
                        opts = [
                            {
                                "key": "__toggle_all__",
                                "label": f"{'▾ Expand all tasks' if tasks_collapsed else '▸ Collapse all tasks'}",
                                "pt_style": "class:accent",
                            }
                        ]
                        for idx, entry in enumerate(task_history):
                            preview = entry["task"].strip().replace("\n", " ")[:40]
                            icon = "▸" if entry["collapsed"] else "▾"
                            tok_badge = f"↑{format_tokens(entry['tokens_up'])} ↓{format_tokens(entry['tokens_down'])}"
                            opts.append({
                                "key": f"task:{idx}",
                                "label": f"{icon} {preview:<42} [{tok_badge}]",
                                "pt_style": "class:item" if entry["collapsed"] else "class:allow",
                            })
                        opts.append({"key": "__done__", "label": "← Done", "pt_style": "class:item"})
                        picked = await _prompt_arrow_menu("Tasks (select to collapse/expand):", opts)
                        if not picked or picked == "__done__":
                            break
                        if picked == "__toggle_all__":
                            tasks_collapsed = not tasks_collapsed
                            for entry in task_history:
                                entry["collapsed"] = tasks_collapsed
                        elif picked.startswith("task:"):
                            idx = int(picked.split(":", 1)[1])
                            task_history[idx]["collapsed"] = not task_history[idx]["collapsed"]
                            console.print(render_task_with_side_tokens(
                                task=task_history[idx]["task"],
                                result=task_history[idx]["result"],
                                tokens_up=task_history[idx]["tokens_up"],
                                tokens_down=task_history[idx]["tokens_down"],
                                approx_context=task_history[idx]["approx_context"],
                                collapsed=task_history[idx]["collapsed"],
                            ))
                    continue

                # Handle other slash commands
                if user_input.strip().startswith("/"):
                    should_continue = await _handle_slash_command(user_input.strip(), controller)
                    if should_continue is False:
                        break
                    continue

                # If the agent is currently working on a task, queue the message!
                is_working = controller.running or (background_worker_task and not background_worker_task.done())
                if is_working:
                    console.print(render_user_message(user_input))
                    q_len = controller.queue_message(user_input)
                    console.print(render_system_message(
                        f"[bold cyan]📥 Message queued for next run[/bold cyan] [dim](Position #{q_len} in queue). "
                        f"It will execute automatically when the current task finishes.[/dim]"
                    ))
                    continue

                # Agent is idle: display user message
                console.print(render_user_message(user_input))

                # Handle conversational messages directly
                if controller._is_conversational(user_input):
                    responses = {
                        "hi": "Hello! How can I help you today?",
                        "hello": "Hello! What would you like to work on?",
                        "hey": "Hey! What can I do for you?",
                        "thanks": "You're welcome!",
                        "thank you": "You're welcome!",
                        "thx": "You're welcome!",
                        "ty": "You're welcome!",
                        "bye": "Goodbye! Have a great day!",
                        "goodbye": "Goodbye! Take care!",
                        "see you": "See you later!",
                        "cya": "See you!",
                        "how are you": "I'm doing well, thanks for asking! How can I help?",
                        "how's it going": "Going great! What would you like to work on?",
                        "what's up": "I'm ready to help! What's on your mind?",
                        "ok": "Great! What would you like to do?",
                        "okay": "Great! What would you like to do?",
                        "sure": "Sounds good! What task would you like me to tackle?",
                        "yeah": "Great! What would you like to work on?",
                        "yep": "Great! What would you like to work on?",
                        "yup": "Great! What would you like to work on?",
                        "no": "No problem! Let me know if you need anything.",
                        "nope": "No problem! Let me know if you need anything.",
                        "nah": "No problem! Let me know if you need anything.",
                    }
                    msg_lower = user_input.strip().lower()
                    response = responses.get(msg_lower, "Hello! How can I help you with your code?")
                    controller.context.add_user_message(user_input)
                    controller.context.add_assistant_message(response)
                    controller.session_manager.log_message("USER", user_input)
                    controller.session_manager.log_message("AGENT", response)
                    controller.save_session()
                    console.print(render_agent_message(response))
                    continue

                # Launch task in background and KEEP THE INPUT PROMPT VISIBLE!
                background_worker_task = asyncio.create_task(_run_task_and_drain_queue(user_input))

            except (typer.Exit, SystemExit):
                break
            except KeyboardInterrupt:
                console.print()
                is_working = controller.running or (background_worker_task and not background_worker_task.done())
                if is_working:
                    thinking.stop()
                    controller.cancel()
                    if background_worker_task:
                        background_worker_task.cancel()
                    if controller.active_approval and not controller.active_approval.done():
                        controller.active_approval.cancel()
                    controller.cancelled = False
                    controller.session_manager.log_error("USER_INTERRUPT", "Task cancelled by Ctrl+C")
                    controller.save_session()
                    console.print(render_system_message("[yellow]⚠ Task interrupted — returning to prompt.[/yellow]"))
                    if controller.has_queued_messages():
                        console.print(render_system_message(
                            f"[dim]Note: {len(controller.message_queue)} queued message(s) preserved. Type '/queue' to view or '/queue clear' to remove.[/dim]"
                        ))
                else:
                    console.print(render_system_message("[dim]Type 'exit' or '/exit' to leave.[/dim]"))
            except EOFError:
                break
            except Exception as e:
                console.print()
                console.print(render_system_message(f"[red]Error: {e}[/red]"))
                controller.session_manager.log_error("CHAT_ERROR", str(e))

    # Run the async chat loop
    await _chat_loop()


# Available slash commands for auto-completion
SLASH_COMMANDS = {
    "/btw": "Ask a direct question to the model & enlighten the agent without disrupting running tasks",
    "/queue": "View, add, or clear queued tasks for upcoming runs",
    "/logs": "View or follow live logs of an ongoing process (/logs [id|follow])",
    "/ps": "List or stop ongoing background processes (/ps [stop <id>])",
    "/tokens": "View session tokens sent/received and approximate context length",
    "/compact": "Compact older conversation history to reduce prompt tokens",
    "/collapse": "Collapse or expand task outputs (with tokens shown at the side)",
    "/tasks": "Browse tasks interactively and toggle collapse/expand per task",
    "/help": "Display all available commands",
    "/provider": "Configure AI providers (Google, NVIDIA, OpenAI, etc.)",
    "/settings": "Open interactive settings & API profiles",
    "/model": "View or switch AI models",
    "/models": "List available models",
    "/status": "Show current agent status",
    "/config": "View, set or edit configuration (e.g. /config set max_iterations 1000)",
    "/history": "View conversation history and past messages",
    "/continue": "Continue previous conversation session",
    "/errors": "View recent errors from debug log",
    "/clear": "Clear terminal conversation",
    "/reset": "Reset current session",
    "/version": "Show agent version",
    "/update": "Check for and install updates",
    "/about": "Information about the agent",
    "/exit": "Exit the agent",
}

# Command icons for auto-completion display
SLASH_COMMAND_ICONS = {
    "/btw": "💡",
    "/queue": "📥",
    "/logs": "📜",
    "/ps": "⚙",
    "/tokens": "📊",
    "/compact": "🗜",
    "/collapse": "▸",
    "/tasks": "☰",
    "/help": "✦",
    "/provider": "⚡",
    "/settings": "⚙",
    "/model": "◈",
    "/models": "≡",
    "/status": "●",
    "/config": "⚙",
    "/history": "◷",
    "/continue": "▶",
    "/errors": "⚠",
    "/clear": "✕",
    "/reset": "↺",
    "/version": "◆",
    "/update": "▲",
    "/about": "ℹ",
    "/exit": "⏻",
}

# Extract just command names for completion
SLASH_COMMAND_NAMES = list(SLASH_COMMANDS.keys())

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.styles import Style

# Modern styled theme for slash command auto-completion popup
SLASH_COMPLETION_STYLE = Style.from_dict({
    # Completion menu popup container
    "completion-menu": "bg:#16161e #c0caf5",
    "completion-menu.completion": "bg:#1e2030 #7aa2f7 bold",
    "completion-menu.completion.current": "bg:#283457 #7dcfff bold",
    "completion-menu.meta.completion": "bg:#16161e #787c99",
    "completion-menu.meta.completion.current": "bg:#283457 #e0af68 bold",
    "completion-menu.multi-column-meta": "bg:#16161e #787c99",
    # Scrollbar
    "scrollbar.background": "bg:#16161e",
    "scrollbar.button": "bg:#3b4261",
})


def _resolve_first_slash_command(prefix: str) -> Optional[str]:
    """Return the first matching slash command for a partial prefix, or None if exact/alias."""
    low = prefix.strip().lower()
    if not low.startswith("/"):
        return None
    if low in SLASH_COMMANDS or low in ("/q", "/quit", "/log", "/processes", "/cancel"):
        return None
    for cmd in SLASH_COMMANDS:
        if cmd.startswith(low):
            return cmd
    return None


class SlashCommandCompleter(Completer):
    """Auto-completion for slash commands with icons and stylish formatting."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd, desc in SLASH_COMMANDS.items():
                if cmd.startswith(text):
                    icon = SLASH_COMMAND_ICONS.get(cmd, "✦")
                    yield Completion(
                        cmd,
                        start_position=-len(text),
                        display=f"{icon} {cmd:<11}",
                        display_meta=f"  {desc}",
                    )


_current_controller: Optional[AgentController] = None


def _get_dynamic_prompt():
    """Show clean prompt when idle, [working] when busy, and 'You: ' dynamically once the user begins typing."""
    from prompt_toolkit.application.current import get_app
    try:
        app = get_app()
        if app and app.current_buffer and app.current_buffer.text:
            return HTML("<ansicyan><b>You: </b></ansicyan>")
    except Exception:
        pass
    if _current_controller:
        if _current_controller.active_approval and not _current_controller.active_approval.done():
            return HTML("<ansiyellow><b>Approve? [y/n/a/d] (or /btw) ❯ </b></ansiyellow>")
        if _current_controller.running:
            return HTML("<ansiyellow><b>[working] ❯ </b></ansiyellow>")
    return HTML("<ansicyan><b>❯ </b></ansicyan>")


def setup_readline_completion():
    """Set up readline for tab completion of slash commands."""
    # Only works on Unix-like systems and Windows with pyreadline3
    try:
        import readline
        readline.parse_and_bind("tab: complete")
        # Create a completer function
        def completer(text, state):
            if text.startswith("/"):
                matches = [cmd for cmd in SLASH_COMMAND_NAMES if cmd.startswith(text)]
            else:
                matches = []
            if state < len(matches):
                return matches[state]
            return None
        readline.set_completer(completer)
        # Also bind '?' to show completions
        readline.parse_and_bind('"?": complete')
    except ImportError:
        pass  # readline not available


def _get_input_with_readline() -> str:
    """Get user input with readline tab completion support (fallback)."""
    try:
        import readline
        # Set up completion each time (in case it wasn't set up)
        readline.parse_and_bind("tab: complete")
        def completer(text, state):
            if text.startswith("/"):
                matches = [cmd for cmd in SLASH_COMMAND_NAMES if cmd.startswith(text)]
            else:
                matches = []
            if state < len(matches):
                return matches[state]
            return None
        readline.set_completer(completer)
        readline.parse_and_bind('"?": complete')
    except ImportError:
        pass
    if _current_controller:
        if _current_controller.active_approval and not _current_controller.active_approval.done():
            return input("\x1b[1;33mApprove? [y/n/a/d] (or /btw) ❯\x1b[0m ")
        if _current_controller.running:
            return input("\x1b[1;33m[working] ❯\x1b[0m ")
    return input("\x1b[1;36m❯\x1b[0m ")


async def _handle_slash_command(command: str, controller: AgentController) -> None:
    """Handle slash commands in chat."""
    from agent.config import get_config, update_config, get_profiles, save_profile, delete_profile, load_profile
    from agent.model import get_model_client, reset_model_client
    from agent.controller import AgentController

    parts = command.split()
    cmd = parts[0].lower()
    resolved = _resolve_first_slash_command(cmd)
    if resolved:
        cmd = resolved

    if cmd == "/btw":
        question = command[4:].strip() if len(command) > 4 else ""
        if not question:
            console.print("[yellow]Usage: /btw <your question or side note>[/yellow]")
            console.print("[dim]Example: /btw what database are we using?[/dim]")
            console.print("[dim]Example: /btw make sure to write clean docstrings[/dim]")
            return True
        await _handle_btw_command(question, controller)
        return True

    elif cmd == "/queue":
        await _handle_queue_command(parts, controller)
        return True

    elif cmd == "/help":
        selected_cmd = await _show_help_interactive()
        if selected_cmd and selected_cmd != "/help":
            return await _handle_slash_command(selected_cmd, controller)
        return True

    elif cmd == "/model":
        config = get_config()
        await _show_models_command(config, controller)
        return True

    elif cmd == "/models":
        config = get_config()
        await _show_models_list(config)
        return True

    elif cmd in ("/settings", "/provider"):
        await _handle_provider_command(parts, controller)
        return True

    elif cmd == "/status":
        _show_status_command(controller)
        return True

    elif cmd == "/tokens":
        _show_tokens_command(controller)
        return True

    elif cmd == "/compact":
        before, after = controller.context.compact_history()
        controller.save_session()
        console.print(render_system_message(
            f"[green]✓ Compacted context:[/green] [yellow]~{format_tokens(before)}[/yellow] → [cyan]~{format_tokens(after)} tokens[/cyan]"
        ))
        return True

    elif cmd == "/config":
        await _handle_config_command(parts, controller)
        return True

    elif cmd == "/history":
        _show_history_command(controller, parts)
        return True

    elif cmd == "/continue":
        if controller.has_saved_session():
            controller.load_session()
            count = len([m for m in controller.context.messages if m.role == "user"])
            console.print(render_system_message(f"[green]Continued previous session ({count} user interactions restored).[/green]"))
        else:
            console.print(render_system_message("[yellow]No saved session found to continue.[/yellow]"))
        return True

    elif cmd in ("/exit", "/quit", "/q"):
        console.print(render_system_message("[yellow]Goodbye from ireri![/yellow]"))
        return False

    elif cmd == "/clear":
        console.clear()
        return True

    elif cmd == "/errors":
        _show_errors_command(controller, parts)
        return True

    elif cmd in ("/logs", "/log"):
        await _handle_logs_command(parts)
        return True

    elif cmd in ("/ps", "/processes"):
        await _handle_ps_command(parts)
        return True

    elif cmd == "/reset":
        if sys.stdin.isatty() and len(parts) == 1:
            confirm = await _prompt_arrow_menu(
                title="Reset current conversation session?",
                options=[
                    {"key": "yes", "label": "✓ Yes, reset session", "pt_style": "class:deny"},
                    {"key": "no", "label": "✗ Cancel", "pt_style": "class:item"},
                ],
            )
            if confirm != "yes":
                console.print("[dim]Session reset cancelled.[/dim]")
                return True
        controller.reset()
        console.print(render_system_message("[green]Session reset.[/green]"))
        return True

    elif cmd == "/version":
        _show_version_command()
        return True

    elif cmd == "/update":
        _show_update_command()
        return True

    elif cmd == "/about":
        _show_about_command()
        return True

    else:
        console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]")
        return True


def _show_history_command(controller: AgentController, parts: list) -> None:
    """Handle /history command - display recent conversation history."""
    limit = 10
    if len(parts) > 1:
        try:
            limit = int(parts[1])
        except ValueError:
            pass

    history = controller.session_manager.get_recent_history(limit=limit)
    if not history:
        console.print(render_system_message("[dim]No previous conversation history found.[/dim]"))
        return

    table = Table(title=f"Recent Conversation History (Last {len(history)} messages)", box=ROUNDED, border_style="bright_blue")
    table.add_column("#", style="dim", width=4)
    table.add_column("Speaker", style="cyan", width=12)
    table.add_column("Message Preview", style="white")

    for idx, msg in enumerate(history, 1):
        role = "[bold cyan]You[/bold cyan]" if msg.get("role") == "user" else "[bold green]ireri[/bold green]"
        content = msg.get("content", "").strip().replace("\n", " ")
        if len(content) > 90:
            content = content[:87] + "..."
        table.add_row(str(idx), role, content)

    console.print()
    console.print(table)
    console.print(render_system_message(
        f"[dim]Full transcript logged at: [underline]{controller.session_manager.history_file}[/underline][/dim]"
    ))
    console.print()


def _show_errors_command(controller: AgentController, parts: list) -> None:
    """Handle /errors command - display recent errors from .ireri/errors.log."""
    limit = 10
    if len(parts) > 1:
        try:
            limit = int(parts[1])
        except ValueError:
            pass

    errors = controller.session_manager.get_recent_errors(limit=limit)
    if not errors:
        console.print(render_system_message(
            f"[dim]No errors encountered yet in: [underline]{controller.session_manager.error_log_file}[/underline][/dim]"
        ))
        return

    table = Table(title=f"Recent Errors (Last {len(errors)} entries)", box=ROUNDED, border_style="red")
    table.add_column("Timestamp", style="dim", width=19)
    table.add_column("Type", style="bold red", width=16)
    table.add_column("Message", style="white")

    for err in errors:
        msg = err.get("message", "").strip().replace("\n", " ")
        if len(msg) > 80:
            msg = msg[:77] + "..."
        table.add_row(err.get("timestamp", ""), err.get("type", "ERROR"), msg)

    console.print()
    console.print(table)
    console.print(render_system_message(
        f"[dim]Full error log at: [underline]{controller.session_manager.error_log_file}[/underline][/dim]"
    ))
    console.print()


async def _handle_ps_command(parts: list) -> None:
    """Handle /ps command - list or stop ongoing/recent processes."""
    from agent.tools.terminal import get_process_manager

    pm = get_process_manager()
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub == "stop":
        target = parts[2] if len(parts) > 2 else ""
        if not target:
            console.print("[yellow]Usage: /ps stop <process_id|all>[/yellow]")
            return
        if target.lower() == "all":
            count = pm.stop_all()
            console.print(render_system_message(f"[yellow]Stopped {count} running process(es).[/yellow]"))
            return
        ok, msg = await pm.stop_process(target)
        color = "green" if ok else "red"
        console.print(render_system_message(f"[{color}]{msg}[/{color}]"))
        return

    procs = pm.list_processes()
    if not procs:
        console.print(render_system_message("[dim]No processes have been started in this session yet.[/dim]"))
        return

    table = Table(title="Managed Processes", box=ROUNDED, border_style="bright_blue")
    table.add_column("ID", style="cyan", width=10)
    table.add_column("PID", style="dim", width=8)
    table.add_column("Status", width=14)
    table.add_column("Mode", style="dim", width=12)
    table.add_column("Uptime", style="dim", width=9)
    table.add_column("Command", style="white")

    for p in procs:
        status_str = "[bold green]RUNNING[/bold green]" if p.is_running else f"[dim]EXIT ({p.returncode})[/dim]"
        mode_str = "background" if p.background else "foreground"
        uptime_str = f"{p.uptime_seconds:.1f}s"
        cmd_str = p.command if len(p.command) <= 60 else p.command[:57] + "..."
        table.add_row(p.process_id, str(p.pid or "-"), status_str, mode_str, uptime_str, cmd_str)

    console.print()
    console.print(table)

    if sys.stdin.isatty():
        while True:
            procs = pm.list_processes()
            if not procs:
                break
            menu_opts = []
            for p in procs:
                st = "RUNNING" if p.is_running else f"EXIT({p.returncode})"
                menu_opts.append({
                    "key": f"logs:{p.process_id}",
                    "label": f"📜 View logs: {p.process_id} [{st}] {p.command[:40]}",
                    "pt_style": "class:accent",
                })
                if p.is_running:
                    menu_opts.append({
                        "key": f"follow:{p.process_id}",
                        "label": f"📡 Follow live: {p.process_id}",
                        "pt_style": "class:allow",
                    })
                    menu_opts.append({
                        "key": f"stop:{p.process_id}",
                        "label": f"⏹ Stop process: {p.process_id}",
                        "pt_style": "class:deny",
                    })
            if any(p.is_running for p in procs):
                menu_opts.append({"key": "stop_all", "label": "⏹ Stop all running processes", "pt_style": "class:deny"})
            menu_opts.append({"key": "done", "label": "← Done", "pt_style": "class:item"})

            action = await _prompt_arrow_menu("Process Actions:", menu_opts)
            if not action or action == "done":
                break
            if action == "stop_all":
                count = pm.stop_all()
                console.print(render_system_message(f"[yellow]Stopped {count} running process(es).[/yellow]"))
            elif action.startswith("logs:"):
                await _handle_logs_command(["/logs", action.split(":", 1)[1]])
            elif action.startswith("follow:"):
                await _handle_logs_command(["/logs", "follow", action.split(":", 1)[1]])
            elif action.startswith("stop:"):
                pid_target = action.split(":", 1)[1]
                ok, msg = await pm.stop_process(pid_target)
                console.print(render_system_message(f"[{'green' if ok else 'red'}]{msg}[/{'green' if ok else 'red'}]"))
        return

    console.print(render_system_message(
        "[dim]Commands: [cyan]/logs <id>[/cyan] (view logs)  •  [cyan]/logs follow <id>[/cyan] (stream live)  •  [cyan]/ps stop <id>[/cyan] (stop)[/dim]"
    ))
    console.print()


async def _handle_logs_command(parts: list) -> None:
    """Handle /logs command - view or follow live logs of an ongoing or recent process."""
    from rich.markup import escape
    from agent.tools.terminal import get_process_manager

    pm = get_process_manager()
    args = parts[1:]

    follow = False
    proc_id: Optional[str] = None
    tail = 80

    for arg in args:
        low = arg.lower()
        if low in ("follow", "-f", "--follow", "live"):
            follow = True
        elif arg.isdigit():
            tail = int(arg)
        elif not proc_id:
            proc_id = arg

    if not args and sys.stdin.isatty():
        procs = pm.list_processes()
        if len(procs) > 1:
            proc_opts = [
                {
                    "key": p.process_id,
                    "label": f"{p.process_id} [{'RUNNING' if p.is_running else f'EXIT({p.returncode})'}] {p.command[:48]}",
                    "pt_style": "class:allow" if p.is_running else "class:item",
                }
                for p in procs
            ]
            proc_opts.append({"key": "__cancel__", "label": "← Cancel", "pt_style": "class:item"})
            picked = await _prompt_arrow_menu("Select process to view logs:", proc_opts)
            if not picked or picked == "__cancel__":
                return
            proc_id = picked
        if procs:
            picked_proc = pm.get_process(proc_id)
            if picked_proc and picked_proc.is_running:
                mode = await _prompt_arrow_menu(
                    f"Logs for {picked_proc.process_id}:",
                    [
                        {"key": "recent", "label": "📜 View recent logs (last 80 lines)", "pt_style": "class:accent"},
                        {"key": "follow", "label": "📡 Follow live logs (stream)", "pt_style": "class:allow"},
                        {"key": "cancel", "label": "← Cancel", "pt_style": "class:item"},
                    ],
                )
                if not mode or mode == "cancel":
                    return
                follow = (mode == "follow")

    proc = pm.get_process(proc_id)
    if not proc:
        msg = f"Process '{proc_id}' not found." if proc_id else "No ongoing or recent processes found."
        console.print(render_system_message(f"[yellow]{msg}[/yellow] [dim]Run /ps to list processes.[/dim]"))
        return

    status_label = "[bold green]RUNNING[/bold green]" if proc.is_running else f"[dim]EXITED ({proc.returncode})[/dim]"
    header = (
        f"[bold cyan]{proc.process_id}[/bold cyan] (PID {proc.pid})  •  {status_label}  •  "
        f"[dim]Uptime: {proc.uptime_seconds:.1f}s[/dim]\n"
        f"[dim]Command: {escape(proc.command)}[/dim]"
    )

    if not follow:
        logs = proc.get_recent_logs(tail=tail)
        body = escape(logs) if logs else "[dim](No output captured yet)[/dim]"
        console.print()
        console.print(Panel(
            f"{header}\n\n{body}",
            title=f"[bold bright_blue]Process Logs (last {tail} lines)[/bold bright_blue]",
            border_style="bright_blue",
            box=ROUNDED,
            padding=(0, 1),
        ))
        if proc.is_running:
            console.print(render_system_message(
                f"[dim]Process is still running. Use [cyan]/logs follow {proc.process_id}[/cyan] to stream live logs or [cyan]/ps stop {proc.process_id}[/cyan] to stop.[/dim]"
            ))
        console.print()
        return

    # Live follow mode
    console.print()
    console.print(Panel(
        f"{header}\n[dim]Streaming live logs... Press Ctrl+C to stop watching (the process will keep running).[/dim]",
        title="[bold bright_cyan]Live Process Logs[/bold bright_cyan]",
        border_style="bright_cyan",
        box=ROUNDED,
    ))

    # Print existing recent lines first
    all_lines = list(proc.combined_logs)
    start_idx = max(0, len(all_lines) - min(tail, 30))
    for line in all_lines[start_idx:]:
        console.print(f"  {escape(line)}")
    seen_total = len(all_lines)

    try:
        while True:
            cur_lines = list(proc.combined_logs)
            if len(cur_lines) > seen_total:
                for line in cur_lines[seen_total:]:
                    console.print(f"  {escape(line)}")
                seen_total = len(cur_lines)
            elif len(cur_lines) < seen_total:
                # Deque wrapped around
                seen_total = len(cur_lines)

            if not proc.is_running:
                # Flush any final lines
                cur_lines = list(proc.combined_logs)
                if len(cur_lines) > seen_total:
                    for line in cur_lines[seen_total:]:
                        console.print(f"  {escape(line)}")
                console.print(render_system_message(f"[dim]Process {proc.process_id} exited with code {proc.returncode}.[/dim]"))
                break
            await asyncio.sleep(0.25)
    except (asyncio.CancelledError, KeyboardInterrupt):
        console.print(render_system_message(f"[yellow]Stopped watching logs for {proc.process_id} (process still {'running' if proc.is_running else 'exited'}).[/yellow]"))


async def _handle_btw_command(question: str, controller: AgentController) -> None:
    """Execute direct side-question to model and display enlightened response."""
    console.print()
    console.print(Panel(
        f"[bold bright_yellow]Q:[/bold bright_yellow] {question}\n\n[dim]Consulting model directly...[/dim]",
        title="[bold yellow]💡 By the Way (Direct Side-Chat)[/bold yellow]",
        border_style="yellow",
        box=ROUNDED,
        padding=(0, 1),
    ))

    answer = await controller.ask_direct(question)

    console.print()
    console.print(Panel(
        Markdown(answer),
        title="[bold bright_green]ireri (Direct Answer)[/bold bright_green]",
        subtitle="[dim]✓ Injected into agent context to guide ongoing & subsequent steps[/dim]",
        border_style="bright_yellow",
        box=ROUNDED,
        padding=(0, 1),
    ))
    console.print()


async def _handle_queue_command(parts: list, controller: AgentController) -> None:
    """Handle /queue slash command to view, add, or clear queued messages."""
    sub = parts[1].lower() if len(parts) > 1 else ""

    if sub in ("clear", "reset"):
        count = controller.clear_message_queue()
        console.print(render_system_message(f"[green]Cleared {count} queued message(s).[/green]"))
        return

    if sub == "add" and len(parts) > 2:
        msg = " ".join(parts[2:]).strip()
        q_len = controller.queue_message(msg)
        console.print(render_system_message(f"[green]✓ Queued task #{q_len}:[/green] [white]{msg}[/white]"))
        return

    # If sub is not a reserved keyword but user typed: /queue do something
    if sub and sub not in ("list", "show") and len(parts) >= 2:
        msg = " ".join(parts[1:]).strip()
        q_len = controller.queue_message(msg)
        console.print(render_system_message(f"[green]✓ Queued task #{q_len}:[/green] [white]{msg}[/white]"))
        return

    queued = controller.get_queued_messages()
    if not queued:
        console.print(render_system_message(
            "[dim]Message queue is currently empty.\n"
            "Queue upcoming tasks with [bold cyan]/queue <message>[/bold cyan] or while running.[/dim]"
        ))
    else:
        table = Table(title=f"Queued Messages ({len(queued)} pending)", box=ROUNDED, border_style="cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Queued Task", style="white")

        for idx, msg in enumerate(queued, 1):
            preview = msg.replace("\n", " ")
            if len(preview) > 85:
                preview = preview[:82] + "..."
            table.add_row(str(idx), preview)

        console.print()
        console.print(table)

    if sys.stdin.isatty() and not sub:
        opts = [{"key": "add", "label": "+ Add a task to queue", "pt_style": "class:allow"}]
        if queued:
            opts.append({"key": "clear", "label": "✕ Clear all queued tasks", "pt_style": "class:deny"})
        opts.append({"key": "done", "label": "← Done", "pt_style": "class:item"})
        action = await _prompt_arrow_menu("Queue Actions:", opts)
        if action == "add":
            loop = asyncio.get_event_loop()
            new_msg = (await loop.run_in_executor(
                None, lambda: Prompt.ask("[bold cyan]Task to queue[/bold cyan]", default="")
            )).strip()
            if new_msg:
                q_len = controller.queue_message(new_msg)
                console.print(render_system_message(f"[green]✓ Queued task #{q_len}:[/green] [white]{new_msg}[/white]"))
        elif action == "clear":
            count = controller.clear_message_queue()
            console.print(render_system_message(f"[green]Cleared {count} queued message(s).[/green]"))
    console.print()


# Configuration key mappings and aliases
CONFIG_KEY_MAP = {
    "max_iterations": "max_iterations",
    "iterations": "max_iterations",
    "iter": "max_iterations",
    "max-iterations": "max_iterations",
    "max_iter": "max_iterations",
    "max_tool_calls": "max_tool_calls",
    "tool_calls": "max_tool_calls",
    "tools": "max_tool_calls",
    "max-tool-calls": "max_tool_calls",
    "max_tools": "max_tool_calls",
    "command_timeout": "command_timeout",
    "timeout": "command_timeout",
    "cmd_timeout": "command_timeout",
    "command-timeout": "command_timeout",
    "model": "openai_model",
    "openai_model": "openai_model",
    "base_url": "openai_base_url",
    "url": "openai_base_url",
    "openai_base_url": "openai_base_url",
    "api_key": "openai_api_key",
    "key": "openai_api_key",
    "openai_api_key": "openai_api_key",
    "log_level": "log_level",
    "log": "log_level",
    "level": "log_level",
}


def set_config_value(key: str, value: Any, controller: Optional[AgentController] = None) -> tuple[bool, str]:
    """Set and persist a configuration value immediately to global config and active controller."""
    from agent.config import get_config, update_config
    from agent.model import reset_model_client, get_model_client

    key_normalized = key.lower().replace("-", "_")
    target_field = CONFIG_KEY_MAP.get(key_normalized)
    if not target_field:
        valid_keys = ", ".join(["max_iterations", "max_tool_calls", "command_timeout", "model", "base_url", "api_key", "log_level"])
        return False, f"Unknown setting '{key}'. Valid settings: {valid_keys}"

    # Type validation & conversion
    if target_field in ("max_iterations", "max_tool_calls", "command_timeout"):
        try:
            val = int(value)
            if val <= 0:
                return False, f"{target_field} must be a positive integer"
        except (ValueError, TypeError):
            return False, f"Invalid integer value for {target_field}: {value}"
    elif target_field == "log_level":
        val = str(value).upper()
        if val not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            return False, f"Invalid log level: {value}. Choose from: DEBUG, INFO, WARNING, ERROR, CRITICAL"
    else:
        val = str(value).strip()
        if not val:
            return False, f"{target_field} cannot be empty"

    update_config(**{target_field: val})

    # Update active controller and model client if in active chat session
    if controller is not None:
        controller.config = get_config()
        if target_field in ("openai_model", "openai_base_url", "openai_api_key"):
            reset_model_client()
            controller.model_client = get_model_client()

    return True, f"Updated [bold cyan]{target_field}[/bold cyan] to [bold green]{val}[/bold green]"


def _show_config_table(config: AgentConfig) -> None:
    """Render the configuration table with editing instructions."""
    table = Table(title="ireri Configuration", box=ROUNDED, border_style="bright_blue")
    table.add_column("Setting", style="cyan", width=18)
    table.add_column("Value", style="white")
    table.add_column("Description", style="dim")

    table.add_row("Model", config.openai_model, "Active LLM model name")
    table.add_row("Base URL", config.openai_base_url, "API endpoint base URL")
    table.add_row("API Key", ("***" + config.openai_api_key[-4:]) if config.openai_api_key else "Not set", "Provider API secret key")
    table.add_row("Workspace", str(config.workspace_root), "Current working workspace directory")
    table.add_row("Max Iterations", str(config.max_iterations), "Max reasoning/tool iterations per task")
    table.add_row("Max Tool Calls", str(config.max_tool_calls), "Max total tool executions per task")
    table.add_row("Command Timeout", f"{config.command_timeout}s", "Timeout for shell commands in seconds")
    table.add_row("Log Level", config.log_level, "Logging verbosity (DEBUG, INFO, WARNING, ERROR)")

    console.print()
    console.print(table)
    console.print()
    help_panel = (
        "[bold cyan]How to change settings:[/bold cyan]\n"
        "  • [yellow]/config set <setting> <value>[/yellow]  (e.g., [green]/config set max_iterations 1000[/green])\n"
        "  • [yellow]/config <setting> <value>[/yellow]      (shorthand, e.g. [green]/config max_tool_calls 1000[/green])\n"
        "  • [yellow]/config edit[/yellow]                   (interactive configuration wizard)\n"
        "  • [yellow]/config get <setting>[/yellow]          (view specific value)\n"
        "  • [yellow]/settings set <setting> <val>[/yellow]  (works in /settings too!)"
    )
    console.print(Panel(help_panel, title="[bold]Configuration Controls[/bold]", border_style="dim", box=ROUNDED))
    console.print()


async def _interactive_tool_access_edit(controller: Optional[AgentController] = None) -> None:
    """Interactive arrow-key editor for per-tool access policies (allow / ask / deny)."""
    from agent.config import _save_global_config

    config = controller.config if controller else get_config()
    registry = get_registry()
    tools = sorted(registry.get_all(), key=lambda t: t.name)

    if not tools:
        console.print("[yellow]No tools registered.[/yellow]")
        return

    policy_order = ["allow", "ask", "deny"]

    def _cycle_policy(tool, direction: int = 1):
        default_p = "ask" if tool.requires_approval else "allow"
        cur = config.get_tool_access(tool.name, default=default_p)
        idx = policy_order.index(cur) if cur in policy_order else 1
        nxt = policy_order[(idx + direction) % len(policy_order)]
        config.set_tool_access(tool.name, nxt)
        _save_global_config(config)

    # Interactive prompt_toolkit mode when TTY is available
    if sys.stdin.isatty():
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import Layout, HSplit, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style

            selected_idx = 0
            total_items = len(tools) + 2  # tools + Reset all + Done
            kb = KeyBindings()

            @kb.add("up")
            @kb.add("k")
            def _up(event):
                nonlocal selected_idx
                selected_idx = (selected_idx - 1) % total_items

            @kb.add("down")
            @kb.add("j")
            def _down(event):
                nonlocal selected_idx
                selected_idx = (selected_idx + 1) % total_items

            @kb.add("right")
            @kb.add("l")
            @kb.add(" ")
            def _cycle_next(event):
                if selected_idx < len(tools):
                    _cycle_policy(tools[selected_idx], 1)

            @kb.add("left")
            @kb.add("h")
            def _cycle_prev(event):
                if selected_idx < len(tools):
                    _cycle_policy(tools[selected_idx], -1)

            @kb.add("enter")
            def _enter(event):
                if selected_idx < len(tools):
                    _cycle_policy(tools[selected_idx], 1)
                elif selected_idx == len(tools):
                    config.reset_tool_access()
                    _save_global_config(config)
                else:
                    event.app.exit(result=True)

            @kb.add("q")
            @kb.add("escape")
            @kb.add("c-c")
            def _exit(event):
                event.app.exit(result=True)

            def _render():
                lines = [
                    ("class:title", "  Tool Access Settings (/settings tools)\n"),
                    ("class:subtitle", "  ↑/↓: navigate  •  ←/→/Space/Enter: change [ALLOW / ASK / DENY]  •  q/Esc: save & exit\n\n"),
                ]
                for idx, tool in enumerate(tools):
                    is_sel = idx == selected_idx
                    default_p = "ask" if tool.requires_approval else "allow"
                    policy = config.get_tool_access(tool.name, default=default_p)
                    danger = " !DANGEROUS" if tool.dangerous else ""
                    pointer = "  ❯ " if is_sel else "    "
                    badge = f"[{policy.upper():5}]"
                    row_text = f"{pointer}{tool.name:<18} {badge}  {tool.description[:42]}{danger}\n"
                    if is_sel:
                        lines.append(("class:selected", row_text))
                    elif policy == "allow":
                        lines.append(("class:allow", row_text))
                    elif policy == "deny":
                        lines.append(("class:deny", row_text))
                    else:
                        lines.append(("class:ask", row_text))

                lines.append(("class:subtitle", "\n"))
                reset_sel = selected_idx == len(tools)
                done_sel = selected_idx == len(tools) + 1
                lines.append((
                    "class:selected" if reset_sel else "class:subtitle",
                    f"{'  ❯ ' if reset_sel else '    '}↺ Reset all tools to defaults\n",
                ))
                lines.append((
                    "class:selected" if done_sel else "class:allow",
                    f"{'  ❯ ' if done_sel else '    '}✓ Save & Done\n",
                ))
                return lines

            style = Style.from_dict({
                "title": "bold ansibrightcyan",
                "subtitle": "ansibrightblack",
                "selected": "bold ansibrightyellow reverse",
                "allow": "ansigreen",
                "ask": "ansiyellow",
                "deny": "ansired",
            })

            app = Application(
                layout=Layout(HSplit([Window(content=FormattedTextControl(_render))])),
                key_bindings=kb,
                style=style,
                full_screen=False,
                erase_when_done=True,
            )
            await app.run_async()
            console.print(render_system_message("[green]✓ Tool access settings saved.[/green]"))
            return
        except Exception:
            pass

    # Fallback numbered prompt for non-TTY / automated environments
    loop = asyncio.get_event_loop()
    while True:
        console.print()
        console.print("[bold bright_cyan]Tool Access Settings[/bold bright_cyan]")
        for i, tool in enumerate(tools, 1):
            default_p = "ask" if tool.requires_approval else "allow"
            policy = config.get_tool_access(tool.name, default=default_p)
            color = "green" if policy == "allow" else ("red" if policy == "deny" else "yellow")
            console.print(f"  [bold]{i}[/bold]) {tool.name:<18} [{color}]{policy.upper()}[/{color}] [dim]- {tool.description}[/dim]")
        console.print("  [bold]r[/bold]) Reset all to defaults")
        console.print("  [bold]0[/bold]) Done")

        valid = [str(i) for i in range(len(tools) + 1)] + ["r"]
        choice = await loop.run_in_executor(
            None,
            lambda: Prompt.ask("[bold cyan]Select tool to toggle[/bold cyan]", choices=valid, default="0")
        )
        if choice == "0":
            break
        elif choice == "r":
            config.reset_tool_access()
            _save_global_config(config)
            console.print(render_system_message("[green]Reset all tool access policies to defaults.[/green]"))
        else:
            tool = tools[int(choice) - 1]
            _cycle_policy(tool, 1)
            new_p = config.get_tool_access(tool.name)
            console.print(render_system_message(f"[green]{tool.name} access set to '{new_p}'.[/green]"))


async def _handle_tools_settings_command(args: list, controller: Optional[AgentController] = None) -> None:
    """Handle `/settings tools` subcommands."""
    from agent.config import _save_global_config

    config = controller.config if controller else get_config()
    registry = get_registry()
    valid_tools = {t.name: t for t in registry.get_all()}

    if not args:
        await _interactive_tool_access_edit(controller)
        return

    if args[0].lower() == "reset":
        if len(args) > 1 and args[1] in valid_tools:
            config.reset_tool_access(args[1])
            _save_global_config(config)
            console.print(render_system_message(f"[green]Reset '{args[1]}' access to default.[/green]"))
        else:
            config.reset_tool_access()
            _save_global_config(config)
            console.print(render_system_message("[green]Reset all tool access policies to defaults.[/green]"))
        return

    if len(args) < 2:
        console.print("[red]Usage: /settings tools <tool_name|all> <allow|ask|deny|reset>[/red]")
        return

    target_tool = args[0]
    policy = args[1].lower()

    if policy == "reset":
        if target_tool == "all":
            config.reset_tool_access()
            _save_global_config(config)
            console.print(render_system_message("[green]Reset all tool access policies to defaults.[/green]"))
        elif target_tool in valid_tools:
            config.reset_tool_access(target_tool)
            _save_global_config(config)
            console.print(render_system_message(f"[green]Reset '{target_tool}' access to default.[/green]"))
        else:
            console.print(render_system_message(f"[red]Unknown tool: {target_tool}[/red]"))
        return

    if policy not in ("allow", "ask", "deny"):
        console.print(render_system_message(f"[red]Invalid policy '{policy}'. Use: allow, ask, deny, or reset.[/red]"))
        return

    if target_tool == "all":
        for name in valid_tools:
            config.set_tool_access(name, policy)
        _save_global_config(config)
        console.print(render_system_message(f"[green]Set access for ALL tools to '{policy}'.[/green]"))
    elif target_tool in valid_tools:
        config.set_tool_access(target_tool, policy)
        _save_global_config(config)
        console.print(render_system_message(f"[green]Set '{target_tool}' access to '{policy}'.[/green]"))
    else:
        console.print(render_system_message(f"[red]Unknown tool: {target_tool}. Available: {', '.join(sorted(valid_tools))}[/red]"))


async def _interactive_config_edit(controller: Optional[AgentController] = None) -> None:
    """Interactive arrow-key wizard to change any configuration setting."""
    while True:
        config = get_config()
        key_preview = ("***" + config.openai_api_key[-4:]) if config.openai_api_key else "Not set"
        options = [
            {"key": "max_iterations", "label": f"Max Iterations     ({config.max_iterations})", "pt_style": "class:item"},
            {"key": "max_tool_calls", "label": f"Max Tool Calls     ({config.max_tool_calls})", "pt_style": "class:item"},
            {"key": "command_timeout", "label": f"Command Timeout    ({config.command_timeout}s)", "pt_style": "class:item"},
            {"key": "model", "label": f"Model              ({config.openai_model})", "pt_style": "class:accent"},
            {"key": "base_url", "label": f"Base URL           ({config.openai_base_url})", "pt_style": "class:accent"},
            {"key": "api_key", "label": f"API Key            ({key_preview})", "pt_style": "class:accent"},
            {"key": "log_level", "label": f"Log Level          ({config.log_level})", "pt_style": "class:item"},
            {"key": "tool_access", "label": f"Tool Access        ({len(config.tool_access)} custom rule(s))", "pt_style": "class:allow"},
            {"key": "done", "label": "← Back / Done", "pt_style": "class:item"},
        ]

        if sys.stdin.isatty():
            choice = await _prompt_arrow_menu("Interactive Configuration Editor", options)
        else:
            choice = None

        if not choice or choice == "done":
            return

        if choice == "tool_access":
            await _interactive_tool_access_edit(controller)
            continue

        defaults = {
            "max_iterations": str(config.max_iterations),
            "max_tool_calls": str(config.max_tool_calls),
            "command_timeout": str(config.command_timeout),
            "model": config.openai_model,
            "base_url": config.openai_base_url,
            "api_key": "",
            "log_level": config.log_level,
        }

        loop = asyncio.get_event_loop()
        new_val = await loop.run_in_executor(
            None,
            lambda: Prompt.ask(f"[bold cyan]Enter new value for {choice}[/bold cyan]", default=defaults[choice]),
        )

        if not new_val.strip():
            console.print("[yellow]No change made.[/yellow]")
            continue

        ok, msg = set_config_value(choice, new_val.strip(), controller)
        if ok:
            console.print(render_system_message(f"[green]{msg}[/green]"))
        else:
            console.print(render_system_message(f"[red]{msg}[/red]"))


async def _handle_config_command(parts: list, controller: Optional[AgentController] = None) -> None:
    """Handle /config slash command and its subcommands."""
    if len(parts) == 1:
        _show_config_table(get_config())
        if sys.stdin.isatty():
            action = await _prompt_arrow_menu(
                "Configuration Actions:",
                [
                    {"key": "edit", "label": "⚙ Edit a setting", "pt_style": "class:accent"},
                    {"key": "tools", "label": "🛡 Edit tool access", "pt_style": "class:allow"},
                    {"key": "reset", "label": "↺ Reset limits to defaults (1000 / 1000 / 60s)", "pt_style": "class:deny"},
                    {"key": "done", "label": "← Done", "pt_style": "class:item"},
                ],
            )
            if action == "edit":
                await _interactive_config_edit(controller)
            elif action == "tools":
                await _interactive_tool_access_edit(controller)
            elif action == "reset":
                set_config_value("max_iterations", 1000, controller)
                set_config_value("max_tool_calls", 1000, controller)
                set_config_value("command_timeout", 60, controller)
                console.print(render_system_message("[green]Reset iterations, tool calls, and timeout to defaults (1000 / 1000 / 60s).[/green]"))
        return

    sub = parts[1].lower()

    if sub in ("help", "-h", "--help"):
        _show_config_table(get_config())
        return

    elif sub in ("edit", "wizard", "interactive"):
        await _interactive_config_edit(controller)
        return

    elif sub == "set":
        if len(parts) < 4:
            console.print("[red]Usage: /config set <setting> <value>[/red]")
            console.print("[dim]Example: /config set max_iterations 1000[/dim]")
            return
        key = parts[2]
        val = " ".join(parts[3:])
        ok, msg = set_config_value(key, val, controller)
        if ok:
            console.print(render_system_message(f"[green]{msg}[/green]"))
        else:
            console.print(render_system_message(f"[red]{msg}[/red]"))
        return

    elif sub == "get":
        if len(parts) < 3:
            console.print("[red]Usage: /config get <setting>[/red]")
            return
        key = parts[2].lower().replace("-", "_")
        mapped = CONFIG_KEY_MAP.get(key, key)
        config = get_config()
        if hasattr(config, mapped):
            val = getattr(config, mapped)
            if mapped == "openai_api_key" and val:
                val = "***" + val[-4:]
            console.print(render_system_message(f"[bold cyan]{mapped}[/bold cyan]: {val}"))
        else:
            console.print(render_system_message(f"[red]Unknown setting: {parts[2]}[/red]"))
        return

    elif sub == "reset":
        set_config_value("max_iterations", 1000, controller)
        set_config_value("max_tool_calls", 1000, controller)
        set_config_value("command_timeout", 60, controller)
        console.print(render_system_message("[green]Reset iterations, tool calls, and timeout to defaults (1000 / 1000 / 60s).[/green]"))
        return

    # Check if user passed shorthand: /config <setting> <value>
    key_normalized = sub.replace("-", "_")
    if key_normalized in CONFIG_KEY_MAP and len(parts) >= 3:
        val = " ".join(parts[2:])
        ok, msg = set_config_value(sub, val, controller)
        if ok:
            console.print(render_system_message(f"[green]{msg}[/green]"))
        else:
            console.print(render_system_message(f"[red]{msg}[/red]"))
        return

    console.print(f"[yellow]Unknown config subcommand: '{sub}'. Type /config for available options.[/yellow]")


async def _handle_provider_command(parts: list, controller: AgentController) -> None:
    """Handle /provider and /settings slash commands for configuring AI providers and settings."""
    from agent.config import get_config, get_profiles, save_profile, delete_profile, load_profile
    from agent.model import reset_model_client, get_model_client

    subcommand = parts[1].lower() if len(parts) > 1 else ""

    if subcommand == "set":
        if len(parts) < 4:
            console.print("[red]Usage: /settings set <setting> <value>[/red]")
            console.print("[dim]Example: /settings set max_iterations 1000[/dim]")
            return
        key = parts[2]
        val = " ".join(parts[3:])
        ok, msg = set_config_value(key, val, controller)
        if ok:
            console.print(render_system_message(f"[green]{msg}[/green]"))
        else:
            console.print(render_system_message(f"[red]{msg}[/red]"))
        return

    elif subcommand in ("edit", "wizard"):
        await _interactive_config_edit(controller)
        return

    elif subcommand in ("tools", "tool", "access"):
        await _handle_tools_settings_command(parts[2:], controller)
        return

    if not subcommand:
        if sys.stdin.isatty():
            while True:
                config = get_config()
                profiles = get_profiles()
                key_preview = ("***" + config.openai_api_key[-4:]) if config.openai_api_key else "Not set"
                console.print()
                console.print(Panel(
                    f"[bold bright_green]Active Model:[/bold bright_green] [cyan]{config.openai_model}[/cyan]  •  "
                    f"[bold bright_green]API Key:[/bold bright_green] [dim]{key_preview}[/dim]\n"
                    f"[bold bright_green]Base URL:[/bold bright_green] [dim]{config.openai_base_url}[/dim]",
                    title="[bold bright_blue]AI Provider & Settings[/bold bright_blue]",
                    border_style="bright_blue",
                    box=ROUNDED,
                    padding=(0, 2),
                ))

                menu_opts = []
                for pname, prof in profiles.items():
                    is_active = (
                        prof.get("api_key") == config.openai_api_key
                        and prof.get("base_url") == config.openai_base_url
                    )
                    pmodel = prof.get("model") or "default"
                    menu_opts.append({
                        "key": f"use:{pname}",
                        "label": f"{'★ ' if is_active else '→ '}Switch to '{pname}' ({pmodel})" + (" [Active]" if is_active else ""),
                        "pt_style": "class:allow" if is_active else "class:accent",
                    })

                menu_opts.extend([
                    {"key": "add_google", "label": "★ Add / Update Google Gemini Preset (gemini-3.5-flash-lite / gemma-4-31b-it)", "pt_style": "class:allow"},
                    {"key": "add_nvidia", "label": "★ Add / Update NVIDIA NIM Preset (Free Tier — only API key)", "pt_style": "class:allow"},
                    {"key": "add_wizard", "label": "+ Add Provider (Google, NVIDIA, OpenAI, OpenRouter, Groq, Custom)", "pt_style": "class:accent"},
                    {"key": "model", "label": "◈ Switch Model (/model)", "pt_style": "class:item"},
                    {"key": "tools", "label": "🛡 Edit Tool Access (/settings tools)", "pt_style": "class:item"},
                    {"key": "edit", "label": "⚙ Edit Settings (/settings edit)", "pt_style": "class:item"},
                ])
                if profiles:
                    menu_opts.append({"key": "delete", "label": "✕ Delete a Saved Profile", "pt_style": "class:deny"})
                menu_opts.append({"key": "done", "label": "← Done", "pt_style": "class:item"})

                action = await _prompt_arrow_menu("Provider & Settings Menu:", menu_opts)
                if not action or action == "done":
                    return
                if action.startswith("use:"):
                    pname = action.split(":", 1)[1]
                    if load_profile(pname):
                        reset_model_client()
                        controller.model_client = get_model_client()
                        console.print(render_system_message(f"[green]✓ Switched active provider to '{pname}' (saved).[/green]"))
                    return
                elif action == "add_google":
                    await _handle_provider_command(["/provider", "google"], controller)
                    return
                elif action == "add_nvidia":
                    await _handle_provider_command(["/provider", "nvidia"], controller)
                    return
                elif action == "add_wizard":
                    await _handle_provider_command(["/provider", "add"], controller)
                    return
                elif action == "model":
                    await _show_models_command(get_config(), controller)
                    return
                elif action == "tools":
                    await _interactive_tool_access_edit(controller)
                elif action == "edit":
                    await _interactive_config_edit(controller)
                elif action == "delete":
                    await _handle_provider_command(["/provider", "delete"], controller)
            return
        subcommand = "list"

    if subcommand == "list":
        config = get_config()
        profiles = get_profiles()

        # Summarize tool access
        custom_rules = len(config.tool_access)
        tool_access_summary = f"{custom_rules} custom rule(s)" if custom_rules else "Default (safe reads=allow, writes/commands=ask)"

        # Header panel with instructions & NVIDIA recommendation
        info_text = (
            f"[bold bright_green]Active Model:[/bold bright_green] [cyan]{config.openai_model}[/cyan]\n"
            f"[bold bright_green]Active Base URL:[/bold bright_green] [dim]{config.openai_base_url}[/dim]\n"
            f"[bold bright_green]API Key:[/bold bright_green] [dim]{('***' + config.openai_api_key[-4:]) if config.openai_api_key else 'Not set'}[/dim]\n"
            f"[bold bright_green]Max Iterations:[/bold bright_green] [green]{config.max_iterations}[/green]  •  "
            f"[bold bright_green]Max Tool Calls:[/bold bright_green] [green]{config.max_tool_calls}[/green]\n"
            f"[bold bright_green]Tool Access:[/bold bright_green] [yellow]{tool_access_summary}[/yellow]\n\n"
            f"[bold bright_yellow]★ RECOMMENDED: NVIDIA NIM Free Models ★[/bold bright_yellow]\n"
            "NVIDIA provides 1,000 free API credits with access to top-tier open models.\n"
            "• [bold]Register for free key:[/bold] [underline cyan]https://build.nvidia.com/[/underline cyan]\n"
            "• [bold]Base URL:[/bold] [cyan]https://integrate.api.nvidia.com/v1[/cyan]\n"
            "• [bold]Preset Model:[/bold] [green]nvidia/nemotron-3.5-super-vl-120b-a12b[/green] (change anytime with /model)\n\n"
            "[bold]Quick Commands:[/bold]\n"
            "• [cyan]/provider nvidia[/cyan]      -> Fast 1-step setup for NVIDIA NIM\n"
            "• [cyan]/provider add[/cyan]         -> Guided wizard to add any provider (NVIDIA, OpenAI, Groq...)\n"
            "• [cyan]/provider use <name>[/cyan] -> Switch active provider profile\n"
            "• [cyan]/provider list[/cyan]        -> List all saved profiles\n"
            "• [cyan]/settings tools[/cyan]       -> Interactive tool access editor (allow / ask / deny)\n"
            "• [cyan]/settings set <name> <val>[/cyan] -> Change setting (e.g. /settings set max_iterations 1000)\n"
            "• [cyan]/settings edit[/cyan]        -> Interactive configuration editor\n"
            "• [cyan]/config[/cyan]               -> Full configuration table"
        )
        console.print()
        console.print(Panel(
            Text.from_markup(info_text),
            title="[bold bright_blue]AI Provider & API Settings[/bold bright_blue]",
            border_style="bright_blue",
            box=ROUNDED,
            padding=(1, 2),
        ))

        if profiles:
            table = Table(title="Saved Profiles", box=ROUNDED, border_style="dim")
            table.add_column("Name", style="cyan")
            table.add_column("Base URL", style="white")
            table.add_column("Model", style="dim")
            table.add_column("Status", style="green")

            for name, profile in profiles.items():
                is_active = (profile.get("api_key") == config.openai_api_key and
                             profile.get("base_url") == config.openai_base_url)
                table.add_row(
                    name,
                    profile.get("base_url", ""),
                    profile.get("model", "default"),
                    "[bold green]Active[/bold green]" if is_active else ""
                )
            console.print()
            console.print(table)
        console.print()
        return

    elif subcommand == "google":
        from agent.config import (
            GOOGLE_GEMINI_BASE_URL,
            GOOGLE_GEMINI_DEFAULT_MODEL,
            GOOGLE_GEMINI_GEMMA_MODEL,
            sync_api_credentials,
        )
        console.print()
        console.print(Panel(
            "[bold bright_cyan]Google Gemini Provider Setup (Preset)[/bold bright_cyan]\n"
            f"• Base URL: [cyan]{GOOGLE_GEMINI_BASE_URL}[/cyan]\n"
            f"• Default Model: [green]{GOOGLE_GEMINI_DEFAULT_MODEL}[/green] (15 calls/min)\n"
            f"• Gemma Model:   [green]{GOOGLE_GEMINI_GEMMA_MODEL}[/green] (30 calls/min)\n"
            "Get your free API key at: [bold underline cyan]https://aistudio.google.com/apikey[/bold underline cyan]",
            title="[bold green]Google Gemini[/bold green]",
            border_style="green",
            box=ROUNDED,
        ))

        api_key = Prompt.ask("[bold cyan]Enter Google Gemini API Key (AIza...)[/bold cyan]").strip()
        if not api_key:
            console.print("[yellow]Setup cancelled: API key required.[/yellow]")
            return

        selected_model = GOOGLE_GEMINI_DEFAULT_MODEL
        if sys.stdin.isatty():
            picked = await _prompt_arrow_menu(
                title="Select Default Google Model:",
                options=[
                    {
                        "key": GOOGLE_GEMINI_DEFAULT_MODEL,
                        "label": f"★ {GOOGLE_GEMINI_DEFAULT_MODEL} (Default — 15 calls/min)",
                        "pt_style": "class:allow",
                    },
                    {
                        "key": GOOGLE_GEMINI_GEMMA_MODEL,
                        "label": f"  {GOOGLE_GEMINI_GEMMA_MODEL} (30 calls/min)",
                        "pt_style": "class:accent",
                    },
                ],
            )
            if picked:
                selected_model = picked

        sync_api_credentials(
            api_key=api_key,
            base_url=GOOGLE_GEMINI_BASE_URL,
            model=selected_model,
            profile_name="google",
        )
        reset_model_client()
        controller.model_client = get_model_client()
        console.print(render_system_message(
            f"[green]Google Gemini configured & synced with model '{selected_model}'![/green]\n"
            f"[dim]Per-minute rate limiting is active automatically. Change model anytime with /model[/dim]"
        ))
        return

    elif subcommand == "nvidia":
        from agent.config import NVIDIA_NIM_BASE_URL, NVIDIA_NIM_DEFAULT_MODEL, sync_api_credentials
        console.print()
        console.print(Panel(
            "[bold bright_cyan]NVIDIA NIM Provider Setup (Preset)[/bold bright_cyan]\n"
            "NVIDIA provides 1,000 free API credits on registration!\n"
            f"• Base URL: [cyan]{NVIDIA_NIM_BASE_URL}[/cyan]\n"
            f"• Default Model: [green]{NVIDIA_NIM_DEFAULT_MODEL}[/green]\n"
            "Register & get your free key at: [bold underline cyan]https://build.nvidia.com/[/bold underline cyan]",
            title="[bold green]NVIDIA Free Tier[/bold green]",
            border_style="green",
            box=ROUNDED,
        ))

        api_key = Prompt.ask("[bold cyan]Enter NVIDIA API Key (nvapi-...)[/bold cyan]").strip()
        if not api_key:
            console.print("[yellow]Setup cancelled: API key required.[/yellow]")
            return

        sync_api_credentials(
            api_key=api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            model=NVIDIA_NIM_DEFAULT_MODEL,
            profile_name="nvidia",
        )
        reset_model_client()
        controller.model_client = get_model_client()
        console.print(render_system_message(
            f"[green]NVIDIA NIM configured & synced with model '{NVIDIA_NIM_DEFAULT_MODEL}'![/green]\n"
            f"[dim]You can change the default model later anytime with /model[/dim]"
        ))
        return

    elif subcommand == "add":
        from agent.config import (
            GOOGLE_GEMINI_BASE_URL,
            GOOGLE_GEMINI_DEFAULT_MODEL,
            NVIDIA_NIM_BASE_URL,
            NVIDIA_NIM_DEFAULT_MODEL,
            sync_api_credentials,
        )
        name = parts[2] if len(parts) > 2 else ""

        if not name:
            if sys.stdin.isatty():
                choice = await _prompt_arrow_menu(
                    title="Select Provider Template:",
                    options=[
                        {"key": "0", "label": "★ Google Gemini (Free Tier — aistudio.google.com/apikey)", "pt_style": "class:allow"},
                        {"key": "1", "label": "★ NVIDIA NIM (Recommended Free Tier — build.nvidia.com)", "pt_style": "class:allow"},
                        {"key": "2", "label": "OpenAI (https://api.openai.com/v1)", "pt_style": "class:accent"},
                        {"key": "3", "label": "OpenRouter (https://openrouter.ai/api/v1)", "pt_style": "class:accent"},
                        {"key": "4", "label": "Groq (https://api.groq.com/openai/v1)", "pt_style": "class:accent"},
                        {"key": "5", "label": "Custom OpenAI-compatible URL", "pt_style": "class:item"},
                        {"key": "cancel", "label": "← Cancel", "pt_style": "class:item"},
                    ],
                )
                if not choice or choice == "cancel":
                    return
            else:
                console.print()
                console.print("[bold cyan]Select provider template:[/bold cyan]")
                console.print("  [bold]0[/bold]) Google Gemini (Free Tier - https://aistudio.google.com/apikey)")
                console.print("  [bold]1[/bold]) NVIDIA NIM ([green]Recommended Free Tier[/green] - https://build.nvidia.com/)")
                console.print("  [bold]2[/bold]) OpenAI (https://api.openai.com/v1)")
                console.print("  [bold]3[/bold]) OpenRouter (https://openrouter.ai/api/v1)")
                console.print("  [bold]4[/bold]) Groq (https://api.groq.com/openai/v1)")
                console.print("  [bold]5[/bold]) Custom OpenAI-compatible URL")
                choice = Prompt.ask("[bold cyan]Choice[/bold cyan]", choices=["0", "1", "2", "3", "4", "5"], default="0")

            if choice == "0":
                name = "google"
                default_url = GOOGLE_GEMINI_BASE_URL
                default_model = GOOGLE_GEMINI_DEFAULT_MODEL
                console.print("[dim]Get free API key: https://aistudio.google.com/apikey[/dim]")
            elif choice == "1":
                name = "nvidia"
                default_url = NVIDIA_NIM_BASE_URL
                default_model = NVIDIA_NIM_DEFAULT_MODEL
                console.print("[dim]Register for free key: https://build.nvidia.com/[/dim]")
            elif choice == "2":
                name = "openai"
                default_url = "https://api.openai.com/v1"
                default_model = "gpt-4o-mini"
            elif choice == "3":
                name = "openrouter"
                default_url = "https://openrouter.ai/api/v1"
                default_model = "meta-llama/llama-3.3-70b-instruct"
            elif choice == "4":
                name = "groq"
                default_url = "https://api.groq.com/openai/v1"
                default_model = "llama-3.3-70b-versatile"
            else:
                name = Prompt.ask("[bold cyan]Profile Name[/bold cyan]", default="custom").strip()
                default_url = "https://api.openai.com/v1"
                default_model = "gpt-4o-mini"
        else:
            if name.lower() == "google":
                default_url = GOOGLE_GEMINI_BASE_URL
                default_model = GOOGLE_GEMINI_DEFAULT_MODEL
                console.print("[dim]Get free API key: https://aistudio.google.com/apikey[/dim]")
            elif name.lower() == "nvidia":
                default_url = NVIDIA_NIM_BASE_URL
                default_model = NVIDIA_NIM_DEFAULT_MODEL
                console.print("[dim]Register for free key: https://build.nvidia.com/[/dim]")
            else:
                default_url = "https://api.openai.com/v1"
                default_model = "gpt-4o-mini"

        base_url = Prompt.ask("[bold cyan]Base URL[/bold cyan]", default=default_url).strip()
        api_key = Prompt.ask("[bold cyan]API Key[/bold cyan]").strip()
        if not api_key:
            console.print("[yellow]API key cannot be empty.[/yellow]")
            return

        model = Prompt.ask("[bold cyan]Default Model[/bold cyan]", default=default_model).strip()

        save_profile(name, api_key, base_url, model)
        console.print(f"[green]Profile '{name}' saved.[/green]")

        if sys.stdin.isatty():
            activate = await _prompt_arrow_menu(
                title=f"Set '{name}' as active provider now?",
                options=[
                    {"key": "yes", "label": f"✓ Yes, activate '{name}' now", "pt_style": "class:allow"},
                    {"key": "no", "label": "Keep current active provider", "pt_style": "class:item"},
                ],
            )
            should_activate = (activate == "yes")
        else:
            should_activate = Confirm.ask(f"Set '{name}' as active profile?", default=True)

        if should_activate:
            sync_api_credentials(api_key=api_key, base_url=base_url, model=model, profile_name=name)
            reset_model_client()
            controller.model_client = get_model_client()
            console.print(render_system_message(f"[green]Active profile set to '{name}'.[/green]"))
        return

    elif subcommand == "delete":
        if len(parts) < 3:
            profiles = get_profiles()
            if sys.stdin.isatty() and profiles:
                del_opts = [
                    {"key": pname, "label": f"✕ Delete '{pname}' ({prof.get('base_url', '')})", "pt_style": "class:deny"}
                    for pname, prof in profiles.items()
                ]
                del_opts.append({"key": "__cancel__", "label": "← Cancel", "pt_style": "class:item"})
                name = await _prompt_arrow_menu("Select profile to delete:", del_opts)
                if not name or name == "__cancel__":
                    return
            else:
                console.print("[red]Usage: /provider delete <name>[/red]")
                return
        else:
            name = parts[2]

        if delete_profile(name):
            console.print(f"[green]Profile '{name}' deleted.[/green]")
        else:
            console.print(f"[red]Profile '{name}' not found.[/red]")
        return

    elif subcommand == "use":
        if len(parts) < 3:
            profiles = get_profiles()
            if sys.stdin.isatty() and profiles:
                use_opts = [
                    {"key": pname, "label": f"→ Switch to '{pname}' ({prof.get('model') or 'default'})", "pt_style": "class:accent"}
                    for pname, prof in profiles.items()
                ]
                use_opts.append({"key": "__cancel__", "label": "← Cancel", "pt_style": "class:item"})
                name = await _prompt_arrow_menu("Select profile to activate:", use_opts)
                if not name or name == "__cancel__":
                    return
            else:
                console.print("[red]Usage: /provider use <name>[/red]")
                return
        else:
            name = parts[2]
        if load_profile(name):
            reset_model_client()
            controller.model_client = get_model_client()
            console.print(render_system_message(f"[green]Active profile set to '{name}'.[/green]"))
        else:
            console.print(f"[red]Profile '{name}' not found.[/red]")
        return

    elif subcommand == "show":
        config = get_config()
        table = Table(title="Current Settings", box=ROUNDED, border_style="bright_blue")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", config.openai_model)
        table.add_row("Base URL", config.openai_base_url)
        table.add_row("API Key", "***" + config.openai_api_key[-4:] if config.openai_api_key else "Not set")
        table.add_row("Max Iterations", str(config.max_iterations))
        table.add_row("Max Tool Calls", str(config.max_tool_calls))
        table.add_row("Command Timeout", f"{config.command_timeout}s")
        table.add_row("Log Level", config.log_level)
        console.print()
        console.print(table)
        console.print()
        return

    else:
        # Check if subcommand is a profile name to use directly
        profiles = get_profiles()
        if subcommand in profiles:
            load_profile(subcommand)
            reset_model_client()
            controller.model_client = get_model_client()
            console.print(render_system_message(f"[green]Active profile set to '{subcommand}'.[/green]"))
            return

        console.print(f"[yellow]Unknown command option: {subcommand}. Use /provider for help.[/yellow]")


async def _fetch_models(base_url: str, api_key: str) -> list:
    """Fetch available models from OpenAI-compatible API."""
    import httpx
    url = base_url.rstrip("/") + "/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
    except Exception:
        return []


@app.callback(invoke_without_command=True)
def default_command(
    ctx: typer.Context,
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", "-u", help="API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", "-i", help="Maximum iterations (default: 1000)"
    ),
    auto_approve: bool = typer.Option(
        False, "--yes", "-y", "--auto-approve", help="Auto-approve all tool calls"
    ),
    new_session: bool = typer.Option(
        False, "--new", help="Start a new session without resuming previous history"
    ),
):
    """Start IRERI CODE interactive chat by default."""
    if ctx.invoked_subcommand is None:
        chat(
            project=project,
            model=model,
            base_url=base_url,
            api_key=api_key,
            max_iterations=max_iterations,
            auto_approve=auto_approve,
            new_session=new_session,
        )


@app.command()
def chat(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", "-u", help="API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", "-i", help="Maximum iterations (default: 1000)"
    ),
    auto_approve: bool = typer.Option(
        False, "--yes", "-y", "--auto-approve", help="Auto-approve all tool calls"
    ),
    new_session: bool = typer.Option(
        False, "--new", help="Start a new session without resuming previous history"
    ),
):
    """Start an interactive chat session with ireri."""
    import asyncio
    # Use existing event loop if running, otherwise create new
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_chat_async(project, model, base_url, api_key, max_iterations, auto_approve, new_session))
    finally:
        # Don't close if we didn't create it
        if not loop.is_closed() and loop != asyncio.get_event_loop():
            loop.close()


@app.command()
def run(
    task: str = typer.Argument(..., help="Task to execute"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    model: Optional[str] = typer.Option(
        None, "--model", "-m", help="Model to use"
    ),
    base_url: Optional[str] = typer.Option(
        None, "--base-url", "-u", help="API base URL"
    ),
    api_key: Optional[str] = typer.Option(
        None, "--api-key", "-k", help="API key"
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", "-i", help="Maximum iterations (default: 1000)"
    ),
    json_output: bool = typer.Option(
        False, "--json", "-j", help="Output result as JSON"
    ),
):
    """Execute a one-shot task with ireri."""
    from agent.config import reset_config
    from agent.model import reset_model_client

    reset_config()
    reset_model_client()
    config = get_config()

    # Override config with CLI args
    # workspace_root always defaults to current directory unless --project specified
    if project:
        config.workspace_root = project.resolve()
    else:
        config.workspace_root = Path.cwd().resolve()
    if model:
        config.openai_model = model
    if base_url:
        config.openai_base_url = base_url
    if api_key:
        config.openai_api_key = api_key
    if max_iterations is not None:
        config.max_iterations = max_iterations

    # Validate
    if not config.openai_api_key:
        console.print("[red]Error: API key required. Set OPENAI_API_KEY in .env or use --api-key[/red]")
        raise typer.Exit(1)

    console.print()
    console.print(render_system_message(f"[bold]Task:[/bold] {task}"))
    console.print(render_system_message(f"[bold]Workspace:[/bold] {config.workspace_root}"))
    console.print(render_system_message("[dim]Working...[/dim]"))
    console.print()

    try:
        result = asyncio.run(run_agent_task(task, str(config.workspace_root)))

        if json_output:
            import json
            console.print(json.dumps({"result": result}))
        else:
            # Replace problematic Unicode characters for Windows console
            safe_result = result.replace('\u2705', '[OK]').replace('\u274c', '[X]').replace('\u26a0', '[!]')
            console.print(render_agent_message(safe_result))

    except Exception as e:
        console.print()
        console.print(render_system_message(f"[red]Error: {e}[/red]"))
        raise typer.Exit(1)


@app.command()
def tools():
    """List available tools."""
    registry = get_registry()
    tools = registry.get_all()
    config = get_config()

    table = Table(title="ireri Available Tools", box=ROUNDED, border_style="bright_blue")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Access Policy", style="yellow")
    table.add_column("Dangerous", style="red")

    for tool in tools:
        default_p = "ask" if tool.requires_approval else "allow"
        policy = config.get_tool_access(tool.name, default=default_p)
        p_style = "green" if policy == "allow" else ("red" if policy == "deny" else "yellow")
        table.add_row(
            tool.name,
            tool.description,
            f"[{p_style}]{policy.upper()}[/{p_style}]",
            "Yes" if tool.dangerous else "No",
        )

    console.print()
    console.print(table)
    console.print()


@app.command(name="config")
def config_cli(
    action: Optional[str] = typer.Argument(None, help="Action: show, set, get, edit"),
    key: Optional[str] = typer.Argument(None, help="Setting name (for set/get)"),
    value: Optional[str] = typer.Argument(None, help="Setting value (for set)"),
):
    """View or modify ireri configuration."""
    if action is None or action == "show":
        _show_config_table(get_config())
    elif action == "set":
        if not key or not value:
            console.print("[red]Usage: ireri config set <setting> <value>[/red]")
            console.print("[dim]Example: ireri config set max_iterations 1000[/dim]")
            raise typer.Exit(1)
        ok, msg = set_config_value(key, value)
        if ok:
            console.print(render_system_message(f"[green]{msg}[/green]"))
        else:
            console.print(render_system_message(f"[red]{msg}[/red]"))
            raise typer.Exit(1)
    elif action == "get":
        if not key:
            console.print("[red]Usage: ireri config get <setting>[/red]")
            raise typer.Exit(1)
        key_norm = key.lower().replace("-", "_")
        mapped = CONFIG_KEY_MAP.get(key_norm, key_norm)
        cfg = get_config()
        if hasattr(cfg, mapped):
            val = getattr(cfg, mapped)
            if mapped == "openai_api_key" and val:
                val = "***" + val[-4:]
            console.print(f"{mapped}: {val}")
        else:
            console.print(f"[red]Unknown setting: {key}[/red]")
            raise typer.Exit(1)
    elif action.lower().replace("-", "_") in CONFIG_KEY_MAP and key is not None:
        # Shorthand: ireri config max_iterations 1000
        ok, msg = set_config_value(action, key)
        if ok:
            console.print(render_system_message(f"[green]{msg}[/green]"))
        else:
            console.print(render_system_message(f"[red]{msg}[/red]"))
            raise typer.Exit(1)
    else:
        _show_config_table(get_config())


@app.command()
def config_show():
    """Show current configuration."""
    _show_config_table(get_config())


@app.command()
def init(
    path: Path = typer.Argument(Path.cwd(), help="Project directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Initialize a new project with ireri configuration."""
    env_path = path / ".env"
    gitignore_path = path / ".gitignore"

    console.print()

    if env_path.exists() and not force:
        console.print(render_system_message("[yellow].env already exists. Use --force to overwrite.[/yellow]"))
    else:
        # Copy from example
        example_path = Path(__file__).parent.parent / ".env.example"
        if example_path.exists():
            env_path.write_text(example_path.read_text())
            console.print(render_system_message(f"[green]Created {env_path}[/green]"))
        else:
            console.print(render_system_message("[red]No .env.example found[/red]"))

    if gitignore_path.exists() and not force:
        console.print(render_system_message("[yellow].gitignore already exists. Use --force to overwrite.[/yellow]"))
    else:
        example_path = Path(__file__).parent.parent / ".gitignore"
        if example_path.exists():
            gitignore_path.write_text(example_path.read_text())
            console.print(render_system_message(f"[green]Created {gitignore_path}[/green]"))

    console.print()
    console.print(render_system_message("[bold]Next steps:[/bold]"))
    console.print(render_system_message("1. Run 'ireri' to start (interactive setup will guide you if needed)"))
    console.print(render_system_message("2. Or edit .env with your API key first"))
    console.print()


async def _show_models_command(config: AgentConfig, controller: Optional[AgentController] = None) -> None:
    """Handle /model command - fetch and switch models interactively with arrow keys."""
    from agent.config import update_config
    from agent.model import get_model_client, reset_model_client
    console.print("[dim]Fetching available models...[/dim]")

    try:
        models = await _fetch_models(config.openai_base_url, config.openai_api_key)

        if not models:
            console.print("[yellow]No models found or error fetching models.[/yellow]")
            if sys.stdin.isatty():
                loop = asyncio.get_event_loop()
                manual = (await loop.run_in_executor(
                    None,
                    lambda: Prompt.ask("[bold cyan]Enter model ID manually (Enter to cancel)[/bold cyan]", default=""),
                )).strip()
                if manual:
                    update_config(openai_model=manual)
                    reset_model_client()
                    if controller:
                        controller.model_client = get_model_client()
                    console.print(render_system_message(f"[green]Model changed to: {manual}[/green]"))
            return

        selected_model: Optional[str] = None
        if sys.stdin.isatty():
            initial_idx = 0
            menu_opts = []
            for idx, m in enumerate(models):
                mid = m.get("id", "unknown")
                is_cur = mid == config.openai_model
                if is_cur:
                    initial_idx = idx
                menu_opts.append({
                    "key": mid,
                    "label": f"{'★ ' if is_cur else '  '}{mid}" + ("  [active]" if is_cur else ""),
                    "pt_style": "class:allow" if is_cur else "class:item",
                })
            selected_model = await _prompt_arrow_menu(
                title=f"Select Model ({len(models)} available):",
                options=menu_opts,
                initial_index=initial_idx,
            )
        else:
            table = Table(title="Available Models")
            table.add_column("#", style="cyan", width=4)
            table.add_column("Model ID", style="white")
            table.add_column("Owned By", style="dim")
            for i, model in enumerate(models, 1):
                table.add_row(str(i), model.get("id", "unknown"), model.get("owned_by", "unknown"))
            console.print(table)
            choice = Prompt.ask("[bold cyan]Select model number[/bold cyan]", default="1")
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx]["id"]

        if selected_model:
            update_config(openai_model=selected_model)
            reset_model_client()
            if controller:
                controller.model_client = get_model_client()
            console.print(Panel(
                f"Model changed to: [green]{selected_model}[/green]\n"
                f"Saved to global config (~/.ireri-code/config.json)",
                title="[bold green]Model Updated[/bold green]",
                border_style="green",
            ))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def _show_models_list(config: AgentConfig) -> None:
    """Handle /models command - just list models."""
    console.print("[dim]Fetching available models...[/dim]")

    try:
        models = await _fetch_models(config.openai_base_url, config.openai_api_key)

        if not models:
            console.print("[yellow]No models found or error fetching models[/yellow]")
            return

        table = Table(title="Available Models", box=ROUNDED, border_style="bright_blue")
        table.add_column("#", style="cyan", width=4)
        table.add_column("Model ID", style="white")
        table.add_column("Created", style="dim")
        table.add_column("Owned By", style="dim")

        for i, model in enumerate(models, 1):
            created = model.get("created", "")
            if created:
                from datetime import datetime
                created = datetime.fromtimestamp(created).strftime("%Y-%m-%d")
            table.add_row(
                str(i),
                model.get("id", "unknown"),
                created,
                model.get("owned_by", "unknown")
            )

        console.print()
        console.print(table)
        console.print()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


async def _show_help_interactive() -> Optional[str]:
    """Interactive arrow-key command browser for /help (falls back to table if non-TTY)."""
    if not sys.stdin.isatty():
        _show_help()
        return None

    options = [{"key": "__close__", "label": "← Close Help", "pt_style": "class:item"}]
    for cmd, desc in SLASH_COMMANDS.items():
        icon = SLASH_COMMAND_ICONS.get(cmd, "✦")
        options.append({
            "key": cmd,
            "label": f"{icon} {cmd:<11} {desc[:58]}",
            "pt_style": "class:accent",
        })

    picked = await _prompt_arrow_menu(
        title="ireri Slash Commands (select a command to run it):",
        options=options,
    )
    if not picked or picked == "__close__":
        return None
    return picked


def _show_help() -> None:
    """Display all available slash commands."""
    table = Table(title="ireri Slash Commands", box=ROUNDED, border_style="bright_blue")
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")

    for cmd, desc in SLASH_COMMANDS.items():
        table.add_row(cmd, desc)

    console.print()
    console.print(table)
    console.print()


def _show_provider_command() -> None:
    """Handle /provider command - configure AI providers."""
    from agent.config import get_config
    config = get_config()
    info_text = (
        f"[bold bright_green]Active Model:[/bold bright_green] [cyan]{config.openai_model}[/cyan]\n"
        f"[bold bright_green]Active Base URL:[/bold bright_green] [dim]{config.openai_base_url}[/dim]\n"
        f"[bold bright_green]API Key:[/bold bright_green] [dim]{('***' + config.openai_api_key[-4:]) if config.openai_api_key else 'Not set'}[/dim]\n\n"
        "[bold bright_yellow]★ RECOMMENDED: NVIDIA NIM Free Models ★[/bold bright_yellow]\n"
        "NVIDIA provides 1,000 free API credits with access to top-tier open models.\n"
        "• [bold]Register for free key:[/bold] [underline cyan]https://build.nvidia.com/[/underline cyan]\n"
        "• [bold]Base URL:[/bold] [cyan]https://integrate.api.nvidia.com/v1[/cyan]\n"
        "• [bold]Preset Model:[/bold] [green]nvidia/nemotron-3.5-super-vl-120b-a12b[/green] (change anytime with /model)\n\n"
        "[bold]Quick Commands:[/bold]\n"
        "• [cyan]/provider nvidia[/cyan]      -> Fast 1-step setup for NVIDIA NIM\n"
        "• [cyan]/provider add[/cyan]         -> Guided wizard to add any provider\n"
        "• [cyan]/provider use <name>[/cyan] -> Switch active provider profile\n"
        "• [cyan]/provider list[/cyan]        -> List all saved profiles"
    )
    console.print()
    console.print(Panel(
        Text.from_markup(info_text),
        title="[bold bright_blue]AI Provider & API Settings[/bold bright_blue]",
        border_style="bright_blue",
        box=ROUNDED,
        padding=(1, 2),
    ))
    console.print()


def _show_tokens_command(controller: AgentController) -> None:
    """Handle /tokens command - display session token analytics and approximate context length."""
    from agent.model import get_rate_limiter
    stats = controller.context.get_token_stats()
    rpd = get_rate_limiter().get_daily_usage(controller.model_client.model, controller.model_client.base_url)
    table = Table(title="Session Token Analytics & Context", box=ROUNDED, border_style="bright_blue")
    table.add_column("Metric", style="cyan", width=26)
    table.add_column("Value", style="bold white")
    table.add_column("Description", style="dim")
    table.add_row("Tokens Sent (↑ Up)", format_tokens(stats['tokens_up']), "Input prompt tokens transmitted to model")
    table.add_row("Tokens Received (↓ Down)", format_tokens(stats['tokens_down']), "Output completion tokens generated by model")
    table.add_row("Total Session Tokens", format_tokens(stats['total_tokens']), "Total tokens consumed during this session")
    table.add_row("Approx. Active Context", f"~{format_tokens(stats['approx_context_tokens'])}", f"Estimated active context memory ({len(controller.context.messages)} conversation items)")
    if rpd.get("is_google"):
        table.add_row("Google Daily Requests (RPD)", f"{rpd['used']} / {rpd['limit']}", f"Requests today ({rpd['date_pt']} PT, resets 12:00 AM PT)")
    console.print()
    console.print(table)
    console.print()


def _show_status_command(controller: AgentController) -> None:
    """Handle /status command - show agent status."""
    from agent.model import get_rate_limiter
    config = get_config()
    stats = controller.context.get_token_stats()
    rpd = get_rate_limiter().get_daily_usage(controller.model_client.model, controller.model_client.base_url)
    table = Table(title="ireri Status", box=ROUNDED, border_style="bright_blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model", config.openai_model)
    table.add_row("Base URL", config.openai_base_url)
    table.add_row("Workspace", str(config.workspace_root))
    table.add_row("Max Iterations", str(config.max_iterations))
    table.add_row("Max Tool Calls", str(config.max_tool_calls))
    table.add_row("Queued Tasks", str(len(controller.message_queue)))
    table.add_row("Tokens Sent (↑)", format_tokens(stats['tokens_up']))
    table.add_row("Tokens Received (↓)", format_tokens(stats['tokens_down']))
    table.add_row("Total Session Tokens", format_tokens(stats['total_tokens']))
    table.add_row("Approx. Context Length", f"~{format_tokens(stats['approx_context_tokens'])}")
    if rpd.get("is_google"):
        table.add_row("Google RPD Today", f"{rpd['used']} / {rpd['limit']} (PT)")
    table.add_row("Session", "Active")

    console.print()
    console.print(table)
    console.print()


def _show_config_command() -> None:
    """Handle /config command - view or edit configuration."""
    _show_config_table(get_config())


def _show_version_command() -> None:
    """Handle /version command."""
    console.print(Panel(
        "[bold bright_blue]IRERI CODE[/bold bright_blue]  [bright_green]v2.1.9[/bright_green]\n"
        "Terminal AI Coding Agent\n"
        "https://github.com/ekusafamily/ireri-code",
        title="[bold]Version Info[/bold]",
        border_style="bright_blue",
    ))


def _show_update_command() -> None:
    """Handle /update command."""
    console.print(Panel(
        "To update ireri, run:\n"
        "[cyan]pip install --upgrade ireri-code[/cyan]\n\n"
        "Or if installed from source:\n"
        "[cyan]git pull && pip install -e .[/cyan]",
        title="[bold]Update[/bold]",
        border_style="bright_blue",
    ))


def _show_about_command() -> None:
    """Handle /about command."""
    console.print(Panel(
        "[bold bright_blue]IRERI CODE[/bold bright_blue] - Terminal AI Coding Agent\n\n"
        "An intelligent coding assistant that lives in your terminal.\n"
        "Features: file operations, code search, command execution,\n"
        "multi-step tasks, and interactive approval system.\n\n"
        "Created with [red]<3[/red] by the ireri team",
        title="[bold]About ireri[/bold]",
        border_style="bright_blue",
    ))


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()