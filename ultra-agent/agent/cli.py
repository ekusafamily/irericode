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
            thinking.start("[bold bright_cyan]Thinking...[/bold bright_cyan]")
    elif event == "model_response":
        if thinking:
            thinking.stop()
        if data.get("tool_calls_count", 0) > 0:
            console.print(f"  [cyan]Model: Planning {data['tool_calls_count']} tool call(s)[/cyan]")
        elif data.get("has_content"):
            console.print("  [cyan]Model: Responding...[/cyan]")
    elif event == "tool_start":
        if thinking:
            thinking.stop()
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
        metadata = data.get("metadata", {})
        if tool == "write_file":
            action = metadata.get("action", "written")
            console.print(f"  [green]OK[/green] File {action}\n")
        elif tool == "edit_file":
            replacements = data.get("output", {}).get("replacements", 0)
            console.print(f"  [yellow]OK[/yellow] File edited ({replacements} replacement(s))\n")
        elif tool == "read_file":
            output = data.get("output", {})
            lines = output.get("total_lines", 0)
            console.print(f"  [blue]OK[/blue] Read {lines} lines\n")
        elif tool == "list_files":
            count = data.get("metadata", {}).get("count", 0)
            console.print(f"  [blue]OK[/blue] Found {count} items\n")
        elif tool == "run_command":
            if success:
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
            if success:
                console.print(f"  [dim]OK[/dim] {tool} completed\n")
            else:
                console.print(f"  [red]FAIL[/red] {tool} failed: {data.get('error', 'unknown')}\n")
    elif event == "tool_error":
        if thinking:
            thinking.stop()
        console.print(f"  [red]FAIL[/red] {data['tool']} error: {data.get('error', 'unknown')}\n")
    elif event == "tool_denied":
        if thinking:
            thinking.stop()
        console.print(f"  [yellow]DENIED[/yellow] {data['tool']} denied\n")
    elif event == "model_error":
        if thinking:
            thinking.stop()
        console.print(f"  [red]Model error:[/red] {data.get('error', 'unknown')}\n")
    elif event == "model_retry":
        if thinking:
            thinking.stop()
        attempt = data.get("attempt", 1)
        max_attempts = data.get("max_attempts", 6)
        delay = data.get("delay", 1)
        err = data.get("error", "Error")
        delay_str = f"{delay // 60}m" if delay >= 60 else f"{delay}s"
        console.print(f"  [yellow]API error: {err}[/yellow]")
        console.print(f"  [yellow]Retrying in {delay_str} (attempt {attempt}/{max_attempts})...[/yellow]\n")
        if thinking:
            thinking.start(f"[bold bright_cyan]Waiting {delay_str} before retry ({attempt}/{max_attempts})...[/bold bright_cyan]")
    elif event == "model_retry_success":
        if thinking:
            thinking.stop()
        console.print(f"  [green]OK[/green] API call recovered on retry attempt {data.get('attempt')}\n")
        if thinking:
            thinking.start("[bold bright_cyan]Thinking...[/bold bright_cyan]")


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

    def stop(self):
        """Stop the status spinner."""
        if self._status is not None:
            try:
                self._status.stop()
            except Exception:
                pass
            self._status = None
        return _AwaitableNone()


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
    if max_iterations is not None:
        config.max_iterations = max_iterations

    # Validate
    if not config.openai_api_key:
        console.print("[red]Error: API key required. Set OPENAI_API_KEY in .env or use --api-key[/red]")
        raise typer.Exit(1)

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
    console.print("[bold bright_green]  Terminal AI Coding Agent  v0.1.0[/bold bright_green]")
    console.print()
    console.print(render_system_message(
        f"Model: [green]{config.openai_model}[/green]  •  "
        f"Workspace: [green]{config.workspace_root}[/green]  •  Max iterations: [green]{config.max_iterations}[/green]"
    ))
    console.print(render_system_message("[dim]Type 'exit' or 'quit' to leave. Type '/help' for commands.[/dim]"))
    console.print()

    # Initialize controller
    controller = AgentController()
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

    # Define approval callback for interactive mode
    async def _approval_callback(tool_name: str, args: Dict[str, Any], dangerous: bool) -> bool:
        """Prompt user for tool approval."""
        thinking.stop()
        import asyncio
        from rich.prompt import Prompt
        from rich.markup import escape

        # Format args for display
        args_display = ", ".join(f"{k}={v}" for k, v in args.items())
        if len(args_display) > 100:
            args_display = args_display[:97] + "..."

        # Build prompt message
        danger_indicator = " [red]! DANGEROUS[/red]" if dangerous else ""
        console.print()
        console.print(render_system_message(f"[yellow]Approval required:[/yellow] {escape(tool_name)}({escape(args_display)}){danger_indicator}"))
        console.print(render_system_message("[dim]Options: [bold]y[/bold]=approve  [bold]n[/bold]=deny  [bold]a[/bold]=approve all  [bold]d[/bold]=deny all[/dim]"))

        # Run blocking Prompt.ask in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        choice = await loop.run_in_executor(
            None,
            lambda: Prompt.ask(
                "[bold cyan]Approve?[/bold cyan]",
                choices=["y", "n", "a", "d"],
                default="y",
                case_sensitive=False,
            )
        )
        choice = str(choice).strip().lower()

        if choice == "a":
            controller.set_approve_all(True)
            console.print(render_system_message("[green]Approved all future tool calls for this session[/green]"))
            return True
        elif choice == "d":
            controller.set_deny_all(True)
            console.print(render_system_message("[red]Denied all future tool calls for this session[/red]"))
            return False
        elif choice == "y":
            return True
        else:
            return False

    controller.approval_callback = _approval_callback

    # Initialize prompt_toolkit session with auto-completion and dynamic prompt
    from prompt_toolkit import PromptSession
    from prompt_toolkit.shortcuts import CompleteStyle
    pt_completer = SlashCommandCompleter()
    pt_session = PromptSession(
        erase_when_done=True,
        completer=pt_completer,
        complete_while_typing=True,
        complete_style=CompleteStyle.COLUMN,
    )

    # Chat loop
    async def _chat_loop():
        while True:
            try:
                console.print()
                # Get user input with prompt_toolkit, falling back to readline/input
                try:
                    user_input = await pt_session.prompt_async(
                        _get_dynamic_prompt,
                        completer=pt_completer,
                        complete_while_typing=True,
                        complete_style=CompleteStyle.COLUMN,
                    )
                except (EOFError, KeyboardInterrupt):
                    raise
                except Exception:
                    loop = asyncio.get_event_loop()
                    user_input = await loop.run_in_executor(
                        None,
                        lambda: _get_input_with_readline()
                    )

                if user_input.lower() in ("exit", "quit", "q"):
                    console.print()
                    console.print(render_system_message("[yellow]Goodbye from ireri![/yellow]"))
                    break

                if not user_input.strip():
                    continue

                # Display user message on the right
                console.print(render_user_message(user_input))

                # Handle slash commands
                if user_input.strip().startswith("/"):
                    should_continue = await _handle_slash_command(user_input.strip(), controller)
                    if should_continue is False:
                        break
                    continue

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

                # Run task with progress and thinking animation
                try:
                    result = await controller.run_task(
                        user_input,
                        progress_callback=lambda event, data: _progress_callback(event, data, thinking),
                    )
                finally:
                    thinking.stop()

                # Display result on the left
                console.print(render_agent_message(result))

            except (typer.Exit, SystemExit):
                break
            except KeyboardInterrupt:
                console.print()
                console.print(render_system_message("[yellow]Interrupted[/yellow]"))
                controller.cancel()
            except EOFError:
                break
            except Exception as e:
                console.print()
                console.print(render_system_message(f"[red]Error: {e}[/red]"))

    # Run the async chat loop
    await _chat_loop()


# Available slash commands for auto-completion
SLASH_COMMANDS = {
    "/help": "Display all available commands",
    "/provider": "Configure AI providers (NVIDIA, OpenAI, etc.)",
    "/settings": "Open interactive settings & API profiles",
    "/model": "View or switch AI models",
    "/models": "List available models",
    "/status": "Show current agent status",
    "/config": "View, set or edit configuration (e.g. /config set max_iterations 1000)",
    "/history": "View conversation history and past messages",
    "/continue": "Continue previous conversation session",
    "/clear": "Clear terminal conversation",
    "/reset": "Reset current session",
    "/version": "Show agent version",
    "/update": "Check for and install updates",
    "/about": "Information about the agent",
    "/exit": "Exit the agent",
}

# Extract just command names for completion
SLASH_COMMAND_NAMES = list(SLASH_COMMANDS.keys())

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML


class SlashCommandCompleter(Completer):
    """Auto-completion for slash commands in prompt_toolkit."""

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if text.startswith("/"):
            for cmd, desc in SLASH_COMMANDS.items():
                if cmd.startswith(text):
                    yield Completion(cmd, start_position=-len(text), display_meta=desc)


def _get_dynamic_prompt():
    """Show clean prompt when idle, and 'You: ' dynamically once the user begins typing."""
    from prompt_toolkit.application.current import get_app
    try:
        app = get_app()
        if app and app.current_buffer and app.current_buffer.text:
            return HTML("<ansicyan><b>You: </b></ansicyan>")
    except Exception:
        pass
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
    return input("\x1b[1;36m❯\x1b[0m ")


async def _handle_slash_command(command: str, controller: AgentController) -> None:
    """Handle slash commands in chat."""
    from agent.config import get_config, update_config, get_profiles, save_profile, delete_profile, load_profile
    from agent.model import get_model_client, reset_model_client
    from agent.controller import AgentController

    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/help":
        _show_help()
        return True

    elif cmd == "/model":
        config = get_config()
        await _show_models_command(config)
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

    elif cmd == "/reset":
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


async def _interactive_config_edit(controller: Optional[AgentController] = None) -> None:
    """Interactive wizard to change any configuration setting."""
    config = get_config()
    console.print()
    console.print("[bold bright_cyan]Interactive Configuration Editor[/bold bright_cyan]")
    console.print(f"  [bold]1[/bold]) Max Iterations     [dim](current: {config.max_iterations})[/dim]")
    console.print(f"  [bold]2[/bold]) Max Tool Calls     [dim](current: {config.max_tool_calls})[/dim]")
    console.print(f"  [bold]3[/bold]) Command Timeout    [dim](current: {config.command_timeout}s)[/dim]")
    console.print(f"  [bold]4[/bold]) Model              [dim](current: {config.openai_model})[/dim]")
    console.print(f"  [bold]5[/bold]) Base URL           [dim](current: {config.openai_base_url})[/dim]")
    console.print(f"  [bold]6[/bold]) API Key            [dim](current: {('***' + config.openai_api_key[-4:]) if config.openai_api_key else 'Not set'})[/dim]")
    console.print(f"  [bold]7[/bold]) Log Level          [dim](current: {config.log_level})[/dim]")
    console.print("  [bold]0[/bold]) Cancel")

    import asyncio
    loop = asyncio.get_event_loop()
    choice = await loop.run_in_executor(
        None,
        lambda: Prompt.ask("[bold cyan]Select setting to edit[/bold cyan]", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="1")
    )

    if choice == "0":
        console.print("[dim]Edit cancelled.[/dim]")
        return

    mapping = {
        "1": ("max_iterations", str(config.max_iterations)),
        "2": ("max_tool_calls", str(config.max_tool_calls)),
        "3": ("command_timeout", str(config.command_timeout)),
        "4": ("model", config.openai_model),
        "5": ("base_url", config.openai_base_url),
        "6": ("api_key", ""),
        "7": ("log_level", config.log_level),
    }

    setting_key, default_val = mapping[choice]
    new_val = await loop.run_in_executor(
        None,
        lambda: Prompt.ask(f"[bold cyan]Enter new value for {setting_key}[/bold cyan]", default=default_val)
    )

    if not new_val.strip():
        console.print("[yellow]No change made.[/yellow]")
        return

    ok, msg = set_config_value(setting_key, new_val.strip(), controller)
    if ok:
        console.print(render_system_message(f"[green]{msg}[/green]"))
    else:
        console.print(render_system_message(f"[red]{msg}[/red]"))


async def _handle_config_command(parts: list, controller: Optional[AgentController] = None) -> None:
    """Handle /config slash command and its subcommands."""
    if len(parts) == 1:
        _show_config_table(get_config())
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

    if not subcommand or subcommand == "list":
        config = get_config()
        profiles = get_profiles()

        # Header panel with instructions & NVIDIA recommendation
        info_text = (
            f"[bold bright_green]Active Model:[/bold bright_green] [cyan]{config.openai_model}[/cyan]\n"
            f"[bold bright_green]Active Base URL:[/bold bright_green] [dim]{config.openai_base_url}[/dim]\n"
            f"[bold bright_green]API Key:[/bold bright_green] [dim]{('***' + config.openai_api_key[-4:]) if config.openai_api_key else 'Not set'}[/dim]\n"
            f"[bold bright_green]Max Iterations:[/bold bright_green] [green]{config.max_iterations}[/green]  •  "
            f"[bold bright_green]Max Tool Calls:[/bold bright_green] [green]{config.max_tool_calls}[/green]\n\n"
            "[bold bright_yellow]★ RECOMMENDED: NVIDIA NIM Free Models ★[/bold bright_yellow]\n"
            "NVIDIA provides 1,000 free API credits with access to top-tier open models.\n"
            "• [bold]Register for free key:[/bold] [underline cyan]https://build.nvidia.com/[/underline cyan]\n"
            "• [bold]Base URL:[/bold] [cyan]https://integrate.api.nvidia.com/v1[/cyan]\n"
            "• [bold]Top Models:[/bold] [green]nvidia/nemotron-3.5-bolt-e4b[/green], [green]meta/llama-3.3-70b-instruct[/green], [green]deepseek-ai/deepseek-r1[/green]\n\n"
            "[bold]Quick Commands:[/bold]\n"
            "• [cyan]/provider nvidia[/cyan]      -> Fast 1-step setup for NVIDIA NIM\n"
            "• [cyan]/provider add[/cyan]         -> Guided wizard to add any provider (NVIDIA, OpenAI, Groq...)\n"
            "• [cyan]/provider use <name>[/cyan] -> Switch active provider profile\n"
            "• [cyan]/provider list[/cyan]        -> List all saved profiles\n"
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

    elif subcommand == "nvidia":
        console.print()
        console.print(Panel(
            "[bold bright_cyan]NVIDIA NIM Provider Setup[/bold bright_cyan]\n"
            "NVIDIA provides 1,000 free API credits on registration!\n"
            "Register & get your free key at: [bold underline cyan]https://build.nvidia.com/[/bold underline cyan]",
            title="[bold green]NVIDIA Free Tier[/bold green]",
            border_style="green",
            box=ROUNDED,
        ))

        api_key = Prompt.ask("[bold cyan]Enter NVIDIA API Key (nvapi-...)[/bold cyan]").strip()
        if not api_key:
            console.print("[yellow]Setup cancelled: API key required.[/yellow]")
            return

        base_url = "https://integrate.api.nvidia.com/v1"
        console.print("\n[dim]Recommended models: nvidia/nemotron-3.5-bolt-e4b, meta/llama-3.3-70b-instruct, deepseek-ai/deepseek-r1[/dim]")
        model = Prompt.ask("[bold cyan]Default Model[/bold cyan]", default="nvidia/nemotron-3.5-bolt-e4b").strip()

        save_profile("nvidia", api_key, base_url, model)
        load_profile("nvidia")
        reset_model_client()
        controller.model_client = get_model_client()
        console.print(render_system_message(f"[green]NVIDIA provider configured and activated with model '{model}'![/green]"))
        return

    elif subcommand == "add":
        name = parts[2] if len(parts) > 2 else ""

        if not name:
            console.print()
            console.print("[bold cyan]Select provider template:[/bold cyan]")
            console.print("  [bold]1[/bold]) NVIDIA NIM ([green]Recommended Free Tier[/green] - https://build.nvidia.com/)")
            console.print("  [bold]2[/bold]) OpenAI (https://api.openai.com/v1)")
            console.print("  [bold]3[/bold]) OpenRouter (https://openrouter.ai/api/v1)")
            console.print("  [bold]4[/bold]) Groq (https://api.groq.com/openai/v1)")
            console.print("  [bold]5[/bold]) Custom OpenAI-compatible URL")
            choice = Prompt.ask("[bold cyan]Choice[/bold cyan]", choices=["1", "2", "3", "4", "5"], default="1")

            if choice == "1":
                name = "nvidia"
                default_url = "https://integrate.api.nvidia.com/v1"
                default_model = "nvidia/nemotron-3.5-bolt-e4b"
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
            if name.lower() == "nvidia":
                default_url = "https://integrate.api.nvidia.com/v1"
                default_model = "nvidia/nemotron-3.5-bolt-e4b"
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

        if Confirm.ask(f"Set '{name}' as active profile?", default=True):
            load_profile(name)
            reset_model_client()
            controller.model_client = get_model_client()
            console.print(render_system_message(f"[green]Active profile set to '{name}'.[/green]"))
        return

    elif subcommand == "delete":
        if len(parts) < 3:
            console.print("[red]Usage: /provider delete <name>[/red]")
            return

        name = parts[2]
        if delete_profile(name):
            console.print(f"[green]Profile '{name}' deleted.[/green]")
        else:
            console.print(f"[red]Profile '{name}' not found.[/red]")
        return

    elif subcommand == "use":
        if len(parts) < 3:
            console.print("[red]Usage: /provider use <name>[/red]")
            return

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

    table = Table(title="ireri Available Tools", box=ROUNDED, border_style="bright_blue")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Approval", style="yellow")
    table.add_column("Dangerous", style="red")

    for tool in tools:
        table.add_row(
            tool.name,
            tool.description,
            "Yes" if tool.requires_approval else "No",
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
    console.print(render_system_message("1. Edit .env with your API key"))
    console.print(render_system_message("2. Run 'ireri chat' to start"))
    console.print()


async def _show_models_command(config: AgentConfig) -> None:
    """Handle /model command - fetch and switch models."""
    from agent.config import update_config
    from agent.model import reset_model_client
    console.print("[dim]Fetching available models...[/dim]")

    try:
        models = await _fetch_models(config.openai_base_url, config.openai_api_key)

        if not models:
            console.print("[yellow]No models found or error fetching models[/yellow]")
            return

        # Display models in a table
        table = Table(title="Available Models")
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

        console.print(table)

        # Prompt for selection
        choice = Prompt.ask(
            "[bold cyan]Select model number[/bold cyan]",
            default="1"
        )

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected_model = models[idx]["id"]
                update_config(openai_model=selected_model)
                reset_model_client()

                console.print(Panel(
                    f"Model changed to: [green]{selected_model}[/green]\n"
                    f"Saved to global config (~/.ireri-code/config.json)",
                    title="[bold green]Model Updated[/bold green]",
                    border_style="green",
                ))
            else:
                console.print("[red]Invalid selection[/red]")
        except ValueError:
            console.print("[red]Invalid input[/red]")

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
        "• [bold]Top Models:[/bold] [green]nvidia/nemotron-3.5-bolt-e4b[/green], [green]meta/llama-3.3-70b-instruct[/green], [green]deepseek-ai/deepseek-r1[/green]\n\n"
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


def _show_status_command(controller: AgentController) -> None:
    """Handle /status command - show agent status."""
    config = get_config()
    table = Table(title="ireri Status", box=ROUNDED, border_style="bright_blue")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model", config.openai_model)
    table.add_row("Base URL", config.openai_base_url)
    table.add_row("Workspace", str(config.workspace_root))
    table.add_row("Max Iterations", str(config.max_iterations))
    table.add_row("Max Tool Calls", str(config.max_tool_calls))
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
        "[bold bright_blue]IRERI CODE[/bold bright_blue]  [bright_green]v0.1.0[/bright_green]\n"
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