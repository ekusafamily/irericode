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
    return Panel(
        Text(content, justify="center"),
        border_style="dim",
        box=ROUNDED,
        padding=(0, 1),
        width=console.width - 4,
    )


def _progress_callback(event: str, data: Dict[str, Any]) -> None:
    """Display progress updates from the agent."""
    if event == "iteration_start":
        console.print(f"[dim]Iteration {data['iteration']}/{data['max_iterations']}[/dim]")
    elif event == "model_response":
        if data["tool_calls_count"] > 0:
            console.print(f"[cyan]Model: Planning {data['tool_calls_count']} tool call(s)[/cyan]")
        elif data["has_content"]:
            console.print("[cyan]Model: Responding...[/cyan]")
    elif event == "tool_start":
        tool = data["tool"]
        args = data["args"]
        if tool == "write_file":
            console.print(f"[green]>[/green] Creating file: {args.get('path', 'unknown')}")
        elif tool == "edit_file":
            console.print(f"[yellow]>[/yellow] Editing file: {args.get('path', 'unknown')}")
        elif tool == "read_file":
            console.print(f"[blue]>[/blue] Reading file: {args.get('path', 'unknown')}")
        elif tool == "list_files":
            console.print(f"[blue]>[/blue] Listing directory: {args.get('path', '.')}")
        elif tool == "run_command":
            cmd = args.get('command', 'unknown')
            console.print(f"[magenta]>[/magenta] Running: {cmd[:60]}...")
        elif tool == "delete_file":
            console.print(f"[red]>[/red] Deleting: {args.get('path', 'unknown')}")
        elif tool == "create_directory":
            console.print(f"[green]>[/green] Creating directory: {args.get('path', 'unknown')}")
        elif tool == "search_files":
            console.print(f"[blue]>[/blue] Searching: {args.get('pattern', 'unknown')}")
        elif tool == "grep":
            console.print(f"[blue]>[/blue] Grepping: {args.get('pattern', 'unknown')}")
        else:
            console.print(f"[dim]>[/dim] {tool}: {args}")
    elif event == "tool_complete":
        tool = data["tool"]
        success = data["success"]
        metadata = data.get("metadata", {})
        if tool == "write_file":
            action = metadata.get("action", "written")
            console.print(f"[green]OK[/green] File {action}")
        elif tool == "edit_file":
            replacements = data.get("output", {}).get("replacements", 0)
            console.print(f"[yellow]OK[/yellow] File edited ({replacements} replacement(s))")
        elif tool == "read_file":
            output = data.get("output", {})
            lines = output.get("total_lines", 0)
            console.print(f"[blue]OK[/blue] Read {lines} lines")
        elif tool == "list_files":
            count = data.get("metadata", {}).get("count", 0)
            console.print(f"[blue]OK[/blue] Found {count} items")
        elif tool == "run_command":
            if success:
                console.print(f"[magenta]OK[/magenta] Command completed")
            else:
                console.print(f"[red]FAIL[/red] Command failed")
        elif tool == "delete_file":
            action = metadata.get("action", "deleted")
            console.print(f"[red]OK[/red] {action}")
        elif tool == "create_directory":
            console.print(f"[green]OK[/green] Directory created")
        elif tool == "search_files" or tool == "grep":
            console.print(f"[blue]OK[/blue] Search completed")
        else:
            if success:
                console.print(f"[dim]OK[/dim] {tool} completed")
            else:
                console.print(f"[red]FAIL[/red] {tool} failed: {data.get('error', 'unknown')}")
    elif event == "tool_error":
        console.print(f"[red]FAIL[/red] {data['tool']} error: {data['error']}")
    elif event == "tool_denied":
        console.print(f"[yellow]DENIED[/yellow] {data['tool']} denied")
    elif event == "model_error":
        console.print(f"[red]Model error:[/red] {data['error']}")


class ThinkingIndicator:
    """Animated thinking indicator with beeping dots."""
    def __init__(self, console: Console):
        self.console = console
        self.running = False
        self.task = None

    async def start(self):
        self.running = True
        # Print "thinking" once at the start
        self.console.print("[bold bright_cyan][bright_white]thinking[/bright_white]", end="")
        self.console.file.flush()
        self.task = asyncio.create_task(self._animate())

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        # Clear the line
        self.console.print("\r", end="")

    async def _animate(self):
        dots = [".  ", ".. ", "...", " ..", "  .", "   "]
        i = 0
        while self.running:
            self.console.print(f"\r[bold bright_cyan]{dots[i % len(dots)]}[/bold bright_cyan]", end="")
            self.console.file.flush()
            await asyncio.sleep(0.1)
            i += 1


async def _chat_async(
    project: Optional[Path],
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    max_iterations: int,
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
    config.max_iterations = max_iterations

    # Validate
    if not config.openai_api_key:
        console.print("[red]Error: API key required. Set OPENAI_API_KEY in .env or use --api-key[/red]")
        raise typer.Exit(1)

    # Welcome banner - IRERI CODE (maximized for visibility)
    console.print()
    console.print("[bold bright_blue]  #######  ##    ##  #######  ########  ########  [/bold bright_blue]")
    console.print("[bold bright_blue] ##       ##    ##  ##   ##  ##          ##       [/bold bright_blue]")
    console.print("[bold bright_blue] ##       ##    ##  ########  ######      ######   [/bold bright_blue]")
    console.print("[bold bright_blue] ##       ##    ##  ##   ##  ##          ##       [/bold bright_blue]")
    console.print("[bold bright_blue]  #######  ########  ##   ##  ########  ########  [/bold bright_blue]")
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

    # Define approval callback for interactive mode
    async def _approval_callback(tool_name: str, args: Dict[str, Any], dangerous: bool) -> bool:
        """Prompt user for tool approval."""
        import asyncio
        from rich.prompt import Prompt

        # Format args for display
        args_display = ", ".join(f"{k}={v}" for k, v in args.items())
        if len(args_display) > 100:
            args_display = args_display[:97] + "..."

        # Build prompt message
        danger_indicator = " [red]! DANGEROUS[/red]" if dangerous else ""
        console.print()
        console.print(render_system_message(f"[yellow]Approval required:[/yellow] {tool_name}({args_display}){danger_indicator}"))
        console.print(render_system_message("[dim]Options: [bold]y[/bold]=approve  [bold]n[/bold]=deny  [bold]a[/bold]=approve all  [bold]d[/bold]=deny all[/dim]"))

        # Run blocking Prompt.ask in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        choice = await loop.run_in_executor(
            None,
            lambda: Prompt.ask("[bold cyan]Approve?[/bold cyan]", choices=["y", "n", "a", "d"], default="y")
        )

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

    # Set up readline completion
    setup_readline_completion()

    # Chat loop - use readline for tab completion
    async def _chat_loop():
        while True:
            try:
                console.print()
                # Get user input using readline for tab completion
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
                    await _handle_slash_command(user_input.strip(), controller)
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
                    console.print(render_agent_message(response))
                    continue

                # Run task with thinking animation
                thinking = ThinkingIndicator(console)
                await thinking.start()
                try:
                    result = await controller.run_task(user_input, progress_callback=_progress_callback)
                finally:
                    await thinking.stop()

                # Display result on the left
                console.print(render_agent_message(result))

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
    "/settings": "Open interactive settings",
    "/model": "View or switch AI models",
    "/models": "List available models",
    "/provider": "Configure AI providers",
    "/status": "Show current agent status",
    "/config": "View or edit configuration",
    "/exit": "Exit the agent",
    "/clear": "Clear terminal conversation",
    "/reset": "Reset current session",
    "/version": "Show agent version",
    "/update": "Check for and install updates",
    "/about": "Information about the agent",
}

# Extract just command names for completion
SLASH_COMMAND_NAMES = list(SLASH_COMMANDS.keys())

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
    """Get user input with readline tab completion support."""
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
    return input("\x1b[1;36mYou\x1b[0m ")


async def _handle_slash_command(command: str, controller: AgentController) -> None:
    """Handle slash commands in chat."""
    from agent.config import get_config, update_config, get_profiles, save_profile, delete_profile, load_profile
    from agent.model import get_model_client, reset_model_client
    from agent.controller import AgentController

    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/help":
        _show_help()
        return

    elif cmd == "/model":
        config = get_config()
        await _show_models_command(config)
        return

    elif cmd == "/models":
        config = get_config()
        await _show_models_list(config)
        return

    elif cmd == "/settings":
        await _handle_settings_command(parts, controller)
        return

    elif cmd == "/provider":
        _show_provider_command()
        return

    elif cmd == "/status":
        _show_status_command(controller)
        return

    elif cmd == "/config":
        _show_config_command()
        return

    elif cmd == "/exit":
        console.print(render_system_message("[yellow]Goodbye from ireri![/yellow]"))
        raise typer.Exit(0)

    elif cmd == "/clear":
        console.clear()
        return

    elif cmd == "/reset":
        controller.reset()
        console.print(render_system_message("[green]Session reset.[/green]"))
        return

    elif cmd == "/version":
        _show_version_command()
        return

    elif cmd == "/update":
        _show_update_command()
        return

    elif cmd == "/about":
        _show_about_command()
        return

    else:
        console.print(f"[yellow]Unknown command: {cmd}. Type /help for available commands.[/yellow]")


async def _handle_settings_command(parts: list, controller: AgentController) -> None:
    """Handle /settings slash command for managing API key profiles."""
    from agent.config import get_config, get_profiles, save_profile, delete_profile, load_profile
    from agent.model import reset_model_client, get_model_client

    subcommand = parts[1].lower() if len(parts) > 1 else "list"

    if subcommand == "list":
        profiles = get_profiles()
        config = get_config()

        if not profiles:
            console.print("[yellow]No profiles saved. Use '/settings add <name>' to add one.[/yellow]")
            return

        table = Table(title="API Key Profiles")
        table.add_column("Name", style="cyan")
        table.add_column("Base URL", style="white")
        table.add_column("Model", style="dim")
        table.add_column("Active", style="green")

        for name, profile in profiles.items():
            is_active = (profile["api_key"] == config.openai_api_key and
                         profile["base_url"] == config.openai_base_url)
            table.add_row(
                name,
                profile["base_url"],
                profile.get("model", "default"),
                "[green]*[/green]" if is_active else ""
            )

        console.print(table)

    elif subcommand == "add":
        if len(parts) < 3:
            console.print("[red]Usage: /settings add <name>[/red]")
            return

        name = parts[2]
        profiles = get_profiles()

        if name in profiles:
            console.print(f"[yellow]Profile '{name}' already exists. Use '/settings delete {name}' first.[/yellow]")
            return

        api_key = Prompt.ask("[bold cyan]API Key[/bold cyan]")
        base_url = Prompt.ask("[bold cyan]Base URL[/bold cyan]", default="https://api.openai.com/v1")
        model = Prompt.ask("[bold cyan]Model (optional)[/bold cyan]", default="")

        save_profile(name, api_key, base_url, model)
        console.print(f"[green]Profile '{name}' saved.[/green]")

        if Confirm.ask("Set as active profile?", default=True):
            load_profile(name)
            reset_model_client()
            controller.model_client = get_model_client()
            console.print(f"[green]Active profile set to '{name}'.[/green]")

    elif subcommand == "delete":
        if len(parts) < 3:
            console.print("[red]Usage: /settings delete <name>[/red]")
            return

        name = parts[2]
        if delete_profile(name):
            console.print(f"[green]Profile '{name}' deleted.[/green]")
        else:
            console.print(f"[red]Profile '{name}' not found.[/red]")

    elif subcommand == "use":
        if len(parts) < 3:
            console.print("[red]Usage: /settings use <name>[/red]")
            return

        name = parts[2]
        if load_profile(name):
            reset_model_client()
            controller.model_client = get_model_client()
            console.print(f"[green]Active profile set to '{name}'.[/green]")
        else:
            console.print(f"[red]Profile '{name}' not found.[/red]")

    elif subcommand == "show":
        config = get_config()
        table = Table(title="Current Settings")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Model", config.openai_model)
        table.add_row("Base URL", config.openai_base_url)
        table.add_row("API Key", "***" + config.openai_api_key[-4:] if config.openai_api_key else "Not set")
        table.add_row("Max Iterations", str(config.max_iterations))
        table.add_row("Max Tool Calls", str(config.max_tool_calls))
        table.add_row("Command Timeout", f"{config.command_timeout}s")
        table.add_row("Log Level", config.log_level)

        console.print(table)

    else:
        console.print("[yellow]Unknown subcommand. Use: list, add, delete, use, show[/yellow]")


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
    max_iterations: int = typer.Option(
        20, "--max-iterations", "-i", help="Maximum iterations"
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
        loop.run_until_complete(_chat_async(project, model, base_url, api_key, max_iterations))
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
    max_iterations: int = typer.Option(
        20, "--max-iterations", "-i", help="Maximum iterations"
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


@app.command()
def config_show():
    """Show current configuration."""
    config = get_config()

    table = Table(title="ireri Configuration", box=ROUNDED, border_style="bright_blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model", config.openai_model)
    table.add_row("Base URL", config.openai_base_url)
    table.add_row("API Key", "***" + config.openai_api_key[-4:] if config.openai_api_key else "Not set")
    table.add_row("Workspace", str(config.workspace_root))
    table.add_row("Max Iterations", str(config.max_iterations))
    table.add_row("Max Tool Calls", str(config.max_tool_calls))
    table.add_row("Command Timeout", f"{config.command_timeout}s")
    table.add_row("Log Level", config.log_level)

    console.print()
    console.print(table)
    console.print()


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
    console.print(Panel(
        "[dim]Provider configuration coming soon.[/dim]\n"
        "Currently supports OpenAI-compatible endpoints.\n"
        "Use /settings to manage API key profiles.",
        title="[bold]AI Provider Settings[/bold]",
        border_style="bright_blue",
    ))


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
    table.add_row("Session", "Active")

    console.print()
    console.print(table)
    console.print()


def _show_config_command() -> None:
    """Handle /config command - view or edit configuration."""
    config = get_config()
    table = Table(title="ireri Configuration", box=ROUNDED, border_style="bright_blue")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    table.add_row("Model", config.openai_model)
    table.add_row("Base URL", config.openai_base_url)
    table.add_row("API Key", "***" + config.openai_api_key[-4:] if config.openai_api_key else "Not set")
    table.add_row("Workspace", str(config.workspace_root))
    table.add_row("Max Iterations", str(config.max_iterations))
    table.add_row("Max Tool Calls", str(config.max_tool_calls))
    table.add_row("Command Timeout", f"{config.command_timeout}s")
    table.add_row("Log Level", config.log_level)

    console.print()
    console.print(table)
    console.print()


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