"""Command-line interface for the Ultra Agent."""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table

from agent.config import get_config, AgentConfig, reset_config
from agent.controller import AgentController, run_agent_task
from agent.context import ContextManager
from agent.model import get_model_client, reset_model_client
from agent.tools import get_registry

console = Console()
app = typer.Typer(
    name="ultra",
    help="Ultra - Terminal AI Coding Agent",
    add_completion=False,
)


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

    console.print(Panel.fit(
        f"[bold cyan]Ultra Agent[/bold cyan] v0.1.0\n"
        f"Model: [green]{config.openai_model}[/green]\n"
        f"Workspace: [green]{config.workspace_root}[/green]\n"
        f"Max iterations: [green]{config.max_iterations}[/green]",
        title="Configuration",
        border_style="cyan",
    ))

    # Initialize controller
    controller = AgentController()

    console.print("\n[bold]Interactive mode started. Type 'exit' or 'quit' to leave.[/bold]")
    console.print("[dim]Tip: Type '/model' to change models[/dim]\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")

            if user_input.lower() in ("exit", "quit", "q"):
                console.print("[yellow]Goodbye![/yellow]")
                break

            if not user_input.strip():
                continue

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
                console.print(Panel(
                    Markdown(response),
                    title="[bold green]Ultra[/bold green]",
                    border_style="green",
                ))
                continue

            # Run task
            console.print("[dim]Working...[/dim]")
            result = await controller.run_task(user_input)

            # Display result
            console.print(Panel(
                Markdown(result),
                title="[bold green]Ultra[/bold green]",
                border_style="green",
            ))

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
            controller.cancel()
        except EOFError:
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def _handle_slash_command(command: str, controller: AgentController) -> None:
    """Handle slash commands in chat."""
    from agent.config import get_config, update_config, get_profiles, save_profile, delete_profile, load_profile
    from agent.model import get_model_client, reset_model_client

    parts = command.split()
    cmd = parts[0].lower()

    if cmd == "/model":
        config = get_config()
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

                    # Recreate controller with new model
                    controller.model_client = get_model_client()

                    console.print(Panel(
                        f"Model changed to: [green]{selected_model}[/green]\n"
                        f"Saved to global config (~/.ultra-agent/config.json)",
                        title="[bold green]Model Updated[/bold green]",
                        border_style="green",
                    ))
                else:
                    console.print("[red]Invalid selection[/red]")
            except ValueError:
                console.print("[red]Invalid input[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    elif cmd == "/settings":
        await _handle_settings_command(parts, controller)

    elif cmd == "/help":
        console.print(Panel(
            "Available slash commands:\n"
            "  [cyan]/model[/cyan] - Change the model\n"
            "  [cyan]/settings[/cyan] - Manage API key profiles\n"
            "  [cyan]/help[/cyan] - Show this help",
            title="[bold]Slash Commands[/bold]",
            border_style="cyan",
        ))

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
                "[green]●[/green]" if is_active else ""
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
    """Start an interactive chat session."""
    asyncio.run(_chat_async(project, model, base_url, api_key, max_iterations))


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
    """Execute a one-shot task."""
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

    console.print(f"[bold]Task:[/bold] {task}")
    console.print(f"[bold]Workspace:[/bold] {config.workspace_root}")
    console.print("[dim]Working...[/dim]\n")

    try:
        result = asyncio.run(run_agent_task(task, str(config.workspace_root)))

        if json_output:
            import json
            console.print(json.dumps({"result": result}))
        else:
            # Replace problematic Unicode characters for Windows console
            safe_result = result.replace('\u2705', '[OK]').replace('\u274c', '[X]').replace('\u26a0', '[!]')
            console.print(Panel(
                Markdown(safe_result),
                title="[bold green]Result[/bold green]",
                border_style="green",
            ))

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def tools():
    """List available tools."""
    registry = get_registry()
    tools = registry.get_all()

    table = Table(title="Available Tools")
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

    console.print(table)


@app.command()
def config_show():
    """Show current configuration."""
    config = get_config()

    table = Table(title="Configuration")
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

    console.print(table)


@app.command()
def init(
    path: Path = typer.Argument(Path.cwd(), help="Project directory"),
    force: bool = typer.Option(False, "--force", "-f", help="Overwrite existing files"),
):
    """Initialize a new project with Ultra configuration."""
    env_path = path / ".env"
    gitignore_path = path / ".gitignore"

    if env_path.exists() and not force:
        console.print(f"[yellow].env already exists. Use --force to overwrite.[/yellow]")
    else:
        # Copy from example
        example_path = Path(__file__).parent.parent / ".env.example"
        if example_path.exists():
            env_path.write_text(example_path.read_text())
            console.print(f"[green]Created {env_path}[/green]")
        else:
            console.print("[red]No .env.example found[/red]")

    if gitignore_path.exists() and not force:
        console.print(f"[yellow].gitignore already exists. Use --force to overwrite.[/yellow]")
    else:
        example_path = Path(__file__).parent.parent / ".gitignore"
        if example_path.exists():
            gitignore_path.write_text(example_path.read_text())
            console.print(f"[green]Created {gitignore_path}[/green]")

    console.print("\n[bold]Next steps:[/bold]")
    console.print("1. Edit .env with your API key")
    console.print("2. Run 'ultra chat' to start")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()