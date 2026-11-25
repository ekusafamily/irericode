# IRERI CODE

[![PyPI version](https://img.shields.io/pypi/v/ireri-code.svg)](https://pypi.org/project/ireri-code/)
[![Python](https://img.shields.io/pypi/pyversions/ireri-code.svg)](https://pypi.org/project/ireri-code/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

```text
  ██╗██████╗ ███████╗██████╗ ██╗    ██████╗ ██████╗ ██████╗ ███████╗
  ██║██╔══██╗██╔════╝██╔══██╗██║   ██╔════╝██╔═══██╗██╔══██╗██╔════╝
  ██║██████╔╝█████╗  ██████╔╝██║   ██║     ██║   ██║██║  ██║█████╗  
  ██║██╔══██╗██╔══╝  ██╔══██╗██║   ██║     ██║   ██║██║  ██║██╔══╝  
  ██║██║  ██║███████╗██║  ██║██║   ╚██████╗╚██████╔╝██████╔╝███████╗
  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
```

**IRERI CODE** (`ireri-code`) is an interactive, terminal-based AI coding agent and Python library. It inspects repositories, plans and executes multi-step coding tasks, edits and creates files, runs terminal commands (including background servers/bots with live log streaming), and recovers automatically from errors—directly from your command line.

---

## Installation

Install directly from PyPI:

```bash
pip install ireri-code
```

Or install in an isolated CLI environment with `pipx`:

```bash
pipx install ireri-code
```

To upgrade to the latest release:

```bash
pip install --upgrade ireri-code
```

---

## Quick Start

Launch the agent from any project directory:

```bash
ireri
```

On first run (if no API key is configured), an interactive onboarding wizard guides you through setup:

1. **NVIDIA NIM Preset (Recommended Free Tier)** — Base URL (`https://integrate.api.nvidia.com/v1`) and default model (`nvidia/nemotron-3.5-super-vl-120b-a12b`) are pre-configured; you only enter your API key from [build.nvidia.com](https://build.nvidia.com/).
2. **Custom Provider** — Enter any OpenAI-compatible Base URL, API key, and default model (OpenAI, OpenRouter, Groq, Ollama, vLLM, LM Studio, etc.).
3. **Skip for now** — Start `ireri` without an API key and configure it later with `/provider` or `/settings`.

Credentials are automatically synchronized across `~/.ireri-code/config.json`, saved profiles (`~/.ireri-code/profiles.json`), and environment variables.

### Non-Interactive / `.env` Configuration

You can also configure via a `.env` file or CLI flags:

```env
OPENAI_API_KEY=nvapi-your-key-here
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_MODEL=nvidia/nemotron-3.5-super-vl-120b-a12b
```

```bash
ireri --model gpt-4o-mini --yes
```

---

## Key Features

- **Interactive Arrow-Key Navigation**:
  - Browse and execute `/provider`, `/settings`, `/model`, `/config`, `/ps`, `/logs`, `/queue`, and `/help` using `↑`/`↓` arrows and `Enter`.
  - Typing a partial slash command (e.g. `/se` or `/m`) and pressing `Enter` automatically selects the first matching suggestion.
- **Interactive Tool Approval & Guidance**:
  - Approve once (`y`), deny once (`n`), always allow/deny in `/settings` (`p`/`b`), or approve/deny for the session (`a`/`d`).
  - When denying a tool call, optionally tell the agent what to do instead—your instruction is sent straight to the model.
- **Background Processes & Live Logs**:
  - Long-running processes (dev servers, bots, watchers) run in the background without blocking the prompt.
  - Inspect or stream live output anytime with `/ps` and `/logs`.
- **Non-Blocking Chat, Queuing & Side-Notes**:
  - Queue follow-up tasks while the agent is working, or use `/btw <question>` to ask a side question and inject guidance into the running context.
- **Session Persistence & Auto-Recovery**:
  - Automatically saves and restores project sessions in `.ireri/`, tracks token usage (`/tokens`), and retries transient API failures with exponential backoff.

---

## Slash Commands

Type `/` in the prompt to open auto-completion:

| Command | Description |
| :--- | :--- |
| `/provider` | Interactive arrow-key menu to switch, add, or delete AI provider profiles |
| `/provider nvidia` | Quick 1-step setup for NVIDIA NIM free tier |
| `/settings` | Open interactive provider, config, and tool access settings |
| `/settings tools` | Interactive editor for per-tool permissions (`allow`, `ask`, `deny`) |
| `/model` | Fetch and switch models interactively with arrow keys |
| `/models` | List available models from the active provider |
| `/config` | View or interactively edit agent configuration (`max_iterations`, timeouts, etc.) |
| `/ps` | List, inspect, or stop ongoing background processes |
| `/logs` | View or follow live logs of a running/recent process (`/logs [follow] [id]`) |
| `/btw <question>` | Ask a direct side-question and guide the agent without stopping the current task |
| `/queue` | View, add, or clear queued follow-up tasks |
| `/tokens` | View session tokens sent/received and estimated context size |
| `/status` | Show current model, workspace, queue, and token status |
| `/history` | View recent conversation history |
| `/continue` | Restore the previous saved session |
| `/errors` | View recent entries from the debug error log |
| `/clear` | Clear the terminal screen |
| `/reset` | Reset the current session context |
| `/help` | Interactive slash command browser |
| `/version` | Show installed version |
| `/exit` | Exit the session (`/quit`, `/q`) |

---

## CLI Reference

```text
Usage: ireri [OPTIONS] COMMAND [ARGS]...

Default (no subcommand): starts interactive chat in the current directory.
  --project, -p         Project directory (default: current directory)
  --model, -m           Model identifier to use
  --base-url, -u        API base URL
  --api-key, -k         API key
  --max-iterations, -i  Maximum loop iterations per task [default: 1000]
  --yes, -y             Auto-approve all tool calls
  --new                 Start a fresh session without resuming history

Commands:
  chat                  Start an interactive chat session (alias for default)
  run <task>            Execute a one-shot task (supports --json)
  tools                 List all available built-in tools
  config                View or update saved configuration
  init                  Initialize a project with .env and .gitignore
```

---

## Python Library Usage

`ireri-code` can also be embedded programmatically in Python applications:

```python
import asyncio
from agent import AgentController, run_agent_task

# Run a one-shot task in a project directory
result = asyncio.run(run_agent_task("Add unit tests for utils.py", project_path="."))
print(result)
```

Or use `AgentController` directly for custom approval and progress hooks:

```python
import asyncio
from agent import AgentController

async def main():
    controller = AgentController()
    controller.set_approve_all(True)
    response = await controller.run_task("Summarize the repository structure")
    print(response)

asyncio.run(main())
```

---

## License

MIT License.
