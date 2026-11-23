# IRERI CODE 🚀
### Terminal AI Coding Agent (v0.1.0)

```
  ██╗██████╗ ███████╗██████╗ ██╗    ██████╗ ██████╗ ██████╗ ███████╗
  ██║██╔══██╗██╔════╝██╔══██╗██║   ██╔════╝██╔═══██╗██╔══██╗██╔════╝
  ██║██████╔╝█████╗  ██████╔╝██║   ██║     ██║   ██║██║  ██║█████╗  
  ██║██╔══██╗██╔══╝  ██╔══██╗██║   ██║     ██║   ██║██║  ██║██╔══╝  
  ██║██║  ██║███████╗██║  ██║██║   ╚██████╗╚██████╔╝██████╔╝███████╗
  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
```

**IRERI CODE** is an intelligent, autonomous terminal-based AI coding assistant. It inspects repositories, plans and executes multi-step coding tasks, edits and creates files, executes shell commands, and iterates until the job is done—all directly from your command line.

---

## ✨ Features

- **Autonomous Multi-Step Agent Loop**:
  - Plans and executes tools autonomously with up to 50 iterations per task.
  - Automatically resets iteration budgets after each message generation.
  - Self-recovering: catches execution errors and adapts strategy.

- **Multi-Provider & Model Support**:
  - Compatible with any OpenAI-compatible endpoint.
  - Built-in support for **NVIDIA NIM**, **OpenAI**, **OpenRouter**, **Groq**, and local models (**Ollama**, **vLLM**, **LM Studio**).
  - Recommended free tier: **NVIDIA NIM** (1,000 free API credits on signup at [build.nvidia.com](https://build.nvidia.com/)).

- **Interactive Terminal UI**:
  - High-impact visual banner and clean message styling powered by **Rich**.
  - Dynamic user prompt: clean `❯ ` prompt when idle, transitioning to `You: ` in real time while typing.
  - Automatic slash-command suggestions pop-up when keying in `/`.
  - Clean tool updates with visual hierarchy and non-intrusive status indicators.

- **Interactive Tool Approval & Safety**:
  - Interactive approval prompts (`y` = approve, `n` = deny, `a` = approve all, `d` = deny all).
  - Dangerous shell command detection (e.g., `rm -rf`, `sudo`, destructive operations).
  - Optional one-shot auto-approval mode (`--yes` / `-y`).

- **Built-in Tool Ecosystem**:
  - **File Operations**: `read_file`, `write_file`, `edit_file`, `delete_file`, `list_files`, `create_directory`.
  - **Code Search**: Fast regex search (`search_files`) and content searching (`grep`).
  - **Terminal**: `run_command` with timeout and working directory isolation.

---

## 📦 Installation

Clone the repository and install it in editable mode:

```bash
git clone https://github.com/ekusafamily/ireri-code.git
cd ireri-code/ultra-agent
pip install -e .
```

---

## ⚡ Quick Start

### 1. Configure an AI Provider

You can configure your provider interactively inside the chat using `/provider`:

```bash
ireri chat
```

Then run:
```text
/provider nvidia
```
*(Prompts for your free NVIDIA API key and configures the default model)*

Alternatively, create a `.env` file in your workspace:

```env
OPENAI_API_KEY=nvapi-your-key-here
OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
OPENAI_MODEL=nvidia/nemotron-3.5-bolt-e4b
```

### 2. Start Chatting

Launch the interactive coding assistant:

```bash
ireri chat
```

Or run in auto-approve mode:

```bash
ireri chat -y
```

### 3. One-Shot Execution

Execute single commands directly from your terminal:

```bash
ireri run "Create a simple modern HTML and CSS website with dark mode"
```

---

## 🤖 Recommended Provider: NVIDIA NIM (Free Tier)

NVIDIA offers 1,000 free API credits to developers to run state-of-the-art open models:

1. **Register for an API key**: [https://build.nvidia.com/](https://build.nvidia.com/)
2. **Base URL**: `https://integrate.api.nvidia.com/v1`
3. **Recommended Models**:
   - `nvidia/nemotron-3.5-bolt-e4b` *(Fast, optimized for code generation)*
   - `meta/llama-3.3-70b-instruct` *(High reasoning capabilities)*
   - `deepseek-ai/deepseek-r1` *(Deep reasoning model)*

Setup takes seconds inside ireri:
```text
/provider nvidia
```

---

## ⌨️ Slash Commands

Keying in `/` inside `ireri chat` will automatically bring up the command completion menu:

| Command | Description |
| :--- | :--- |
| `/provider` | View active provider, list profiles, and configure new providers |
| `/provider nvidia` | Quick 1-step setup for NVIDIA NIM free models |
| `/provider add` | Guided wizard to add any AI provider |
| `/provider use <name>` | Switch between configured provider profiles |
| `/model` | Switch the active AI model |
| `/models` | List all available models from the current provider |
| `/settings` | Open interactive settings and API profiles |
| `/config` | View current agent configuration |
| `/status` | Show current agent runtime status |
| `/clear` | Clear the terminal conversation view |
| `/reset` | Reset the agent session context |
| `/version` | Show agent version information |
| `/update` | Check for updates |
| `/about` | Information about IRERI CODE |
| `/exit` | Exit the session cleanly (`/quit`, `/q`) |

---

## 🛠️ CLI Reference

```text
Usage: ireri [OPTIONS] COMMAND [ARGS]...

Commands:
  chat         Start an interactive chat session with ireri.
    --project, -p         Project directory (default: current directory)
    --model, -m           Model identifier to use
    --base-url, -u        API base URL
    --api-key, -k         API key
    --max-iterations, -i  Maximum loop iterations per task [default: 50]
    --yes, -y             Auto-approve all tool calls

  run          Execute a one-shot task with ireri.
    --project, -p         Project directory
    --model, -m           Model identifier to use
    --max-iterations, -i  Maximum loop iterations [default: 50]
    --json, -j            Output result as JSON

  tools        List all available built-in tools.
  config-show  Show current configuration settings.
  init         Initialize a new project with .env configuration.
```

---

## 🧪 Running Tests

To run the test suite:

```bash
pytest ultra-agent
```

---

## 📄 License

MIT License. Created by the ireri team.
