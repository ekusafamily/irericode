"""Terminal/command execution tools for the Ultra Agent."""

import asyncio
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from . import Tool, ToolArgument, ToolResult, register_tool
from agent.config import get_config

import logging

logger = logging.getLogger(__name__)


# Dangerous command patterns that require extra approval
DANGEROUS_PATTERNS = [
    r"rm\s+-rf\s+/",
    r"rm\s+-rf\s+~",
    r"sudo\s+",
    r"chmod\s+777",
    r">\s*/dev/",
    r"curl\s+.*\|\s*sh",
    r"wget\s+.*\|\s*sh",
    r"pip\s+install\s+.*--break-system-packages",
    r"npm\s+install\s+-g",
    r"docker\s+",
    r"kubectl\s+",
    r"systemctl\s+",
    r"service\s+",
    r"mount\s+",
    r"dd\s+if=",
    r"mkfs\s+",
    r"fdisk\s+",
    r":\(\)\{\s*:\|\:&\s*\};:",  # fork bomb
]


def _is_dangerous(command: str) -> bool:
    """Check if a command matches dangerous patterns."""
    import re
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False


def _resolve_workspace_path(path: str) -> Path:
    """Resolve a path relative to workspace root."""
    config = get_config()
    workspace = config.workspace_root

    # Normalize path: handle Unix-style Windows paths like /c/..., /d/...
    # Also handle special paths like /workspace
    normalized_path = path
    if path == "/workspace" or path == "/workspace/":
        normalized_path = "."
    elif path.startswith('/') and len(path) >= 3 and path[2] == '/':
        # Looks like /c/..., /d/... - Unix-style Windows path
        # Strip the leading /c/ and treat as relative
        normalized_path = path[3:]

        # If the normalized path equals the workspace root path, use "."
        workspace_str = str(workspace).replace('\\', '/')
        if normalized_path == workspace_str or normalized_path == workspace_str.rstrip('/'):
            normalized_path = "."

    target = (workspace / normalized_path).resolve()

    try:
        target.relative_to(workspace)
    except ValueError:
        raise ValueError(f"Path '{path}' attempts to escape workspace")

    return target


class RunCommandArgs(ToolArgument):
    """Arguments for running a command."""
    command: str = Field(description="Command to execute")
    cwd: Optional[str] = Field(default=None, description="Working directory (relative to workspace)")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds")
    shell: bool = Field(default=True, description="Run through shell")


class RunCommandTool(Tool):
    """Execute a terminal command."""

    name = "run_command"
    description = "Execute a shell command in the workspace"
    args_schema = RunCommandArgs
    requires_approval = True

    async def execute(self, args: RunCommandArgs) -> ToolResult:
        try:
            config = get_config()

            # Determine working directory
            if args.cwd:
                cwd = _resolve_workspace_path(args.cwd)
            else:
                cwd = config.workspace_root

            # Check for dangerous commands
            if _is_dangerous(args.command):
                return ToolResult(
                    success=False,
                    error="Command matches dangerous pattern. Requires explicit approval.",
                    metadata={"dangerous": True}
                )

            # Set timeout
            timeout = args.timeout or config.command_timeout

            logger.info(f"Executing command in {cwd}: {args.command}")

            # Run command
            process = await asyncio.create_subprocess_shell(
                args.command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return ToolResult(
                    success=False,
                    error=f"Command timed out after {timeout} seconds",
                    metadata={"timeout": True}
                )

            stdout_text = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            return ToolResult(
                success=process.returncode == 0,
                output={
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "returncode": process.returncode,
                    "command": args.command,
                },
                error=stderr_text if process.returncode != 0 else None,
                metadata={
                    "returncode": process.returncode,
                    "cwd": str(cwd),
                }
            )
        except Exception as e:
            logger.error(f"run_command error: {e}")
            return ToolResult(success=False, error=str(e))


# Register terminal tool
register_tool(RunCommandTool())