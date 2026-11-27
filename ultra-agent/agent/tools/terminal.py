"""Terminal/command execution tools for ireri."""

import asyncio
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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


def _kill_process_tree(process: asyncio.subprocess.Process) -> None:
    """Terminate a process and all its children cleanly across Windows and POSIX."""
    if process.returncode is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        else:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception:
                process.kill()
    except Exception:
        try:
            process.kill()
        except Exception:
            pass


class ManagedProcess:
    """Represents a running or recently finished subprocess with live log buffering."""

    def __init__(
        self,
        process_id: str,
        command: str,
        cwd: str,
        process: asyncio.subprocess.Process,
        background: bool = False,
    ):
        self.process_id = process_id
        self.command = command
        self.cwd = cwd
        self.process = process
        self.background = background
        self.pid = process.pid
        self.started_at = time.monotonic()
        self.started_wall = datetime.now()
        self.ended_at: Optional[float] = None
        self.stdout_lines: deque[str] = deque(maxlen=2000)
        self.stderr_lines: deque[str] = deque(maxlen=2000)
        self.combined_logs: deque[str] = deque(maxlen=4000)
        self._reader_tasks: List[asyncio.Task] = []
        self._done_event = asyncio.Event()
        self._monitor_task: Optional[asyncio.Task] = None

    @property
    def returncode(self) -> Optional[int]:
        return self.process.returncode

    @property
    def is_running(self) -> bool:
        return self.process.returncode is None

    @property
    def uptime_seconds(self) -> float:
        end = self.ended_at if self.ended_at is not None else time.monotonic()
        return max(0.0, end - self.started_at)

    def start_readers(self) -> None:
        """Start background async stream readers for live stdout/stderr capture."""
        async def _read_stream(stream: Optional[asyncio.StreamReader], target: deque, prefix: str = ""):
            if stream is None:
                return
            try:
                while True:
                    line_bytes = await stream.readline()
                    if not line_bytes:
                        break
                    text = line_bytes.decode("utf-8", errors="replace").rstrip("\r\n")
                    target.append(text)
                    self.combined_logs.append(f"{prefix}{text}")
            except Exception:
                pass

        async def _monitor():
            try:
                await self.process.wait()
                if self._reader_tasks:
                    await asyncio.gather(*self._reader_tasks, return_exceptions=True)
            finally:
                if self.ended_at is None:
                    self.ended_at = time.monotonic()
                self._done_event.set()

        self._reader_tasks = [
            asyncio.create_task(_read_stream(self.process.stdout, self.stdout_lines, "")),
            asyncio.create_task(_read_stream(self.process.stderr, self.stderr_lines, "[stderr] ")),
        ]
        self._monitor_task = asyncio.create_task(_monitor())

    async def wait_done(self) -> int:
        """Wait for the process and all stream readers to finish."""
        await self._done_event.wait()
        return self.process.returncode if self.process.returncode is not None else -1

    def get_stdout_text(self) -> str:
        return "\n".join(self.stdout_lines)

    def get_stderr_text(self) -> str:
        return "\n".join(self.stderr_lines)

    def get_recent_logs(self, tail: int = 100) -> str:
        """Return the last `tail` combined log lines."""
        if tail <= 0:
            return "\n".join(self.combined_logs)
        lines = list(self.combined_logs)[-tail:]
        return "\n".join(lines)


class ProcessManager:
    """Tracks ongoing and recent commands so they can run in background and stream logs."""

    def __init__(self):
        self._processes: Dict[str, ManagedProcess] = {}
        self._counter = 0

    async def start_process(
        self,
        command: str,
        cwd: Path,
        background: bool = False,
    ) -> ManagedProcess:
        """Start a subprocess and begin live log streaming."""
        self._counter += 1
        process_id = f"proc-{self._counter}"

        kwargs: Dict[str, Any] = {
            "cwd": cwd,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.PIPE,
        }
        if sys.platform != "win32":
            kwargs["start_new_session"] = True

        process = await asyncio.create_subprocess_shell(command, **kwargs)
        managed = ManagedProcess(
            process_id=process_id,
            command=command,
            cwd=str(cwd),
            process=process,
            background=background,
        )
        managed.start_readers()
        self._processes[process_id] = managed
        return managed

    def get_process(self, process_id: Optional[str] = None) -> Optional[ManagedProcess]:
        """Get a process by ID, or the most recent running/available process if omitted."""
        if process_id:
            return self._processes.get(process_id)
        if not self._processes:
            return None
        running = [p for p in self._processes.values() if p.is_running]
        if running:
            return running[-1]
        return list(self._processes.values())[-1]

    def list_processes(self) -> List[ManagedProcess]:
        """List all tracked processes."""
        return list(self._processes.values())

    def get_logs(self, process_id: Optional[str] = None, tail: int = 100) -> Tuple[Optional[ManagedProcess], str]:
        """Get recent logs for a process."""
        proc = self.get_process(process_id)
        if not proc:
            return None, ""
        return proc, proc.get_recent_logs(tail=tail)

    async def stop_process(self, process_id: str) -> Tuple[bool, str]:
        """Stop a running process by ID."""
        proc = self._processes.get(process_id)
        if not proc:
            return False, f"Unknown process ID: {process_id}"
        if not proc.is_running:
            return True, f"Process {process_id} already exited with code {proc.returncode}"

        _kill_process_tree(proc.process)
        try:
            await asyncio.wait_for(proc.wait_done(), timeout=3.0)
        except asyncio.TimeoutError:
            pass
        return True, f"Stopped process {process_id} (PID {proc.pid})"

    def stop_all(self) -> int:
        """Synchronously terminate all still-running processes (used on CLI exit)."""
        stopped = 0
        for proc in self._processes.values():
            if proc.is_running:
                _kill_process_tree(proc.process)
                stopped += 1
        return stopped


_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """Get the global ProcessManager instance."""
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager


class RunCommandArgs(ToolArgument):
    """Arguments for running a command."""
    command: str = Field(description="Command to execute")
    cwd: Optional[str] = Field(default=None, description="Working directory (relative to workspace)")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds for foreground commands")
    shell: bool = Field(default=True, description="Run through shell")
    background: bool = Field(
        default=False,
        description="Set true for long-running processes (bots, servers, watchers like 'python bot.py'). Returns immediately with process_id and initial logs."
    )
    wait_seconds: float = Field(
        default=2.0,
        description="When background=true, seconds to wait to capture initial startup logs (default 2.0s)"
    )


class RunCommandTool(Tool):
    """Execute a terminal command (foreground or background)."""

    name = "run_command"
    description = (
        "Execute a shell command in the workspace. For long-running processes like bots, servers, "
        "or watchers (e.g., 'python bot.py', 'npm run dev'), set background=true so it returns "
        "immediately with a process_id instead of blocking."
    )
    args_schema = RunCommandArgs
    requires_approval = True
    dangerous = True

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

            timeout = args.timeout or config.command_timeout
            cmd_lower = args.command.lower()
            is_installer_cmd = any(
                kw in cmd_lower
                for kw in ("pip install", "npm install", "npm i ", "yarn install", "pnpm install", "cargo build", "cargo check")
            )
            if args.timeout is None and is_installer_cmd:
                timeout = max(timeout, 300)
            pm = get_process_manager()

            logger.info(f"Executing command in {cwd} (background={args.background}): {args.command}")
            managed = await pm.start_process(args.command, cwd=cwd, background=args.background)

            if args.background:
                startup_wait = min(max(args.wait_seconds, 0.2), float(timeout))
                try:
                    await asyncio.wait_for(managed.wait_done(), timeout=startup_wait)
                    # Process exited during startup window
                    stdout_text = managed.get_stdout_text()
                    stderr_text = managed.get_stderr_text()
                    rc = managed.returncode
                    error_msg = None
                    if rc != 0:
                        error_msg = stderr_text.strip() or stdout_text.strip() or f"Command '{args.command}' exited immediately with code {rc}"
                    return ToolResult(
                        success=rc == 0,
                        output={
                            "process_id": managed.process_id,
                            "pid": managed.pid,
                            "status": "exited",
                            "stdout": stdout_text,
                            "stderr": stderr_text,
                            "returncode": rc,
                            "command": args.command,
                        },
                        error=error_msg,
                        metadata={
                            "background": True,
                            "process_id": managed.process_id,
                            "returncode": rc,
                            "cwd": str(cwd),
                        },
                    )
                except asyncio.TimeoutError:
                    # Still running in background as expected!
                    initial_logs = managed.get_recent_logs(tail=50)
                    return ToolResult(
                        success=True,
                        output={
                            "process_id": managed.process_id,
                            "pid": managed.pid,
                            "status": "running",
                            "command": args.command,
                            "initial_logs": initial_logs,
                            "message": (
                                f"Process '{managed.process_id}' (PID {managed.pid}) is running in the background. "
                                f"Use get_process_logs(process_id='{managed.process_id}') to inspect logs or stop_process to stop it."
                            ),
                        },
                        metadata={
                            "background": True,
                            "process_id": managed.process_id,
                            "pid": managed.pid,
                            "cwd": str(cwd),
                        },
                    )

            # Foreground command
            try:
                await asyncio.wait_for(managed.wait_done(), timeout=timeout)
            except asyncio.TimeoutError:
                partial_logs = managed.get_recent_logs(tail=50)
                await pm.stop_process(managed.process_id)
                return ToolResult(
                    success=False,
                    error=(
                        f"Command timed out after {timeout} seconds."
                        + (
                            " Package installation took longer than expected; re-run with a higher timeout."
                            if is_installer_cmd
                            else f" If '{args.command}' is a long-running bot, server, or watcher, re-run with background=true."
                        )
                    ),
                    output={
                        "process_id": managed.process_id,
                        "stdout": managed.get_stdout_text(),
                        "stderr": managed.get_stderr_text(),
                        "partial_logs": partial_logs,
                        "command": args.command,
                    },
                    metadata={"timeout": True, "process_id": managed.process_id}
                )

            stdout_text = managed.get_stdout_text()
            stderr_text = managed.get_stderr_text()
            rc = managed.returncode

            error_msg = None
            if rc != 0:
                err_s = stderr_text.strip()
                out_s = stdout_text.strip()
                if err_s and out_s and out_s != err_s:
                    # Include stdout tail so warnings on stderr (e.g. pytest) don't hide the real error
                    error_msg = f"{err_s}\n\n{out_s[-1500:]}"
                else:
                    error_msg = err_s or out_s or f"Command '{args.command}' failed with exit code {rc}"

            return ToolResult(
                success=rc == 0,
                output={
                    "process_id": managed.process_id,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                    "returncode": rc,
                    "command": args.command,
                },
                error=error_msg,
                metadata={
                    "process_id": managed.process_id,
                    "returncode": rc,
                    "cwd": str(cwd),
                }
            )
        except Exception as e:
            logger.error(f"run_command error: {e}")
            return ToolResult(success=False, error=str(e))


class ListProcessesArgs(ToolArgument):
    """Arguments for listing ongoing/recent processes."""
    pass


class ListProcessesTool(Tool):
    """List ongoing and recent commands/processes started by ireri."""

    name = "list_processes"
    description = "List all ongoing and recent terminal processes with their process_id, PID, status, uptime, and command"
    args_schema = ListProcessesArgs
    requires_approval = False

    async def execute(self, args: ListProcessesArgs) -> ToolResult:
        pm = get_process_manager()
        items = []
        for proc in pm.list_processes():
            items.append({
                "process_id": proc.process_id,
                "pid": proc.pid,
                "status": "running" if proc.is_running else f"exited ({proc.returncode})",
                "background": proc.background,
                "uptime_seconds": round(proc.uptime_seconds, 1),
                "command": proc.command,
                "cwd": proc.cwd,
            })
        return ToolResult(success=True, output={"processes": items, "count": len(items)})


class GetProcessLogsArgs(ToolArgument):
    """Arguments for retrieving logs from an ongoing or finished process."""
    process_id: Optional[str] = Field(default=None, description="Process ID (e.g. 'proc-1'). Defaults to latest running/recent process.")
    tail: int = Field(default=100, description="Number of recent log lines to return (0 for all buffered lines)")


class GetProcessLogsTool(Tool):
    """Get live stdout/stderr logs of an ongoing or finished process."""

    name = "get_process_logs"
    description = "Read the latest stdout/stderr logs of an ongoing background process or recent command"
    args_schema = GetProcessLogsArgs
    requires_approval = False

    async def execute(self, args: GetProcessLogsArgs) -> ToolResult:
        pm = get_process_manager()
        proc, logs = pm.get_logs(process_id=args.process_id, tail=args.tail)
        if not proc:
            available = [p.process_id for p in pm.list_processes()]
            hint = f"Available processes: {', '.join(available)}" if available else "No processes have been started yet."
            return ToolResult(
                success=False,
                error=f"Process not found: {args.process_id or 'none'}. {hint}",
            )
        return ToolResult(
            success=True,
            output={
                "process_id": proc.process_id,
                "pid": proc.pid,
                "status": "running" if proc.is_running else f"exited ({proc.returncode})",
                "uptime_seconds": round(proc.uptime_seconds, 1),
                "command": proc.command,
                "logs": logs,
                "total_buffered_lines": len(proc.combined_logs),
            },
        )


class StopProcessArgs(ToolArgument):
    """Arguments for stopping a running process."""
    process_id: str = Field(description="Process ID to stop (e.g. 'proc-1')")


class StopProcessTool(Tool):
    """Stop an ongoing background process."""

    name = "stop_process"
    description = "Stop/terminate a running background process by process_id"
    args_schema = StopProcessArgs
    requires_approval = True

    async def execute(self, args: StopProcessArgs) -> ToolResult:
        pm = get_process_manager()
        ok, msg = await pm.stop_process(args.process_id)
        return ToolResult(success=ok, output={"message": msg} if ok else None, error=None if ok else msg)


# Register terminal tools
register_tool(RunCommandTool())
register_tool(ListProcessesTool())
register_tool(GetProcessLogsTool())
register_tool(StopProcessTool())