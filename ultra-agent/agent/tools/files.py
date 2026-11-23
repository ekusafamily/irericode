"""File system tools for ireri."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from . import Tool, ToolArgument, ToolResult, register_tool, get_registry
from agent.config import get_config

import logging

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
    """Resolve a path relative to workspace root, preventing traversal."""
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

    # Convert to Path and resolve
    target = (workspace / normalized_path).resolve()

    # Ensure the target is within workspace
    try:
        target.relative_to(workspace)
    except ValueError:
        raise ValueError(f"Path '{path}' attempts to escape workspace")

    return target


class ListFilesArgs(ToolArgument):
    """Arguments for listing files."""
    path: str = Field(default=".", description="Directory path to list")
    recursive: bool = Field(default=False, description="List recursively")


class ListFilesTool(Tool):
    """List files in a directory."""

    name = "list_files"
    description = "List files and directories in a given path"
    args_schema = ListFilesArgs

    async def execute(self, args: ListFilesArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)

            if not target.exists():
                return ToolResult(success=False, error=f"Path does not exist: {args.path}")

            if not target.is_dir():
                return ToolResult(success=False, error=f"Path is not a directory: {args.path}")

            files = []
            if args.recursive:
                for item in target.rglob("*"):
                    rel = item.relative_to(get_config().workspace_root)
                    files.append({
                        "path": str(rel),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })
            else:
                for item in target.iterdir():
                    rel = item.relative_to(get_config().workspace_root)
                    files.append({
                        "path": str(rel),
                        "type": "directory" if item.is_dir() else "file",
                        "size": item.stat().st_size if item.is_file() else None,
                    })

            return ToolResult(
                success=True,
                output={"files": files, "path": args.path},
                metadata={"count": len(files)}
            )
        except Exception as e:
            logger.error(f"list_files error: {e}")
            return ToolResult(success=False, error=str(e))


class ReadFileArgs(ToolArgument):
    """Arguments for reading a file."""
    path: str = Field(description="File path to read")
    offset: Optional[int] = Field(default=None, description="Line offset to start reading from")
    limit: Optional[int] = Field(default=None, description="Maximum number of lines to read")


class ReadFileTool(Tool):
    """Read a file's contents."""

    name = "read_file"
    description = "Read the contents of a file"
    args_schema = ReadFileArgs

    async def execute(self, args: ReadFileArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)

            if not target.exists():
                return ToolResult(success=False, error=f"File does not exist: {args.path}")

            if not target.is_file():
                return ToolResult(success=False, error=f"Path is not a file: {args.path}")

            # Check file size (limit to 1MB for safety)
            if target.stat().st_size > 1_000_000:
                return ToolResult(
                    success=False,
                    error=f"File too large ({target.stat().st_size} bytes). Max 1MB."
                )

            content = target.read_text(encoding="utf-8")
            lines = content.splitlines()

            # Apply offset and limit
            start = args.offset or 0
            end = start + (args.limit or len(lines))
            selected_lines = lines[start:end]

            return ToolResult(
                success=True,
                output={
                    "path": args.path,
                    "content": "\n".join(selected_lines),
                    "total_lines": len(lines),
                    "start_line": start + 1,
                    "end_line": min(end, len(lines)),
                },
                metadata={"size": target.stat().st_size}
            )
        except UnicodeDecodeError:
            return ToolResult(success=False, error="File is not valid UTF-8 text")
        except Exception as e:
            logger.error(f"read_file error: {e}")
            return ToolResult(success=False, error=str(e))


class WriteFileArgs(ToolArgument):
    """Arguments for writing a file."""
    path: str = Field(description="File path to write")
    content: str = Field(description="Content to write")
    create_dirs: bool = Field(default=True, description="Create parent directories if needed")


class WriteFileTool(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Create a new file or overwrite an existing file"
    args_schema = WriteFileArgs
    requires_approval = True

    async def execute(self, args: WriteFileArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)

            if args.create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)

            target.write_text(args.content, encoding="utf-8")

            return ToolResult(
                success=True,
                output={"path": args.path, "bytes_written": len(args.content.encode("utf-8"))},
                metadata={"action": "created" if not target.exists() else "overwritten"}
            )
        except Exception as e:
            logger.error(f"write_file error: {e}")
            return ToolResult(success=False, error=str(e))


class EditFileArgs(ToolArgument):
    """Arguments for editing a file."""
    path: str = Field(description="File path to edit")
    old_string: str = Field(description="Exact string to replace")
    new_string: str = Field(description="New string to insert")
    replace_all: bool = Field(default=False, description="Replace all occurrences")


class EditFileTool(Tool):
    """Edit a file by replacing a specific string."""

    name = "edit_file"
    description = "Make a targeted edit to a file by replacing old_string with new_string"
    args_schema = EditFileArgs
    requires_approval = True

    async def execute(self, args: EditFileArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)

            if not target.exists():
                return ToolResult(success=False, error=f"File does not exist: {args.path}")

            content = target.read_text(encoding="utf-8")

            if args.old_string not in content:
                return ToolResult(
                    success=False,
                    error=f"String not found in file: {args.old_string[:100]}..."
                )

            if args.replace_all:
                new_content = content.replace(args.old_string, args.new_string)
                count = content.count(args.old_string)
            else:
                count = content.count(args.old_string)
                if count > 1:
                    return ToolResult(
                        success=False,
                        error=f"String appears {count} times. Use replace_all=true or provide more context."
                    )
                new_content = content.replace(args.old_string, args.new_string, 1)

            target.write_text(new_content, encoding="utf-8")

            return ToolResult(
                success=True,
                output={"path": args.path, "replacements": count},
                metadata={"action": "edited"}
            )
        except Exception as e:
            logger.error(f"edit_file error: {e}")
            return ToolResult(success=False, error=str(e))


class DeleteFileArgs(ToolArgument):
    """Arguments for deleting a file."""
    path: str = Field(description="File or directory path to delete")
    recursive: bool = Field(default=False, description="Delete directories recursively")


class DeleteFileTool(Tool):
    """Delete a file or directory."""

    name = "delete_file"
    description = "Delete a file or directory"
    args_schema = DeleteFileArgs
    requires_approval = True
    dangerous = True

    async def execute(self, args: DeleteFileArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)

            if not target.exists():
                return ToolResult(success=False, error=f"Path does not exist: {args.path}")

            if target.is_dir():
                if not args.recursive:
                    return ToolResult(success=False, error="Directory not empty. Use recursive=true.")
                import shutil
                shutil.rmtree(target)
                action = "deleted_directory"
            else:
                target.unlink()
                action = "deleted_file"

            return ToolResult(
                success=True,
                output={"path": args.path},
                metadata={"action": action}
            )
        except Exception as e:
            logger.error(f"delete_file error: {e}")
            return ToolResult(success=False, error=str(e))


class CreateDirectoryArgs(ToolArgument):
    """Arguments for creating a directory."""
    path: str = Field(description="Directory path to create")
    parents: bool = Field(default=True, description="Create parent directories")


class CreateDirectoryTool(Tool):
    """Create a directory."""

    name = "create_directory"
    description = "Create a new directory"
    args_schema = CreateDirectoryArgs
    requires_approval = True

    async def execute(self, args: CreateDirectoryArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)
            target.mkdir(parents=args.parents, exist_ok=True)

            return ToolResult(
                success=True,
                output={"path": args.path},
                metadata={"action": "created"}
            )
        except Exception as e:
            logger.error(f"create_directory error: {e}")
            return ToolResult(success=False, error=str(e))


# Register all file tools
register_tool(ListFilesTool())
register_tool(ReadFileTool())
register_tool(WriteFileTool())
register_tool(EditFileTool())
register_tool(DeleteFileTool())
register_tool(CreateDirectoryTool())