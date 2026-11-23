"""Code search tools for the Ultra Agent."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from . import Tool, ToolArgument, ToolResult, register_tool
from agent.config import get_config

import logging

logger = logging.getLogger(__name__)


def _resolve_path(path: str) -> Path:
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


class SearchCodeArgs(ToolArgument):
    """Arguments for searching code."""
    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(default=".", description="Directory to search in")
    file_pattern: Optional[str] = Field(default=None, description="Glob pattern for files (e.g., '*.py')")
    case_sensitive: bool = Field(default=False, description="Case sensitive search")
    max_results: int = Field(default=50, description="Maximum results to return")


class SearchCodeTool(Tool):
    """Search for code patterns in files."""

    name = "search_code"
    description = "Search for a regex pattern in source files"
    args_schema = SearchCodeArgs

    async def execute(self, args: SearchCodeArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)

            if not target.exists():
                return ToolResult(success=False, error=f"Path does not exist: {args.path}")

            if not target.is_dir():
                return ToolResult(success=False, error=f"Path is not a directory: {args.path}")

            # Compile regex
            flags = 0 if args.case_sensitive else re.IGNORECASE
            try:
                regex = re.compile(args.pattern, flags)
            except re.error as e:
                return ToolResult(success=False, error=f"Invalid regex: {e}")

            results = []
            files_searched = 0

            # Determine file glob
            glob_pattern = args.file_pattern or "**/*"

            for file_path in target.rglob(glob_pattern):
                if not file_path.is_file():
                    continue

                # Skip binary files and large files
                try:
                    if file_path.stat().st_size > 500_000:
                        continue
                except OSError:
                    continue

                files_searched += 1

                try:
                    content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    continue  # Skip binary files
                except OSError:
                    continue

                lines = content.splitlines()
                for i, line in enumerate(lines):
                    matches = list(regex.finditer(line))
                    if matches:
                        for match in matches:
                            rel_path = file_path.relative_to(get_config().workspace_root)
                            results.append({
                                "file": str(rel_path),
                                "line": i + 1,
                                "match": line.strip(),
                                "span": match.span(),
                            })
                            if len(results) >= args.max_results:
                                break
                    if len(results) >= args.max_results:
                        break
                if len(results) >= args.max_results:
                    break

            return ToolResult(
                success=True,
                output={
                    "results": results,
                    "pattern": args.pattern,
                    "files_searched": files_searched,
                    "total_matches": len(results),
                },
                metadata={"truncated": len(results) >= args.max_results}
            )
        except Exception as e:
            logger.error(f"search_code error: {e}")
            return ToolResult(success=False, error=str(e))


class GetProjectInfoArgs(ToolArgument):
    """Arguments for getting project info."""
    path: str = Field(default=".", description="Project directory path")


class GetProjectInfoTool(Tool):
    """Get project information and structure."""

    name = "get_project_info"
    description = "Get an overview of the project structure and key files"
    args_schema = GetProjectInfoArgs

    async def execute(self, args: GetProjectInfoArgs) -> ToolResult:
        try:
            target = _resolve_path(args.path)

            if not target.exists():
                return ToolResult(success=False, error=f"Path does not exist: {args.path}")

            info = {
                "root": str(target.relative_to(get_config().workspace_root)),
                "directories": [],
                "files": [],
                "key_files": [],
                "git": False,
            }

            # Check for git
            if (target / ".git").exists():
                info["git"] = True

            # Look for key files
            key_patterns = [
                "README*", "package.json", "pyproject.toml", "setup.py",
                "requirements.txt", "Cargo.toml", "go.mod", "pom.xml",
                "tsconfig.json", "vite.config.*", "next.config.*",
                ".env*", "docker-compose*", "Dockerfile",
            ]

            for item in target.rglob("*"):
                if not item.is_file():
                    continue

                try:
                    rel = item.relative_to(target)
                except ValueError:
                    continue

                # Check if it's a key file
                name = item.name
                for pattern in key_patterns:
                    if self._match_pattern(name, pattern):
                        info["key_files"].append(str(rel))
                        break

                # Limit file listing
                if len(info["files"]) < 100:
                    info["files"].append(str(rel))

                if len(info["key_files"]) >= 20:
                    pass  # Keep collecting but don't add more

            # Get directories
            for item in target.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    info["directories"].append(item.name)

            return ToolResult(
                success=True,
                output=info,
                metadata={}
            )
        except Exception as e:
            logger.error(f"get_project_info error: {e}")
            return ToolResult(success=False, error=str(e))

    def _match_pattern(self, name: str, pattern: str) -> bool:
        """Simple glob pattern matching."""
        import fnmatch
        return fnmatch.fnmatch(name, pattern)


# Register search tools
register_tool(SearchCodeTool())
register_tool(GetProjectInfoTool())