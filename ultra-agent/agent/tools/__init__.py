"""Tool system base classes and registry."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


class ToolResult(BaseModel):
    """Result of a tool execution."""
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ToolArgument(BaseModel):
    """Base class for tool arguments."""
    pass


class Tool(ABC):
    """Base class for all tools."""

    name: str = ""
    description: str = ""
    args_schema: Type[ToolArgument] = ToolArgument
    requires_approval: bool = False
    dangerous: bool = False

    @abstractmethod
    async def execute(self, args: ToolArgument) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def format_for_model(self) -> Dict[str, Any]:
        """Format tool definition for model consumption."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema(),
            },
        }


class ToolRegistry:
    """Registry for managing available tools."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            logger.warning(f"Tool '{tool.name}' already registered, overwriting")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> List[Tool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> List[Dict[str, Any]]:
        """Get all tool definitions formatted for the model."""
        return [tool.format_for_model() for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered."""
        return name in self._tools


# Global registry instance
_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get the global tool registry."""
    return _registry


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    _registry.register(tool)


# Import tool modules to trigger registration
from . import files, terminal, search  # noqa: F401