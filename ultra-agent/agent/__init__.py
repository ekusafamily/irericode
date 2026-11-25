"""ireri - A terminal-based AI coding agent."""

from agent.config import AgentConfig, get_config, set_config
from agent.model import ModelClient, get_model_client, set_model_client
from agent.context import ContextManager
from agent.controller import AgentController, run_agent_task
from agent.tools import Tool, ToolRegistry, get_registry, register_tool, ToolResult, ToolArgument

__version__ = "2.1.2"

__all__ = [
    "AgentConfig",
    "get_config",
    "set_config",
    "ModelClient",
    "get_model_client",
    "set_model_client",
    "ContextManager",
    "AgentController",
    "run_agent_task",
    "Tool",
    "ToolRegistry",
    "get_registry",
    "register_tool",
    "ToolResult",
    "ToolArgument",
]