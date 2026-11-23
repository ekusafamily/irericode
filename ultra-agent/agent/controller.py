"""Main agent controller for ireri."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.config import get_config
from agent.model import get_model_client, ModelClient, ModelResponse
from agent.context import ContextManager
from agent.prompts import get_system_prompt, get_task_prompt, get_continuation_prompt, get_error_recovery_prompt
from agent.tools import get_registry, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# Progress callback type
ProgressCallback = Callable[[str, Dict[str, Any]], None]

# Approval callback type
ApprovalCallback = Callable[[str, Dict[str, Any], bool], bool]


class AgentController:
    """Main agent control loop."""

    def __init__(
        self,
        model_client: Optional[ModelClient] = None,
        tool_registry: Optional[ToolRegistry] = None,
        context: Optional[ContextManager] = None,
        model: Optional[str] = None,
        progress_callback: Optional[ProgressCallback] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ):
        from agent.config import reset_config
        from agent.model import reset_model_client, ModelClient

        reset_config()
        reset_model_client()
        self.config = get_config()

        # Use provided model or config model (avoids shell env var override)
        effective_model = model or self.config.openai_model
        self.model_client = model_client or ModelClient(model=effective_model)
        self.tool_registry = tool_registry or get_registry()
        self.context = context or ContextManager()
        self.progress_callback = progress_callback
        self.approval_callback = approval_callback

        # Add system prompt
        self.context.add_system_message(get_system_prompt())

        # State
        self.running = False
        self.cancelled = False
        self.last_error: Optional[str] = None

        # Approval state
        self._approve_all = False
        self._deny_all = False

    def reset(self) -> None:
        """Reset the controller state for a new session."""
        self.context.clear()
        self.context.add_system_message(get_system_prompt())
        self.running = False
        self.cancelled = False
        self.last_error = None
        self._approve_all = False
        self._deny_all = False

    async def run_task(self, task: str, project_context: str = "", progress_callback: Optional[ProgressCallback] = None) -> str:
        """Run a complete task and return the result."""
        self.running = True
        self.cancelled = False
        if progress_callback:
            self.progress_callback = progress_callback

        # Add task to context
        prompt = get_task_prompt(task, project_context)
        self.context.add_user_message(prompt)

        try:
            result = await self._agent_loop()
            return result
        finally:
            self.running = False

    async def _agent_loop(self) -> str:
        """Main agent loop."""
        config = self.config
        max_iterations = config.max_iterations
        max_tool_calls = config.max_tool_calls

        while self.running and not self.cancelled:
            # Check limits
            if self.context.get_iteration_count() >= max_iterations:
                return f"Stopped: Maximum iterations ({max_iterations}) reached"

            if self.context.get_tool_calls_count() >= max_tool_calls:
                return f"Stopped: Maximum tool calls ({max_tool_calls}) reached"

            # Increment iteration
            iteration = self.context.increment_iteration()
            logger.info(f"Agent iteration {iteration}/{max_iterations}")

            # Emit progress: iteration started
            if self.progress_callback:
                self.progress_callback("iteration_start", {
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                })

            # Get messages for model
            messages = self.context.get_messages_for_model()

            # Get tool definitions
            tools = self.tool_registry.get_definitions()

            # Call model
            try:
                response = await self.model_client.chat(
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=0.1,
                )
            except Exception as e:
                logger.error(f"Model call failed: {e}")
                self.last_error = str(e)
                if self.progress_callback:
                    self.progress_callback("model_error", {"error": str(e)})
                # Try to recover
                error_prompt = get_error_recovery_prompt(str(e))
                self.context.add_user_message(error_prompt)
                continue

            logger.info(f"Model response: content={response.content[:100] if response.content else None}, tool_calls={len(response.tool_calls) if response.tool_calls else 0}")

            # Emit progress: model response
            if self.progress_callback:
                self.progress_callback("model_response", {
                    "has_content": bool(response.content),
                    "tool_calls_count": len(response.tool_calls) if response.tool_calls else 0,
                    "finish_reason": response.finish_reason,
                })

            # Handle response
            if response.tool_calls:
                # Add assistant message with tool calls
                self.context.add_assistant_message(
                    content=response.content,
                    tool_calls=response.tool_calls,
                )

                # Execute tool calls
                for tool_call in response.tool_calls:
                    if self.cancelled:
                        break

                    result = await self._execute_tool_call(tool_call)

                    # Add tool result to context
                    self.context.add_tool_result(
                        tool_call_id=tool_call["id"],
                        name=tool_call["function"]["name"],
                        content=json.dumps(result.output) if result.success else result.error or "Tool failed",
                    )

                    # Check if tool failed
                    if not result.success:
                        self.last_error = result.error
                        logger.warning(f"Tool {tool_call['function']['name']} failed: {result.error}")

            elif response.content:
                # Regular response - add to context and continue
                self.context.add_assistant_message(content=response.content)

                # Check if task seems complete
                if self._is_task_complete(response.content):
                    return response.content

                # Continue with continuation prompt
                self.context.add_user_message(get_continuation_prompt())
            else:
                # Empty response
                logger.warning("Empty response from model")
                self.context.add_user_message(get_continuation_prompt())

        if self.cancelled:
            return "Task cancelled by user"

        return "Agent loop ended without completion"

    async def _execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """Execute a single tool call."""
        tool_name = tool_call["function"]["name"]
        tool = self.tool_registry.get(tool_name)

        if not tool:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {tool_name}"
            )

        # Parse arguments
        try:
            args_json = tool_call["function"]["arguments"]
            if isinstance(args_json, str):
                args_dict = json.loads(args_json)
            else:
                args_dict = args_json
            args = tool.args_schema(**args_dict)
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Invalid tool arguments: {e}"
            )

        # Emit progress: tool started
        if self.progress_callback:
            self.progress_callback("tool_start", {
                "tool": tool_name,
                "args": args_dict,
            })

        # Check approval
        if tool.requires_approval:
            approved = await self._request_approval(tool_name, args_dict, tool.dangerous)
            if not approved:
                if self.progress_callback:
                    self.progress_callback("tool_denied", {"tool": tool_name})
                return ToolResult(
                    success=False,
                    error="User denied approval"
                )

        # Execute tool
        logger.info(f"Executing tool: {tool_name} with args: {args_dict}")
        try:
            result = await tool.execute(args)
            logger.info(f"Tool {tool_name} result: success={result.success}")

            # Emit progress: tool completed
            if self.progress_callback:
                self.progress_callback("tool_complete", {
                    "tool": tool_name,
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "metadata": result.metadata,
                })

            return result
        except Exception as e:
            logger.error(f"Tool {tool_name} execution error: {e}")
            if self.progress_callback:
                self.progress_callback("tool_error", {
                    "tool": tool_name,
                    "error": str(e),
                })
            return ToolResult(success=False, error=str(e))

    async def _request_approval(self, tool_name: str, args: Dict[str, Any], dangerous: bool = False) -> bool:
        """Request user approval for a tool call."""
        # Check session-level approvals first
        if self._approve_all:
            return True
        if self._deny_all:
            return False

        # If we have an approval callback, use it
        if self.approval_callback:
            return await self.approval_callback(tool_name, args, dangerous)

        # Fallback: auto-approve in non-interactive mode
        logger.info(f"Approval requested for {tool_name}: {args}")
        return True  # Auto-approve for now

    def set_approve_all(self, value: bool) -> None:
        """Set approve-all mode for the session."""
        self._approve_all = value

    def set_deny_all(self, value: bool) -> None:
        """Set deny-all mode for the session."""
        self._deny_all = value

    def _is_task_complete(self, response: str) -> bool:
        """Heuristic to detect if task is complete."""
        completion_indicators = [
            "task complete",
            "completed successfully",
            "finished",
            "done",
            "all done",
            "is complete",
            "has been completed",
            "complete.",
        ]
        response_lower = response.lower()
        return any(indicator in response_lower for indicator in completion_indicators)

    def _is_conversational(self, message: str) -> bool:
        """Check if message is a simple conversational input (not a coding task)."""
        msg = message.strip().lower()
        conversational_patterns = [
            "hi", "hello", "hey", "hi there", "hello there",
            "thanks", "thank you", "thx", "ty",
            "bye", "goodbye", "see you", "cya",
            "how are you", "how's it going", "what's up",
            "ok", "okay", "sure", "yeah", "yep", "yup",
            "no", "nope", "nah",
            "lol", "haha", "hehe",
        ]
        # Exact match or very short message
        return msg in conversational_patterns or (len(msg) < 30 and msg in ["hi", "hello", "hey", "thanks", "bye", "ok"])

    def cancel(self) -> None:
        """Cancel the current task."""
        self.cancelled = True
        self.running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "running": self.running,
            "cancelled": self.cancelled,
            "iteration": self.context.get_iteration_count(),
            "tool_calls": self.context.get_tool_calls_count(),
            "last_error": self.last_error,
        }


async def run_agent_task(task: str, project_path: Optional[str] = None) -> str:
    """Convenience function to run a single task."""
    from agent.model import reset_model_client

    reset_model_client()
    config = get_config()

    if project_path:
        config.workspace_root = Path(project_path).resolve()

    controller = AgentController()
    return await controller.run_task(task)