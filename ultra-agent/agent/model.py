"""Model client for the Ultra Agent."""

import os
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic import BaseModel

from agent.config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ModelResponse:
    """Response from the model."""
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None
    raw: Any = None


class ModelClient:
    """Client for interacting with OpenAI-compatible API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        config = get_config()

        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        # Use explicitly passed model; caller handles priority resolution
        self.model = model if model is not None else config.openai_model

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY in .env")

        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        logger.info(f"Model client initialized: {self.model} @ {self.base_url}")

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> ModelResponse:
        """Send a chat completion request."""
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            }

            if max_tokens:
                kwargs["max_tokens"] = max_tokens

            if tools:
                kwargs["tools"] = tools
                if tool_choice:
                    kwargs["tool_choice"] = tool_choice

            if stream:
                return await self._stream_chat(**kwargs)
            else:
                response = await self.client.chat.completions.create(**kwargs)
                return self._parse_response(response)

        except Exception as e:
            logger.error(f"Model API error: {e}")
            raise

    async def _stream_chat(self, **kwargs) -> ModelResponse:
        """Handle streaming response."""
        # For now, use non-streaming and simulate
        # Full streaming implementation would accumulate chunks
        response = await self.client.chat.completions.create(**kwargs, stream=False)
        return self._parse_response(response)

    def _parse_response(self, response) -> ModelResponse:
        """Parse OpenAI response into ModelResponse."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return ModelResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=usage,
            raw=response,
        )

    async def close(self):
        """Close the client."""
        await self.client.close()


# Global client instance
_client: Optional[ModelClient] = None


def get_model_client() -> ModelClient:
    """Get the global model client instance."""
    global _client
    if _client is None:
        _client = ModelClient()
    return _client


def set_model_client(client: ModelClient) -> None:
    """Set the global model client instance."""
    global _client
    _client = client


def reset_model_client() -> None:
    """Reset the global model client instance (useful after config changes)."""
    global _client
    _client = None