"""Model client for ireri."""

import os
import time
import asyncio
import logging
from collections import deque
from typing import Any, Callable, Dict, List, Optional, AsyncGenerator
from dataclasses import dataclass

from openai import AsyncOpenAI
from pydantic import BaseModel

from agent.config import get_config, MODEL_RATE_LIMITS_RPM

logger = logging.getLogger(__name__)

# Incremental retry delays in seconds: 1s, 5s, 15s, 30s, 1m (60s), 3m (180s)
RETRY_DELAYS = [1, 5, 15, 30, 60, 180]

# Maximum time to wait for a model response before diagnosing & reloading connection (3 mins = 180s)
MODEL_CALL_TIMEOUT = 180.0


class ModelRateLimiter:
    """Per-model sliding-window rate limiter (calls per minute)."""

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._calls: Dict[str, deque] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def get_rpm_limit(model_name: str) -> Optional[int]:
        """Return configured calls-per-minute limit for a model, or None if unlimited."""
        if not model_name:
            return None
        normalized = model_name.strip().lower()
        if normalized.startswith("models/"):
            normalized = normalized[len("models/"):]
        if normalized in MODEL_RATE_LIMITS_RPM:
            return MODEL_RATE_LIMITS_RPM[normalized]
        for key, limit in MODEL_RATE_LIMITS_RPM.items():
            if normalized.startswith(key):
                return limit
        return None

    async def acquire(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> float:
        """Wait if the model's per-minute limit is reached until the next minute window opens.

        Returns the total seconds waited.
        """
        rpm_limit = self.get_rpm_limit(model_name)
        if rpm_limit is None or rpm_limit <= 0:
            return 0.0

        key = model_name.strip().lower()
        if key.startswith("models/"):
            key = key[len("models/"):]

        total_waited = 0.0
        while True:
            async with self._lock:
                now = time.monotonic()
                history = self._calls.setdefault(key, deque())
                while history and (now - history[0]) >= self.window_seconds:
                    history.popleft()

                if len(history) < rpm_limit:
                    history.append(now)
                    return total_waited

                wait_seconds = max(0.05, self.window_seconds - (now - history[0]) + 0.05)

            if progress_callback:
                progress_callback("rate_limit_wait", {
                    "model": model_name,
                    "rpm_limit": rpm_limit,
                    "wait_seconds": wait_seconds,
                })
            logger.info(
                f"Rate limit reached for {model_name} ({rpm_limit} calls/min). "
                f"Waiting {wait_seconds:.1f}s for next minute reset..."
            )
            await asyncio.sleep(wait_seconds)
            total_waited += wait_seconds


_rate_limiter = ModelRateLimiter()


def get_rate_limiter() -> ModelRateLimiter:
    """Get the global per-model rate limiter."""
    return _rate_limiter


async def diagnose_connection(base_url: str, api_key: str, timeout: float = 8.0) -> Dict[str, Any]:
    """Diagnose connectivity to model API endpoint to find out what the problem is."""
    import httpx
    import time
    from urllib.parse import urlparse

    result = {
        "reachable": False,
        "latency_ms": None,
        "status_code": None,
        "error": None,
        "summary": "",
    }

    parsed = urlparse(base_url)
    hostname = parsed.hostname or base_url

    start = time.time()
    try:
        test_url = base_url.rstrip("/") + "/models"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(test_url, headers=headers)
            elapsed_ms = int((time.time() - start) * 1000)
            result["reachable"] = True
            result["latency_ms"] = elapsed_ms
            result["status_code"] = resp.status_code
            if resp.status_code == 200:
                result["summary"] = f"Endpoint is reachable and responsive ({elapsed_ms}ms latency). The provider may be experiencing queue delays or high load."
            elif resp.status_code == 401:
                result["summary"] = f"Endpoint is reachable ({elapsed_ms}ms), but returned 401 Unauthorized (Invalid API Key)."
            elif resp.status_code == 429:
                result["summary"] = f"Endpoint is reachable ({elapsed_ms}ms), but rate limit (429) was exceeded."
            else:
                result["summary"] = f"Endpoint is reachable ({elapsed_ms}ms) but returned HTTP status {resp.status_code}."
    except httpx.ConnectTimeout:
        result["error"] = "ConnectTimeout"
        result["summary"] = f"Could not connect to {hostname} within {timeout}s. Check internet connection, proxy, or firewall."
    except httpx.ConnectError as e:
        result["error"] = f"ConnectError: {e}"
        result["summary"] = f"Connection failed to {hostname}. Host unreachable or offline."
    except httpx.ReadTimeout:
        result["error"] = "ReadTimeout"
        result["summary"] = f"Connected to {hostname}, but server timed out without returning response headers."
    except Exception as e:
        result["error"] = str(e)
        result["summary"] = f"Diagnostic probe error: {e}"

    return result


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
        timeout: float = MODEL_CALL_TIMEOUT,
    ):
        config = get_config()

        self.api_key = api_key or config.openai_api_key
        self.base_url = base_url or config.openai_base_url
        # Use explicitly passed model; caller handles priority resolution
        self.model = model if model is not None else config.openai_model
        self.timeout = timeout

        self.client = AsyncOpenAI(
            api_key=self.api_key or "not-configured",
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self.progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        logger.info(f"Model client initialized: {self.model} @ {self.base_url} (timeout: {self.timeout}s)")

    def reload_client(self) -> None:
        """Reload and recreate the AsyncOpenAI client instance (restarts connection pool)."""
        logger.info(f"Reloading AsyncOpenAI client connection pool for {self.model} @ {self.base_url}")
        try:
            self.client = AsyncOpenAI(
                api_key=self.api_key or "not-configured",
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception as e:
            logger.error(f"Failed to reload model client: {e}")

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        retry_delays: Optional[List[int]] = None,
        request_timeout: Optional[float] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> ModelResponse:
        """Send a chat completion request with incremental retry delay and 3-min timeout recovery."""
        if not self.api_key:
            raise ValueError("API key is not configured. Run /provider or /settings to set your API key.")

        delays = retry_delays if retry_delays is not None else RETRY_DELAYS
        cb = progress_callback or self.progress_callback
        total_attempts = len(delays) + 1
        call_timeout = request_timeout if request_timeout is not None else self.timeout

        for attempt_idx in range(total_attempts):
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

                try:
                    await _rate_limiter.acquire(self.model, progress_callback=cb)
                    if stream:
                        response = await asyncio.wait_for(
                            self._stream_chat(
                                progress_callback=cb,
                                est_prompt_tokens=sum(len(str(m.get("content", ""))) for m in messages) // 4,
                                **kwargs,
                            ),
                            timeout=call_timeout,
                        )
                    else:
                        raw_response = await asyncio.wait_for(
                            self.client.chat.completions.create(**kwargs),
                            timeout=call_timeout,
                        )
                        response = self._parse_response(raw_response)
                except (asyncio.TimeoutError, TimeoutError):
                    # Model did not respond within 3 minutes (or call_timeout)
                    logger.warning(
                        f"Model did not respond within {int(call_timeout)}s (3 mins). Diagnosing issue..."
                    )
                    diagnostic = await diagnose_connection(self.base_url, self.api_key)
                    self.reload_client()
                    if cb:
                        cb("model_timeout", {
                            "timeout": call_timeout,
                            "diagnostic": diagnostic,
                            "attempt": attempt_idx + 1,
                            "max_attempts": len(delays),
                        })
                    raise TimeoutError(
                        f"No response from model within {int(call_timeout)}s (3 mins). "
                        f"Diagnostic: {diagnostic['summary']}"
                    )

                if attempt_idx > 0:
                    logger.info(f"Model API call recovered on retry {attempt_idx}")
                    if cb:
                        cb("model_retry_success", {"attempt": attempt_idx})

                return response

            except Exception as e:
                if attempt_idx < len(delays):
                    delay = delays[attempt_idx]
                    retry_num = attempt_idx + 1
                    logger.warning(
                        f"Model API error: {e}. Retrying in {delay}s (attempt {retry_num}/{len(delays)})..."
                    )
                    if cb:
                        cb("model_retry", {
                            "error": str(e),
                            "attempt": retry_num,
                            "max_attempts": len(delays),
                            "delay": delay,
                        })
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"Model API error exhausted all {len(delays)} retries: {e}")
                    raise

    async def _stream_chat(
        self,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        est_prompt_tokens: int = 0,
        **kwargs,
    ) -> ModelResponse:
        """Handle streaming response and emit real-time token_stream progress updates."""
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs["stream_options"] = {"include_usage": True}

        try:
            stream_resp = await self.client.chat.completions.create(**stream_kwargs)
        except Exception:
            # Some OpenAI-compatible providers reject stream_options; retry plain stream=True
            stream_kwargs.pop("stream_options", None)
            try:
                stream_resp = await self.client.chat.completions.create(**stream_kwargs)
            except Exception:
                # Fallback to non-streaming if provider doesn't support streaming
                raw_response = await self.client.chat.completions.create(**kwargs, stream=False)
                return self._parse_response(raw_response)

        # If a mocked create() returned a non-async-iterable object, parse it directly
        if not hasattr(stream_resp, "__aiter__"):
            return self._parse_response(stream_resp)

        content_parts: List[str] = []
        tool_calls_map: Dict[int, Dict[str, Any]] = {}
        finish_reason: Optional[str] = None
        usage: Optional[Dict[str, Any]] = None
        est_down = 0
        char_count = 0

        async for chunk in stream_resp:
            if getattr(chunk, "usage", None):
                u = chunk.usage
                usage = {
                    "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
                    "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
                    "total_tokens": getattr(u, "total_tokens", 0) or 0,
                }
                if usage["prompt_tokens"]:
                    est_prompt_tokens = usage["prompt_tokens"]
                if usage["completion_tokens"]:
                    est_down = usage["completion_tokens"]

            choices = getattr(chunk, "choices", None) or []
            if choices:
                choice = choices[0]
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    delta_content = getattr(delta, "content", None)
                    if delta_content:
                        content_parts.append(delta_content)
                        char_count += len(delta_content)
                        est_down = max(est_down + 1, char_count // 4)

                    delta_tcs = getattr(delta, "tool_calls", None)
                    if delta_tcs:
                        for tc in delta_tcs:
                            idx = getattr(tc, "index", 0) or 0
                            if idx not in tool_calls_map:
                                tool_calls_map[idx] = {
                                    "id": getattr(tc, "id", None) or f"call_{idx}",
                                    "type": getattr(tc, "type", None) or "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            elif getattr(tc, "id", None):
                                tool_calls_map[idx]["id"] = tc.id
                            fn = getattr(tc, "function", None)
                            if fn is not None:
                                if getattr(fn, "name", None):
                                    tool_calls_map[idx]["function"]["name"] += fn.name
                                if getattr(fn, "arguments", None):
                                    arg_chunk = fn.arguments
                                    tool_calls_map[idx]["function"]["arguments"] += arg_chunk
                                    char_count += len(arg_chunk)
                                    est_down = max(est_down + 1, char_count // 4)

            if progress_callback:
                progress_callback("token_stream", {
                    "tokens_up": est_prompt_tokens,
                    "tokens_down": est_down,
                })

        tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else None
        full_content = "".join(content_parts) if content_parts else None
        return ModelResponse(
            content=full_content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            raw=None,
        )

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