"""
OpenAI API interface for LLMs

This module also supports a "manual mode" (human-in-the-loop) where prompts are written
to a task queue directory and the system waits for a corresponding *.answer.json file
"""

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai

from dio_agent.llm.base import LLMInterface

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _build_display_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Render messages into a single plain-text prompt for the manual UI.
    """
    chunks: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = m.get("content", "")
        chunks.append(f"### {role}\n{content}\n")
    return "\n".join(chunks).rstrip() + "\n"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _format_exception_details(exc: Exception, api_base: str, model: str) -> str:
    """
    Build a detailed, single-line error string for network/API failures.
    """
    def _preview(value: Any, max_len: int = 500) -> str:
        text = str(value)
        return text if len(text) <= max_len else text[:max_len] + "...(truncated)"

    exc_type = type(exc).__name__
    exc_message = str(exc).strip() or repr(exc)
    details = [f"{exc_type}: {exc_message}", f"model={model}", f"api_base={api_base}"]

    # OpenAI SDK exception fields
    status_code = getattr(exc, "status_code", None)
    request_id = getattr(exc, "request_id", None) or getattr(exc, "_request_id", None)
    body = getattr(exc, "body", None)

    # Try to extract richer response/request info from SDK response object
    response_obj = getattr(exc, "response", None)
    request_url = None
    retry_after = None
    if response_obj is not None:
        if status_code is None:
            status_code = getattr(response_obj, "status_code", None)
        headers = getattr(response_obj, "headers", None)
        if headers:
            retry_after = headers.get("retry-after") or headers.get("Retry-After")
        req = getattr(response_obj, "request", None)
        if req is not None:
            request_url = getattr(req, "url", None)
        if body is None:
            # Best effort: httpx.Response.text may be unavailable in some SDK wrappers.
            text_attr = getattr(response_obj, "text", None)
            if text_attr:
                body = text_attr

    if status_code is not None:
        details.append(f"status_code={status_code}")
    if request_id:
        details.append(f"request_id={request_id}")
    if request_url:
        details.append(f"request_url={request_url}")
    if retry_after:
        details.append(f"retry_after={retry_after}s")
    if body:
        details.append(f"response_body={_preview(body)}")

    # Include underlying transport cause (often httpx timeout / DNS / TLS details)
    cause = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    if cause is not None:
        details.append(f"cause={type(cause).__name__}: {_preview(cause)}")

    # Human-readable issue category + hints
    issue = "unknown"
    hint = ""
    if isinstance(exc, openai.RateLimitError) or status_code == 429:
        issue = "rate_limit"
        hint = "Likely too many requests or provider-side throttling"
    elif isinstance(exc, openai.APITimeoutError):
        issue = "timeout"
        hint = "Provider did not return in time; consider lower timeout/model load"
    elif isinstance(exc, openai.APIConnectionError):
        issue = "connection_error"
        hint = "Network/region/TLS/proxy issue while connecting to provider"
    elif isinstance(exc, openai.AuthenticationError) or status_code == 401:
        issue = "auth_error"
        hint = "Invalid/expired API key or key not allowed for this provider"
    elif isinstance(exc, openai.PermissionDeniedError) or status_code == 403:
        issue = "permission_error"
        hint = "Access denied by provider/policy/account restrictions"
    elif isinstance(exc, openai.BadRequestError) or status_code == 400:
        issue = "bad_request"
        hint = "Request payload/model parameters invalid for this endpoint"
    elif isinstance(exc, openai.NotFoundError) or status_code == 404:
        issue = "not_found"
        hint = "Model or endpoint not found"
    elif isinstance(exc, openai.UnprocessableEntityError) or status_code == 422:
        issue = "unprocessable_entity"
        hint = "Request understood but semantically invalid"
    elif isinstance(exc, openai.InternalServerError) or (status_code is not None and status_code >= 500):
        issue = "provider_server_error"
        hint = "Provider/OpenRouter upstream internal error"

    details.append(f"issue={issue}")
    if hint:
        details.append(f"hint={hint}")

    return " | ".join(details)


def _extract_usage_counts(response: Any) -> Dict[str, int]:
    """Best-effort token usage extraction from OpenAI-compatible responses."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    usage_dict: Dict[str, Any] = {}
    if isinstance(usage, dict):
        usage_dict = usage
    elif hasattr(usage, "model_dump"):
        try:
            dumped = usage.model_dump()
            if isinstance(dumped, dict):
                usage_dict = dumped
        except Exception:
            usage_dict = {}

    def _lookup(field_names: List[str]) -> Any:
        for field_name in field_names:
            value = getattr(usage, field_name, None)
            if value is not None:
                return value
            if field_name in usage_dict:
                return usage_dict[field_name]
        return None

    prompt_tokens = _lookup(["prompt_tokens", "input_tokens"])
    completion_tokens = _lookup(["completion_tokens", "output_tokens"])
    total_tokens = _lookup(["total_tokens"])

    def _safe_int(value: Any) -> int:
        return int(value) if isinstance(value, (int, float)) else 0

    prompt_tokens = _safe_int(prompt_tokens)
    completion_tokens = _safe_int(completion_tokens)
    total_tokens = _safe_int(total_tokens)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _drop_none_values(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _normalize_proxy_base_url(url: Optional[str]) -> str:
    if not url:
        return ""

    normalized = str(url).strip().rstrip("/")
    if normalized.endswith("/chat/completions"):
        normalized = normalized[: -len("/chat/completions")]
    if normalized.endswith("/v1"):
        normalized = normalized[: -len("/v1")]
    return normalized.rstrip("/")


def _should_use_raw_anthropic_auth(api_base: Optional[str], api_key: Optional[str]) -> bool:
    proxy_base = os.environ.get("ANTHROPIC_BASE_URL")
    proxy_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
    if not proxy_base or not proxy_token or not api_key:
        return False
    return (
        api_key == proxy_token
        and _normalize_proxy_base_url(api_base) == _normalize_proxy_base_url(proxy_base)
    )


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)
        self._last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        # Manual mode: enabled via llm.manual_mode in config.yaml
        self.manual_mode = (getattr(model_cfg, "manual_mode", False) is True)
        self.manual_queue_dir: Optional[Path] = None

        if self.manual_mode:
            qdir = getattr(model_cfg, "_manual_queue_dir", None)
            if not qdir:
                raise ValueError(
                    "Manual mode is enabled but manual_queue_dir is missing. "
                    "This should be injected by the DIOAgent controller."
                )
            self.manual_queue_dir = Path(str(qdir)).expanduser().resolve()
            self.manual_queue_dir.mkdir(parents=True, exist_ok=True)
            self.client = None
        else:
            # Set up API client (normal mode)
            # OpenAI client requires max_retries to be int, not None
            max_retries = self.retries if self.retries is not None else 0
            use_raw_anthropic_auth = _should_use_raw_anthropic_auth(self.api_base, self.api_key)
            client_api_key = "" if use_raw_anthropic_auth else self.api_key
            default_headers = None
            if use_raw_anthropic_auth:
                default_headers = {"Authorization": str(self.api_key)}
            self.client = openai.OpenAI(
                api_key=client_api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=max_retries,
                default_headers=default_headers,
            )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define OpenAI reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is an OpenAI reasoning model based on model name pattern
        # This works for all endpoints (OpenAI, Azure, OptiLLM, OpenRouter, etc.)
        model_lower = str(self.model).lower()
        is_openai_reasoning_model = model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)

        if is_openai_reasoning_model:
            # For OpenAI reasoning models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Add optional reasoning parameters if provided
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Handle reasoning_effort for open source reasoning models.
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        # Seed only makes sense for actual API calls
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None and not self.manual_mode:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Some OpenAI-compatible providers reject explicit nulls (e.g. top_p=null).
        # Omit None-valued optional parameters entirely.
        params = _drop_none_values(params)

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        # Manual mode: no timeout unless explicitly passed by the caller
        if self.manual_mode:
            timeout = kwargs.get("timeout", None)
            self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            return await self._manual_wait_for_answer(params, timeout=timeout)

        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await self._call_api(params, timeout=timeout)
                return response
            except Exception as e:
                error_details = _format_exception_details(e, self.api_base, self.model)
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {error_details}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"All {retries + 1} attempts failed with error: {error_details}"
                    )
                    raise RuntimeError(error_details) from e

    async def _call_api(self, params: Dict[str, Any], timeout: Optional[Union[int, float]] = None) -> str:
        """Make the actual API call"""
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")

        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        call_params = dict(params)
        if timeout is not None:
            call_params["timeout"] = timeout
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**call_params)
        )
        # Best-effort transport metadata (SDK-dependent): useful for diagnosing
        # provider/network behavior without breaking compatibility.
        response_obj = getattr(response, "response", None) or getattr(response, "_response", None)
        status_code = getattr(response_obj, "status_code", None)
        request_id = getattr(response, "_request_id", None) or getattr(response, "request_id", None)
        if status_code is not None or request_id is not None:
            logger.info(
                "LLM API call succeeded: model=%s api_base=%s status_code=%s request_id=%s",
                self.model,
                self.api_base,
                status_code,
                request_id,
            )
        # Logging of system prompt, user message and response content
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        self._last_usage = _extract_usage_counts(response)
        return response.choices[0].message.content

    def consume_last_usage(self) -> Dict[str, int]:
        usage = self._last_usage or {}
        consumed = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
        self._last_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return consumed

    async def _manual_wait_for_answer(
        self, params: Dict[str, Any], timeout: Optional[Union[int, float]]
    ) -> str:
        """
        Manual mode: write a task JSON file and poll for *.answer.json
        If timeout is provided, we respect it; otherwise we wait indefinitely
        """

        if self.manual_queue_dir is None:
            raise RuntimeError("manual_queue_dir is not initialized")

        task_id = str(uuid.uuid4())
        messages = params.get("messages", [])
        display_prompt = _build_display_prompt(messages)

        task_payload: Dict[str, Any] = {
            "id": task_id,
            "created_at": _iso_now(),
            "model": params.get("model"),
            "display_prompt": display_prompt,
            "messages": messages,
            "meta": {
                "max_tokens": params.get("max_tokens"),
                "max_completion_tokens": params.get("max_completion_tokens"),
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "reasoning_effort": params.get("reasoning_effort"),
                "verbosity": params.get("verbosity"),
            },
        }

        task_path = self.manual_queue_dir / f"{task_id}.json"
        answer_path = self.manual_queue_dir / f"{task_id}.answer.json"

        _atomic_write_json(task_path, task_payload)
        logger.info(f"[manual_mode] Task enqueued: {task_path}")

        start = time.time()
        poll_interval = 0.5

        while True:
            if answer_path.exists():
                try:
                    data = json.loads(answer_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"[manual_mode] Failed to parse answer JSON for {task_id}: {e}")
                    await asyncio.sleep(poll_interval)
                    continue

                answer = str(data.get("answer") or "")
                logger.info(f"[manual_mode] Answer received for {task_id}")
                return answer

            if timeout is not None and (time.time() - start) > float(timeout):
                raise asyncio.TimeoutError(
                    f"Manual mode timed out after {timeout} seconds waiting for answer of task {task_id}"
                )

            await asyncio.sleep(poll_interval)
