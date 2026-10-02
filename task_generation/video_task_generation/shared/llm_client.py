"""OpenAI-compatible LLM / VLM client wrapper.

Calls the compatible service pointed at by `OPENAI_BASE_URL` through the
official openai SDK. Two kinds of call are supported:
  * `call_llm(messages)`                text only
  * `call_vlm(text, images_b64)`        multimodal (text plus base64 JPEGs)

Every call carries retries, a timeout, and structured logging. It returns the
string content, or "" on failure.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# openai client, imported lazily so the package still imports without it installed
_client_lock = threading.Lock()
_CLIENT = None


def _get_client():
    global _CLIENT
    with _client_lock:
        if _CLIENT is not None:
            return _CLIENT
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "openai package is required — install with `pip install openai>=1.30`"
            ) from e

        api_key = os.environ.get("OPENAI_API_KEY")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set — see .env.example")
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _CLIENT = OpenAI(**kwargs)
        logger.info("OpenAI client initialised (base_url=%s)", base_url or "<default>")
        return _CLIENT


# ──────────────────────────────────────────────────────────────────────
# Global default params (filled by config.toml via set_defaults())
# ──────────────────────────────────────────────────────────────────────


@dataclass
class _Defaults:
    model_text: str = "your-synthesis-model"
    model_vision: str = "your-synthesis-model"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    max_retries: int = 3


_DEFAULTS = _Defaults()


def set_defaults(
    model_text: Optional[str] = None,
    model_vision: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
) -> None:
    """Overwrite module-level defaults (called once at runner startup)."""
    if model_text is not None:
        _DEFAULTS.model_text = model_text
    if model_vision is not None:
        _DEFAULTS.model_vision = model_vision
    if temperature is not None:
        _DEFAULTS.temperature = temperature
    if max_tokens is not None:
        _DEFAULTS.max_tokens = max_tokens
    if timeout is not None:
        _DEFAULTS.timeout = timeout
    if max_retries is not None:
        _DEFAULTS.max_retries = max_retries
    logger.info(
        "LLM defaults: text=%s vision=%s temp=%.2f max_tokens=%d timeout=%ds retries=%d",
        _DEFAULTS.model_text, _DEFAULTS.model_vision,
        _DEFAULTS.temperature, _DEFAULTS.max_tokens,
        _DEFAULTS.timeout, _DEFAULTS.max_retries,
    )


def get_defaults() -> _Defaults:
    return _DEFAULTS


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────


def call_llm(
    messages: List[Dict[str, Any]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    **extra: Any,
) -> str:
    """Plain text chat.completions call. Returns response content or "" on failure."""
    return _chat_call(
        messages=messages,
        model=model or _DEFAULTS.model_text,
        temperature=_DEFAULTS.temperature if temperature is None else temperature,
        max_tokens=_DEFAULTS.max_tokens if max_tokens is None else max_tokens,
        timeout=_DEFAULTS.timeout if timeout is None else timeout,
        max_retries=_DEFAULTS.max_retries if max_retries is None else max_retries,
        extra=extra,
    )


def call_vlm(
    prompt: str,
    images_b64: List[str],
    *,
    system_prompt: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    image_mime: str = "image/jpeg",
    **extra: Any,
) -> str:
    """Multi-modal call: one text prompt + list of base64-encoded images."""
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{image_mime};base64,{b64}"},
        })
    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})

    return _chat_call(
        messages=messages,
        model=model or _DEFAULTS.model_vision,
        temperature=_DEFAULTS.temperature if temperature is None else temperature,
        max_tokens=_DEFAULTS.max_tokens if max_tokens is None else max_tokens,
        timeout=_DEFAULTS.timeout if timeout is None else timeout,
        max_retries=_DEFAULTS.max_retries if max_retries is None else max_retries,
        extra=extra,
    )


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _chat_call(
    messages: List[Dict[str, Any]],
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    extra: Dict[str, Any],
) -> str:
    client = _get_client()
    last_err: Optional[Exception] = None

    # Qwen3/3.5 models default to "thinking" mode which generates thousands of
    # reasoning tokens before the actual answer, often exceeding gateway timeouts.
    # Inject /no_think as system prefix to disable thinking for task-generation
    # calls (we don't need chain-of-thought here, just structured output).
    if "qwen3" in model.lower():
        if not any(m.get("role") == "system" and "/no_think" in (m.get("content") or "") for m in messages):
            messages = [{"role": "system", "content": "/no_think"}] + list(messages)

    # kimi-k2.5 only allows temperature=1; clamp it to avoid 400 errors.
    if "kimi" in model.lower():
        temperature = 1.0

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                **extra,
            )
            choice = resp.choices[0] if resp.choices else None
            if choice is None:
                raise RuntimeError("empty choices in LLM response")
            content = choice.message.content or ""
            if not content.strip():
                raise RuntimeError("LLM returned empty content")
            return content
        except Exception as exc:  # broad: upstream openai exceptions are many
            last_err = exc
            wait = min(2 ** (attempt - 1), 10)
            logger.warning(
                "LLM call failed (attempt %d/%d, model=%s): %s — retrying in %ds",
                attempt, max_retries, model, exc, wait,
            )
            time.sleep(wait)
    logger.error("LLM call gave up after %d attempts (model=%s): %s", max_retries, model, last_err)
    return ""


# ──────────────────────────────────────────────────────────────────────
# Utility: extract <tag>...</tag> content from LLM output
# ──────────────────────────────────────────────────────────────────────


def extract_tag(text: str, tag: str) -> Optional[str]:
    """Return the first ``<tag>...</tag>`` payload, or None if missing."""
    if not text:
        return None
    open_tok, close_tok = f"<{tag}>", f"</{tag}>"
    if open_tok not in text or close_tok not in text:
        return None
    inner = text.split(open_tok, 1)[1].split(close_tok, 1)[0]
    return inner.strip()
