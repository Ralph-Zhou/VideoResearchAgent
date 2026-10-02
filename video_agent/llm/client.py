"""OpenAI-compatible LLM client with reasoning trace support."""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from openai import OpenAI
from video_agent.config import AppConfig

logger = logging.getLogger(__name__)


def _reasoning(msg: Any) -> Optional[str]:
    """Read both legacy provider and vLLM 0.17 reasoning fields."""
    return getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or None


@dataclass
class ChatResult:
    content: str
    reasoning_content: Optional[str]
    token_usage: Dict[str, int]
    latency_ms: float
    raw_response: Any = field(repr=False, default=None)

    def to_assistant_message(self) -> Dict:
        msg: Dict[str, Any] = {"role": "assistant", "content": self.content}
        if self.reasoning_content:
            msg["reasoning_content"] = self.reasoning_content
        return msg


def serialize_message(msg: Any) -> Dict[str, Any]:
    """Preserve content, tool calls, and either provider's reasoning field."""
    if isinstance(msg, dict):
        return dict(msg)
    out: Dict[str, Any] = {
        "role": getattr(msg, "role", "assistant"),
        "content": getattr(msg, "content", None) or None,
    }
    reasoning = _reasoning(msg)
    if reasoning:
        out["reasoning_content"] = reasoning
    if getattr(msg, "reasoning", None):
        out["reasoning"] = msg.reasoning
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [{
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        } for tc in msg.tool_calls]
    for key in ("tool_call_id", "name"):
        value = getattr(msg, key, None)
        if value is not None:
            out[key] = value
    return out


class LLMClient:
    def __init__(self, cfg: AppConfig, node_name: str = "default"):
        node = cfg.llm.get_node_config(node_name)
        self.client = OpenAI(api_key=node.api_key, base_url=node.base_url)
        self.model = node.model
        self.temperature = node.temperature
        self.max_tokens = node.max_tokens
        self.enable_thinking = node.enable_thinking
        self.repetition_penalty = node.repetition_penalty
        self.node_name = node_name

    def chat(self, messages: List[Dict], temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> ChatResult:
        start = time.time()
        kwargs: Dict[str, Any] = {}
        if self.enable_thinking is not None:
            kwargs["extra_body"] = {"chat_template_kwargs": {
                "enable_thinking": self.enable_thinking,
            }}
        if self.repetition_penalty is not None:
            kwargs.setdefault("extra_body", {})[
                "repetition_penalty"
            ] = self.repetition_penalty
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs,
        )
        latency = (time.time() - start) * 1000
        msg = response.choices[0].message
        reasoning = _reasoning(msg)
        usage = self._extract_usage(response)
        logger.debug(
            "[LLM:%s] model=%s latency=%.0fms tokens=%s reasoning=%s",
            self.node_name, self.model, latency, usage,
            f"{len(reasoning)}chars" if reasoning else "none",
        )
        return ChatResult(msg.content or "", reasoning, usage, latency, response)

    @staticmethod
    def _extract_usage(response: Any) -> Dict[str, int]:
        usage = response.usage
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        return {
            "input_tokens": usage.prompt_tokens or 0,
            "output_tokens": usage.completion_tokens or 0,
            "total_tokens": usage.total_tokens or 0,
        }
