"""VideoResearchAgent: single VLM + tool-calling loop.

Replaces the previous node-centric workflow with an agent-centric approach
where a single VLM decides what to do at each step via function calling.
The complete messages history (including reasoning_content) serves as
the trajectory and can be directly used as SFT training data.

Message format (hybrid scaffold):
  assistant → tool_calls: [{name, arguments}]      (standard OpenAI format)
  tool      → "OK"                                  (minimal placeholder per API requirement)
  user      → "[Tool Result] full text + images"    (all substantive content here)

This keeps tool_call in standard format while unifying all tool results
(text + images) into a single user message, enabling cleaner SFT data
where images and text coexist naturally.
"""

import copy
import json
import os
import re
import time
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from openai import OpenAI

from video_agent.config import AppConfig
from video_agent.llm.client import serialize_message
from video_agent.tools.tool_registry import ToolRegistry
from video_agent.utils.logger import TrajectoryLogger
from video_agent.utils.swift_exporter import export_case_to_swift
from video_agent.utils.colors import (
    header, subheader, key_value, success, error, dim, warning,
    BOLD, RESET, CYAN, GREEN, YELLOW, MAGENTA, DIM, RED, BLUE,
)

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _resolve_prompt_path(name_or_path: str) -> Path:
    """Resolve a system-prompt identifier to an absolute Path.

    Accepts either a bare name (e.g. "default_system_prompt") — which is
    resolved to `<prompts_dir>/<name>.md` — or a direct path to a .md file.
    """
    p = Path(name_or_path)
    if p.suffix and p.exists():
        return p
    candidate = PROMPTS_DIR / f"{name_or_path}.md"
    if candidate.exists():
        return candidate
    # Last resort: allow callers to pass an explicit .md path that may be
    # relative to the repo root.
    if p.suffix:
        return p
    raise FileNotFoundError(
        f"System prompt '{name_or_path}' not found. Tried: {candidate} and {p}"
    )

FORCE_ANSWER_MSG = (
    "You are running out of iterations. You MUST provide your final answer NOW. "
    "Do NOT call any more tools. Instead, output your reasoning and wrap your "
    "final answer in <answer>YOUR ANSWER</answer> tags. "
    "If you have no evidence, make your best inference based on what you know."
)

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

TOOL_RESULT_PREFIX = (
    "[Tool Execution Result]\n"
    "The content below is the return value from executing the tool(s) in the environment, "
    "NOT a new user request. Analyze the result and decide your next action.\n\n"
)

_TOOL_PLACEHOLDER = "OK"

class VideoResearchAgent:
    """Agent-centric video search system: single VLM with tool-calling."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg

        self.client_list = []

        if self.cfg.llm.num_instances > 1:
            # Parse base_url to extract scheme://host (without port/path) so we can
            # dispatch requests across multiple vLLM instances running on adjacent ports.
            # Supported base_url formats:
            #   - "http://host:port/v1"  (full URL with port and /v1)
            #   - "http://host:port"     (full URL with port, no path)
            #   - "http://host"          (bare host, no port)
            parsed = urlparse(self.cfg.llm.base_url)
            scheme = parsed.scheme or "http"
            host = parsed.hostname or "localhost"
            base_host = f"{scheme}://{host}"
            for i in range(self.cfg.llm.num_instances):
                self.client_list.append(OpenAI(
                    api_key=cfg.llm.api_key,
                    base_url=f"{base_host}:{self.cfg.llm.start_port + i}/v1",
                ))
        else:
            self.client_list.append(OpenAI(
                api_key=cfg.llm.api_key,
                base_url=cfg.llm.base_url,
            ))

        self.model = cfg.llm.model
        self.temperature = cfg.llm.temperature
        self.max_tokens = cfg.llm.max_tokens
        self.enable_thinking = cfg.llm.enable_thinking
        self.repetition_penalty = cfg.llm.repetition_penalty
        self.max_iterations = cfg.agent.max_iterations
        self.image_detail = cfg.watcher.image_detail

        self.scaffold_mode = cfg.agent.scaffold_mode  # "hybrid" or "standard"

        self.full_trajectory = cfg.agent.full_trajectory
        self.full_trajectory_dir = cfg.agent.full_trajectory_dir

        self.tools = ToolRegistry(cfg)
        self.tool_schemas = self.tools.get_openai_tools()
        self.enabled_tool_names = [s["function"]["name"] for s in self.tool_schemas]
        self.system_prompt_path = _resolve_prompt_path(cfg.agent.system_prompt)
        self.system_prompt = self.system_prompt_path.read_text(encoding="utf-8")

        self.traj_logger = TrajectoryLogger(
            trajectory_dir=cfg.logging.trajectory_dir,
            print_steps=cfg.logging.print_steps,
        )
    
    def _random_client(self) -> OpenAI:
        return random.choice(self.client_list)

    def run(
        self,
        query: str,
        row_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run open-web research on a query and return the answer and trajectory."""
        start_time = time.time()
        system_prompt = self.system_prompt
        user_content = query

        if self.cfg.logging.save_trajectory:
            self.traj_logger.start_run(
                user_query=query,
                config_snapshot=self._config_snapshot(),
                row_id=row_id,
                ground_truth=ground_truth,
            )

        print(header("VideoResearchAgent START"))
        print(key_value("Query:", query[:120], CYAN))
        print(key_value("Model:", self.model, CYAN))
        print(key_value("System Prompt:", self.system_prompt_path.name, CYAN))
        print(key_value("Max Iters:", str(self.max_iterations), CYAN))
        print(key_value("Temperature:", str(self.temperature), CYAN))
        if self.enable_thinking is not None:
            print(key_value("Thinking:", "ON" if self.enable_thinking else "OFF", CYAN))
        print(key_value("Tools:", ", ".join(self.enabled_tool_names), CYAN))
        print(key_value("Scaffold:", self.scaffold_mode, CYAN))
        if self.full_trajectory:
            print(key_value("Full Trajectory:", f"ON (dir={self.full_trajectory_dir})", YELLOW))
        print()

        messages: List[Any] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        # Preserve complete observations for SFT export.
        archival_messages: List[Any] = copy.deepcopy(messages)

        answer = ""
        confidence = ""
        explanation = ""
        iteration = 0
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        for iteration in range(self.max_iterations):
            print(subheader(f"Iteration {iteration + 1}/{self.max_iterations}"))

            if iteration == self.max_iterations - 1:
                force_message = {"role": "user", "content": FORCE_ANSWER_MSG}
                messages.append(force_message)
                archival_messages.append(copy.deepcopy(force_message))
                print(f"  {RED}{BOLD}[FORCE]{RESET} Injecting force-answer prompt")


            try:
                client = self._random_client()
                request_kwargs: Dict[str, Any] = {}
                if self.enable_thinking is not None:
                    request_kwargs["extra_body"] = {
                        "chat_template_kwargs": {
                            "enable_thinking": self.enable_thinking,
                        }
                    }
                if self.repetition_penalty is not None:
                    request_kwargs.setdefault("extra_body", {})[
                        "repetition_penalty"
                    ] = self.repetition_penalty
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=self.tool_schemas,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **request_kwargs,
                )
            except Exception as e:
                print(error(f"LLM call failed: {e}"))
                logger.error("LLM call error at iteration %d: %s", iteration, e)
                break

            msg = response.choices[0].message
            self._accumulate_usage(total_usage, response)

            reasoning = (
                getattr(msg, "reasoning_content", None)
                or getattr(msg, "reasoning", None)
                or None
            )
            if reasoning:
                preview = reasoning[:300].replace("\n", " ")
                print(f"  {DIM}[reasoning] {preview}...{RESET}")

            if msg.content:
                print(f"  {CYAN}[content]{RESET} {msg.content[:200]}")

            assistant_message = serialize_message(msg)
            messages.append(assistant_message)
            archival_messages.append(copy.deepcopy(assistant_message))

            if not getattr(msg, "tool_calls", None):
                # No tool call → check for <answer> tag in content
                raw_content = msg.content or ""
                extracted = self._extract_answer(raw_content)
                if extracted is not None:
                    answer = extracted
                    explanation = raw_content
                    print(f"\n  {GREEN}{BOLD}Agent submitted answer via <answer> tag{RESET}")
                else:
                    answer = raw_content
                    print(f"\n  {GREEN}Agent finished without tool call (text answer){RESET}")
                break

            # ── Execute tool calls ──
            image_parts: List[Any] = []
            tool_results: List[Dict[str, Any]] = []

            for tc in msg.tool_calls:
                func_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                print(f"  {MAGENTA}{BOLD}[tool_call]{RESET} {func_name}({json.dumps(args, ensure_ascii=False)[:100]})")

                result = self.tools.execute(func_name, args)

                text_parts = [result.text]
                if result.frames:
                    ts_list = ", ".join(f"{f.timestamp:.1f}s" for f in result.frames)
                    text_parts.append(f"[{len(result.frames)} video frames at: {ts_list}]")
                    for f in result.frames:
                        image_parts.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{f.image_b64}",
                                "detail": self.image_detail,
                            },
                        })
                    print(f"  {YELLOW}[frames]{RESET} Injected {len(result.frames)} frames into context")

                tool_results.append({
                    "tc": tc,
                    "func_name": func_name,
                    "text": "\n".join(text_parts),
                })

                result_preview = result.text[:150].replace("\n", " ")
                print(f"  {DIM}[result] {result_preview}...{RESET}")

            # ── Build messages based on scaffold mode ──
            if self.scaffold_mode == "standard":
                for tr in tool_results:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tr["tc"].id,
                        "name": tr["func_name"],
                        "content": tr["text"],
                    }
                    messages.append(tool_message)
                    archival_messages.append(copy.deepcopy(tool_message))
                if image_parts:
                    visual_message = {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "[Visual frames from the tool execution above]"},
                        ] + image_parts,
                    }
                    messages.append(visual_message)
                    archival_messages.append(copy.deepcopy(visual_message))
            else:
                for tr in tool_results:
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tr["tc"].id,
                        "name": tr["func_name"],
                        "content": _TOOL_PLACEHOLDER,
                    }
                    messages.append(tool_message)
                    archival_messages.append(copy.deepcopy(tool_message))
                all_text = TOOL_RESULT_PREFIX + "\n\n".join(
                    f"--- {tr['func_name']} ---\n{tr['text']}" for tr in tool_results
                )
                if image_parts:
                    user_msg: Dict[str, Any] = {
                        "role": "user",
                        "content": [{"type": "text", "text": all_text}] + image_parts,
                    }
                else:
                    user_msg = {"role": "user", "content": all_text}
                messages.append(user_msg)
                archival_messages.append(copy.deepcopy(user_msg))
        else:
            print(warning(f"\n  Max iterations ({self.max_iterations}) reached"))

        duration = time.time() - start_time

        # Export full (image-bearing) trajectory in ms-swift format BEFORE we
        # strip base64, since swift needs the raw image bytes to write JPGs.
        if self.full_trajectory and self.full_trajectory_dir:
            try:
                raw_messages = [
                    serialize_message(m) if not isinstance(m, dict) else dict(m)
                    for m in archival_messages
                ]
                export_case_to_swift(
                    out_root=Path(self.full_trajectory_dir),
                    row_id=str(row_id) if row_id is not None else (self.traj_logger._run_id or "anon"),
                    messages=raw_messages,
                    config_snapshot=self._config_snapshot(),
                    metadata={
                        "row_id": row_id,
                        "user_query": query,
                        "ground_truth": ground_truth,
                        "final_answer": answer,
                        "confidence": confidence,
                        "explanation": explanation,
                        "model": self.model,
                        "iterations": min(iteration + 1, self.max_iterations),
                        "duration_s": round(duration, 2),
                        "metrics": total_usage,
                    },
                )
                print(f"  {GREEN}[full-trajectory]{RESET} exported to {self.full_trajectory_dir}")
            except Exception as e:
                logger.error("Full-trajectory export failed for row=%s: %s", row_id, e, exc_info=True)
                print(error(f"[full-trajectory] export failed: {e}"))

        if self.cfg.logging.save_trajectory:
            serialized_messages = [
                self._safe_serialize(m) for m in messages
            ]
            self.traj_logger.log_messages_trajectory(
                messages=serialized_messages,
                final_answer=answer,
                confidence=confidence,
                explanation=explanation,
                metrics=total_usage,
                duration=duration,
            )

        print(header("RESULT"))
        print(key_value("Answer:", f"{GREEN}{BOLD}{answer}{RESET}", GREEN))
        print(key_value("Confidence:", f"{YELLOW}{confidence}{RESET}", GREEN))
        print(key_value("Explanation:", (explanation or "")[:200], GREEN))
        print(f"\n  {DIM}{'─' * 40}{RESET}")
        print(key_value("Duration:", f"{duration:.1f}s", BLUE))
        print(key_value("Total Tokens:", str(total_usage.get("total_tokens", 0)), BLUE))
        print(key_value("Iterations:", str(min(iteration + 1, self.max_iterations)), BLUE))
        print()

        return {
            "final_answer": answer,
            "confidence": confidence,
            "explanation": explanation,
            "messages": [self._safe_serialize(m) for m in messages],
            "metrics": total_usage,
            "duration": duration,
            "model": self.model,
            "iterations": min(iteration + 1, self.max_iterations),
        }

    @staticmethod
    def _extract_answer(content: str) -> Optional[str]:
        """Extract answer from <answer>...</answer> tags. Returns None if no tag found."""
        match = _ANSWER_RE.search(content)
        if match:
            return match.group(1).strip()
        return None



    @staticmethod
    def _safe_serialize(msg: Any) -> Dict[str, Any]:
        """Serialize a message, stripping base64 image data for storage."""
        if isinstance(msg, dict):
            result = {}
            for k, v in msg.items():
                if k == "content" and isinstance(v, list):
                    stripped = []
                    for item in v:
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            url_val = item.get("image_url", {}).get("url", "")
                            if url_val.startswith("data:"):
                                stripped.append({
                                    "type": "image_url",
                                    "image_url": {
                                        "url": "[base64_image_stripped_for_storage]",
                                        "detail": item.get("image_url", {}).get("detail", "low"),
                                    },
                                })
                            else:
                                stripped.append(item)
                        else:
                            stripped.append(item)
                    result[k] = stripped
                else:
                    result[k] = v
            return result
        return serialize_message(msg)

    @staticmethod
    def _accumulate_usage(total: dict, response: Any):
        usage = response.usage
        if usage:
            total["input_tokens"] += usage.prompt_tokens or 0
            total["output_tokens"] += usage.completion_tokens or 0
            total["total_tokens"] += usage.total_tokens or 0

    def _config_snapshot(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "enable_thinking": self.enable_thinking,
            "repetition_penalty": self.repetition_penalty,
            "max_iterations": self.max_iterations,
            "max_tokens": self.max_tokens,
            "image_detail": self.image_detail,
            "enabled_tools": self.enabled_tool_names,
            "scaffold_mode": self.scaffold_mode,
            "full_trajectory": self.full_trajectory,
            "full_trajectory_dir": self.full_trajectory_dir,
            "search_max_results": self.cfg.search.max_results,
            "sparse_frames": self.cfg.watcher.num_sparse_frames,
            "dense_fps": self.cfg.watcher.dense_fps,
        }

