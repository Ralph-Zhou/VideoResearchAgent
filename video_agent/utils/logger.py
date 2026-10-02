"""Trajectory logger for recording agent execution as messages (SFT-ready format)."""

import json
import time
import uuid
import logging
from pathlib import Path
from typing import Optional, Any, Dict, List

from video_agent.utils.colors import node_tag, dim, RESET, DIM

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """
    Records agent execution trajectories in JSONL format.
    New format: trajectory = complete messages list (user/assistant/tool),
    directly usable as SFT training data.
    """

    def __init__(self, trajectory_dir: str, print_steps: bool = True):
        self.trajectory_dir = Path(trajectory_dir)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.print_steps = print_steps
        self._current_file: Optional[Path] = None
        self._run_id: Optional[str] = None

    def start_run(
        self,
        user_query: str,
        config_snapshot: dict,
        row_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
    ) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._run_id = f"run_{ts}_{uuid.uuid4().hex[:6]}"

        prefix = f"row{row_id}_" if row_id else ""
        filename = f"{prefix}{self._run_id}.jsonl"
        self._current_file = self.trajectory_dir / filename

        metadata = {
            "type": "metadata",
            "run_id": self._run_id,
            "row_id": row_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "config": config_snapshot,
            "user_query": user_query,
            "ground_truth": ground_truth,
        }
        self._append(metadata)
        return self._run_id

    def log_messages_trajectory(
        self,
        messages: List[Dict[str, Any]],
        final_answer: str,
        confidence: str = "",
        explanation: str = "",
        metrics: Optional[Dict] = None,
        duration: Optional[float] = None,
    ):
        """Log the complete messages trajectory (SFT-ready format)."""
        self._append({
            "type": "messages",
            "run_id": self._run_id,
            "messages": messages,
        })
        result = {
            "type": "result",
            "run_id": self._run_id,
            "final_answer": final_answer,
            "confidence": confidence,
            "explanation": explanation,
            "metrics": metrics or {},
            "duration": duration,
        }
        self._append(result)

    # Keep legacy log_step for backward compat if needed
    def log_step(
        self,
        node: str,
        loop_round: int,
        input_data: Any,
        output_data: Any,
        llm_call: Optional[dict] = None,
        tool_call: Optional[dict] = None,
        sub_stage: Optional[str] = None,
        reasoning_content: Optional[str] = None,
    ):
        step = {
            "type": "step",
            "run_id": self._run_id,
            "node": node,
            "loop_round": loop_round,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input": input_data,
            "output": output_data,
        }
        if sub_stage:
            step["sub_stage"] = sub_stage
        if llm_call:
            step["llm_call"] = llm_call
        if tool_call:
            step["tool_call"] = tool_call
        if reasoning_content:
            step["reasoning_content"] = reasoning_content
        self._append(step)

        if self.print_steps:
            out_str = json.dumps(output_data, ensure_ascii=False, default=str)
            print(f"  {node_tag(node)} {dim(out_str[:300])}")

    def log_result(
        self,
        final_answer: str,
        explanation: str,
        confidence: str,
        metrics: dict,
        is_correct: Optional[bool] = None,
    ):
        result = {
            "type": "result",
            "run_id": self._run_id,
            "final_answer": final_answer,
            "explanation": explanation,
            "confidence": confidence,
            "metrics": metrics,
        }
        if is_correct is not None:
            result["is_correct"] = is_correct
        self._append(result)

    def _append(self, data: dict):
        if self._current_file is None:
            return
        with open(self._current_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, default=str) + "\n")
