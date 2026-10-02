"""Robust JSON extraction from LLM outputs."""

import re
import json
from typing import Any


def extract_json(text: str) -> Any:
    """
    Extract JSON from text, handling:
    1. Plain JSON strings
    2. ```json ... ``` Markdown code blocks
    3. JSON embedded in other text
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in text: {text[:200]}")
