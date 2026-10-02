"""Convert agent trajectories into ms-swift multimodal agent SFT format.

Output layout (compact, portable):

    <out_root>/
    ├── all.jsonl                     # one swift sample per line (training entry point)
    ├── metadata.jsonl                # one metadata line per case (for traceability)
    └── images/
        └── case_<row_id>/
            ├── 0001.jpg
            ├── 0002.jpg
            └── ...

ms-swift agent dataset spec (multimodal, hermes/qwen agent_template):

    {
      "tools": "<json string of tool list>",
      "messages": [
        {"role": "system",        "content": "..."},
        {"role": "user",          "content": "..."},
        {"role": "assistant",     "content": "<thought text>"},
        {"role": "tool_call",     "content": "<json: name+arguments>"},
        {"role": "tool_response", "content": "<json or text>"},
        {"role": "user",          "content": "<image><image> see frames"},
        ...
      ],
      "images": ["images/case_xxx/0001.jpg", "images/case_xxx/0002.jpg", ...]
    }

Key rules implemented here:
- Each image-bearing user/tool-response message has its base64 images decoded to
  jpg files, and inline `<image>` tags are appended to that message's text in
  the SAME ORDER as the image list grows. Number of `<image>` tags across the
  whole `messages` list MUST equal len(images).
- Original openai-style assistant messages with `tool_calls` are split into
  one "assistant" entry (the textual reasoning/content) plus one
  "tool_call" entry per call (json string of {name, arguments}).
- Tool-result `user` messages are converted to `tool_response` (without `<image>`
  tags moved out — images stay attached so the model still sees them).
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Benchmark workers share one process and may finish simultaneously.  Keep the
# two combined JSONL files line-safe while allowing per-case image/JSON writes
# to proceed independently.
_JSONL_APPEND_LOCK = threading.Lock()

# Marker used in the agent's tool-result user messages; we strip it from the
# final swift `tool_response` content because swift's agent_template will wrap
# the content into the model-expected format itself.
_TOOL_RESULT_PREFIX_MARKERS = (
    "[Tool Execution Result]",
    "[Tool Result]",
)

_IMAGE_TAG = "<image>"
# Text that incidentally contains "<image>" (e.g. an LLM's own reasoning_content
# referring to an image literally) would otherwise inflate the visual-tag count
# beyond the actual image list and break ms-swift template encoding. We replace
# such literal occurrences with a benign string before injecting our own tags.
_LITERAL_IMAGE_TAG_ESCAPE = "&lt;image&gt;"


@dataclass
class _ExtractedImage:
    """An image extracted from a message, ready to be saved to disk."""
    b64: str
    detail: str = "high"


def _split_data_uri(url: str) -> Tuple[Optional[str], str]:
    """Return (mime, base64_payload) for a data: URL, or (None, url) if not."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, url
    # data:image/jpeg;base64,XXXX
    try:
        header, payload = url.split(",", 1)
    except ValueError:
        return None, url
    mime = header.split(";")[0][len("data:"):]  # e.g. image/jpeg
    return mime, payload


def _save_b64_image(b64_data: str, dest: Path) -> None:
    raw = base64.b64decode(b64_data)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)


def _sanitize_text(text: str) -> str:
    """Escape any literal `<image>` substrings so they don't get mistaken for
    multimodal placeholders by the ms-swift template engine."""
    if not text:
        return text
    return text.replace(_IMAGE_TAG, _LITERAL_IMAGE_TAG_ESCAPE)


def _extract_images_from_content(content: Any) -> Tuple[str, List[_ExtractedImage]]:
    """Pull images out of a multimodal `content` field, returning (text, images).

    Accepts:
      - str → returned as-is, no images.
      - list of {type: text | image_url} parts → concatenates text in order,
        collects images in order. The text is augmented with `<image>` tags
        IN-PLACE (one tag at the position where each image_url part appears),
        so that the resulting text preserves the original image-text interleave.
    Any `<image>` substring inside text parts is escaped to keep the final tag
    count accurate.
    """
    if content is None:
        return "", []
    if isinstance(content, str):
        return _sanitize_text(content), []

    if not isinstance(content, list):
        return _sanitize_text(str(content)), []

    text_chunks: List[str] = []
    images: List[_ExtractedImage] = []
    for item in content:
        if not isinstance(item, dict):
            text_chunks.append(_sanitize_text(str(item)))
            continue
        t = item.get("type")
        if t == "text":
            text_chunks.append(_sanitize_text(item.get("text", "") or ""))
        elif t == "image_url":
            url = item.get("image_url", {}).get("url", "")
            mime, payload = _split_data_uri(url)
            if mime is None:
                # Already a path / non-data URL — skip but still note it as a tag
                # so message order stays sane.
                text_chunks.append(_IMAGE_TAG)
                continue
            detail = item.get("image_url", {}).get("detail", "high")
            images.append(_ExtractedImage(b64=payload, detail=detail))
            text_chunks.append(_IMAGE_TAG)
        else:
            # Unknown part type — best effort
            text_chunks.append(_sanitize_text(json.dumps(item, ensure_ascii=False)))

    return "".join(text_chunks).strip(), images


def _strip_tool_result_marker(text: str) -> str:
    """Drop the `[Tool Execution Result] ...` preamble from tool-result text."""
    for marker in _TOOL_RESULT_PREFIX_MARKERS:
        if text.startswith(marker):
            # Remove the whole first paragraph up to the first blank line.
            parts = text.split("\n\n", 1)
            if len(parts) == 2:
                return parts[1].lstrip()
            return text[len(marker):].lstrip()
    return text


def _format_tool_call_arguments(raw: str) -> str:
    """Normalize tool-call arguments to a compact JSON string."""
    if raw is None:
        return "{}"
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    return json.dumps(obj, ensure_ascii=False)


def _convert_messages_to_swift(
    messages: List[Dict[str, Any]],
    image_dir: Path,
    image_basename_template: str = "{idx:04d}.jpg",
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Walk the full openai-style messages and return (swift_messages, image_paths).

    image_paths are absolute paths to extracted JPGs, in the order they appear
    across all messages (so the count and order matches `<image>` tags in
    swift_messages).
    """
    swift_msgs: List[Dict[str, Any]] = []
    image_paths: List[str] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        tool_calls = msg.get("tool_calls")

        # ── system / user (no tool_call_id) ──
        if role == "system":
            text, imgs = _extract_images_from_content(content)
            if not text:
                continue
            swift_msgs.append({"role": "system", "content": text})
            for im in imgs:
                idx = len(image_paths) + 1
                p = image_dir / image_basename_template.format(idx=idx)
                _save_b64_image(im.b64, p)
                image_paths.append(str(p.resolve()))
            continue

        if role == "user":
            text, imgs = _extract_images_from_content(content)
            # Heuristic: a "user" message that begins with our tool-result prefix
            # is actually a tool_response in swift terms.
            is_tool_response = any(
                text.lstrip().startswith(m) for m in _TOOL_RESULT_PREFIX_MARKERS
            )
            for im in imgs:
                idx = len(image_paths) + 1
                p = image_dir / image_basename_template.format(idx=idx)
                _save_b64_image(im.b64, p)
                image_paths.append(str(p.resolve()))

            if is_tool_response:
                # Keep <image> tags inline so swift template inserts them where
                # the agent saw them.
                stripped = _strip_tool_result_marker(text)
                swift_msgs.append({
                    "role": "tool_response",
                    "content": stripped,
                })
            else:
                swift_msgs.append({"role": "user", "content": text})
            continue

        # ── assistant (may carry tool_calls) ──
        if role == "assistant":
            text, imgs = _extract_images_from_content(content)
            for im in imgs:
                idx = len(image_paths) + 1
                p = image_dir / image_basename_template.format(idx=idx)
                _save_b64_image(im.b64, p)
                image_paths.append(str(p.resolve()))

            reasoning = msg.get("reasoning_content")
            assistant_text_parts: List[str] = []
            if reasoning:
                # Wrap reasoning in <think> tags (qwen3/hermes convention).
                # Sanitize literal "<image>" inside reasoning text to avoid
                # inflating the visual-token count.
                assistant_text_parts.append(
                    f"<think>\n{_sanitize_text(reasoning)}\n</think>"
                )
            if text:
                assistant_text_parts.append(text)
            assistant_text = "\n".join(assistant_text_parts).strip()

            if assistant_text:
                swift_msgs.append({"role": "assistant", "content": assistant_text})

            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    name = fn.get("name", "")
                    args_str = _format_tool_call_arguments(fn.get("arguments", "{}"))
                    swift_msgs.append({
                        "role": "tool_call",
                        "content": json.dumps(
                            {"name": name, "arguments": json.loads(args_str)
                             if args_str.startswith("{") else args_str},
                            ensure_ascii=False,
                        ),
                    })
            continue

        # ── tool placeholder (hybrid scaffold writes "OK" here) ──
        if role == "tool":
            # In hybrid scaffold the actual tool result lives in the next user
            # message; here `content` is "OK". We still emit an empty
            # tool_response so message order is preserved IF content is
            # non-trivial; otherwise we DROP it because the next user msg will
            # carry the real payload (and ms-swift expects each tool_call to be
            # paired with one tool_response, which our user-msg conversion
            # provides).
            text = ""
            if isinstance(content, str):
                text = content.strip()
            elif isinstance(content, list):
                text, _ = _extract_images_from_content(content)
            if not text or text.upper() == "OK":
                # Drop placeholder — user msg right after carries the real result
                continue
            swift_msgs.append({"role": "tool_response", "content": text})
            continue

    return swift_msgs, image_paths


def _build_tools_string(config_snapshot: Optional[Dict[str, Any]]) -> str:
    """Produce the `tools` JSON string from the config snapshot's enabled tools.

    We import TOOL_SCHEMAS lazily here so this module remains importable in
    contexts where heavy tool deps (ffmpeg, torch...) aren't installed.
    """
    enabled = (config_snapshot or {}).get("enabled_tools") or []
    if not enabled:
        return "[]"
    try:
        from video_agent.tools.tool_registry import TOOL_SCHEMAS  # local import
    except Exception:
        logger.warning("Could not import TOOL_SCHEMAS for swift export; tools=[]")
        return "[]"
    keep = [s for s in TOOL_SCHEMAS if s.get("function", {}).get("name") in enabled]
    return json.dumps(keep, ensure_ascii=False)


def _validate_image_tag_count(swift_msgs: List[Dict[str, Any]], n_images: int) -> bool:
    """Sanity-check: total `<image>` tag count across messages equals n_images."""
    total = 0
    for m in swift_msgs:
        c = m.get("content")
        if isinstance(c, str):
            total += len(re.findall(re.escape(_IMAGE_TAG), c))
    if total != n_images:
        logger.warning(
            "Swift export tag-count mismatch: %d <image> tags vs %d image files",
            total, n_images,
        )
        return False
    return True


def _externalize_openai_messages(
    messages: List[Dict[str, Any]],
    out_root: Path,
    case_id: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Preserve OpenAI-style messages while replacing data URIs with files.

    The returned messages retain roles, reasoning content, tool calls, tool
    call IDs, text parts, image order, and image detail.  Each inline base64
    image is decoded under ``images/case_<id>/`` and recorded in a manifest so
    a paper-case bundle can be inspected without loading enormous data URIs.
    Paths are relative to ``out_root``.
    """
    archived = copy.deepcopy(messages)
    image_dir = out_root / "images" / f"case_{case_id}"
    image_manifest: List[Dict[str, Any]] = []
    image_index = 0

    for message_index, message in enumerate(archived):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for content_index, part in enumerate(content):
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict):
                continue
            url = image_url.get("url", "")
            mime, payload = _split_data_uri(url)
            if mime is None:
                image_manifest.append({
                    "message_index": message_index,
                    "content_index": content_index,
                    "path_or_url": url,
                    "detail": image_url.get("detail", "high"),
                    "embedded": False,
                })
                continue

            image_index += 1
            destination = image_dir / f"{image_index:04d}.jpg"
            raw = base64.b64decode(payload)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            relative_path = str(destination.relative_to(out_root))
            image_url["url"] = relative_path
            image_manifest.append({
                "image_index": image_index,
                "message_index": message_index,
                "content_index": content_index,
                "path": relative_path,
                "mime_type": mime,
                "detail": image_url.get("detail", "high"),
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "embedded": True,
            })

    return archived, image_manifest


def _write_case_archive(
    *,
    out_root: Path,
    case_id: str,
    row_id: str,
    messages: List[Dict[str, Any]],
    config_snapshot: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
) -> Path:
    """Write a self-contained, human-inspectable trajectory for one case."""
    archived_messages, image_manifest = _externalize_openai_messages(
        messages, out_root, case_id
    )
    case_dir = out_root / "cases" / f"case_{case_id}"
    case_dir.mkdir(parents=True, exist_ok=True)
    archive_path = case_dir / "trajectory.json"
    payload = {
        "schema_version": "video-searcher.full-trajectory.v1",
        "row_id": row_id,
        "bundle_root": "../..",
        "config_snapshot": config_snapshot or {},
        "metadata": metadata or {},
        "messages": archived_messages,
        "image_manifest": image_manifest,
        "image_count": sum(bool(item.get("embedded")) for item in image_manifest),
    }
    temporary_path = archive_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(archive_path)
    return archive_path


def export_case_to_swift(
    *,
    out_root: Path,
    row_id: str,
    messages: List[Dict[str, Any]],
    config_snapshot: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a single trajectory as ms-swift agent-multimodal sample.

    Compact layout: images land in <out_root>/images/case_<row_id>/ with
    relative paths in the jsonl so the dataset is portable across machines.
    """
    import re as _re

    out_root = Path(out_root)
    case_id = _re.sub(r'[^\w\-.]', '_', str(row_id)).strip('_')[:120]
    image_dir = out_root / "images" / f"case_{case_id}"
    out_root.mkdir(parents=True, exist_ok=True)

    swift_msgs, image_paths_abs = _convert_messages_to_swift(messages, image_dir)

    # Preserve the original OpenAI-style trajectory as a separate per-case
    # archive.  It points at the same decoded image files as the compact swift
    # sample and is the authoritative artifact for paper examples.
    _write_case_archive(
        out_root=out_root,
        case_id=case_id,
        row_id=str(row_id),
        messages=messages,
        config_snapshot=config_snapshot,
        metadata=metadata,
    )

    # Convert absolute image paths to relative (from out_root)
    image_paths_rel = []
    for p in image_paths_abs:
        try:
            image_paths_rel.append(str(Path(p).relative_to(out_root.resolve())))
        except ValueError:
            image_paths_rel.append(p)

    sample: Dict[str, Any] = {
        "tools": _build_tools_string(config_snapshot),
        "messages": swift_msgs,
    }
    if image_paths_rel:
        sample["images"] = image_paths_rel

    _validate_image_tag_count(swift_msgs, len(image_paths_rel))

    # Append to combined jsonl (training entry point)
    all_jsonl = out_root / "all.jsonl"
    with _JSONL_APPEND_LOCK:
        with open(all_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        # Append metadata to metadata.jsonl under the same lock so both files
        # preserve a consistent completion order across concurrent workers.
        if metadata is not None:
            meta_out = dict(metadata)
            meta_out["_case_id"] = case_id
            meta_jsonl = out_root / "metadata.jsonl"
            with open(meta_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(meta_out, ensure_ascii=False, default=str) + "\n")

    return sample
