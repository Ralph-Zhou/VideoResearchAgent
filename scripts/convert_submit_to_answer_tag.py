#!/usr/bin/env python3
"""Convert old trajectories from submit_answer tool to <answer> tag format.

Old format (last 3 messages):
  assistant:     <think>reasoning...</think>
  tool_call:     {"name": "submit_answer", "arguments": {"answer": "X", ...}}
  tool_response: --- submit_answer --- Answer submitted: X

New format (last 1 message):
  assistant:     <think>reasoning...</think>\n\n<answer>X</answer>

Also updates:
  - system prompt → new prompt (no submit_answer)
  - tools schema → removes submit_answer entry
"""

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

NEW_SYSTEM_PROMPT = (Path(__file__).parent.parent / "video_agent" / "prompts" / "default_system_prompt.md").read_text(encoding="utf-8")


def remove_submit_tool_schema(tools_str: str) -> str:
    """Remove the submit_answer tool from a JSON-encoded tools list."""
    try:
        tools = json.loads(tools_str)
    except (json.JSONDecodeError, TypeError):
        return tools_str
    filtered = [t for t in tools if t.get("function", {}).get("name") != "submit_answer"]
    return json.dumps(filtered, ensure_ascii=False)


def extract_answer_from_tool_call(tool_call_content: str) -> tuple:
    """Parse the tool_call JSON to get answer, confidence, explanation."""
    try:
        obj = json.loads(tool_call_content)
        args = obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        return (
            args.get("answer", ""),
            args.get("confidence", ""),
            args.get("explanation", ""),
        )
    except (json.JSONDecodeError, TypeError):
        return ("", "", "")


def convert_messages(messages: list, new_system_prompt: str) -> list:
    """Convert a message list: replace system prompt, remove submit_answer tail."""
    if not messages:
        return messages

    result = list(messages)

    # 1. Replace system prompt
    if result[0].get("role") == "system":
        result[0] = dict(result[0])
        result[0]["content"] = new_system_prompt

    # 2. Find and convert the submit_answer pattern at the end
    # Pattern: ..., assistant, tool_call(submit_answer), tool_response
    if len(result) >= 3:
        last3 = result[-3:]
        is_submit = (
            last3[1].get("role") == "tool_call"
            and "submit_answer" in (last3[1].get("content") or "")
            and last3[2].get("role") == "tool_response"
        )
        if is_submit:
            answer, confidence, explanation = extract_answer_from_tool_call(
                last3[1].get("content", "")
            )

            # Merge into the preceding assistant message
            assistant_msg = dict(last3[0])
            old_content = assistant_msg.get("content", "") or ""
            assistant_msg["content"] = f"{old_content}\n\n<answer>{answer}</answer>"
            result[-3] = assistant_msg
            # Remove tool_call and tool_response
            result = result[:-2]

    # 3. Also handle cases where submit_answer appears elsewhere (non-tail)
    # This shouldn't normally happen, but just in case: remove any remaining
    # tool_call/tool_response pairs for submit_answer
    cleaned = []
    i = 0
    while i < len(result):
        m = result[i]
        if (m.get("role") == "tool_call"
                and "submit_answer" in (m.get("content") or "")):
            answer, _, _ = extract_answer_from_tool_call(m.get("content", ""))
            # Append answer tag to previous assistant message
            if cleaned and cleaned[-1].get("role") == "assistant":
                prev = dict(cleaned[-1])
                old_c = prev.get("content", "") or ""
                if "<answer>" not in old_c:
                    prev["content"] = f"{old_c}\n\n<answer>{answer}</answer>"
                    cleaned[-1] = prev
            # Skip tool_call
            i += 1
            # Skip tool_response if it follows
            if i < len(result) and result[i].get("role") == "tool_response":
                resp_content = result[i].get("content", "")
                if "submit_answer" in resp_content or "Answer submitted" in resp_content:
                    i += 1
            continue
        cleaned.append(m)
        i += 1

    return cleaned


def convert_sample_json(sample: dict, new_system_prompt: str) -> dict:
    """Convert a single sample.json dict."""
    out = dict(sample)

    # Update tools
    if "tools" in out:
        out["tools"] = remove_submit_tool_schema(out["tools"])

    # Update messages
    if "messages" in out:
        out["messages"] = convert_messages(out["messages"], new_system_prompt)

    return out


def convert_all_jsonl(input_path: str, output_path: str, new_system_prompt: str) -> dict:
    """Convert all.jsonl line by line. Returns stats."""
    stats = {"total": 0, "converted": 0, "skipped": 0}
    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            try:
                obj = json.loads(line)
                converted = convert_sample_json(obj, new_system_prompt)
                fout.write(json.dumps(converted, ensure_ascii=False) + "\n")
                stats["converted"] += 1
            except Exception as e:
                print(f"  Warning: failed to convert line {stats['total']}: {e}")
                fout.write(line + "\n")
                stats["skipped"] += 1
    return stats


def convert_case_dirs(traj_dir: str, output_dir: str, new_system_prompt: str) -> dict:
    """Convert all case_*/sample.json files."""
    stats = {"total": 0, "converted": 0, "skipped": 0}
    traj_path = Path(traj_dir)
    out_path = Path(output_dir)

    for case_dir in sorted(traj_path.iterdir()):
        if not case_dir.is_dir() or not case_dir.name.startswith("case_"):
            continue
        stats["total"] += 1
        sample_file = case_dir / "sample.json"
        if not sample_file.exists():
            stats["skipped"] += 1
            continue

        try:
            with open(sample_file) as f:
                sample = json.load(f)

            converted = convert_sample_json(sample, new_system_prompt)

            # Write to output dir, preserving case structure
            out_case_dir = out_path / case_dir.name
            out_case_dir.mkdir(parents=True, exist_ok=True)

            with open(out_case_dir / "sample.json", "w") as f:
                json.dump(converted, f, ensure_ascii=False, indent=2)

            # Copy meta.json and images
            meta_file = case_dir / "meta.json"
            if meta_file.exists():
                shutil.copy2(meta_file, out_case_dir / "meta.json")

            images_dir = case_dir / "images"
            if images_dir.exists():
                out_images = out_case_dir / "images"
                if not out_images.exists():
                    os.symlink(str(images_dir.resolve()), str(out_images))

            stats["converted"] += 1
        except Exception as e:
            print(f"  Warning: failed to convert {case_dir.name}: {e}")
            stats["skipped"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert old trajectories from submit_answer to <answer> tag format."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Input distill directory (e.g. data/results/distill_stage4_tasks/distill)",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Output directory for converted data",
    )
    parser.add_argument(
        "--system-prompt", type=str, default=None,
        help="Path to new system prompt .md file (default: video_agent/prompts/default_system_prompt.md)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.system_prompt:
        new_prompt = Path(args.system_prompt).read_text(encoding="utf-8")
    else:
        new_prompt = NEW_SYSTEM_PROMPT

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"System prompt: {len(new_prompt)} chars")
    print()

    # Convert all.jsonl
    all_jsonl = input_dir / "all.jsonl"
    if all_jsonl.exists():
        print("Converting all.jsonl ...")
        stats = convert_all_jsonl(
            str(all_jsonl),
            str(output_dir / "all.jsonl"),
            new_prompt,
        )
        print(f"  Total: {stats['total']}, Converted: {stats['converted']}, Skipped: {stats['skipped']}")
    else:
        print(f"  all.jsonl not found at {all_jsonl}, skipping.")
    print()

    # Convert case directories
    traj_dir = input_dir / "trajectories"
    if traj_dir.exists():
        print("Converting case_* directories ...")
        out_traj = output_dir / "trajectories"
        stats = convert_case_dirs(str(traj_dir), str(out_traj), new_prompt)
        print(f"  Total: {stats['total']}, Converted: {stats['converted']}, Skipped: {stats['skipped']}")
    else:
        print(f"  trajectories/ not found at {traj_dir}, skipping.")
    print()

    # Verification: check a few converted samples
    converted_all = output_dir / "all.jsonl"
    if converted_all.exists():
        print("=== Verification ===")
        with open(converted_all) as f:
            for idx, line in enumerate(f):
                if idx >= 3:
                    break
                obj = json.loads(line.strip())
                msgs = obj.get("messages", [])
                last = msgs[-1] if msgs else {}
                last_role = last.get("role", "?")
                last_content = str(last.get("content", ""))[:200]
                has_answer_tag = "<answer>" in last_content
                has_submit = any(
                    "submit_answer" in (m.get("content") or "")
                    for m in msgs
                    if m.get("role") == "tool_call"
                )
                print(f"  Sample {idx}: last_role={last_role}, "
                      f"has_<answer>={has_answer_tag}, "
                      f"has_submit_answer={has_submit}")
                if last_role == "assistant":
                    # Show tail of content
                    tail = str(last.get("content", ""))[-150:]
                    print(f"    tail: ...{tail}")
        print()

    print("Done!")


if __name__ == "__main__":
    main()
