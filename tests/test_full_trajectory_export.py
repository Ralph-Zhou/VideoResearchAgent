import base64
import json
from types import SimpleNamespace

from video_agent.llm.client import serialize_message
from video_agent.utils.swift_exporter import export_case_to_swift


def _data_uri(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def test_serialize_message_preserves_vllm_reasoning_field():
    message = SimpleNamespace(
        role="assistant",
        content=None,
        reasoning="I should verify the scene before answering.",
        tool_calls=[],
    )
    serialized = serialize_message(message)
    assert serialized["reasoning"] == message.reasoning
    assert serialized["reasoning_content"] == message.reasoning


def test_full_trajectory_export_preserves_openai_messages_and_images(tmp_path):
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": None,
            "reasoning_content": "I should inspect the video.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "watch_video",
                        "arguments": '{"url":"https://youtu.be/example0000"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "watch_video",
            "content": "Video metadata and frame timestamps",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "frames follow"},
                {
                    "type": "image_url",
                    "image_url": {"url": _data_uri(b"first-image"), "detail": "low"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": _data_uri(b"second-image"), "detail": "high"},
                },
            ],
        },
        {"role": "assistant", "content": "<answer>example</answer>"},
    ]
    metadata = {"row_id": "67", "ground_truth": "example", "final_answer": "example"}

    swift_sample = export_case_to_swift(
        out_root=tmp_path,
        row_id="67",
        messages=messages,
        config_snapshot={"enabled_tools": []},
        metadata=metadata,
    )

    archive_path = tmp_path / "cases" / "case_67" / "trajectory.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    assert archive["schema_version"] == "video-searcher.full-trajectory.v1"
    assert archive["row_id"] == "67"
    assert archive["config_snapshot"] == {"enabled_tools": []}
    assert archive["metadata"] == metadata
    assert [message["role"] for message in archive["messages"]] == [
        "system", "user", "assistant", "tool", "user", "assistant"
    ]
    assert archive["messages"][2]["reasoning_content"] == "I should inspect the video."
    assert archive["messages"][2]["tool_calls"][0]["function"]["name"] == "watch_video"
    assert archive["messages"][3]["content"] == "Video metadata and frame timestamps"

    image_urls = [
        part["image_url"]["url"]
        for part in archive["messages"][4]["content"]
        if part.get("type") == "image_url"
    ]
    assert image_urls == ["images/case_67/0001.jpg", "images/case_67/0002.jpg"]
    assert (tmp_path / image_urls[0]).read_bytes() == b"first-image"
    assert (tmp_path / image_urls[1]).read_bytes() == b"second-image"
    assert archive["image_count"] == 2
    assert [item["message_index"] for item in archive["image_manifest"]] == [4, 4]
    assert [item["content_index"] for item in archive["image_manifest"]] == [1, 2]
    assert all(len(item["sha256"]) == 64 for item in archive["image_manifest"])

    assert len(swift_sample["images"]) == 2
    assert sum(
        message["content"].count("<image>")
        for message in swift_sample["messages"]
    ) == 2
    assert len((tmp_path / "all.jsonl").read_text(encoding="utf-8").splitlines()) == 1
    assert len((tmp_path / "metadata.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_external_image_reference_is_preserved_without_fake_embedded_count(tmp_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "existing image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/frame.jpg", "detail": "low"},
                },
            ],
        }
    ]
    export_case_to_swift(
        out_root=tmp_path,
        row_id="external",
        messages=messages,
        config_snapshot={"enabled_tools": []},
        metadata={},
    )
    archive = json.loads(
        (tmp_path / "cases" / "case_external" / "trajectory.json").read_text()
    )
    assert archive["messages"][0]["content"][1]["image_url"]["url"] == (
        "https://example.com/frame.jpg"
    )
    assert archive["image_count"] == 0
    assert archive["image_manifest"][0]["embedded"] is False
