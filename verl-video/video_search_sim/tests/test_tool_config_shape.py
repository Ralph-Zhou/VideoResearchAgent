"""Verify that ``configs/tool_config.yaml`` has the shape verl expects.

We deliberately do *not* construct the tool itself (that requires verl + ray
+ a running FastAPI service). We only check that:

- the YAML parses,
- every entry has the required top-level keys,
- ``config.type`` is one of {native, mcp} so ``ToolType(...)`` in verl's
  ``tool_registry.initialize_tools_from_config`` will accept it,
- the tool schema parses as an OpenAI function schema,
- **every parameter type is one that ``Qwen3XMLToolParser`` can round-trip
  safely** — this is the critical alignment for qwen3_coder XML tool calls.

If the second check fails, the agent will still *emit* tool calls, but the
parser will silently degrade array/object params to ``eval()`` or lose them
entirely, which manifests as opaque runtime errors deep inside the rollout
rather than at config-validation time. Catching it here keeps CI green.
"""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "tool_config.yaml"

# Subset of types that Qwen3XMLToolParser converts without ``eval()``.
# See verl/experimental/agent_loop/tool_parser.py :: _parse_xml_function_call
# ``convert_param_value`` — branches it recognises: string/str/text/varchar/
# char/enum, int/uint/long/short/unsigned, num/float, boolean/bool/binary.
# Everything else (array / object / dict) falls through to json.loads / eval
# which is fragile under even minor model drift.
_XML_SAFE_PARAM_TYPES = {
    "string",
    "str",
    "text",
    "varchar",
    "char",
    "enum",
    "integer",
    "int",  # "int" appears in some loose schemas; Qwen parser accepts both
    "uint",
    "long",
    "short",
    "unsigned",
    "number",
    "num",
    "float",
    "boolean",
    "bool",
    "binary",
}

EXPECTED_TOOL_NAMES = {
    "search_youtube",
    "web_search",
    "watch_video",
    "visual_grounding",
}


def _load_tools() -> list[dict]:
    with CONFIG_PATH.open("r") as fh:
        data = yaml.safe_load(fh)
    assert "tools" in data and isinstance(data["tools"], list) and data["tools"]
    return data["tools"]


def test_tool_config_structure() -> None:
    tools = _load_tools()
    for entry in tools:
        assert "class_name" in entry
        assert "config" in entry
        assert "tool_schema" in entry

        cfg = entry["config"]
        assert cfg.get("type") in {"native", "mcp"}, "verl tool_registry requires a valid `type`"

        schema = entry["tool_schema"]
        assert schema.get("type") == "function"
        fn = schema.get("function", {})
        assert isinstance(fn.get("name"), str) and fn["name"]
        assert isinstance(fn.get("description"), str)
        params = fn.get("parameters", {})
        assert params.get("type") == "object"
        assert "properties" in params and isinstance(params["properties"], dict)


def test_all_expected_tools_present() -> None:
    names = {e["tool_schema"]["function"]["name"] for e in _load_tools()}
    assert names == EXPECTED_TOOL_NAMES, f"tool_config.yaml must declare exactly {EXPECTED_TOOL_NAMES}; got {names}"


def test_param_types_are_xml_parser_safe() -> None:
    """All parameters must use types that qwen3_coder's XML parser handles cleanly."""
    offenders: list[tuple[str, str, str]] = []
    for entry in _load_tools():
        fn = entry["tool_schema"]["function"]
        props = fn.get("parameters", {}).get("properties", {}) or {}
        for param_name, param_schema in props.items():
            t = str(param_schema.get("type", "")).lower()
            if t not in _XML_SAFE_PARAM_TYPES:
                offenders.append((fn["name"], param_name, t))
    assert not offenders, (
        "These parameters declare types that qwen3_coder XML parser cannot safely "
        f"round-trip: {offenders}. Rewrite them as strings (and parse inside the tool)."
    )


def test_required_params_reference_existing_properties() -> None:
    """Every name in ``required`` must exist in ``properties``."""
    broken = []
    for entry in _load_tools():
        fn = entry["tool_schema"]["function"]
        params = fn.get("parameters", {})
        props = set(params.get("properties", {}).keys())
        required = params.get("required", []) or []
        for r in required:
            if r not in props:
                broken.append((fn["name"], r))
    assert not broken, f"required params missing from properties: {broken}"


def test_video_search_tool_config_contract() -> None:
    """Lock down the specific shape the VideoSearchTool expects."""
    entries = [e for e in _load_tools() if e["class_name"].endswith("VideoSearchTool")]
    assert len(entries) == 1, "exactly one VideoSearchTool entry expected"
    entry = entries[0]

    cfg = entry["config"]
    assert "retrieval_service_url" in cfg and cfg["retrieval_service_url"].startswith("http")
    assert cfg.get("type") == "native"

    fn = entry["tool_schema"]["function"]
    assert fn["name"] == "search_youtube"
    assert "query" in fn["parameters"]["properties"]
    assert "query" in fn["parameters"].get("required", [])


def test_watch_video_tool_mode_is_string_enum() -> None:
    """The `mode` parameter must be typed as string — XML parser would eval() an array."""
    entries = [e for e in _load_tools() if e["class_name"].endswith("WatchVideoTool")]
    assert len(entries) == 1
    props = entries[0]["tool_schema"]["function"]["parameters"]["properties"]
    assert "mode" in props and props["mode"]["type"] == "string"


def test_visual_grounding_timestamps_is_string() -> None:
    """``timestamps`` must be a string, not an array: avoids eval() in the parser."""
    entries = [e for e in _load_tools() if e["class_name"].endswith("VisualGroundingTool")]
    assert len(entries) == 1
    props = entries[0]["tool_schema"]["function"]["parameters"]["properties"]
    assert props["timestamps"]["type"] == "string", (
        "timestamps must be string (comma-separated floats); qwen3_coder XML parser "
        "cannot safely parse array parameters."
    )


def test_answer_termination_is_not_a_tool() -> None:
    # Appendix D: the policy terminates with an assistant <answer> span.
    assert not any(e["class_name"].endswith("SubmitAnswerTool") for e in _load_tools())
