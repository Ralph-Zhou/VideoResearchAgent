"""Unit tests for the shared BM25 tokenizer."""

from __future__ import annotations

from video_search_sim.video_corpus.indexer import tokenize


def test_lowercase_default() -> None:
    toks = tokenize("Hello World")
    assert toks == ["hello", "world"]


def test_preserve_case() -> None:
    toks = tokenize("Hello World", lowercase=False)
    assert toks == ["Hello", "World"]


def test_punctuation_stripped() -> None:
    toks = tokenize("foo, bar; baz! baz?")
    assert toks == ["foo", "bar", "baz", "baz"]


def test_alphanumeric_kept_together() -> None:
    toks = tokenize("abc123 XYZ9")
    assert toks == ["abc123", "xyz9"]


def test_empty_inputs() -> None:
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!!!") == []
