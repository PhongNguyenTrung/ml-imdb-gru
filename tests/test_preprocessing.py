"""Tests for the regex tokenizer (Req 2)."""

from __future__ import annotations

from imdb_gru.data.preprocessing import RegexTokenizer


def test_strips_html_tags() -> None:
    tok = RegexTokenizer()
    out = tok("Hello <br /> world <p>foo</p>")
    assert out == ["hello", "world", "foo"]


def test_lowercases() -> None:
    tok = RegexTokenizer(lowercase=True)
    assert tok("THIS Movie ROCKS") == ["this", "movie", "rocks"]


def test_strips_punctuation() -> None:
    tok = RegexTokenizer()
    out = tok("Wow!!! Amazing... truly?")
    assert out == ["wow", "amazing", "truly"]


def test_keeps_apostrophes_by_default() -> None:
    tok = RegexTokenizer()
    out = tok("I don't think it's bad")
    assert "don't" in out
    assert "it's" in out


def test_drops_apostrophes_when_flag_false() -> None:
    tok = RegexTokenizer(keep_apostrophes=False)
    out = tok("I don't think")
    assert "dont" in out
    assert "don't" not in out


def test_decodes_html_entities() -> None:
    tok = RegexTokenizer()
    out = tok("Tom &amp; Jerry &quot;classic&quot;")
    assert "tom" in out and "jerry" in out and "classic" in out


def test_empty_input() -> None:
    assert RegexTokenizer()("") == []
    assert RegexTokenizer()("   ") == []
    assert RegexTokenizer()("<br />") == []
