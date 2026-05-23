"""Text preprocessing for IMDB reviews (Req 2).

The IMDB corpus contains HTML break-tags ``<br />`` and a fair amount of
punctuation noise. This module exposes a single ``RegexTokenizer`` that:

1. Strips HTML tags.
2. Lower-cases.
3. Decodes a small set of HTML entities (``&amp;``, ``&quot;``, ...).
4. Replaces non-alphanumeric runs with whitespace.
5. Splits on whitespace into tokens.

We deliberately keep the tokenizer **rule-based and reproducible** — the
research contribution lies in the GRU architecture, not in preprocessing —
and we avoid heavyweight dependencies (NLTK, spaCy) so the pipeline runs
deterministically on any machine.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

# Pre-compiled regex patterns (module-level for speed).
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9']+")
_MULTI_SPACE_RE = re.compile(r"\s+")


class RegexTokenizer:
    """Deterministic regex-based tokenizer.

    Parameters
    ----------
    lowercase : bool
        If True, lower-case the input before tokenizing.
    keep_apostrophes : bool
        If True, preserve apostrophes inside words (e.g. ``don't`` stays one
        token); otherwise they are stripped.

    Examples
    --------
    >>> tok = RegexTokenizer()
    >>> tok("This <br /> movie was *AMAZING* — I'd watch again!")
    ['this', 'movie', 'was', 'amazing', "i'd", 'watch', 'again']
    """

    def __init__(self, lowercase: bool = True, keep_apostrophes: bool = True) -> None:
        self.lowercase = lowercase
        self.keep_apostrophes = keep_apostrophes

    def __call__(self, text: str) -> list[str]:
        return self.tokenize(text)

    def tokenize(self, text: str) -> list[str]:
        text = html.unescape(text)
        text = _HTML_TAG_RE.sub(" ", text)
        if self.lowercase:
            text = text.lower()
        cleaned = _NON_ALNUM_RE.sub(" ", text)
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
        if not cleaned:
            return []
        tokens = cleaned.split(" ")
        if not self.keep_apostrophes:
            tokens = [t.replace("'", "") for t in tokens if t.replace("'", "")]
        return tokens

    def tokenize_batch(self, texts: Iterable[str]) -> list[list[str]]:
        return [self.tokenize(t) for t in texts]
