"""
HybridGuard canonicalization front-end.

A deterministic pre-classification pipeline that inverts common obfuscation
transforms BEFORE any feature is extracted. Motivated by the observation
(paper Table 2) that homoglyph-lite perturbations collapse the baseline
HG_MULTIFEAT AUROC from 0.998 to 0.574 on the prompt-injection benchmark.

Design goals:
    1. Pure standard library. No model dependency. CPU-only, sub-millisecond
       per call on typical (<=4 kB) prompts, so it is safe to run on every
       inference call in a production gate.
    2. Deterministic and idempotent. The same input always produces the same
       canonical form, and canonicalize(canonicalize(x)) == canonicalize(x).
    3. Depth-bounded. Nested encodings (e.g., base64 of rot13 of a payload)
       are detected and unwrapped up to MAX_DECODE_DEPTH layers to prevent
       decoder-chain denial of service.
    4. Audit-friendly. Every canonicalization step records the transform name
       in a CanonicalResult.trace list, so downstream features and humans
       can reason about what was changed and why.

Pipeline (applied in order):
    a. NFKC normalization. Folds compatibility codepoints (ligatures, full-
       width digits, mathematical alphanumerics) to their canonical forms.
    b. Zero-width / invisible character stripping. Removes U+200B-U+200D,
       U+2060, U+FEFF, U+180E and the Tag block (U+E0000..U+E007F).
    c. Confusable-character folding (Unicode TR39, reduced table). Maps
       commonly abused Latin/Cyrillic/Greek look-alikes to ASCII. The table
       shipped here is a conservative subset chosen to minimize false
       folding on legitimate non-English content.
    d. Whitespace / punctuation collapse. Collapses runs of >=3 identical
       punctuation characters and >=2 whitespace characters to a single
       occurrence. Keeps decision boundaries stable under punctuation-
       stuffing transforms.
    e. Case folding for feature extraction only. Produced as a parallel
       lowercase trace; the original case is preserved in the primary
       output so downstream models that rely on case (e.g., CNN character
       features) still see the un-case-folded canonical form.
    f. Encoding unwrap. Detects base64, hex, URL-encoded, and ROT13 blobs
       that decode to printable ASCII and substitutes the decoded content
       inline, annotated with an <ENC:base64> marker so the downstream
       classifier can use "prompt contained decoded base64" as a signal.

Public API:
    canonicalize(text: str) -> CanonicalResult
    CanonicalResult.canonical: str, the canonical form
    CanonicalResult.lowercase: str, case-folded canonical form
    CanonicalResult.trace: list[str], ordered transform names applied
    CanonicalResult.decoded_payloads: list[tuple[str, str]], (encoding, decoded)

References:
    Unicode Technical Report 39 (Unicode Security Mechanisms), sec. 4
    Unicode Normalization Forms, UAX #15
    OWASP LLM01:2025 Prompt Injection, canonicalization recommendation
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_DECODE_DEPTH: int = 3
"""Maximum layers of nested encoding to unwrap. Bounds CPU cost per call."""

MIN_ENCODED_LEN: int = 12
"""Minimum span length to attempt base64/hex decoding on. Avoids false
triggers on short alphanumeric tokens."""

# ---------------------------------------------------------------------------
# Character tables
# ---------------------------------------------------------------------------

_INVISIBLE_CHARS: str = (
    "\u200B"  # zero-width space
    "\u200C"  # zero-width non-joiner
    "\u200D"  # zero-width joiner
    "\u2060"  # word joiner
    "\uFEFF"  # zero-width no-break space / BOM
    "\u180E"  # Mongolian vowel separator (deprecated but still appears)
    "\u00AD"  # soft hyphen
    "\u034F"  # combining grapheme joiner
)

# Unicode Tag block (U+E0000..U+E007F). Historically proposed for language
# tagging, now commonly abused to smuggle invisible instructions into LLMs.
_TAG_BLOCK_RANGE: Tuple[int, int] = (0xE0000, 0xE007F)

# Conservative confusable table, derived from Unicode TR39 confusables.txt.
# Each key is a non-ASCII codepoint that visually resembles the ASCII value.
# Only Latin/Cyrillic/Greek high-risk confusables are included to minimize
# incorrect folding on legitimate non-Latin content.
_CONFUSABLES: dict = {
    # Cyrillic -> Latin
    "\u0430": "a",  # а
    "\u0435": "e",  # е
    "\u043E": "o",  # о
    "\u0440": "p",  # р
    "\u0441": "c",  # с
    "\u0445": "x",  # х
    "\u0443": "y",  # у
    "\u0456": "i",  # і
    "\u0458": "j",  # ј
    "\u0455": "s",  # ѕ
    "\u0459": "lj", # љ (leave for now)
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041A": "K",
    "\u041C": "M", "\u041D": "H", "\u041E": "O", "\u0420": "P",
    "\u0421": "C", "\u0422": "T", "\u0425": "X", "\u0423": "Y",
    # Greek -> Latin
    "\u03B1": "a", "\u03BF": "o", "\u03C1": "p", "\u03C4": "t",
    "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H",
    "\u0399": "I", "\u039A": "K", "\u039C": "M", "\u039D": "N",
    "\u039F": "O", "\u03A1": "P", "\u03A4": "T", "\u03A5": "Y",
    "\u03A7": "X", "\u03A9": "O",  # Omega-> O
    # Fullwidth -> ASCII (NFKC usually handles these, included for safety)
    "\uFF41": "a", "\uFF42": "b", "\uFF43": "c", "\uFF44": "d",
    "\uFF45": "e", "\uFF49": "i", "\uFF4E": "n", "\uFF4F": "o",
    "\uFF52": "r", "\uFF53": "s", "\uFF54": "t",
    # Mathematical alphanumerics commonly used in jailbreak prompts
    "\U0001D41A": "a", "\U0001D41B": "b", "\U0001D41C": "c",
    "\U0001D41D": "d", "\U0001D41E": "e", "\U0001D41F": "f",
    "\U0001D420": "g", "\U0001D421": "h", "\U0001D422": "i",
    "\U0001D423": "j", "\U0001D424": "k", "\U0001D425": "l",
    "\U0001D426": "m", "\U0001D427": "n", "\U0001D428": "o",
    "\U0001D429": "p", "\U0001D42A": "q", "\U0001D42B": "r",
    "\U0001D42C": "s", "\U0001D42D": "t", "\U0001D42E": "u",
    "\U0001D42F": "v", "\U0001D430": "w", "\U0001D431": "x",
    "\U0001D432": "y", "\U0001D433": "z",
}

# Regex for candidate base64 spans. Require length >= MIN_ENCODED_LEN, valid
# base64 charset, and optional '=' padding.
_B64_RE = re.compile(
    r"(?<![A-Za-z0-9+/=])"
    r"((?:[A-Za-z0-9+/]{4}){" + str(MIN_ENCODED_LEN // 4) + r",}"
    r"(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?)"
    r"(?![A-Za-z0-9+/=])"
)

# Regex for candidate hex spans. Require length >= MIN_ENCODED_LEN, even.
_HEX_RE = re.compile(
    r"(?<![A-Fa-f0-9])"
    r"((?:[A-Fa-f0-9]{2}){" + str(MIN_ENCODED_LEN // 2) + r",})"
    r"(?![A-Fa-f0-9])"
)

# Regex for URL-encoded sequences. Spans may contain ASCII letters between
# %XX groups (e.g. '%3Cscript%3E' has 'script' between encoded markers), so
# we greedily match the widest span that contains at least 3 %XX instances.
_URL_RE = re.compile(r"(?:[A-Za-z0-9_\-./]|%[0-9A-Fa-f]{2})+")
_URL_MIN_PCTXX = 3  # require at least this many %XX in a matched span


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class CanonicalResult:
    """Result of canonicalize(). See module docstring for field semantics."""
    canonical: str
    lowercase: str
    trace: List[str] = field(default_factory=list)
    decoded_payloads: List[Tuple[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return self.canonical


# ---------------------------------------------------------------------------
# Internal transforms
# ---------------------------------------------------------------------------

def _strip_invisibles(text: str) -> Tuple[str, bool]:
    """Remove zero-width and Tag-block characters. Returns (cleaned, changed)."""
    if not any(c in _INVISIBLE_CHARS for c in text) and not any(
        _TAG_BLOCK_RANGE[0] <= ord(c) <= _TAG_BLOCK_RANGE[1] for c in text
    ):
        return text, False
    cleaned = []
    for c in text:
        if c in _INVISIBLE_CHARS:
            continue
        if _TAG_BLOCK_RANGE[0] <= ord(c) <= _TAG_BLOCK_RANGE[1]:
            continue
        cleaned.append(c)
    new = "".join(cleaned)
    return new, new != text


def _fold_confusables(text: str) -> Tuple[str, bool]:
    """Apply TR39 confusable folding. Returns (folded, changed)."""
    if not any(c in _CONFUSABLES for c in text):
        return text, False
    out = []
    for c in text:
        out.append(_CONFUSABLES.get(c, c))
    new = "".join(out)
    return new, new != text


def _collapse_runs(text: str) -> Tuple[str, bool]:
    """Collapse runs of 3+ identical punctuation and 2+ whitespace."""
    # Collapse runs of 3+ identical non-alphanumeric characters to a single one.
    # E.g. '!!!!!!' -> '!', '........' -> '.', '------' -> '-'.
    collapsed = re.sub(r"([^\w\s])\1{2,}", r"\1", text)
    # Collapse runs of whitespace (but preserve single newlines for structure).
    collapsed = re.sub(r"[ \t]{2,}", " ", collapsed)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
    return collapsed, collapsed != text


def _try_base64(span: str) -> str | None:
    """Attempt to base64-decode a candidate span. Returns decoded ASCII or None."""
    try:
        # Enforce valid base64 alphabet and padding.
        decoded = base64.b64decode(span, validate=True)
    except (binascii.Error, ValueError):
        return None
    try:
        as_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # Require that the decoded content is mostly printable, otherwise reject
    # (reduces false positives on arbitrary binary blobs).
    printable = sum(c.isprintable() or c in "\t\n" for c in as_text)
    if not as_text or printable / max(len(as_text), 1) < 0.9:
        return None
    return as_text


def _try_hex(span: str) -> str | None:
    """Attempt to hex-decode a candidate span."""
    try:
        decoded = bytes.fromhex(span)
    except ValueError:
        return None
    try:
        as_text = decoded.decode("utf-8")
    except UnicodeDecodeError:
        return None
    printable = sum(c.isprintable() or c in "\t\n" for c in as_text)
    if not as_text or printable / max(len(as_text), 1) < 0.9:
        return None
    return as_text


def _try_url(span: str) -> str | None:
    """Attempt to URL-decode a candidate span."""
    try:
        from urllib.parse import unquote
        decoded = unquote(span)
    except Exception:
        return None
    if decoded == span:
        return None
    return decoded


def _try_rot13(span: str) -> str | None:
    """Apply ROT13. Only returned if the result looks more English-like
    (contains common English trigrams) than the input."""
    # Cheap English-likeness heuristic: count occurrences of ' the ', ' and ',
    # ' ignore ', ' system ', ' prompt ', ' assistant ' in the result.
    import codecs
    decoded = codecs.encode(span, "rot_13")
    needles = (" the ", " and ", " ignore", " system", " prompt",
               " assistant", " user", " please", " you ")
    score_in = sum(n in span.lower() for n in needles)
    score_out = sum(n in decoded.lower() for n in needles)
    if score_out > score_in:
        return decoded
    return None


def _unwrap_encodings(
    text: str, depth: int = 0, decoded_out: List[Tuple[str, str]] | None = None
) -> Tuple[str, List[Tuple[str, str]]]:
    """Detect and inline-decode base64/hex/URL/rot13 spans up to MAX_DECODE_DEPTH."""
    if decoded_out is None:
        decoded_out = []
    if depth >= MAX_DECODE_DEPTH:
        return text, decoded_out

    changed = False

    # Base64
    def _b64_sub(m: re.Match) -> str:
        nonlocal changed
        span = m.group(1)
        dec = _try_base64(span)
        if dec is None:
            return span
        changed = True
        decoded_out.append(("base64", dec))
        return f" <ENC:base64>{dec}</ENC> "

    text = _B64_RE.sub(_b64_sub, text)

    # Hex
    def _hex_sub(m: re.Match) -> str:
        nonlocal changed
        span = m.group(1)
        dec = _try_hex(span)
        if dec is None:
            return span
        changed = True
        decoded_out.append(("hex", dec))
        return f" <ENC:hex>{dec}</ENC> "

    text = _HEX_RE.sub(_hex_sub, text)

    # URL
    def _url_sub(m: re.Match) -> str:
        nonlocal changed
        span = m.group(0)
        # Gate on minimum number of %XX groups to avoid false positives on
        # plain identifiers like 'tokenId_abc123'.
        if len(re.findall(r"%[0-9A-Fa-f]{2}", span)) < _URL_MIN_PCTXX:
            return span
        dec = _try_url(span)
        if dec is None:
            return span
        changed = True
        decoded_out.append(("url", dec))
        return dec

    text = _URL_RE.sub(_url_sub, text)

    # ROT13: apply to the full text if it looks ROT13-ish and the decode
    # increases English-likeness.
    rot = _try_rot13(text)
    if rot is not None:
        changed = True
        decoded_out.append(("rot13", rot))
        text = rot

    # Recurse if anything changed and we have depth budget remaining.
    if changed and depth + 1 < MAX_DECODE_DEPTH:
        return _unwrap_encodings(text, depth=depth + 1, decoded_out=decoded_out)
    return text, decoded_out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def canonicalize(text: str) -> CanonicalResult:
    """Run the full canonicalization pipeline on a single prompt.

    Parameters
    ----------
    text : str
        Raw prompt text as received at the gate.

    Returns
    -------
    CanonicalResult
        The canonical form, the case-folded canonical form, the ordered list
        of transform names that modified the text, and any decoded payloads
        recovered by the encoding-unwrap stage.
    """
    if not isinstance(text, str):
        raise TypeError(f"canonicalize expects str, got {type(text).__name__}")

    trace: List[str] = []
    decoded: List[Tuple[str, str]] = []

    # a. NFKC normalization
    nfkc = unicodedata.normalize("NFKC", text)
    if nfkc != text:
        trace.append("nfkc")
    text = nfkc

    # b. Invisible stripping
    text, changed = _strip_invisibles(text)
    if changed:
        trace.append("invisibles")

    # c. Confusable folding
    text, changed = _fold_confusables(text)
    if changed:
        trace.append("confusables")

    # d. Whitespace / punctuation collapse
    text, changed = _collapse_runs(text)
    if changed:
        trace.append("collapse")

    # f. Encoding unwrap (depth-bounded)
    unwrapped, decoded = _unwrap_encodings(text, depth=0, decoded_out=[])
    if unwrapped != text:
        trace.append(f"encoding_unwrap:{len(decoded)}")
    text = unwrapped

    # e. Case-folded parallel trace
    lowercase = text.casefold()

    return CanonicalResult(
        canonical=text,
        lowercase=lowercase,
        trace=trace,
        decoded_payloads=decoded,
    )


__all__ = ["canonicalize", "CanonicalResult", "MAX_DECODE_DEPTH"]
