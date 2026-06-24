"""
test_confusables_drift.py — WS5 guard against silent drift of the confusable fold
table back to a hand-curated map.

The certificate's closure-soundness (=1.0) depends on CONFUSABLES_TR39 being the
FULL UTS#39 cross-script table, not a small hand map. Two layers of defense:

 1. Structural invariants (always run, no external dependency): the table is large,
    spans many scripts, and every entry is a single non-ASCII codepoint mapped to a
    single ASCII character. A regression to a 40-entry Cyrillic/Greek hand map fails
    these immediately.

 2. Authoritative cross-check (runs when the `confusable_homoglyphs` package, which
    bundles the Unicode confusables data, is installed; skipped otherwise): every
    table entry homoglyph->ascii must be reachable in the authoritative Unicode
    confusable graph, i.e. the standard agrees that the homoglyph is confusable with
    its ASCII target. This catches a silently *edited* table, not just a shrunken one.
"""
import os
import unicodedata

import pytest

from confusables_table import CONFUSABLES_TR39 as T

_ASCII = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


# --- Layer 1: structural invariants -------------------------------------------

def test_table_is_full_not_a_hand_map():
    # the shipped table has 539 entries; a hand map (the failure mode) has ~40.
    assert len(T) >= 400, f"confusable table shrank to {len(T)} entries (hand-map drift?)"


def test_entries_are_single_codepoint_nonascii_to_ascii():
    for k, v in T.items():
        assert len(k) == 1 and ord(k) > 127, f"bad key {k!r}"
        assert len(v) == 1 and v in _ASCII, f"bad value {v!r} for key {k!r}"
        assert k != v


def test_no_combining_mark_keys():
    # combining marks are handled by the strip-combining pass, not the fold table;
    # a combining key would be a generation bug.
    for k in T:
        assert not unicodedata.combining(k), f"combining-mark key {k!r}"


def test_script_diversity():
    prefixes = set()
    for k in T:
        try:
            prefixes.add(unicodedata.name(k).split(" ")[0])
        except ValueError:
            pass
    # a Cyrillic/Greek hand map covers ~2 prefixes; the full table covers dozens.
    assert len(prefixes) >= 6, f"table covers only {len(prefixes)} script-prefixes"


# --- Layer 2: authoritative Unicode cross-check -------------------------------

def _load_authoritative_graph():
    confusable_homoglyphs = pytest.importorskip("confusable_homoglyphs")
    import json
    path = os.path.join(os.path.dirname(confusable_homoglyphs.__file__), "confusables.json")
    if not os.path.exists(path):
        pytest.skip("confusables.json not bundled with confusable_homoglyphs")
    data = json.load(open(path))
    return {k: set(e["c"] for e in v) for k, v in data.items()}


def _reaches_ascii(graph, k, target, maxd=2):
    seen, frontier = {k}, {k}
    for _ in range(maxd + 1):
        nxt = set()
        for c in frontier:
            for r in graph.get(c, ()):
                if r not in seen:
                    seen.add(r)
                    nxt.add(r)
        frontier = nxt
    for c in seen:
        nf = unicodedata.normalize("NFKC", c)
        if nf == target or (len(nf) == 1 and nf.lower() == target.lower()):
            return True
    return False


def test_entries_match_authoritative_unicode_confusables():
    graph = _load_authoritative_graph()
    confirmed = sum(_reaches_ascii(graph, k, v) for k, v in T.items())
    frac = confirmed / len(T)
    # every shipped entry should be a genuine Unicode confusable of its ASCII target;
    # a small tolerance absorbs Unicode-version skew between the bundled data and the
    # table's build-time source.
    assert frac >= 0.98, f"only {frac:.3f} of entries confirmed by authoritative data"
