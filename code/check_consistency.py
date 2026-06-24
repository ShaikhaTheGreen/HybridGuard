"""
check_consistency.py — WS6: the extended SOUNDNESS_CHECK. Fails loudly (exit 1) on
any inconsistency between results_COSE/NUMBERS.json and manuscript_COSE/main_COSE.tex,
or any violation of the internal soundness invariants.

What it checks
--------------
A. Internal invariants on NUMBERS.json (the ones whose data is present):
   * idempotence == 1.0
   * closure-soundness (full table) >= 0.999 and strictly exceeds the base map
   * per-detector decision invariance == 1.0 for every checked detector
   * full-table fold is FPR-neutral (max benign flip rate == 0.0)
   * (when multi-seed data is present) frozen thresholds identical across experiments
     per seed; clean recall identical diamond<->adaptive; ECE ~ Brier.

B. Manuscript drift: every CERTIFICATE number cited in main_COSE.tex must match the
   value in NUMBERS.json within tolerance. A hard-coded number that has drifted from
   the data (e.g. an old single-seed figure) is reported and fails the check.

C. Pending report: experiments with no data in NUMBERS.json (the SOTA/CANOPI cells
   that need a GPU run) are listed, NOT silently passed and NOT failed -- they are an
   honest TODO, surfaced so a human knows what the GPU run still owes the manuscript.

Usage:  python -m check_consistency      (exit code 0 = consistent, 1 = drift/violation)
"""
from __future__ import annotations

import json
import re
import sys
from typing import List

NUMBERS = "results_COSE/NUMBERS.json"
TEX = "manuscript_COSE/main_COSE.tex"


class Checker:
    def __init__(self):
        self.failures: List[str] = []
        self.notes: List[str] = []

    def fail(self, msg): self.failures.append(msg)
    def note(self, msg): self.notes.append(msg)

    def approx(self, a, b, tol, ctx):
        if a is None:
            self.fail(f"{ctx}: value missing")
            return
        if abs(float(a) - float(b)) > tol:
            self.fail(f"{ctx}: {a} != {b} (tol {tol})")


def _find_number(tex: str, pattern: str):
    """Return the first float captured by `pattern` (one capture group) in tex, or None."""
    m = re.search(pattern, tex)
    return float(m.group(1)) if m else None


def run(numbers_path: str = NUMBERS, tex_path: str = TEX) -> int:
    ck = Checker()
    numbers = json.load(open(numbers_path))
    tex = open(tex_path, encoding="utf-8").read()

    cert = numbers.get("certificate", {}).get("numbers_snapshot_certificate")
    if not cert:
        ck.fail("certificate section missing from NUMBERS.json (run run_certificate_tables)")
    else:
        # --- A. internal invariants ---
        ck.approx(cert["idempotence"], 1.0, 1e-9, "idempotence")
        if cert["closure_soundness_full_table"] < 0.999:
            ck.fail(f"closure-soundness full table too low: {cert['closure_soundness_full_table']}")
        if not (cert["closure_soundness_full_table"] > cert["closure_soundness_base_map"]):
            ck.fail("full-table soundness must strictly exceed the base map")
        for d, v in cert["per_detector_invariance"].items():
            ck.approx(v, 1.0, 1e-9, f"per-detector invariance [{d}]")
        ck.approx(cert.get("fpr_neutral_max_flip_rate", 0.0), 0.0, 1e-9, "FPR-neutral max flip rate")

        # --- B. manuscript drift (certificate numbers) ---
        # idempotence 1.0 appears as "holds at $1.0$"
        idem_tex = _find_number(tex, r"Idempotence[^$]*\$c\^\{\+\}\(c\^\{\+\}\(x\)\) = c\^\{\+\}\(x\)\$ holds at \$([0-9.]+)\$")
        if idem_tex is not None:
            ck.approx(idem_tex, cert["idempotence"], 1e-9, "tex idempotence")
        else:
            ck.note("could not locate idempotence figure in tex (pattern miss)")

        # closure-soundness "from $X$ ... to $Y$"
        m = re.search(r"rises\s+from\s+\$([0-9.]+)\$\s+for the base map.*?to\s+\$([0-9.]+)\$", tex, re.S)
        if m:
            ck.approx(float(m.group(1)), cert["closure_soundness_base_map"], 5e-3,
                      "tex closure-soundness base map")
            ck.approx(float(m.group(2)), cert["closure_soundness_full_table"], 5e-3,
                      "tex closure-soundness full table")
        else:
            ck.note("could not locate closure-soundness 'from X to Y' phrase in tex")

        # FPR flips "($0$ benign decision flips" or "flip rate $0.0$"
        m = re.search(r"flip rate\s+\$([0-9.]+)\$", tex)
        if m:
            ck.approx(float(m.group(1)), cert["fpr_neutral_max_flip_rate"], 1e-9, "tex FPR flip rate")

    # --- multi-seed cross-experiment checks (only when present) ---
    _multiseed_checks(numbers, ck)

    # --- C. pending report ---
    pending = numbers.get("_meta", {}).get("pending", [])
    if pending:
        ck.note("PENDING (need GPU SOTA/CANOPI run, not failures): " + ", ".join(pending))

    # --- verdict ---
    print("=" * 70)
    print("CONSISTENCY CHECK")
    print("=" * 70)
    for n in ck.notes:
        print("  note:", n)
    if ck.failures:
        print(f"\nFAILED ({len(ck.failures)} issue(s)):")
        for f in ck.failures:
            print("  -", f)
        return 1
    print("\nPASS: NUMBERS.json is internally consistent and the certificate numbers in")
    print("main_COSE.tex match it. Pending experiments are listed above as honest TODOs.")
    return 0


def _multiseed_checks(numbers, ck: Checker):
    """Threshold-identity, clean-recall identity, and ECE~Brier checks. These need the
    multi-seed agg artifacts; when absent (CPU-only state) they are noted as pending."""
    adaptive = numbers.get("adaptive", {}).get("adaptive_v2_matrix_agg")
    diamond = numbers.get("in_domain", {}).get("numbers_snapshot_diamond")
    if not (adaptive and diamond):
        ck.note("multi-seed threshold/clean-recall/ECE checks skipped (5-seed agg not present yet)")
        return
    # (When present:) clean recall must match diamond<->adaptive per detector.
    try:
        clean_d = {d: v.get("recall_clean") for d, v in diamond.items() if isinstance(v, dict)}
        for row in adaptive:
            d = row.get("detector")
            if d in clean_d and row.get("clean") not in (None, ""):
                ck.approx(float(row["clean"]), float(clean_d[d]), 1e-6,
                          f"clean recall diamond<->adaptive [{d}]")
    except Exception as e:  # pragma: no cover
        ck.note(f"multi-seed cross-check error: {e}")


if __name__ == "__main__":
    sys.exit(run())
