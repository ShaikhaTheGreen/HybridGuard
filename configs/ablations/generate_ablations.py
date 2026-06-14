"""
Generate the E2 ablation configs from the CANOPI main config.

One YAML per ablation (the repo's per-experiment-config convention), written as
plain text so this runs without PyYAML. Reproduce with:

    python configs/ablations/generate_ablations.py

Ablations (each reports Delta R@1%FPR vs canopi_main, CI must exclude 0 to "matter"):
  loss term off    : no_inv, no_hardneg, no_drift, no_pauc
  augmentation off : no_paraphrase, no_persona, no_encoding, no_crosslingual
  encoder          : finetune (unfreeze E), monolingual (EN-only encoder)
  head depth       : depth1, depth3   (main is depth2)
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
MULTI = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MONO = "sentence-transformers/all-MiniLM-L6-v2"

BASE = dict(
    seeds="[42, 2025, 7, 1337, 314]",
    encoder_model=MULTI,
    head_depth=2,
    freeze=True,
    lam=dict(inv=1.0, hardneg=0.5, drift=0.5, pauc=1.0),
    families='[paraphrase, persona, "encoding:homoglyph", "encoding:base64"]',
    crosslingual="[ar, es, fr, de, zh, hi, ru, pt]",   # match canopi_main diverse-8 panel
)

# (name, description, mutation dict)
ABLATIONS = [
    ("no_inv", "L_inv off (lam1=0)", {"lam.inv": 0.0}),
    ("no_hardneg", "L_hardneg off (lam2=0)", {"lam.hardneg": 0.0}),
    ("no_drift", "L_drift off (lam3=0)", {"lam.drift": 0.0}),
    ("no_pauc", "L_pAUC off (lam4=0) == B6", {"lam.pauc": 0.0}),
    ("no_paraphrase", "paraphrase view family off", {"families": '[persona, "encoding:homoglyph", "encoding:base64"]'}),
    ("no_persona", "persona view family off", {"families": '[paraphrase, "encoding:homoglyph", "encoding:base64"]'}),
    ("no_encoding", "encoding view families off", {"families": "[paraphrase, persona]"}),
    ("no_crosslingual", "no NLLB AR/ES views", {"crosslingual": "[]"}),
    ("finetune", "unfreeze encoder E", {"freeze": False}),
    ("monolingual", "EN-only encoder", {"encoder_model": MONO}),
    ("depth1", "projection head depth 1", {"head_depth": 1}),
    ("depth3", "projection head depth 3", {"head_depth": 3}),
]

TEMPLATE = """# ABLATION: {desc}. Compare Delta R@1%FPR vs canopi_main (CI must exclude 0).
name: ablation_{name}
seeds: {seeds}
split: {{scheme: "60/20/20", seed: 1337, stratified: true}}
encoder:
  backend: sentence-transformers
  model_name: {encoder_model}
  canonicalize_first: true
  cache: true
model: {{head_depth: {head_depth}, hidden: 256, out_dim: 128, dropout: 0.1, freeze_encoder: {freeze}}}
loss:  {{lam1_inv: {inv}, lam2_hardneg: {hardneg}, lam3_drift: {drift}, lam4_pauc: {pauc}, tau: 0.1, hardneg_margin: 0.2, beta: 0.01, pauc_margin: 1.0}}
augment:
  families: {families}
  crosslingual: {crosslingual}
  crosslingual_max: 300
  nllb: facebook/nllb-200-distilled-600M
  intent_filter: true
  tau_keep: 0.5
train: {{epochs: 30, batch_size: 256, lr: 0.001, wd: 0.0001}}
eval:  {{target_fpr: 0.01, bootstrap_resamples: 1000, bootstrap_seed: 1337}}
"""


def _apply(mut: dict) -> dict:
    cfg = dict(BASE)
    cfg["lam"] = dict(BASE["lam"])
    for k, v in mut.items():
        if k.startswith("lam."):
            cfg["lam"][k.split(".", 1)[1]] = v
        else:
            cfg[k] = v
    return cfg


def main():
    for name, desc, mut in ABLATIONS:
        c = _apply(mut)
        text = TEMPLATE.format(
            desc=desc, name=name, seeds=c["seeds"], encoder_model=c["encoder_model"],
            head_depth=c["head_depth"], freeze=str(c["freeze"]).lower(),
            inv=c["lam"]["inv"], hardneg=c["lam"]["hardneg"], drift=c["lam"]["drift"], pauc=c["lam"]["pauc"],
            families=c["families"], crosslingual=c["crosslingual"],
        )
        (HERE / f"{name}.yaml").write_text(text, encoding="utf-8")
        print(f"wrote {name}.yaml — {desc}")


if __name__ == "__main__":
    main()
