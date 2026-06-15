"""
npl_multilingual_experiment.py  —  Linguistic Canonicalization (experiment E6)
=============================================================================
Answers the reviewer/author question: does *bridging other languages to English*
recover prompt-injection detection, and does the same idea fix romanized Arabic / accents?

Thesis extension: "Canonicalize First" generalizes from CHARACTER canonicalization
(homoglyph/zero-width/leet/spacing) to LINGUISTIC canonicalization:
    raw non-English / romanized Arabic  ->  script+diacritic normalize  ->  MT pivot to English
    ->  hand to the SAME English-trained detector.
If recall on Arabic/Spanish/romanized-Arabic injections jumps from ~0 to ~clean after this
stage, the detector-agnostic canonicalization story now covers language shift too.

Pipeline reuse: detectors + frozen 1%-FPR thresholds come from the same harness as
npl_diamond_experiment.py. Translation uses NLLB-200 (runs on the Colab GPU).

------------------------------------------------------------------------------
HOW TO RUN (Colab GPU), after your notebook has X_val/y_val and the trained HG model:

    from npl_multilingual_experiment import run
    run(X_val, y_val,
        ml_sets={                       # your curated multilingual injection sets
            "ar":         (ar_texts,  ar_y),     # Arabic script
            "es":         (es_texts,  es_y),     # Spanish
            # optional: romanized Arabic (Latin-script Arabic). If you only have Arabic script,
            # pass make_romanized=True to auto-generate a romanized-Arabic attack set from "ar".
        },
        hg_detectors={"HG_MULTIFEAT": hg_multifeat},
        out_dir="npl_mling_out")

Outputs (zipped + auto-downloaded): linguistic_recovery.{csv,tex},
numbers_snapshot_mling.json, fig_mling_recovery.{png,pdf}.
"""
import os, json, warnings
import numpy as np
from canonicalize import canonicalize
from npl_diamond_experiment import build_detectors, threshold_at_fpr, _write_csv, _bundle

NLLB_MODEL = "facebook/nllb-200-distilled-600M"
LANG2NLLB  = {"ar": "arb_Arab", "es": "spa_Latn", "fr": "fra_Latn", "en": "eng_Latn"}

# Minimal Arabic<->romanized (Latin-script) character map (numerals stand in for Arabic letters).
_AR2ABZ = {"ا":"a","ب":"b","ت":"t","ث":"th","ج":"j","ح":"7","خ":"5","د":"d","ذ":"z",
           "ر":"r","ز":"z","س":"s","ش":"sh","ص":"9","ض":"d","ط":"6","ظ":"z","ع":"3",
           "غ":"gh","ف":"f","ق":"8","ك":"k","ل":"l","م":"m","ن":"n","ه":"h","و":"w",
           "ي":"y","ى":"a","ة":"a","ء":"2"}

def romanize_arabic(text: str) -> str:
    return "".join(_AR2ABZ.get(c, c) for c in str(text))

# Inverse map (romanized/Latin-script Arabic -> Arabic script), greedy longest-match.
# Multi-char and digit codes first; ambiguous Latin picks the most common Arabic letter.
_ABZ2AR = {
    "th":"ث","sh":"ش","gh":"غ",
    "7":"ح","5":"خ","9":"ص","6":"ط","3":"ع","2":"ء","8":"ق",
    "a":"ا","b":"ب","t":"ت","j":"ج","d":"د","z":"ز","r":"ر",
    "s":"س","f":"ف","k":"ك","l":"ل","m":"م","n":"ن","h":"ه",
    "w":"و","y":"ي",
}
def romanized_to_arabic(text: str) -> str:
    s = str(text).lower(); out = []; i = 0
    while i < len(s):
        two = s[i:i+2]
        if two in _ABZ2AR:
            out.append(_ABZ2AR[two]); i += 2; continue
        one = s[i]
        out.append(_ABZ2AR.get(one, one)); i += 1
    return "".join(out)


def load_translator():
    """Return translate(texts, src_lang)->list[str] (English). GPU if available."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    cuda = torch.cuda.is_available()
    if cuda:
        mdl = mdl.cuda()
    mdl.eval()
    eng_id = tok.convert_tokens_to_ids("eng_Latn")

    def translate(texts, src_lang, bs=16):
        src = LANG2NLLB.get(src_lang, "eng_Latn")
        tok.src_lang = src
        out = []
        for i in range(0, len(texts), bs):
            b = [str(t) for t in texts[i:i+bs]]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=256)
            if cuda:
                enc = {k: v.cuda() for k, v in enc.items()}
            with torch.no_grad():
                gen = mdl.generate(**enc, forced_bos_token_id=eng_id, max_length=256)
            out.extend(tok.batch_decode(gen, skip_special_tokens=True))
        return out
    return translate


def linguistic_canonicalize(texts, src_lang, translate):
    """script/diacritic normalize -> MT pivot to English."""
    norm = [canonicalize(t) for t in texts]      # strips diacritics/homoglyphs (accents)
    return translate(norm, src_lang)


def run(X_val, y_val, ml_sets, hg_detectors=None, out_dir="npl_mling_out",
        load_sota=True, make_romanized=False):
    os.makedirs(out_dir, exist_ok=True)
    det = build_detectors(hg_detectors, load_sota=load_sota)
    assert len(det) >= 1, "need >=1 English-trained detector"
    print("Detectors:", list(det))

    # optionally synthesize a romanized-Arabic attack set from the Arabic set
    sets = dict(ml_sets)
    if make_romanized and "ar" in sets:
        at, ay = sets["ar"]
        sets["ar_roman"] = ([romanize_arabic(t) for t in at], ay)

    translate = load_translator()
    rows, snap = [], {}
    for dname, fn in det.items():
        thr = threshold_at_fpr(y_val, fn(list(X_val)))          # frozen on English val
        snap[dname] = {"threshold": thr}
        for lang, (texts, y) in sets.items():
            y = np.asarray(y); pos = y == 1
            if not pos.any():
                continue
            Xp = [texts[i] for i in np.where(pos)[0]]
            rec_raw = float((fn(Xp) >= thr).mean())
            if lang.startswith("ar_roman"):
                # romanized Arabic: transliterate Latin->Arabic script, THEN MT-pivot to English
                Xc = translate([romanized_to_arabic(canonicalize(t)) for t in Xp], "ar")
            else:
                Xc = linguistic_canonicalize(Xp, "ar" if lang.startswith("ar") else lang, translate)
            rec_can = float((fn(Xc) >= thr).mean())
            rows.append(dict(detector=dname, lang=lang,
                             recall_raw=round(rec_raw, 4),
                             recall_canon=round(rec_can, 4),
                             gain=round(rec_can - rec_raw, 4)))
            snap[dname][lang] = {"raw": rec_raw, "canon": rec_can}
            print(f"  {dname:18s} {lang:12s} raw={rec_raw:.3f} -> canon={rec_can:.3f}")

    _write_csv(os.path.join(out_dir, "linguistic_recovery.csv"), rows)
    _emit_tex(rows, os.path.join(out_dir, "linguistic_recovery.tex"))
    _figure(rows, os.path.join(out_dir, "fig_mling_recovery"))
    json.dump(snap, open(os.path.join(out_dir, "numbers_snapshot_mling.json"), "w"), indent=2)
    bundle = _bundle(out_dir)
    print("\nDONE ->", out_dir, "| bundle:", bundle)
    try:
        from google.colab import files  # type: ignore
        files.download(bundle)
    except Exception:
        pass
    return out_dir


def _emit_tex(rows, path):
    langs = sorted({r["lang"] for r in rows}); dets = sorted({r["detector"] for r in rows})
    L = [r"\begin{table}[ht]\centering",
         r"\caption{Linguistic canonicalization. Recall@1\%FPR on non-English injections, raw vs after script-normalization + MT pivot to English. The English-trained detector recovers detection across languages.}",
         r"\label{tab:mling_recovery}", r"\small",
         r"\begin{tabular}{ll" + "c"*len(dets) + "}", r"\toprule",
         "Lang & Cond. & " + " & ".join(dets) + r" \\", r"\midrule"]
    for lang in langs:
        for cond, key in [("raw", "recall_raw"), ("+canon", "recall_canon")]:
            cells = [f"{next((r[key] for r in rows if r['lang']==lang and r['detector']==d), float('nan')):.3f}" for d in dets]
            L.append(f"{lang if cond=='raw' else ''} & {cond} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(path, "w").write("\n".join(L))


def _figure(rows, stem):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        dets = sorted({r["detector"] for r in rows}); langs = sorted({r["lang"] for r in rows})
        fig, axes = plt.subplots(1, len(dets), figsize=(4*len(dets), 3.2), sharey=True)
        if len(dets) == 1:
            axes = [axes]
        for ax, d in zip(axes, dets):
            raw = [next(r["recall_raw"] for r in rows if r["lang"]==l and r["detector"]==d) for l in langs]
            can = [next(r["recall_canon"] for r in rows if r["lang"]==l and r["detector"]==d) for l in langs]
            x = np.arange(len(langs))
            ax.bar(x - 0.2, raw, 0.4, label="raw")
            ax.bar(x + 0.2, can, 0.4, label="+canon")
            ax.set_title(d); ax.set_xticks(x); ax.set_xticklabels(langs, rotation=30, ha="right"); ax.set_ylim(0, 1)
        axes[0].set_ylabel("Recall @ 1% FPR"); axes[-1].legend()
        fig.tight_layout(); fig.savefig(stem + ".png", dpi=160); fig.savefig(stem + ".pdf")
    except Exception as e:
        warnings.warn(f"figure skipped: {e}")
