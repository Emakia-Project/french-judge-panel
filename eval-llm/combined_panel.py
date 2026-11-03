#!/usr/bin/env python3
"""
combined_panel.py — LLM0/LLM3/LLM4 against a COMBINED reference that adds
Gemini Vision and X's platform flag to the five LLM judges, broken down by
vote band, under both criteria.

    python combined_panel.py \
        --v1 data/consensus_fixed_2026-09-04_0423__..._b4d9cce4.csv \
        --v2 data/consensus_2026-09-08_0035__all_8436_8436__..._2709a522.csv \
        --joined paper/joined_for_tables.csv \
        --vision runs_vision/20260904T213938Z/results.jsonl \
        --outdir paper

THE COMBINED REFERENCE

    Five LLM judges, one vote each, plus Gemini Vision and the platform flag
    at one vote each: seven signals.

    Vision only votes when media is attached, and 76% of the corpus is
    text-only. A fixed threshold over seven therefore means a stricter bar
    for text-only posts than for media-bearing ones -- the structural
    problem the English design handled with "max 10, or 8 when Vision
    abstains". Two readings are reported:

      FIXED        k of 7, accepting that text-only tops out at 6
      PROPORTIONAL share of AVAILABLE votes, so abstention does not penalise

    Both are shown because the choice changes what the top band means, and
    neither is obviously right.

WHAT THIS COSTS

    Nothing is free here. Folding vision and the flag into the reference
    means neither can be evaluated AGAINST it any more. The 8.6% of
    media-bearing posts that vision catches and the text panel misses is
    measurable only while vision stays outside. This script therefore
    reports the LLM-only reference alongside, so the two can be compared and
    the cost is visible rather than assumed.

    Prior measurement, which predicts what folding them in will do:
      - the flag fires on 1.70% of unanimous-neutral texts and 0.31% of
        unanimous-harassment texts, so it is ANTI-correlated with panel
        severity on this corpus
      - vision tracks the panel monotonically (8.8% -> 42.6% across bands)
        but only reaches the ~24% of posts carrying media
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POS, NEG = "Harassment", "Neutral"
MODELS = ["llm0", "llm3", "llm4"]


def metrics(pred: pd.Series, ref: pd.Series) -> dict:
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum()); tn = int((~pred & ~ref).sum())
    n = tp + fp + fn + tn
    base = (tp + fn) / n if n else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    triv = 2 * base / (1 + base) if base else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    return dict(n=n, base=base, precision=p, recall=r, f1=f, trivial=triv,
                lift=f - triv, mcc=mcc, agree=(tp + tn) / n if n else 0.0,
                tp=tp, fp=fp, fn=fn, tn=tn)


def load_vision(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    v = pd.DataFrame(rows)
    v = v[v.tweet_id.notna()].copy()
    v["tweet_id"] = v.tweet_id.astype(str)
    # A post counts as vision-flagged if ANY of its media is flagged: the
    # question is whether abuse is present in the media, not in all of it.
    return v.groupby("tweet_id").agg(
        n_media=("media_key", "size"),
        vis_pos=("verdict", lambda s: bool((s == POS).any())),
    ).reset_index().rename(columns={"tweet_id": "id"})


def build_combined(d: pd.DataFrame, votecol: str) -> pd.DataFrame:
    """Add the two auxiliary signals as one vote each."""
    d = d.copy()
    d["vis_vote"] = d.vis_pos.fillna(False).astype(bool).astype(int)
    d["x_vote"] = d.x_flag.fillna(False).astype(bool).astype(int)
    d["has_media"] = d.n_media.notna()

    d["combined_votes"] = d[votecol] + d.vis_vote + d.x_vote
    # Vision abstains on text-only posts, so available votes differ per row.
    d["available"] = 5 + 1 + d.has_media.astype(int)
    d["combined_share"] = d.combined_votes / d.available
    return d


def band_table(d: pd.DataFrame, votecol: str, ref: pd.Series, label: str,
               out: Path, tag: str):
    """Classifiers against a boolean reference Series, by LLM-only vote band."""
    print("\n" + "=" * 96)
    print(f"{label}")
    print("=" * 96)
    ref = ref.reindex(d.index).astype(bool)
    print(f"  reference positive rate {100*ref.mean():.1f}%  "
          f"({int(ref.sum()):,} of {len(d):,})\n")

    print(f"  {'votes':>6s}{'n':>7s}{'ref POS':>9s}   " +
          "   ".join(f"{m.upper():>22s}" for m in MODELS))
    print(f"  {'':>6s}{'':>7s}{'':>9s}   " +
          "   ".join(f"{'agree   P     R   MCC':>22s}" for m in MODELS))

    rows = []
    for v in range(6):
        g = d[d[votecol] == v]
        if g.empty:
            continue
        gref = ref.loc[g.index]
        cells, rec = [], {"votes": v, "n": len(g),
                          "ref_pos_rate": float(gref.mean())}
        for m in MODELS:
            c = metrics(g[f"prediction_{m}"].eq(POS), gref)
            cells.append(f"{100*c['agree']:5.1f}% {c['precision']:5.3f} "
                         f"{c['recall']:5.3f} {c['mcc']:+5.2f}")
            rec[f"{m}_agree"] = c["agree"]; rec[f"{m}_mcc"] = c["mcc"]
            rec[f"{m}_precision"] = c["precision"]; rec[f"{m}_recall"] = c["recall"]
        rows.append(rec)
        print(f"  {v:>6d}{len(g):>7,}{100*gref.mean():8.1f}%   " + "   ".join(cells))

    # whole corpus
    cells = []
    for m in MODELS:
        c = metrics(d[f"prediction_{m}"].eq(POS), ref)
        cells.append(f"{100*c['agree']:5.1f}% {c['precision']:5.3f} "
                     f"{c['recall']:5.3f} {c['mcc']:+5.2f}")
    print(f"  {'ALL':>6s}{len(d):>7,}{100*ref.mean():8.1f}%   " + "   ".join(cells))

    t = pd.DataFrame(rows)
    t.to_csv(out / f"combined_{tag}.csv", index=False)
    return t


def compare_references(d: pd.DataFrame, votecol: str, llm_ref: pd.Series,
                       comb_ref: pd.Series, label: str):
    """What did adding the two signals actually change?"""
    moved = int((llm_ref != comb_ref).sum())
    to_pos = int(((~llm_ref) & comb_ref).sum())
    to_neg = int((llm_ref & (~comb_ref)).sum())
    print(f"\n  Adding vision and the platform flag moved {moved:,} of "
          f"{len(d):,} labels ({100*moved/len(d):.2f}%)")
    print(f"    -> Harassment: {to_pos:,}     -> Neutral: {to_neg:,}")
    if moved:
        print("\n  where the movement lands, by LLM-only vote band:")
        for v in range(6):
            g = d[d[votecol] == v]
            if g.empty:
                continue
            mv = int((llm_ref[g.index] != comb_ref[g.index]).sum())
            if mv:
                print(f"    {v} votes: {mv:,} of {len(g):,} "
                      f"({100*mv/len(g):.1f}%)")
    if moved / len(d) < 0.03:
        print("\n  Under 3%. The combined reference is close to the LLM-only")
        print("  one, so folding the two signals in buys little -- and it")
        print("  costs the ability to evaluate them independently, which is")
        print("  how their behaviour was measured in the first place.")


def run_one(d: pd.DataFrame, votecol: str, consensus_col: str,
            criterion: str, a, out: Path):
    print("\n" + "#" * 96)
    print(f"# CRITERION {criterion.upper()}")
    print("#" * 96)

    d = build_combined(d, votecol)

    llm_ref = d[consensus_col].eq(POS)
    fixed_ref = d.combined_votes >= a.k
    prop_ref = d.combined_share >= a.share

    print(f"\n  signal availability: {int(d.has_media.sum()):,} posts carry "
          f"media ({100*d.has_media.mean():.1f}%), so vision votes on those "
          f"only.")
    print(f"  platform flag fires on {int(d.x_vote.sum()):,} "
          f"({100*d.x_vote.mean():.2f}%); vision flags "
          f"{int(d.vis_vote.sum()):,} ({100*d.vis_vote.mean():.2f}%).")

    print("\n  combined vote distribution (5 judges + vision + flag):")
    print(f"  {'votes':>6s}{'n':>8s}{'share':>8s}")
    for v in sorted(d.combined_votes.unique()):
        g = d[d.combined_votes == v]
        print(f"  {int(v):>6d}{len(g):>8,}{100*len(g)/len(d):7.1f}%")

    band_table(d, votecol, llm_ref,
               f"TABLE {criterion.upper()}-1 — reference: five LLM judges only "
               f"(k>={a.k_llm})", out, f"{criterion}_llm_only")

    band_table(d, votecol, fixed_ref,
               f"TABLE {criterion.upper()}-2 — reference: COMBINED, fixed "
               f"k>={a.k} of 7 (text-only posts top out at 6)",
               out, f"{criterion}_combined_fixed")

    band_table(d, votecol, prop_ref,
               f"TABLE {criterion.upper()}-3 — reference: COMBINED, "
               f"proportional >={a.share:.2f} of available votes",
               out, f"{criterion}_combined_prop")

    compare_references(d, votecol, llm_ref, fixed_ref, criterion)
    return d


def main(a):
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    v1 = pd.read_csv(a.v1, dtype={"id": str})
    v2 = pd.read_csv(a.v2, dtype={"id": str})
    j = pd.read_csv(a.joined, dtype={"id": str})
    for x in (v1, v2, j):
        x["id"] = x["id"].astype(str)

    pred_cols = [c for c in j.columns if c.startswith("prediction_")]
    keep_j = ["id"] + pred_cols + [c for c in ("stream", "x_flag",
                                               "possibly_sensitive")
                                   if c in j.columns]
    d = (v1[["id", "votes", "consensus"]]
         .rename(columns={"votes": "votes_v1", "consensus": "cons_v1"})
         .merge(v2[["id", "votes", "consensus"]]
                .rename(columns={"votes": "votes_v2", "consensus": "cons_v2"}),
                on="id")
         .merge(j[keep_j], on="id"))

    if "x_flag" not in d.columns:
        d["x_flag"] = d.get("possibly_sensitive", False)

    if a.vision:
        d = d.merge(load_vision(Path(a.vision)), on="id", how="left")
    else:
        d["n_media"] = np.nan; d["vis_pos"] = False

    print(f"{len(d):,} texts with both criteria, CoreML predictions, "
          f"vision and the platform flag")

    run_one(d, "votes_v1", "cons_v1", "v1", a, out)
    run_one(d, "votes_v2", "cons_v2", "v2", a, out)

    print("\n" + "=" * 96)
    print("READING THESE")
    print("=" * 96)
    print("  Rows are the five-LLM vote band, so band 5 means all five judges")
    print("  said Harassment. Columns are each classifier's agreement,")
    print("  precision, recall and MCC against the reference named in the")
    print("  table heading.")
    print("\n  Precision and recall inside a single band are unstable where the")
    print("  band is nearly all one class: at 0 votes almost every reference")
    print("  label is Neutral, so recall is computed over a handful of")
    print("  positives. Agreement and MCC are the safer reads there.")
    print("\n  Every figure is agreement with a panel of models, not accuracy.")
    print("  The panel is not ground truth, and folding vision and the flag")
    print("  into it does not make it one.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--joined", default="paper/joined_for_tables.csv")
    ap.add_argument("--vision")
    ap.add_argument("--k", type=int, default=4,
                    help="combined threshold, fixed count of 7 (default 4)")
    ap.add_argument("--k-llm", type=int, default=3,
                    help="LLM-only threshold, for the comparison table")
    ap.add_argument("--share", type=float, default=0.5,
                    help="combined threshold as a share of available votes")
    ap.add_argument("--outdir", default="paper")
    main(ap.parse_args())
