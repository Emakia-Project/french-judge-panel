#!/usr/bin/env python3
"""
The 1,516 duplicate texts are a free natural experiment.

The same text appears in two source datasets, annotated INDEPENDENTLY by two
different research teams under two different protocols. Where the two
published labels disagree, at least one is wrong, and no LLM is involved in
establishing that.

This is the external anchor the design was missing. It is not a proxy or a
model opinion: it is two sets of human researchers looking at identical text
and reaching different conclusions.

FOUR THINGS THIS MEASURES
  1. Human-human agreement across published corpora, with no model in the
     loop. A ceiling on what any label set can be worth.
  2. Whether the panel sides with one source systematically.
  3. Whether panel disagreement PREDICTS researcher disagreement. If texts
     the panel finds contested are also the texts where two research teams
     disagreed, the panel is detecting real ambiguity rather than producing
     noise. That is convergent validity from outside the system.
  4. Leakage: duplicates spanning train and test inflate every metric.

USAGE
    ./env/bin/python duplicate_audit.py --run RUN_ID
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

POS, NEG = "Harassment", "Neutral"


def norm(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.lower()
            .str.replace(r"http\S+", " ", regex=True)
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())


def main(a):
    f = Path("data") / f"consensus_{a.run}.csv"
    d = pd.read_csv(f, dtype={"id": str})
    d["_k"] = norm(d.text)
    d = d[d._k.str.len() > 3]           # ignore degenerate keys

    dup_keys = d._k[d._k.duplicated(keep=False)]
    dups = d[d._k.isin(set(dup_keys))].copy()
    print(f"{len(d):,} texts   {dups._k.nunique():,} texts appear more than "
          f"once ({len(dups):,} rows)\n")
    if dups.empty:
        return

    g = dups.groupby("_k")
    rows = []
    for k, grp in g:
        labs = set(grp.label.dropna())
        rows.append({
            "key": k,
            "n": len(grp),
            "sources": " + ".join(sorted(set(grp.source))),
            "labels_agree": len(labs) <= 1,
            "labels": " / ".join(sorted(labs)),
            "votes": grp.votes.iloc[0],
            "consensus": grp.consensus.iloc[0],
            "text": grp.text.iloc[0],
        })
    r = pd.DataFrame(rows)

    # ------------------------------------------------------------ 1
    print("=" * 70)
    print("1. Do the published labels agree, human to human?")
    print("=" * 70)
    cross = r[r.sources.str.contains(r"\+")]
    print(f"  {len(r):,} duplicated texts")
    print(f"  {len(cross):,} span TWO DIFFERENT source datasets")
    if len(cross):
        agree = cross.labels_agree.mean()
        print(f"\n  researcher-researcher agreement: {100*agree:.1f}%")
        print(f"  {int((~cross.labels_agree).sum()):,} texts where two teams "
              f"published OPPOSITE labels for identical text")
        print("\n  by source pair:")
        for pair, gg in cross.groupby("sources"):
            print(f"    {pair:45s} {100*gg.labels_agree.mean():5.1f}% agree "
                  f"(n={len(gg):,})")

    # ------------------------------------------------------------ 2
    print("\n" + "=" * 70)
    print("2. Does panel disagreement PREDICT researcher disagreement?")
    print("=" * 70)
    if len(cross):
        # The panel never saw that these texts were duplicated, so this is a
        # genuine out-of-sample test of whether its uncertainty is meaningful.
        print(f"  {'votes':>7s}{'texts':>9s}{'researchers agree':>20s}")
        for v in range(6):
            m = cross.votes == v
            if m.sum() < 10:
                continue
            print(f"{v:>7d}{int(m.sum()):>9,}"
                  f"{100*cross.loc[m,'labels_agree'].mean():>19.1f}%")
        mid = cross[cross.votes.isin([2, 3])]
        ext = cross[cross.votes.isin([0, 5])]
        if len(mid) >= 10 and len(ext) >= 10:
            print(f"\n  contested bands (2-3): researchers agree "
                  f"{100*mid.labels_agree.mean():.1f}%  (n={len(mid):,})")
            print(f"  extreme bands  (0,5): researchers agree "
                  f"{100*ext.labels_agree.mean():.1f}%  (n={len(ext):,})")
            gap = ext.labels_agree.mean() - mid.labels_agree.mean()
            print(f"  gap: {100*gap:+.1f} points")
            if gap > 0.05:
                print("\n  The panel's contested band is ALSO where independent")
                print("  research teams disagreed. That is convergent validity")
                print("  from outside the system, and it is the strongest")
                print("  evidence available that the middle is genuinely")
                print("  ambiguous rather than merely hard for models.")

    # ------------------------------------------------------------ 3
    print("\n" + "=" * 70)
    print("3. Where the two teams disagreed, who does the panel side with?")
    print("=" * 70)
    conflict = cross[~cross.labels_agree]
    if len(conflict):
        for pair, gg in conflict.groupby("sources"):
            srcs = pair.split(" + ")
            print(f"\n  {pair}  (n={len(gg):,})")
            for s in srcs:
                sub = dups[(dups.source == s) & (dups._k.isin(set(gg.key)))]
                if len(sub):
                    agree_panel = (sub.label.values ==
                                   sub.consensus.values).mean()
                    print(f"    panel agrees with {s:28s} "
                          f"{100*agree_panel:5.1f}%")
        print("\n  examples where two research teams published opposite "
              "labels:")
        for _, x in conflict.head(6).iterrows():
            print(f"    [{x.labels}] panel={x.consensus} votes={x.votes}")
            print(f"      {str(x.text)[:105]}")

    # ------------------------------------------------------------ 4
    print("\n" + "=" * 70)
    print("4. Leakage")
    print("=" * 70)
    print(f"  {len(dups) - r.n.count():,} rows would be removed by keeping one")
    print(f"  copy of each text. Deduplicate on the normalised key BEFORE the")
    print(f"  split, or {r.n.sum() - len(r):,} texts appear in both train and")
    print(f"  test and every metric is inflated.")

    out = Path("data/duplicate_conflicts.csv")
    if len(conflict):
        conflict[["text", "sources", "labels", "votes", "consensus"]].to_csv(
            out, index=False)
        print(f"\n  wrote {out} for manual inspection")

    keep = d.drop_duplicates("_k", keep="first")
    dedup = Path("data/french_corpus_dedup.csv")
    keep.drop(columns=["_k"]).to_csv(dedup, index=False)
    print(f"  wrote {dedup} ({len(keep):,} unique texts)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    main(ap.parse_args())
