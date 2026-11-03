#!/usr/bin/env python3
"""
Recompute consensus from stored responses, and emit the three label sets.

Separate from label_corpus.py on purpose. That script returns early when
there is nothing left to call, which is correct for a labelling run and
useless when the labels are already complete and you want the tables again.
Consensus is a pure function of results.jsonl; it should not require the
labelling machinery to run.

Makes ZERO API calls.

OUTPUTS
    data/consensus_<run_id>.csv   one row per text: votes, consensus, per-judge
    labels/LLM0.csv               original researcher labels
    labels/LLM3.csv               consensus at votes >= 3
    labels/LLM4.csv               consensus at votes >= 4

    LLM0/3/4 are the training inputs for the three CoreML classifiers. They
    share the same texts and differ only in the label column, so the label
    set is the only variable in the downstream comparison.

USAGE
    ./env/bin/python consensus.py --run RUN_ID
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"
CORPUS = "data/french_corpus_merged.csv"
JUDGES = ["claude", "gemini", "grok", "mistral", "openai"]


def load(run_id: str) -> pd.DataFrame:
    path = Path("runs") / run_id / "results.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(r.get("raw_response", "")).strip():
                rows.append((str(r.get("text_id")), r.get("judge"),
                             r.get("parsed_label"), r.get("model_version")))
    df = pd.DataFrame(rows, columns=["id", "judge", "label", "version"])
    print(f"{len(df):,} stored responses")
    return df


def main(a):
    raw = load(a.run)

    print("\nper judge")
    for j, g in raw.groupby("judge"):
        pos = (g.label == POS).mean()
        vers = sorted(set(g.version.dropna()))
        print(f"  {j:9s} n={len(g):7,}  flagged {100*pos:5.1f}%  "
              f"abstain {(g.label==ABSTAIN).sum():4,}  {vers}")

    piv = raw.pivot_table(index="id", columns="judge", values="label",
                          aggfunc="first")
    present = [j for j in JUDGES if j in piv.columns]

    corpus = pd.read_csv(CORPUS, dtype={"id": str})
    df = corpus.merge(piv, left_on="id", right_index=True, how="left")

    # An abstention and a missing judge both contribute nothing. Same rule
    # the labelling script uses, so these numbers are directly comparable.
    df["votes"] = sum((df[j] == POS).astype(int) for j in present)
    df["n_judges"] = sum(df[j].notna().astype(int) for j in present)

    orphans = set(piv.index) - set(corpus.id)
    if orphans:
        print(f"\n{len(orphans):,} stored ids are not in the corpus and are "
              f"ignored: {sorted(orphans)[:5]}")

    print(f"\ntexts {len(df):,}")
    print("\njudges per text")
    for k, v in df.n_judges.value_counts().sort_index().items():
        flag = "   <-- incomplete" if k < len(present) else ""
        print(f"  {k}: {v:,}{flag}")

    print("\nagreement bands")
    print(f"{'votes':>7s}{'texts':>9s}{'share':>9s}")
    for k in range(len(present) + 1):
        n = int((df.votes == k).sum())
        print(f"{k:>7d}{n:>9,}{100*n/len(df):>8.1f}%")

    print("\nconsensus thresholds")
    for k in (3, 4, 5):
        n = int((df.votes >= k).sum())
        print(f"  k>={k}: {n:>7,} positives  ({100*n/len(df):5.2f}%)")

    df["consensus"] = (df.votes >= 3).map({True: POS, False: NEG})

    if "label" in df.columns:
        agree = (df.consensus == df.label).mean()
        print(f"\nagreement with the researcher label: {100*agree:.1f}%")
        print(f"  {int((~(df.consensus == df.label)).sum()):,} texts where the "
              f"panel contests the published label")
        # Direction matters: a procedure that only ever adds positives is
        # indistinguishable from a lowered threshold.
        n2p = int(((df.label == NEG) & (df.consensus == POS)).sum())
        p2n = int(((df.label == POS) & (df.consensus == NEG)).sum())
        print(f"  Neutral -> Harassment: {n2p:,}")
        print(f"  Harassment -> Neutral: {p2n:,}")

        print("\ncontested rate by band")
        for k in range(len(present) + 1):
            m = df.votes == k
            if m.sum() == 0:
                continue
            d = (df.loc[m, "consensus"] != df.loc[m, "label"]).mean()
            print(f"  votes {k}: {100*d:5.1f}% contested  ({int(m.sum()):,} texts)")

        if "nb_annotators" in df.columns:
            # The strongest available evidence that panel disagreement tracks
            # annotation quality rather than model noise.
            sub = df[df.nb_annotators.notna()]
            if len(sub):
                print("\ncontested rate by annotator count "
                      f"({len(sub):,} texts with the field)")
                for n, g in sub.groupby("nb_annotators"):
                    d = (g.consensus != g.label).mean()
                    print(f"  {int(n)} annotator(s): {100*d:5.1f}%  "
                          f"({len(g):,} texts)")

    out = Path("data") / f"consensus_{a.run}.csv"
    cols = ["id", "source", "text", "label"] + present + ["votes", "n_judges",
                                                          "consensus"]
    df[[c for c in cols if c in df.columns]].to_csv(out, index=False)
    print(f"\nwrote {out}")

    lab = Path("labels")
    lab.mkdir(exist_ok=True)
    for name, series in (("LLM0", df["label"]),
                         ("LLM3", (df.votes >= 3).map({True: POS, False: NEG})),
                         ("LLM4", (df.votes >= 4).map({True: POS, False: NEG}))):
        t = pd.DataFrame({"id": df.id, "text": df.text, "label": series})
        t = t[t.text.notna()]
        t.to_csv(lab / f"{name}.csv", index=False)
        print(f"  {name}: {len(t):,} rows, "
              f"{100*(t.label==POS).mean():.1f}% {POS}")

    print("\nThese three files differ only in the label column. Train with "
          "identical features and an identical seeded split so the label set "
          "is the only variable.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    main(ap.parse_args())
