#!/usr/bin/env python3
"""
Merge the BigQuery export with the panel votes into posts.csv.

  stream_preds.csv   post_id, stream, round, possibly_sensitive,
                     prediction_llm0/3/4, score_llm0/3/4
  v1_v2_full.csv     id, votes_v1, votes_v2, ...

  -> posts.csv       post_id, stream, round, possibly_sensitive,
                     votes_v1, votes_v2, pred_llm0/3/4, score_llm0/3/4

Usage:
  python3 merge_posts.py stream_preds.csv v1_v2_full.csv posts.csv
"""

import sys
import pandas as pd

POS = {"harassment", "harass", "1", "true", "positive", "yes", "toxic", "abusive"}
NEG = {"neutral", "0", "false", "negative", "no", "none", "benign", "clean"}
STREAM_MAP = {"stream_a_reply_directed": "A", "stream_b_topic": "B",
              "stream_c_lexicon": "C", "stream_d_neutral": "D",
              "stream_e_account": "E", "stream_e_roster": "E"}


def to_binary(s, col):
    v = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=v.index, dtype="Int64")
    out[v.isin(POS)] = 1
    out[v.isin(NEG)] = 0
    bad = v[out.isna()].value_counts()
    if len(bad):
        sys.exit(f"{col}: unmapped values {dict(bad)}\n"
                 f"add them to POS or NEG at the top of this script")
    return out.astype(int)


def short_stream(s):
    v = s.astype(str).str.strip().str.lower()
    out = v.map(STREAM_MAP)
    # fall back: take the letter after 'stream_', else leave as given
    fallback = v.str.extract(r"^stream[_\- ]?([a-e])", expand=False).str.upper()
    out = out.fillna(fallback).fillna(s.astype(str).str.strip().str.upper())
    return out


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    preds_path, votes_path, out_path = sys.argv[1:]

    p = pd.read_csv(preds_path, dtype={"post_id": str})
    v = pd.read_csv(votes_path, dtype={"id": str})
    print(f"predictions {len(p):,}   votes {len(v):,}")

    if p["post_id"].duplicated().any():
        n = int(p["post_id"].duplicated().sum())
        sys.exit(f"{n} duplicate post_id in the export. A post is in more than one "
                 f"stream, or rn>1 rows were included. Decide the rule before merging.")

    p["stream"] = short_stream(p["stream"])
    for c in ("llm0", "llm3", "llm4"):
        p[f"pred_{c}"] = to_binary(p[f"prediction_{c}"], f"prediction_{c}")

    keep = ["id", "votes_v1"] + (["votes_v2"] if "votes_v2" in v.columns else [])
    m = p.merge(v[keep], left_on="post_id", right_on="id", how="inner").drop(columns=["id"])

    print(f"merged {len(m):,}")
    if len(m) < len(v):
        missing = set(v["id"]) - set(p["post_id"])
        print(f"warning: {len(missing):,} posts have votes but no prediction")
    if len(m) < len(p):
        print(f"warning: {len(p) - len(m):,} predictions have no votes row")

    cols = (["post_id", "stream", "round", "possibly_sensitive", "votes_v1"]
            + (["votes_v2"] if "votes_v2" in m.columns else [])
            + [f"pred_{c}" for c in ("llm0", "llm3", "llm4")]
            + [f"score_{c}" for c in ("llm0", "llm3", "llm4") if f"score_{c}" in m.columns])
    m = m[[c for c in cols if c in m.columns]]
    m.to_csv(out_path, index=False)

    print("\nstream x panel-positive (votes_v1 >= 3):")
    x = m.assign(pos=(m["votes_v1"] >= 3).astype(int))
    t = x.groupby("stream").agg(n=("post_id", "size"), prevalence=("pos", "mean"))
    t["prevalence"] = (t["prevalence"] * 100).round(1).astype(str) + "%"
    print(t.to_string())
    print(f"\ntotal {len(m):,}, prevalence {x['pos'].mean() * 100:.1f}%")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
