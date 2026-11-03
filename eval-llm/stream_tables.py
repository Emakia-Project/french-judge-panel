#!/usr/bin/env python3
"""
Per-stream breakdown of classifier agreement and metrics.

Input: one CSV, one row per post.

  Required : post_id, stream, votes_v1, pred_llm0, pred_llm3, pred_llm4
  Optional : votes_v2            -> every table is also produced for criterion v2
             score_llm0/3/4      -> reported only, not used in the tables

Panel label = 1 if votes >= K (default 3) else 0.
Agreement   = prediction equals panel label.

Outputs, written to --outdir:
  table_bands_<crit>.md / .tex      full 6-band grid, per stream, per classifier
  table_bands3_<crit>.md / .tex     collapsed 3-band grid for the body text
  table_metrics_<crit>.md / .tex    F1 / Trivial / Lift / MCC / P / R per stream

Usage:
  python3 stream_tables.py posts.csv --outdir out
  python3 stream_tables.py posts.csv --outdir out --bootstrap 10000
"""

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd

CLASSIFIERS = ["llm0", "llm3", "llm4"]
REQUIRED = ["post_id", "stream", "votes_v1"] + [f"pred_{c}" for c in CLASSIFIERS]
MIN_CELL = 50          # cells below this are flagged as unreliable
FLAG = "\u2020"        # dagger


# ---------------------------------------------------------------- loading

def load(path, k):
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"missing required column(s): {', '.join(missing)}")

    df["stream"] = df["stream"].astype(str).str.strip().str.upper()

    crits = ["v1"] + (["v2"] if "votes_v2" in df.columns else [])
    for c in crits:
        v = pd.to_numeric(df[f"votes_{c}"], errors="coerce")
        if v.isna().any():
            sys.exit(f"votes_{c} has {int(v.isna().sum())} non-numeric or empty values")
        if not v.between(0, 5).all():
            sys.exit(f"votes_{c} outside 0-5")
        df[f"votes_{c}"] = v.astype(int)
        df[f"panel_{c}"] = (v >= k).astype(int)

    for c in CLASSIFIERS:
        p = pd.to_numeric(df[f"pred_{c}"], errors="coerce")
        if not p.isin([0, 1]).all():
            sys.exit(f"pred_{c} must be 0/1")
        df[f"pred_{c}"] = p.astype(int)

    if df["post_id"].duplicated().any():
        n = int(df["post_id"].duplicated().sum())
        print(f"warning: {n} duplicate post_id values", file=sys.stderr)

    return df, crits


# ---------------------------------------------------------------- metrics

def confusion(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return (int(((y == 1) & (p == 1)).sum()),
            int(((y == 0) & (p == 1)).sum()),
            int(((y == 0) & (p == 0)).sum()),
            int(((y == 1) & (p == 0)).sum()))


def metrics(y, p):
    tp, fp, tn, fn = confusion(y, p)
    n = tp + fp + tn + fn
    if n == 0:
        return {k: float("nan") for k in
                ("n", "prev", "f1", "trivial", "lift", "mcc", "prec", "rec", "flag_rate")}
    prev = (tp + fn) / n
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    trivial = 2 * prev / (1 + prev) if prev else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    return {"n": n, "prev": prev, "f1": f1, "trivial": trivial,
            "lift": f1 - trivial, "mcc": mcc, "prec": prec, "rec": rec,
            "flag_rate": (tp + fp) / n}


def mcc_of(y, p):
    return metrics(y, p)["mcc"]


def paired_bootstrap(y, pa, pb, reps, seed=0):
    """CI for MCC(a) - MCC(b), resampling posts."""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y, pa, pb))
    n = len(y)
    d = np.empty(reps)
    for i in range(reps):
        idx = rng.integers(0, n, n)
        d[i] = mcc_of(y[idx], pa[idx]) - mcc_of(y[idx], pb[idx])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return mcc_of(y, pa) - mcc_of(y, pb), lo, hi, float((d <= 0).mean())


# ---------------------------------------------------------------- tables

BAND3 = {0: "0 unanimous neutral", 1: "1-2 majority neutral", 2: "1-2 majority neutral",
         3: "3-4 contested", 4: "3-4 contested", 5: "5 unanimous harassment"}
BAND3_ORDER = ["0 unanimous neutral", "1-2 majority neutral",
               "3-4 contested", "5 unanimous harassment"]
BAND6 = {0: "0 unanimous neutral", 1: "1", 2: "2",
         3: "3 minimum threshold", 4: "4", 5: "5 unanimous harassment"}
BAND6_ORDER = [BAND6[i] for i in range(6)]


def agreement_grid(df, crit, band_map, band_order):
    """rows = band x stream, cols = per-classifier agreement %."""
    d = df.copy()
    d["band"] = d[f"votes_{crit}"].map(band_map)
    streams = sorted(d["stream"].unique())
    rows = []
    for band in band_order:
        for stream in streams + ["All"]:
            sub = d[d["band"] == band]
            if stream != "All":
                sub = sub[sub["stream"] == stream]
            if len(sub) == 0:
                continue
            row = {"Band": band, "Stream": stream, "n": len(sub)}
            for c in CLASSIFIERS:
                agree = (sub[f"pred_{c}"] == sub[f"panel_{crit}"]).mean() * 100
                row[c.upper()] = f"{agree:.1f}" + (FLAG if len(sub) < MIN_CELL else "")
            rows.append(row)
    # overall
    for stream in streams + ["All"]:
        sub = d if stream == "All" else d[d["stream"] == stream]
        row = {"Band": "All bands", "Stream": stream, "n": len(sub)}
        for c in CLASSIFIERS:
            row[c.upper()] = f"{(sub[f'pred_{c}'] == sub[f'panel_{crit}']).mean() * 100:.1f}"
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_table(df, crit):
    streams = sorted(df["stream"].unique())
    rows = []
    for stream in streams + ["All"]:
        sub = df if stream == "All" else df[df["stream"] == stream]
        y = sub[f"panel_{crit}"].values
        for c in CLASSIFIERS:
            m = metrics(y, sub[f"pred_{c}"].values)
            rows.append({"Stream": stream, "n": m["n"],
                         "Base": f"{m['prev'] * 100:.1f}%",
                         "Variant": c.upper(),
                         "F1": f"{m['f1']:.3f}",
                         "Trivial": f"{m['trivial']:.3f}",
                         "Lift": f"{m['lift']:+.3f}",
                         "MCC": f"{m['mcc']:+.3f}",
                         "P": f"{m['prec']:.3f}",
                         "R": f"{m['rec']:.3f}",
                         "Flag": f"{m['flag_rate'] * 100:.1f}%"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- output

def to_latex(df, caption, label):
    cols = list(df.columns)
    spec = "l" * 2 + "r" * (len(cols) - 2)
    out = ["\\begin{table*}[t]", "\\centering", "\\small",
           f"\\begin{{tabular}}{{{spec}}}", "\\toprule",
           " & ".join(f"\\textbf{{{c}}}" for c in cols) + " \\\\", "\\midrule"]
    prev = None
    for _, r in df.iterrows():
        if prev is not None and r[cols[0]] != prev:
            out.append("\\midrule")
        prev = r[cols[0]]
        out.append(" & ".join(str(r[c]).replace("%", "\\%").replace(FLAG, "$\\dagger$")
                              for c in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{table*}"]
    return "\n".join(out)


def to_markdown(df):
    """Plain-text markdown table; avoids the optional tabulate dependency."""
    cols = [str(c) for c in df.columns]
    rows = [[str(v) for v in r] for r in df.astype(str).values]
    w = [max(len(c), *(len(r[i]) for r in rows)) if rows else len(c)
         for i, c in enumerate(cols)]
    line = lambda cells: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
    out = [line(cols), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
    out += [line(r) for r in rows]
    return "\n".join(out)


def write(df, outdir, stem, caption, label):
    df.to_csv(os.path.join(outdir, stem + ".csv"), index=False)
    with open(os.path.join(outdir, stem + ".md"), "w") as f:
        f.write(f"**{caption}**\n\n" + to_markdown(df) + "\n")
    with open(os.path.join(outdir, stem + ".tex"), "w") as f:
        f.write(to_latex(df, caption, label) + "\n")
    print(f"  {stem}.md / .tex / .csv")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="out")
    ap.add_argument("-k", type=int, default=3, help="consensus threshold (default 3)")
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="reps for the paired MCC bootstrap (e.g. 10000)")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    df, crits = load(a.csv, a.k)
    print(f"{len(df):,} posts, streams {sorted(df['stream'].unique())}, criteria {crits}\n")

    for crit in crits:
        print(f"criterion {crit}:")
        write(agreement_grid(df, crit, BAND6, BAND6_ORDER), a.outdir,
              f"table_bands_{crit}",
              f"Classifier agreement with the panel by stream and vote band, "
              f"criterion {crit}. {FLAG} marks cells with n < {MIN_CELL}.",
              f"tab:bands-{crit}")
        write(agreement_grid(df, crit, BAND3, BAND3_ORDER), a.outdir,
              f"table_bands3_{crit}",
              f"Agreement by stream, bands collapsed, criterion {crit}.",
              f"tab:bands3-{crit}")
        write(metrics_table(df, crit), a.outdir, f"table_metrics_{crit}",
              f"Out-of-domain performance by stream, criterion {crit}. "
              f"Trivial is the all-positive baseline at that stream's base rate.",
              f"tab:metrics-{crit}")
        print()

    if a.bootstrap:
        print(f"paired bootstrap, {a.bootstrap:,} reps, MCC differences\n")
        for crit in crits:
            for stream in sorted(df["stream"].unique()) + ["All"]:
                sub = df if stream == "All" else df[df["stream"] == stream]
                y = sub[f"panel_{crit}"].values
                d, lo, hi, p = paired_bootstrap(y, sub["pred_llm3"].values,
                                                sub["pred_llm4"].values, a.bootstrap)
                star = "" if lo <= 0 <= hi else "  *"
                print(f"  {crit} {stream:>3}  LLM3-LLM4  {d:+.3f}  "
                      f"[{lo:+.3f}, {hi:+.3f}]  p={p:.3f}{star}")
            print()


if __name__ == "__main__":
    main()
