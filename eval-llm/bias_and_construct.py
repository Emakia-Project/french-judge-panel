#!/usr/bin/env python3
"""
bias_and_construct.py — items 4 and 5 from the still-needed list.

    python bias_and_construct.py \
        --joined paper/joined_for_tables.csv \
        --v2 data/consensus_2026-09-08_0035__all_8436_8436__..._2709a522.csv \
        --outdir paper

ITEM 4 — CONSTRUCT MISMATCH

    LLM0 is the harmonised published corpus. A classifier trained on it
    barely exceeds the trivial baseline here. Part of that is error and part
    is that the question changed: the source corpora label HATE SPEECH,
    meaning attacks that reference identity, while this panel was asked
    about HARASSMENT, which includes insults and threats with no identity
    dimension at all.

    An insult with no identity reference is Neutral under their definition
    and Harassment under ours. Every such text is scored as an LLM0 failure
    while being a correct application of the labels it was trained on.

    This measures the share. It converts "those labels are bad" into "those
    labels answer a different question, and here is how much of the gap that
    accounts for" -- which is both more defensible and more useful to the
    people who published them.

ITEM 5 — PER-GROUP DIFFERENTIAL FLAG RATE

    The tier configuration deliberately keeps identity terms out of every
    matching rule, because crossing attack verbs with protected-class terms
    is the best-documented source of false-positive bias in toxicity
    classifiers (Dixon et al., 2018). The terms were reserved as a
    MEASUREMENT axis. Nothing has measured them yet.

    What is computable without human labels is the differential flag rate:
    the share flagged among texts mentioning group X, against the corpus
    base rate. That is a real bias signal.

    What is NOT computable is a true per-group false-positive rate. A false
    positive means the model said Harassment and the truth said no, and
    there is no truth column. Calling a differential flag rate an FPR would
    be the same class of error as thresholding CoreML's label confidence as
    if it were P(harassment). The two are reported under their own names.

TERM LIST PROVENANCE

    The French identity terms below are a working list, not a published
    one. Dixon et al. supply English terms; no equivalent French list is
    cited here because none was used. Before this appears in a paper the
    list needs either a citable source or an explicit statement that it was
    assembled for this measurement, with the terms enumerated. A
    hand-assembled list used to MEASURE bias is defensible in a way that a
    hand-assembled list used to DETECT abuse is not, but it still has to be
    shown.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

POS, NEG = "Harassment", "Neutral"
MODELS = ["llm0", "llm3", "llm4"]

# --------------------------------------------------------------- terms
#
# Grouped by protected category. Neutral group names only -- no slurs. The
# point is to find texts that MENTION a group, not texts that attack one.
IDENTITY_TERMS = {
    "religion": [
        "musulman", "musulmane", "musulmans", "islam", "islamique", "mosquée",
        "juif", "juive", "juifs", "judaïsme", "synagogue", "chrétien",
        "chrétienne", "catholique", "protestant", "bouddhiste", "hindou",
        "athée", "laïc", "voile", "hijab", "kippa",
    ],
    "origin_ethnicity": [
        "arabe", "arabes", "maghrébin", "maghrébine", "africain", "africaine",
        "noir", "noire", "noirs", "blanc", "blanche", "asiatique", "rom",
        "roms", "gitan", "tsigane", "immigré", "immigrée", "immigrés",
        "étranger", "étrangère", "migrant", "migrants", "réfugié", "réfugiés",
        "banlieue", "banlieusard",
    ],
    "gender": [
        "femme", "femmes", "féminin", "féminine", "féministe", "féminisme",
        "homme", "hommes", "masculin", "masculinité", "sexisme", "sexiste",
    ],
    "lgbtq": [
        "gay", "gays", "lesbienne", "lesbiennes", "homosexuel", "homosexuelle",
        "homosexualité", "bisexuel", "bisexuelle", "trans", "transgenre",
        "transsexuel", "queer", "lgbt", "lgbtq", "non-binaire", "homophobe",
        "homophobie", "transphobe", "transphobie",
    ],
    "disability": [
        "handicap", "handicapé", "handicapée", "handicapés", "autiste",
        "autisme", "trisomique", "aveugle", "sourd", "sourde",
        "malvoyant", "malentendant",
    ],
    "age": [
        "vieux", "vieille", "vieillard", "retraité", "retraitée", "jeune",
        "jeunes", "boomer", "senior",
    ],
}

ALL_TERMS = sorted({t for v in IDENTITY_TERMS.values() for t in v})


def norm(s: str) -> str:
    """Match the pipeline's normalisation so counts are comparable."""
    s = unicodedata.normalize("NFKC", str(s or ""))
    s = re.sub(r"[\u200b-\u200d\ufeff]", "", s)
    return s.casefold()


def build_matchers(terms):
    # Word boundaries, so "noir" does not match "noircir" and "gay" does not
    # match "gayette". Accents are preserved: folding them would merge terms
    # that are different words in French.
    return {t: re.compile(rf"(?:^|\W){re.escape(t)}(?:\W|$)") for t in terms}


def wilson(k, n):
    if not n:
        return 0.0, 0.0
    z, p = 1.96, k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# --------------------------------------------------------------- item 5
def item5(d: pd.DataFrame, out: Path, min_n: int):
    print("=" * 90)
    print("ITEM 5 — differential flag rate by identity term")
    print("=" * 90)
    print("  Share flagged among texts MENTIONING a group, against the corpus")
    print("  rate. Not a false-positive rate: there is no truth column.\n")

    text = d.text.map(norm)
    matchers = build_matchers(ALL_TERMS)
    base = {m: float(d[f"prediction_{m}"].eq(POS).mean()) for m in MODELS}
    panel_base = float(d.consensus.eq(POS).mean())
    print(f"  corpus rates — panel {100*panel_base:.1f}%  " +
          "  ".join(f"{m} {100*base[m]:.1f}%" for m in MODELS) + "\n")

    rows = []
    for cat, terms in IDENTITY_TERMS.items():
        hit = pd.Series(False, index=d.index)
        for t in terms:
            hit |= text.str.contains(matchers[t], regex=True)
        n = int(hit.sum())
        if n < min_n:
            print(f"  {cat:18s} n={n:<5d}  below --min-n, skipped")
            continue
        g = d[hit]
        rec = {"category": cat, "n": n,
               "panel_rate": float(g.consensus.eq(POS).mean())}
        cells = []
        for m in MODELS:
            r = float(g[f"prediction_{m}"].eq(POS).mean())
            lo, hi = wilson(int(g[f"prediction_{m}"].eq(POS).sum()), n)
            rec[f"{m}_rate"] = r
            rec[f"{m}_ratio"] = r / base[m] if base[m] else np.nan
            cells.append(f"{100*r:5.1f}% [{100*lo:3.0f},{100*hi:3.0f}] "
                         f"{r/base[m] if base[m] else 0:4.2f}x")
        rows.append(rec)
        print(f"  {cat:18s} n={n:<5d} panel {100*rec['panel_rate']:5.1f}% "
              f"({rec['panel_rate']/panel_base:4.2f}x)   " + "   ".join(cells))

    t = pd.DataFrame(rows)
    t.to_csv(out / "item5_identity_differential.csv", index=False)

    print("\n  Ratios are flag rate relative to the corpus rate for that model.")
    print("  A ratio above 1 means the model flags texts mentioning that group")
    print("  more often than average. That is expected in part -- abuse really")
    print("  does target these groups, and the panel column shows how much.")
    print("\n  The bias question is whether a MODEL's ratio exceeds the PANEL's")
    print("  ratio for the same group. That difference is the part not")
    print("  explained by the content:")
    if len(t):
        print(f"\n  {'category':18s}{'panel':>8s}" +
              "".join(f"{m:>10s}" for m in MODELS) + "   excess over panel")
        for _, r in t.iterrows():
            pr = r.panel_rate / panel_base
            ex = [(r[f"{m}_ratio"] - pr) for m in MODELS]
            flag = "  <<<" if max(ex) > 0.5 else ""
            print(f"  {r.category:18s}{pr:8.2f}" +
                  "".join(f"{r[f'{m}_ratio']:10.2f}" for m in MODELS) +
                  "   " + " ".join(f"{e:+.2f}" for e in ex) + flag)
        print("\n  A model whose ratio sits well above the panel's for one group")
        print("  is keying on the group rather than on the abuse. That is the")
        print("  Dixon et al. failure mode, and it is the reason the tier")
        print("  configuration keeps these terms out of every matching rule.")

    # per-term, for the categories that fire
    per_term = []
    for t_ in ALL_TERMS:
        hit = text.str.contains(matchers[t_], regex=True)
        n = int(hit.sum())
        if n < min_n:
            continue
        g = d[hit]
        rec = {"term": t_, "n": n, "panel_rate": float(g.consensus.eq(POS).mean())}
        for m in MODELS:
            rec[f"{m}_rate"] = float(g[f"prediction_{m}"].eq(POS).mean())
        per_term.append(rec)
    if per_term:
        pt = pd.DataFrame(per_term)
        pt.to_csv(out / "item5_per_term.csv", index=False)
        print(f"\n  per-term table written for {len(pt)} terms above n={min_n}")


# --------------------------------------------------------------- item 4
def item4(d: pd.DataFrame, out: Path, min_n: int):
    print("\n" + "=" * 90)
    print("ITEM 4 — construct mismatch: hate speech vs harassment")
    print("=" * 90)
    print("  Of the texts where LLM0 and the panel disagree, how many carry")
    print("  NO identity reference? Those are texts the source corpora would")
    print("  correctly call Neutral under a hate-speech definition while this")
    print("  panel calls them Harassment under a harassment definition.\n")

    text = d.text.map(norm)
    matchers = build_matchers(ALL_TERMS)
    has_identity = pd.Series(False, index=d.index)
    for t in ALL_TERMS:
        has_identity |= text.str.contains(matchers[t], regex=True)
    print(f"  {int(has_identity.sum()):,} of {len(d):,} texts mention an "
          f"identity term ({100*has_identity.mean():.1f}%)\n")

    ref = d.consensus.eq(POS)
    rows = []
    for m in MODELS:
        pred = d[f"prediction_{m}"].eq(POS)
        fn = (~pred) & ref            # panel says Harassment, model does not
        fp = pred & (~ref)
        rec = {
            "model": m,
            "false_neg": int(fn.sum()),
            "fn_no_identity": int((fn & ~has_identity).sum()),
            "false_pos": int(fp.sum()),
            "fp_no_identity": int((fp & ~has_identity).sum()),
        }
        rec["fn_no_identity_share"] = (rec["fn_no_identity"] / rec["false_neg"]
                                       if rec["false_neg"] else np.nan)
        rows.append(rec)
        print(f"  {m.upper()}")
        print(f"    misses (panel POS, model NEG): {rec['false_neg']:,}")
        print(f"      of which carry NO identity term: "
              f"{rec['fn_no_identity']:,} "
              f"({100*rec['fn_no_identity_share']:.1f}%)")
        print(f"    false alarms (model POS, panel NEG): {rec['false_pos']:,}")
        print(f"      of which carry NO identity term: {rec['fp_no_identity']:,}")

    t = pd.DataFrame(rows)
    t.to_csv(out / "item4_construct_mismatch.csv", index=False)

    llm0 = t[t.model == "llm0"].iloc[0]
    print(f"\n  READING THIS")
    print(f"  {100*llm0.fn_no_identity_share:.1f}% of LLM0's misses carry no")
    print("  identity reference at all. Under a hate-speech definition those")
    print("  are correct Neutrals; under a harassment definition they are")
    print("  misses. That share is the part of LLM0's gap attributable to")
    print("  construct rather than to label error.")
    print("\n  This does not make the remainder acceptable, and it does not")
    print("  rescue the Brier score of 0.2651, which is a calibration failure")
    print("  independent of what the labels meant. But it converts a")
    print("  criticism into a measurement, which is what the published")
    print("  corpora's authors are owed.")

    # Same cut restricted to the settled bands, where the panel is reliable.
    if "votes" in d.columns:
        print("\n  restricted to the unanimous bands (0 and 5 votes), where the")
        print("  panel survives both a judge swap and a criterion change:")
        ext = d[d.votes.isin([0, 5])]
        e_id = has_identity.loc[ext.index]
        e_ref = ext.consensus.eq(POS)
        for m in MODELS:
            pred = ext[f"prediction_{m}"].eq(POS)
            fn = (~pred) & e_ref
            if fn.sum() < min_n:
                continue
            share = float((fn & ~e_id).sum() / fn.sum())
            print(f"    {m}: {int(fn.sum()):,} misses, "
                  f"{100*share:.1f}% with no identity term")
        print("\n    If the share is similar here, the construct explanation")
        print("    holds across the corpus rather than living in the")
        print("    contested band where the panel is least reliable anyway.")


def main(a):
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(a.joined, dtype={"id": str})
    if "text" not in d.columns:
        raise SystemExit("the joined frame has no 'text' column; re-run "
                         "paper_tables.py with a consensus file that carries it")
    print(f"{len(d):,} texts, panel base rate "
          f"{100*d.consensus.eq(POS).mean():.1f}%\n")

    item5(d, out, a.min_n)
    item4(d, out, a.min_n)

    if a.v2:
        v2 = pd.read_csv(a.v2, dtype={"id": str})
        v2["id"] = v2["id"].astype(str)
        d2 = d.drop(columns=["consensus", "votes"], errors="ignore").merge(
            v2[["id", "votes", "consensus"]], on="id")
        print("\n\n" + "#" * 90)
        print("# SAME TWO ITEMS UNDER THE v2 CRITERION")
        print("#" * 90)
        print("  v2 asks whether the content TARGETS a person or identifiable")
        print("  group, which is closer to the hate-speech construct than v1's")
        print("  'contains harassment'. If the construct gap narrows here, that")
        print("  is direct evidence the gap was construct rather than error.\n")
        item5(d2, out.joinpath(), a.min_n)
        item4(d2, out, a.min_n)

    print(f"\nwrote tables to {out}/")
    print("\nThe identity term list is a working list assembled for this")
    print("measurement, not a published one. Enumerate it in the paper or")
    print("replace it with a citable source before publication.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--joined", default="paper/joined_for_tables.csv")
    ap.add_argument("--v2", help="v2 consensus csv, to repeat both items")
    ap.add_argument("--min-n", type=int, default=30)
    ap.add_argument("--outdir", default="paper")
    main(ap.parse_args())
