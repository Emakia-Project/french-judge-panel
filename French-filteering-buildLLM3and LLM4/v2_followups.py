#!/usr/bin/env python3
"""
v2_followups.py — items 1, 2, 3 and 9 from the still-needed list.

    python v2_followups.py \
        --v1 data/consensus_fixed_2026-09-04_0423__french2026_d2_8436__prompt-fr__rep1__k3__5judges__b4d9cce4.csv \
        --v2 data/consensus_2026-09-08_0035__all_8436_8436__prompt-fr__rep1__k3__5judges__2709a522.csv \
        --run-v1 runs/2026-09-04_0423__french2026_d2_8436__prompt-fr__rep1__k3__5judges__b4d9cce4 \
        --run-v2 runs/2026-09-08_0035__all_8436_8436__prompt-fr__rep1__k3__5judges__2709a522 \
        --joined paper/joined_for_tables.csv \
        --vision runs_vision/20260904T213938Z/results.jsonl \
        --outdir paper

No API calls. Everything here is already on disk.

  1  Classifiers rescored against the v2 labels. Does LLM3 > LLM4 > LLM0
     survive a change of criterion, or was the ordering a property of what
     v1 happened to call harassment?

  2  Why Claude flagged MORE under the stricter criterion while every other
     judge flagged less. If Opus reads "targets" more broadly than the
     others, v2 does not mean the same thing across the panel.

  3  Vision as a contested-band tie-break, measured rather than asserted.
     A 1.8x separation is a correlation; whether using it as a rule
     IMPROVES the panel is a different question and needs the second label
     set as the reference.

  9  Vision three-way join under v2, so the modality comparison uses a
     matched criterion. Vision already ran under v2; comparing it against a
     v1 text panel confounded modality with criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"
MODELS = ["llm0", "llm3", "llm4"]


def confusion(pred, ref):
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum())
    fn = int((~pred & ref).sum()); tn = int((~pred & ~ref).sum())
    n = tp + fp + fn + tn
    base = (tp + fn) / n if n else 0
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    triv = 2 * base / (1 + base) if base else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    return dict(n=n, base=base, precision=p, recall=r, f1=f, trivial=triv,
                lift=f - triv, mcc=mcc, tp=tp, fp=fp, fn=fn, tn=tn)


def load_run(d: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in
            (d / "results.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    df["text_id"] = df.text_id.astype(str)
    return df


# ---------------------------------------------------------------- item 1
def item1(d: pd.DataFrame, out: Path):
    print("=" * 84)
    print("ITEM 1 — classifiers against v1 and v2 labels")
    print("=" * 84)
    print("  Same predictions, two references. If the ordering holds under")
    print("  both, it is a property of the labels rather than of the")
    print("  criterion that scored them.\n")

    rows = []
    for tag, refcol in (("v1", "consensus_v1"), ("v2", "consensus_v2")):
        ref = d[refcol].eq(POS)
        print(f"  {tag}  (base rate {100*ref.mean():.1f}%)")
        for m in MODELS:
            c = confusion(d[f"prediction_{m}"].eq(POS), ref)
            c.update({"criterion": tag, "variant": m.upper()})
            rows.append(c)
            print(f"    {m.upper():5s} F1={c['f1']:.3f}  triv={c['trivial']:.3f}  "
                  f"lift={c['lift']:+.3f}  MCC={c['mcc']:+.3f}  "
                  f"P={c['precision']:.3f} R={c['recall']:.3f}")
        print()

    t = pd.DataFrame(rows)
    t.to_csv(out / "item1_classifiers_v1_v2.csv", index=False)

    print("  ordering by MCC:")
    for tag in ("v1", "v2"):
        g = t[t.criterion == tag].sort_values("mcc", ascending=False)
        print(f"    {tag}: " + "  >  ".join(
            f"{r.variant} ({r.mcc:+.3f})" for _, r in g.iterrows()))
    o1 = list(t[t.criterion == "v1"].sort_values("mcc", ascending=False).variant)
    o2 = list(t[t.criterion == "v2"].sort_values("mcc", ascending=False).variant)
    print(f"\n  {'SAME ordering under both criteria.' if o1 == o2 else 'ORDERING CHANGES with the criterion.'}")
    if o1 == o2:
        print("  The label-quality result does not depend on which question")
        print("  was asked of the panel, which is the stronger form of it.")
    else:
        print("  The label-quality claim is criterion-dependent and must be")
        print("  stated with the criterion attached.")

    # per stream, since base rate drives F1
    if "stream" in d.columns:
        print("\n  by stream, lift over the trivial baseline:")
        print(f"  {'stream':34s}" + "".join(f"{m.upper():>9s}" for m in MODELS) * 1)
        for tag, refcol in (("v1", "consensus_v1"), ("v2", "consensus_v2")):
            print(f"    --- {tag} ---")
            for stream, g in d.groupby("stream"):
                ref = g[refcol].eq(POS)
                cells = "".join(
                    f"{confusion(g[f'prediction_{m}'].eq(POS), ref)['lift']:+9.3f}"
                    for m in MODELS)
                print(f"    {stream:32s}{cells}   base {100*ref.mean():4.1f}%")


# ---------------------------------------------------------------- item 2
def item2(r1: pd.DataFrame, r2: pd.DataFrame, d: pd.DataFrame, out: Path):
    print("\n" + "=" * 84)
    print("ITEM 2 — why did Claude flag MORE under the stricter criterion?")
    print("=" * 84)

    p1 = r1.pivot_table(index="text_id", columns="judge",
                        values="parsed_label", aggfunc="first")
    p2 = r2.pivot_table(index="text_id", columns="judge",
                        values="parsed_label", aggfunc="first")
    common = p1.index.intersection(p2.index)
    p1, p2 = p1.loc[common], p2.loc[common]
    judges = sorted(set(p1.columns) & set(p2.columns))

    print(f"  {len(common):,} texts judged under both prompts\n")
    print(f"  {'judge':10s}{'v1 flag':>9s}{'v2 flag':>9s}{'change':>9s}"
          f"{'POS->NEG':>10s}{'NEG->POS':>10s}{'net':>8s}")
    rows = []
    for j in judges:
        a, b = p1[j].eq(POS), p2[j].eq(POS)
        valid = p1[j].ne(ABSTAIN) & p2[j].ne(ABSTAIN)
        a, b = a[valid], b[valid]
        pn = int((a & ~b).sum()); np_ = int((~a & b).sum())
        rows.append({"judge": j, "v1_flag": float(a.mean()),
                     "v2_flag": float(b.mean()),
                     "pos_to_neg": pn, "neg_to_pos": np_, "net": np_ - pn})
        print(f"  {j:10s}{100*a.mean():8.1f}%{100*b.mean():8.1f}%"
              f"{100*(b.mean()-a.mean()):+8.1f}%{pn:>10,}{np_:>10,}{np_-pn:>+8,}")

    t = pd.DataFrame(rows)
    t.to_csv(out / "item2_judge_shift.csv", index=False)

    up = t[t.net > 0]
    if len(up):
        print(f"\n  Judges that flagged MORE under v2: "
              f"{', '.join(up.judge)}")
        print("  A stricter criterion should reduce every judge's flag rate.")
        print("  A judge moving the other way is reading the two prompts")
        print("  differently from the rest of the panel.")

    # Where does the outlier disagree with the others, under each prompt?
    print("\n  agreement with the OTHER four judges, per prompt:")
    print(f"  {'judge':10s}{'v1':>9s}{'v2':>9s}{'change':>9s}")
    for j in judges:
        others = [x for x in judges if x != j]
        res = []
        for p in (p1, p2):
            maj = (p[others].eq(POS).sum(axis=1) >= 2)
            valid = p[j].ne(ABSTAIN)
            res.append(float((p[j].eq(POS)[valid] == maj[valid]).mean()))
        print(f"  {j:10s}{100*res[0]:8.1f}%{100*res[1]:8.1f}%"
              f"{100*(res[1]-res[0]):+8.1f}%")
    print("\n  A judge whose agreement with the others FALLS under v2 is the")
    print("  one reading the new criterion differently. That is the sentence")
    print("  the paper needs, and it names which judge rather than guessing.")

    # texts only the outlier flags under v2
    if len(up):
        j = up.sort_values("net", ascending=False).judge.iloc[0]
        others = [x for x in judges if x != j]
        only = p2[(p2[j].eq(POS)) & (p2[others].eq(POS).sum(axis=1) == 0)]
        print(f"\n  {len(only):,} texts where only {j} flagged under v2:")
        txt = d.set_index("id").text if "text" in d.columns else None
        if txt is not None:
            for tid in only.index[:10]:
                if tid in txt.index:
                    print(f"    - {str(txt.loc[tid])[:100]}")


# ---------------------------------------------------------------- items 3 & 9
def items39(d: pd.DataFrame, vision: pd.DataFrame, out: Path):
    print("\n" + "=" * 84)
    print("ITEMS 3 & 9 — vision under a matched criterion, and as a tie-break")
    print("=" * 84)

    d = d.merge(vision, on="id", how="left")
    d["vis_pos"] = d.vis_pos.fillna(False).astype(bool)
    d["has_media"] = d.n_media.notna()
    hm = d[d.has_media]
    print(f"  {len(hm):,} media-bearing posts of {len(d):,}\n")

    # ---- item 9: matched criterion ----
    print("  ITEM 9 — vision (v2) against each text panel")
    print("  Vision ran under v2, so comparing it to a v1 panel confounded")
    print("  modality with criterion. Both are shown.")
    for tag, col in (("v1 panel", "consensus_v1"), ("v2 panel  <-- matched", "consensus_v2")):
        tp_ = hm[col].eq(POS); vp = hm.vis_pos
        only_v = int((~tp_ & vp).sum())
        c = confusion(vp, tp_)
        print(f"\n    {tag}")
        print(f"      text POS & vision POS  {int((tp_ & vp).sum()):>5,}")
        print(f"      text POS & vision NEG  {int((tp_ & ~vp).sum()):>5,}")
        print(f"      text NEG & vision POS  {only_v:>5,}   "
              f"({100*only_v/len(hm):.1f}% of media-bearing)")
        print(f"      agreement {100*((tp_ == vp).mean()):.1f}%  MCC {c['mcc']:+.3f}")
    print("\n    The 'vision caught, text missed' cell is the number a")
    print("    text-only filter cannot reach. Under the matched criterion it")
    print("    is the defensible version of that claim.")

    # ---- item 3: tie-break, measured ----
    print("\n  ITEM 3 — vision as a contested-band tie-break")
    print("  A 1.8x separation is a correlation. Whether USING it improves")
    print("  the panel is a different question, and needs a reference the")
    print("  rule did not help produce. The v2 panel is that reference.\n")

    cb = hm[hm.votes_v1.isin([2, 3])]
    if cb.empty:
        print("    no media-bearing texts in the v1 contested band")
        return
    ref = cb.consensus_v2.eq(POS)
    print(f"    {len(cb):,} media-bearing texts in the v1 contested band, "
          f"{100*ref.mean():.1f}% positive under v2\n")

    rules = {
        "v1 verdict as-is": cb.consensus_v1.eq(POS),
        "always Neutral": pd.Series(False, index=cb.index),
        "always Harassment": pd.Series(True, index=cb.index),
        "vision decides": cb.vis_pos,
        "v1 POS AND vision": cb.consensus_v1.eq(POS) & cb.vis_pos,
        "v1 POS OR vision": cb.consensus_v1.eq(POS) | cb.vis_pos,
    }
    print(f"    {'rule':22s}{'F1':>7s}{'triv':>7s}{'lift':>8s}{'MCC':>8s}"
          f"{'P':>7s}{'R':>7s}")
    best, best_mcc = None, -2
    for name, pred in rules.items():
        c = confusion(pred, ref)
        if c["mcc"] > best_mcc:
            best_mcc, best = c["mcc"], name
        print(f"    {name:22s}{c['f1']:7.3f}{c['trivial']:7.3f}"
              f"{c['lift']:+8.3f}{c['mcc']:+8.3f}{c['precision']:7.3f}"
              f"{c['recall']:7.3f}")

    baseline = confusion(cb.consensus_v1.eq(POS), ref)["mcc"]
    print(f"\n    best by MCC: {best}")
    if best != "v1 verdict as-is" and best_mcc > baseline + 0.02:
        print(f"    Vision improves on the v1 verdict in this band "
              f"({baseline:+.3f} -> {best_mcc:+.3f}). The tie-break earns")
        print("    its place, on the ~11% of contested texts that carry media.")
    else:
        print(f"    Vision does NOT improve on the v1 verdict here "
              f"({baseline:+.3f} vs {best_mcc:+.3f}). The 1.8x separation is")
        print("    real but does not survive being turned into a rule, which")
        print("    is worth reporting: correlation at the band level is not")
        print("    the same as usable signal per text.")

    pd.DataFrame([{**confusion(p, ref), "rule": n} for n, p in rules.items()]
                 ).to_csv(out / "item3_vision_tiebreak.csv", index=False)


# ---------------------------------------------------------------- main
def main(a):
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    v1 = pd.read_csv(a.v1, dtype={"id": str})
    v2 = pd.read_csv(a.v2, dtype={"id": str})
    j = pd.read_csv(a.joined, dtype={"id": str})
    for x in (v1, v2, j):
        x["id"] = x["id"].astype(str)

    keep_v1 = [c for c in ("id", "votes", "consensus", "stream", "text") if c in v1.columns]
    keep_v2 = [c for c in ("id", "votes", "consensus") if c in v2.columns]
    d = (v1[keep_v1].rename(columns={"votes": "votes_v1", "consensus": "consensus_v1"})
         .merge(v2[keep_v2].rename(columns={"votes": "votes_v2",
                                            "consensus": "consensus_v2"}), on="id")
         .merge(j[["id"] + [c for c in j.columns
                            if c.startswith(("prediction_", "p_"))]], on="id"))
    if "stream" not in d.columns and "stream" in j.columns:
        d = d.merge(j[["id", "stream"]], on="id", how="left")
    if "text" not in d.columns and "text" in j.columns:
        d = d.merge(j[["id", "text"]], on="id", how="left")
    print(f"{len(d):,} texts with both criteria and CoreML predictions\n")

    item1(d, out)

    if a.run_v1 and a.run_v2:
        item2(load_run(Path(a.run_v1)), load_run(Path(a.run_v2)), d, out)

    if a.vision:
        rows = [json.loads(l) for l in
                Path(a.vision).read_text(encoding="utf-8").splitlines() if l.strip()]
        v = pd.DataFrame(rows)
        v = v[v.tweet_id.notna()].copy()
        v["tweet_id"] = v.tweet_id.astype(str)
        per = v.groupby("tweet_id").agg(
            n_media=("media_key", "size"),
            vis_pos=("verdict", lambda s: bool((s == POS).any()))
        ).reset_index().rename(columns={"tweet_id": "id"})
        items39(d, per, out)

    d.to_csv(out / "v1_v2_joined.csv", index=False)
    print(f"\nwrote tables to {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--run-v1")
    ap.add_argument("--run-v2")
    ap.add_argument("--joined", default="paper/joined_for_tables.csv")
    ap.add_argument("--vision")
    ap.add_argument("--outdir", default="paper")
    main(ap.parse_args())
