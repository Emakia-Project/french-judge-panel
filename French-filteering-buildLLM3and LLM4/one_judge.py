#!/usr/bin/env python3
"""
one_judge.py — run a single judge over a corpus and compare against that
judge's verdicts in an existing run.

    python one_judge.py \
        --corpus data/haiku_check.csv \
        --model claude-haiku-4-5-20251001 \
        --against runs/2026-09-04_0423__french2026_d2_8436__prompt-fr__rep1__k3__5judges__b4d9cce4 \
        --out data/claude_haiku_840.csv

WHY THIS EXISTS

    label_corpus.py runs all five judges, which is right when you want a
    consensus. It is wrong when the question is narrower: does swapping one
    model change that model's own behaviour? Paying for four judges whose
    verdicts you already have answers nothing extra.

WHAT IT ANSWERS AND WHAT IT DOES NOT

    ANSWERS: does the replacement model agree with the original on the same
    texts, overall and per vote band.

    DOES NOT ANSWER: whether any CONSENSUS label changed. Consensus is k>=3
    of five, so one judge differing on a text usually leaves the outcome
    intact. This script estimates the consensus impact from the other four
    judges' stored votes, which is exact -- their verdicts do not change
    when Claude's model does.

    That estimate is the number that matters operationally: it is how many
    corpus labels would move if the model were discontinued and replaced.

    Prompts are read from label_corpus.py so the two runs are asking the
    same question. If that file changes, this script changes with it.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"


def load_label_corpus(path="label_corpus.py"):
    """
    Import the pipeline so prompt, parser and handler are shared, not copied.

    label_corpus.py builds its argparse at module level, so importing it
    makes IT parse OUR command line and reject our flags. Hide argv during
    the import; restore it immediately after.
    """
    saved = sys.argv
    sys.argv = [path]
    try:
        spec = importlib.util.spec_from_file_location("lc", path)
        lc = importlib.util.module_from_spec(spec)
        sys.modules["lc"] = lc
        spec.loader.exec_module(lc)
    except SystemExit:
        # its parser may exit on the stripped argv; the module is still bound
        lc = sys.modules["lc"]
    finally:
        sys.argv = saved
    return lc


async def run_judge(lc, texts, judge, model, concurrency, interval):
    """Call one judge over every text. Same handler label_corpus.py uses."""
    import httpx

    lc.MODELS[judge] = model          # override the pinned model
    if hasattr(lc, "MODEL_PREFIX"):
        lc.MODEL_PREFIX[judge] = model.rsplit("-", 1)[0]
    fn = lc.JUDGES[judge]

    sem = asyncio.Semaphore(concurrency)
    results = {}
    done = 0

    async def one(client, tid, text):
        nonlocal done
        async with sem:
            if interval:
                await asyncio.sleep(interval)
            try:
                raw, ver = await fn(client, text)
                label = lc.parse_label(raw) if hasattr(lc, "parse_label") else None
                err = ""
            except Exception as e:
                raw, ver, label = "", model, None
                err = f"{type(e).__name__}: {str(e)[:120]}"
            results[tid] = {"text_id": tid, "raw_response": raw,
                            "model_version": ver, "parsed_label": label or ABSTAIN,
                            "error": err}
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(texts)}")

    async with httpx.AsyncClient(timeout=lc.TIMEOUT) as client:
        await asyncio.gather(*(one(client, t, x) for t, x in texts))
    return results


def main(a):
    lc = load_label_corpus(a.pipeline)
    if hasattr(lc, "set_prompt"):
        lc.set_prompt(a.prompt)
    elif hasattr(lc, "PROMPTS"):
        lc.PROMPT = lc.PROMPTS[a.prompt]
    print(f"judge={a.judge}  model={a.model}  prompt={a.prompt}")

    df = pd.read_csv(a.corpus, dtype={"id": str})
    texts = [(str(r.id), str(r.text)) for r in df.itertuples()]
    print(f"{len(texts):,} texts\n")

    t0 = time.time()
    res = asyncio.run(run_judge(lc, texts, a.judge, a.model,
                                a.concurrency, a.interval))
    print(f"\ncompleted in {(time.time()-t0)/60:.1f} min")

    new = pd.DataFrame(res.values())
    new["text_id"] = new.text_id.astype(str)
    flagged = int(new.parsed_label.eq(POS).sum())
    abst = int(new.parsed_label.eq(ABSTAIN).sum())
    print(f"  flagged {flagged} ({100*flagged/len(new):.1f}%)  abstain {abst}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(a.out, index=False)
    print(f"  wrote {a.out}")

    if not a.against:
        return

    # ---- compare against the stored run ------------------------------
    rows = [json.loads(l) for l in
            (Path(a.against) / "results.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    old = pd.DataFrame(rows)
    old["text_id"] = old.text_id.astype(str)
    old_j = old[old.judge == a.judge].set_index("text_id")

    m = new.set_index("text_id").join(old_j, rsuffix="_old", how="inner")
    m = m[m.parsed_label.ne(ABSTAIN) & m.parsed_label_old.ne(ABSTAIN)]
    if m.empty:
        print("\nno overlapping non-abstain verdicts")
        return

    print("\n" + "=" * 70)
    print(f"{a.judge}: new model vs stored run, {len(m):,} texts")
    print("=" * 70)
    old_model = m.model_version_old.mode().iloc[0] if len(m) else "?"
    print(f"  stored:  {old_model}")
    print(f"  new:     {a.model}")
    a_pos = m.parsed_label_old.eq(POS)
    b_pos = m.parsed_label.eq(POS)
    print(f"\n  stored flags {100*a_pos.mean():5.1f}%")
    print(f"  new flags    {100*b_pos.mean():5.1f}%")
    print(f"  agree        {100*(a_pos == b_pos).mean():5.1f}%")
    print(f"  stored POS -> new NEG: {int((a_pos & ~b_pos).sum())}")
    print(f"  stored NEG -> new POS: {int((~a_pos & b_pos).sum())}")

    # ---- per vote band -----------------------------------------------
    if a.consensus:
        cons = pd.read_csv(a.consensus, dtype={"id": str})
        cons["id"] = cons["id"].astype(str)
        m2 = m.join(cons.set_index("id")[["votes", "consensus"]], how="left")
        print("\n  agreement by panel vote band:")
        print(f"  {'votes':>6s}{'n':>7s}{'agree':>9s}")
        for v in range(6):
            g = m2[m2.votes == v]
            if g.empty:
                continue
            ag = (g.parsed_label_old.eq(POS) == g.parsed_label.eq(POS)).mean()
            print(f"  {v:>6d}{len(g):>7d}{100*ag:8.1f}%")
        print("\n  If the extremes hold and only 2-3 moves, the panel is")
        print("  robust to this swap where it matters.")

        # ---- consensus impact, exact -----------------------------------
        # The other four judges' verdicts do not change when this judge's
        # model does, so their votes can be reused directly.
        others = old[(old.judge != a.judge) & old.parsed_label.ne(ABSTAIN)]
        other_votes = (others.assign(pos=others.parsed_label.eq(POS))
                       .groupby("text_id").pos.sum())
        idx = m.index.intersection(other_votes.index)
        base = other_votes.loc[idx]
        old_total = base + m.loc[idx, "parsed_label_old"].eq(POS).astype(int)
        new_total = base + m.loc[idx, "parsed_label"].eq(POS).astype(int)
        moved = int(((old_total >= 3) != (new_total >= 3)).sum())
        print("\n" + "=" * 70)
        print("CONSENSUS IMPACT")
        print("=" * 70)
        print(f"  {len(idx):,} texts with the other four judges' votes on file")
        print(f"  consensus labels that would CHANGE: {moved} "
              f"({100*moved/len(idx):.2f}%)")
        print(f"    -> POS: {int(((old_total<3)&(new_total>=3)).sum())}")
        print(f"    -> NEG: {int(((old_total>=3)&(new_total<3)).sum())}")
        print("\n  This is the operational number: how many corpus labels")
        print("  would move if this model were discontinued and replaced.")
        print("  One judge of five rarely flips a k>=3 outcome, so a large")
        print("  per-judge difference can still leave the corpus intact --")
        print("  and a small one can still move the boundary cases.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--judge", default="claude")
    ap.add_argument("--model", required=True, help="model string to test")
    ap.add_argument("--prompt", default="v1", choices=("v1", "v2"))
    ap.add_argument("--against", help="runs/<run_id> holding the stored verdicts")
    ap.add_argument("--consensus", help="consensus_fixed_*.csv, for vote bands")
    ap.add_argument("--pipeline", default="label_corpus.py")
    ap.add_argument("--concurrency", type=int, default=5)
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds between calls, if the provider throttles")
    ap.add_argument("--out", default="data/one_judge.csv")
    main(ap.parse_args())
