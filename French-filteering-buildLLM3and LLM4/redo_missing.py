#!/usr/bin/env python3
"""
Find and re-call every (text, judge) pair that has no usable label.

WHY NOT redo_ids.csv
    That file was written before the Mistral backfill and reflects an older
    state. Regenerate from results.jsonl instead of trusting a stale list.

WHY NOT --resume
    A row stored with an empty or unparseable response still counts as
    present, so store.have skips it forever. That is the same bug class as
    recording a rate-limited call as ABSTAIN: the record exists, so the
    system believes the work is done. These rows need to be found by their
    CONTENT, not by their absence.

TWO MODES
    --diagnose   report what is missing and why. No API calls.
    --fix        re-call the missing pairs and append the results.

USAGE
    ./env/bin/python redo_missing.py --run RUN_ID --diagnose
    ./env/bin/python redo_missing.py --run RUN_ID --fix
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CORPUS = "data/french_corpus_merged.csv"
POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"
JUDGE_NAMES = ["claude", "gemini", "grok", "mistral", "openai"]


def scan(run_dir: Path):
    """
    Returns (usable, stored_bad) where usable maps judge -> set of text_ids
    with a real label, and stored_bad lists rows that exist but carry no
    verdict.
    """
    usable = defaultdict(set)
    stored_bad = []
    with (run_dir / "results.jsonl").open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            j, tid = r.get("judge"), str(r.get("text_id"))
            raw = str(r.get("raw_response", "")).strip()
            lab = r.get("parsed_label")
            if raw and lab in (POS, NEG):
                usable[j].add(tid)
            else:
                stored_bad.append({
                    "text_id": tid, "judge": j, "parsed_label": lab,
                    "raw_response": raw[:80], "error": str(r.get("error", ""))[:80],
                })
    return usable, stored_bad


def missing(run_dir: Path):
    corpus = pd.read_csv(CORPUS, dtype={"id": str})
    corpus = corpus[corpus.text.notna()]
    ids = set(corpus.id)
    usable, stored_bad = scan(run_dir)

    gaps = []
    for j in JUDGE_NAMES:
        for tid in sorted(ids - usable.get(j, set())):
            gaps.append({"text_id": tid, "judge": j})
    return corpus, gaps, stored_bad


def diagnose(a):
    run_dir = Path("runs") / a.run
    corpus, gaps, bad = missing(run_dir)

    print(f"corpus {len(corpus):,} texts x 5 judges = {5*len(corpus):,} pairs")
    print(f"missing a usable label: {len(gaps)}")
    if not gaps:
        print("\nnothing to redo; the matrix is complete")
        return

    by_judge = defaultdict(list)
    for g in gaps:
        by_judge[g["judge"]].append(g["text_id"])
    print("\nby judge")
    for j, v in sorted(by_judge.items()):
        print(f"  {j:9s} {len(v):4d}   e.g. {v[:3]}")

    # A row that exists but carries no verdict is why --resume will not fix
    # this: the record is present, so the run believes the call was made.
    bad_keys = {(b["text_id"], b["judge"]) for b in bad}
    gap_keys = {(g["text_id"], g["judge"]) for g in gaps}
    overlap = bad_keys & gap_keys
    print(f"\n{len(overlap)} of the gaps are STORED with an unusable response.")
    print(f"{len(gap_keys - bad_keys)} are absent from the file entirely.")

    if bad:
        print("\nstored-but-unusable rows (first 20):")
        for b in bad[:20]:
            print(f"  {b['judge']:9s} {b['text_id']:18s} "
                  f"label={str(b['parsed_label']):10s} "
                  f"raw={b['raw_response']!r:20s} err={b['error']!r}")

    out = Path("redo_pairs.csv")
    pd.DataFrame(gaps).to_csv(out, index=False)
    print(f"\nwrote {out} ({len(gaps)} pairs)")
    print("re-call them with:  ./env/bin/python redo_missing.py "
          f"--run {a.run} --fix")


def fix(a):
    # Imported lazily so --diagnose works even if label_corpus is not
    # importable. The judge functions live there and there is no reason to
    # reimplement five API clients for eighteen calls.
    try:
        from label_corpus import JUDGES, call_judge, init_limits, parse_verdict
    except Exception as e:
        raise SystemExit(
            f"could not import label_corpus: {e}\n\n"
            "If this is 'ap.parse_args()' running at import time, guard the\n"
            "bottom of label_corpus.py with:\n"
            "    if __name__ == '__main__':\n"
            "        args = ap.parse_args()\n"
            "        ...\n"
            "and re-run. Nothing else needs to change.")

    import httpx
    from label_corpus import TIMEOUT

    run_dir = Path("runs") / a.run
    corpus, gaps, _ = missing(run_dir)
    if not gaps:
        print("nothing to redo")
        return
    text_of = dict(zip(corpus.id, corpus.text))
    print(f"re-calling {len(gaps)} pairs")

    async def go():
        init_limits()
        n_ok = n_fail = 0
        with (run_dir / "results.jsonl").open("a", encoding="utf-8") as out:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                for g in gaps:
                    tid, j = g["text_id"], g["judge"]
                    if j not in JUDGES:
                        print(f"  {j}: not in JUDGES, skipped")
                        continue
                    res = await call_judge(client, j, str(text_of[tid]))
                    raw, ver = res[0], res[1]
                    lab = parse_verdict(raw)
                    if not str(raw).strip() or lab == ABSTAIN:
                        # Still not a verdict. Do not record it -- storing a
                        # non-answer is what created this situation.
                        print(f"  {j:9s} {tid:18s} still unusable "
                              f"(raw={str(raw)[:40]!r})")
                        n_fail += 1
                        continue
                    out.write(json.dumps({
                        "text_id": tid, "text": str(text_of[tid]), "judge": j,
                        "model_version": ver, "replicate_idx": 0,
                        "raw_response": raw, "parsed_label": lab,
                        "latency_ms": res[2] if len(res) > 2 else 0,
                        "error": "",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }, ensure_ascii=False) + "\n")
                    out.flush()
                    print(f"  {j:9s} {tid:18s} -> {lab}")
                    n_ok += 1
        print(f"\nrecovered {n_ok}   still failing {n_fail}")
        print("re-run consensus.py to pick these up")

    asyncio.run(go())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--fix", action="store_true")
    a = ap.parse_args()
    if a.fix:
        fix(a)
    else:
        diagnose(a)
