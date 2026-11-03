#!/usr/bin/env python3
"""
Re-parse stored raw responses. Makes ZERO API calls.

THE PROBLEM
    Three Grok responses parsed as ABSTAIN despite containing a clear verdict:

        "**Neutral**\\n\\nLe texte dénonce le racisme ... il ne contient ni
         propos haineux, ni insultes, ni harcèlement ..."

    The verdict is Neutral. The EXPLANATION names every harassment term in
    the prompt, because the model is explaining what the text is not.

THE TRAP
    The obvious fix -- search the whole string for a harassment token -- is
    worse than the bug. All three explanations contain "harcèlement", so a
    substring search flips all three to Harassment: three silent wrong labels
    in place of three visible gaps. Failing closed to ABSTAIN was correct.

THE FIX
    Take the verdict from the FRONT of the response, before any explanation
    begins. Strip markdown, take the first line, and match only there. If the
    first line is not a verdict, fall back to the first line that is, and
    abstain if none is. Explanatory prose is never searched.

USAGE
    ./env/bin/python reparse.py --run RUN_ID              # report only
    ./env/bin/python reparse.py --run RUN_ID --apply      # rewrite labels
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from collections import Counter
from pathlib import Path

POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"

# Accent-stripped, lowercase forms. Order does not matter because each is
# matched against a whole token, not searched for inside a sentence.
POS_WORDS = {"harassment", "harcelement", "harcelant", "abusif", "haineux"}
NEG_WORDS = {"neutral", "neutre", "non", "aucun"}


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower()


def _verdict_of_line(line: str) -> str | None:
    """A verdict only if the LINE is essentially just the verdict word."""
    t = _norm(line)
    t = re.sub(r"[*_`#>\s]+", " ", t)          # markdown and whitespace
    t = re.sub(r"^[\s\-\d.):]+", "", t)         # list markers
    t = t.strip(" .:!,;\"'()[]")
    if not t:
        return None
    words = t.split()
    # Allow a short lead-in such as "reponse: neutral" but never a sentence.
    if len(words) > 4:
        return None
    for w in words:
        w = w.strip(" .:!,;\"'()[]")
        if w in POS_WORDS:
            return POS
        if w in NEG_WORDS:
            return NEG
    return None


def parse_verdict(raw) -> str:
    """
    Front-anchored. Never searches explanatory prose, because an explanation
    of why something is NOT harassment necessarily contains the harassment
    vocabulary.
    """
    s = str(raw).strip()
    if not s:
        return ABSTAIN
    for line in s.splitlines():
        v = _verdict_of_line(line)
        if v:
            return v
        if line.strip():
            # First non-empty line was prose, not a verdict. Keep scanning
            # only blank-separated headers, not the body.
            break
    # Some models emit "Label: Neutral" on a later line after a preamble.
    for line in s.splitlines()[:4]:
        m = re.match(r"^\s*[*_`#\s]*(?:label|reponse|response|verdict|"
                     r"classification)\s*[:=]\s*(.+)$", _norm(line))
        if m:
            v = _verdict_of_line(m.group(1))
            if v:
                return v
    return ABSTAIN


def main(a):
    path = Path("runs") / a.run / "results.jsonl"
    rows, changed = [], []
    counts = Counter()

    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                rows.append(None)
                continue
            old = r.get("parsed_label")
            new = parse_verdict(r.get("raw_response", ""))
            if new != old:
                counts[f"{old} -> {new}"] += 1
                changed.append((r.get("judge"), r.get("text_id"), old, new,
                                str(r.get("raw_response", ""))[:70]))
                r["parsed_label"] = new
            rows.append(r)

    print(f"{len(rows):,} rows scanned")
    if not changed:
        print("no labels change under the corrected parser")
        return

    print(f"\n{len(changed):,} labels change")
    for k, v in counts.most_common():
        print(f"  {k:26s} {v:,}")

    print("\nby judge")
    for j, n in Counter(c[0] for c in changed).most_common():
        print(f"  {j:9s} {n:,}")

    print("\nexamples")
    for j, tid, old, new, raw in changed[:12]:
        print(f"  {j:8s} {tid:18s} {old:8s} -> {new:11s} {raw!r}")

    # Anything leaving ABSTAIN for a real verdict is a recovery. Anything
    # moving BETWEEN real verdicts means the old parser was actively wrong,
    # which is a different and more serious claim -- look at those by hand.
    flips = [c for c in changed if c[2] in (POS, NEG) and c[3] in (POS, NEG)]
    if flips:
        print(f"\n{len(flips)} rows change between two real verdicts. "
              f"Inspect these individually before applying.")

    if not a.apply:
        print("\nreport only. re-run with --apply to rewrite the file.")
        return

    backup = path.with_suffix(".jsonl.bak")
    shutil.copy2(path, backup)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            if r is not None:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nbacked up to {backup}")
    print(f"rewrote {path}")
    print("re-run consensus.py to pick up the corrected labels")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--apply", action="store_true")
    main(ap.parse_args())
