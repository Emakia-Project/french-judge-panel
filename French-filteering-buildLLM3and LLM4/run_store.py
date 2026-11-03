"""
run_store.py -- append-only result storage. Nothing is ever overwritten.

WHY THIS EXISTS
---------------
A rerun script wrote to a single fixed filename. The gemini/grok pass wrote
its results there; the mistral/openai/claude pass read fresh from BigQuery
and wrote to the same path, discarding 2,050 paid-for calls silently. The
loss was only noticed because the flag rates in the printout looked wrong.

Paper 2 makes that failure mode far worse: 25,825 texts x 5 judges x 2
prompt languages x 5 replicates is over a million calls, across many reruns
as prompts, models and configs change. Any scheme where a run writes to one
named file will destroy earlier work.

THE RULE
--------
Every result is a row in an append-only log, keyed by everything that
distinguishes it:

    run_id, text_id, judge, model_version, prompt_lang, replicate_idx

An English-prompt call and a French-prompt call on the same text by the same
judge are DIFFERENT ROWS, not competing versions of one row. A rerun after a
config change is a new run_id; the old rows stay.

Three guarantees:

  1. APPEND, NEVER REPLACE. Results are appended to results.jsonl.
  2. REFUSE TO CLOBBER. A run directory that already exists is never
     silently reused with different settings; the manifest is compared and
     the run aborts on mismatch.
  3. RESUME, NOT RESTART. Before calling an API, ask whether that exact key
     already has a row. A crash at call 800,000 costs nothing.

The run directory name is built from the settings themselves, so what a
file contains is visible without opening it:

    runs/2026-08-07_1432__fr__prompt-en__k3__gemini-grok__a1b2c3/
    runs/2026-08-07_1610__fr__prompt-fr__k3__all5__d4e5f6/

USAGE
-----
    from run_store import RunStore, RunSpec

    spec = RunSpec(
        corpus="fr_eval_1025",
        prompt_lang="en",
        judges={"gemini": "gemini-2.5-flash",
                "grok": "grok-4.20-non-reasoning"},
        replicates=1,
        temperature=0,
        max_tokens=100,
        prompt_text=PROMPT,
        notes="config-matched rerun of the two truncated judges",
    )
    store = RunStore(spec)

    for text_id, text in corpus:
        for judge in spec.judges:
            for rep in range(spec.replicates):
                if store.have(text_id, judge, rep):
                    continue                       # resume
                raw, ver = call(judge, text)
                store.record(text_id, judge, rep, raw, ver, latency_ms=...)

    store.close()
    df = store.to_frame()
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

RUNS_DIR = Path("runs")
ABSTAIN = "ABSTAIN"


def _slug(s: str, maxlen: int = 40) -> str:
    s = re.sub(r"[^\w\-.]+", "-", str(s)).strip("-").lower()
    return s[:maxlen]


@dataclass
class RunSpec:
    """
    Everything that makes one run different from another. The directory name
    and the fingerprint are both derived from this, so two runs that differ
    in any field cannot collide.
    """
    corpus: str                          # "fr_eval_1025", "fr_corpus_25825"
    prompt_lang: str                     # "en" | "fr" | "es"
    judges: Dict[str, str]               # judge name -> pinned model string
    prompt_text: str
    replicates: int = 1
    temperature: float = 0.0
    max_tokens: int = 100
    consensus_k: int = 3
    notes: str = ""
    extra: Dict[str, str] = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Stable hash over everything that affects the result."""
        payload = {
            "corpus": self.corpus,
            "prompt_lang": self.prompt_lang,
            "judges": dict(sorted(self.judges.items())),
            "prompt_text": self.prompt_text,
            "replicates": self.replicates,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra": dict(sorted(self.extra.items())),
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:8]

    def dirname(self, ts: Optional[datetime] = None) -> str:
        """
        Human-readable and unique. What the run did is visible in the name:

            2026-08-07_1432__fr_eval_1025__prompt-en__rep1__k3__gemini-grok__a1b2c3
        """
        ts = ts or datetime.now()
        judges = "-".join(sorted(self.judges))
        if len(judges) > 30:
            judges = f"{len(self.judges)}judges"
        return "__".join([
            ts.strftime("%Y-%m-%d_%H%M"),
            _slug(self.corpus),
            f"prompt-{_slug(self.prompt_lang)}",
            f"rep{self.replicates}",
            f"k{self.consensus_k}",
            _slug(judges),
            self.fingerprint(),
        ])


class RunStore:
    """Append-only JSONL store with a manifest, resume support, and no clobber."""

    def __init__(self, spec: RunSpec, root: Path = RUNS_DIR,
                 resume_from: Optional[str] = None):
        self.spec = spec
        root.mkdir(parents=True, exist_ok=True)

        if resume_from:
            self.dir = root / resume_from
            if not self.dir.exists():
                raise FileNotFoundError(f"cannot resume: {self.dir} not found")
            self._check_manifest_matches()
        else:
            self.dir = root / spec.dirname()
            # A directory name includes a timestamp to the minute, so a
            # collision means two runs started in the same minute with
            # identical settings. Refuse rather than append into it blindly.
            if self.dir.exists():
                raise FileExistsError(
                    f"{self.dir} already exists. Pass resume_from='{self.dir.name}' "
                    f"to continue it, or wait a minute to start a new run. "
                    f"Never write two different runs into one directory.")
            self.dir.mkdir(parents=True)
            self._write_manifest()

        self.results_path = self.dir / "results.jsonl"
        self._seen = self._load_keys()
        self._fh = open(self.results_path, "a", encoding="utf-8")
        print(f"[run_store] {self.dir}")
        if self._seen:
            print(f"[run_store] resuming, {len(self._seen)} rows already present")

    # ------------------------------------------------------------ manifest
    def _manifest(self) -> dict:
        d = asdict(self.spec)
        d["fingerprint"] = self.spec.fingerprint()
        d["created_utc"] = datetime.now(timezone.utc).isoformat()
        return d

    def _write_manifest(self) -> None:
        with open(self.dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(self._manifest(), f, indent=2, ensure_ascii=False)

    def _check_manifest_matches(self) -> None:
        """Resuming into a directory whose settings differ is the overwrite
        bug wearing a different hat. Refuse."""
        path = self.dir / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; cannot verify settings")
        old = json.load(open(path, encoding="utf-8"))
        if old.get("fingerprint") != self.spec.fingerprint():
            raise ValueError(
                f"SETTINGS MISMATCH\n"
                f"  {self.dir} was created with fingerprint "
                f"{old.get('fingerprint')}\n"
                f"  current spec is {self.spec.fingerprint()}\n\n"
                f"  Resuming would mix results from two configurations in one\n"
                f"  file. Start a new run instead.\n"
                f"  differing fields: {self._diff(old)}")

    def _diff(self, old: dict) -> list:
        cur = asdict(self.spec)
        return [k for k in cur if k in old and old[k] != cur[k]]

    # ---------------------------------------------------------------- keys
    @staticmethod
    def _key(text_id, judge, replicate) -> str:
        return f"{text_id}|{judge}|{replicate}"

    def _load_keys(self) -> set:
        if not self.results_path.exists():
            return set()
        seen = set()
        with open(self.results_path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue          # tolerate a truncated final line
                seen.add(self._key(r["text_id"], r["judge"], r["replicate_idx"]))
        return seen

    def have(self, text_id, judge: str, replicate: int = 0) -> bool:
        """True if this exact call already has a row. Use to resume."""
        return self._key(text_id, judge, replicate) in self._seen

    # -------------------------------------------------------------- record
    def record(self, text_id, judge: str, replicate: int,
               raw_response: str, model_version: str,
               parsed_label: Optional[str] = None,
               text: Optional[str] = None,
               latency_ms: Optional[int] = None,
               error: str = "") -> None:
        """
        Append one row.

        RAW RESPONSE IS ALWAYS STORED, including on parse failure and
        refusal, so labels can be re-derived later without re-spending and an
        empty completion stays visible rather than silently becoming a
        negative vote.

        TEXT IS STORED TOO. It duplicates the corpus and costs roughly 20 MB
        across a full run, which is worth it: a log you cannot read without a
        join is a log nobody checks.

        Four fields that were previously written on every row are gone. They
        are constant for a run and already recorded in manifest.json:

            run_id           -- the directory name
            prompt_lang      -- manifest
            model_requested  -- manifest.judges
            input/output_tokens -- never populated

        model_version STAYS. It varies, and it is the field that catches a
        provider silently substituting a model.
        """
        row = {
            "text_id": str(text_id),
            "text": text,
            "judge": judge,
            "model_version": model_version,
            "replicate_idx": replicate,
            "raw_response": raw_response,
            "parsed_label": parsed_label,
            "latency_ms": latency_ms,
            "error": error,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()          # crash-safe: never lose a paid-for call
        self._seen.add(self._key(text_id, judge, replicate))

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --------------------------------------------------------------- reads
    def to_frame(self) -> pd.DataFrame:
        if not self.results_path.exists():
            return pd.DataFrame()
        return pd.read_json(self.results_path, lines=True)

    def summary(self) -> None:
        df = self.to_frame()
        if df.empty:
            print("[run_store] no rows yet")
            return
        print(f"\n[run_store] {self.dir.name}")
        print(f"  run settings are in {self.dir / 'manifest.json'}")
        print(f"  rows {len(df)}   texts {df.text_id.nunique()}")
        for j, g in df.groupby("judge"):
            lab = g.parsed_label
            n = len(g)
            ab = int((lab == ABSTAIN).sum())
            pos = int((lab == "Harassment").sum())
            vers = sorted(set(g.model_version.dropna()))
            flag = "  ! version changed mid-run" if len(vers) > 1 else ""
            print(f"  {j:10s} n={n:6d}  abstain {ab:5d} ({100*ab/n:4.1f}%)  "
                  f"flagged {pos:5d} ({100*pos/max(n-ab,1):5.1f}%)  {vers}{flag}")


# ---------------------------------------------------------------- helpers
def list_runs(root: Path = RUNS_DIR) -> pd.DataFrame:
    """Every run ever done, with its settings. The index of your work."""
    rows = []
    for d in sorted(root.glob("*")):
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        m = json.load(open(mf, encoding="utf-8"))
        rp = d / "results.jsonl"
        n = sum(1 for _ in open(rp, encoding="utf-8")) if rp.exists() else 0
        rows.append({
            "run_id": d.name,
            "corpus": m.get("corpus"),
            "prompt_lang": m.get("prompt_lang"),
            "judges": ",".join(sorted(m.get("judges", {}))),
            "replicates": m.get("replicates"),
            "temperature": m.get("temperature"),
            "rows": n,
            "created": m.get("created_utc", "")[:16],
            "notes": m.get("notes", "")[:40],
        })
    return pd.DataFrame(rows)


def load_runs(*run_ids: str, root: Path = RUNS_DIR) -> pd.DataFrame:
    """Load and concatenate several runs. Compare prompt languages, models,
    replicates -- without any of them having overwritten the others."""
    frames = []
    for rid in run_ids:
        p = root / rid / "results.jsonl"
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_json(p, lines=True))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------- demo
if __name__ == "__main__":
    import shutil
    demo = Path("runs_demo")
    shutil.rmtree(demo, ignore_errors=True)

    base = dict(corpus="fr_eval_1025", prompt_text="Classify: {text}",
                judges={"gemini": "gemini-2.5-flash",
                        "grok": "grok-4.20-non-reasoning"})

    en = RunSpec(prompt_lang="en", notes="English prompt", **base)
    fr = RunSpec(prompt_lang="fr", notes="French prompt", **base)

    print("Two runs differing ONLY in prompt language:\n")
    print(" ", en.dirname())
    print(" ", fr.dirname())
    assert en.dirname() != fr.dirname()
    print("\n-> different directories, neither can overwrite the other\n")

    s1 = RunStore(en, root=demo)
    for i in range(3):
        s1.record(f"tweet{i}", "gemini", 0, "Neutral", "gemini-2.5-flash",
                  parsed_label="Neutral", latency_ms=610)
    s1.close()

    print("\nresuming the same run:")
    s2 = RunStore(en, root=demo, resume_from=s1.dir.name)
    print("  have(tweet0, gemini, 0):", s2.have("tweet0", "gemini", 0), "-> skip")
    print("  have(tweet0, gemini, 1):", s2.have("tweet0", "gemini", 1), "-> call")
    print("  have(tweet9, gemini, 0):", s2.have("tweet9", "gemini", 0), "-> call")
    s2.record("tweet0", "gemini", 1, "Harassment", "gemini-2.5-flash",
              parsed_label="Harassment")
    s2.summary()
    s2.close()

    print("\nresuming with CHANGED settings:")
    changed = RunSpec(prompt_lang="en", temperature=0.7, notes="different", **base)
    try:
        RunStore(changed, root=demo, resume_from=s1.dir.name)
        print("  FAIL: should have refused")
    except ValueError as e:
        print(" ", str(e).splitlines()[0])
        print("  -> refused, settings mismatch")

    print("\nall runs:")
    print(list_runs(demo).to_string(index=False))
    shutil.rmtree(demo, ignore_errors=True)
