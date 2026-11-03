#!/usr/bin/env python3
"""
Stage 2 -- parallel five-judge labelling of the merged French corpus.

Applies the configuration frozen 9 August 2026. Every value here was settled
on the 1,025-tweet stream before this runs at scale; do not change one without
re-running the experiment that justified it.

  French instruction          dispersion across judges 13.4 -> 6.8, and
                              over-flagging of political speech reduced
  temperature 0               Mistral self-flip 8.8% -> 2.5%
  max_tokens 100              10 starves a reasoning model
  thinking_budget 0 (Gemini)  without it, 0.0% flag rate on 1,025 tweets
  grok-4.20-non-reasoning     grok-3-mini retired, silently served as 4.3
  claude default temperature  endpoint rejects an explicit value
  5 attempts, exponential     1 attempt made transient failures permanent
  empty -> ABSTAIN            never a label; excluded from the vote
  1 replicate                 all judges stable to within 2.5% at temp 0
  k >= 3 of 5                 retained from the original design

WHY THE WHOLE CORPUS, INCLUDING THE 9,506 ALREADY LABELLED
----------------------------------------------------------
The existing labels came from an English instruction and from grok-3-mini,
which the provider has retired. Reusing them would give one corpus under two
labelling policies -- the defect found in MLMA, where two binarisations of
identical rows produced 19.3% and 98.4% positive. The extra 16,319 rows cost
about $62. Training and evaluation data must share one policy.

THROUGHPUT FIXES, 10 August 2026
--------------------------------
None of these touch the labelling policy: same prompt, temperature, max_tokens,
models, consensus rule. They change only how calls are scheduled and how a
rate-limited call is recorded.

THE CONSTRAINT IS THE ACCOUNT TIER, NOT THE CLIENT. Mistral returns
x-ratelimit-limit-req-minute: 4 on its 429. Four requests per minute is a
hard quota, so 2,544 calls take 11.3 hours and 23,651 would have taken 98.
No client change moves that number; a paid Mistral tier does.

  1. No batch barrier. asyncio.gather over a fixed batch made every finished
     call wait for the slowest one in its group. A worker pool draining a
     queue keeps every provider busy instead.
  2. Backoff sleeps outside the semaphore. A call sleeping through a 429 no
     longer holds one of Mistral's few concurrency slots.
  3. Mistral paced to one call per 16s at concurrency 1, matching the 4/min
     quota with a one-second margin for the fixed window boundary.
  4. Backoff sized to the window (15s) and CAPPED (60s). Uncapped doubling
     let a single call cost 8.5 minutes of pure sleeping.
  5. Retry-After honoured, and ONLY Retry-After. Reading a remaining/reset
     counter as a delay yields 0 and burns the retry budget instantly.
  6. A call that exhausts its rate-limit budget is NOT recorded. Previously
     it was stored as an ABSTAIN, which made resume skip it forever and put
     a non-verdict into the corpus. Leaving it unrecorded keeps it in todo.
  7. Retry counts are stored and printed, so 429s are visible instead of
     having to be inferred from a latency histogram.
  8. latency_ms times the API call, not the queue wait ahead of it.

CORPUS ORDERING
  Rows are grouped by source and the head is far more positive than the tail:
  the first 2,192 texts draw 58-85% flags from every judge, against 37.7%
  overall. Shuffle with a fixed seed before any train/test split, and never
  read a partial run's flag rate as the corpus rate.

STORAGE
  Append-only. runs/<run_id>/results.jsonl, one row per call, raw response
  retained. Resumes on restart; nothing is overwritten.

USAGE
  python3 label_corpus.py --preflight          # ~50 calls
  python3 label_corpus.py --n-rows 10          # smoke test, 50 calls
  python3 label_corpus.py --estimate           # cost only, no calls
  python3 label_corpus.py                      # full 25,825
  python3 label_corpus.py --resume RUN_ID      # continue after a stop
  python3 label_corpus.py --ids pivotal_ids.csv --resume RUN_ID

  Overnight, on a Mac that must not sleep:
    caffeinate -is nohup ./env/bin/python label_corpus.py \
      --ids pivotal_ids.csv --resume RUN_ID > run_overnight.log 2>&1 &
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil as _shutil
import sys
import threading
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from run_store import RunSpec, RunStore

load_dotenv()

DATA = Path("data")
CORPUS = DATA / "french_corpus_merged.csv"
POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"

K_CONSENSUS = 3
MAX_TOKENS = 100
TEMPERATURE = 0
MAX_ATTEMPTS = 5
TIMEOUT = 60.0

# Per-provider simultaneous requests. Mistral's quota is 4 requests per
# MINUTE (confirmed 10 Aug from the 429 response header
# x-ratelimit-limit-req-minute: 4). Concurrency above 1 is pointless against
# a per-minute cap; it only converts spacing into rejections.
CONCURRENCY = {
    "gemini": 5, "grok": 5, "openai": 5, "claude": 5, "mistral": 1,
}

# Minimum seconds between successive requests to a provider.
#
# 4 req/min is one every 15 seconds. 16 rather than 15 because the window is
# fixed rather than sliding: at exactly 15s a call lands on the boundary and
# is refused. The extra second costs 2,544 * 1s = 42 minutes over the run and
# buys a 429 rate near zero, which matters because a call that exhausts its
# rate-limit budget is dropped unrecorded rather than stored.
#
# This is a floor set by the account tier, not a tuning choice. 2,544 calls
# at 16s is 11.3 hours. A paid Mistral tier removes the constraint entirely;
# nothing in the client can.
MIN_INTERVAL = {
    "gemini": 0.0, "grok": 0.0, "openai": 0.0, "claude": 0.0, "mistral": 16.0,
}

# A 429 is not a transient network fault and must not share its backoff.
# Sized to the window: the quota resets once a minute, so waiting 15s and
# retrying is right, and doubling past 60s only wastes time. The cap is what
# stops one call from costing 8.5 minutes, which is what 2.0 doubling through
# 8 attempts did on 10 Aug.
RATE_LIMIT_BACKOFF = 15.0       # seconds, doubled per attempt
RATE_LIMIT_MAX_WAIT = 60.0      # ceiling on any single backoff sleep
RATE_LIMIT_ATTEMPTS = 8

# Pinned. The version the API returns is asserted against these on every call.
MODELS = {
    "gemini":  "gemini-2.5-flash",
    "grok":    "grok-4.20-non-reasoning",
    "mistral": "mistral-large-latest",
    "openai":  "gpt-4o-mini",
    "claude":  "claude-haiku-4-5-20251001",
}
# Substring that must appear in the returned version string.
VERSION_MUST_CONTAIN = {
    "gemini": "gemini-2.5-flash", "grok": "grok-4.20",
    "mistral": "mistral-large", "openai": "gpt-4o-mini",
    "claude": "claude-haiku",
}

# Approximate input price per million tokens, for the pre-run estimate only.
PRICE_PER_M = {
    "gemini": 0.30, "grok": 3.00, "mistral": 2.00,
    "openai": 0.15, "claude": 1.00,     # Haiku tier; Opus was 15.00
}
EST_INPUT_TOKENS = 320

PROMPT = (
    "Vous êtes un classificateur de modération de contenu pour des "
    "publications francophones sur les réseaux sociaux. Classez le texte "
    "suivant en exactement un mot :\n"
    "'Harassment' s'il contient du harcèlement, des propos haineux, des "
    "insultes, des menaces, du sexisme, du racisme ou un langage abusif ; "
    "sinon 'Neutral'. Répondez uniquement par le mot Harassment ou Neutral.\n"
    "Texte : {text}"
)


class JudgeRefusal(Exception):
    """
    The provider declined to answer.

    Distinct from a network fault and from a rate limit. A refusal is
    DETERMINISTIC: the same prompt returns the same block, so retrying
    re-spends the call for the same answer. Carries the provider's own
    reason so the log says why.
    """

    def __init__(self, reason: str, version: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.version = version


# ===========================================================================
# parsing -- an empty or unparseable reply is ABSTAIN, never the negative class
# ===========================================================================
def parse_verdict(raw) -> str:
    if raw is None:
        return ABSTAIN
    t = str(raw).strip().lower()
    if not t or t == "nan":
        return ABSTAIN
    pos = any(w in t for w in ("harassment", "harcèlement", "harcelement"))
    neg = any(w in t for w in ("neutral", "neutre"))
    if pos and not neg:
        return POS
    if neg and not pos:
        return NEG
    return ABSTAIN


# ===========================================================================
# judges -- each returns (raw_text, model_version_returned)
# ===========================================================================
async def ask_gemini(client: httpx.AsyncClient, text: str):
    key = os.environ.get("GEMINI_API_KEY") or os.environ["GOOGLE_API_KEY"]
    r = await client.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{MODELS['gemini']}:generateContent",
        params={"key": key},
        json={
            "contents": [{"parts": [{"text": PROMPT.format(text=text)}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_TOKENS,
                # Load-bearing. Without it the model spends its whole output
                # budget on reasoning and returns no content field.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        },
    )
    r.raise_for_status()
    d = r.json()
    ver = d.get("modelVersion", MODELS["gemini"])

    # A blocked prompt returns HTTP 200 with NO candidates array and the
    # reason in promptFeedback. Reading candidates first turns that into a
    # KeyError and loses the only useful information in the response.
    block = (d.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise JudgeRefusal(f"BLOCKED:{block}", ver)

    cands = d.get("candidates") or []
    if not cands:
        raise JudgeRefusal("BLOCKED:NO_CANDIDATES", ver)

    cand = cands[0]
    try:
        txt = cand["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        txt = ""

    if not str(txt).strip():
        # Candidates present but no text: finishReason says why.
        fr = cand.get("finishReason", "UNKNOWN")
        if fr in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION",
                  "SPII"):
            raise JudgeRefusal(f"BLOCKED:{fr}", ver)
        if fr == "MAX_TOKENS":
            # Deterministic at temperature 0, and a config problem rather
            # than a transient one. Surface it instead of retrying.
            raise JudgeRefusal("EMPTY:MAX_TOKENS", ver)
    return txt, ver


async def _openai_style(client, url, key, model, text):
    r = await client.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model,
              "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
              "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
    )

    # Content-policy rejections, before raise_for_status turns them into a
    # truncated exception string. 429 is deliberately NOT in this list: it
    # must keep flowing to the rate-limit backoff path.
    if r.status_code in (400, 403, 422, 451):
        try:
            msg = str((r.json().get("error") or {}).get("message", ""))[:140]
        except Exception:
            msg = r.text[:140]
        raise JudgeRefusal(f"HTTP_{r.status_code}:{msg}", model)

    r.raise_for_status()
    d = r.json()
    choice = (d.get("choices") or [{}])[0]
    txt = (choice.get("message") or {}).get("content") or ""
    ver = d.get("model", model)
    if not str(txt).strip():
        fin = choice.get("finish_reason")
        if fin in ("content_filter", "refusal"):
            raise JudgeRefusal(f"BLOCKED:{fin}", ver)
        if fin == "length":
            raise JudgeRefusal("EMPTY:MAX_TOKENS", ver)
    return txt, ver


async def ask_grok(client, text):
    return await _openai_style(client, "https://api.x.ai/v1/chat/completions",
                               os.environ["XAI_API_KEY"], MODELS["grok"], text)


async def ask_openai(client, text):
    return await _openai_style(client, "https://api.openai.com/v1/chat/completions",
                               os.environ["OPENAI_API_KEY"], MODELS["openai"], text)


async def ask_mistral(client, text):
    return await _openai_style(client, "https://api.mistral.ai/v1/chat/completions",
                               os.environ["MISTRAL_API_KEY"], MODELS["mistral"], text)


async def ask_claude(client, text):
    # Temperature omitted deliberately: matches the refinement pass, and this
    # endpoint rejects an explicit value on this model.
    r = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                 "anthropic-version": "2023-06-01"},
        json={"model": MODELS["claude"], "max_tokens": MAX_TOKENS,
              "messages": [{"role": "user", "content": PROMPT.format(text=text)}]},
    )
    r.raise_for_status()
    d = r.json()
    txt = "".join(b.get("text", "") for b in d.get("content", [])
                  if b.get("type") == "text")
    return txt, d.get("model", MODELS["claude"])


JUDGES = {"gemini": ask_gemini, "grok": ask_grok, "mistral": ask_mistral,
          "openai": ask_openai, "claude": ask_claude}


class ModelSubstitution(RuntimeError):
    pass


def assert_version(judge: str, returned: str) -> None:
    want = VERSION_MUST_CONTAIN[judge]
    if want.lower() not in str(returned or "").lower():
        raise ModelSubstitution(
            f"{judge}: requested {MODELS[judge]}, provider returned "
            f"{returned!r}. grok-3-mini was retired this way and a request "
            f"for it was silently served by grok-4.3. Verify against the "
            f"provider's model list before continuing.")


class RateLimiter:
    """Enforces a minimum interval between requests to one provider."""

    def __init__(self, interval: float):
        self.interval = interval
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval


LIMITERS: dict = {}
SEMS: dict = {}


def init_limits() -> None:
    """Rebuild both per-provider gates. Must run inside the event loop."""
    LIMITERS.clear()
    SEMS.clear()
    LIMITERS.update({j: RateLimiter(MIN_INTERVAL[j]) for j in JUDGES})
    SEMS.update({j: asyncio.Semaphore(CONCURRENCY[j]) for j in JUDGES})


def _is_rate_limit(e: Exception) -> bool:
    t = str(e).lower()
    return "429" in t or "rate limit" in t or "too many requests" in t


def _retry_after(e: Exception) -> float | None:
    """
    Seconds the provider asked us to wait, if it said.

    Only Retry-After means that. An earlier version also read
    x-ratelimit-reset-requests, which is a quota-window counter, not a delay.
    Mistral sends x-ratelimit-remaining-req-minute: 0 alongside its 429, and
    reading any remaining/reset header as a sleep duration gives 0, which
    burns the whole retry budget in milliseconds.
    """
    resp = getattr(e, "response", None)
    if resp is None:
        return None
    v = resp.headers.get("retry-after")
    if v:
        try:
            return max(0.0, min(float(v), RATE_LIMIT_MAX_WAIT))
        except ValueError:
            pass          # HTTP-date form; fall back to the doubling
    return None


async def call_judge(client, judge, text):
    """
    Returns (raw, version, latency_ms, error, attempts, rl_attempts).

    A 429 gets its own retry budget and its own backoff. Both are now sized
    to the measured behaviour: a rejection arrives in 0.3s, so the sleep that
    follows it starts at 2s, not 8s.

    The concurrency slot is held only across the request itself. Sleeping
    through a backoff while holding one of two Mistral slots is how a
    provider that answers in 1.2s produced a 94-hour estimate.
    """
    fn = JUDGES[judge]
    lim = LIMITERS[judge]
    sem = SEMS[judge]
    err = ""
    t_enter = time.time()
    t0 = t_enter
    attempt = rl_attempt = 0
    while attempt < MAX_ATTEMPTS and rl_attempt < RATE_LIMIT_ATTEMPTS:
        await lim.wait()
        try:
            async with sem:
                # Timed HERE, not at function entry. With a pool of pumps
                # every task starts at once and then waits on the limiter,
                # so entry-relative timing measured queue depth and made
                # 1-second calls look like 300-second ones.
                t0 = time.time()
                raw, ver = await fn(client, text)
            assert_version(judge, ver)
            if str(raw).strip():
                return (raw, ver, int(1000 * (time.time() - t0)), "",
                        attempt, rl_attempt)
            err = "empty completion"
            attempt += 1
            await asyncio.sleep(2 ** attempt)
        except ModelSubstitution:
            raise
        except JudgeRefusal as e:
            # Deterministic, so no retry. Recorded as ABSTAIN with the
            # provider's reason: a refusal is a fact about the pipeline, not
            # a verdict about the text, and it must not become either a
            # label or an invisible gap.
            return ("", e.version, int(1000 * (time.time() - t0)), e.reason,
                    attempt, rl_attempt)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"
            if _is_rate_limit(e):
                wait = _retry_after(e)
                if wait is None:
                    wait = min(RATE_LIMIT_BACKOFF * (2 ** rl_attempt),
                               RATE_LIMIT_MAX_WAIT)
                rl_attempt += 1
                await asyncio.sleep(wait)
            else:
                attempt += 1
                await asyncio.sleep(2 ** attempt)
    return ("", "", int(1000 * (time.time() - t_enter)), err,
            attempt, rl_attempt)


# ===========================================================================
# preflight
# ===========================================================================
PROBES = [
    ("Merci beaucoup pour votre aide, c'est très gentil.", NEG),
    ("Il fait beau aujourd'hui, on va se promener.", NEG),
    ("La réunion est reportée à mardi prochain.", NEG),
    ("J'ai adoré ce film, je le recommande vivement.", NEG),
    ("Le train de 8h a du retard comme d'habitude.", NEG),
    ("Ferme ta gueule sale idiot, personne ne t'a rien demandé.", POS),
    ("T'es qu'une merde, dégage de là connard.", POS),
    ("Les gens comme toi ne devraient pas exister, va crever.", POS),
    ("Espèce de grosse débile, retourne d'où tu viens.", POS),
    ("Je vais te retrouver et te faire regretter d'être né.", POS),
]


async def preflight() -> None:
    print("\nPREFLIGHT -- 10 probes per judge")
    init_limits()
    dead = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for j in JUDGES:
            verdicts, empties, vers = [], 0, set()
            for text, _ in PROBES:
                raw, ver, _, _, _, _ = await call_judge(client, j, text)
                if not str(raw).strip():
                    empties += 1
                vers.add(ver)
                verdicts.append(parse_verdict(raw))
            correct = sum(1 for v, (_, e) in zip(verdicts, PROBES)
                          if v == e and v != ABSTAIN)
            labels = {v for v in verdicts if v != ABSTAIN}
            bad = empties > 5 or len(labels) < 2
            print(f"  {j:9s} correct {correct}/10  empty {empties}/10  "
                  f"{sorted(v for v in vers if v)}{'   <-- DEAD' if bad else ''}")
            if bad:
                dead.append(j)
    if dead:
        raise SystemExit(
            f"\nPREFLIGHT FAILED: {dead}\nFix the configuration before "
            f"spending the run. An empty completion must never count as "
            f"the negative class.")
    print("  all five judges alive")


# ===========================================================================
# storage helper
# ===========================================================================
_RL_FIELD_OK = True


def record(store, tid, judge, raw, ver, lab, text, ms, err, rl) -> None:
    """
    Store one call, including how many 429s it survived.

    run_store may predate the rl_attempts field. If it rejects the kwarg we
    fall back once and stop trying, rather than failing the run over a
    diagnostic.
    """
    global _RL_FIELD_OK
    kw = dict(parsed_label=lab, text=str(text), latency_ms=ms, error=err)
    if _RL_FIELD_OK:
        try:
            store.record(tid, judge, 0, raw, ver, rl_attempts=rl, **kw)
            return
        except TypeError:
            _RL_FIELD_OK = False
            print("  note: run_store does not accept rl_attempts; "
                  "429 counts will show in the console only", flush=True)
    store.record(tid, judge, 0, raw, ver, **kw)


# ===========================================================================
# cost estimate
# ===========================================================================
def estimate(n_texts: int) -> float:
    per_text = sum(EST_INPUT_TOKENS * PRICE_PER_M[j] / 1e6 for j in MODELS)
    total = n_texts * per_text
    print(f"\nESTIMATE  {n_texts:,} texts x {len(MODELS)} judges = "
          f"{n_texts*len(MODELS):,} calls")
    for j in sorted(MODELS, key=lambda x: -PRICE_PER_M[x]):
        c = n_texts * EST_INPUT_TOKENS * PRICE_PER_M[j] / 1e6
        print(f"  {j:9s} ~${c:8,.2f}   {100*c/total:5.1f}%")
    print(f"  {'TOTAL':9s} ~${total:8,.2f}")
    print("  (input tokens only, approximate prices; record measured spend)")
    return total


def corpus_fingerprint(path: Path, df) -> dict:
    """
    Two hashes, because they answer different questions.

    file_sha256 changes if anything at all changed, including a re-quoted
    column. ids_sha256 changes only if the SET of texts changed, so a corpus
    regenerated with the same rows but a different column order still matches.
    Report both: they disagree exactly when the file moved but the sample
    did not, which is the case worth knowing about.
    """
    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ids = sorted(str(i) for i in df["id"])
    ids_sha = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    return {
        "corpus_path": str(path.resolve()),
        "corpus_bytes": path.stat().st_size,
        "n_texts": len(ids),
        "file_sha256": file_sha,
        "ids_sha256": ids_sha,
        "first_id": ids[0] if ids else None,
        "last_id": ids[-1] if ids else None,
    }


def archive_corpus(path: Path, fp: dict, dest: str) -> str | None:
    """Copy the corpus to GCS keyed by ids_sha256. Never overwrite."""
    from google.cloud import storage

    assert dest.startswith("gs://"), "--archive-corpus must start with gs://"
    bucket_name, _, prefix = dest[5:].rstrip("/").partition("/")
    blob_name = f"{prefix}/{fp['ids_sha256'][:16]}_{path.name}" if prefix \
        else f"{fp['ids_sha256'][:16]}_{path.name}"
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    if blob.exists():
        # Same id-set hash means the same texts. Leave the original in place.
        print(f"corpus already archived at gs://{bucket_name}/{blob_name}")
    else:
        blob.upload_from_filename(str(path), content_type="text/csv")
        print(f"corpus archived to gs://{bucket_name}/{blob_name}")
    return f"gs://{bucket_name}/{blob_name}"


# ===========================================================================
# main
# ===========================================================================
async def run(args) -> None:
    # --corpus overrides the CORPUS default. read_csv and the fingerprint
    # must see the SAME path, or the run records a hash of a file it did not
    # read.
    corpus = Path(args.corpus) if getattr(args, "corpus", None) else CORPUS
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")
    full = pd.read_csv(corpus, dtype={"id": str})
    df = full
    print(f"corpus {corpus}  {len(df)} rows")
    assert df["id"].is_unique, "duplicate ids in corpus"

    if args.n_rows:
        df = df.head(args.n_rows).copy()
        print(f"  --n-rows {args.n_rows}: smoke test only, not a labelling run")

    if args.ids:
        # Restrict the CALLS, not the corpus. With four judges complete, a
        # k>=3 outcome is already fixed unless exactly two of them said POS,
        # so the rest of the fifth judge's calls cannot change any label.
        # Consensus below is still computed over the whole corpus.
        keep = set(pd.read_csv(args.ids, dtype={"id": str})["id"])
        missing = keep - set(df["id"])
        if missing:
            raise SystemExit(
                f"{len(missing)} ids in {args.ids} are not in the corpus, "
                f"e.g. {sorted(missing)[:5]}")
        df = df[df["id"].isin(keep)].copy()
        print(f"  --ids {args.ids}: {len(df):,} rows selected for calls")

    # Before any spend, so --estimate also reports what it would have judged.
    fp = corpus_fingerprint(corpus, df)
    print(f"\ncorpus fingerprint")
    print(f"  texts       {fp['n_texts']:,}")
    print(f"  ids_sha256  {fp['ids_sha256']}")
    print(f"  file_sha256 {fp['file_sha256']}")

    cost = estimate(len(df))
    if args.estimate:
        return
    if cost > args.max_cost:
        raise SystemExit(
            f"\nestimate ${cost:,.2f} exceeds --max-cost ${args.max_cost:,.2f}. "
            f"Raise the limit deliberately or reduce the corpus.")

    await preflight()

    spec = RunSpec(
        corpus=(f"french_corpus_merged_{len(df)}" if args.n_rows
                else "french_corpus_merged_25825"),
        prompt_lang="fr",
        judges=dict(MODELS),
        prompt_text=PROMPT,
        replicates=1,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        consensus_k=K_CONSENSUS,
        notes=("Stage 2 corpus labelling, frozen configuration of 9 Aug 2026"
               f" | corpus={corpus.name}"
               f" | n_texts={fp['n_texts']}"
               f" | file_sha256={fp['file_sha256']}"),
        # extra IS hashed by RunSpec.fingerprint(); notes is NOT. The id-set
        # hash therefore has to live here, or two runs over different texts
        # would share a fingerprint and run_store would let one resume into
        # the other. file_sha256 stays out on purpose: a cosmetic re-export
        # of the same rows should not invalidate a resumable run.
        extra={"corpus_ids_sha256": fp["ids_sha256"],
               "corpus_n_texts": str(fp["n_texts"])},
    )

    lock = threading.Lock()
    counts = {j: {"done": 0, "pos": 0, "abstain": 0, "err": 0,
                  "retried": 0, "ratelimited": 0} for j in JUDGES}
    t_start = time.time()

    with RunStore(spec, resume_from=args.resume) as store:
        # The run now carries its own inputs. Answering "which texts produced
        # this number" no longer depends on a file outside the run directory.
        if args.archive_corpus:
            try:
                fp["gcs_uri"] = archive_corpus(corpus, fp, args.archive_corpus)
            except Exception as e:
                print(f"corpus archive failed ({e}); continuing")
                fp["gcs_uri"] = None

        side = store.dir / "corpus.json"
        if side.exists():
            # Resuming: the corpus must be the same texts, or the run is a
            # mixture of two samples and every rate computed over it is wrong.
            prev = json.loads(side.read_text(encoding="utf-8"))
            if prev.get("ids_sha256") != fp["ids_sha256"]:
                raise SystemExit(
                    f"\nCorpus mismatch on resume.\n"
                    f"  run {store.dir.name} was started on ids_sha256 "
                    f"{prev.get('ids_sha256')}\n"
                    f"  this corpus is             {fp['ids_sha256']}\n"
                    f"Resuming would blend two different samples into one run. "
                    f"Start a new run, or pass the original corpus.")
            if prev.get("file_sha256") != fp["file_sha256"]:
                print("note: same texts, different file bytes "
                      "(cosmetic rewrite). Continuing.")
        else:
            side.write_text(json.dumps(fp, indent=2), encoding="utf-8")
            try:
                _shutil.copy2(corpus, store.dir / corpus.name)
            except Exception as e:
                print(f"could not copy corpus into run dir ({e})")

        todo = [(r.id, r.text, j) for r in df.itertuples()
                for j in JUDGES if not store.have(r.id, j, 0)]
        print(f"\n{len(todo):,} calls to make "
              f"({len(df)*len(JUDGES) - len(todo):,} already present)")
        if not todo:
            print("nothing to do")
            return

        init_limits()

        async def worker(client, tid, text, judge):
            raw, ver, ms, err, att, rl = await call_judge(
                client, judge, str(text))
            lab = parse_verdict(raw)

            # A call that ran out of rate-limit budget is not a verdict, and
            # storing it as ABSTAIN would make every future resume skip it.
            # Leave it unrecorded so it stays in todo.
            if not str(raw).strip() and rl >= RATE_LIMIT_ATTEMPTS:
                with lock:
                    counts[judge]["ratelimited"] += 1
                return

            with lock:
                record(store, tid, judge, raw, ver, lab, text, ms, err, rl)
                c = counts[judge]
                c["done"] += 1
                c["pos"] += (lab == POS)
                c["abstain"] += (lab == ABSTAIN)
                c["err"] += bool(err)
                c["retried"] += bool(rl)
                total = sum(x["done"] for x in counts.values())
                # Every 10, not 25. At 16s/call a 25-call interval is 6.7
                # minutes of silence, and the last run made only 20 calls so
                # the progress line never printed at all.
                if total % 10 == 0:
                    el = time.time() - t_start
                    rate = total / max(el, 1)
                    eta = (len(todo) - total) / max(rate, 1e-9) / 60
                    done_at = time.strftime(
                        "%a %H:%M", time.localtime(time.time() + eta * 60))
                    line = (f"  {total:7,}/{len(todo):,}  "
                            f"{rate*60:5.1f} calls/min  "
                            f"eta {eta:6.1f} min (done ~{done_at})")
                    # Surface abstentions AS THEY HAPPEN, not in the summary
                    # an hour later. 88 of 200 Mistral calls were 429s in an
                    # earlier run and nothing said so until the end.
                    bad = {j: c["abstain"] for j, c in counts.items()
                           if c["abstain"]}
                    if bad:
                        worst = max(bad.items(),
                                    key=lambda kv: kv[1] / max(counts[kv[0]]["done"], 1))
                        rate_worst = worst[1] / max(counts[worst[0]]["done"], 1)
                        line += f"   ABSTAIN {bad}"
                        if rate_worst > 0.02:
                            line += (f"   <-- {worst[0]} at "
                                     f"{100*rate_worst:.0f}%, STOP and raise "
                                     f"MIN_INTERVAL")
                    # A retried 429 that eventually succeeds leaves err="".
                    # It was invisible before, and 10% of Mistral calls were
                    # paying 56s of backoff without ever showing up here.
                    hit = {j: c["retried"] for j, c in counts.items()
                           if c["retried"]}
                    if hit:
                        line += f"   429-retried {hit}"
                        w = max(hit, key=lambda j: hit[j] / max(counts[j]["done"], 1))
                        frac = hit[w] / max(counts[w]["done"], 1)
                        if frac > 0.15:
                            line += (f"   <-- {w} retrying {100*frac:.0f}% "
                                     f"of calls, raise MIN_INTERVAL['{w}']")
                    dropped = {j: c["ratelimited"] for j, c in counts.items()
                               if c["ratelimited"]}
                    if dropped:
                        line += f"   UNRECORDED {dropped}"
                    print(line, flush=True)

        # No batching. A fixed batch is a barrier: every finished call in the
        # group waits for the slowest one before the next group starts, so a
        # single call sleeping off a 429 stalled 49 others. The pool below
        # lets each provider run at its own pace against its own semaphore.
        from collections import Counter
        print("  judges in queue:",
              dict(Counter(j for _, _, j in todo)), flush=True)

        queue: asyncio.Queue = asyncio.Queue()
        for item in todo:
            queue.put_nowait(item)

        async def pump(client):
            while True:
                try:
                    tid, text, judge = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await worker(client, tid, text, judge)

        # One pump per available slot across all providers. Pumps are not
        # judge-specific; the per-provider semaphores do the limiting.
        n_pumps = sum(CONCURRENCY[j] for j in JUDGES)
        print(f"  {n_pumps} pump tasks, per-provider limits {dict(CONCURRENCY)}",
              flush=True)

        # Say up front how long this will take, per provider. The floor is
        # set by whichever judge is paced slowest, and that is a tier limit,
        # not something the run can improve on.
        from collections import Counter as _C
        per_judge = _C(j for _, _, j in todo)
        worst_h = 0.0
        for j, n in per_judge.items():
            gap = max(MIN_INTERVAL[j], 0.001) / max(CONCURRENCY[j], 1)
            hrs = n * gap / 3600
            worst_h = max(worst_h, hrs)
            if MIN_INTERVAL[j]:
                print(f"    {j}: {n:,} calls at {MIN_INTERVAL[j]:.0f}s "
                      f"spacing = {hrs:.1f} h", flush=True)
        if worst_h > 1.0:
            eta = time.strftime("%a %H:%M",
                                time.localtime(time.time() + worst_h * 3600))
            print(f"  FLOOR {worst_h:.1f} h, finishing about {eta}. "
                  f"This is a provider quota, not a client setting. "
                  f"Use caffeinate -is so the machine stays awake.",
                  flush=True)

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                await asyncio.gather(*(pump(client) for _ in range(n_pumps)))
            except ModelSubstitution as e:
                print(f"\nABORTED: {e}")
                print(f"Partial results kept in {store.dir}. "
                      f"Fix the model string and resume with "
                      f"--resume {store.dir.name}")
                raise SystemExit(1)

        print(f"\ncompleted in {(time.time()-t_start)/60:.1f} min")
        for j in sorted(counts):
            c = counts[j]
            valid = c["done"] - c["abstain"]
            print(f"  {j:9s} {c['done']:6,}  flagged {c['pos']:6,} "
                  f"({100*c['pos']/max(valid,1):5.1f}%)  "
                  f"abstain {c['abstain']:4}  errors {c['err']:4}")
        store.summary()
        run_id = store.dir.name

    # ---- consensus, computed from the stored rows
    res = pd.read_json(Path("runs") / run_id / "results.jsonl", lines=True)
    piv = res.pivot_table(index="text_id", columns="judge",
                          values="parsed_label", aggfunc="first")
    votes = (piv == POS).sum(axis=1)
    valid = (piv != ABSTAIN).sum(axis=1)
    out = pd.DataFrame({"id": piv.index.astype(str), "votes": votes.values,
                        "valid_judges": valid.values})
    out["consensus"] = (out.votes >= K_CONSENSUS).map({True: POS, False: NEG})
    out["judges_seen"] = piv.notna().sum(axis=1).values

    # Merge against the WHOLE corpus, not the possibly --ids-filtered df.
    # results.jsonl carries every judge's rows from every resume, so a run
    # that only made the pivotal calls still writes all 25,825 rows here.
    merged = full[["id", "source", "text", "label"]].merge(
        out, on="id", how="left")
    dest = DATA / f"consensus_{run_id}.csv"
    merged.to_csv(dest, index=False)

    print(f"\n{'votes':>6s}{'texts':>8s}")
    for k in range(6):
        print(f"{k:6d}{int((out.votes == k).sum()):8,}")
    n = int((out.consensus == POS).sum())
    print(f"\nk>={K_CONSENSUS}: {n:,} positives ({100*n/len(out):.2f}%)")

    # With --ids, most rows are decided by four judges. That is exact for the
    # k>=3 rule, but say so plainly rather than leaving it implicit.
    cov = out.judges_seen.value_counts().sort_index()
    print("\njudges per text")
    for k, v in cov.items():
        print(f"  {k}: {v:,}")
    gaps = int(merged.consensus.isna().sum())
    if gaps:
        print(f"\n{gaps:,} corpus rows have no labels at all in this run")
    if "label" in merged.columns:
        ok = (merged.label == merged.consensus).mean()
        print(f"agreement with the human gold label: {100*ok:.1f}%")
    print(f"\nwrote {dest}")
    print(f"run_id {run_id}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--estimate", action="store_true", help="cost only, no calls")
    ap.add_argument("--n-rows", type=int, default=0,
                    help="first N rows -- smoke test only")
    ap.add_argument("--corpus",
                    help="path to the corpus CSV "
                         "(id, source, text, label). "
                         "Defaults to CORPUS.")
    ap.add_argument("--resume", help="run_id to continue")
    ap.add_argument("--archive-corpus", metavar="GS_URI",
                    help="also copy the corpus CSV to gs://bucket/prefix/, "
                         "keyed by ids_sha256")
    ap.add_argument("--ids", help="csv with an id column; call only these rows")
    ap.add_argument("--max-cost", type=float, default=250.0,
                    help="abort if the estimate exceeds this")
    a = ap.parse_args()

    if a.preflight:
        asyncio.run(preflight())
        sys.exit(0)

    asyncio.run(run(a))
