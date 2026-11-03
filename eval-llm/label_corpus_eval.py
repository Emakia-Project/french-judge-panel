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

STORAGE
  Append-only. runs/<run_id>/results.jsonl, one row per call, raw response
  retained. Resumes on restart; nothing is overwritten.

USAGE
  python3 label_corpus.py --preflight          # ~50 calls
  python3 label_corpus.py --n-rows 10          # smoke test, 50 calls
  python3 label_corpus.py --estimate           # cost only, no calls
  python3 label_corpus.py                      # full 25,825
  python3 label_corpus.py --resume RUN_ID      # continue after a stop
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
# Default corpus. Override with --corpus to label a different one (e.g. the
# 2026 French collection exported by export_for_judging.py).
CORPUS = DATA / "french_corpus_merged.csv"
POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"

K_CONSENSUS = 3
MAX_TOKENS = 100
TEMPERATURE = 0
MAX_ATTEMPTS = 5
TIMEOUT = 60.0

# Per-provider simultaneous requests. Mistral throttles hard on bursts and
# gets its own low limit; the others tolerate more.
CONCURRENCY = {
    "gemini": 5, "grok": 5, "openai": 5, "claude": 5, "mistral": 1,
}

# Minimum seconds between calls to each provider, as a starting point. The
# pacer widens these on 429 and lets them decay back on sustained success, so
# these are an opening bid rather than a fixed budget. Mistral throttles on
# requests per minute, which is why concurrency alone never fixed it.
MIN_INTERVAL = {
    "gemini": 0.0, "grok": 0.0, "openai": 0.0, "claude": 0.0, "mistral": 1.2,
}

# A 429 gets its own, larger budget: it is a deferred call, not a failed one.
RATE_LIMIT_ATTEMPTS = 8
MAX_BACKOFF = 90.0


class RateLimiter:
    """
    Per-provider pacer. Enforces a minimum gap between calls and widens that
    gap when the provider pushes back.

    Adaptive rather than fixed because the right interval is not knowable in
    advance: it depends on the tier, the model, and what else is running. The
    run finds it by being told.
    """

    def __init__(self, interval: float = 0.0):
        self.interval = interval
        self.floor = interval
        self._next = 0.0
        self._ok_since_penalty = 0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        if self.interval <= 0:
            return
        async with self._lock:
            now = time.monotonic()
            if now < self._next:
                await asyncio.sleep(self._next - now)
                now = time.monotonic()
            self._next = now + self.interval

    def penalise(self, retry_after: float | None = None) -> float:
        """Called on 429. Returns how long to sleep before retrying."""
        self._ok_since_penalty = 0
        self.interval = min(max(self.interval * 2, 0.5), MAX_BACKOFF)
        # Retry-After is the provider stating the answer. Prefer it to a guess.
        return min(retry_after if retry_after else self.interval * 4, MAX_BACKOFF)

    def succeeded(self) -> None:
        """Decay back toward the floor so one early 429 does not throttle the
        whole run."""
        self._ok_since_penalty += 1
        if self._ok_since_penalty >= 50 and self.interval > self.floor:
            self.interval = max(self.floor, self.interval * 0.8)
            self._ok_since_penalty = 0


LIMITERS: dict = {}


def limiter_for(judge: str) -> RateLimiter:
    if judge not in LIMITERS:
        LIMITERS[judge] = RateLimiter(MIN_INTERVAL.get(judge, 0.0))
    return LIMITERS[judge]


def retry_after_seconds(exc) -> float | None:
    """Read Retry-After off an httpx error, if the provider sent one."""
    resp = getattr(exc, "response", None)
    if resp is None:
        return None
    raw = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None      # HTTP-date form; fall back to our own backoff


# Pinned. The version the API returns is asserted against these on every call.
MODELS = {
    "gemini":  "gemini-2.5-flash",
    "grok":    "grok-4.20-non-reasoning",
    "mistral": "mistral-large-latest",
    "openai":  "gpt-4o-mini",
    "claude":  "claude-opus-4-8",
}
# Substring that must appear in the returned version string.
VERSION_MUST_CONTAIN = {
    "gemini": "gemini-2.5-flash", "grok": "grok-4.20",
    "mistral": "mistral-large", "openai": "gpt-4o-mini",
    "claude": "claude-opus",
}

# Approximate input price per million tokens, for the pre-run estimate only.
PRICE_PER_M = {
    "gemini": 0.30, "grok": 3.00, "mistral": 2.00,
    "openai": 0.15, "claude": 15.00,
}
EST_INPUT_TOKENS = 320

# ===========================================================================
# prompts -- v1 (rounds up to and including French round 1) and v2
#
# v1 asked whether the text CONTAINS harassment. The same criterion applied
# to media conflated depicting abuse with committing it. v2 replaces
# "contains" with "targets": harassment is directed at a person or an
# identifiable group, while reporting, testimony and quotation are not.
#
# Note the scope change this implies. v1 is the prompt behind the published
# English results, so v2 supersedes rather than revises them; a like-for-like
# English comparison would need its own re-run. v1 is kept verbatim so both
# can be run over the same corpus and the disagreement set compared.
# Select with --prompt v1|v2 (default v2).
# ===========================================================================

# --- v1, superseded. Kept for the ablation. --------------------------------
PROMPT_V1 = (
    "Vous êtes un classificateur de modération de contenu pour des "
    "publications francophones sur les réseaux sociaux. Classez le texte "
    "suivant en exactement un mot :\n"
    "'Harassment' s'il contient du harcèlement, des propos haineux, des "
    "insultes, des menaces, du sexisme, du racisme ou un langage abusif ; "
    "sinon 'Neutral'. Répondez uniquement par le mot Harassment ou Neutral.\n"
    "Texte : {text}"
)

# --- v2, target criterion. --------------------------------------------------
PROMPT_V2 = (
    "Vous êtes un classificateur de modération de contenu pour des "
    "publications francophones sur les réseaux sociaux. Classez le texte "
    "suivant en exactement un mot.\n"
    "\n"
    "'Harassment' UNIQUEMENT si le texte vise une personne ou un groupe "
    "identifiable : insulte, propos haineux, menace, sexisme ou racisme "
    "dirigé contre quelqu'un ou contre un groupe (femmes, migrants, une "
    "communauté), incitation à s'en prendre à une personne.\n"
    "\n"
    "'Neutral' si le texte documente, informe, témoigne, cite ou débat sans "
    "viser quelqu'un. En particulier : reportage ou compte rendu de presse, "
    "témoignage d'une personne sur ce qu'elle a subi, dénonciation d'un "
    "harcèlement, paroles de chanson ou citation rapportée, registre "
    "grossier ou vulgaire sans destinataire, désaccord politique exprimé "
    "sans attaque personnelle.\n"
    "\n"
    "Rapporter ou dénoncer une agression n'est pas commettre un "
    "harcèlement. Un langage cru n'est pas en soi du harcèlement. En cas de "
    "doute sur l'existence d'une cible, répondez Neutral.\n"
    "Répondez uniquement par le mot Harassment ou Neutral.\n"
    "Texte : {text}"
)

PROMPTS = {"v1": PROMPT_V1, "v2": PROMPT_V2}

# Bound at parse time from --prompt. Default v2.
PROMPT_VERSION = "v2"
PROMPT = PROMPTS[PROMPT_VERSION]


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
    # KeyError and discards the only useful field in the response.
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
    # Content-policy rejections, before raise_for_status truncates the
    # exception to a URL fragment. 429 is NOT in this list: it must keep
    # flowing to the existing backoff path.
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
    if r.status_code in (400, 403, 422, 451):
        try:
            msg = str((r.json().get("error") or {}).get("message", ""))[:140]
        except Exception:
            msg = r.text[:140]
        raise JudgeRefusal(f"HTTP_{r.status_code}:{msg}", MODELS["claude"])

    r.raise_for_status()
    d = r.json()
    ver = d.get("model", MODELS["claude"])
    txt = "".join(b.get("text", "") for b in d.get("content", [])
                  if b.get("type") == "text")

    if not str(txt).strip():
        stop = d.get("stop_reason")
        if stop == "refusal":
            raise JudgeRefusal("BLOCKED:refusal", ver)
        if stop == "max_tokens":
            raise JudgeRefusal("EMPTY:MAX_TOKENS", ver)
    return txt, ver


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


async def call_judge(client, judge, text):
    """Returns (raw, version, latency_ms, error). Retries on transient faults."""
    fn = JUDGES[judge]
    err = ""
    t0 = time.time()
    pacer = limiter_for(judge)
    rl_attempt = 0
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            await pacer.wait()
            raw, ver = await fn(client, text)
            assert_version(judge, ver)
            if str(raw).strip():
                pacer.succeeded()
                return raw, ver, int(1000 * (time.time() - t0)), ""
            err = "empty completion"
        except ModelSubstitution:
            raise
        except JudgeRefusal as e:
            # Deterministic, so no retry. Recorded as ABSTAIN with the
            # provider's reason: a refusal is a fact about the pipeline, not
            # a verdict about the text, and must become neither a label nor
            # an invisible gap.
            return "", e.version, int(1000 * (time.time() - t0)), e.reason
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 429:
                err = f"{type(e).__name__}: {str(e)[:160]}"
                attempt += 1
                await asyncio.sleep(2 ** attempt)
                continue

            # A rate limit is a DEFERRED call, not a failed one. It does not
            # consume the normal attempt budget, and the wait comes from the
            # provider's own Retry-After when it sends one.
            rl_attempt += 1
            if rl_attempt > RATE_LIMIT_ATTEMPTS:
                err = (f"RATE_LIMITED: gave up after {rl_attempt} waits "
                       f"(interval now {pacer.interval:.1f}s)")
                break
            sleep_s = pacer.penalise(retry_after_seconds(e))
            await asyncio.sleep(sleep_s)
            continue
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"

        attempt += 1
        await asyncio.sleep(2 ** attempt)
    return "", "", int(1000 * (time.time() - t0)), err


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
    dead = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for j in JUDGES:
            verdicts, empties, vers = [], 0, set()
            for text, _ in PROBES:
                raw, ver, _, _ = await call_judge(client, j, text)
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


# ===========================================================================
# main
# ===========================================================================
def corpus_fingerprint(path: Path, df) -> dict:
    """
    Two hashes, answering different questions.

    file_sha256 changes if anything changed, including a re-quoted column.
    ids_sha256 changes only if the SET of texts changed, so a corpus
    regenerated with the same rows in a different column order still
    matches. They disagree exactly when the file moved but the sample did
    not, which is the case worth knowing about.
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


def archive_corpus(path: Path, fp: dict, dest: str):
    """Copy the corpus to GCS keyed by ids_sha256. Never overwrite."""
    from google.cloud import storage

    assert dest.startswith("gs://"), "--archive-corpus must start with gs://"
    bucket_name, _, prefix = dest[5:].rstrip("/").partition("/")
    name = f"{fp['ids_sha256'][:16]}_{path.name}"
    blob_name = f"{prefix}/{name}" if prefix else name
    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    if blob.exists():
        print(f"corpus already archived at gs://{bucket_name}/{blob_name}")
    else:
        blob.upload_from_filename(str(path), content_type="text/csv")
        print(f"corpus archived to gs://{bucket_name}/{blob_name}")
    return f"gs://{bucket_name}/{blob_name}"


async def run(args) -> None:
    corpus = Path(args.corpus) if args.corpus else CORPUS
    if not corpus.exists():
        raise SystemExit(f"corpus not found: {corpus}")
    df = pd.read_csv(corpus, dtype={"id": str})
    print(f"corpus {corpus}  {len(df)} rows")
    assert df["id"].is_unique, "duplicate ids in corpus"

    if args.n_rows:
        df = df.head(args.n_rows).copy()
        print(f"  --n-rows {args.n_rows}: smoke test only, not a labelling run")

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
        corpus=f"{corpus.stem}_{len(df)}",
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
        # hash has to live here, or two runs over different texts share a
        # fingerprint and run_store lets one resume into the other.
        # file_sha256 stays out on purpose: a cosmetic re-export of the same
        # rows should not invalidate a resumable run.
        extra={"corpus_ids_sha256": fp["ids_sha256"],
               "corpus_n_texts": str(fp["n_texts"])},
    )

    lock = threading.Lock()
    counts = {j: {"done": 0, "pos": 0, "abstain": 0, "err": 0} for j in JUDGES}
    t_start = time.time()

    with RunStore(spec, resume_from=args.resume) as store:
        # The run now carries its own inputs.
        if getattr(args, "archive_corpus", None):
            try:
                fp["gcs_uri"] = archive_corpus(corpus, fp, args.archive_corpus)
            except Exception as e:
                print(f"corpus archive failed ({e}); continuing")

        side = store.dir / "corpus.json"
        if side.exists():
            prev = json.loads(side.read_text(encoding="utf-8"))
            if prev.get("ids_sha256") != fp["ids_sha256"]:
                raise SystemExit(
                    f"\nCorpus mismatch on resume.\n"
                    f"  run {store.dir.name} used ids_sha256 "
                    f"{prev.get('ids_sha256')}\n"
                    f"  this corpus is            {fp['ids_sha256']}\n"
                    f"Resuming would blend two samples into one run.")
            if prev.get("file_sha256") != fp["file_sha256"]:
                print("note: same texts, different file bytes. Continuing.")
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

        sems = {j: asyncio.Semaphore(CONCURRENCY[j]) for j in JUDGES}

        async def worker(client, tid, text, judge):
            async with sems[judge]:
                raw, ver, ms, err = await call_judge(client, judge, str(text))
            lab = parse_verdict(raw)
            with lock:
                store.record(tid, judge, 0, raw, ver, parsed_label=lab,
                             latency_ms=ms, error=err)
                c = counts[judge]
                c["done"] += 1
                c["pos"] += (lab == POS)
                c["abstain"] += (lab == ABSTAIN)
                c["err"] += bool(err)
                total = sum(x["done"] for x in counts.values())
                if total % 500 == 0:
                    el = time.time() - t_start
                    rate = total / max(el, 1)
                    eta = (len(todo) - total) / max(rate, 1e-9) / 60
                    print(f"  {total:7,}/{len(todo):,}  "
                          f"{rate:5.1f} calls/s  eta {eta:5.1f} min")

        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                await asyncio.gather(
                    *(worker(client, t, x, j) for t, x, j in todo))
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
    # dtype is load-bearing. read_json parses 19-digit tweet ids as float64,
    # which holds ~15-16 significant digits: 2092152608249044999 becomes
    # 2092152608249044992 and the merge below silently drops the row. On the
    # 1,082-row French 2026 run that lost 1,076 of 1,082 verdicts.
    res = pd.read_json(Path("runs") / run_id / "results.jsonl", lines=True,
                       dtype={"text_id": str})
    res["text_id"] = res["text_id"].astype(str)
    piv = res.pivot_table(index="text_id", columns="judge",
                          values="parsed_label", aggfunc="first")
    votes = (piv == POS).sum(axis=1)
    valid = (piv != ABSTAIN).sum(axis=1)
    out = pd.DataFrame({"id": piv.index.astype(str), "votes": votes.values,
                        "valid_judges": valid.values})
    out["consensus"] = (out.votes >= K_CONSENSUS).map({True: POS, False: NEG})
    keep = [c for c in ("id", "source", "text", "label") if c in df.columns]
    merged = df[keep].merge(out, on="id", how="left")
    _lost = merged.consensus.isna().sum()
    if _lost:
        print(f"\n!! {_lost:,} of {len(merged):,} corpus rows got no verdict. "
              f"Do not report these numbers until that is explained.")
    dest = DATA / f"consensus_{run_id}.csv"
    merged.to_csv(dest, index=False)

    print(f"\n{'votes':>6s}{'texts':>8s}")
    for k in range(6):
        print(f"{k:6d}{int((out.votes == k).sum()):8,}")
    n = int((out.consensus == POS).sum())
    print(f"\nk>={K_CONSENSUS}: {n:,} positives ({100*n/len(out):.2f}%)")
    if "label" in merged.columns and merged.label.notna().any() \
            and (merged.label.astype(str).str.strip() != "").any():
        ok = (merged.label == merged.consensus).mean()
        print(f"agreement with the human gold label: {100*ok:.1f}%")
    print(f"\nwrote {dest}")
    print(f"run_id {run_id}")


ap = argparse.ArgumentParser()
ap.add_argument("--preflight", action="store_true")
ap.add_argument("--estimate", action="store_true", help="cost only, no calls")
ap.add_argument("--n-rows", type=int, default=0,
                help="first N rows -- smoke test only")
ap.add_argument("--corpus", help="path to the corpus CSV "
                "(id, source, text, label). Defaults to the merged French "
                "corpus.")
ap.add_argument("--resume", help="run_id to continue")
ap.add_argument("--archive-corpus", metavar="GS_URI",
                help="also copy the corpus CSV to "
                     "gs://bucket/prefix/, keyed by ids_sha256")
ap.add_argument("--max-cost", type=float, default=250.0,
                help="abort if the estimate exceeds this")
ap.add_argument("--prompt", choices=("v1", "v2"), default="v1",
                help="v1 = 'contains' criterion (default). Every text run in "
                     "this corpus uses v1, so the runs stay comparable with "
                     "the published English results; v2 = 'targets' "
                     "criterion, the default for vision_media.py, kept here "
                     "for the ablation. The prompt text is part of the run "
                     "fingerprint, so a run under the wrong version cannot be "
                     "silently merged with the others -- but you only find "
                     "out after paying for it.")
a = ap.parse_args()

# Bind PROMPT before preflight or any judge call reads it. Runs under
# different --prompt values must go to different run_ids; RunSpec carries
# prompt_text, so the stored spec already distinguishes them.
PROMPT_VERSION = a.prompt
PROMPT = PROMPTS[PROMPT_VERSION]
print(f"prompt version: {PROMPT_VERSION}")

if a.preflight:
    asyncio.run(preflight())
    sys.exit(0)

asyncio.run(run(a))
