#!/usr/bin/env python3
"""
Gemini Vision pass over the archived media.

A third verdict source, independent of the five text judges and of X's
possibly_sensitive flag. Runs on the archived bytes in media_local/ (or GCS),
never on X's URLs, which expire — storing URLs instead of bytes is what made
Vision abstain on 100% of the previous evaluation stream.

Deliberately blind by default. The prompt contains the image or video and
nothing else: no post text, no platform flag, no text-judge verdict. That
isolates the question the corpus can actually answer — is the abuse visible
in the media alone — and keeps the comparison with the text panel meaningful.
Pass --with-text to run the multimodal variant as a separate labelled run.

Conventions follow label_corpus.py: pinned model with a version assertion,
ABSTAIN never collapsed into the negative class, append-only JSONL so a
partial run resumes, and a cost estimate before any call.

Usage:
    export GEMINI_API_KEY=...
    python vision_media.py --estimate
    python vision_media.py --n 8 --flagged-first   # the X-flagged ones first
    python vision_media.py                          # everything
    python vision_media.py --resume runs_vision/20260828T1200Z
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

POS, NEG, ABSTAIN = "Harassment", "Neutral", "ABSTAIN"

MODEL = "gemini-2.5-flash"
VERSION_MUST_CONTAIN = "gemini-2.5-flash"

TEMPERATURE = 0
MAX_TOKENS = 100
TIMEOUT = 300.0
CONCURRENCY = 4

# Inline base64 is capped by the request size. Anything larger goes through
# the Files API. 15 MB leaves headroom under the ~20 MB request ceiling.
INLINE_LIMIT = 15 * 1024 * 1024
FILES_API_LIMIT = 2 * 1024 * 1024 * 1024

BASE = "https://generativelanguage.googleapis.com"

# ===========================================================================
# prompts -- v1 (round 1, pipeline validation) and v2 (target criterion)
#
# v1 asked whether the media CONTAINS harassment. Manual inspection of the
# round 1 flagged set showed that criterion conflates depicting abuse with
# committing it: press photographs of a battered face, a victim posting her
# own injuries, conspiracy imagery with no named target, and quoted song
# lyrics all satisfy "contains" while none of them is an act of harassment
# directed at anyone.
#
# v2 replaces "contains" with "targets". Harassment is directed at a person
# or an identifiable group; documentation, reporting and testimony are not.
# The rule is one-directional by design: material that merely depicts abuse
# falls to Neutral, so a survivor's own testimony is never suppressed.
#
# v1 is kept verbatim so the two can be run over the same corpus and the
# disagreement set compared. Select with --prompt v1|v2 (default v2).
# ===========================================================================

# --- v1, round 1. Superseded by v2, kept for the ablation. -----------------
PROMPT_MEDIA_ONLY_V1 = (
    "Vous êtes un classificateur de modération de contenu. Examinez "
    "uniquement cette image ou vidéo, sans aucun texte accompagnant. "
    "Classez-la en exactement un mot :\n"
    "'Harassment' si elle contient du harcèlement, des propos haineux, des "
    "insultes, des menaces, du sexisme, du racisme, un symbole haineux, ou "
    "un mème ou une capture d'écran à caractère abusif ; sinon 'Neutral'.\n"
    "Répondez uniquement par le mot Harassment ou Neutral."
)

PROMPT_WITH_TEXT_V1 = (
    "Vous êtes un classificateur de modération de contenu. Examinez cette "
    "image ou vidéo accompagnée du texte de la publication. Classez "
    "l'ensemble en exactement un mot :\n"
    "'Harassment' s'il contient du harcèlement, des propos haineux, des "
    "insultes, des menaces, du sexisme, du racisme ou un langage abusif ; "
    "sinon 'Neutral'.\n"
    "Répondez uniquement par le mot Harassment ou Neutral.\n"
    "Texte de la publication : {text}"
)

# --- v2, target criterion. --------------------------------------------------
PROMPT_MEDIA_ONLY_V2 = (
    "Vous êtes un classificateur de modération de contenu. Examinez "
    "uniquement cette image ou vidéo, sans aucun texte accompagnant. "
    "Classez-la en exactement un mot.\n"
    "\n"
    "'Harassment' UNIQUEMENT si l'image elle-même vise une personne ou un "
    "groupe identifiable : insulte visuelle, mème dégradant visant "
    "quelqu'un, symbole haineux, capture d'écran diffusée pour humilier, "
    "menace explicite.\n"
    "\n"
    "'Neutral' si l'image documente, informe, témoigne ou illustre sans "
    "viser quelqu'un. En particulier : photo ou reportage de presse, image "
    "d'une victime ou de ses blessures, paroles de chanson, image "
    "complotiste sans cible nommée, contenu choquant ou violent qui ne "
    "s'adresse à personne.\n"
    "\n"
    "Montrer une agression n'est pas commettre un harcèlement. En cas de "
    "doute sur l'existence d'une cible, répondez Neutral.\n"
    "Répondez uniquement par le mot Harassment ou Neutral."
)

PROMPT_WITH_TEXT_V2 = (
    "Vous êtes un classificateur de modération de contenu. Examinez cette "
    "image ou vidéo accompagnée du texte de la publication. Classez "
    "l'ensemble en exactement un mot.\n"
    "\n"
    "'Harassment' UNIQUEMENT si la publication vise une personne ou un "
    "groupe identifiable : insulte, mème dégradant, symbole haineux, "
    "capture d'écran diffusée pour humilier, menace explicite.\n"
    "\n"
    "'Neutral' si la publication documente, informe, témoigne, cite ou "
    "illustre sans viser quelqu'un. En particulier : reportage de presse, "
    "témoignage d'une victime sur ce qu'elle a subi, paroles de chanson "
    "citées, image complotiste sans cible nommée, registre grossier sans "
    "destinataire.\n"
    "\n"
    "Montrer ou raconter une agression n'est pas commettre un harcèlement. "
    "En cas de doute sur l'existence d'une cible, répondez Neutral.\n"
    "Répondez uniquement par le mot Harassment ou Neutral.\n"
    "Texte de la publication : {text}"
)

PROMPTS = {
    "v1": (PROMPT_MEDIA_ONLY_V1, PROMPT_WITH_TEXT_V1),
    "v2": (PROMPT_MEDIA_ONLY_V2, PROMPT_WITH_TEXT_V2),
}

# Bound at parse time from --prompt. Default v2.
PROMPT_VERSION = "v2"
PROMPT_MEDIA_ONLY, PROMPT_WITH_TEXT = PROMPTS[PROMPT_VERSION]


# ===========================================================================
# parsing -- unparseable is ABSTAIN, never the negative class
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


class ModelSubstitution(RuntimeError):
    pass


def assert_version(returned: str) -> None:
    if VERSION_MUST_CONTAIN.lower() not in str(returned or "").lower():
        raise ModelSubstitution(
            f"requested {MODEL}, provider returned {returned!r}. A silently "
            f"substituted model gives one corpus two labelling regimes. "
            f"Verify before resuming.")


# ===========================================================================
# media input
# ===========================================================================
def mime_for(path: Path, media_type: str) -> str:
    guess, _ = mimetypes.guess_type(str(path))
    if guess:
        return guess
    return {"photo": "image/jpeg", "video": "video/mp4",
            "animated_gif": "video/mp4"}.get(media_type, "application/octet-stream")


async def upload_file(client: httpx.AsyncClient, key: str, path: Path,
                      mime: str) -> str:
    """Files API for anything too large to inline. Returns the file URI."""
    size = path.stat().st_size
    start = await client.post(
        f"{BASE}/upload/v1beta/files",
        params={"key": key},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": mime,
        },
        json={"file": {"display_name": path.name}},
    )
    start.raise_for_status()
    upload_url = start.headers.get("x-goog-upload-url")
    if not upload_url:
        raise RuntimeError("no upload URL returned")

    up = await client.post(
        upload_url,
        headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                 "X-Goog-Upload-Command": "upload, finalize"},
        content=path.read_bytes(),
    )
    up.raise_for_status()
    info = up.json()["file"]

    # Video must finish processing before it can be referenced.
    name, uri = info["name"], info["uri"]
    for _ in range(60):
        if info.get("state") == "ACTIVE":
            return uri
        if info.get("state") == "FAILED":
            raise RuntimeError("file processing FAILED")
        await asyncio.sleep(3)
        st = await client.get(f"{BASE}/v1beta/{name}", params={"key": key})
        st.raise_for_status()
        info = st.json()
    raise RuntimeError("file never became ACTIVE")


async def ask_vision(client: httpx.AsyncClient, key: str, path: Path,
                     media_type: str, post_text: str | None):
    mime = mime_for(path, media_type)
    size = path.stat().st_size

    if size <= INLINE_LIMIT:
        part = {"inline_data": {"mime_type": mime,
                                "data": base64.b64encode(
                                    path.read_bytes()).decode()}}
    else:
        uri = await upload_file(client, key, path, mime)
        part = {"file_data": {"mime_type": mime, "file_uri": uri}}

    prompt = (PROMPT_WITH_TEXT.format(text=post_text) if post_text is not None
              else PROMPT_MEDIA_ONLY)

    r = await client.post(
        f"{BASE}/v1beta/models/{MODEL}:generateContent",
        params={"key": key},
        json={
            "contents": [{"parts": [part, {"text": prompt}]}],
            "generationConfig": {
                "temperature": TEMPERATURE,
                "maxOutputTokens": MAX_TOKENS,
                # Load-bearing, same as the text judges: without it the model
                # spends its output budget on reasoning and returns no content.
                "thinkingConfig": {"thinkingBudget": 0},
            },
        },
    )
    r.raise_for_status()
    d = r.json()
    ver = d.get("modelVersion", MODEL)

    # A blocked image returns HTTP 200 with NO candidates array and the reason
    # in promptFeedback. Reading candidates first turns that into a KeyError
    # and discards the only informative field in the response. On a harassment
    # corpus the refused images are plausibly the most severe ones, so losing
    # the reason biases every downstream comparison.
    block = (d.get("promptFeedback") or {}).get("blockReason")
    if block:
        return "", ver, f"BLOCKED:{block}"

    cands = d.get("candidates") or []
    if not cands:
        return "", ver, "BLOCKED:NO_CANDIDATES"

    cand = cands[0]
    try:
        txt = cand["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        txt = ""

    if not str(txt).strip():
        fr = cand.get("finishReason", "UNKNOWN")
        if fr in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION",
                  "SPII", "IMAGE_SAFETY"):
            return "", ver, f"BLOCKED:{fr}"
        if fr == "MAX_TOKENS":
            # Deterministic at temperature 0: a config problem, not a
            # transient one. Surface it rather than retrying.
            return "", ver, "EMPTY:MAX_TOKENS"
    return txt, ver, ""


# ===========================================================================
# work list
# ===========================================================================
def load_manifest(manifest: Path) -> dict:
    if not manifest.exists():
        sys.exit(
            f"manifest not found: {manifest}\n"
            f"It lives next to the downloaded media, in the collection "
            f"directory, not here. Pass the full path:\n"
            f"  --manifest '/path/to/collect-French data/media_local/manifest.jsonl'")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["media_key"]] = r
    return rows


def resolve(local_path: str, manifest: Path) -> Path:
    """
    local_path in the manifest is relative to the directory download_media.py
    ran in, not to wherever this script runs. Resolve against the manifest's
    grandparent (manifest lives at <root>/media_local/manifest.jsonl).
    """
    p = Path(local_path)
    if p.is_absolute():
        return p
    root = manifest.parent.parent
    cand = root / p
    if cand.exists():
        return cand
    # also try alongside the manifest itself
    alt = manifest.parent / p.name
    return alt if alt.exists() else cand


def build_worklist(args) -> list[dict]:
    """Join media_archive to tweets in BigQuery, or fall back to the manifest."""
    mpath = Path(args.manifest)
    manifest = load_manifest(mpath)

    if args.no_bq:
        return [{"media_key": k, "tweet_id": None, "type": v.get("type"),
                 "local_path": str(resolve(v["local_path"], mpath)),
                 "x_flag": None, "text": None}
                for k, v in manifest.items() if v.get("local_path")]

    from google.cloud import bigquery
    client = bigquery.Client(project=args.project)
    sql = f"""
        SELECT t.id AS tweet_id, mk AS media_key, m.type,
               t.possibly_sensitive AS x_flag, t.text
        FROM `{args.project}.{args.dataset}.tweets_dedup` t,
             UNNEST(t.media_keys) AS mk
        JOIN `{args.project}.{args.dataset}.media_dedup` m
          ON m.media_key = mk
        ORDER BY t.possibly_sensitive DESC, t.id
    """
    rows = []
    for r in client.query(sql).result():
        man = manifest.get(r.media_key)
        if not man or not man.get("local_path"):
            continue
        rows.append({"media_key": r.media_key, "tweet_id": r.tweet_id,
                     "type": r.type,
                     "local_path": str(resolve(man["local_path"], mpath)),
                     "x_flag": r.x_flag, "text": r.text})
    return rows


# ===========================================================================
# main
# ===========================================================================
async def run(args):
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        sys.exit("Set GEMINI_API_KEY.")

    work = build_worklist(args)

    if args.flagged_first:
        work.sort(key=lambda w: (w["x_flag"] is not True,))
    if args.n:
        work = work[:args.n]

    missing = [w for w in work if not Path(w["local_path"]).exists()]
    if missing:
        print(f"WARNING {len(missing)} archived files not on disk; skipping")
        work = [w for w in work if Path(w["local_path"]).exists()]

    total_bytes = sum(Path(w["local_path"]).stat().st_size for w in work)
    print(f"\n{len(work):,} media objects, {total_bytes/1e6:,.1f} MB")
    by_type = {}
    for w in work:
        by_type[w["type"]] = by_type.get(w["type"], 0) + 1
    print(f"  {by_type}")
    print(f"  {sum(1 for w in work if Path(w['local_path']).stat().st_size > INLINE_LIMIT)}"
          f" go through the Files API (over {INLINE_LIMIT/1e6:.0f} MB)")
    if args.estimate:
        print("\n--estimate, no calls made. Video is billed by duration, so a "
              "byte count is not a price; run a small --n first and read the "
              "measured spend in the console.")
        return

    if args.resume:
        run_dir = Path(args.resume)
        done = set()
        res = run_dir / "results.jsonl"
        if res.exists():
            for line in res.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(json.loads(line)["media_key"])
        work = [w for w in work if w["media_key"] not in done]
        print(f"resuming {run_dir}: {len(done)} done, {len(work)} left")
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path("runs_vision") / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "spec.json").write_text(json.dumps({
            "model": MODEL, "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "prompt": PROMPT_WITH_TEXT if args.with_text else PROMPT_MEDIA_ONLY,
            "prompt_version": PROMPT_VERSION,
            "mode": "with_text" if args.with_text else "media_only",
            "started": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    if not work:
        print("nothing to do")
        return

    out = open(run_dir / "results.jsonl", "a", encoding="utf-8")
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(CONCURRENCY)
    counts = {"done": 0, "pos": 0, "abstain": 0, "err": 0}
    t0 = time.time()

    async def worker(client, w):
        async with sem:
            t = time.time()
            raw, ver, err = None, None, None
            try:
                raw, ver, refusal = await ask_vision(
                    client, key, Path(w["local_path"]), w["type"],
                    w["text"] if args.with_text else None)
                assert_version(ver)
                # A refusal is a fact about the pipeline, not a verdict about
                # the image. It stays ABSTAIN; only the reason is recorded.
                if refusal:
                    err = refusal
            except ModelSubstitution:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {str(e)[:200]}"
            lab = parse_verdict(raw)
            row = {"media_key": w["media_key"], "tweet_id": w["tweet_id"],
                   "type": w["type"], "verdict": lab, "raw": raw,
                   "model_version": ver, "error": err,
                   "abstained": lab == ABSTAIN,
                   "latency_ms": int((time.time() - t) * 1000)}
            async with lock:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                counts["done"] += 1
                counts["pos"] += lab == POS
                counts["abstain"] += lab == ABSTAIN
                counts["err"] += bool(err)
                if counts["done"] % 25 == 0:
                    print(f"  {counts['done']}/{len(work)}  "
                          f"flagged {counts['pos']}  abstain {counts['abstain']}")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        try:
            await asyncio.gather(*(worker(client, w) for w in work))
        except ModelSubstitution as e:
            out.close()
            print(f"\nABORTED: {e}")
            print(f"Partial results in {run_dir}. Resume with "
                  f"--resume {run_dir}")
            raise SystemExit(1)
    out.close()

    valid = counts["done"] - counts["abstain"]
    print(f"\n{'='*60}")
    print(f"completed in {(time.time()-t0)/60:.1f} min")
    print(f"  judged   {counts['done']}")
    print(f"  flagged  {counts['pos']} ({100*counts['pos']/max(valid,1):.1f}% "
          f"of non-abstained)")
    print(f"  abstain  {counts['abstain']}")
    print(f"  errors   {counts['err']}")
    if counts["abstain"] == counts["done"] and counts["done"]:
        print("\n  >>> universal abstention. Check the media actually opens "
              "before reporting any rate. This is the failure mode from the "
              "previous evaluation.")
    print(f"\nresults: {run_dir/'results.jsonl'}")
    print("\nload to BigQuery as vision_verdicts, keyed on media_key, then "
          "join to tweets_dedup on tweet_id.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="media_local/manifest.jsonl")
    ap.add_argument("--dataset", default=os.environ.get("BQ_DATASET"))
    ap.add_argument("--project",
                    default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR-PROJECT-ID"))
    ap.add_argument("--no-bq", action="store_true",
                    help="use the manifest alone; no tweet_id or x_flag join")
    ap.add_argument("--with-text", action="store_true",
                    help="multimodal: show the post text alongside the media. "
                         "Run as a SEPARATE run from the blind default so the "
                         "two are comparable.")
    ap.add_argument("--flagged-first", action="store_true",
                    help="order X-flagged posts first")
    ap.add_argument("--n", type=int, default=0, help="first N only")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--resume", help="run dir to continue")
    ap.add_argument("--prompt", choices=("v1", "v2"), default="v2",
                    help="v1 = round 1 'contains' criterion, kept for the "
                         "ablation; v2 = 'targets' criterion (default)")
    args = ap.parse_args()

    # Bind the prompt globals before anything reads them. Runs under different
    # --prompt values must go to different run dirs; spec.json records which.
    global PROMPT_VERSION, PROMPT_MEDIA_ONLY, PROMPT_WITH_TEXT
    PROMPT_VERSION = args.prompt
    PROMPT_MEDIA_ONLY, PROMPT_WITH_TEXT = PROMPTS[PROMPT_VERSION]
    print(f"prompt version: {PROMPT_VERSION}")

    if not args.no_bq and not args.dataset:
        sys.exit("Set BQ_DATASET, or pass --no-bq to run from the manifest.")

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
