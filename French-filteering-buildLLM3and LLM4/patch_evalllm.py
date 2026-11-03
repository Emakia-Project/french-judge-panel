#!/usr/bin/env python3
"""
patch_evalllm.py — all outstanding fixes for evaluate-COREML-prediction/eval-llm/
label_corpus.py (the 507-line copy that produced both french2026_1082 runs).

Targeted at THAT file. Do not run it against the 915-line copy in
French-filtering-buildLLM3andLLM4; the anchors differ and it is a different
project.

WHAT IT FIXES

1. REFUSALS ARE INVISIBLE AND RETRIED
   Gemini returns HTTP 200 with no candidates and
   promptFeedback.blockReason = PROHIBITED_CONTENT. The handler swallows the
   KeyError, returns "", call_judge reports "empty completion" and retries
   three times at 2/4/8s. The block is deterministic, so every retry returns
   the same block. OpenAI (finish_reason=content_filter) and Claude
   (stop_reason=refusal) are the same class and equally invisible.

   A refusal is a fact about the pipeline, not a verdict about the text. It
   is still recorded as ABSTAIN; only the error field becomes specific, and
   the retries stop. 429 is deliberately untouched and still backs off.

2. THE RUN DOES NOT KNOW WHICH TEXTS IT SAW
   RunSpec carries a corpus stem and a row count. Two exports with the same
   stem and count but different rows are indistinguishable, and run_store
   would let one resume into the other. ids_sha256 goes into RunSpec.extra,
   which RunSpec.fingerprint() hashes, so the run directory name changes when
   the sample changes and run_store refuses the resume itself. notes is NOT
   hashed, so putting it there would do nothing.

3. THE CORPUS IS NOT ARCHIVED WITH THE RUN
   runs/<run_id>/corpus.json plus a copy of the CSV. Optionally to GCS with
   --archive-corpus.

USAGE
    python3 patch_evalllm.py --check
    python3 patch_evalllm.py
"""

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("label_corpus.py")
EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


# ------------------------------------------------------------------ imports
edit("hashlib + shutil imports",
'''import argparse
import asyncio
import json
import os''',
'''import argparse
import asyncio
import hashlib
import json
import os
import shutil as _shutil''')

# ---------------------------------------------------------------- exception
edit("JudgeRefusal exception",
'''async def ask_gemini(client: httpx.AsyncClient, text: str):''',
'''class JudgeRefusal(Exception):
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


async def ask_gemini(client: httpx.AsyncClient, text: str):''')

# ------------------------------------------------------------------- gemini
edit("ask_gemini refusal detection",
'''    r.raise_for_status()
    d = r.json()
    try:
        txt = d["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        txt = ""
    return txt, d.get("modelVersion", MODELS["gemini"])''',
'''    r.raise_for_status()
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
    return txt, ver''')

# ------------------------------------------------------------ openai-style
edit("_openai_style refusal detection",
'''    r.raise_for_status()
    d = r.json()
    return (d["choices"][0]["message"].get("content") or ""), d.get("model", model)''',
'''    # Content-policy rejections, before raise_for_status truncates the
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
    return txt, ver''')

# ------------------------------------------------------------------- claude
edit("ask_claude refusal detection",
'''    r.raise_for_status()
    d = r.json()
    txt = "".join(b.get("text", "") for b in d.get("content", [])
                  if b.get("type") == "text")
    return txt, d.get("model", MODELS["claude"])''',
'''    if r.status_code in (400, 403, 422, 451):
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
    return txt, ver''')

# --------------------------------------------------------------- call_judge
# NOTE: this copy returns a 4-tuple (raw, ver, ms, err), not the 6-tuple in
# the buildLLM3 copy. The refusal return must match that shape.
edit("call_judge handles refusal without retrying",
'''        except ModelSubstitution:
            raise
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"''',
'''        except ModelSubstitution:
            raise
        except JudgeRefusal as e:
            # Deterministic, so no retry. Recorded as ABSTAIN with the
            # provider's reason: a refusal is a fact about the pipeline, not
            # a verdict about the text, and must become neither a label nor
            # an invisible gap.
            return "", e.version, int(1000 * (time.time() - t0)), e.reason
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:160]}"''')

# ------------------------------------------------------------- fingerprint
edit("corpus_fingerprint helper",
'''async def run(args) -> None:''',
'''def corpus_fingerprint(path: Path, df) -> dict:
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
    ids_sha = hashlib.sha256("\\n".join(ids).encode("utf-8")).hexdigest()
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


async def run(args) -> None:''')

edit("compute fingerprint before any spend",
'''    cost = estimate(len(df))
    if args.estimate:
        return''',
'''    # Before any spend, so --estimate also reports what it would have judged.
    fp = corpus_fingerprint(corpus, df)
    print(f"\\ncorpus fingerprint")
    print(f"  texts       {fp['n_texts']:,}")
    print(f"  ids_sha256  {fp['ids_sha256']}")
    print(f"  file_sha256 {fp['file_sha256']}")

    cost = estimate(len(df))
    if args.estimate:
        return''')

edit("corpus identity into RunSpec.extra",
'''        notes="Stage 2 corpus labelling, frozen configuration of 9 Aug 2026",
    )''',
'''        notes=("Stage 2 corpus labelling, frozen configuration of 9 Aug 2026"
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
    )''')

edit("write corpus.json into the run dir",
'''    with RunStore(spec, resume_from=args.resume) as store:
        todo = [(r.id, r.text, j) for r in df.itertuples()''',
'''    with RunStore(spec, resume_from=args.resume) as store:
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
                    f"\\nCorpus mismatch on resume.\\n"
                    f"  run {store.dir.name} used ids_sha256 "
                    f"{prev.get('ids_sha256')}\\n"
                    f"  this corpus is            {fp['ids_sha256']}\\n"
                    f"Resuming would blend two samples into one run.")
            if prev.get("file_sha256") != fp["file_sha256"]:
                print("note: same texts, different file bytes. Continuing.")
        else:
            side.write_text(json.dumps(fp, indent=2), encoding="utf-8")
            try:
                _shutil.copy2(corpus, store.dir / corpus.name)
            except Exception as e:
                print(f"could not copy corpus into run dir ({e})")

        todo = [(r.id, r.text, j) for r in df.itertuples()''')


def add_flag(src):
    """Insert --archive-corpus after --resume, matching its indentation."""
    if 'add_argument("--archive-corpus"' in src:
        return src, "already present"
    lines = src.splitlines(keepends=True)
    hits = [n for n, l in enumerate(lines) if 'add_argument("--resume"' in l]
    if len(hits) != 1:
        return src, f"anchor MATCHES {len(hits)}"
    n = hits[0]
    indent = lines[n][:len(lines[n]) - len(lines[n].lstrip())]
    ap = lines[n].strip().split(".")[0]
    lines.insert(n + 1,
                 f'{indent}{ap}.add_argument("--archive-corpus", metavar="GS_URI",\n'
                 f'{indent}                help="also copy the corpus CSV to "\n'
                 f'{indent}                     "gs://bucket/prefix/, keyed by ids_sha256")\n')
    return "".join(lines), "ok"


def main(a):
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found; run from eval-llm/")
    src = TARGET.read_text(encoding="utf-8")

    n_lines = len(src.splitlines())
    if "french_corpus_merged" in src and n_lines > 800:
        sys.exit(f"This looks like the {n_lines}-line buildLLM3 copy, not the "
                 f"507-line eval-llm one. Refusing.")

    if "class JudgeRefusal" in src and "def corpus_fingerprint" in src:
        sys.exit("already fully patched")

    problems = []
    for name, old, _ in EDITS:
        n = src.count(old)
        print(f"  {'ok' if n == 1 else f'MATCHES {n}':12s} {name}")
        if n != 1:
            problems.append(name)

    _, status = add_flag(src)
    print(f"  {status:12s} --archive-corpus flag")
    if status.startswith("anchor"):
        problems.append("--archive-corpus flag")

    if problems:
        sys.exit(f"\n{len(problems)} block(s) did not match exactly once. "
                 f"Nothing written.")

    if a.check:
        print("\ncheck only, nothing written")
        return

    backup = TARGET.with_suffix(".py.evalbak")
    shutil.copy2(TARGET, backup)
    for _, old, new in EDITS:
        src = src.replace(old, new, 1)
    src, _ = add_flag(src)
    TARGET.write_text(src, encoding="utf-8")

    try:
        ast.parse(src)
    except SyntaxError as e:
        shutil.copy2(backup, TARGET)
        sys.exit(f"patched file does not parse ({e}); restored from backup")

    print(f"\nbacked up to {backup}")
    print(f"patched {TARGET}")
    print("\nVerify:")
    print("  ./env/bin/python label_corpus.py --preflight")
    print("  ./env/bin/python label_corpus.py --corpus data/<export>.csv --estimate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args())
