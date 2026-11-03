#!/usr/bin/env python3
"""
Patch label_corpus.py so a provider REFUSAL is distinguishable from an
empty response, and is not retried.

WHAT IS WRONG NOW
    Gemini returns HTTP 200 with no candidates array and
    promptFeedback.blockReason = PROHIBITED_CONTENT. The handler's
    try/except swallows the KeyError, returns "", and call_judge reports
    "empty completion" -- then retries three times at 2/4/8s backoff. The
    block is deterministic, so all three retries return the same block.
    Fifteen blocked rows cost 45 extra calls and produced a log line that
    named neither the cause nor the provider's own explanation.

    Grok's 403 raises HTTPStatusError, and str(e) truncates to
    "Client error '403 Forbidden' for url 'https://api.x.ai/v1/chat/" --
    the status code survives, the provider's message does not.

    OpenAI (finish_reason=content_filter) and Claude (stop_reason=refusal)
    have the same failure class and are equally invisible today.

WHAT THIS CHANGES
    A new JudgeRefusal exception carries a reason string. Handlers raise it
    on a block; call_judge catches it, records the reason in the error field,
    and returns WITHOUT retrying. The stored label is still ABSTAIN, so no
    verdict is invented and nothing downstream changes -- only the error
    field becomes specific:

        BLOCKED:PROHIBITED_CONTENT        instead of  empty completion
        HTTP_403:<provider message>       instead of  a truncated URL
        BLOCKED:content_filter
        EMPTY:MAX_TOKENS

    Rate limits are untouched: 429 is not intercepted and still flows to the
    existing backoff path.

USAGE
    python3 patch_judges.py              # apply
    python3 patch_judges.py --check      # report only
"""

import argparse
import shutil
import sys
from pathlib import Path

TARGET = Path("label_corpus.py")

EDITS = []


def edit(name, old, new):
    EDITS.append((name, old, new))


# --------------------------------------------------------------- exception
edit("JudgeRefusal class",
"""# ===========================================================================
# parsing -- an empty or unparseable reply is ABSTAIN, never the negative class
# ===========================================================================""",
"""class JudgeRefusal(Exception):
    \"\"\"
    The provider declined to answer.

    Distinct from a network fault and from a rate limit. A refusal is
    DETERMINISTIC: the same prompt returns the same block, so retrying
    re-spends the call for the same answer. Carries the provider's own
    reason so the log says why.
    \"\"\"

    def __init__(self, reason: str, version: str = ""):
        super().__init__(reason)
        self.reason = reason
        self.version = version


# ===========================================================================
# parsing -- an empty or unparseable reply is ABSTAIN, never the negative class
# ===========================================================================""")

# --------------------------------------------------------------- gemini
edit("ask_gemini refusal detection",
"""    r.raise_for_status()
    d = r.json()
    try:
        txt = d["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        txt = ""
    return txt, d.get("modelVersion", MODELS["gemini"])""",
"""    r.raise_for_status()
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
    return txt, ver""")

# --------------------------------------------------------------- openai-style
edit("_openai_style refusal detection",
"""    r = await client.post(
        url,
        headers={"Authorization": f"Bearer {key}"},
        json={"model": model,
              "messages": [{"role": "user", "content": PROMPT.format(text=text)}],
              "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS},
    )
    r.raise_for_status()
    d = r.json()
    return (d["choices"][0]["message"].get("content") or ""), d.get("model", model)""",
"""    r = await client.post(
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
    return txt, ver""")

# --------------------------------------------------------------- call_judge
edit("call_judge handles refusal without retrying",
"""        except ModelSubstitution:
            raise
        except Exception as e:""",
"""        except ModelSubstitution:
            raise
        except JudgeRefusal as e:
            # Deterministic, so no retry. Recorded as ABSTAIN with the
            # provider's reason: a refusal is a fact about the pipeline, not
            # a verdict about the text, and it must not become either a
            # label or an invisible gap.
            return ("", e.version, int(1000 * (time.time() - t0)), e.reason,
                    attempt, rl_attempt)
        except Exception as e:""")


def main(a):
    if not TARGET.exists():
        sys.exit(f"{TARGET} not found; run this from the project directory")
    src = TARGET.read_text(encoding="utf-8")

    if "class JudgeRefusal" in src:
        sys.exit("already patched (JudgeRefusal is present)")

    problems = []
    for name, old, _ in EDITS:
        n = src.count(old)
        status = "ok" if n == 1 else f"MATCHES {n}"
        print(f"  {status:12s} {name}")
        if n != 1:
            problems.append(name)

    if problems:
        sys.exit(f"\n{len(problems)} block(s) did not match exactly once. "
                 f"The file has drifted from what this patch expects; "
                 f"apply those by hand.")

    if a.check:
        print("\ncheck only, nothing written")
        return

    backup = TARGET.with_suffix(".py.bak")
    shutil.copy2(TARGET, backup)
    for _, old, new in EDITS:
        src = src.replace(old, new, 1)
    TARGET.write_text(src, encoding="utf-8")

    import ast
    try:
        ast.parse(src)
    except SyntaxError as e:
        shutil.copy2(backup, TARGET)
        sys.exit(f"patched file does not parse ({e}); restored from backup")

    print(f"\nbacked up to {backup}")
    print(f"patched {TARGET}")
    print("\nCLAUDE HANDLER NOT PATCHED: ask_claude was truncated in what I "
          "saw. Add the same check by hand -- after r.raise_for_status(), if "
          "the content is empty and stop_reason is 'refusal', raise "
          "JudgeRefusal(\"BLOCKED:refusal\", <returned model>).")
    print("\nVerify:  ./env/bin/python label_corpus.py --preflight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    main(ap.parse_args())
