"""
queries_loader.py — read queries.json (produced by emit-queries.mjs) so
collect.py stops building its own X API queries.

filters.yaml is the single source of truth. Regenerate with:
    node emit-queries.mjs > queries.json

Wire into collect.py in three places:

    from queries_loader import QuerySet
    qs = QuerySet.load("queries.json")

    # 1. where the query string is built
    for query in qs.queries(stream, locale="fr"):
        page = fetch(query, ...)

    # 2. where fetched posts are accumulated
    kept, dropped = qs.apply_exclusions(posts, stream, locale="fr",
                                        text_key="text")

    # 3. in the run summary, so the drop is visible and reportable
    print(qs.summary(stream, "fr"))
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

# collect.py's ids -> filters.yaml's ids
ALIASES = {
    "stream_a_reply_directed": "A_reply_directed",
    "stream_b_topic": "B_topic",
    "stream_c_lexicon_seeded": "C_lexicon_seeded",
    "stream_d_neutral_baseline": "D_neutral_baseline",
    # collect.py's --help uses the shorter forms
    "stream_c_lexicon": "C_lexicon_seeded",
    "stream_d_neutral": "D_neutral_baseline",
}

_ZERO_WIDTH = re.compile(r"[\u200b-\u200d\ufeff]")
_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ѕ": "s", "ν": "v", "ο": "o", "ρ": "p", "τ": "t", "α": "a",
})


def normalize(text: str) -> str:
    """Mirror the JS normalizer. Must stay in sync with filters.yaml."""
    s = unicodedata.normalize("NFKC", text or "")
    s = _ZERO_WIDTH.sub("", s)
    s = s.translate(_HOMOGLYPHS)
    s = re.sub(r"(.)\1{3,}", lambda m: m.group(1) * 3, s)
    return s.casefold()


@dataclass
class StreamSpec:
    queries: list[str]
    residual_exclusions: list[str]
    anchor: str | None = None
    max_chars: int = 0
    _patterns: list[re.Pattern] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._patterns = [
            re.compile(rf"(?:^|\W){re.escape(normalize(t))}(?:\W|$)")
            for t in self.residual_exclusions
        ]

    def excluded_by(self, text: str) -> str | None:
        """Return the first residual term present, or None."""
        norm = normalize(text)
        for term, pat in zip(self.residual_exclusions, self._patterns):
            if pat.search(norm):
                return term
        return None


class QuerySet:
    def __init__(self, data: dict):
        self.generated = data.get("generated")
        self.config_version = data.get("config_version")
        self._streams = {
            s: {
                loc: StreamSpec(
                    queries=v["queries"],
                    residual_exclusions=v.get("residual_exclusions", []),
                    anchor=v.get("anchor"),
                    max_chars=v.get("max_chars", 0),
                )
                for loc, v in locales.items()
            }
            for s, locales in data["streams"].items()
        }

    @classmethod
    def load(cls, path: str | Path = "queries.json") -> "QuerySet":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"{p} not found. Generate it first: node emit-queries.mjs > {p}"
            )
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def _spec(self, stream: str, locale: str) -> StreamSpec:
        key = ALIASES.get(stream, stream)
        if key not in self._streams:
            raise KeyError(f"unknown stream {stream!r}; have {sorted(self._streams)}")
        if locale not in self._streams[key]:
            raise KeyError(
                f"stream {key} has no locale {locale!r}; "
                f"have {sorted(self._streams[key])}"
            )
        return self._streams[key][locale]

    def queries(self, stream: str, locale: str = "fr") -> list[str]:
        return list(self._spec(stream, locale).queries)

    def apply_exclusions(
        self,
        posts: list[dict],
        stream: str,
        locale: str = "fr",
        text_key: str = "text",
    ) -> tuple[list[dict], list[dict]]:
        """
        Drop posts containing a residual exclusion term.

        Only streams with a negate: list have residuals, so this is a no-op
        everywhere else and is safe to call unconditionally. Dropped posts are
        returned rather than discarded: the count belongs in the run log, and
        an unexpectedly large drop usually means the query budget, not the data.
        """
        spec = self._spec(stream, locale)
        if not spec.residual_exclusions:
            return list(posts), []

        kept, dropped = [], []
        for p in posts:
            hit = spec.excluded_by(p.get(text_key, ""))
            if hit is None:
                kept.append(p)
            else:
                dropped.append({**p, "_excluded_by": hit})
        return kept, dropped

    def summary(self, stream: str, locale: str = "fr") -> str:
        spec = self._spec(stream, locale)
        key = ALIASES.get(stream, stream)
        bits = [
            f"{key}/{locale}: {len(spec.queries)} query(ies), "
            f"max {spec.max_chars} chars"
        ]
        if spec.anchor:
            bits.append(f"anchor={spec.anchor}")
        if spec.residual_exclusions:
            bits.append(
                f"{len(spec.residual_exclusions)} residual exclusions applied post-fetch"
            )
        return " | ".join(bits)


if __name__ == "__main__":
    qs = QuerySet.load()
    for stream in ALIASES:
        try:
            print(qs.summary(stream, "fr"))
        except KeyError as e:
            print(f"skip: {e}")

    sample = [
        {"id": 1, "text": "le conseil municipal se réunit demain"},
        {"id": 2, "text": "mdr quel génie"},
        {"id": 3, "text": "retourne dans ton pays"},
    ]
    kept, dropped = qs.apply_exclusions(sample, "stream_d_neutral_baseline", "fr")
    print(f"\nexclusion check: kept {len(kept)}, dropped {len(dropped)}")
    for d in dropped:
        print(f"  dropped {d['id']}: {d['_excluded_by']!r}")
