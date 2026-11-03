#!/usr/bin/env python3
"""
X API v2 recent search collector.

Replaces the Node/Express toolkit. No server, no webhook, no curl.

Design rules, from collection_queries.yaml:
  - You pay per post RETURNED. Retrieve once, store raw, filter offline.
  - Raw JSON hits disk BEFORE BigQuery. A BQ failure never costs a second call.
  - Hard budget cap enforced inside the pagination loop, not at the entry point.
  - Every row carries its stream label so sampling design stays auditable.

Usage:
    export X_BEARER_TOKEN="..."
    export GOOGLE_CLOUD_PROJECT="YOUR-PROJECT-ID"

    # smoke test, 10 posts, nothing written to BigQuery
    python collect.py --query "(retraites OR grève) lang:fr -is:retweet" \
        --stream stream_b_topic --limit 10 --no-bq

    # same but land it in BigQuery
    python collect.py --query "..." --stream stream_b_topic --limit 10

    # load a previous raw run into BigQuery without re-calling the API
    python collect.py --replay raw/20260817T1430Z --stream stream_b_topic

    # use the compiled queries from filters.yaml instead of --query
    node emit-queries.mjs > queries.json
    python collect.py --from-config --stream stream_c_lexicon --locale fr \
        --round 1 --limit 200 --no-bq
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

ENDPOINT = "https://api.x.com/2/tweets/search/recent"

POST_READ_USD = 0.005
USER_READ_USD = 0.010

# Free. You pay per post returned, not per field on it. Ask for everything.
TWEET_FIELDS = [
    "id", "text", "created_at", "author_id", "conversation_id",
    "in_reply_to_user_id", "referenced_tweets", "possibly_sensitive",
    "public_metrics", "entities", "attachments", "context_annotations",
    "lang", "source", "geo", "withheld", "reply_settings",
]

# url and variants are the two the old Node schema dropped. Without them the
# media rows are unfetchable, which is what made Vision abstain.
MEDIA_FIELDS = [
    "media_key", "type", "url", "preview_image_url", "variants",
    "duration_ms", "alt_text", "width", "height",
]

USER_FIELDS = ["id", "username", "name", "public_metrics", "verified", "created_at"]


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------

def build_params(query, page_size, with_parents, with_users):
    expansions = ["attachments.media_keys"]
    if with_parents:
        expansions.append("referenced_tweets.id")
    if with_users:
        expansions.append("author_id")

    params = {
        "query": query,
        "max_results": page_size,
        "tweet.fields": ",".join(TWEET_FIELDS),
        "media.fields": ",".join(MEDIA_FIELDS),
        "expansions": ",".join(expansions),
    }
    if with_users:
        params["user.fields"] = ",".join(USER_FIELDS)
    return params


def get_page(params, token, next_token=None):
    """One call. Retries only on 429 and 5xx — never on auth or bad-query."""
    if next_token:
        params = {**params, "next_token": next_token}

    headers = {"Authorization": f"Bearer {token}", "User-Agent": "research-collector/2.0"}

    for attempt in range(4):
        r = requests.get(ENDPOINT, params=params, headers=headers, timeout=45)

        if r.status_code == 200:
            return r.json()

        if r.status_code in (401, 403):
            sys.exit(f"Auth failed ({r.status_code}). Not retrying.\n{r.text[:500]}")

        if r.status_code == 400:
            sys.exit(f"Bad query. Not retrying.\n{r.text[:800]}")

        if r.status_code == 429:
            reset = r.headers.get("x-rate-limit-reset")
            wait = max(5, int(reset) - int(time.time())) if reset else 30
            wait = min(wait, 300)
            print(f"  429, sleeping {wait}s")
            time.sleep(wait)
            continue

        if 500 <= r.status_code < 600:
            wait = 2 ** attempt
            print(f"  {r.status_code}, retry in {wait}s")
            time.sleep(wait)
            continue

        sys.exit(f"Unexpected {r.status_code}\n{r.text[:500]}")

    sys.exit("Gave up after 4 attempts.")


def collect(query, limit, token, page_size, with_parents, with_users, raw_dir):
    """
    Paginate to `limit` posts. The cap is checked every loop, not once.

    Note the API floor: max_results cannot go below 10, so the final page can
    overshoot `limit` by up to 9 posts. You pay for those. It is not a bug,
    but do not treat `limit` as an exact spend ceiling.
    """
    params = build_params(query, page_size, with_parents, with_users)
    raw_dir.mkdir(parents=True, exist_ok=True)

    pages, total, next_token, page_no = [], 0, None, 0

    while total < limit:
        remaining = limit - total
        # The API floor is 10. Asking for the floor to satisfy a remainder of 1
        # bills 10 posts to gain 1. Stop instead; the shortfall rolls forward
        # to the next query's share.
        if remaining < 10 and total > 0:
            print(f"  {remaining} short of limit, stopping rather than "
                  f"buying a 10-post page")
            break

        page_no += 1
        params["max_results"] = max(10, min(page_size, remaining))

        print(f"page {page_no} (have {total}/{limit})")
        payload = get_page(params, token, next_token)

        path = raw_dir / f"page_{page_no:03d}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8")

        got = len(payload.get("data", []))
        total += got
        pages.append(payload)
        print(f"  +{got} posts -> {path.name}")

        next_token = payload.get("meta", {}).get("next_token")
        if not next_token or got == 0:
            print("  no more pages")
            break

    return pages


# --------------------------------------------------------------------------
# flatten
# --------------------------------------------------------------------------

def to_datetime(iso):
    """X returns 2026-08-17T14:30:00.000Z. BigQuery DATETIME wants naive."""
    if not iso:
        return None
    return iso.replace("Z", "").split(".")[0]


def as_json(value):
    """
    entities, geo and withheld have nested shapes that vary by post type.
    A strict RECORD column breaks on the first unusual post and takes the whole
    batch with it, after you have already paid for those posts. Store the JSON
    text instead and PARSE_JSON it on the way out.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def flatten(pages, category, subcategory, stream, round_no):
    tweets, media, users = [], [], []
    seen_t, seen_m, seen_u = set(), set(), set()

    for payload in pages:
        for t in payload.get("data", []):
            if t["id"] in seen_t:
                continue
            seen_t.add(t["id"])
            tweets.append({
                "id": t["id"],
                "text": t.get("text"),
                "category": category,
                "subcategory": subcategory,
                "stream": stream,
                "round": round_no,
                "tweet_url": f"https://x.com/i/status/{t['id']}",
                "lang": t.get("lang"),
                "author_id": t.get("author_id"),
                "created_at": to_datetime(t.get("created_at")),
                "source": t.get("source"),
                "possibly_sensitive": t.get("possibly_sensitive"),
                "public_metrics": t.get("public_metrics"),
                "referenced_tweets": t.get("referenced_tweets", []),
                "in_reply_to_user_id": t.get("in_reply_to_user_id"),
                "conversation_id": t.get("conversation_id"),
                "withheld": as_json(t.get("withheld")),
                "geo": as_json(t.get("geo")),
                "context_annotations": t.get("context_annotations", []),
                "entities": as_json(t.get("entities")),
                "media_keys": (t.get("attachments") or {}).get("media_keys", []),
            })

        inc = payload.get("includes", {})

        for m in inc.get("media", []):
            if m["media_key"] in seen_m:
                continue
            seen_m.add(m["media_key"])
            variants = m.get("variants", [])
            best = max(
                (v for v in variants if v.get("bit_rate") is not None),
                key=lambda v: v["bit_rate"], default=None,
            )
            media.append({
                "media_key": m["media_key"],
                "type": m.get("type"),
                "url": m.get("url"),
                "preview_image_url": m.get("preview_image_url"),
                "best_variant_url": best.get("url") if best else None,
                "variants": variants,
                "duration_ms": m.get("duration_ms"),
                "alt_text": m.get("alt_text"),
                "width": m.get("width"),
                "height": m.get("height"),
                "category": category,
                "subcategory": subcategory,
                "stream": stream,
                "round": round_no,
            })

        for u in inc.get("users", []):
            if u["id"] in seen_u:
                continue
            seen_u.add(u["id"])
            users.append({
                "id": u["id"],
                "username": u.get("username"),
                "name": u.get("name"),
                "created_at": to_datetime(u.get("created_at")),
                "public_metrics": u.get("public_metrics"),
                "verified": u.get("verified"),
                "category": category,
                "subcategory": subcategory,
            })

    return tweets, media, users


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def report(tweets, media, users, pages, dropped=None):
    n = len(tweets)
    print(f"\n{'='*60}\nunique posts : {n}")
    print(f"media objects: {len(media)}")
    print(f"user objects : {len(users)}")

    parents = sum(len(p.get("includes", {}).get("tweets", [])) for p in pages)
    if parents:
        print(f"parent posts : {parents}")

    # Cost counts dropped posts too. You paid for them; the exclusion is
    # post-fetch. Reporting cost on kept posts only understates the spend.
    # Billed on posts RETURNED, not posts kept. Duplicates across queries and
    # posts removed by residual exclusions were both paid for.
    returned = sum(len(p.get("data", [])) for p in pages)
    dupes = returned - n - len(dropped or [])
    cost = (returned + parents) * POST_READ_USD + len(users) * USER_READ_USD
    if dupes > 0:
        print(f"duplicates   : {dupes} (billed, deduped across queries)")
    if dropped:
        print(f"excluded     : {len(dropped)} (residual terms, billed anyway)")
        seen = {}
        for d in dropped:
            seen[d["_excluded_by"]] = seen.get(d["_excluded_by"], 0) + 1
        top = sorted(seen.items(), key=lambda kv: -kv[1])[:5]
        print("  top terms  : " + ", ".join(f"{t}({c})" for t, c in top))
    print(f"est. cost    : ${cost:.3f}")

    if not n:
        return

    absent = sum(1 for t in tweets if t["possibly_sensitive"] is None)
    true_ = sum(1 for t in tweets if t["possibly_sensitive"] is True)
    print(f"\npossibly_sensitive: absent={absent}  True={true_}  "
          f"False={n - absent - true_}")
    if absent == n:
        print("  >>> field never returned. Any 0% rate is a request bug,")
        print("      not a platform property. Do not report it.")

    replies = sum(1 for t in tweets if t["in_reply_to_user_id"])
    langs = {}
    for t in tweets:
        langs[t["lang"]] = langs.get(t["lang"], 0) + 1
    print(f"replies: {replies}/{n}   lang: {langs}")

    unfetchable = [m for m in media
                   if not m["url"] and not m["best_variant_url"]]
    if media:
        print(f"media with no fetchable URL: {len(unfetchable)}/{len(media)}")


# --------------------------------------------------------------------------
# bigquery
# --------------------------------------------------------------------------

def load(dataset, tweets, media, users):
    from google.cloud import bigquery

    client = bigquery.Client()

    def push(table, rows, key):
        if not rows:
            return
        ref = f"{client.project}.{dataset}.{table}"
        # row_ids make re-runs idempotent within BigQuery's dedup window.
        errors = client.insert_rows_json(ref, rows, row_ids=[r[key] for r in rows])
        if errors:
            print(f"!! {table}: {len(errors)} rows rejected")
            print(json.dumps(errors[:3], indent=2)[:1500])
            raise SystemExit(f"insert failed for {table}")
        print(f"   {len(rows):>5} -> {ref}")

    print("\nloading to BigQuery")
    push("tweets", tweets, "id")
    push("media", media, "media_key")
    push("users", users, "id")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query")
    ap.add_argument("--from-config", nargs="?", const="queries.json",
                    default=None, metavar="QUERIES_JSON",
                    help="read compiled queries from queries.json instead of "
                         "--query. filters.yaml stays the single source of "
                         "truth; regenerate with: node emit-queries.mjs")
    ap.add_argument("--locale", default="fr",
                    help="locale key inside queries.json (default: fr)")
    ap.add_argument("--limit", type=int, default=10,
                    help="hard cap on posts. 10 is the API minimum per page.")
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--round", type=int, required=True, dest="round_no",
                    help="collection round. 1 = unsteered original query list; "
                         "2+ = adjusted after seeing predictions. Stamped on "
                         "every row so the two are separable at analysis time.")
    ap.add_argument("--stream", required=True,
                    help="stream_a_reply_directed | stream_b_topic | "
                         "stream_c_lexicon | stream_d_neutral")
    ap.add_argument("--category", default="")
    ap.add_argument("--subcategory", default="")
    ap.add_argument("--dataset", default=os.environ.get("BQ_DATASET"),
                    help="BigQuery dataset. Defaults to $BQ_DATASET. No "
                         "built-in default: a wrong one writes 2026 French "
                         "rows into the 2024 English tables.")
    ap.add_argument("--with-parents", action="store_true",
                    help="expand referenced_tweets.id — each parent bills as a post read")
    ap.add_argument("--with-users", action="store_true",
                    help="expand author_id — bills $0.010 per user, usually not worth it")
    ap.add_argument("--no-bq", action="store_true")
    ap.add_argument("--replay", help="load an existing raw dir, no API calls")
    ap.add_argument("--yes", action="store_true", help="skip cost confirmation")
    args = ap.parse_args()

    # Abort before spending, not after. An abort costs nothing; a
    # mis-targeted load mixes two corpora with different sampling designs.
    if not args.no_bq and not args.dataset:
        sys.exit("No dataset. Set BQ_DATASET in .env or pass --dataset. "
                 "Use --no-bq to collect without loading.")

    if args.query and args.from_config:
        sys.exit("Use --query or --from-config, not both.")

    qs = None
    if args.from_config:
        from queries_loader import QuerySet
        qs = QuerySet.load(args.from_config)

    if args.replay:
        raw_dir = Path(args.replay)
        # rglob, not glob: --from-config runs nest pages under q01/, q02/...
        pages = [json.loads(p.read_text(encoding="utf-8"))
                 for p in sorted(raw_dir.rglob("page_*.json"))]
        print(f"replaying {len(pages)} pages from {raw_dir} (no API calls)")
    else:
        if args.from_config:
            queries = qs.queries(args.stream, args.locale)
            if not queries:
                sys.exit(f"No queries for {args.stream}/{args.locale} in "
                         f"{args.from_config}")
        elif args.query:
            queries = [args.query]
        else:
            sys.exit("--query or --from-config is required unless --replay is used")

        token = os.environ.get("X_BEARER_TOKEN")
        if not token:
            sys.exit("Set X_BEARER_TOKEN first.")

        # --limit is a budget for the WHOLE stream, not per query. A stream can
        # compile to several queries (C is 7), and a per-query cap would
        # multiply the spend by that count without saying so.
        worst = args.limit * POST_READ_USD
        if args.with_users:
            worst += args.limit * USER_READ_USD
        if args.with_parents:
            worst += args.limit * POST_READ_USD

        print(f"stream : {args.stream}/{args.locale}")
        print(f"queries: {len(queries)}")
        for i, q in enumerate(queries, 1):
            print(f"  [{i}] {q}")
        if qs is not None:
            print(f"  {qs.summary(args.stream, args.locale)}")
        print(f"limit  : {args.limit} posts total across all queries")
        print(f"max cost: ${worst:.2f}")
        if not args.yes and worst > 1.0:
            if input("proceed? [y/N] ").strip().lower() != "y":
                sys.exit("aborted")
        print()

        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        raw_dir = Path("raw") / run_id
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Manifest first, so an interrupted run is still auditable. The exact
        # query list is part of the sampling design and belongs with the data.
        (raw_dir / "manifest.json").write_text(json.dumps({
            "run_id": run_id,
            "stream": args.stream,
            "locale": args.locale,
            "round": args.round_no,
            "limit": args.limit,
            "source": args.from_config or "--query",
            "config_generated": getattr(qs, "generated", None),
            "config_version": getattr(qs, "config_version", None),
            "queries": queries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        # Adaptive fair share: unused budget rolls forward to later queries
        # instead of the first query consuming the whole allowance.
        pages = []
        budget = args.limit
        for i, q in enumerate(queries):
            if budget <= 0:
                print(f"budget exhausted, skipping {len(queries) - i} query(ies)")
                break
            share = min(budget, max(10, budget // (len(queries) - i)))
            print(f"\n[query {i+1}/{len(queries)}] share {share} of {budget} left")
            print(f"  {q}")
            got_pages = collect(q, share, token, args.page_size,
                                args.with_parents, args.with_users,
                                raw_dir / f"q{i+1:02d}")
            budget -= sum(len(p.get("data", [])) for p in got_pages)
            pages += got_pages

        print(f"\nraw pages kept in {raw_dir}")

    tweets, media, users = flatten(pages, args.category, args.subcategory,
                                   args.stream, args.round_no)

    # Terms the X API query budget could not hold. Without this drop, stream D
    # contains posts the config claims it excludes and the prevalence anchor
    # is wrong in a direction that is easy to miss later.
    dropped = []
    if qs is not None:
        tweets, dropped = qs.apply_exclusions(tweets, args.stream, args.locale)
        if dropped:
            kept_keys = {k for t in tweets for k in t["media_keys"]}
            kept_authors = {t["author_id"] for t in tweets}
            media = [m for m in media if m["media_key"] in kept_keys]
            users = [u for u in users if u["id"] in kept_authors]
            (raw_dir / "excluded.json").write_text(
                json.dumps(dropped, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8")

    report(tweets, media, users, pages, dropped)

    if args.no_bq:
        print("\n--no-bq, nothing written. Replay later with:")
        print(f"  python collect.py --replay {raw_dir} --stream {args.stream}"
              + (f" --from-config {args.from_config} --locale {args.locale}"
                 if args.from_config else "")
              + f" --round {args.round_no}")
        return

    load(args.dataset, tweets, media, users)


if __name__ == "__main__":
    main()
