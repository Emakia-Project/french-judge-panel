#!/usr/bin/env python3
"""
download X media at capture time and archive to GCS.

The failure this exists to prevent: X media URLs expire. The previous
evaluation stored links, the links died, and Gemini Vision abstained on
100% of the media stream. Bytes are the archive; URLs are not.

Reads media rows from the raw collector pages (no BigQuery needed) or from
the media table, downloads each object, uploads to GCS keyed by media_key,
and writes a manifest so gcs_uri can be joined back.

Cost note: media downloads are NOT billed by X. GCS storage runs about
$0.02/GB/month, so the full 12k-post collection is well under a dollar.

Usage:
    export MEDIA_STORE="gs://YOUR-BUCKET/media/fr/"

    # from raw pages, no BigQuery
    python download_media.py --raw-dir raw/

    # one run only
    python download_media.py --raw-dir raw/20260825T1430Z

    # local only, no upload (inspect first)
    python download_media.py --raw-dir raw/ --local-only

    # then push the manifest into BigQuery
    python download_media.py --raw-dir raw/ --load-manifest
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests


def pick_url(m):
    """
    Photos carry url. Video and animated_gif carry variants.

    bit_rate 0 is a legitimate value meaning animated_gif, so test
    IS NOT NONE, never truthiness. Testing truthiness silently drops
    every GIF in the corpus.
    """
    if m.get("url"):
        return m["url"], "url"

    variants = [v for v in m.get("variants", []) if v.get("bit_rate") is not None]
    if variants:
        best = max(variants, key=lambda v: v["bit_rate"])
        return best["url"], "variant"

    if m.get("preview_image_url"):
        return m["preview_image_url"], "preview"

    return None, None


def extension_for(media_type, content_type):
    if content_type:
        if "jpeg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
        if "mp4" in content_type:
            return ".mp4"
        if "gif" in content_type:
            return ".gif"
    return {"photo": ".jpg", "video": ".mp4", "animated_gif": ".mp4"}.get(
        media_type, ".bin")


def collect_media_rows(raw_dir):
    """Walk raw/*/page_*.json and pull includes.media, deduped by media_key."""
    seen, rows = set(), []
    pages = sorted(Path(raw_dir).rglob("page_*.json"))
    if not pages:
        sys.exit(f"No page_*.json under {raw_dir}")

    for p in pages:
        payload = json.loads(p.read_text(encoding="utf-8"))
        for m in payload.get("includes", {}).get("media", []):
            key = m.get("media_key")
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(m)

    print(f"{len(pages)} raw pages -> {len(rows)} unique media objects")
    return rows


EXPIRED_CODES = (403, 404, 410)


def download(url, timeout=120, retries=3, max_bytes=None):
    """
    Returns (data, content_type, kind, why).

    kind is None on success, "expired" when the CDN says the object is gone,
    "toolarge" when it exceeds max_bytes, "transient" otherwise.

    The distinction matters. Expired is unrecoverable and is the loss this
    script exists to catch. Transient is a dropped connection and must not be
    reported as data loss, or the operator investigates the wrong thing during
    the only window in which the bytes are still there.
    """
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=timeout,
                             headers={"User-Agent": "research-collector/1.0"})
        except requests.RequestException as e:
            last = f"network: {str(e)[:100]}"
            time.sleep(2 ** attempt)
            continue

        if r.status_code in EXPIRED_CODES:
            r.close()
            return None, None, "expired", f"http {r.status_code}"

        if r.status_code != 200:
            last = f"http {r.status_code}"
            r.close()
            time.sleep(2 ** attempt)
            continue

        # Check the declared size BEFORE pulling the body, so an oversized
        # object is not fetched in full and then discarded.
        declared = r.headers.get("content-length")
        if max_bytes and declared and int(declared) > max_bytes:
            r.close()
            return None, None, "toolarge", f"content-length {int(declared):,}"

        content_type = r.headers.get("content-type", "")
        try:
            # With stream=True the body is pulled HERE, not at get(). This read
            # must be inside the retry loop: a mid-transfer reset raises here.
            data = r.content
        except (requests.RequestException, OSError) as e:
            last = f"body read: {str(e)[:100]}"
            try:
                r.close()
            except Exception:
                pass
            time.sleep(2 ** attempt)
            continue
        r.close()

        # A truncated body is worse than none: it archives corrupt bytes that
        # look valid until something tries to decode them.
        if declared and len(data) != int(declared):
            last = f"truncated: got {len(data):,} of {int(declared):,}"
            time.sleep(2 ** attempt)
            continue

        return data, content_type, None, None

    return None, None, "transient", last or "gave up"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="raw",
                    help="directory of collector raw pages")
    ap.add_argument("--out-dir", default="media_local",
                    help="local staging directory")
    ap.add_argument("--bucket", default=os.environ.get("MEDIA_STORE"),
                    help="gs://bucket/prefix/ . Defaults to $MEDIA_STORE")
    ap.add_argument("--local-only", action="store_true",
                    help="download but do not upload")
    ap.add_argument("--load-manifest", action="store_true",
                    help="write manifest rows to BigQuery media_archive")
    ap.add_argument("--dataset", default=os.environ.get("BQ_DATASET"))
    ap.add_argument("--max-bytes", type=int, default=100 * 1024 * 1024,
                    help="skip objects larger than this")
    args = ap.parse_args()

    if not args.local_only and not args.bucket:
        sys.exit("Set MEDIA_STORE or pass --bucket, or use --local-only.")

    rows = collect_media_rows(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    bucket = client = None
    if not args.local_only:
        from google.cloud import storage
        assert args.bucket.startswith("gs://"), "--bucket must start with gs://"
        path = args.bucket[5:].rstrip("/")
        bucket_name, _, prefix = path.partition("/")
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        print(f"uploading to gs://{bucket_name}/{prefix}/")

    manifest, total_bytes = [], 0
    failed, expired, skipped, recovered = [], [], [], []

    # Resume. Bytes from an interrupted run are already on disk; the manifest
    # may not have caught up, because the original wrote it only after the
    # whole loop finished. Index from both, prefer the manifest.
    done, on_disk = set(), {}
    for f in out.glob("*"):
        if f.is_file() and f.name != "manifest.jsonl":
            on_disk.setdefault(f.stem, f)
    if on_disk:
        print(f"{len(on_disk)} files already on disk in {out}")

    mpath = out / "manifest.jsonl"
    if mpath.exists():
        # Last line wins: a key re-processed later (e.g. downloaded local-only,
        # then uploaded) appends an updated row rather than mutating the file.
        prior = {}
        for line in mpath.read_text(encoding="utf-8").splitlines():
            try:
                prev = json.loads(line)
            except Exception:
                continue
            if prev.get("media_key"):
                prior[prev["media_key"]] = prev

        needs_upload = 0
        for k, prev in prior.items():
            # A row archived under --local-only has no gcs_uri. If a bucket is
            # configured now, that row is NOT done: the bytes are on disk but
            # not in GCS, and skipping it would silently upload nothing.
            if bucket is not None and not prev.get("gcs_uri"):
                needs_upload += 1
                continue
            done.add(k)
            manifest.append(prev)
            total_bytes += prev.get("bytes", 0)

        print(f"{len(prior)} in manifest, {len(done)} complete")
        if needs_upload:
            print(f"{needs_upload} downloaded but not yet in GCS - uploading now")

    ctype_for_ext = {".jpg": "image/jpeg", ".png": "image/png",
                     ".mp4": "video/mp4", ".gif": "image/gif"}

    # Append and flush per item. Writing the manifest only at the end means a
    # crash discards the index for every success before it.
    mf = open(mpath, "a", encoding="utf-8")

    for i, m in enumerate(rows, 1):
        key = m["media_key"]
        if key in done:
            continue

        url, source = pick_url(m)
        local_hit = on_disk.get(key)
        data = content_type = kind = why = None

        if local_hit is not None:
            data = local_hit.read_bytes()
            content_type = ctype_for_ext.get(local_hit.suffix, "")
            recovered.append(key)
        elif not url:
            failed.append((key, "no usable url"))
            print(f"[{i}/{len(rows)}] {key} NO URL")
            continue
        else:
            data, content_type, kind, why = download(url, max_bytes=args.max_bytes)

        if kind == "expired":
            expired.append((key, why))
            print(f"[{i}/{len(rows)}] {key} EXPIRED {why}")
            continue
        if kind == "toolarge":
            skipped.append((key, why))
            print(f"[{i}/{len(rows)}] {key} SKIP {why}")
            continue
        if kind:
            failed.append((key, why))
            print(f"[{i}/{len(rows)}] {key} FAILED {why}")
            continue

        ext = extension_for(m.get("type"), content_type)
        fname = f"{key}{ext}"
        local = out / fname
        if local_hit is None:
            local.write_bytes(data)
        total_bytes += len(data)

        gcs_uri = None
        if bucket is not None:
            _, _, prefix = args.bucket[5:].rstrip("/").partition("/")
            blob_name = f"{prefix}/{fname}" if prefix else fname
            blob = bucket.blob(blob_name)
            if not blob.exists():
                blob.upload_from_filename(str(local),
                                          content_type=content_type or None)
            gcs_uri = f"gs://{bucket.name}/{blob_name}"

        row = {
            "media_key": key,
            "type": m.get("type"),
            "gcs_uri": gcs_uri,
            "local_path": str(local),
            "bytes": len(data),
            "content_type": content_type,
            "sha256": hashlib.sha256(data).hexdigest(),
            "source_field": source,
            "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        }
        manifest.append(row)
        mf.write(json.dumps(row, ensure_ascii=False) + "\n")
        mf.flush()
        done.add(key)

        tag = "recovered" if local_hit is not None else (m.get("type") or "?")
        print(f"[{i}/{len(rows)}] {tag:12} {len(data):>10,} B  {fname}")

    mf.close()

    print(f"\n{'='*60}")
    print(f"archived   : {len(manifest)}/{len(rows)}")
    if recovered:
        print(f"recovered  : {len(recovered)} from disk, no re-download")
    print(f"total bytes: {total_bytes/1e6:,.1f} MB")
    print(f"est. GCS   : ${total_bytes/1e9 * 0.02:.4f}/month")
    print(f"manifest   : {mpath}")

    if expired:
        print(f"\nEXPIRED: {len(expired)} - unrecoverable, the bytes are gone")
        for k, why in expired[:10]:
            print(f"  {k}  {why}")
        print("This is the loss the script exists to prevent. A high count means")
        print("media is being archived too long after collection.")
    if failed:
        print(f"\ntransient failures: {len(failed)} - rerun to pick them up")
        for k, why in failed[:10]:
            print(f"  {k}  {why}")
    if skipped:
        print(f"\nover --max-bytes: {len(skipped)} - raise the flag if you need them")
        for k, why in skipped[:5]:
            print(f"  {k}  {why}")

    if args.load_manifest:
        if not args.dataset:
            sys.exit("--load-manifest needs BQ_DATASET or --dataset")
        from google.cloud import bigquery
        bq = bigquery.Client()
        ref = f"{bq.project}.{args.dataset}.media_archive"
        errors = bq.insert_rows_json(
            ref, manifest, row_ids=[r["media_key"] for r in manifest])
        if errors:
            print(json.dumps(errors[:3], indent=2)[:1500])
            sys.exit("manifest insert failed")
        print(f"\n{len(manifest)} -> {ref}")


if __name__ == "__main__":
    main()
