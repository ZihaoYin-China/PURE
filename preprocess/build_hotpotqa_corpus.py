"""
Build the HotpotQA document corpus by downloading Wikipedia articles for the
gold-standard page IDs referenced in the query annotations.

This fixes the corpus mismatch where gt_texts reference original Wikipedia
page IDs but the retrieval corpus only has LongRAG documents with sequential IDs.

Downloads Wikipedia pages in batched API calls with proper rate limiting.
"""

import json
import os
import re
import sys
import time
import urllib.request
import urllib.parse
import urllib.error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QUERY_PATHS = [
    os.path.join(BASE_DIR, "dataset/query/hotpotqa.json"),
]
_extra = [
    "dataset/query_nonvideo_split/test/hotpotqa.json",
    "dataset/query_nonvideo_split/train/hotpotqa.json",
    "dataset/query_nonvideo_split/full/hotpotqa.json",
    "dataset/query_nonvideo_large_strict_d40/test/hotpotqa.json",
    "dataset/query_nonvideo_large_strict_d40/dev/hotpotqa.json",
    "dataset/query_nonvideo_large_strict_d40/train_fit/hotpotqa.json",
]
for p in _extra:
    full = os.path.join(BASE_DIR, p)
    if os.path.exists(full):
        QUERY_PATHS.append(full)

OUTPUT_DIR = os.path.join(BASE_DIR, "dataset/hotpotqa/text")
API_URL = "https://en.wikipedia.org/w/api.php"
BATCH_SIZE = 50
REQUEST_DELAY = 8.0  # seconds; Wikipedia allows ~5000 req/hr anonymously
UA = "PURE-CorpusBuilder/1.0 (research bot; contact via repo README)"


def extract_page_ids():
    page_ids = set()
    for qpath in QUERY_PATHS:
        if not os.path.exists(qpath):
            continue
        with open(qpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            for gt_path in row.get("gt_texts", []):
                filename = os.path.basename(gt_path)
                page_id_str = os.path.splitext(filename)[0]
                if page_id_str.isdigit():
                    page_ids.add(int(page_id_str))
    return sorted(page_ids)


def strip_wikitext(text):
    """Basic wikitext → plain text conversion."""
    # Remove templates {{...}}
    text = re.sub(r'\{\{[^}]*\}\}', '', text)
    # Remove refs <ref>...</ref>
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*?/>', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove wikitext formatting
    text = re.sub(r"''+", '', text)  # bold/italic
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)  # [[link|text]]
    text = re.sub(r'\[\[([^\]]+)\]\]', r'\1', text)  # [[link]]
    text = re.sub(r'\[https?://[^\]]+\]', '', text)  # [url]
    # Remove headings markup
    text = re.sub(r'=+([^=]+)=+', r'\1', text)
    # Remove lists
    text = re.sub(r'^[\*#:;]+\s*', '', text, flags=re.MULTILINE)
    return text.strip()


def download_pages_batch(page_ids_batch, use_extracts=True):
    """Download a batch of Wikipedia pages. Tries extracts first, falls back to wikitext."""
    if use_extracts:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": "1",
            "exsectionformat": "plain",
            "exlimit": "max",
            "maxlag": "5",
            "pageids": "|".join(str(pid) for pid in page_ids_batch),
        }
    else:
        params = {
            "action": "query",
            "format": "json",
            "prop": "revisions",
            "rvprop": "content",
            "rvslots": "main",
            "maxlag": "5",
            "pageids": "|".join(str(pid) for pid in page_ids_batch),
        }

    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA})

    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 30 * (2 ** attempt)
                print(f"    HTTP 429 (rate limit), waiting {wait}s...")
                time.sleep(wait)
            elif attempt == 3:
                raise
            else:
                time.sleep(5 * (2 ** attempt))
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(5 * (2 ** attempt))

    pages = data.get("query", {}).get("pages", {})
    results = {}
    for _pid, page_info in pages.items():
        pid = int(_pid)
        if pid < 0:
            continue
        title = page_info.get("title", "")

        if use_extracts:
            text = page_info.get("extract", "")
        else:
            revisions = page_info.get("revisions", [])
            if revisions:
                slots = revisions[0].get("slots", {})
                main_slot = slots.get("main", {})
                text = main_slot.get("*", "")
            else:
                text = ""

        if text:
            if not use_extracts:
                text = strip_wikitext(text)
            results[pid] = {"title": title, "text": text}
    return results


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_page_ids = extract_page_ids()
    print(f"Found {len(all_page_ids)} unique Wikipedia page IDs from query files")

    existing = set()
    for pid in all_page_ids:
        path = os.path.join(OUTPUT_DIR, f"{pid}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            if len(content) > 200:
                existing.add(pid)

    missing_ids = [pid for pid in all_page_ids if pid not in existing]
    print(f"Already have: {len(existing)}, need to download: {len(missing_ids)}")

    if not missing_ids:
        print("All pages already exist.")
        return

    # Phase 1: Download via extracts API
    batches = [
        missing_ids[i : i + BATCH_SIZE]
        for i in range(0, len(missing_ids), BATCH_SIZE)
    ]
    print(f"Phase 1: downloading {len(missing_ids)} pages in {len(batches)} batches "
          f"(extracts API, {REQUEST_DELAY}s delay)...")

    downloaded = {}
    failed = []
    for batch_idx, batch in enumerate(batches):
        try:
            results = download_pages_batch(batch, use_extracts=True)
        except Exception as e:
            print(f"  Batch {batch_idx + 1}/{len(batches)} ERROR: {e}")
            failed.extend(batch)
            continue

        for pid, info in results.items():
            path = os.path.join(OUTPUT_DIR, f"{pid}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"{info['title']}\n\n{info['text']}")
            downloaded[pid] = info

        missing_in_batch = set(batch) - set(results.keys())
        failed.extend(missing_in_batch)

        if (batch_idx + 1) % 5 == 0:
            print(f"  Progress: {batch_idx + 1}/{len(batches)} batches, "
                  f"{len(downloaded)} saved, {len(failed)} pending/failed")

        time.sleep(REQUEST_DELAY)

    # Phase 2: Retry failed with wikitext API
    retry_ids = sorted(set(failed))
    if retry_ids:
        print(f"\nPhase 2: retrying {len(retry_ids)} failed/missing pages "
              f"(wikitext API)...")
        retry_downloaded = {}
        still_failed = []
        batches2 = [
            retry_ids[i : i + BATCH_SIZE]
            for i in range(0, len(retry_ids), BATCH_SIZE)
        ]
        for batch_idx, batch in enumerate(batches2):
            try:
                results = download_pages_batch(batch, use_extracts=False)
            except Exception as e:
                print(f"  Batch {batch_idx + 1}/{len(batches2)} ERROR: {e}")
                still_failed.extend(batch)
                continue

            for pid, info in results.items():
                path = os.path.join(OUTPUT_DIR, f"{pid}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(f"{info['title']}\n\n{info['text']}")
                retry_downloaded[pid] = info

            still_missing = set(batch) - set(results.keys())
            still_failed.extend(still_missing)

            if (batch_idx + 1) % 5 == 0:
                print(f"  Progress: {batch_idx + 1}/{len(batches2)} batches, "
                      f"{len(retry_downloaded)} saved, "
                      f"{len(still_failed)} still failed")

            time.sleep(REQUEST_DELAY)

        downloaded.update(retry_downloaded)
        failed = still_failed

    # Report
    print(f"\n{'='*60}")
    print(f"Download complete.")
    print(f"  Phase 1 (extracts):    saved most pages")
    print(f"  Phase 2 (wikitext):    recovered some failed pages")
    print(f"  Total saved:           {len(downloaded)}")
    print(f"  Still failed:          {len(failed)}")
    if failed:
        print(f"  Sample failed IDs:     {failed[:10]}")

    all_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".txt")]
    print(f"  Total corpus files:    {len(all_files)}")

    # Verify gt_texts
    corpus_ids = set()
    for fname in all_files:
        corpus_ids.add(os.path.join("dataset/hotpotqa/text", fname))

    total_gt = 0
    matched_gt = 0
    for qpath in QUERY_PATHS:
        if not os.path.exists(qpath):
            continue
        with open(qpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for row in data:
            for gt_path in row.get("gt_texts", []):
                total_gt += 1
                if gt_path in corpus_ids:
                    matched_gt += 1

    pct = 100 * matched_gt / total_gt if total_gt > 0 else 0
    print(f"\nGT text verification: {matched_gt}/{total_gt} ({pct:.1f}%)")

    # For pages that truly don't exist anymore, note them
    if failed:
        print(f"\nNote: {len(failed)} Wikipedia page IDs could not be downloaded.")
        print(f"These pages may have been deleted since the 2017 HotpotQA dump.")
        print(f"Their documents will not be in the retrieval corpus.")


if __name__ == "__main__":
    main()
