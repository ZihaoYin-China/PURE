import argparse
import faulthandler
import io
import json
import os
import shutil
import time
from pathlib import Path


def load_required_paths(query_file: Path) -> set[str]:
    with query_file.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    required = set()
    prefix = "dataset/visual_rag/images/"
    for row in rows:
        for key in ("candidate_images", "gt_images"):
            for path in row.get(key, []) or []:
                path = str(path)
                if path.startswith(prefix):
                    path = path[len(prefix):]
                required.add(path)
    return required


def find_example_path(example: dict) -> str | None:
    preferred = ("file_name", "filepath", "path", "file_id", "image_path")
    for key in preferred:
        value = example.get(key)
        if isinstance(value, str) and value.lower().endswith((".jpg", ".jpeg", ".png")):
            return value
    for value in example.values():
        if isinstance(value, str) and value.lower().endswith((".jpg", ".jpeg", ".png")):
            return value
    return None


def find_image_value(example: dict):
    preferred = ("image", "jpg", "jpeg", "png", "bytes")
    for key in preferred:
        if key in example:
            return example[key]
    for value in example.values():
        if hasattr(value, "save"):
            return value
        if isinstance(value, (bytes, bytearray)):
            return value
        if isinstance(value, dict) and (
            value.get("bytes") is not None or value.get("path")
        ):
            return value
    return None


def normalize_dataset_path(path: str) -> str:
    path = path.replace("\\", "/")
    for prefix in ("train/", "./train/"):
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


def write_raw_image(raw, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, (bytes, bytearray)):
        output_path.write_bytes(bytes(raw))
        return
    raise ValueError(f"Expected image bytes for {output_path}, got {type(raw).__name__}")


def save_image(example: dict, output_path: Path):
    image = find_image_value(example)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(image, "save"):
        image.save(output_path)
        return

    if isinstance(image, (bytes, bytearray)):
        output_path.write_bytes(bytes(image))
        return

    if isinstance(image, dict):
        if image.get("bytes") is not None:
            raw = image["bytes"]
            if isinstance(raw, (bytes, bytearray)):
                output_path.write_bytes(bytes(raw))
                return
            from PIL import Image

            img = Image.open(io.BytesIO(raw))
            img.save(output_path)
            return
        if image.get("path") and os.path.exists(image["path"]):
            shutil.copyfile(image["path"], output_path)
            return

    raise ValueError(
        "Could not save image from dataset row. Expected a PIL image or "
        "an image dict with bytes/path."
    )


def maybe_disable_image_decode(dataset, decode_images: bool):
    if decode_images:
        return dataset

    try:
        from datasets import Image
    except Exception as exc:
        print(f"[WARN] Could not import datasets.Image; image decoding remains enabled: {exc}", flush=True)
        return dataset

    features = getattr(dataset, "features", None) or {}
    for column in ("image", "jpg", "jpeg", "png"):
        if column not in features:
            continue
        try:
            dataset = dataset.cast_column(column, Image(decode=False))
            print(f"[INFO] Disabled decoding for image column: {column}", flush=True)
        except Exception as exc:
            print(f"[WARN] Could not disable decoding for column {column}: {exc}", flush=True)
    return dataset


def load_streaming_dataset(args, data_file: str | None = None, columns: list[str] | None = None):
    from datasets import load_dataset

    kwargs = {
        "split": args.split,
        "streaming": True,
        "columns": columns or ["file_name", "image"],
    }
    if data_file is not None:
        kwargs["data_files"] = {args.split: [data_file]}
    try:
        return load_dataset(args.dataset, **kwargs)
    except TypeError as exc:
        print(
            f"[WARN] Dataset loader did not accept column projection ({exc}); "
            "falling back to all columns.",
            flush=True,
        )
        kwargs.pop("columns", None)
        return load_dataset(args.dataset, **kwargs)


def find_matching_paths_in_shard(args, data_file: str, remaining: set[str]) -> set[str]:
    dataset = load_streaming_dataset(args, data_file=data_file, columns=["file_name"])
    matches = set()
    seen = 0
    for example in dataset:
        seen += 1
        if args.report_every > 0 and seen % args.report_every == 0:
            print(
                f"[SCAN_NAMES] shard_seen={seen}, matches={len(matches)}, remaining={len(remaining)}",
                flush=True,
            )
        dataset_path = find_example_path(example)
        if not dataset_path:
            continue
        rel_path = normalize_dataset_path(dataset_path)
        if rel_path in remaining:
            matches.add(rel_path)
    return matches


def shard_data_file(template: str, split: str, shard: int, num_shards: int) -> str:
    return template.format(split=split, shard=shard, num_shards=num_shards)


def checkpoint_matches(data: dict, args, required_count: int) -> bool:
    return (
        data.get("dataset") == args.dataset
        and data.get("split") == args.split
        and data.get("query_file") == str(args.query_file)
        and data.get("required_count") == required_count
        and data.get("num_shards") == args.num_shards
    )


def load_completed_shards(args, required_count: int) -> set[int]:
    if not args.checkpoint_file.is_file():
        return set()
    try:
        with args.checkpoint_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Could not read checkpoint {args.checkpoint_file}: {exc}", flush=True)
        return set()
    if not checkpoint_matches(data, args, required_count):
        print(f"[INFO] Ignoring checkpoint with different dataset/query: {args.checkpoint_file}", flush=True)
        return set()
    return {int(x) for x in data.get("completed_shards", [])}


def write_checkpoint(args, required_count: int, completed_shards: set[int], remaining_count: int):
    args.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "dataset": args.dataset,
        "split": args.split,
        "query_file": str(args.query_file),
        "required_count": required_count,
        "num_shards": args.num_shards,
        "completed_shards": sorted(completed_shards),
        "remaining": remaining_count,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with args.checkpoint_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def process_examples(
    dataset,
    args,
    remaining: set[str],
    required_count: int,
    existing_count: int,
    counters: dict,
    shard: int | None = None,
    shard_targets: set[str] | None = None,
):
    for example in dataset:
        counters["seen"] += 1
        counters["shard_seen"] += 1
        if args.report_every > 0 and counters["shard_seen"] % args.report_every == 0:
            shard_part = f"shard={shard:05d}, " if shard is not None else ""
            print(
                f"[SCAN] {shard_part}seen={counters['seen']}, "
                f"shard_seen={counters['shard_seen']}, saved={counters['saved']}, "
                f"remaining={len(remaining)}",
                flush=True,
            )
        dataset_path = find_example_path(example)
        if not dataset_path:
            continue
        rel_path = normalize_dataset_path(dataset_path)
        if rel_path not in remaining:
            continue
        if shard_targets is not None and rel_path not in shard_targets:
            continue

        save_image(example, args.output_dir / rel_path)
        remaining.remove(rel_path)
        counters["saved"] += 1
        print(f"[SAVE] {counters['saved']}/{required_count - existing_count} {rel_path}", flush=True)

        if not remaining:
            break


def download_shard_file(args, data_file: str, force_download: bool = False) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=args.dataset,
        repo_type="dataset",
        filename=data_file,
        force_download=force_download,
    )
    return Path(path)


def looks_like_parquet_corruption(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        token in text
        for token in (
            "snappy",
            "parquet",
            "thrift",
            "deserialize",
            "page header",
            "checksum",
        )
    )


def save_matching_images_from_local_parquet(
    args,
    data_file: str,
    shard_targets: set[str],
    remaining: set[str],
    required_count: int,
    existing_count: int,
    counters: dict,
    shard: int,
):
    last_exc = None
    for attempt in range(2):
        local_path = download_shard_file(args, data_file, force_download=attempt > 0)
        try:
            return save_matching_images_from_parquet_path(
                local_path,
                args,
                shard_targets,
                remaining,
                required_count,
                existing_count,
                counters,
                shard,
            )
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and looks_like_parquet_corruption(exc):
                print(
                    f"[WARN] Local parquet read failed for shard={shard:05d} ({type(exc).__name__}: {exc}); "
                    "forcing a fresh shard download.",
                    flush=True,
                )
                continue
            raise
    if last_exc is not None:
        raise last_exc


def save_matching_images_from_parquet_path(
    local_path: Path,
    args,
    shard_targets: set[str],
    remaining: set[str],
    required_count: int,
    existing_count: int,
    counters: dict,
    shard: int,
):
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(local_path)
    for row_group in range(parquet_file.num_row_groups):
        names_table = parquet_file.read_row_group(row_group, columns=["file_name"])
        names = names_table.column("file_name").to_pylist()
        row_group_targets = {
            normalize_dataset_path(name)
            for name in names
            if normalize_dataset_path(name) in shard_targets and normalize_dataset_path(name) in remaining
        }
        if not row_group_targets:
            continue

        table = parquet_file.read_row_group(row_group, columns=["file_name", "image"])
        file_names = table.column("file_name").to_pylist()
        images = table.column("image").to_pylist()
        for dataset_path, image_bytes in zip(file_names, images):
            counters["seen"] += 1
            counters["shard_seen"] += 1
            rel_path = normalize_dataset_path(dataset_path)
            if rel_path not in row_group_targets or rel_path not in remaining:
                continue
            write_raw_image(image_bytes, args.output_dir / rel_path)
            remaining.remove(rel_path)
            counters["saved"] += 1
            print(f"[SAVE] {counters['saved']}/{required_count - existing_count} {rel_path}", flush=True)

            if not remaining:
                break
        if not remaining:
            break


def process_whole_stream(args, remaining: set[str], required_count: int, existing_count: int) -> int:
    counters = {"seen": 0, "shard_seen": 0, "saved": 0}
    restarts = 0
    while remaining:
        counters["shard_seen"] = 0
        try:
            dataset = load_streaming_dataset(args)
            dataset = maybe_disable_image_decode(dataset, args.decode_images)
            process_examples(dataset, args, remaining, required_count, existing_count, counters)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            restarts += 1
            print(
                f"[WARN] Streaming download interrupted ({type(exc).__name__}: {exc}). "
                f"Restart {restarts}/{args.max_restarts}; remaining={len(remaining)}.",
                flush=True,
            )
            if restarts > args.max_restarts:
                print("[ERROR] Too many restarts; writing missing list and stopping.", flush=True)
                break
            time.sleep(args.restart_sleep)
    return counters["saved"]


def process_shards(args, remaining: set[str], required_count: int, existing_count: int) -> int:
    counters = {"seen": 0, "shard_seen": 0, "saved": 0}
    completed_shards = load_completed_shards(args, required_count)
    if completed_shards:
        print(f"[INFO] Loaded checkpoint with {len(completed_shards)} completed shards.", flush=True)

    restarts = 0
    for shard in range(args.start_shard, args.num_shards):
        if not remaining:
            break
        if shard in completed_shards:
            continue

        data_file = shard_data_file(args.data_file_template, args.split, shard, args.num_shards)
        while remaining:
            counters["shard_seen"] = 0
            try:
                print(f"[SHARD] {shard + 1}/{args.num_shards} {data_file}", flush=True)
                shard_targets = find_matching_paths_in_shard(args, data_file, remaining)
                if not shard_targets:
                    completed_shards.add(shard)
                    write_checkpoint(args, required_count, completed_shards, len(remaining))
                    print(
                        f"[SHARD_DONE] shard={shard:05d}, matches=0, saved={counters['saved']}, "
                        f"remaining={len(remaining)}",
                        flush=True,
                    )
                    break

                if args.stream_image_bytes:
                    print(
                        f"[SHARD_MATCH] shard={shard:05d}, matches={len(shard_targets)}; loading image bytes via datasets.",
                        flush=True,
                    )
                    dataset = load_streaming_dataset(args, data_file=data_file, columns=["file_name", "image"])
                    dataset = maybe_disable_image_decode(dataset, args.decode_images)
                    process_examples(
                        dataset,
                        args,
                        remaining,
                        required_count,
                        existing_count,
                        counters,
                        shard=shard,
                        shard_targets=shard_targets,
                    )
                else:
                    print(
                        f"[SHARD_MATCH] shard={shard:05d}, matches={len(shard_targets)}; downloading shard parquet locally.",
                        flush=True,
                    )
                    save_matching_images_from_local_parquet(
                        args,
                        data_file,
                        shard_targets,
                        remaining,
                        required_count,
                        existing_count,
                        counters,
                        shard,
                    )
                completed_shards.add(shard)
                write_checkpoint(args, required_count, completed_shards, len(remaining))
                print(
                    f"[SHARD_DONE] shard={shard:05d}, saved={counters['saved']}, "
                    f"remaining={len(remaining)}",
                    flush=True,
                )
                break
            except KeyboardInterrupt:
                write_checkpoint(args, required_count, completed_shards, len(remaining))
                raise
            except Exception as exc:
                restarts += 1
                print(
                    f"[WARN] Shard {shard:05d} interrupted ({type(exc).__name__}: {exc}). "
                    f"Restart {restarts}/{args.max_restarts}; remaining={len(remaining)}.",
                    flush=True,
                )
                if restarts > args.max_restarts:
                    print("[ERROR] Too many restarts; writing missing list and stopping.", flush=True)
                    return counters["saved"]
                time.sleep(args.restart_sleep)
    return counters["saved"]


def configure_hf_cache(args):
    if args.hf_cache_dir is None and any(
        os.environ.get(key)
        for key in ("HF_HOME", "HF_HUB_CACHE", "HF_DATASETS_CACHE")
    ):
        print("[INFO] Using existing Hugging Face cache environment.", flush=True)
        return

    cache_dir = args.hf_cache_dir or (args.output_dir.parent / "hf_cache")
    hf_home = cache_dir.expanduser().resolve()
    hub_cache = hf_home / "hub"
    datasets_cache = hf_home / "datasets"
    hub_cache.mkdir(parents=True, exist_ok=True)
    datasets_cache.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    print(f"[INFO] Using Hugging Face cache: {hf_home}", flush=True)


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser(
        description="Download only the iNat2021 images referenced by VisualRAG query json."
    )
    parser.add_argument("--query_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("dataset/visual_rag/images"))
    parser.add_argument("--dataset", default="MVRL/iNat-2021-train")
    parser.add_argument("--split", default="train")
    parser.add_argument("--report_every", type=int, default=5000)
    parser.add_argument("--missing_out", type=Path, default=Path("dataset/visual_rag/missing_images.txt"))
    parser.add_argument("--max_restarts", type=int, default=50)
    parser.add_argument("--restart_sleep", type=float, default=10.0)
    parser.add_argument("--num_shards", type=int, default=212)
    parser.add_argument("--start_shard", type=int, default=0)
    parser.add_argument("--data_file_template", default="data/{split}-{shard:05d}-of-{num_shards:05d}.parquet")
    parser.add_argument(
        "--checkpoint_file",
        type=Path,
        default=Path("dataset/visual_rag/download_visual_rag_images_from_hf.checkpoint.json"),
    )
    parser.add_argument(
        "--decode_images",
        action="store_true",
        help="Decode image rows with PIL before saving. Default is to save raw bytes/path to avoid native decoder crashes.",
    )
    parser.add_argument(
        "--enable_xet",
        action="store_true",
        help="Allow the hf-xet native downloader. By default this script disables it because it can crash in some environments.",
    )
    parser.add_argument(
        "--legacy_streaming",
        action="store_true",
        help="Use the old whole-dataset streaming loop instead of shard checkpoints.",
    )
    parser.add_argument(
        "--stream_image_bytes",
        action="store_true",
        help="Use datasets streaming for image bytes. Default downloads matching parquet shards locally and reads them with pyarrow.",
    )
    parser.add_argument(
        "--hf_cache_dir",
        type=Path,
        default=None,
        help=(
            "Directory for Hugging Face hub/datasets cache. Defaults to "
            "<output_dir>/../hf_cache unless HF cache environment variables are already set."
        ),
    )
    args = parser.parse_args()

    configure_hf_cache(args)

    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        print("[INFO] Set HF_HUB_DISABLE_XET=1 for a more stable pure-Python download path.", flush=True)

    required = load_required_paths(args.query_file)
    existing = {p for p in required if (args.output_dir / p).is_file()}
    remaining = set(required) - existing
    print(f"Required images: {len(required)}")
    print(f"Already present : {len(existing)}")
    print(f"To download     : {len(remaining)}")
    if not remaining:
        return

    if args.legacy_streaming:
        saved = process_whole_stream(args, remaining, len(required), len(existing))
    else:
        saved = process_shards(args, remaining, len(required), len(existing))

    args.missing_out.parent.mkdir(parents=True, exist_ok=True)
    with args.missing_out.open("w", encoding="utf-8") as f:
        for path in sorted(remaining):
            f.write(path + "\n")
    print(f"Done. saved={saved}, missing={len(remaining)}")
    if remaining:
        print(f"Missing list written to: {args.missing_out}")


if __name__ == "__main__":
    main()
