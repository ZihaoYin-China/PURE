import argparse
import pickle
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Merge pickle shards into one dict pickle.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    merged = {}
    for raw_path in args.inputs:
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input pickle not found: {path}")
        with path.open("rb") as f:
            shard = pickle.load(f)
        if not isinstance(shard, dict):
            raise ValueError(f"Expected dict pickle, got {type(shard)} from {path}")
        overlap = set(merged).intersection(shard)
        if overlap:
            raise ValueError(f"Found {len(overlap)} duplicated keys while merging {path}")
        merged.update(shard)
        print(f"Loaded {path}: {len(shard)} entries")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(merged, f)
    print(f"Saved {output_path}: {len(merged)} entries")


if __name__ == "__main__":
    main()
