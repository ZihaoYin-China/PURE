#!/usr/bin/env python
"""Build fixed-routing baseline files for end-to-end comparison experiments.

The generated files keep the original query rows but overwrite ``retrieval``.
They can be evaluated with ``eval/eval.py`` through ``script/4_eval.sh`` or
``script/23_eval_large_baseline_all.sh`` by pointing ``ROUTE_DIR`` at one of
the generated policy directories.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable, List


LABELS = ["no", "paragraph", "document", "image"]
DEFAULT_TARGETS = ["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"]
DEFAULT_POLICIES = ["no", "paragraph", "document", "image", "oracle"]


def parse_list(text: str, defaults: Iterable[str]) -> List[str]:
    if not text:
        return list(defaults)
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def probs_for(label: str) -> List[float]:
    probs = [0.0] * len(LABELS)
    if label in LABELS:
        probs[LABELS.index(label)] = 1.0
    return probs


def policy_retrieval(row: Dict, policy: str) -> str:
    policy = str(policy).strip().lower()
    if policy == "oracle":
        label = str(row.get("gt_retrieval", "no")).strip().lower()
        return label if label in LABELS else "no"
    if policy in LABELS:
        return policy
    raise ValueError(f"Unsupported policy: {policy}")


def build_policy_rows(rows: List[Dict], policy: str) -> List[Dict]:
    output = []
    for row in rows:
        new_row = dict(row)
        original = str(new_row.get("retrieval", "")).strip().lower()
        label = policy_retrieval(new_row, policy)
        new_row["retrieval_original"] = original
        new_row["retrieval"] = label
        new_row["retrieval_conf"] = 1.0
        new_row["retrieval_probs"] = probs_for(label)
        new_row["retrieval_probs_order"] = LABELS
        new_row["retrieval_probs_source"] = f"fixed_{policy}"
        new_row["fixed_route_policy"] = policy
        output.append(new_row)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source_route_dir",
        required=True,
        help="Existing route directory, organized as source_route_dir/router/target.json.",
    )
    parser.add_argument("--router_model", required=True)
    parser.add_argument("--output_root", default="route/fixed_route_baselines")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGETS))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    args = parser.parse_args()

    targets = parse_list(args.targets, DEFAULT_TARGETS)
    policies = parse_list(args.policies, DEFAULT_POLICIES)

    summary = []
    for target in targets:
        input_file = os.path.join(args.source_route_dir, args.router_model, f"{target}.json")
        if not os.path.isfile(input_file):
            raise FileNotFoundError(f"Missing source route file: {input_file}")
        rows = load_json(input_file)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a list of rows: {input_file}")

        for policy in policies:
            policy = policy.strip().lower()
            out_rows = build_policy_rows(rows, policy)
            out_file = os.path.join(args.output_root, policy, args.router_model, f"{target}.json")
            save_json(out_file, out_rows)
            summary.append((policy, target, len(out_rows), out_file))

    print("policy\ttarget\trows\toutput")
    for policy, target, n_rows, out_file in summary:
        print(f"{policy}\t{target}\t{n_rows}\t{out_file}")


if __name__ == "__main__":
    main()
