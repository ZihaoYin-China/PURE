import argparse
import json
import os
from typing import Dict, List

try:
    from tabulate import tabulate
except ImportError:  # pragma: no cover
    tabulate = None


MODALITIES = ["no", "paragraph", "document", "image"]


def _parse_targets(text: str) -> List[str]:
    parts = [x.strip() for x in str(text or "").replace(" ", ",").split(",")]
    return [x for x in parts if x]


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _validate_alignment(base_rows, donor_rows, target: str):
    if len(base_rows) != len(donor_rows):
        raise ValueError(
            f"Target {target}: row count mismatch base={len(base_rows)} donor={len(donor_rows)}"
        )
    for idx, (base_row, donor_row) in enumerate(zip(base_rows, donor_rows)):
        base_index = base_row.get("index")
        donor_index = donor_row.get("index")
        if base_index != donor_index:
            raise ValueError(
                f"Target {target}: index mismatch at row {idx}: "
                f"base={base_index!r}, donor={donor_index!r}"
            )


def _attach_probs_from_donor(base_row: Dict, donor_row: Dict, source: str) -> Dict:
    merged = dict(base_row)
    probs = donor_row.get("retrieval_probs")
    if not isinstance(probs, list) or len(probs) != len(MODALITIES):
        raise ValueError(
            f"Invalid donor retrieval_probs for index={base_row.get('index')}: {probs}"
        )
    probs = [float(x) for x in probs]
    total = sum(max(0.0, x) for x in probs)
    if total <= 0:
        raise ValueError(f"Non-positive donor probability sum for index={base_row.get('index')}")
    probs = [max(0.0, x) / total for x in probs]

    ranked = sorted(probs, reverse=True)
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))

    merged["retrieval_probs"] = probs
    merged["retrieval_probs_order"] = donor_row.get("retrieval_probs_order", MODALITIES)
    merged["retrieval_probs_source"] = source
    merged["retrieval_probs_pred"] = MODALITIES[pred_idx]
    merged["retrieval_probs_conf"] = float(probs[pred_idx])
    merged["retrieval_probs_margin"] = (
        float(ranked[0] - ranked[1]) if len(ranked) > 1 else float(ranked[0])
    )
    merged["retrieval_probs_match_original"] = (
        str(base_row.get("retrieval", "")).strip().lower() == MODALITIES[pred_idx]
    )
    return merged


def build_hybrid_probs(
    *,
    base_prob_dir: str,
    base_router: str,
    donor_router: str,
    output_dir: str,
    all_targets: List[str],
    donor_targets: List[str],
):
    donor_target_set = {str(x).strip().lower() for x in donor_targets}
    base_router_dir = os.path.join(base_prob_dir, base_router)
    donor_router_dir = os.path.join(base_prob_dir, donor_router)
    output_router_dir = os.path.join(output_dir, base_router)
    os.makedirs(output_router_dir, exist_ok=True)

    summary = []
    source_tag = f"hybrid_{donor_router}_probs_on_{base_router}"

    for target in all_targets:
        base_path = os.path.join(base_router_dir, f"{target}.json")
        if not os.path.isfile(base_path):
            raise FileNotFoundError(f"Base probability file not found: {base_path}")

        base_rows = _load_json(base_path)
        use_donor = target.lower() in donor_target_set

        if use_donor:
            donor_path = os.path.join(donor_router_dir, f"{target}.json")
            if not os.path.isfile(donor_path):
                raise FileNotFoundError(f"Donor probability file not found: {donor_path}")
            donor_rows = _load_json(donor_path)
            _validate_alignment(base_rows, donor_rows, target)
            output_rows = [
                _attach_probs_from_donor(base_row, donor_row, source=source_tag)
                for base_row, donor_row in zip(base_rows, donor_rows)
            ]
        else:
            output_rows = base_rows

        output_path = os.path.join(output_router_dir, f"{target}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_rows, f, indent=4, ensure_ascii=False)

        summary.append(
            {
                "target": target,
                "rows": len(output_rows),
                "mode": f"{base_router}+{donor_router}" if use_donor else base_router,
                "output": output_path,
            }
        )

    if tabulate is not None:
        print(tabulate(summary, headers="keys", tablefmt="fancy_grid"))
    else:
        for row in summary:
            print(
                f"{row['target']}: mode={row['mode']}, rows={row['rows']}, output={row['output']}"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Build a hybrid Bayes probability directory by swapping selected targets to donor router probabilities."
    )
    parser.add_argument("--base_prob_dir", type=str, required=True)
    parser.add_argument("--base_router", type=str, default="distilbert")
    parser.add_argument("--donor_router", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--targets",
        type=str,
        default="mmlu,squad,natural_questions,hotpotqa,webqa",
        help="All targets to emit into the output directory.",
    )
    parser.add_argument(
        "--donor_targets",
        type=str,
        required=True,
        help="Targets whose retrieval_probs should come from the donor router.",
    )
    args = parser.parse_args()

    build_hybrid_probs(
        base_prob_dir=args.base_prob_dir,
        base_router=args.base_router,
        donor_router=args.donor_router,
        output_dir=args.output_dir,
        all_targets=_parse_targets(args.targets),
        donor_targets=_parse_targets(args.donor_targets),
    )


if __name__ == "__main__":
    main()
