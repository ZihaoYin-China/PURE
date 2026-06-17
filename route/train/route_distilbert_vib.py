import argparse
import json
import os

import torch
from tqdm import tqdm

from route.train.vib_prototype_router import DEFAULT_LABELS, load_router_checkpoint, load_router_tokenizer, resolve_device

TEXT_EQUIV_TARGETS = {"squad", "nq", "natural_questions"}
TEXT_ACTIONS = {"paragraph", "document"}


def _target_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]


def _is_text_equiv_target(target):
    return str(target or "").strip().lower() in TEXT_EQUIV_TARGETS


def _route_match(target, pred, gold):
    pred = str(pred or "").strip().lower()
    gold = str(gold or "").strip().lower()
    if _is_text_equiv_target(target) and pred in TEXT_ACTIONS and gold in TEXT_ACTIONS:
        return True
    return pred == gold



def _move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _parse_target_list(raw_value: str):
    if not raw_value:
        return set()
    return {
        item.strip()
        for item in str(raw_value).split(",")
        if item.strip()
    }


def main():
    parser = argparse.ArgumentParser(description="Route queries with isolated VIB+Prototype+Evidential router.")
    parser.add_argument("--checkpoint_dir", type=str, default="route/train/checkpoints/distilbert_vib")
    parser.add_argument("--input_dir", type=str, default="dataset/query")
    parser.add_argument("--output_dir", type=str, default="route/results_vib")
    parser.add_argument("--router_name", type=str, default="distilbert")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--include_targets", type=str, default="")
    parser.add_argument("--exclude_targets", type=str, default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    tokenizer = load_router_tokenizer(args.checkpoint_dir)
    model = load_router_checkpoint(args.checkpoint_dir, device=device)
    include_targets = _parse_target_list(args.include_targets)
    exclude_targets = _parse_target_list(args.exclude_targets)

    if os.path.isfile(args.input_dir):
        input_paths = [args.input_dir]
    else:
        input_paths = [
            os.path.join(args.input_dir, fname)
            for fname in sorted(os.listdir(args.input_dir))
            if fname.endswith(".json")
        ]
        filtered_paths = []
        for path in input_paths:
            target_name = os.path.splitext(os.path.basename(path))[0]
            if include_targets and target_name not in include_targets:
                continue
            if target_name in exclude_targets:
                continue
            filtered_paths.append(path)
        input_paths = filtered_paths

    if not input_paths:
        raise ValueError(
            f"No input json files selected from {args.input_dir!r}. "
            f"include_targets={sorted(include_targets)}, exclude_targets={sorted(exclude_targets)}"
        )

    output_root = os.path.join(args.output_dir, args.router_name)
    os.makedirs(output_root, exist_ok=True)

    overall_results = []
    for path in input_paths:
        data = _read_json(path)
        questions = [str(item.get("question", "")).strip() for item in data]

        counts = {label: 0 for label in DEFAULT_LABELS}
        total_conf = 0.0
        total_uncertainty = 0.0
        correct_4class = 0
        correct_text_equiv = 0
        target = _target_from_path(path)

        for start in tqdm(
            range(0, len(data), args.batch_size),
            desc=f"Routing {os.path.basename(path)} with VIB router",
        ):
            batch_questions = questions[start : start + args.batch_size]
            batch = tokenizer(
                batch_questions,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )
            batch = _move_batch_to_device(batch, device)

            outputs = model.predict_batch(**batch)
            probs = outputs["probs"].detach().cpu().tolist()
            pred_ids = outputs["pred_ids"].detach().cpu().tolist()
            confs = outputs["conf"].detach().cpu().tolist()
            uncertainty = outputs["uncertainty"].detach().cpu().tolist()
            logits = outputs["logits"].detach().cpu().tolist()
            alphas = outputs["alpha"].detach().cpu().tolist()
            dirichlet_mean = outputs["dirichlet_mean"].detach().cpu().tolist()
            proto_dist = outputs["proto_dist"].detach().cpu().tolist()
            proto_logits = outputs["proto_logits"].detach().cpu().tolist()
            classifier_logits = outputs["classifier_logits"].detach().cpu().tolist()
            evidence_logits = outputs["evidence_logits"].detach().cpu().tolist()
            mus = outputs["mu"].detach().cpu().tolist()

            for offset, pred_id in enumerate(pred_ids):
                row = data[start + offset]
                pred_label = DEFAULT_LABELS[pred_id]
                row["retrieval"] = pred_label
                row["retrieval_conf"] = float(confs[offset])
                row["retrieval_probs"] = [float(x) for x in probs[offset]]
                row["retrieval_probs_order"] = list(DEFAULT_LABELS)
                row["retrieval_probs_source"] = f"{args.router_name}_vib_router"
                row["retrieval_logits"] = [float(x) for x in logits[offset]]
                row["retrieval_uncertainty"] = float(uncertainty[offset])
                row["retrieval_alpha"] = [float(x) for x in alphas[offset]]
                row["retrieval_dirichlet_mean"] = [float(x) for x in dirichlet_mean[offset]]
                row["retrieval_proto_dist"] = [float(x) for x in proto_dist[offset]]
                row["retrieval_proto_logits"] = [float(x) for x in proto_logits[offset]]
                row["retrieval_classifier_logits"] = [float(x) for x in classifier_logits[offset]]
                row["retrieval_evidence_logits"] = [float(x) for x in evidence_logits[offset]]
                row["retrieval_latent_mu"] = [float(x) for x in mus[offset]]
                row["retrieval_margin"] = float(sorted(probs[offset], reverse=True)[0] - sorted(probs[offset], reverse=True)[1])

                counts[pred_label] += 1
                total_conf += confs[offset]
                total_uncertainty += uncertainty[offset]

                gt = str(row.get("gt_retrieval", "")).strip().lower()
                if gt == pred_label:
                    correct_4class += 1
                if _route_match(target, pred_label, gt):
                    correct_text_equiv += 1

        output_path = os.path.join(output_root, os.path.basename(path))
        _write_json(output_path, data)
        total = max(1, len(data))
        overall_results.append(
            {
                "Path": os.path.basename(path),
                "accuracy": round(correct_text_equiv / total, 4),
                "accuracy_4class": round(correct_4class / total, 4),
                "text_equiv_accuracy": round(correct_text_equiv / total, 4),
                "text_equiv_applied": _is_text_equiv_target(target),
                "avg_conf": round(total_conf / total, 4),
                "avg_uncert": round(total_uncertainty / total, 4),
                "no": counts["no"],
                "paragraph": counts["paragraph"],
                "document": counts["document"],
                "image": counts["image"],
            }
        )

    print("============================================================")
    for row in overall_results:
        print(
            f"{row['Path']}: acc={row['accuracy']:.4f}, raw_acc={row['accuracy_4class']:.4f}, conf={row['avg_conf']:.4f}, "
            f"uncert={row['avg_uncert']:.4f}, no={row['no']}, para={row['paragraph']}, "
            f"doc={row['document']}, image={row['image']}"
        )
    print("============================================================")
    print(f"Saved routed files to: {output_root}")


if __name__ == "__main__":
    main()
