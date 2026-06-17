import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_bayes_vib_posterior as PV

MODALITIES = ["no", "paragraph", "document", "image"]


def _load_json_list(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(str(path))
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def _row_key(row, fallback_idx):
    return str(row.get("index", fallback_idx))


def _index_rows(rows):
    return {_row_key(row, idx): row for idx, row in enumerate(rows)}


def _parse_fixed_root_overrides(value):
    overrides = {}
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "fixed_root_overrides entries must look like "
                "document=eval/results_.../qwen-api:qwen3.6-plus/t5-large"
            )
        modality, root = item.split("=", 1)
        modality = modality.strip().lower()
        root = root.strip()
        if modality not in MODALITIES:
            raise ValueError(f"Invalid modality in fixed_root_overrides: {modality}")
        if not root:
            raise ValueError(f"Empty fixed-root override for modality: {modality}")
        overrides[modality] = root
    return overrides


def _valid_modalities(target):
    valid = ["no", "paragraph", "document"]
    if str(target).lower() in {"webqa", "visual_rag"}:
        valid.append("image")
    return valid


def _parse_probs(row, prob_field):
    fields = []
    if prob_field == "auto":
        fields = ["retrieval_probs", "retrieval_dirichlet_mean"]
    else:
        fields = [prob_field]
    for field in fields:
        value = row.get(field)
        if isinstance(value, list) and len(value) == len(MODALITIES):
            try:
                probs = [max(0.0, float(x)) for x in value]
            except (TypeError, ValueError):
                continue
            total = sum(probs)
            if total > 0:
                return [x / total for x in probs], field
    return None, None


def _top_modalities(row, target, top_n, prob_field):
    valid = _valid_modalities(target)
    probs, used_field = _parse_probs(row, prob_field)
    if probs is None:
        primary = str(row.get("retrieval", "")).lower()
        if primary == "image" and "image" not in valid:
            primary = "document"
        if primary not in valid:
            primary = valid[0]
        weights = [1.0]
        return [primary], weights, used_field

    ranked = sorted(
        valid,
        key=lambda modality: probs[MODALITIES.index(modality)],
        reverse=True,
    )
    selected = ranked[: max(1, int(top_n))]
    raw_weights = [probs[MODALITIES.index(modality)] for modality in selected]
    total = sum(raw_weights)
    if total <= 0:
        weights = [1.0 / len(selected)] * len(selected)
    else:
        weights = [x / total for x in raw_weights]
    return selected, weights, used_field


def _load_fixed_maps(args):
    maps = {}
    fixed_root_overrides = _parse_fixed_root_overrides(args.fixed_root_overrides)
    for modality in _valid_modalities(args.target):
        root = fixed_root_overrides.get(
            modality,
            args.fixed_root_template.format(modality=modality),
        )
        path = Path(root) / f"{args.target}_top{args.top_k}_{args.alpha}_{args.nframes}.json"
        rows = _load_json_list(path)
        maps[modality] = _index_rows(rows)
    return maps


def _candidate_from_fixed(fixed_maps, key, modality, weight):
    fixed_row = fixed_maps.get(modality, {}).get(key)
    if fixed_row is None:
        return {
            "modality": modality,
            "weight": float(weight),
            "retrieved": [],
            "response": "",
            "missing_fixed_candidate": True,
        }
    return {
        "modality": modality,
        "weight": float(weight),
        "retrieved": fixed_row.get("retrieved", []),
        "response": fixed_row.get("response", ""),
    }


def _result_path(args):
    model_name = args.model_path.split("/")[-1]
    out_dir = Path(args.output_root) / model_name / args.router_model
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = PV.BASE._sanitize_tag(args.tag or args.mode)
    return out_dir / f"{args.target}_top{args.top_k}_{args.alpha}_{args.nframes}_{tag}.json"


def main():
    parser = argparse.ArgumentParser(
        description="Build no-Bayes ablation rows from cached fixed-branch generations."
    )
    parser.add_argument("--model_path", type=str, default="qwen-api:qwen3.6-plus")
    parser.add_argument("--router_model", type=str, required=True, choices=["distilbert", "t5-large"])
    parser.add_argument("--target", type=str, required=True, choices=PV.BASE.SUPPORTED_TARGETS)
    parser.add_argument("--route_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--fixed_root_template", type=str, default="eval/results_qwen36plus_api_compare_fixed_{modality}/qwen-api:qwen3.6-plus/t5-large")
    parser.add_argument(
        "--fixed_root_overrides",
        type=str,
        default="",
        help=(
            "Semicolon-separated modality-specific fixed result roots. "
            "Example: document=eval/results_qwen36plus_api_compare_fixed_document_corrected/"
            "qwen-api:qwen3.6-plus/t5-large"
        ),
    )
    parser.add_argument("--mode", type=str, required=True, choices=["top1", "topk_verifier"])
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--prob_field", type=str, default="auto")
    parser.add_argument("--top_n", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--alpha", type=str, default="0.2")
    parser.add_argument("--nframes", type=str, default="1")
    parser.add_argument("--posterior_agreement_weight", type=float, default=1.0)
    parser.add_argument("--posterior_conflict_weight", type=float, default=0.35)
    parser.add_argument("--posterior_route_weight", type=float, default=0.15)
    parser.add_argument("--posterior_evidence_weight", type=float, default=0.05)
    parser.add_argument("--posterior_empty_penalty", type=float, default=1.0)
    parser.add_argument("--posterior_non_answer_penalty", type=float, default=0.85)
    parser.add_argument("--posterior_verifier", type=int, default=1, choices=[0, 1])
    parser.add_argument("--posterior_verifier_choice_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--posterior_verifier_max_new_tokens", type=int, default=64)
    parser.add_argument("--posterior_evidence_max_chars", type=int, default=1200)
    parser.add_argument("--resume", type=int, default=1, choices=[0, 1])
    parser.add_argument("--partial_save_every", type=int, default=25)
    args = parser.parse_args()

    route_file = Path(args.route_dir) / args.router_model / f"{args.target}.json"
    data = _load_json_list(route_file)
    fixed_maps = _load_fixed_maps(args)
    output_file = _result_path(args)
    partial_file = Path(str(output_file) + ".partial")

    resume_rows = []
    if args.resume and partial_file.is_file():
        resume_rows = _load_json_list(partial_file)
    elif args.resume and output_file.is_file():
        resume_rows = _load_json_list(output_file)
    resume_map = _index_rows(resume_rows) if resume_rows else {}

    model = None
    verifier_args = SimpleNamespace(
        posterior_agreement_weight=args.posterior_agreement_weight,
        posterior_conflict_weight=args.posterior_conflict_weight,
        posterior_route_weight=args.posterior_route_weight,
        posterior_evidence_weight=args.posterior_evidence_weight,
        posterior_empty_penalty=args.posterior_empty_penalty,
        posterior_non_answer_penalty=args.posterior_non_answer_penalty,
        posterior_verifier=args.posterior_verifier,
        posterior_verifier_choice_only=args.posterior_verifier_choice_only,
        posterior_verifier_max_new_tokens=args.posterior_verifier_max_new_tokens,
        posterior_evidence_max_chars=args.posterior_evidence_max_chars,
    )
    if args.mode == "topk_verifier" and args.posterior_verifier == 1:
        model = PV.BASE.ModelLoader(args.model_path)

    out = []
    for idx, row in enumerate(tqdm(data, desc=f"{args.mode}:{args.router_model}:{args.target}")):
        key = _row_key(row, idx)
        old = resume_map.get(key)
        if old is not None and old.get("response") not in (None, ""):
            out.append(old)
            continue

        modalities, weights, used_prob_field = _top_modalities(row, args.target, args.top_n, args.prob_field)
        candidates = [_candidate_from_fixed(fixed_maps, key, m, w) for m, w in zip(modalities, weights)]

        result = dict(row)
        result["retrieval_original"] = row.get("retrieval")
        result["retrieval_no_bayes_top_modalities"] = modalities
        result["retrieval_no_bayes_top_weights"] = weights
        result["retrieval_no_bayes_prob_field"] = used_prob_field
        result["retrieval_no_bayes_mode"] = args.mode
        result["retrieval_no_bayes_candidates"] = candidates

        if args.mode == "top1" or len(candidates) == 1:
            selected_idx = 0
            response = str(candidates[0].get("response", ""))
            retrieved = candidates[0].get("retrieved", [])
            fusion_mode = "raw_top1"
            scores = []
            verifier_raw = ""
        else:
            response, scores, selected_idx, fusion_mode, verifier_raw = PV._posterior_predictive_fusion(
                args.target,
                row,
                candidates,
                model,
                verifier_args,
            )
            if selected_idx is None or selected_idx < 0 or selected_idx >= len(candidates):
                selected_idx = 0
            retrieved = candidates[selected_idx].get("retrieved", [])

        result["retrieval"] = candidates[selected_idx].get("modality", modalities[0])
        result["retrieved"] = retrieved
        result["response"] = response
        result["retrieval_no_bayes_selected_index"] = selected_idx
        result["retrieval_no_bayes_fusion_mode"] = fusion_mode
        result["retrieval_no_bayes_scores"] = scores
        if verifier_raw:
            result["retrieval_no_bayes_verifier_response"] = verifier_raw
        out.append(result)

        if args.partial_save_every and len(out) % args.partial_save_every == 0:
            partial_file.write_text(json.dumps(out, indent=4, ensure_ascii=False), encoding="utf-8")

    output_file.write_text(json.dumps(out, indent=4, ensure_ascii=False), encoding="utf-8")
    if partial_file.is_file():
        partial_file.unlink()

    meta = vars(args).copy()
    meta.update({
        "route_file": str(route_file),
        "output_file": str(output_file),
        "fixed_root_template": args.fixed_root_template,
        "bayes_selector": False,
        "uses_cached_fixed_branch_generations": True,
    })
    meta_file = Path(str(output_file).replace(".json", ".meta.json"))
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {output_file}")
    print(f"Saved meta: {meta_file}")


if __name__ == "__main__":
    main()
