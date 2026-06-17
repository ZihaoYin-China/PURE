import argparse
import gc
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_bayes_vib_posterior as PV
import eval_topk_verifier_from_fixed as TF
from route.bayes_dirichlet_router_vib import BayesDirichletRouterVIB, MODALITIES


def _get_env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _nframes_tag(value):
    return str(value).replace(",", "_").replace(":", "")


def _output_file(args):
    model_name = args.model_path.split("/")[-1]
    suffix = "bayes"
    if args.bayes_tag.strip():
        suffix = f"bayes_{PV.BASE._sanitize_tag(args.bayes_tag)}"
    elif args.soft_top_n > 1:
        suffix = f"bayes_vibposterior_softtop{args.soft_top_n}_{args.soft_weight_mode}"
    out_dir = Path(args.output_root) / model_name / args.router_model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{args.target}_top{args.top_k}_{args.alpha}_{_nframes_tag(args.nframes)}_{suffix}.json"


def _is_complete_row(row):
    return row.get("response") not in (None, "") and row.get("retrieval_bayes") is not None


def _candidate_from_fixed(fixed_maps, key, modality, weight):
    candidate = TF._candidate_from_fixed(fixed_maps, key, modality, weight)
    if candidate.get("missing_fixed_candidate"):
        raise KeyError(f"Missing fixed candidate for index={key}, modality={modality}")
    return candidate


def main():
    parser = argparse.ArgumentParser(
        description="Fast COVER-style Bayes+VIB posterior verifier from cached fixed branch generations."
    )
    parser.add_argument("--model_path", type=str, default="qwen-api:qwen3.6-plus")
    parser.add_argument("--router_model", type=str, required=True, choices=["distilbert", "t5-large"])
    parser.add_argument("--target", type=str, required=True, choices=PV.BASE.SUPPORTED_TARGETS)
    parser.add_argument("--route_dir", type=str, required=True)
    parser.add_argument("--base_route_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--fixed_root_template", type=str, default="eval/results_qwen36plus_api_compare_fixed_{modality}/qwen-api:qwen3.6-plus/t5-large")
    parser.add_argument("--fixed_root_overrides", type=str, default="")
    parser.add_argument("--bayes_tag", type=str, default="")
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--alpha", type=str, default="0.2")
    parser.add_argument("--nframes", type=str, default="1")
    parser.add_argument("--alpha_prior", type=str, default="1,1,1,1")
    parser.add_argument("--alpha_prior_by_target", type=str, default="")
    parser.add_argument("--tau", type=float, default=10.0)
    parser.add_argument("--beta_cost", type=float, default=0.1)
    parser.add_argument("--modality_costs", type=str, default="0.0,0.25,0.45,0.60")
    parser.add_argument("--default_confidence", type=float, default=0.72)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.35)
    parser.add_argument("--fallback_when_uncertain", type=int, default=1, choices=[0, 1])
    parser.add_argument("--decision_mode", type=str, default="mean", choices=["mean", "thompson"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vib_uncertainty_low", type=float, default=0.28)
    parser.add_argument("--vib_uncertainty_high", type=float, default=0.45)
    parser.add_argument("--vib_weight_low", type=float, default=0.35)
    parser.add_argument("--vib_weight_high", type=float, default=0.85)
    parser.add_argument("--dynamic_tau_min", type=float, default=0.35)
    parser.add_argument("--dynamic_tau_max", type=float, default=1.8)
    parser.add_argument("--evidence_saturation", type=float, default=8.0)
    parser.add_argument("--soft_top_n", type=int, default=2)
    parser.add_argument("--soft_weight_mode", type=str, default="theta", choices=["theta", "utility"])
    parser.add_argument("--soft_store_candidates", type=int, default=1, choices=[0, 1])
    parser.add_argument("--hybrid_use_base", type=int, default=1, choices=[0, 1])
    parser.add_argument("--vib_prob_field", type=str, default="probs")
    parser.add_argument("--posterior_agreement_weight", type=float, default=1.0)
    parser.add_argument("--posterior_conflict_weight", type=float, default=0.3)
    parser.add_argument("--posterior_route_weight", type=float, default=0.12)
    parser.add_argument("--posterior_evidence_weight", type=float, default=0.05)
    parser.add_argument("--posterior_empty_penalty", type=float, default=1.0)
    parser.add_argument("--posterior_non_answer_penalty", type=float, default=0.85)
    parser.add_argument("--posterior_verifier", type=int, default=1, choices=[0, 1])
    parser.add_argument("--posterior_verifier_choice_only", type=int, default=1, choices=[0, 1])
    parser.add_argument("--posterior_verifier_max_new_tokens", type=int, default=64)
    parser.add_argument("--posterior_evidence_max_chars", type=int, default=1200)
    args = parser.parse_args()

    route_file = Path(args.route_dir) / args.router_model / f"{args.target}.json"
    data = TF._load_json_list(route_file)
    fixed_maps = TF._load_fixed_maps(args)

    alpha_prior = PV.BASE._parse_float_list(args.alpha_prior, expected_len=len(MODALITIES))
    modality_costs = PV.BASE._parse_float_list(args.modality_costs, expected_len=len(MODALITIES))
    alpha_prior_by_target = PV.BASE._parse_prior_map(args.alpha_prior_by_target)
    bayes_router = BayesDirichletRouterVIB(
        alpha_prior=alpha_prior,
        alpha_prior_by_target=alpha_prior_by_target,
        tau=args.tau,
        beta_cost=args.beta_cost,
        modality_costs=modality_costs,
        default_confidence=args.default_confidence,
        uncertainty_threshold=args.uncertainty_threshold,
        fallback_when_uncertain=bool(args.fallback_when_uncertain),
        decision_mode=args.decision_mode,
        seed=args.seed,
        vib_uncertainty_low=args.vib_uncertainty_low,
        vib_uncertainty_high=args.vib_uncertainty_high,
        vib_weight_low=args.vib_weight_low,
        vib_weight_high=args.vib_weight_high,
        dynamic_tau_min=args.dynamic_tau_min,
        dynamic_tau_max=args.dynamic_tau_max,
        evidence_saturation=args.evidence_saturation,
    )

    base_route_map = {}
    if args.hybrid_use_base == 1:
        base_route_map = PV._load_base_route_map(args.base_route_dir, args.router_model, args.target)

    output_file = _output_file(args)
    partial_file = Path(str(output_file) + ".partial")
    partial_save_every = max(0, _get_env_int("EVAL_PARTIAL_SAVE_EVERY", 25))
    gc_every = max(0, _get_env_int("EVAL_GC_EVERY", 100))

    resume_rows = []
    if os.environ.get("EVAL_RESUME_PARTIAL", "1") not in {"0", "false", "False"}:
        if partial_file.is_file():
            resume_rows = TF._load_json_list(partial_file)
        elif output_file.is_file():
            resume_rows = TF._load_json_list(output_file)
    resume_map = TF._index_rows(resume_rows) if resume_rows else {}
    if resume_map:
        resumed = 0
        for idx, row in enumerate(data):
            old = resume_map.get(TF._row_key(row, idx))
            if old is not None and _is_complete_row(old):
                data[idx] = old
                resumed += 1
        print(f"[INFO] Resumed {resumed} completed rows from cached output/partial")

    model = PV.BASE.ModelLoader(args.model_path) if args.posterior_verifier == 1 else None

    for row_idx, row in enumerate(tqdm(data, desc=f"fast-cover-fixed:{args.router_model}:{args.target}")):
        if _is_complete_row(row):
            continue

        key = TF._row_key(row, row_idx)
        original_modality = str(row.get("retrieval", "error")).lower()
        retrieval_conf = row.get("retrieval_conf", None)
        vib_uncertainty = row.get("retrieval_uncertainty", None)
        vib_probs, vib_probs_field = PV._select_vib_probs(row, args.vib_prob_field)
        vib_alpha = PV._parse_float_vector(row, "retrieval_alpha")
        base_probs, base_row = PV._base_probs_for_row(base_route_map, row)

        decision = bayes_router.decide(
            retrieval=original_modality,
            retrieval_conf=retrieval_conf,
            target=args.target,
            vib_probs=vib_probs,
            vib_alpha=vib_alpha,
            vib_uncertainty=vib_uncertainty,
            base_probs=base_probs,
        )
        modality = decision["selected"]
        if args.target not in {"webqa", "visual_rag"} and modality == "image":
            modality = "document"

        if args.soft_top_n > 1:
            soft_modalities, soft_weights = PV.BASE._select_soft_modalities(
                decision=decision,
                target=args.target,
                soft_top_n=args.soft_top_n,
                weight_mode=args.soft_weight_mode,
                primary_modality=modality,
            )
        else:
            soft_modalities, soft_weights = [modality], [1.0]

        candidates = [
            _candidate_from_fixed(fixed_maps, key, candidate_modality, candidate_weight)
            for candidate_modality, candidate_weight in zip(soft_modalities, soft_weights)
        ]

        if len(candidates) <= 1:
            selected_idx = 0
            response = str(candidates[0].get("response", ""))
            retrieved = list(candidates[0].get("retrieved", []))
            posterior_scores = []
            fusion_mode = "cached_single_branch"
            verifier_raw = ""
            retrieved_union = list(retrieved)
        else:
            retrieved_union = []
            seen = set()
            for candidate in candidates:
                for item in candidate.get("retrieved", []):
                    if item not in seen:
                        seen.add(item)
                        retrieved_union.append(item)
            response, posterior_scores, selected_idx, fusion_mode, verifier_raw = PV._posterior_predictive_fusion(
                args.target,
                row,
                candidates,
                model,
                args,
            )
            if selected_idx is None or selected_idx < 0 or selected_idx >= len(candidates):
                selected_idx = 0
            retrieved = list(candidates[selected_idx].get("retrieved", []))

        row["retrieved"] = retrieved
        row["response"] = response
        row["retrieval_original"] = original_modality
        row["retrieval_bayes"] = modality
        row["retrieval_bayes_selective_mode"] = "soft_top_n_cached_fixed" if len(candidates) > 1 else "cached_single_branch"
        row["retrieval_bayes_selective_details"] = {"uses_cached_fixed_branch_generations": True}
        row["retrieval_bayes_uncertainty"] = decision["uncertainty"]
        row["retrieval_bayes_posterior_alpha"] = decision["posterior_alpha"]
        row["retrieval_bayes_router_probs"] = decision["router_probs"]
        row["retrieval_bayes_theta"] = decision["theta"]
        row["retrieval_bayes_utility"] = decision["utility"]
        row["retrieval_bayes_alpha_prior"] = decision["alpha_prior"]
        row["retrieval_bayes_vib_probs"] = decision["vib_probs"]
        row["retrieval_bayes_base_probs"] = decision["base_probs"]
        row["retrieval_bayes_vib_weight"] = decision["vib_weight"]
        row["retrieval_bayes_tau_scale"] = decision["tau_scale"]
        row["retrieval_bayes_tau_effective"] = decision["tau_effective"]
        row["retrieval_bayes_probs_source"] = decision["probs_source"]
        row["retrieval_bayes_vib_prob_field"] = vib_probs_field
        row["retrieval_bayes_used_base_probs"] = bool(base_probs is not None)
        if base_row is not None:
            row["retrieval_bayes_base_probs_source"] = base_row.get("retrieval_probs_source", "base_route_file")

        row["retrieval_bayes_soft_enabled"] = len(candidates) > 1
        row["retrieval_bayes_soft_modalities"] = soft_modalities
        row["retrieval_bayes_soft_weights"] = soft_weights
        row["retrieval_bayes_soft_weight_mode"] = args.soft_weight_mode
        row["retrieval_bayes_soft_fusion_mode"] = fusion_mode
        row["retrieval_bayes_soft_top_n"] = len(soft_modalities)
        row["retrieval_bayes_soft_retrieved_union"] = retrieved_union
        row["retrieval_bayes_posterior_generation_scores"] = posterior_scores
        row["retrieval_bayes_posterior_generation_selected_index"] = selected_idx
        row["retrieval_bayes_posterior_score_gap"] = PV._posterior_score_gap(posterior_scores)
        row["retrieval_bayes_posterior_safe_fallback"] = False
        if verifier_raw:
            row["retrieval_bayes_posterior_generation_verifier_response"] = verifier_raw
        if args.soft_store_candidates == 1:
            row["retrieval_bayes_soft_candidates"] = candidates
        row["retrieval_bayes_posterior_generation_objective"] = {
            "agreement_weight": args.posterior_agreement_weight,
            "conflict_weight": args.posterior_conflict_weight,
            "route_weight": args.posterior_route_weight,
            "evidence_weight": args.posterior_evidence_weight,
            "empty_penalty": args.posterior_empty_penalty,
            "non_answer_penalty": args.posterior_non_answer_penalty,
            "verifier": bool(args.posterior_verifier),
            "verifier_choice_only": bool(args.posterior_verifier_choice_only),
            "verifier_max_new_tokens": args.posterior_verifier_max_new_tokens,
            "evidence_max_chars": args.posterior_evidence_max_chars,
            "uses_cached_fixed_branch_generations": True,
        }

        if partial_save_every and (row_idx + 1) % partial_save_every == 0:
            partial_file.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
            print(f"[INFO] Saved partial results to: {partial_file}")
        if gc_every and (row_idx + 1) % gc_every == 0:
            gc.collect()

    output_file.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")
    if partial_file.is_file():
        partial_file.unlink()

    meta = vars(args).copy()
    meta.update({
        "route_file": str(route_file),
        "output_file": str(output_file),
        "fixed_root_template": args.fixed_root_template,
        "fixed_root_overrides": args.fixed_root_overrides,
        "bayes_selector": True,
        "uses_cached_fixed_branch_generations": True,
        "fast_cover_from_fixed": True,
    })
    meta_file = Path(str(output_file).replace(".json", ".meta.json"))
    meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved fast fixed COVER results to: {output_file}")
    print(f"Saved meta to: {meta_file}")


if __name__ == "__main__":
    main()
