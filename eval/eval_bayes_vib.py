import argparse
import gc
import importlib.util
import json
import os

from tqdm import tqdm

from retrieve.retrieve_image import InternImgRetriever
from retrieve.retrieve_image_bge import BGEImageRetriever
from retrieve.retrieve_text import BGETextRetriever
from route.bayes_dirichlet_router_vib import BayesDirichletRouterVIB, MODALITIES


def _load_base_eval_module():
    module_path = os.path.join(os.path.dirname(__file__), "eval_bayes.py")
    spec = importlib.util.spec_from_file_location("_base_eval_bayes_module", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base_eval_module()


def _get_env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _clear_inactive_retrievers(active_name, enabled):
    if not enabled:
        return

    global retriever_paragraph, retriever_document, retriever_image
    cleared = False

    active_is_text = active_name in {"paragraph", "document"}

    if not active_is_text and active_name != "paragraph" and retriever_paragraph is not None:
        retriever_paragraph = None
        cleared = True
    if not active_is_text and active_name != "document" and retriever_document is not None:
        retriever_document = None
        cleared = True
    if active_name != "image" and retriever_image is not None:
        retriever_image = None
        cleared = True

    if cleared:
        gc.collect()


def _parse_float_vector(row, key):
    value = row.get(key)
    if not isinstance(value, list) or len(value) != len(MODALITIES):
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _select_vib_probs(row, prob_field):
    prob_field = str(prob_field or "auto").strip().lower()
    if prob_field == "dirichlet_mean":
        return _parse_float_vector(row, "retrieval_dirichlet_mean"), "retrieval_dirichlet_mean"
    if prob_field == "probs":
        return _parse_float_vector(row, "retrieval_probs"), "retrieval_probs"

    dirichlet_mean = _parse_float_vector(row, "retrieval_dirichlet_mean")
    if dirichlet_mean is not None:
        return dirichlet_mean, "retrieval_dirichlet_mean"
    return _parse_float_vector(row, "retrieval_probs"), "retrieval_probs"


def _load_base_route_map(base_route_dir, router_model, target):
    if not base_route_dir:
        return {}
    route_file = os.path.join(base_route_dir, router_model, f"{target}.json")
    if not os.path.isfile(route_file):
        return {}
    with open(route_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for idx, row in enumerate(data):
        key = str(row.get("index", idx))
        mapping[key] = row
    return mapping


def _load_base_result_map(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    mapping = {}
    for idx, row in enumerate(data):
        key = str(row.get("index", idx))
        mapping[key] = row
    return mapping


def _same_execution_plan(base_row, selected_modality, soft_modalities):
    if not base_row:
        return False
    if not str(base_row.get("response", "")).strip():
        return False

    base_selected = str(base_row.get("retrieval_bayes", "")).strip().lower()
    if base_selected != str(selected_modality or "").strip().lower():
        return False

    base_soft = base_row.get("retrieval_bayes_soft_modalities")
    if isinstance(base_soft, list):
        base_soft = [str(x).strip().lower() for x in base_soft]
        current_soft = [str(x).strip().lower() for x in soft_modalities]
        return base_soft == current_soft

    return len(soft_modalities) <= 1


def _base_probs_for_row(
    base_route_map,
    row,
    *,
    temperature=1.0,
    blend_with_original=0.0,
):
    key = str(row.get("index", ""))
    if key in base_route_map:
        base_row = base_route_map[key]
    else:
        base_row = None
    if base_row is None:
        return None, None
    original_retrieval = str(base_row.get("retrieval", row.get("retrieval", ""))).lower()
    return BASE._parse_router_probs(
        base_row,
        temperature=temperature,
        original_retrieval=original_retrieval,
        blend_with_original=blend_with_original,
    ), base_row


def _parse_vib_target_params(text):
    out = {}
    raw = str(text or "").strip()
    if not raw:
        return out
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        target, values = chunk.split("=", 1)
        parts = [x.strip() for x in values.split(",") if x.strip()]
        if len(parts) != 4:
            raise ValueError(
                "Each entry in --vib_target_params must have 4 comma-separated values: "
                "vib_uncertainty_low,vib_uncertainty_high,vib_weight_low,vib_weight_high"
            )
        out[str(target).strip().lower()] = {
            "vib_uncertainty_low": float(parts[0]),
            "vib_uncertainty_high": float(parts[1]),
            "vib_weight_low": float(parts[2]),
            "vib_weight_high": float(parts[3]),
        }
    return out


def _parse_name_set(raw_value, valid_values=None):
    values = {
        x.strip().lower()
        for x in str(raw_value or "").replace(" ", ",").split(",")
        if x.strip()
    }
    if valid_values is None:
        return values
    valid_values = set(valid_values)
    return {x for x in values if x in valid_values}


def _top_conf_margin(probs):
    if not isinstance(probs, list) or len(probs) != len(MODALITIES):
        return None, None, None
    try:
        values = [float(x) for x in probs]
    except (TypeError, ValueError):
        return None, None, None
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
    top_idx = order[0]
    second_idx = order[1] if len(order) > 1 else order[0]
    return (
        MODALITIES[top_idx],
        float(values[top_idx]),
        float(values[top_idx] - values[second_idx]),
    )


def _risk_control_override(
    *,
    enabled,
    target,
    active_targets,
    stable_base_modalities,
    allow_vib_modalities,
    max_uncertainty,
    min_vib_conf,
    min_vib_margin,
    require_disagreement,
    base_probs,
    vib_probs,
    vib_uncertainty,
):
    info = {
        "enabled": bool(enabled),
        "active": False,
        "blocked": False,
        "reason": "disabled",
        "override": None,
    }
    if not enabled:
        return None, info

    target_key = str(target or "").strip().lower()
    if active_targets and target_key not in active_targets:
        info["reason"] = "target_not_enabled"
        return None, info

    base_top, base_conf, base_margin = _top_conf_margin(base_probs)
    vib_top, vib_conf, vib_margin = _top_conf_margin(vib_probs)
    info.update({
        "active": True,
        "reason": "pass",
        "base_top": base_top,
        "base_conf": base_conf,
        "base_margin": base_margin,
        "vib_top": vib_top,
        "vib_conf": vib_conf,
        "vib_margin": vib_margin,
    })

    if base_top is None or vib_top is None:
        info["reason"] = "missing_probs"
        return None, info

    if base_top not in stable_base_modalities:
        info["reason"] = "base_not_stable"
        return None, info

    vib_uncertainty_value = None
    try:
        vib_uncertainty_value = float(vib_uncertainty)
    except (TypeError, ValueError):
        pass
    info["vib_uncertainty"] = vib_uncertainty_value

    allow_by_modality = (not allow_vib_modalities) or (vib_top in allow_vib_modalities)
    allow_by_uncertainty = (
        vib_uncertainty_value is None
        or vib_uncertainty_value <= float(max_uncertainty)
    )
    allow_by_conf = float(vib_conf) >= float(min_vib_conf)
    allow_by_margin = float(vib_margin) >= float(min_vib_margin)
    allow_by_disagreement = (not require_disagreement) or (vib_top != base_top)

    info.update({
        "allow_by_modality": bool(allow_by_modality),
        "allow_by_uncertainty": bool(allow_by_uncertainty),
        "allow_by_conf": bool(allow_by_conf),
        "allow_by_margin": bool(allow_by_margin),
        "allow_by_disagreement": bool(allow_by_disagreement),
    })

    if (
        allow_by_modality
        and allow_by_uncertainty
        and allow_by_conf
        and allow_by_margin
        and allow_by_disagreement
    ):
        return None, info

    info["blocked"] = True
    info["reason"] = "stable_base_risk_control"
    info["override"] = 0.0
    return 0.0, info


def _confidence_guard_override(
    *,
    enabled,
    target,
    active_targets,
    threshold,
    retrieval_conf,
    base_probs,
):
    info = {
        "enabled": bool(enabled),
        "active": False,
        "blocked": False,
        "reason": "disabled",
        "override": None,
    }
    if not enabled:
        return None, info

    target_key = str(target or "").strip().lower()
    if active_targets and target_key not in active_targets:
        info["reason"] = "target_not_enabled"
        return None, info

    info["active"] = True
    info["reason"] = "pass"
    info["threshold"] = float(threshold)

    conf_value = None
    try:
        conf_value = float(retrieval_conf)
    except (TypeError, ValueError):
        pass
    info["retrieval_conf"] = conf_value

    if conf_value is None:
        info["reason"] = "missing_retrieval_conf"
        return None, info

    if base_probs is None:
        info["reason"] = "missing_base_probs"
        return None, info

    if conf_value < float(threshold):
        info["blocked"] = True
        info["reason"] = "retrieval_conf_below_threshold"
        info["override"] = 0.0
        return 0.0, info

    return None, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="qwen3-vl:8b")
    parser.add_argument(
        "--router_model",
        type=str,
        default="distilbert",
        choices=["gpt", "qwen", "t5-large", "distilbert", "selfrag"],
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"],
    )
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--nframes", type=str, default="1")
    parser.add_argument("--route_dir", type=str, default="route/results_vib")
    parser.add_argument("--output_root", type=str, default="eval/results_bayes_vib_hybrid")
    parser.add_argument(
        "--query_bge_dir",
        type=str,
        default="eval/features/query/bge-large",
        help="Directory containing BGE query feature pickles.",
    )
    parser.add_argument(
        "--query_internvideo_dir",
        type=str,
        default="eval/features/query/internvideo",
        help="Directory containing InternVideo query feature pickles.",
    )
    parser.add_argument(
        "--bge_image_retrieval",
        action="store_true",
        help=(
            "Use BGE caption embeddings for image retrieval instead of "
            "InternVideo image embeddings."
        ),
    )
    parser.add_argument("--bayes_tag", type=str, default="")

    parser.add_argument("--alpha_prior", type=str, default="1,1,1,1")
    parser.add_argument("--tau", type=float, default=8.0)
    parser.add_argument("--beta_cost", type=float, default=0.1)
    parser.add_argument("--modality_costs", type=str, default="0.0,0.25,0.45,0.60")
    parser.add_argument("--default_confidence", type=float, default=0.72)
    parser.add_argument("--uncertainty_threshold", type=float, default=0.35)
    parser.add_argument("--decision_mode", type=str, default="mean", choices=["mean", "thompson"])
    parser.add_argument("--fallback_when_uncertain", type=int, default=1, choices=[0, 1])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alpha_prior_by_target",
        type=str,
        default="",
        help="Target-specific priors: mmlu=8,1,1,0.5;squad=1,8,2,0.5",
    )
    parser.add_argument(
        "--router_probs_temperature",
        type=float,
        default=1.0,
        help="Temperature for base router probabilities. >1 flattens, <1 sharpens.",
    )
    parser.add_argument(
        "--router_probs_temperature_by_target",
        type=str,
        default="",
        help="Target-specific base router prob temperatures: webqa=2.0;hotpotqa=1.2",
    )
    parser.add_argument(
        "--router_probs_blend_with_original",
        type=float,
        default=0.0,
        help="Blend base router probs toward the original router label one-hot. 0 disables.",
    )
    parser.add_argument(
        "--router_probs_blend_with_original_by_target",
        type=str,
        default="",
        help="Target-specific blend strengths in [0,1]: webqa=0.4;hotpotqa=0.2",
    )

    parser.add_argument("--online_update", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--penalty", type=float, default=0.5)
    parser.add_argument("--spread", type=float, default=0.25)
    parser.add_argument("--use_penalty_update", type=int, default=1, choices=[0, 1])

    parser.add_argument("--soft_top_n", type=int, default=1)
    parser.add_argument(
        "--soft_weight_mode",
        type=str,
        default="theta",
        choices=["theta", "utility"],
    )
    parser.add_argument(
        "--soft_fusion_mode",
        type=str,
        default="auto",
        choices=["auto", "none", "llm"],
    )
    parser.add_argument("--soft_fusion_max_new_tokens", type=int, default=64)
    parser.add_argument("--soft_store_candidates", type=int, default=1, choices=[0, 1])

    parser.add_argument("--base_route_dir", type=str, default="route/results_bayes_probs")
    parser.add_argument("--hybrid_use_base", type=int, default=1, choices=[0, 1])
    parser.add_argument(
        "--reuse_base_results_file",
        type=str,
        default="",
        help="Optional Bayes-only result JSON to reuse when VIB keeps the same execution plan.",
    )
    parser.add_argument("--reuse_base_when_same_plan", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--vib_prob_field",
        type=str,
        default="auto",
        choices=["auto", "dirichlet_mean", "probs"],
    )
    parser.add_argument("--vib_uncertainty_low", type=float, default=0.28)
    parser.add_argument("--vib_uncertainty_high", type=float, default=0.45)
    parser.add_argument("--vib_weight_low", type=float, default=0.15)
    parser.add_argument("--vib_weight_high", type=float, default=0.85)
    parser.add_argument(
        "--vib_target_params",
        type=str,
        default="",
        help=(
            "Target-specific VIB parameters: "
            "mmlu=0.24,0.38,0.05,0.20;squad=0.23,0.37,0.10,0.35"
        ),
    )
    parser.add_argument("--dynamic_tau_min", type=float, default=0.35)
    parser.add_argument("--dynamic_tau_max", type=float, default=1.15)
    parser.add_argument("--evidence_saturation", type=float, default=8.0)
    parser.add_argument(
        "--protect_base_modalities",
        type=str,
        default="",
        help="Comma-separated modalities whose base prediction should be protected from VIB override.",
    )
    parser.add_argument(
        "--allow_vib_modalities",
        type=str,
        default="",
        help="Comma-separated modalities where VIB is still allowed to intervene when protection is enabled.",
    )
    parser.add_argument(
        "--protect_base_selected_modalities",
        type=str,
        default="",
        help=(
            "Comma-separated final Bayes-only selected modalities to protect from VIB overrides. "
            "Requires --reuse_base_results_file so the base selected modality can be read."
        ),
    )
    parser.add_argument("--vib_risk_control", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--vib_risk_control_targets",
        type=str,
        default="",
        help="Comma-separated targets where risk-controlled VIB fallback is enabled. Empty means all targets.",
    )
    parser.add_argument(
        "--vib_risk_control_base_modalities",
        type=str,
        default="no,paragraph",
        help="Base top modalities treated as stable enough to protect from risky VIB overrides.",
    )
    parser.add_argument(
        "--vib_risk_control_allow_vib_modalities",
        type=str,
        default="document,image",
        help="VIB top modalities allowed to override a stable base prediction when confidence tests pass.",
    )
    parser.add_argument("--vib_risk_control_max_uncertainty", type=float, default=0.32)
    parser.add_argument("--vib_risk_control_min_vib_conf", type=float, default=0.70)
    parser.add_argument("--vib_risk_control_min_vib_margin", type=float, default=0.20)
    parser.add_argument("--vib_risk_control_require_disagreement", type=int, default=1, choices=[0, 1])
    parser.add_argument("--vib_conf_guard", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--vib_conf_guard_targets",
        type=str,
        default="",
        help="Comma-separated targets where low VIB retrieval_conf falls back to base probabilities.",
    )
    parser.add_argument("--vib_conf_guard_threshold", type=float, default=0.75)
    args = parser.parse_args()

    alpha_prior = BASE._parse_float_list(args.alpha_prior, expected_len=len(MODALITIES))
    modality_costs = BASE._parse_float_list(args.modality_costs, expected_len=len(MODALITIES))
    alpha_prior_by_target = BASE._parse_prior_map(args.alpha_prior_by_target)
    router_probs_temperature_by_target = BASE._parse_scalar_map(args.router_probs_temperature_by_target)
    router_probs_blend_with_original_by_target = BASE._parse_scalar_map(
        args.router_probs_blend_with_original_by_target
    )
    target_router_probs_temperature = router_probs_temperature_by_target.get(
        args.target.lower(),
        args.router_probs_temperature,
    )
    target_router_probs_blend_with_original = router_probs_blend_with_original_by_target.get(
        args.target.lower(),
        args.router_probs_blend_with_original,
    )
    vib_target_params = _parse_vib_target_params(args.vib_target_params)

    print(
        f"[VIB-HYBRID BAYES EVAL] model={args.model_path}, router={args.router_model}, "
        f"target={args.target}, top_k={args.top_k}, alpha={args.alpha}, nframes={args.nframes}, "
        f"soft_top_n={args.soft_top_n}, soft_weight_mode={args.soft_weight_mode}, "
        f"soft_fusion_mode={args.soft_fusion_mode}, vib_prob_field={args.vib_prob_field}, "
        f"hybrid_use_base={bool(args.hybrid_use_base)}, "
        f"router_probs_temperature={target_router_probs_temperature}, "
        f"router_probs_blend_with_original={target_router_probs_blend_with_original}"
    )
    protect_base_modalities = [
        x.strip().lower()
        for x in str(args.protect_base_modalities or "").replace(" ", ",").split(",")
        if x.strip()
    ]
    allow_vib_modalities = [
        x.strip().lower()
        for x in str(args.allow_vib_modalities or "").replace(" ", ",").split(",")
        if x.strip()
    ]
    protect_base_selected_modalities = _parse_name_set(
        args.protect_base_selected_modalities,
        valid_values=set(MODALITIES),
    )
    vib_risk_control_targets = _parse_name_set(
        args.vib_risk_control_targets,
        valid_values={"mmlu", "squad", "natural_questions", "hotpotqa", "webqa"},
    )
    vib_risk_control_base_modalities = _parse_name_set(
        args.vib_risk_control_base_modalities,
        valid_values=set(MODALITIES),
    )
    vib_risk_control_allow_vib_modalities = _parse_name_set(
        args.vib_risk_control_allow_vib_modalities,
        valid_values=set(MODALITIES),
    )
    vib_conf_guard_targets = _parse_name_set(
        args.vib_conf_guard_targets,
        valid_values={"mmlu", "squad", "natural_questions", "hotpotqa", "webqa"},
    )

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
        vib_target_params=vib_target_params,
        dynamic_tau_min=args.dynamic_tau_min,
        dynamic_tau_max=args.dynamic_tau_max,
        evidence_saturation=args.evidence_saturation,
        protect_base_modalities=protect_base_modalities,
        allow_vib_modalities=allow_vib_modalities,
    )

    model = BASE.ModelLoader(args.model_path)

    route_file = os.path.join(args.route_dir, args.router_model, f"{args.target}.json")
    if not os.path.exists(route_file):
        raise FileNotFoundError(
            f"Route result file not found: {route_file}\n"
            "Please run VIB routing first."
        )

    with open(route_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    base_route_map = {}
    if args.hybrid_use_base == 1:
        base_route_map = _load_base_route_map(
            base_route_dir=args.base_route_dir,
            router_model=args.router_model,
            target=args.target,
        )
    base_result_map = _load_base_result_map(args.reuse_base_results_file)

    retriever_paragraph = None
    retriever_document = None
    retriever_image = None
    single_retriever_cache = _get_env_bool("EVAL_SINGLE_RETRIEVER_CACHE", False)

    if single_retriever_cache:
        print("[INFO] EVAL_SINGLE_RETRIEVER_CACHE=1: keeping only the active retriever in memory.")

    def execute_modality(current_row, formatted_query, modality):
        global retriever_paragraph, retriever_document, retriever_image

        retrieved = []

        if modality == "no":
            response = model.inference(formatted_query)
            return response, retrieved

        if modality in ["paragraph", "document"]:
            if modality == "paragraph":
                _clear_inactive_retrievers("paragraph", single_retriever_cache)
                if retriever_paragraph is None:
                    retriever_paragraph = BGETextRetriever(
                        queryfeats_path=os.path.join(args.query_bge_dir, f"{args.target}.pkl"),
                        textfeats_path=[
                            "eval/features/text/squad.pkl",
                            "eval/features/text/natural_questions.pkl",
                        ],
                    )
                retrieved, _ = retriever_paragraph.retrieve(current_row["index"], top_k=args.top_k)
            else:
                _clear_inactive_retrievers("document", single_retriever_cache)
                if retriever_document is None:
                    retriever_document = BGETextRetriever(
                        queryfeats_path=os.path.join(args.query_bge_dir, f"{args.target}.pkl"),
                        textfeats_path=[
                            os.environ.get("HOTPOTQA_TEXT_FEATS", "eval/features/text/hotpotqa.pkl"),
                        ],
                    )
                retrieved, _ = retriever_document.retrieve(current_row["index"], top_k=args.top_k)

            retrieved_texts = []
            for doc in retrieved:
                with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                    retrieved_texts.append(f.read()[:3000])

            response = model.inference(
                formatted_query,
                retrieved_texts=retrieved_texts,
                max_new_tokens=128,
            )
            return response, retrieved

        if modality == "image":
            _clear_inactive_retrievers("image", single_retriever_cache)
            if retriever_image is None:
                if args.bge_image_retrieval:
                    bge_query_path = os.path.join(args.query_bge_dir, f"{args.target}.pkl")
                    bge_caption_path = os.environ.get(
                        "WEBQA_BGE_CAPTION_FEATS",
                        os.path.join("eval/features/image", f"{args.target}_bge_captions.pkl"),
                    )
                    if not os.path.exists(bge_caption_path):
                        raise FileNotFoundError(
                            f"BGE caption features not found: {bge_caption_path}."
                        )
                    retriever_image = BGEImageRetriever(
                        queryfeats_path=bge_query_path,
                        captionfeats_path=bge_caption_path,
                    )
                else:
                    query_img_feat = os.path.join(args.query_internvideo_dir, f"{args.target}.pkl")
                    if not os.path.exists(query_img_feat):
                        raise FileNotFoundError(
                            f"Image query features not found: {query_img_feat}. "
                            "This sample needs image retrieval, but no image query features are prepared."
                        )
                    retriever_image = InternImgRetriever(
                        queryfeats_path=query_img_feat,
                        imgfeats_path=["eval/features/image/webqa.pkl"],
                        imgcapfeats_path=["eval/features/image/webqa_imgcap.pkl"],
                        alpha=args.alpha,
                    )

            retrieved, _ = retriever_image.retrieve(current_row["index"], top_k=args.top_k)
            response = model.inference(
                formatted_query,
                retrieved_images=retrieved,
                max_new_tokens=128,
            )
            return response, retrieved

        raise ValueError(f"Invalid modality after Bayes decision: {modality}")

    for row in tqdm(
        data,
        desc=f"VIB-hybrid Bayes-evaluating {args.target} with {args.model_path} + {args.router_model}",
    ):
        query = BASE.reformat(row)
        original_modality = str(row.get("retrieval", "error")).lower()
        retrieval_conf = row.get("retrieval_conf", None)
        vib_uncertainty = row.get("retrieval_uncertainty", None)
        vib_probs, vib_probs_field = _select_vib_probs(row, args.vib_prob_field)
        vib_alpha = _parse_float_vector(row, "retrieval_alpha")
        base_probs, base_row = _base_probs_for_row(
            base_route_map,
            row,
            temperature=target_router_probs_temperature,
            blend_with_original=target_router_probs_blend_with_original,
        )
        base_result_row = base_result_map.get(str(row.get("index", "")))
        vib_weight_override, risk_control_info = _risk_control_override(
            enabled=args.vib_risk_control == 1,
            target=args.target,
            active_targets=vib_risk_control_targets,
            stable_base_modalities=vib_risk_control_base_modalities,
            allow_vib_modalities=vib_risk_control_allow_vib_modalities,
            max_uncertainty=args.vib_risk_control_max_uncertainty,
            min_vib_conf=args.vib_risk_control_min_vib_conf,
            min_vib_margin=args.vib_risk_control_min_vib_margin,
            require_disagreement=bool(args.vib_risk_control_require_disagreement),
            base_probs=base_probs,
            vib_probs=vib_probs,
            vib_uncertainty=vib_uncertainty,
        )
        conf_guard_override, conf_guard_info = _confidence_guard_override(
            enabled=args.vib_conf_guard == 1,
            target=args.target,
            active_targets=vib_conf_guard_targets,
            threshold=args.vib_conf_guard_threshold,
            retrieval_conf=retrieval_conf,
            base_probs=base_probs,
        )
        if conf_guard_override is not None:
            vib_weight_override = conf_guard_override

        force_base_result_by_selected_guard = False
        decision = bayes_router.decide(
            retrieval=original_modality,
            retrieval_conf=retrieval_conf,
            target=args.target,
            vib_probs=vib_probs,
            vib_alpha=vib_alpha,
            vib_uncertainty=vib_uncertainty,
            base_probs=base_probs,
            vib_weight_override=vib_weight_override,
        )
        modality = decision["selected"]

        selected_guard_info = {
            "enabled": bool(protect_base_selected_modalities),
            "active": False,
            "blocked": False,
            "reason": "disabled" if not protect_base_selected_modalities else "pass",
            "override": None,
        }
        if protect_base_selected_modalities:
            if base_result_row is None:
                selected_guard_info["reason"] = "missing_base_result"
            else:
                base_selected = str(
                    base_result_row.get("retrieval_bayes")
                    or base_result_row.get("retrieval")
                    or ""
                ).strip().lower()
                selected_guard_info.update({
                    "active": True,
                    "base_selected": base_selected,
                    "candidate_selected": modality,
                })
                if (
                    base_selected in protect_base_selected_modalities
                    and modality != base_selected
                ):
                    selected_guard_info["blocked"] = True
                    selected_guard_info["reason"] = "base_selected_protected"
                    selected_guard_info["override"] = "force_base_result"
                    force_base_result_by_selected_guard = True
                    modality = base_selected
                    selected_guard_info["selected_after_override"] = modality

        if args.target != "webqa" and modality == "image":
            modality = "document"

        soft_top_n_effective = int(args.soft_top_n)
        if args.target != "webqa" and modality == "image":
            modality = "document"

        retrieved = []
        response = ""
        soft_modalities = [modality]
        soft_weights = [1.0]
        soft_fusion_mode_used = "hard"

        if soft_top_n_effective > 1:
            soft_modalities, soft_weights = BASE._select_soft_modalities(
                decision=decision,
                target=args.target,
                soft_top_n=soft_top_n_effective,
                weight_mode=args.soft_weight_mode,
                primary_modality=modality,
            )

        if force_base_result_by_selected_guard and base_result_row is not None:
            base_soft_modalities = base_result_row.get("retrieval_bayes_soft_modalities")
            base_soft_weights = base_result_row.get("retrieval_bayes_soft_weights")
            if isinstance(base_soft_modalities, list) and base_soft_modalities:
                soft_modalities = [str(x).strip().lower() for x in base_soft_modalities]
                if isinstance(base_soft_weights, list) and len(base_soft_weights) == len(soft_modalities):
                    soft_weights = base_soft_weights
                else:
                    soft_weights = [1.0 / len(soft_modalities)] * len(soft_modalities)
            else:
                soft_modalities = [modality]
                soft_weights = [1.0]

        reused_base_result = (
            args.reuse_base_when_same_plan == 1
            and (
                force_base_result_by_selected_guard
                or _same_execution_plan(base_result_row, modality, soft_modalities)
            )
        )

        if reused_base_result:
            retrieved = base_result_row.get("retrieved", [])
            response = base_result_row.get("response", "")
            if len(soft_modalities) > 1:
                row["retrieval_bayes_soft_enabled"] = True
                row["retrieval_bayes_soft_modalities"] = soft_modalities
                row["retrieval_bayes_soft_weights"] = soft_weights
                row["retrieval_bayes_soft_weight_mode"] = args.soft_weight_mode
                row["retrieval_bayes_soft_fusion_mode"] = base_result_row.get(
                    "retrieval_bayes_soft_fusion_mode",
                    "reused_base",
                )
                row["retrieval_bayes_soft_top_n"] = len(soft_modalities)
                row["retrieval_bayes_soft_retrieved_union"] = base_result_row.get(
                    "retrieval_bayes_soft_retrieved_union",
                    retrieved,
                )
                if args.soft_store_candidates == 1 and base_result_row.get(
                    "retrieval_bayes_soft_candidates"
                ) is not None:
                    row["retrieval_bayes_soft_candidates"] = base_result_row.get(
                        "retrieval_bayes_soft_candidates"
                    )

        elif len(soft_modalities) <= 1:
            response, retrieved = execute_modality(row, query, modality)
        else:
            candidates = []
            retrieved_union = []
            seen_retrieved = set()

            for candidate_modality, candidate_weight in zip(soft_modalities, soft_weights):
                candidate_response, candidate_retrieved = execute_modality(
                    row,
                    query,
                    candidate_modality,
                )
                candidates.append(
                    {
                        "modality": candidate_modality,
                        "weight": float(candidate_weight),
                        "retrieved": candidate_retrieved,
                        "response": candidate_response,
                    }
                )
                for item in candidate_retrieved:
                    if item not in seen_retrieved:
                        seen_retrieved.add(item)
                        retrieved_union.append(item)

            response, soft_fusion_mode_used = BASE._resolve_trivial_fusion(args.target, candidates)
            if not response:
                if args.soft_fusion_mode == "none":
                    response = BASE._weighted_vote_response(args.target, candidates)
                    soft_fusion_mode_used = "weighted_pick"
                else:
                    fusion_query = BASE._build_fusion_query(args.target, row, candidates)
                    fused_response = model.inference(
                        fusion_query,
                        max_new_tokens=args.soft_fusion_max_new_tokens,
                    )
                    if args.target == "mmlu":
                        fused_choice = BASE._extract_choice_from_prediction(fused_response)
                        if fused_choice is not None:
                            response = fused_choice
                            soft_fusion_mode_used = "llm"
                        else:
                            response = BASE._weighted_vote_response(args.target, candidates)
                            soft_fusion_mode_used = "llm_fallback_weighted_pick"
                    else:
                        response = str(fused_response or "").strip()
                        if response:
                            soft_fusion_mode_used = "llm"
                        else:
                            response = BASE._weighted_vote_response(args.target, candidates)
                            soft_fusion_mode_used = "llm_fallback_weighted_pick"

            retrieved = list(candidates[0]["retrieved"])
            row["retrieval_bayes_soft_enabled"] = True
            row["retrieval_bayes_soft_modalities"] = soft_modalities
            row["retrieval_bayes_soft_weights"] = soft_weights
            row["retrieval_bayes_soft_weight_mode"] = args.soft_weight_mode
            row["retrieval_bayes_soft_fusion_mode"] = soft_fusion_mode_used
            row["retrieval_bayes_soft_top_n"] = len(soft_modalities)
            row["retrieval_bayes_soft_retrieved_union"] = retrieved_union
            if args.soft_store_candidates == 1:
                row["retrieval_bayes_soft_candidates"] = candidates

        if len(soft_modalities) <= 1:
            row["retrieval_bayes_soft_enabled"] = False

        row["retrieved"] = retrieved
        row["response"] = response
        row["retrieval_bayes_reused_base_result"] = bool(reused_base_result)

        row["retrieval_original"] = original_modality
        row["retrieval_bayes"] = modality
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
        row["retrieval_bayes_base_top_modality"] = decision["base_top_modality"]
        row["retrieval_bayes_vib_top_modality"] = decision["vib_top_modality"]
        row["retrieval_bayes_protected_by_base"] = decision["protected_by_base"]
        row["retrieval_bayes_vib_weight_overridden"] = decision["vib_weight_overridden"]
        row["retrieval_bayes_risk_control"] = risk_control_info
        row["retrieval_bayes_conf_guard"] = conf_guard_info
        row["retrieval_bayes_selected_guard"] = selected_guard_info
        if base_row is not None:
            row["retrieval_bayes_base_probs_source"] = base_row.get(
                "retrieval_probs_source",
                "base_route_file",
            )

        if args.online_update == 1:
            reward = BASE._light_reward(args.target, row, response)
            if args.use_penalty_update == 1:
                bayes_router.update_prior_with_penalty(
                    selected_modality=modality,
                    reward=reward,
                    eta=args.eta,
                    penalty=args.penalty,
                    rho=args.rho,
                    spread=args.spread,
                )
            else:
                bayes_router.update_prior(
                    selected_modality=modality,
                    reward=reward,
                    eta=args.eta,
                    rho=args.rho,
                )
            row["retrieval_bayes_reward"] = float(reward)
            row["retrieval_bayes_alpha_prior_after"] = bayes_router.alpha_prior.tolist()

    model_name = args.model_path.split("/")[-1]
    nframes_tag = args.nframes.replace(",", "_").replace(":", "")
    bayes_suffix = "bayes"
    if args.bayes_tag.strip():
        bayes_suffix = f"bayes_{BASE._sanitize_tag(args.bayes_tag)}"
    elif args.soft_top_n > 1:
        bayes_suffix = (
            f"bayes_softtop{args.soft_top_n}_{args.soft_weight_mode}_{args.soft_fusion_mode}"
        )

    output_dir = os.path.join(args.output_root, model_name, args.router_model)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{args.target}_top{args.top_k}_{args.alpha}_{nframes_tag}_{bayes_suffix}.json",
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved VIB-hybrid Bayes results to: {output_file}")

    meta = {
        "model_path": args.model_path,
        "router_model": args.router_model,
        "target": args.target,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "nframes": args.nframes,
        "route_dir": args.route_dir,
        "base_route_dir": args.base_route_dir,
        "output_root": args.output_root,
        "query_bge_dir": args.query_bge_dir,
        "query_internvideo_dir": args.query_internvideo_dir,
        "bge_image_retrieval": args.bge_image_retrieval,
        "bayes_tag": args.bayes_tag,
        "alpha_prior_init": alpha_prior,
        "alpha_prior_by_target": alpha_prior_by_target,
        "tau": args.tau,
        "beta_cost": args.beta_cost,
        "modality_costs": modality_costs,
        "router_probs_temperature": target_router_probs_temperature,
        "router_probs_blend_with_original": target_router_probs_blend_with_original,
        "default_confidence": args.default_confidence,
        "uncertainty_threshold": args.uncertainty_threshold,
        "decision_mode": args.decision_mode,
        "fallback_when_uncertain": bool(args.fallback_when_uncertain),
        "online_update": bool(args.online_update),
        "eta": args.eta,
        "rho": args.rho,
        "penalty": args.penalty,
        "spread": args.spread,
        "use_penalty_update": bool(args.use_penalty_update),
        "soft_top_n": args.soft_top_n,
        "soft_weight_mode": args.soft_weight_mode,
        "soft_fusion_mode": args.soft_fusion_mode,
        "soft_fusion_max_new_tokens": args.soft_fusion_max_new_tokens,
        "soft_store_candidates": bool(args.soft_store_candidates),
        "hybrid_use_base": bool(args.hybrid_use_base),
        "reuse_base_results_file": args.reuse_base_results_file,
        "reuse_base_when_same_plan": bool(args.reuse_base_when_same_plan),
        "vib_prob_field": args.vib_prob_field,
        "vib_uncertainty_low": args.vib_uncertainty_low,
        "vib_uncertainty_high": args.vib_uncertainty_high,
        "vib_weight_low": args.vib_weight_low,
        "vib_weight_high": args.vib_weight_high,
        "vib_target_params": vib_target_params,
        "dynamic_tau_min": args.dynamic_tau_min,
        "dynamic_tau_max": args.dynamic_tau_max,
        "evidence_saturation": args.evidence_saturation,
        "protect_base_modalities": protect_base_modalities,
        "allow_vib_modalities": allow_vib_modalities,
        "protect_base_selected_modalities": sorted(protect_base_selected_modalities),
        "vib_risk_control": bool(args.vib_risk_control),
        "vib_risk_control_targets": sorted(vib_risk_control_targets),
        "vib_risk_control_base_modalities": sorted(vib_risk_control_base_modalities),
        "vib_risk_control_allow_vib_modalities": sorted(vib_risk_control_allow_vib_modalities),
        "vib_risk_control_max_uncertainty": args.vib_risk_control_max_uncertainty,
        "vib_risk_control_min_vib_conf": args.vib_risk_control_min_vib_conf,
        "vib_risk_control_min_vib_margin": args.vib_risk_control_min_vib_margin,
        "vib_risk_control_require_disagreement": bool(args.vib_risk_control_require_disagreement),
        "vib_conf_guard": bool(args.vib_conf_guard),
        "vib_conf_guard_targets": sorted(vib_conf_guard_targets),
        "vib_conf_guard_threshold": args.vib_conf_guard_threshold,
    }
    meta_file = output_file.replace(".json", ".meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved VIB-hybrid Bayes meta to: {meta_file}")
