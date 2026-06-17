import argparse
import gc
import importlib.util
import json
import os
import re

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


def _get_env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _clear_inactive_retrievers(active_name, enabled):
    if not enabled:
        return

    global retriever_paragraph, retriever_document, retriever_image
    cleared = False

    if active_name != "paragraph" and retriever_paragraph is not None:
        retriever_paragraph = None
        cleared = True
    if active_name != "document" and retriever_document is not None:
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


def _load_safe_fallback_rows(path):
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Safe fallback file must contain a JSON list: {path}")
    return data


def _posterior_score_gap(scores):
    values = []
    for score in scores or []:
        if not isinstance(score, dict):
            continue
        try:
            values.append(float(score.get("score", 0.0)))
        except (TypeError, ValueError):
            continue
    if len(values) < 2:
        return None
    values.sort(reverse=True)
    return float(values[0] - values[1])


def _extract_verifier_candidate_index(text, num_candidates):
    if not text:
        return None
    text = str(text).strip()
    match = re.search(r"\bcandidate\s*#?\s*(\d+)\b", text, flags=re.IGNORECASE)
    if match is None:
        match = re.match(r"^\s*(\d+)\s*$", text)
    if match is None:
        return None
    idx = int(match.group(1))
    if 0 <= idx < num_candidates:
        return idx
    return None


def _parse_modality_pairs(value):
    pairs = set()
    for item in str(value or "").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip().lower() for part in item.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(
                "posterior_accept_modality_pairs entries must look like 'no,paragraph'"
            )
        pairs.add(tuple(parts))
    return pairs


def _parse_target_filter(value):
    value = str(value or "").strip().lower()
    if not value or value in {"all", "*"}:
        return None
    return {item.strip() for item in re.split(r"[,;\s]+", value) if item.strip()}


def _target_allowed(target, target_filter):
    return target_filter is None or str(target or "").lower() in target_filter


def _valid_modalities_for_target(target):
    valid = ["no", "paragraph", "document"]
    if str(target or "").lower() in {"webqa", "visual_rag"}:
        valid.append("image")
    return valid


def _decision_maps(decision):
    theta = {modality: float(value) for modality, value in zip(MODALITIES, decision["theta"])}
    utility = {modality: float(value) for modality, value in zip(MODALITIES, decision["utility"])}
    return theta, utility


def _top_utility_margin(target, utility, primary_modality):
    valid_modalities = _valid_modalities_for_target(target)
    primary_utility = utility.get(primary_modality, float("-inf"))
    competitors = [
        utility.get(modality, float("-inf"))
        for modality in valid_modalities
        if modality != primary_modality
    ]
    if not competitors:
        return float("inf")
    return float(primary_utility - max(competitors))


def _should_force_no_retrieval(target, decision, args, target_filter):
    if int(getattr(args, "selective_no_retrieval", 0)) != 1:
        return False, {}
    if not _target_allowed(target, target_filter):
        return False, {"reason": "target_not_enabled"}

    uncertainty = float(decision.get("uncertainty", 1.0))
    max_uncertainty = float(getattr(args, "selective_no_uncertainty_max", -1.0))
    if max_uncertainty >= 0 and uncertainty > max_uncertainty:
        return False, {
            "reason": "uncertainty_above_max",
            "uncertainty": uncertainty,
            "uncertainty_max": max_uncertainty,
        }

    theta, utility = _decision_maps(decision)
    valid_modalities = _valid_modalities_for_target(target)
    best_utility = max(utility.get(modality, float("-inf")) for modality in valid_modalities)
    no_theta = theta.get("no", 0.0)
    no_utility = utility.get("no", float("-inf"))
    utility_gap = best_utility - no_utility

    theta_ok = no_theta >= float(getattr(args, "selective_no_theta_min", 1.0))
    utility_ok = utility_gap <= float(getattr(args, "selective_no_utility_margin", -1.0))
    ok = theta_ok or utility_ok
    return ok, {
        "reason": "force_no" if ok else "no_not_confident",
        "theta_no": float(no_theta),
        "utility_no": float(no_utility),
        "best_utility": float(best_utility),
        "utility_gap": float(utility_gap),
        "theta_min": float(getattr(args, "selective_no_theta_min", 1.0)),
        "utility_margin": float(getattr(args, "selective_no_utility_margin", -1.0)),
        "uncertainty": uncertainty,
    }


def _should_use_single_branch(target, decision, primary_modality, args, target_filter):
    if int(getattr(args, "selective_single_branch", 0)) != 1:
        return False, {}
    if not _target_allowed(target, target_filter):
        return False, {"reason": "target_not_enabled"}

    uncertainty = float(decision.get("uncertainty", 1.0))
    max_uncertainty = float(getattr(args, "selective_single_branch_uncertainty_max", -1.0))
    if max_uncertainty >= 0 and uncertainty > max_uncertainty:
        return False, {
            "reason": "uncertainty_above_max",
            "uncertainty": uncertainty,
            "uncertainty_max": max_uncertainty,
        }

    theta, utility = _decision_maps(decision)
    primary_theta = theta.get(primary_modality, 0.0)
    margin = _top_utility_margin(target, utility, primary_modality)
    theta_ok = primary_theta >= float(getattr(args, "selective_single_branch_theta_min", 1.0))
    margin_ok = margin >= float(getattr(args, "selective_single_branch_utility_margin", float("inf")))
    ok = theta_ok or margin_ok
    return ok, {
        "reason": "confident_single_branch" if ok else "route_not_confident",
        "primary_modality": primary_modality,
        "primary_theta": float(primary_theta),
        "utility_margin": float(margin),
        "theta_min": float(getattr(args, "selective_single_branch_theta_min", 1.0)),
        "utility_margin_min": float(getattr(args, "selective_single_branch_utility_margin", float("inf"))),
        "uncertainty": uncertainty,
    }


def _weights_for_modalities(decision, modalities, weight_mode):
    theta, utility = _decision_maps(decision)
    weight_mode = str(weight_mode or "theta").strip().lower()
    if weight_mode == "utility":
        weights = BASE._softmax([utility.get(modality, 0.0) for modality in modalities])
    else:
        raw_weights = [max(0.0, theta.get(modality, 0.0)) for modality in modalities]
        total = sum(raw_weights)
        if total <= 0:
            weights = [1.0 / len(modalities)] * len(modalities)
        else:
            weights = [value / total for value in raw_weights]
    return weights


def _maybe_include_no_candidate(target, decision, soft_modalities, args, target_filter):
    if int(getattr(args, "selective_include_no_candidate", 0)) != 1:
        return soft_modalities, None
    if not _target_allowed(target, target_filter):
        return soft_modalities, {"reason": "target_not_enabled"}
    if "no" in soft_modalities or len(soft_modalities) <= 1:
        return soft_modalities, {"reason": "no_already_present_or_single_branch"}

    max_utility_gap = float(getattr(args, "selective_include_no_utility_gap_max", -1.0))
    theta, utility = _decision_maps(decision)
    best_selected_utility = max(utility.get(modality, float("-inf")) for modality in soft_modalities)
    no_utility = utility.get("no", float("-inf"))
    utility_gap = best_selected_utility - no_utility
    if max_utility_gap >= 0 and utility_gap > max_utility_gap:
        return soft_modalities, {
            "reason": "no_utility_gap_too_large",
            "theta_no": float(theta.get("no", 0.0)),
            "utility_no": float(no_utility),
            "best_selected_utility": float(best_selected_utility),
            "utility_gap": float(utility_gap),
            "utility_gap_max": max_utility_gap,
        }

    updated = list(soft_modalities)
    updated[-1] = "no"
    deduped = []
    for modality in updated:
        if modality not in deduped:
            deduped.append(modality)
    return deduped, {
        "reason": "included_no_candidate",
        "theta_no": float(theta.get("no", 0.0)),
        "utility_no": float(no_utility),
        "best_selected_utility": float(best_selected_utility),
        "utility_gap": float(utility_gap),
        "utility_gap_max": max_utility_gap,
    }


def _base_probs_for_row(base_route_map, row):
    key = str(row.get("index", ""))
    base_row = base_route_map.get(key)
    if base_row is None:
        return None, None
    return BASE._parse_router_probs(base_row), base_row


def _normalize_candidate_weights(candidates):
    raw_weights = [max(0.0, float(candidate.get("weight", 0.0))) for candidate in candidates]
    total = sum(raw_weights)
    if total <= 0:
        return [1.0 / len(candidates)] * len(candidates)
    return [weight / total for weight in raw_weights]


def _candidate_evidence_preview(candidate, max_chars):
    modality = str(candidate.get("modality", "")).lower()
    retrieved = candidate.get("retrieved") or []
    if modality == "no":
        return "No external retrieval evidence; the answer comes from the model prior."
    if not retrieved:
        return "No retrieved evidence was returned for this branch."

    previews = []
    remaining = max(0, int(max_chars))
    for idx, item in enumerate(retrieved, start=1):
        if remaining <= 0:
            break
        path = str(item)
        header = f"Evidence {idx}: {path}"
        body = ""
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    body = f.read(max(0, remaining - len(header) - 2)).strip()
            except OSError:
                body = ""
        text = f"{header}\n{body}" if body else header
        previews.append(text[:remaining])
        remaining -= len(previews[-1])

    return "\n\n".join(previews).strip()


def _answer_instruction(target):
    mode = BASE._answer_mode_for_target(target)
    if mode == "mcq_letter":
        return "Return only a single letter: A, B, C, or D."
    if mode == "exact_short":
        return "Return only the exact short answer."
    return "Return one complete sentence."


def _looks_like_non_answer(text):
    text = str(text or "").strip().lower()
    if not text:
        return True
    patterns = [
        "cannot determine",
        "can't determine",
        "unable to determine",
        "not possible to determine",
        "not possible to answer",
        "cannot answer",
        "can't answer",
        "does not depict",
        "doesn't depict",
        "does not show",
        "doesn't show",
        "not shown",
        "not visible",
        "no evidence",
        "not enough information",
        "insufficient information",
        "does not provide information",
        "there is no answer",
        "question cannot be answered",
        "answer cannot be determined",
    ]
    return any(pattern in text for pattern in patterns)


def _score_posterior_candidates(target, candidates, args):
    if not candidates:
        return [], [], []

    weights = _normalize_candidate_weights(candidates)
    canonicals = [
        BASE._canonical_prediction(target, candidate.get("response", ""))
        for candidate in candidates
    ]
    non_answer_flags = [
        _looks_like_non_answer(candidate.get("response", "")) for candidate in candidates
    ]
    has_direct_candidate = any(
        canonical and not non_answer
        for canonical, non_answer in zip(canonicals, non_answer_flags)
    )

    scores = []
    for idx, candidate in enumerate(candidates):
        canonical = canonicals[idx]
        posterior_answer_mass = 0.0
        posterior_conflict_mass = 0.0

        for other_weight, other_canonical in zip(weights, canonicals):
            if not canonical or not other_canonical:
                continue
            if other_canonical == canonical:
                posterior_answer_mass += other_weight
            else:
                posterior_conflict_mass += other_weight

        modality = str(candidate.get("modality", "")).lower()
        retrieved = candidate.get("retrieved") or []
        branch_has_evidence = 1.0 if modality == "no" or len(retrieved) > 0 else 0.0
        empty_answer = 0.0 if canonical else 1.0
        non_answer_penalty = 1.0 if has_direct_candidate and non_answer_flags[idx] else 0.0

        score = (
            args.posterior_agreement_weight * posterior_answer_mass
            - args.posterior_conflict_weight * posterior_conflict_mass
            + args.posterior_route_weight * weights[idx]
            + args.posterior_evidence_weight * branch_has_evidence
            - args.posterior_empty_penalty * empty_answer
            - args.posterior_non_answer_penalty * non_answer_penalty
        )

        scores.append(
            {
                "candidate_index": idx,
                "modality": candidate.get("modality", ""),
                "weight": float(weights[idx]),
                "canonical_answer": canonical,
                "posterior_answer_mass": float(posterior_answer_mass),
                "posterior_conflict_mass": float(posterior_conflict_mass),
                "branch_has_evidence": float(branch_has_evidence),
                "empty_answer": float(empty_answer),
                "non_answer_penalty": float(non_answer_penalty),
                "score": float(score),
            }
        )

    return weights, canonicals, scores


def _build_posterior_verifier_query(target, row, candidates, scores, args):
    answer_mode = BASE._answer_mode_for_target(target)
    allowed_answers = []
    if answer_mode == "mcq_letter":
        for candidate in candidates:
            choice = BASE._extract_choice_from_prediction(candidate.get("response", ""))
            if choice and choice not in allowed_answers:
                allowed_answers.append(choice)

    candidate_sections = []
    for idx, candidate in enumerate(candidates):
        score = scores[idx]
        evidence_preview = _candidate_evidence_preview(
            candidate,
            max_chars=args.posterior_evidence_max_chars,
        )
        candidate_sections.append(
            "\n".join(
                [
                    f"Candidate {idx}",
                    f"Modality: {candidate.get('modality', '')}",
                    f"Bayes posterior branch weight: {score['weight']:.4f}",
                    f"Posterior answer mass: {score['posterior_answer_mass']:.4f}",
                    f"Posterior conflict mass: {score['posterior_conflict_mass']:.4f}",
                    f"Non-answer/refusal penalty: {score['non_answer_penalty']:.1f}",
                    f"Risk-selection score: {score['score']:.4f}",
                    f"Answer: {str(candidate.get('response', '')).strip()}",
                    f"Evidence preview:\n{evidence_preview}",
                ]
            )
        )

    if allowed_answers:
        answer_constraint = (
            "Allowed final answers are only the candidate letters: "
            + ", ".join(allowed_answers)
            + ". Do not output any other letter.\n"
        )
    else:
        answer_constraint = ""

    if int(getattr(args, "posterior_verifier_choice_only", 0)) == 1:
        answer_constraint += (
            "Output exactly one candidate label from the list, for example: Candidate 0. "
            "Do not write a new answer and do not explain.\n"
        )

    return (
        "You are a posterior predictive verifier for multimodal retrieval-augmented QA.\n"
        "The retrieval modality is latent. Each candidate answer was generated from one "
        "possible modality branch, and the Bayes posterior branch weight is only a prior, "
        "not an automatic truth label.\n"
        "Choose the final answer that best answers the question and is best supported by "
        "its branch evidence. Penalize answers that are off-topic, unsupported, or say "
        "there is no answer when another candidate is clearly supported.\n"
        "If one candidate refuses to answer because the retrieved image or evidence is "
        "off-topic, and another candidate gives a direct answer to the question, prefer "
        "the direct answer unless it is clearly contradicted.\n"
        "Use posterior weights to break ties, but do not blindly choose the highest-weight "
        "candidate when its answer conflicts with the question or evidence.\n"
        f"{answer_constraint}"
        "Do not explain your reasoning.\n"
        f"{_answer_instruction(target)}\n\n"
        f"Question:\n{row['question']}\n\n"
        "Candidate answers and evidence:\n"
        + "\n\n".join(candidate_sections)
    )


def _posterior_predictive_fusion(target, row, candidates, model, args):
    """
    Approximate posterior predictive generation:

      p(y | x) ~= sum_z q(z | x) p(y | x, e_z)

    The score estimates posterior answer mass and conflict risk. When candidates
    disagree, an LLM verifier receives the Bayes posterior weights plus branch
    evidence previews, so uncertainty is used at generation time instead of
    collapsing to the top-1 route.
    """
    if not candidates:
        return "", [], None, "posterior_empty", ""

    weights, canonicals, scores = _score_posterior_candidates(target, candidates, args)

    trivial_response, trivial_mode = BASE._resolve_trivial_fusion(target, candidates)
    if trivial_response:
        best_idx = max(
            range(len(candidates)),
            key=lambda idx: (scores[idx]["score"], weights[idx], -idx),
        )
        return trivial_response, scores, best_idx, f"posterior_{trivial_mode}", ""

    best_idx = max(
        range(len(candidates)),
        key=lambda idx: (scores[idx]["score"], weights[idx], -idx),
    )
    best_candidate = candidates[best_idx]
    best_canonical = canonicals[best_idx]
    deterministic_response = ""

    if best_canonical:
        if str(target or "").lower() == "mmlu":
            deterministic_response = best_canonical
        else:
            deterministic_response = str(best_candidate.get("response", "")).strip()
    else:
        deterministic_response = BASE._weighted_vote_response(target, candidates)

    if args.posterior_verifier == 0:
        mode = "posterior_predictive_score" if best_canonical else "posterior_fallback_weighted_pick"
        return deterministic_response, scores, best_idx, mode, ""

    verifier_query = _build_posterior_verifier_query(target, row, candidates, scores, args)
    verifier_response = model.inference(
        verifier_query,
        max_new_tokens=args.posterior_verifier_max_new_tokens,
    )
    verifier_response = str(verifier_response or "").strip()

    if int(getattr(args, "posterior_verifier_choice_only", 0)) == 1:
        verifier_idx = _extract_verifier_candidate_index(verifier_response, len(candidates))
        if verifier_idx is not None:
            verifier_candidate = candidates[verifier_idx]
            verifier_answer = str(verifier_candidate.get("response", "")).strip()
            if verifier_answer:
                return (
                    verifier_answer,
                    scores,
                    verifier_idx,
                    "posterior_verifier_choice_only",
                    verifier_response,
                )
        return (
            deterministic_response,
            scores,
            best_idx,
            "posterior_verifier_choice_only_fallback_score",
            verifier_response,
        )

    if str(target or "").lower() == "mmlu":
        verifier_choice = BASE._extract_choice_from_prediction(verifier_response)
        candidate_choices = {
            BASE._extract_choice_from_prediction(candidate.get("response", ""))
            for candidate in candidates
        }
        candidate_choices = {choice for choice in candidate_choices if choice}
        if verifier_choice and verifier_choice in candidate_choices:
            return verifier_choice, scores, best_idx, "posterior_verifier_llm", verifier_response
    elif verifier_response:
        return verifier_response, scores, best_idx, "posterior_verifier_llm", verifier_response

    return deterministic_response, scores, best_idx, "posterior_verifier_fallback_score", verifier_response


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
        choices=BASE.SUPPORTED_TARGETS,
    )
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--nframes", type=str, default="1")
    parser.add_argument("--route_dir", type=str, default="route/results_vib")
    parser.add_argument("--output_root", type=str, default="eval/results_bayes_vib_posterior")
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

    parser.add_argument("--online_update", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--penalty", type=float, default=0.5)
    parser.add_argument("--spread", type=float, default=0.25)
    parser.add_argument("--use_penalty_update", type=int, default=1, choices=[0, 1])

    parser.add_argument("--soft_top_n", type=int, default=2)
    parser.add_argument(
        "--soft_weight_mode",
        type=str,
        default="theta",
        choices=["theta", "utility"],
    )
    parser.add_argument("--soft_store_candidates", type=int, default=1, choices=[0, 1])
    parser.add_argument("--selective_no_retrieval", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--selective_no_targets",
        type=str,
        default="",
        help="Targets where high-confidence no-retrieval can collapse to a single no branch.",
    )
    parser.add_argument("--selective_no_theta_min", type=float, default=0.55)
    parser.add_argument("--selective_no_utility_margin", type=float, default=0.03)
    parser.add_argument(
        "--selective_no_uncertainty_max",
        type=float,
        default=-1.0,
        help="Disable when < 0. Otherwise force no only below this route uncertainty.",
    )
    parser.add_argument("--selective_single_branch", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--selective_single_branch_targets",
        type=str,
        default="",
        help="Targets where confident routes execute only the primary branch.",
    )
    parser.add_argument("--selective_single_branch_theta_min", type=float, default=0.62)
    parser.add_argument("--selective_single_branch_utility_margin", type=float, default=0.08)
    parser.add_argument(
        "--selective_single_branch_uncertainty_max",
        type=float,
        default=-1.0,
        help="Disable when < 0. Otherwise use one branch only below this route uncertainty.",
    )
    parser.add_argument("--selective_include_no_candidate", type=int, default=0, choices=[0, 1])
    parser.add_argument(
        "--selective_include_no_targets",
        type=str,
        default="",
        help="Targets where uncertain soft routing should include no-retrieval as a cheap candidate.",
    )
    parser.add_argument(
        "--selective_include_no_utility_gap_max",
        type=float,
        default=-1.0,
        help="Disable when < 0. Otherwise include no only if its utility is within this gap.",
    )

    parser.add_argument("--base_route_dir", type=str, default="route/results_bayes_probs")
    parser.add_argument("--hybrid_use_base", type=int, default=1, choices=[0, 1])
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
    parser.add_argument("--dynamic_tau_min", type=float, default=0.35)
    parser.add_argument("--dynamic_tau_max", type=float, default=1.15)
    parser.add_argument("--evidence_saturation", type=float, default=8.0)

    parser.add_argument("--posterior_agreement_weight", type=float, default=1.0)
    parser.add_argument("--posterior_conflict_weight", type=float, default=0.35)
    parser.add_argument("--posterior_route_weight", type=float, default=0.15)
    parser.add_argument("--posterior_evidence_weight", type=float, default=0.05)
    parser.add_argument("--posterior_empty_penalty", type=float, default=1.0)
    parser.add_argument("--posterior_non_answer_penalty", type=float, default=0.85)
    parser.add_argument("--posterior_verifier", type=int, default=1, choices=[0, 1])
    parser.add_argument("--posterior_verifier_choice_only", type=int, default=0, choices=[0, 1])
    parser.add_argument("--posterior_verifier_max_new_tokens", type=int, default=64)
    parser.add_argument("--posterior_evidence_max_chars", type=int, default=1200)
    parser.add_argument(
        "--posterior_safe_fallback_file",
        type=str,
        default="",
        help="Optional VIB+Bayes result file used as a fallback when posterior generation is not confident.",
    )
    parser.add_argument(
        "--posterior_accept_max_score_gap",
        type=float,
        default=-1.0,
        help="Enable safe fallback when >= 0: accept posterior only if top score gap is below this value.",
    )
    parser.add_argument(
        "--posterior_accept_min_score_gap",
        type=float,
        default=-1.0,
        help="Enable safe fallback when >= 0: accept posterior only if top score gap is at least this value.",
    )
    parser.add_argument(
        "--posterior_accept_modality_pairs",
        type=str,
        default="",
        help=(
            "Optional semicolon-separated ordered modality pairs allowed to accept "
            "posterior output, e.g. 'no,paragraph;paragraph,no'. Other pairs fall "
            "back to posterior_safe_fallback_file when provided."
        ),
    )
    args = parser.parse_args()
    posterior_accept_modality_pairs = _parse_modality_pairs(
        args.posterior_accept_modality_pairs
    )
    selective_no_targets = _parse_target_filter(args.selective_no_targets)
    selective_single_branch_targets = _parse_target_filter(args.selective_single_branch_targets)
    selective_include_no_targets = _parse_target_filter(args.selective_include_no_targets)

    print(
        f"[VIB-BAYES POSTERIOR GEN EVAL] model={args.model_path}, "
        f"router={args.router_model}, target={args.target}, top_k={args.top_k}, "
        f"alpha={args.alpha}, nframes={args.nframes}, soft_top_n={args.soft_top_n}, "
        f"soft_weight_mode={args.soft_weight_mode}, vib_prob_field={args.vib_prob_field}, "
        f"hybrid_use_base={bool(args.hybrid_use_base)}"
    )

    alpha_prior = BASE._parse_float_list(args.alpha_prior, expected_len=len(MODALITIES))
    modality_costs = BASE._parse_float_list(args.modality_costs, expected_len=len(MODALITIES))
    alpha_prior_by_target = BASE._parse_prior_map(args.alpha_prior_by_target)

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

    model = BASE.ModelLoader(args.model_path)

    route_file = os.path.join(args.route_dir, args.router_model, f"{args.target}.json")
    if not os.path.exists(route_file):
        raise FileNotFoundError(
            f"Route result file not found: {route_file}\n"
            "Please run VIB routing first."
        )

    with open(route_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    model_name = args.model_path.split("/")[-1]
    nframes_tag = args.nframes.replace(",", "_").replace(":", "")
    bayes_suffix = "bayes"
    if args.bayes_tag.strip():
        bayes_suffix = f"bayes_{BASE._sanitize_tag(args.bayes_tag)}"
    elif args.soft_top_n > 1:
        bayes_suffix = f"bayes_vibposterior_softtop{args.soft_top_n}_{args.soft_weight_mode}"

    output_dir = os.path.join(args.output_root, model_name, args.router_model)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{args.target}_top{args.top_k}_{args.alpha}_{nframes_tag}_{bayes_suffix}.json",
    )
    partial_file = output_file + ".partial"
    partial_save_every = max(0, _get_env_int("EVAL_PARTIAL_SAVE_EVERY", 25))
    gc_every = max(0, _get_env_int("EVAL_GC_EVERY", 10))
    max_new_rows = max(0, _get_env_int("EVAL_MAX_NEW_ROWS", 0))
    processed_new_rows = 0
    stopped_early = False

    if _get_env_bool("EVAL_RESUME_PARTIAL", True) and os.path.exists(partial_file):
        with open(partial_file, "r", encoding="utf-8") as f:
            partial_data = json.load(f)
        partial_by_index = {
            str(partial_row.get("index", idx)): partial_row
            for idx, partial_row in enumerate(partial_data)
            if partial_row.get("response") not in (None, "")
            and partial_row.get("retrieval_bayes") is not None
        }
        resumed = 0
        for idx, row in enumerate(data):
            key = str(row.get("index", idx))
            if key in partial_by_index:
                data[idx] = partial_by_index[key]
                resumed += 1
        if resumed:
            print(f"[INFO] Resumed {resumed} rows from partial: {partial_file}")

    safe_fallback_rows = _load_safe_fallback_rows(args.posterior_safe_fallback_file)
    safe_fallback_map = {
        str(fallback_row.get("index", idx)): fallback_row
        for idx, fallback_row in enumerate(safe_fallback_rows)
    }

    base_route_map = {}
    if args.hybrid_use_base == 1:
        base_route_map = _load_base_route_map(
            base_route_dir=args.base_route_dir,
            router_model=args.router_model,
            target=args.target,
        )

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
                        textfeats_path=BASE.get_text_feature_paths(args.target, modality),
                    )
                retrieved, _ = retriever_paragraph.retrieve(current_row["index"], top_k=args.top_k)
            else:
                _clear_inactive_retrievers("document", single_retriever_cache)
                if retriever_document is None:
                    retriever_document = BGETextRetriever(
                        queryfeats_path=os.path.join(args.query_bge_dir, f"{args.target}.pkl"),
                        textfeats_path=BASE.get_text_feature_paths(args.target, modality),
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
                    imgfeats_path, imgcapfeats_path = BASE.get_image_feature_paths(args.target)
                    retriever_image = InternImgRetriever(
                        queryfeats_path=query_img_feat,
                        imgfeats_path=imgfeats_path,
                        imgcapfeats_path=imgcapfeats_path,
                        alpha=args.alpha,
                    )

            candidate_images = current_row.get("candidate_images") if args.target == "visual_rag" else None
            retrieved, _ = retriever_image.retrieve(current_row["index"], top_k=args.top_k, candidate_ids=candidate_images)
            response = model.inference(
                formatted_query,
                retrieved_images=retrieved,
                max_new_tokens=128,
            )
            return response, retrieved

        raise ValueError(f"Invalid modality after Bayes decision: {modality}")

    for row_idx, row in enumerate(tqdm(
        data,
        desc=(
            f"VIB-Bayes posterior-generating {args.target} with "
            f"{args.model_path} + {args.router_model}"
        ),
    )):
        if (
            row.get("response") not in (None, "")
            and row.get("retrieval_bayes") is not None
        ):
            continue

        query = BASE.reformat(row)
        original_modality = str(row.get("retrieval", "error")).lower()
        retrieval_conf = row.get("retrieval_conf", None)
        vib_uncertainty = row.get("retrieval_uncertainty", None)
        vib_probs, vib_probs_field = _select_vib_probs(row, args.vib_prob_field)
        vib_alpha = _parse_float_vector(row, "retrieval_alpha")
        base_probs, base_row = _base_probs_for_row(base_route_map, row)

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

        retrieved = []
        response = ""
        soft_modalities = [modality]
        soft_weights = [1.0]
        soft_fusion_mode_used = "hard"
        candidates = []
        retrieved_union = []

        selective_mode_used = "disabled"
        selective_details = {}
        force_no, force_no_details = _should_force_no_retrieval(
            args.target,
            decision,
            args,
            selective_no_targets,
        )
        if force_no:
            modality = "no"
            soft_modalities = ["no"]
            soft_weights = [1.0]
            selective_mode_used = "force_no_single_branch"
            selective_details = force_no_details
        elif args.soft_top_n > 1:
            single_branch, single_branch_details = _should_use_single_branch(
                args.target,
                decision,
                modality,
                args,
                selective_single_branch_targets,
            )
            if single_branch:
                soft_modalities = [modality]
                soft_weights = [1.0]
                selective_mode_used = "confident_single_branch"
                selective_details = single_branch_details
            else:
                soft_modalities, soft_weights = BASE._select_soft_modalities(
                    decision=decision,
                    target=args.target,
                    soft_top_n=args.soft_top_n,
                    weight_mode=args.soft_weight_mode,
                    primary_modality=modality,
                )
                updated_modalities, include_no_details = _maybe_include_no_candidate(
                    args.target,
                    decision,
                    soft_modalities,
                    args,
                    selective_include_no_targets,
                )
                if updated_modalities != soft_modalities:
                    soft_modalities = updated_modalities
                    soft_weights = _weights_for_modalities(
                        decision,
                        soft_modalities,
                        args.soft_weight_mode,
                    )
                    selective_mode_used = "soft_top_n_include_no_candidate"
                    selective_details = include_no_details
                else:
                    selective_mode_used = "soft_top_n"
                    selective_details = include_no_details or single_branch_details or force_no_details

        if len(soft_modalities) <= 1:
            response, retrieved = execute_modality(row, query, modality)
        else:
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

            response, posterior_scores, selected_idx, soft_fusion_mode_used, verifier_raw = (
                _posterior_predictive_fusion(args.target, row, candidates, model, args)
            )
            if selected_idx is not None and 0 <= selected_idx < len(candidates):
                retrieved = list(candidates[selected_idx]["retrieved"])
            else:
                retrieved = list(candidates[0]["retrieved"])

            row["retrieval_bayes_soft_enabled"] = True
            row["retrieval_bayes_soft_modalities"] = soft_modalities
            row["retrieval_bayes_soft_weights"] = soft_weights
            row["retrieval_bayes_soft_weight_mode"] = args.soft_weight_mode
            row["retrieval_bayes_soft_fusion_mode"] = soft_fusion_mode_used
            row["retrieval_bayes_soft_top_n"] = len(soft_modalities)
            row["retrieval_bayes_soft_retrieved_union"] = retrieved_union
            row["retrieval_bayes_posterior_generation_scores"] = posterior_scores
            row["retrieval_bayes_posterior_generation_selected_index"] = selected_idx
            if verifier_raw:
                row["retrieval_bayes_posterior_generation_verifier_response"] = verifier_raw
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
                "accept_min_score_gap": args.posterior_accept_min_score_gap,
                "accept_max_score_gap": args.posterior_accept_max_score_gap,
                "accept_modality_pairs": args.posterior_accept_modality_pairs,
            }
            if args.soft_store_candidates == 1:
                row["retrieval_bayes_soft_candidates"] = candidates

            score_gap = _posterior_score_gap(posterior_scores)
            row["retrieval_bayes_posterior_score_gap"] = score_gap
            row["retrieval_bayes_posterior_safe_fallback"] = False
            if safe_fallback_map and (
                args.posterior_accept_max_score_gap >= 0
                or args.posterior_accept_min_score_gap >= 0
            ):
                fallback_key = str(row.get("index", row_idx))
                fallback_row = safe_fallback_map.get(fallback_key)
                fallback_reasons = []
                if score_gap is None:
                    fallback_reasons.append("missing_score_gap")
                else:
                    if (
                        args.posterior_accept_max_score_gap >= 0
                        and score_gap >= args.posterior_accept_max_score_gap
                    ):
                        fallback_reasons.append("score_gap_above_max")
                    if (
                        args.posterior_accept_min_score_gap >= 0
                        and score_gap < args.posterior_accept_min_score_gap
                    ):
                        fallback_reasons.append("score_gap_below_min")
                    if posterior_accept_modality_pairs:
                        modality_pair = tuple(
                            str(modality_item).lower()
                            for modality_item in soft_modalities[:2]
                        )
                        if modality_pair not in posterior_accept_modality_pairs:
                            fallback_reasons.append("modality_pair_not_allowed")
                should_fallback = bool(fallback_reasons)
                if fallback_row is not None and should_fallback:
                    response = fallback_row.get("response", response)
                    retrieved = fallback_row.get("retrieved", retrieved)
                    row["retrieval_bayes_posterior_safe_fallback"] = True
                    row["retrieval_bayes_posterior_safe_fallback_reason"] = (
                        ",".join(fallback_reasons)
                    )
                    row["retrieval_bayes_posterior_safe_fallback_source"] = (
                        args.posterior_safe_fallback_file
                    )

        if len(soft_modalities) <= 1:
            row["retrieval_bayes_soft_enabled"] = False

        row["retrieved"] = retrieved
        row["response"] = response

        row["retrieval_original"] = original_modality
        row["retrieval_bayes"] = modality
        row["retrieval_bayes_selective_mode"] = selective_mode_used
        row["retrieval_bayes_selective_details"] = selective_details
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

        processed_new_rows += 1

        if partial_save_every and (row_idx + 1) % partial_save_every == 0:
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Saved partial results to: {partial_file}")
        if gc_every and (row_idx + 1) % gc_every == 0:
            gc.collect()
        if max_new_rows and processed_new_rows >= max_new_rows:
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            print(f"[INFO] Reached EVAL_MAX_NEW_ROWS={max_new_rows}; saved partial results to: {partial_file}")
            stopped_early = True
            break

    if stopped_early:
        print(f"[INFO] Stopped after {processed_new_rows} new rows. Re-run the same command to continue from partial.")
        raise SystemExit(0)

    model_name = args.model_path.split("/")[-1]
    nframes_tag = args.nframes.replace(",", "_").replace(":", "")
    bayes_suffix = "bayes"
    if args.bayes_tag.strip():
        bayes_suffix = f"bayes_{BASE._sanitize_tag(args.bayes_tag)}"
    elif args.soft_top_n > 1:
        bayes_suffix = f"bayes_vibposterior_softtop{args.soft_top_n}_{args.soft_weight_mode}"

    output_dir = os.path.join(args.output_root, model_name, args.router_model)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir,
        f"{args.target}_top{args.top_k}_{args.alpha}_{nframes_tag}_{bayes_suffix}.json",
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved VIB-Bayes posterior generation results to: {output_file}")
    if os.path.exists(partial_file):
        os.remove(partial_file)

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
        "innovation_stack": [
            "vib_uncertainty_aware_router_evidence",
            "bayesian_dirichlet_task_prior_posterior_routing",
            "posterior_predictive_generation_over_latent_modalities",
        ],
        "alpha_prior_init": alpha_prior,
        "alpha_prior_by_target": alpha_prior_by_target,
        "tau": args.tau,
        "beta_cost": args.beta_cost,
        "modality_costs": modality_costs,
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
        "soft_fusion_mode": "posterior_predictive",
        "soft_store_candidates": bool(args.soft_store_candidates),
        "selective_no_retrieval": bool(args.selective_no_retrieval),
        "selective_no_targets": args.selective_no_targets,
        "selective_no_theta_min": args.selective_no_theta_min,
        "selective_no_utility_margin": args.selective_no_utility_margin,
        "selective_no_uncertainty_max": args.selective_no_uncertainty_max,
        "selective_single_branch": bool(args.selective_single_branch),
        "selective_single_branch_targets": args.selective_single_branch_targets,
        "selective_single_branch_theta_min": args.selective_single_branch_theta_min,
        "selective_single_branch_utility_margin": args.selective_single_branch_utility_margin,
        "selective_single_branch_uncertainty_max": args.selective_single_branch_uncertainty_max,
        "selective_include_no_candidate": bool(args.selective_include_no_candidate),
        "selective_include_no_targets": args.selective_include_no_targets,
        "selective_include_no_utility_gap_max": args.selective_include_no_utility_gap_max,
        "hybrid_use_base": bool(args.hybrid_use_base),
        "vib_prob_field": args.vib_prob_field,
        "vib_uncertainty_low": args.vib_uncertainty_low,
        "vib_uncertainty_high": args.vib_uncertainty_high,
        "vib_weight_low": args.vib_weight_low,
        "vib_weight_high": args.vib_weight_high,
        "dynamic_tau_min": args.dynamic_tau_min,
        "dynamic_tau_max": args.dynamic_tau_max,
        "evidence_saturation": args.evidence_saturation,
        "posterior_agreement_weight": args.posterior_agreement_weight,
        "posterior_conflict_weight": args.posterior_conflict_weight,
        "posterior_route_weight": args.posterior_route_weight,
        "posterior_evidence_weight": args.posterior_evidence_weight,
        "posterior_empty_penalty": args.posterior_empty_penalty,
        "posterior_non_answer_penalty": args.posterior_non_answer_penalty,
        "posterior_verifier": bool(args.posterior_verifier),
        "posterior_verifier_choice_only": bool(args.posterior_verifier_choice_only),
        "posterior_verifier_max_new_tokens": args.posterior_verifier_max_new_tokens,
        "posterior_evidence_max_chars": args.posterior_evidence_max_chars,
        "posterior_safe_fallback_file": args.posterior_safe_fallback_file,
        "posterior_accept_max_score_gap": args.posterior_accept_max_score_gap,
        "posterior_accept_min_score_gap": args.posterior_accept_min_score_gap,
        "posterior_accept_modality_pairs": args.posterior_accept_modality_pairs,
    }
    meta_file = output_file.replace(".json", ".meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved VIB-Bayes posterior generation meta to: {meta_file}")
