import argparse
import gc
import importlib
import json
import math
import os
import pickle
import random
import re
import string
from collections import Counter

from tqdm import tqdm

from retrieve.retrieve_image import InternImgRetriever
from retrieve.retrieve_image_bge import BGEImageRetriever
from retrieve.retrieve_text import BGETextRetriever
from route.bayes_dirichlet_router import BayesDirichletRouter, MODALITIES


random.seed(42)


def _sanitize_tag(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text.strip("_")


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


def _has_response(row):
    response = row.get("response")
    return response is not None and str(response).strip() != ""


def _row_key(row):
    return (
        str(row.get("source", "")),
        str(row.get("index", "")),
        str(row.get("question", "")),
    )


def _merge_existing_results(data, existing):
    existing_by_key = { _row_key(row): row for row in existing if _has_response(row) }
    restored = 0
    for idx, row in enumerate(data):
        old_row = existing_by_key.get(_row_key(row))
        if old_row is None:
            continue
        merged = dict(row)
        merged.update(old_row)
        data[idx] = merged
        restored += 1
    return restored


def _save_json_atomic(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


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


def _normalize_answer(s: str) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(ground_truth).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _flatten_to_str_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        out = []
        for v in value:
            out.extend(_flatten_to_str_list(v))
        return [x for x in out if x != ""]
    if isinstance(value, dict):
        out = []
        for v in value.values():
            out.extend(_flatten_to_str_list(v))
        return [x for x in out if x != ""]
    return [str(value).strip()]


def _extract_refs(item):
    keys = [
        "answers",
        "answer",
        "gt_answers",
        "gt_answer",
        "ground_truths",
        "ground_truth",
        "gold_answers",
        "gold_answer",
        "gold",
        "label",
        "labels",
        "target",
        "targets",
        "reference",
        "references",
    ]
    refs = []
    for k in keys:
        if k in item:
            refs.extend(_flatten_to_str_list(item[k]))
    refs = [x.strip() for x in refs if str(x).strip() != ""]
    # Keep order but deduplicate
    seen = set()
    out = []
    for x in refs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _extract_choice_from_prediction(text: str):
    text = str(text or "").strip().upper()
    patterns = [
        r"ANSWER\s*[:：]?\s*\(?([A-E])\)?",
        r"OPTION\s*[:：]?\s*\(?([A-E])\)?",
        r"CHOICE\s*[:：]?\s*\(?([A-E])\)?",
        r"\(([A-E])\)",
        r"\b([A-E])\b",
    ]
    for pat in patterns:
        matches = re.findall(pat, text)
        if matches:
            return matches[-1]
    return None


def _safe_int_to_choice(v):
    letters = "ABCDE"
    if isinstance(v, int):
        if 0 <= v < len(letters):
            return letters[v]
        return None
    if isinstance(v, str):
        t = v.strip().upper()
        if t in letters:
            return t
        m = re.search(r"\b([A-E])\b", t)
        if m:
            return m.group(1)
        if t.isdigit():
            idx = int(t)
            if 0 <= idx < len(letters):
                return letters[idx]
    return None


def _extract_mmlu_gold(item):
    candidate_keys = [
        "answer",
        "gt_answer",
        "gold",
        "gold_answer",
        "label",
        "correct_answer",
        "target",
        "answer_idx",
        "label_idx",
        "gold_idx",
    ]
    for k in candidate_keys:
        if k in item:
            gold = _safe_int_to_choice(item[k])
            if gold is not None:
                return gold
    return None


def _light_reward(target, item, prediction):
    target = str(target or "").lower()
    prediction = str(prediction or "")

    if target == "mmlu":
        pred = _extract_choice_from_prediction(prediction)
        gold = _extract_mmlu_gold(item)
        if pred is None or gold is None:
            return 0.0
        return float(pred == gold)

    refs = _extract_refs(item)
    if not refs:
        return 0.0
    # For short/long answers, use max token-level F1 as dense reward.
    return max(_f1_score(prediction, ref) for ref in refs)


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_module = self._load_model_module()
        self.model, self.processor, self.tokenizer = self.model_module.load_model(model_path)

    def _load_model_module(self):
        model_lower = self.model_path.lower()

        api_prefixes = (
            "qwen-api:",
            "dashscope:",
            "openai:",
            "gpt:",
            "dmxapi:",
            "deepseek:",
            "glm:",
            "zhipu:",
            "openai-compatible:",
        )
        api_model_names = ("gpt-", "deepseek-", "glm-")
        if model_lower.startswith(api_prefixes) or model_lower.startswith(api_model_names):
            module_name = "qwen_api"
        elif "internvl" in model_lower:
            module_name = "internvl2_5"
        elif "qwen" in model_lower and "vl" in model_lower:
            module_name = "qwen2_5_vl"
        elif "phi" in model_lower and "vision" in model_lower:
            module_name = "phi_3_5_vision"
        else:
            raise ValueError(f"Unsupported model type: {self.model_path}")

        return importlib.import_module(f"utils.models.{module_name}")

    def inference(self, query, **kwargs):
        return self.model_module.inference(
            self.model, self.processor, self.tokenizer, query, **kwargs
        )


def reformat(row):
    query, data_type = row["question"], row["source"]

    if data_type in {"mmlu", "truthfulqa"}:
        choices = row.get("choices") or []
        if choices:
            last_letter = chr(ord("A") + min(len(choices), 26) - 1)
            return f"{query} Please respond with only a single letter (A-{last_letter})."
        return f"{query} Please respond with only a single letter."
    if data_type in ["natural_questions", "hotpotqa", "squad", "triviaqa"]:
        return f"{query} Please respond with only the exact answer."
    if data_type in {"webqa", "lara", "visual_rag"}:
        return f"{query} Please respond in a complete sentence."
    raise ValueError(f"Invalid data type: {data_type}")


SUPPORTED_TARGETS = [
    "mmlu",
    "squad",
    "natural_questions",
    "hotpotqa",
    "webqa",
    "truthfulqa",
    "triviaqa",
    "lara",
    "visual_rag",
]


def get_text_feature_paths(target, modality):
    target = str(target or "").lower()
    modality = str(modality or "").lower()
    if modality == "paragraph":
        if target in {"triviaqa", "lara"}:
            return [f"eval/features/text/{target}.pkl"]
        return [
            "eval/features/text/squad.pkl",
            "eval/features/text/natural_questions.pkl",
        ]
    if modality == "document":
        if target == "lara":
            return ["eval/features/text/lara.pkl"]
        if target == "hotpotqa":
            return [os.environ.get("HOTPOTQA_TEXT_FEATS", "eval/features/text/hotpotqa.pkl")]
        if target in {"squad", "natural_questions"}:
            return get_text_feature_paths(target, "paragraph")
        return ["eval/features/text/hotpotqa.pkl"]
    raise ValueError(f"Invalid text modality: {modality}")


def get_image_feature_paths(target):
    if str(target or "").lower() == "visual_rag":
        img_path = os.environ.get(
            "VISUAL_RAG_IMAGE_FEATS",
            "eval/features/image/visual_rag.pkl",
        )
        imgcap_path = os.environ.get(
            "VISUAL_RAG_IMGCAP_FEATS",
            "eval/features/image/visual_rag_imgcap.pkl",
        )
        return [img_path], [imgcap_path] if imgcap_path else None
    return ["eval/features/image/webqa.pkl"], ["eval/features/image/webqa_imgcap.pkl"]


def _parse_float_list(csv_like: str, expected_len: int):
    parts = [p.strip() for p in str(csv_like).split(",") if p.strip() != ""]
    vals = [float(x) for x in parts]
    if len(vals) != expected_len:
        raise ValueError(
            f"Expected {expected_len} comma-separated values, got {len(vals)}: {csv_like}"
        )
    return vals


def _parse_prior_map(text: str):
    """
    Parse target-specific priors:
      "mmlu=8,1,1,0.5;squad=1,8,2,0.5"
    """
    text = str(text or "").strip()
    if not text:
        return {}
    mapping = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid prior map entry: {part}")
        key, vals = part.split("=", 1)
        key = key.strip().lower()
        mapping[key] = _parse_float_list(vals, expected_len=len(MODALITIES))
    return mapping


def _parse_scalar_map(text: str):
    """
    Parse target-specific scalar values:
      "webqa=2.0;hotpotqa=1.2"
    """
    text = str(text or "").strip()
    if not text:
        return {}
    mapping = {}
    for part in text.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid scalar map entry: {part}")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        mapping[key] = float(value.strip())
    return mapping


def _apply_temperature_to_probs(probs, temperature: float):
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"router_probs temperature must be > 0, got {temperature}")

    probs = [max(1e-8, float(x)) for x in probs]
    total = sum(probs)
    if total <= 0:
        return None
    probs = [x / total for x in probs]

    if abs(temperature - 1.0) < 1e-8:
        return probs

    logits = [math.log(x) / temperature for x in probs]
    max_logit = max(logits)
    exp_values = [math.exp(x - max_logit) for x in logits]
    exp_total = sum(exp_values)
    if exp_total <= 0:
        return None
    return [x / exp_total for x in exp_values]


def _blend_probs_with_original_retrieval(probs, original_retrieval: str, strength: float):
    strength = float(strength)
    if strength < 0 or strength > 1:
        raise ValueError(f"router_probs blend strength must be in [0, 1], got {strength}")
    if strength <= 0:
        return probs

    original_retrieval = str(original_retrieval or "").strip().lower()
    if original_retrieval not in MODALITIES:
        return probs

    one_hot = [0.0] * len(MODALITIES)
    one_hot[MODALITIES.index(original_retrieval)] = 1.0
    blended = [
        (1.0 - strength) * float(prob) + strength * float(anchor)
        for prob, anchor in zip(probs, one_hot)
    ]
    total = sum(blended)
    if total <= 0:
        return probs
    return [value / total for value in blended]


def _parse_router_probs(
    row,
    temperature: float = 1.0,
    original_retrieval: str = "",
    blend_with_original: float = 0.0,
):
    # Optional explicit vector support:
    # "retrieval_probs": [p_no, p_paragraph, p_document, p_image]
    probs = row.get("retrieval_probs")
    if probs is None:
        return None
    if not isinstance(probs, list) or len(probs) != len(MODALITIES):
        return None
    try:
        probs = [float(x) for x in probs]
    except (TypeError, ValueError):
        return None
    s = sum(max(0.0, x) for x in probs)
    if s <= 0:
        return None
    probs = [max(0.0, x) / s for x in probs]
    probs = _apply_temperature_to_probs(probs, temperature=temperature)
    if probs is None:
        return None
    return _blend_probs_with_original_retrieval(
        probs,
        original_retrieval=original_retrieval,
        strength=blend_with_original,
    )


def _answer_mode_for_target(target: str) -> str:
    target = str(target or "").lower()
    if target in {"mmlu", "truthfulqa"}:
        return "mcq_letter"
    if target in {"squad", "natural_questions", "hotpotqa", "triviaqa"}:
        return "exact_short"
    return "sentence"


def _canonical_prediction(target: str, prediction: str) -> str:
    target = str(target or "").lower()
    prediction = str(prediction or "").strip()
    if not prediction:
        return ""
    if target in {"mmlu", "truthfulqa"}:
        return _extract_choice_from_prediction(prediction) or ""
    return _normalize_answer(prediction)


def _resolve_trivial_fusion(target: str, candidates):
    non_empty = []
    for candidate in candidates:
        canonical = _canonical_prediction(target, candidate.get("response", ""))
        if canonical:
            non_empty.append((candidate, canonical))

    if len(non_empty) == 1:
        return non_empty[0][0]["response"], "single_non_empty"

    canonical_values = {canonical for _, canonical in non_empty}
    if non_empty and len(canonical_values) == 1:
        best_candidate = max(non_empty, key=lambda item: float(item[0].get("weight", 0.0)))[0]
        return best_candidate["response"], "consensus"

    return "", ""


def _weighted_vote_response(target: str, candidates):
    if not candidates:
        return ""

    if str(target or "").lower() in {"mmlu", "truthfulqa"}:
        votes = {}
        for candidate in candidates:
            letter = _extract_choice_from_prediction(candidate.get("response", ""))
            if not letter:
                continue
            votes[letter] = votes.get(letter, 0.0) + float(candidate.get("weight", 0.0))
        if votes:
            return max(votes.items(), key=lambda item: (item[1], item[0]))[0]

    return max(candidates, key=lambda item: float(item.get("weight", 0.0))).get("response", "")


def _softmax(values):
    values = [float(v) for v in values]
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    if total <= 0:
        return [1.0 / len(values)] * len(values)
    return [v / total for v in exps]


def _select_soft_modalities(
    decision,
    target: str,
    soft_top_n: int,
    weight_mode: str,
    primary_modality: str,
):
    target = str(target or "").lower()
    soft_top_n = max(1, int(soft_top_n))

    valid_modalities = ["no", "paragraph", "document"]
    if target in {"webqa", "visual_rag"}:
        valid_modalities.append("image")

    theta = {modality: float(value) for modality, value in zip(MODALITIES, decision["theta"])}
    utility = {modality: float(value) for modality, value in zip(MODALITIES, decision["utility"])}

    ranked_modalities = sorted(
        valid_modalities,
        key=lambda modality: utility.get(modality, float("-inf")),
        reverse=True,
    )

    ordered_modalities = []
    if target in {"webqa", "visual_rag"} and primary_modality == "image" and soft_top_n > 1:
        no_utility = utility.get("no", float("-inf"))
        paragraph_utility = utility.get("paragraph", float("-inf"))
        document_utility = utility.get("document", float("-inf"))

        # WebQA is mostly image-first, but a weak text branch can rescue cases
        # where the retrieved image is off-topic. Default to "no" as the safer
        # second branch and only admit paragraph when it is clearly stronger.
        paragraph_margin = 0.015
        document_margin = 0.02
        secondary_modality = "no"
        if (
            paragraph_utility >= no_utility + paragraph_margin
            and paragraph_utility >= document_utility + document_margin
        ):
            secondary_modality = "paragraph"

        ordered_modalities.append("image")
        if secondary_modality not in ordered_modalities:
            ordered_modalities.append(secondary_modality)

        for modality in ranked_modalities:
            if modality == "document":
                continue
            if modality not in ordered_modalities:
                ordered_modalities.append(modality)
            if len(ordered_modalities) >= soft_top_n:
                break
    elif primary_modality in valid_modalities:
        ordered_modalities.append(primary_modality)
        for modality in ranked_modalities:
            if modality not in ordered_modalities:
                ordered_modalities.append(modality)
            if len(ordered_modalities) >= soft_top_n:
                break

    selected_modalities = ordered_modalities[:soft_top_n]
    if not selected_modalities:
        selected_modalities = [primary_modality]

    weight_mode = str(weight_mode or "theta").strip().lower()
    if weight_mode == "utility":
        weights = _softmax([utility.get(modality, 0.0) for modality in selected_modalities])
    else:
        raw_weights = [max(0.0, theta.get(modality, 0.0)) for modality in selected_modalities]
        total = sum(raw_weights)
        if total <= 0:
            weights = [1.0 / len(selected_modalities)] * len(selected_modalities)
        else:
            weights = [value / total for value in raw_weights]

    return selected_modalities, weights


def _build_fusion_query(target: str, row, candidates):
    answer_mode = _answer_mode_for_target(target)
    if answer_mode == "mcq_letter":
        answer_instruction = "Return only a single letter: A, B, C, or D."
    elif answer_mode == "exact_short":
        answer_instruction = "Return only the exact short answer."
    else:
        answer_instruction = "Return one complete sentence."

    candidate_sections = []
    for idx, candidate in enumerate(candidates, start=1):
        candidate_sections.append(
            "\n".join(
                [
                    f"Candidate {idx}",
                    f"Modality: {candidate['modality']}",
                    f"Bayes weight: {candidate['weight']:.4f}",
                    f"Answer: {str(candidate['response']).strip()}",
                ]
            )
        )

    return (
        "You are given a question and candidate answers from different retrieval branches.\n"
        "Higher Bayes weight means the routing policy trusts that branch more.\n"
        "Use the candidate answers to produce one final answer.\n"
        "If candidates conflict, prefer the answer better supported by the higher-weight branches.\n"
        "Do not explain your reasoning.\n"
        f"{answer_instruction}\n\n"
        f"Question:\n{row['question']}\n\n"
        "Candidate answers:\n"
        + "\n\n".join(candidate_sections)
    )


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
        choices=SUPPORTED_TARGETS,
    )
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--nframes", type=str, default="1")
    parser.add_argument("--route_dir", type=str, default="route/results")
    parser.add_argument("--output_root", type=str, default="eval/results_bayes")
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

    # Bayesian routing args
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
        help="Temperature for explicit router probabilities. >1 flattens, <1 sharpens.",
    )
    parser.add_argument(
        "--router_probs_temperature_by_target",
        type=str,
        default="",
        help="Target-specific router prob temperatures: webqa=2.0;hotpotqa=1.2",
    )
    parser.add_argument(
        "--router_probs_blend_with_original",
        type=float,
        default=0.0,
        help="Blend explicit probs toward the original router label one-hot. 0 disables.",
    )
    parser.add_argument(
        "--router_probs_blend_with_original_by_target",
        type=str,
        default="",
        help="Target-specific blend strengths in [0,1]: webqa=0.4;hotpotqa=0.2",
    )

    # Optional online adaptation
    parser.add_argument("--online_update", type=int, default=0, choices=[0, 1])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.0)
    parser.add_argument("--penalty", type=float, default=0.5)
    parser.add_argument("--spread", type=float, default=0.25)
    parser.add_argument("--use_penalty_update", type=int, default=1, choices=[0, 1])

    # Bayes-only soft execution / fusion controls.
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

    args = parser.parse_args()

    model_name = args.model_path.split("/")[-1]
    nframes_tag = args.nframes.replace(",", "_").replace(":", "")
    bayes_suffix = "bayes"
    if args.bayes_tag.strip():
        bayes_suffix = f"bayes_{_sanitize_tag(args.bayes_tag)}"
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
    partial_file = f"{output_file}.partial"
    save_every = max(1, _get_env_int("EVAL_SAVE_EVERY", 25))
    resume_eval = _get_env_bool("EVAL_RESUME", True)

    alpha_prior = _parse_float_list(args.alpha_prior, expected_len=len(MODALITIES))
    modality_costs = _parse_float_list(args.modality_costs, expected_len=len(MODALITIES))
    alpha_prior_by_target = _parse_prior_map(args.alpha_prior_by_target)
    router_probs_temperature_by_target = _parse_scalar_map(args.router_probs_temperature_by_target)
    router_probs_blend_with_original_by_target = _parse_scalar_map(
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

    print(
        f"[BAYES EVAL] model={args.model_path}, router={args.router_model}, target={args.target}, "
        f"top_k={args.top_k}, alpha={args.alpha}, nframes={args.nframes}, "
        f"soft_top_n={args.soft_top_n}, soft_weight_mode={args.soft_weight_mode}, "
        f"soft_fusion_mode={args.soft_fusion_mode}, "
        f"router_probs_temperature={target_router_probs_temperature}, "
        f"router_probs_blend_with_original={target_router_probs_blend_with_original}"
    )

    bayes_router = BayesDirichletRouter(
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
    )

    model = ModelLoader(args.model_path)

    route_file = os.path.join(args.route_dir, args.router_model, f"{args.target}.json")
    if not os.path.exists(route_file):
        raise FileNotFoundError(
            f"Route result file not found: {route_file}\n"
            "Please run routing first."
        )

    with open(route_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if resume_eval:
        resume_file = partial_file if os.path.exists(partial_file) else None
        if resume_file is None and os.environ.get("FORCE_EVAL", "0") != "1":
            resume_file = output_file if os.path.exists(output_file) else None
        if resume_file is not None and os.path.exists(resume_file):
            with open(resume_file, "r", encoding="utf-8") as f:
                restored = _merge_existing_results(data, json.load(f))
            if restored:
                print(f"[INFO] Resuming from {resume_file}: restored {restored}/{len(data)} rows.")

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
                        textfeats_path=get_text_feature_paths(args.target, modality),
                    )
                retrieved, _ = retriever_paragraph.retrieve(current_row["index"], top_k=args.top_k)
            else:
                _clear_inactive_retrievers("document", single_retriever_cache)
                if retriever_document is None:
                    retriever_document = BGETextRetriever(
                        queryfeats_path=os.path.join(args.query_bge_dir, f"{args.target}.pkl"),
                        textfeats_path=get_text_feature_paths(args.target, modality),
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
                    imgfeats_path, imgcapfeats_path = get_image_feature_paths(args.target)
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

    completed_since_save = 0
    for row in tqdm(data, desc=f"Bayes-evaluating {args.target} with {args.model_path} + {args.router_model}"):
        if resume_eval and _has_response(row):
            continue

        query = reformat(row)
        original_modality = str(row.get("retrieval", "error")).lower()
        retrieval_conf = row.get("retrieval_conf", None)
        retrieval_probs = _parse_router_probs(
            row,
            temperature=target_router_probs_temperature,
            original_retrieval=original_modality,
            blend_with_original=target_router_probs_blend_with_original,
        )
        used_explicit_probs = retrieval_probs is not None

        decision = bayes_router.decide(
            retrieval=original_modality,
            retrieval_conf=retrieval_conf,
            retrieval_probs=retrieval_probs,
            target=args.target,
        )
        modality = decision["selected"]

        # Keep the same non-webqa hard guard as baseline.
        if args.target not in {"webqa", "visual_rag"} and modality == "image":
            modality = "document"

        retrieved = []
        response = ""
        soft_modalities = [modality]
        soft_weights = [1.0]
        soft_fusion_mode_used = "hard"

        if args.soft_top_n > 1:
            soft_modalities, soft_weights = _select_soft_modalities(
                decision=decision,
                target=args.target,
                soft_top_n=args.soft_top_n,
                weight_mode=args.soft_weight_mode,
                primary_modality=modality,
            )

        if len(soft_modalities) <= 1:
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

            response, soft_fusion_mode_used = _resolve_trivial_fusion(args.target, candidates)
            if not response:
                if args.soft_fusion_mode == "none":
                    response = _weighted_vote_response(args.target, candidates)
                    soft_fusion_mode_used = "weighted_pick"
                else:
                    fusion_query = _build_fusion_query(args.target, row, candidates)
                    fused_response = model.inference(
                        fusion_query,
                        max_new_tokens=args.soft_fusion_max_new_tokens,
                    )
                    if args.target in {"mmlu", "truthfulqa"}:
                        fused_choice = _extract_choice_from_prediction(fused_response)
                        if fused_choice is not None:
                            response = fused_choice
                            soft_fusion_mode_used = "llm"
                        else:
                            response = _weighted_vote_response(args.target, candidates)
                            soft_fusion_mode_used = "llm_fallback_weighted_pick"
                    else:
                        response = str(fused_response or "").strip()
                        if response:
                            soft_fusion_mode_used = "llm"
                        else:
                            response = _weighted_vote_response(args.target, candidates)
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

        row["retrieval_original"] = original_modality
        row["retrieval_bayes"] = modality
        row["retrieval_bayes_uncertainty"] = decision["uncertainty"]
        row["retrieval_bayes_posterior_alpha"] = decision["posterior_alpha"]
        row["retrieval_bayes_router_probs"] = decision["router_probs"]
        row["retrieval_bayes_theta"] = decision["theta"]
        row["retrieval_bayes_utility"] = decision["utility"]
        row["retrieval_bayes_used_explicit_probs"] = bool(used_explicit_probs)
        if used_explicit_probs:
            row["retrieval_bayes_probs_source"] = row.get("retrieval_probs_source", "row.retrieval_probs")
            row["retrieval_bayes_probs_temperature"] = float(target_router_probs_temperature)
            row["retrieval_bayes_probs_blend_with_original"] = float(
                target_router_probs_blend_with_original
            )

        if args.online_update == 1:
            reward = _light_reward(args.target, row, response)
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

        completed_since_save += 1
        if completed_since_save % save_every == 0:
            _save_json_atomic(partial_file, data)
            print(f"[INFO] Saved partial results to: {partial_file}")

    if completed_since_save:
        _save_json_atomic(partial_file, data)

    _save_json_atomic(output_file, data)
    if os.path.exists(partial_file):
        os.remove(partial_file)
    print(f"Saved Bayes results to: {output_file}")

    meta = {
        "model_path": args.model_path,
        "router_model": args.router_model,
        "target": args.target,
        "top_k": args.top_k,
        "alpha": args.alpha,
        "nframes": args.nframes,
        "route_dir": args.route_dir,
        "output_root": args.output_root,
        "query_bge_dir": args.query_bge_dir,
        "query_internvideo_dir": args.query_internvideo_dir,
        "bge_image_retrieval": args.bge_image_retrieval,
        "bayes_tag": args.bayes_tag,
        "alpha_prior_init": alpha_prior,
        "alpha_prior_final": bayes_router.alpha_prior.tolist(),
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
        "soft_fusion_mode": args.soft_fusion_mode,
        "soft_fusion_max_new_tokens": args.soft_fusion_max_new_tokens,
        "soft_store_candidates": bool(args.soft_store_candidates),
    }
    meta_file = output_file.replace(".json", ".meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Saved Bayes meta to: {meta_file}")
