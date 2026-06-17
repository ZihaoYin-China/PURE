#!/usr/bin/env python
"""Sampled wall-clock and token-cost benchmark for the in-domain QA suite.

Default design:
  * 5 in-domain targets.
  * 500 stratified examples per target.
  * 6 key methods used in the paper discussion.
  * Replays the stored evidence/candidate branches, so the latency study uses
    the same route decisions and retrieved files as the reported QA results.

The script has three practical modes:

  prepare      Build the fixed sample manifest.
  run-latency  Replay generator/verifier calls on the fixed sample and time them.
  summarize    Aggregate cached branch-budget stats and measured latency.
  all          prepare + run-latency + summarize.

Historical result files are not modified.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "eval"
for path in (EVAL_DIR, REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from score import (  # noqa: E402
    score_long_answers,
    score_mmlu,
    score_short_answers,
)
from eval import reformat  # noqa: E402
from utils.models import qwen_api  # noqa: E402
import eval_bayes_vib_posterior as posterior_verifier  # noqa: E402


TARGETS = ["mmlu", "squad", "natural_questions", "hotpotqa", "webqa"]
MODALITIES = ["no", "paragraph", "document", "image"]
DEFAULT_COSTS = {"no": 0.0, "paragraph": 0.25, "document": 0.45, "image": 0.60}

DEFAULT_METHODS = [
    {
        "name": "Naive",
        "policy": "as_is",
        "path": "eval/results_qwen36plus_api_compare_fixed_no/"
        "qwen-api:qwen3.6-plus/t5-large/{target}_top1_0.2_1.json",
        "overrides": {
            "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_fixed_no/"
            "qwen-api:qwen3.6-plus/t5-large/hotpotqa_top1_0.2_1.json",
            "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_fixed_no/"
            "qwen-api:qwen3.6-plus/t5-large/webqa_top1_0.2_1.json",
        },
    },
    {
        "name": "UniversalRAG-T5",
        "policy": "as_is",
        "path": "eval/results_qwen36plus_api_universalrag_test/"
        "qwen-api:qwen3.6-plus/t5-large/{target}_top1_0.2_1.json",
        "overrides": {
            "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_universalrag/"
            "qwen-api:qwen3.6-plus/t5-large/hotpotqa_top1_0.2_1.json",
            "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_universalrag/"
            "qwen-api:qwen3.6-plus/t5-large/webqa_top1_0.2_1.json",
        },
    },
    {
        "name": "Hard-T5-top1",
        "policy": "as_is",
        "path": "eval/results_qwen36plus_api_compare_hard_router/"
        "qwen-api:qwen3.6-plus/t5-large/{target}_top1_0.2_1.json",
        "overrides": {
            "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_hard/"
            "qwen-api:qwen3.6-plus/t5-large/hotpotqa_top1_0.2_1.json",
            "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_hard/"
            "qwen-api:qwen3.6-plus/t5-large/webqa_top1_0.2_1.json",
        },
    },
    {
        "name": "Hard-T5-top2+Ver",
        "policy": "as_is",
        "path": "eval/results_ablation_classifier_verifier_no_bayes/"
        "qwen-api:qwen3.6-plus/t5-large/{target}_top1_0.2_1_classifier_verifier_no_bayes_top2.json",
        "overrides": {
            "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_no_bayes_verifier/"
            "qwen-api:qwen3.6-plus/t5-large/"
            "hotpotqa_top1_0.2_1_classifier_t5large_verifier_no_bayes_top2_refresh.json",
            "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_no_bayes_verifier/"
            "qwen-api:qwen3.6-plus/t5-large/"
            "webqa_top1_0.2_1_classifier_t5large_verifier_no_bayes_top2_refresh.json",
        },
    },
    {
        "name": "COVER-T5-final",
        "policy": "as_is",
        "path": "eval/results_cover_candidate_verifier_qwen36plus_indomain/"
        "qwen-api:qwen3.6-plus/t5-large/"
        "{target}_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier.json",
        "overrides": {
            "mmlu": "eval/results_qwen36plus_api_mmlu_noonly_t5/"
            "qwen-api:qwen3.6-plus/t5-large/"
            "mmlu_top1_0.2_1_bayes_mmlu_noonly_qwen36plus_api_t5_tau10_beta0p1.json",
            "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/"
            "qwen-api:qwen3.6-plus/t5-large/"
            "hotpotqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
            "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/"
            "qwen-api:qwen3.6-plus/t5-large/"
            "webqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
        },
    },
    {
        "name": "COVER-T5-selective50",
        "policy": "selective",
        "selective_fraction": 0.50,
        "path": "eval/results_cover_candidate_verifier_qwen36plus_indomain/"
        "qwen-api:qwen3.6-plus/t5-large/"
        "{target}_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier.json",
        "overrides": {
            "hotpotqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/"
            "qwen-api:qwen3.6-plus/t5-large/"
            "hotpotqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
            "webqa": "eval/results_webqa_hotpot_fair_refresh_20260528_cover/"
            "qwen-api:qwen3.6-plus/t5-large/"
            "webqa_top1_0.2_1_bayes_cover_t5_candidate_verifier_softtop2_theta_posteriorverifier_refresh.json",
        },
    },
]


@dataclass(frozen=True)
class MethodSpec:
    name: str
    path: str
    policy: str = "as_is"
    selective_fraction: float = 0.0
    overrides: Dict[str, str] | None = None


_write_lock = threading.Lock()


def repo_path(path: str) -> str:
    return str((REPO_ROOT / path).resolve()) if not os.path.isabs(path) else path


def parse_csv_list(text: str, default: Sequence[str]) -> List[str]:
    if not text:
        return list(default)
    return [x.strip() for x in text.replace(",", " ").split() if x.strip()]


def load_methods(path: Optional[str]) -> List[MethodSpec]:
    data = DEFAULT_METHODS
    if path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    specs = []
    for item in data:
        specs.append(
            MethodSpec(
                name=str(item["name"]),
                path=str(item["path"]),
                policy=str(item.get("policy", "as_is")),
                selective_fraction=float(item.get("selective_fraction", 0.0)),
                overrides=dict(item.get("overrides", {})),
            )
        )
    return specs


def find_result_file(template: str, target: str) -> str:
    raw = template.format(target=target)
    path = repo_path(raw)
    if os.path.isfile(path):
        return path
    matches = [
        p
        for p in sorted(glob.glob(path))
        if os.path.isfile(p) and not p.endswith(".meta.json") and not p.endswith(".partial")
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No result file found for target={target}: {raw}")
    raise ValueError(f"Template matched multiple files for target={target}: {matches[:8]}")


def method_file(spec: MethodSpec, target: str) -> str:
    template = (spec.overrides or {}).get(target, spec.path)
    return find_result_file(template, target)


def load_json_list(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def row_key(row: dict) -> str:
    return "\t".join(
        [
            str(row.get("source", "")),
            str(row.get("index", "")),
            str(row.get("question", "")),
        ]
    )


def key_parts(key: str) -> Tuple[str, str, str]:
    parts = key.split("\t", 2)
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def normalize_modality(value: object, target: str) -> str:
    modality = str(value or "no").strip().lower()
    if modality not in MODALITIES:
        modality = "no"
    if target not in {"webqa", "visual_rag"} and modality == "image":
        modality = "document"
    return modality


def get_candidates(row: dict, target: str) -> List[dict]:
    for key in ("retrieval_bayes_soft_candidates", "retrieval_no_bayes_candidates"):
        candidates = row.get(key)
        if isinstance(candidates, list) and candidates:
            out = []
            for cand in candidates:
                cand = dict(cand)
                cand["modality"] = normalize_modality(cand.get("modality"), target)
                cand["weight"] = float(cand.get("weight", 1.0 / max(1, len(candidates))))
                cand["retrieved"] = list(cand.get("retrieved") or [])
                cand["response"] = str(cand.get("response", ""))
                out.append(cand)
            return out

    modality = row.get("retrieval_bayes") or row.get("retrieval")
    return [
        {
            "modality": normalize_modality(modality, target),
            "weight": 1.0,
            "retrieved": list(row.get("retrieved") or []),
            "response": str(row.get("response", "")),
        }
    ]


def margin_value(row: dict) -> float:
    for key in (
        "retrieval_margin",
        "retrieval_probs_margin",
        "retrieval_bayes_posterior_score_gap",
    ):
        value = row.get(key)
        try:
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            pass
    probs = row.get("retrieval_probs")
    if isinstance(probs, list) and len(probs) >= 2:
        vals = sorted([float(x) for x in probs], reverse=True)
        return vals[0] - vals[1]
    return float("inf")


def selective_threshold(rows: Sequence[dict], fraction: float) -> float:
    margins = sorted(margin_value(row) for row in rows if math.isfinite(margin_value(row)))
    if not margins:
        return float("-inf")
    idx = min(len(margins) - 1, max(0, int(math.ceil(len(margins) * fraction)) - 1))
    return margins[idx]


def materialize_row(row: dict, target: str, spec: MethodSpec, threshold: float | None) -> dict:
    out = deepcopy(row)
    candidates = get_candidates(row, target)
    use_multi = len(candidates) > 1

    if spec.policy == "selective":
        use_multi = len(candidates) > 1 and margin_value(row) <= float(threshold or float("-inf"))
        out["latency_cost_selective_fraction"] = spec.selective_fraction
        out["latency_cost_selective_threshold"] = threshold
        out["latency_cost_selective_multi"] = use_multi
        if not use_multi and candidates:
            first = candidates[0]
            out["response"] = first.get("response", "")
            out["retrieved"] = list(first.get("retrieved") or [])
            out["retrieval"] = first.get("modality", row.get("retrieval", "no"))

    out["latency_cost_candidates"] = candidates if use_multi else candidates[:1]
    return out


def executed_modalities(row: dict, target: str) -> List[str]:
    candidates = row.get("latency_cost_candidates") or get_candidates(row, target)
    out = []
    for cand in candidates:
        modality = normalize_modality(cand.get("modality"), target)
        if modality not in out:
            out.append(modality)
    return out or ["no"]


def proxy_cost(row: dict, target: str, costs: Dict[str, float]) -> float:
    return sum(costs[m] for m in executed_modalities(row, target))


def approx_tokens(text: object, image_count: int = 0, image_token_proxy: int = 0) -> int:
    text = str(text or "")
    # Conservative English/Chinese-agnostic fallback when API usage is absent.
    return int(math.ceil(len(text) / 4.0)) + image_count * int(image_token_proxy)


def read_retrieved_texts(paths: Sequence[str], max_chars: int = 3000) -> List[str]:
    texts = []
    for path in paths[:1]:
        try:
            with open(repo_path(str(path)), "r", encoding="utf-8", errors="ignore") as f:
                texts.append(qwen_api._truncate_text(f.read(), max_chars=max_chars))
        except OSError:
            texts.append("")
    return texts


def branch_prompt(row: dict, target: str, candidate: dict, model_config: Optional[dict] = None) -> Tuple[str, List[str]]:
    query = reformat(row)
    modality = normalize_modality(candidate.get("modality"), target)
    retrieved = list(candidate.get("retrieved") or [])
    image_paths: List[str] = []

    if modality == "no":
        return query, image_paths

    if modality in {"paragraph", "document"}:
        retrieved_texts = read_retrieved_texts(retrieved)
        doc_text = "\n\n".join(
            [f"Relevant document {idx + 1}:\n{text}" for idx, text in enumerate(retrieved_texts)]
        )
        return (
            "Answer the question using the retrieved document.\n"
            "Keep the answer short and exact.\n\n"
            f"{doc_text}\n\nQuestion:\n{query}",
            image_paths,
        )

    if modality == "image":
        provider_config = (model_config or {}).get("provider_config", qwen_api.API_PROVIDERS["qwen-api"])
        image_mode = str((model_config or {}).get("image_mode", "image") or "image").lower()
        max_images = max(1, int(os.environ.get("GENERATOR_API_MAX_IMAGES", "1") or 1))
        images = [repo_path(str(p)) for p in retrieved[:max_images]]
        use_caption = image_mode in {"caption", "both"}
        send_images = image_mode in {"image", "both"}

        caption_texts = []
        if use_caption:
            for idx, image_path in enumerate(images):
                caption = qwen_api._caption_for_image(image_path)
                caption_texts.append(
                    f"Relevant image {idx + 1} caption:\n"
                    f"{qwen_api._truncate_text(caption, max_chars=1000)}"
                )
        if send_images:
            image_paths = images

        if use_caption and send_images:
            return (
                "Considering the given image and its retrieved caption,\n"
                + "\n\n".join(caption_texts)
                + "\n\n"
                + query,
                image_paths,
            )
        if use_caption:
            return (
                "Answer the question using the retrieved image caption as visual evidence.\n"
                "Keep the answer grounded in the caption; if the caption is insufficient, "
                "answer as best as possible.\n\n"
                + "\n\n".join(caption_texts)
                + "\n\nQuestion:\n"
                + query,
                image_paths,
            )
        return f"Considering the given image,\n{query}", image_paths

    return query, image_paths


def verifier_args(max_new_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        posterior_agreement_weight=1.00,
        posterior_conflict_weight=0.35,
        posterior_route_weight=0.15,
        posterior_evidence_weight=0.05,
        posterior_empty_penalty=1.00,
        posterior_non_answer_penalty=0.85,
        posterior_verifier=1,
        posterior_verifier_choice_only=1,
        posterior_verifier_max_new_tokens=max_new_tokens,
        posterior_evidence_max_chars=1200,
    )


def row_has_observed_verifier(row: dict) -> bool:
    return bool(
        str(row.get("retrieval_bayes_posterior_generation_verifier_response", "")).strip()
        or str(row.get("retrieval_no_bayes_verifier_response", "")).strip()
    )


def should_call_verifier(row: dict, verifier_policy: str) -> bool:
    candidates = row.get("latency_cost_candidates") or []
    if len(candidates) <= 1:
        return False
    if verifier_policy == "always":
        return True
    if verifier_policy == "never":
        return False
    return row_has_observed_verifier(row)


def build_verifier_prompt(row: dict, target: str, max_new_tokens: int) -> str:
    candidates = row.get("latency_cost_candidates") or []
    args = verifier_args(max_new_tokens)
    _, _, scores = posterior_verifier._score_posterior_candidates(target, candidates, args)
    return posterior_verifier._build_posterior_verifier_query(target, row, candidates, scores, args)


def call_chat(
    model_config: dict,
    prompt: str,
    *,
    image_paths: Sequence[str] | None = None,
    max_tokens: int = 128,
    image_token_proxy: int = 0,
    max_retries: int = 3,
) -> dict:
    provider_config = model_config.get("provider_config", qwen_api.API_PROVIDERS["qwen-api"])
    messages = qwen_api._messages_for_query(
        prompt,
        image_paths=list(image_paths or []),
        provider_config=provider_config,
    )

    extra_body = None
    if model_config.get("enable_thinking") is not None:
        extra_body = {"enable_thinking": bool(model_config.get("enable_thinking"))}
    if model_config.get("thinking_type"):
        extra_body = extra_body or {}
        extra_body["thinking"] = {"type": model_config["thinking_type"]}

    request_kwargs = {
        "model": model_config["model_name"],
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max(1, int(max_tokens)),
    }
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    if model_config.get("reasoning_effort"):
        request_kwargs["reasoning_effort"] = model_config["reasoning_effort"]

    last_error = None
    start = time.perf_counter()
    completion = None
    for attempt in range(1, max_retries + 1):
        try:
            completion = model_config["client"].chat.completions.create(**request_kwargs)
            break
        except Exception as exc:  # pragma: no cover - depends on remote API
            last_error = exc
            if qwen_api._is_data_inspection_failed(exc):
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                answer_mode = qwen_api._infer_answer_mode(prompt)
                content = qwen_api._fallback_answer_for_blocked_output(answer_mode)
                prompt_tokens = approx_tokens(
                    prompt,
                    image_count=len(image_paths or []),
                    image_token_proxy=image_token_proxy,
                )
                completion_tokens = approx_tokens(content)
                return {
                    "latency_ms": elapsed_ms,
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                    "total_tokens": int(prompt_tokens + completion_tokens),
                    "response_chars": len(str(content or "")),
                    "error": "",
                    "warning": "data_inspection_failed_fallback",
                }
            if attempt == max_retries:
                raise
            time.sleep(min(30.0, 2.0 * attempt))
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    content = ""
    if completion is not None and getattr(completion, "choices", None):
        content, _reasoning = qwen_api._extract_choice_text(completion.choices[0])

    usage = getattr(completion, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None

    image_count = len(image_paths or [])
    if prompt_tokens is None:
        prompt_tokens = approx_tokens(prompt, image_count=image_count, image_token_proxy=image_token_proxy)
    if completion_tokens is None:
        completion_tokens = approx_tokens(content)
    if total_tokens is None:
        total_tokens = int(prompt_tokens) + int(completion_tokens)

    return {
        "latency_ms": elapsed_ms,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(total_tokens),
        "response_chars": len(str(content or "")),
        "error": "" if last_error is None else str(last_error),
    }


def load_rows_by_method_target(methods: Sequence[MethodSpec], targets: Sequence[str]) -> Dict[Tuple[str, str], List[dict]]:
    out = {}
    for spec in methods:
        for target in targets:
            out[(spec.name, target)] = load_json_list(method_file(spec, target))
    return out


def sample_manifest_path(output_dir: str) -> str:
    return os.path.join(output_dir, "sample_manifest.jsonl")


def build_sample_manifest(
    methods: Sequence[MethodSpec],
    targets: Sequence[str],
    output_dir: str,
    sample_per_target: int,
    seed: int,
) -> List[dict]:
    rows_by_method = load_rows_by_method_target(methods, targets)
    rng = random.Random(seed)
    manifest = []

    for target in targets:
        maps = []
        for spec in methods:
            row_map = {row_key(row): row for row in rows_by_method[(spec.name, target)]}
            maps.append(row_map)
        common_keys = set(maps[0])
        for row_map in maps[1:]:
            common_keys &= set(row_map)
        if not common_keys:
            raise RuntimeError(f"No common examples across methods for target={target}")

        reference = maps[0]
        strata: Dict[str, List[str]] = {}
        for key in sorted(common_keys):
            row = reference[key]
            stratum = str(row.get("gt_retrieval") or row.get("retrieval") or "all")
            strata.setdefault(stratum, []).append(key)

        target_n = min(sample_per_target, len(common_keys))
        selected: List[str] = []
        quotas: Dict[str, int] = {}
        total = sum(len(v) for v in strata.values())
        remainders = []
        for stratum, keys in strata.items():
            exact = target_n * len(keys) / max(1, total)
            base = min(len(keys), int(math.floor(exact)))
            quotas[stratum] = base
            remainders.append((exact - base, stratum))

        remaining = target_n - sum(quotas.values())
        for _rem, stratum in sorted(remainders, reverse=True):
            if remaining <= 0:
                break
            if quotas[stratum] < len(strata[stratum]):
                quotas[stratum] += 1
                remaining -= 1

        for stratum, keys in sorted(strata.items()):
            shuffled = list(keys)
            rng.shuffle(shuffled)
            selected.extend(shuffled[: quotas[stratum]])
        rng.shuffle(selected)

        for key in selected:
            source, index, question = key_parts(key)
            manifest.append(
                {
                    "target": target,
                    "source": source,
                    "index": index,
                    "question": question,
                    "stratum": str(reference[key].get("gt_retrieval") or reference[key].get("retrieval") or "all"),
                }
            )

    os.makedirs(output_dir, exist_ok=True)
    path = sample_manifest_path(output_dir)
    with open(path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return manifest


def load_manifest(output_dir: str) -> List[dict]:
    path = sample_manifest_path(output_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Sample manifest not found: {path}. Run --mode prepare first.")
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def manifest_keys(manifest: Sequence[dict], target: str) -> set[str]:
    return {
        "\t".join([str(item.get("source", "")), str(item.get("index", "")), str(item.get("question", ""))])
        for item in manifest
        if item.get("target") == target
    }


def quality_metric(target: str, rows: List[dict], score_long: bool) -> Tuple[str, Optional[float]]:
    try:
        if target == "mmlu":
            metrics = score_mmlu(rows)
            return "Accuracy", float(metrics["Accuracy"])
        if target in {"squad", "natural_questions", "hotpotqa"}:
            metrics = score_short_answers(rows, target)
            return "F1", float(metrics["F1"])
        if target == "webqa" and score_long:
            metrics = score_long_answers(rows, target)
            return "BERTScore", float(metrics["BERTScore"])
    except Exception as exc:
        return f"score_error:{exc}", None
    return ("BERTScore" if target == "webqa" else "Score"), None


def summarize_cached(
    methods: Sequence[MethodSpec],
    targets: Sequence[str],
    output_dir: str,
    costs: Dict[str, float],
    score_quality: bool,
    score_long: bool,
) -> str:
    manifest = load_manifest(output_dir)
    rows_by_method = load_rows_by_method_target(methods, targets)
    out_rows = []

    for spec in methods:
        for target in targets:
            raw_rows = rows_by_method[(spec.name, target)]
            threshold = selective_threshold(raw_rows, spec.selective_fraction) if spec.policy == "selective" else None
            full_rows = [materialize_row(row, target, spec, threshold) for row in raw_rows]
            sample_keys = manifest_keys(manifest, target)
            sample_rows = [row for row in full_rows if row_key(row) in sample_keys]

            branches = [len(row.get("latency_cost_candidates") or []) for row in sample_rows]
            costs_sample = [proxy_cost(row, target, costs) for row in sample_rows]
            multi_rate = sum(1 for x in branches if x > 1) / max(1, len(branches))

            metric_name, full_quality = ("", None)
            sample_quality = None
            if score_quality:
                metric_name, full_quality = quality_metric(target, full_rows, score_long)
                _, sample_quality = quality_metric(target, sample_rows, score_long)

            out_rows.append(
                {
                    "method": spec.name,
                    "target": target,
                    "sample_count": len(sample_rows),
                    "full_count": len(full_rows),
                    "metric": metric_name,
                    "full_quality": full_quality if full_quality is not None else "",
                    "sample_quality": sample_quality if sample_quality is not None else "",
                    "avg_branches_sample": statistics.fmean(branches) if branches else "",
                    "multi_rate_sample": multi_rate * 100.0,
                    "proxy_cost_sample": statistics.fmean(costs_sample) if costs_sample else "",
                    "selective_threshold": threshold if threshold is not None else "",
                    "result_file": method_file(spec, target),
                }
            )

    path = os.path.join(output_dir, "cached_sample_summary.csv")
    write_csv(path, out_rows)
    return path


def latency_output_path(output_dir: str) -> str:
    return os.path.join(output_dir, "latency_events.jsonl")


def completed_event_keys(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error"):
                continue
            key = str(row.get("event_key", ""))
            if key:
                keys.add(key)
    return keys


def make_event_key(method: str, target: str, row: dict) -> str:
    return "|".join([method, target, str(row.get("source", "")), str(row.get("index", ""))])


def run_one_latency_task(
    spec: MethodSpec,
    target: str,
    row: dict,
    model_config: dict,
    args: argparse.Namespace,
) -> dict:
    candidates = row.get("latency_cost_candidates") or []
    branch_results = []
    branch_prompt_tokens = 0
    branch_completion_tokens = 0
    branch_total_tokens = 0
    branch_latency_ms = 0.0

    for idx, candidate in enumerate(candidates):
        prompt, image_paths = branch_prompt(row, target, candidate, model_config=model_config)
        if args.dry_run_api:
            result = {
                "latency_ms": 0.0,
                "prompt_tokens": approx_tokens(
                    prompt,
                    image_count=len(image_paths),
                    image_token_proxy=args.image_token_proxy,
                ),
                "completion_tokens": approx_tokens(candidate.get("response", "")),
                "total_tokens": 0,
                "response_chars": len(str(candidate.get("response", ""))),
                "error": "",
            }
            result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
        else:
            result = call_chat(
                model_config,
                prompt,
                image_paths=image_paths,
                max_tokens=args.max_new_tokens,
                image_token_proxy=args.image_token_proxy,
                max_retries=args.max_retries,
            )

        result.update(
            {
                "candidate_index": idx,
                "modality": candidate.get("modality", ""),
                "image_count": len(image_paths),
            }
        )
        branch_results.append(result)
        branch_latency_ms += float(result["latency_ms"])
        branch_prompt_tokens += int(result["prompt_tokens"])
        branch_completion_tokens += int(result["completion_tokens"])
        branch_total_tokens += int(result["total_tokens"])

    verifier_result = None
    verifier_called = should_call_verifier(row, args.verifier_policy)
    if verifier_called:
        verifier_prompt = build_verifier_prompt(row, target, args.verifier_max_new_tokens)
        if args.dry_run_api:
            verifier_result = {
                "latency_ms": 0.0,
                "prompt_tokens": approx_tokens(verifier_prompt),
                "completion_tokens": approx_tokens(
                    row.get("retrieval_bayes_posterior_generation_verifier_response")
                    or row.get("retrieval_no_bayes_verifier_response")
                    or ""
                ),
                "total_tokens": 0,
                "response_chars": 0,
                "error": "",
            }
            verifier_result["total_tokens"] = (
                verifier_result["prompt_tokens"] + verifier_result["completion_tokens"]
            )
        else:
            verifier_result = call_chat(
                model_config,
                verifier_prompt,
                max_tokens=args.verifier_max_new_tokens,
                image_token_proxy=args.image_token_proxy,
                max_retries=args.max_retries,
            )

    verifier_prompt_tokens = int((verifier_result or {}).get("prompt_tokens", 0))
    verifier_completion_tokens = int((verifier_result or {}).get("completion_tokens", 0))
    verifier_total_tokens = int((verifier_result or {}).get("total_tokens", 0))
    verifier_latency_ms = float((verifier_result or {}).get("latency_ms", 0.0))

    prompt_tokens = branch_prompt_tokens + verifier_prompt_tokens
    completion_tokens = branch_completion_tokens + verifier_completion_tokens
    total_tokens = branch_total_tokens + verifier_total_tokens
    api_cost = (
        prompt_tokens * args.input_price_per_m / 1_000_000.0
        + completion_tokens * args.output_price_per_m / 1_000_000.0
    )

    return {
        "event_key": make_event_key(spec.name, target, row),
        "method": spec.name,
        "target": target,
        "source": row.get("source", ""),
        "index": row.get("index", ""),
        "stratum": row.get("gt_retrieval") or row.get("retrieval") or "",
        "branch_count": len(candidates),
        "modalities": ";".join([str(c.get("modality", "")) for c in candidates]),
        "proxy_cost": proxy_cost(row, target, DEFAULT_COSTS),
        "branch_latency_ms": branch_latency_ms,
        "verifier_latency_ms": verifier_latency_ms,
        "total_latency_ms": branch_latency_ms + verifier_latency_ms,
        "branch_prompt_tokens": branch_prompt_tokens,
        "branch_completion_tokens": branch_completion_tokens,
        "verifier_prompt_tokens": verifier_prompt_tokens,
        "verifier_completion_tokens": verifier_completion_tokens,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "api_cost": api_cost,
        "verifier_called": int(verifier_called),
        "branch_results": branch_results,
        "verifier_result": verifier_result or {},
    }


def run_latency(
    methods: Sequence[MethodSpec],
    targets: Sequence[str],
    output_dir: str,
    args: argparse.Namespace,
) -> str:
    manifest = load_manifest(output_dir)
    rows_by_method = load_rows_by_method_target(methods, targets)
    sample_keys_by_target = {target: manifest_keys(manifest, target) for target in targets}
    tasks = []

    for spec in methods:
        for target in targets:
            raw_rows = rows_by_method[(spec.name, target)]
            threshold = selective_threshold(raw_rows, spec.selective_fraction) if spec.policy == "selective" else None
            for raw in raw_rows:
                if row_key(raw) not in sample_keys_by_target[target]:
                    continue
                tasks.append((spec, target, materialize_row(raw, target, spec, threshold)))

    if args.max_tasks:
        tasks = tasks[: args.max_tasks]

    os.makedirs(output_dir, exist_ok=True)
    out_path = latency_output_path(output_dir)
    done = completed_event_keys(out_path) if args.resume else set()
    tasks = [t for t in tasks if make_event_key(t[0].name, t[1], t[2]) not in done]
    print(f"[INFO] latency tasks to run: {len(tasks)}; already completed: {len(done)}")

    if not tasks:
        return out_path

    if args.dry_run_api:
        provider, model_name = qwen_api._provider_from_model_path(args.model_path)
        provider_config = qwen_api.API_PROVIDERS.get(provider, qwen_api.API_PROVIDERS["qwen-api"])
        model_config = {
            "provider_config": provider_config,
            "image_mode": provider_config.get("image_mode", "image"),
            "model_name": model_name,
            "client": None,
        }
    else:
        model_config, _, _ = qwen_api.load_model(args.model_path)

    def worker(task):
        spec, target, row = task
        try:
            return run_one_latency_task(spec, target, row, model_config, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            return {
                "event_key": make_event_key(spec.name, target, row),
                "method": spec.name,
                "target": target,
                "source": row.get("source", ""),
                "index": row.get("index", ""),
                "error": str(exc),
            }

    with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = [pool.submit(worker, task) for task in tasks]
        for n, fut in enumerate(as_completed(futures), start=1):
            event = fut.result()
            with _write_lock:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            if n % max(1, args.log_every) == 0:
                print(f"[INFO] completed {n}/{len(tasks)} sampled latency tasks")

    return out_path


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * pct / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def mean_ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


def summarize_latency(output_dir: str) -> str:
    path = latency_output_path(output_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Latency event file not found: {path}")

    # Keep one successful event per method-target-example. This makes summaries
    # robust to interrupted/restarted runs that append duplicate JSONL rows,
    # and ignores transient failed attempts when a later retry succeeded.
    deduped: Dict[str, dict] = {}
    error_only: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("event_key", ""))
            if not key:
                continue
            if row.get("error"):
                error_only.setdefault(key, row)
                continue
            deduped[key] = row

    groups: Dict[Tuple[str, str], List[dict]] = {}
    for row in deduped.values():
        groups.setdefault((row["method"], row["target"]), []).append(row)

    out_rows = []
    for (method, target), rows in sorted(groups.items()):
        lat = [float(r.get("total_latency_ms", 0.0)) for r in rows]
        branch_lat = [float(r.get("branch_latency_ms", 0.0)) for r in rows]
        verifier_lat = [float(r.get("verifier_latency_ms", 0.0)) for r in rows]
        prompt_tokens = [int(r.get("prompt_tokens", 0)) for r in rows]
        completion_tokens = [int(r.get("completion_tokens", 0)) for r in rows]
        total_tokens = [int(r.get("total_tokens", 0)) for r in rows]
        api_costs = [float(r.get("api_cost", 0.0)) for r in rows]
        branches = [float(r.get("branch_count", 0.0)) for r in rows]
        proxy = [float(r.get("proxy_cost", 0.0)) for r in rows]
        out_rows.append(
            {
                "method": method,
                "target": target,
                "n": len(rows),
                "latency_mean_ms": statistics.fmean(lat),
                "latency_ci95_ms": mean_ci95(lat),
                "latency_median_ms": statistics.median(lat),
                "latency_p95_ms": percentile(lat, 95),
                "branch_latency_mean_ms": statistics.fmean(branch_lat),
                "verifier_latency_mean_ms": statistics.fmean(verifier_lat),
                "prompt_tokens_mean": statistics.fmean(prompt_tokens),
                "completion_tokens_mean": statistics.fmean(completion_tokens),
                "total_tokens_mean": statistics.fmean(total_tokens),
                "api_cost_mean": statistics.fmean(api_costs),
                "avg_branches": statistics.fmean(branches),
                "multi_rate": 100.0 * sum(1 for b in branches if b > 1) / max(1, len(branches)),
                "verifier_rate": 100.0 * sum(1 for r in rows if int(r.get("verifier_called", 0)) == 1) / max(1, len(rows)),
                "proxy_cost": statistics.fmean(proxy),
            }
        )

    out_path = os.path.join(output_dir, "latency_summary.csv")
    write_csv(out_path, out_rows)
    return out_path


def write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_costs(text: str) -> Dict[str, float]:
    if not text:
        return dict(DEFAULT_COSTS)
    out = dict(DEFAULT_COSTS)
    for part in text.replace(",", " ").split():
        if not part.strip():
            continue
        key, value = part.split("=", 1)
        out[key.strip()] = float(value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["prepare", "run-latency", "summarize", "all"], default="all")
    parser.add_argument("--output-dir", default="analysis/results/cpu8c32g_latency_cost")
    parser.add_argument("--methods-config", default="")
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--sample-per-target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-path", default="qwen-api:qwen3.6-plus")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--dry-run-api", action="store_true", help="Do not call the remote API; estimate tokens only.")
    parser.add_argument("--continue-on-error", action="store_true", default=True)
    parser.add_argument("--max-tasks", type=int, default=0, help="Smoke-test limit across all method-target examples.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--verifier-max-new-tokens", type=int, default=64)
    parser.add_argument("--verifier-policy", choices=["observed", "always", "never"], default="observed")
    parser.add_argument("--image-token-proxy", type=int, default=0)
    parser.add_argument("--input-price-per-m", type=float, default=0.0)
    parser.add_argument("--output-price-per-m", type=float, default=0.0)
    parser.add_argument("--costs", default="no=0,paragraph=0.25,document=0.45,image=0.60")
    parser.add_argument("--score-quality", action="store_true")
    parser.add_argument("--score-long", action="store_true", help="Also compute ROUGE/BERTScore for WebQA.")
    args = parser.parse_args()

    targets = parse_csv_list(args.targets, TARGETS)
    methods = load_methods(args.methods_config or None)
    costs = parse_costs(args.costs)
    args.cost_map = costs
    output_dir = repo_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    if args.mode in {"prepare", "all"}:
        manifest = build_sample_manifest(methods, targets, output_dir, args.sample_per_target, args.seed)
        print(f"[INFO] wrote sample manifest: {sample_manifest_path(output_dir)} ({len(manifest)} rows)")

    if args.mode in {"run-latency", "all"}:
        path = run_latency(methods, targets, output_dir, args)
        print(f"[INFO] wrote latency events: {path}")

    if args.mode in {"summarize", "all"}:
        cached_csv = summarize_cached(methods, targets, output_dir, costs, args.score_quality, args.score_long)
        print(f"[INFO] wrote cached summary: {cached_csv}")
        if os.path.exists(latency_output_path(output_dir)):
            latency_csv = summarize_latency(output_dir)
            print(f"[INFO] wrote latency summary: {latency_csv}")


if __name__ == "__main__":
    main()
