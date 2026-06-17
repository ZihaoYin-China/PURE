import argparse
import json
import os
import sys
from typing import Iterable, List

import torch
import torch.nn.functional as F
from tabulate import tabulate
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from route.gpt.prompt import ROUTER_PROMPT


MODALITIES = ["no", "paragraph", "document", "image"]


def _parse_targets(text: str) -> List[str]:
    parts = [x.strip() for x in str(text or "").replace(" ", ",").split(",")]
    return [x for x in parts if x]


def _resolve_device(device: str) -> torch.device:
    device = str(device or "").strip().lower()
    if device in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _clean_question(row) -> str:
    return str(row["question"]).rsplit("\n", 1)[0]


class DistilBertSidecar:
    def __init__(self, checkpoint_dir: str, device: torch.device):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir).to(device)
        self.model.eval()

    def predict_probs(self, questions: List[str], max_input_length: int = 512) -> List[List[float]]:
        inputs = self.tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        return probs.detach().cpu().tolist()


class T5Sidecar:
    def __init__(self, checkpoint_dir: str, device: torch.device):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(checkpoint_dir)
        dtype = torch.bfloat16 if device.type == "cuda" else None
        kwargs = {"torch_dtype": dtype} if dtype is not None else {}
        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint_dir, **kwargs).to(device)
        self.model.eval()

        label_token_ids = []
        for label in MODALITIES:
            token_ids = self.tokenizer.encode(label, add_special_tokens=False)
            if len(token_ids) != 1:
                raise ValueError(
                    f"T5 sidecar expects single-token labels, but {label!r} tokenized to {token_ids}"
                )
            label_token_ids.append(token_ids[0])
        self.label_token_ids = torch.tensor(label_token_ids, dtype=torch.long, device=device)

        decoder_start_token_id = self.model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = self.model.config.pad_token_id
        if decoder_start_token_id is None:
            raise ValueError("Could not resolve decoder_start_token_id for T5 router sidecar.")
        self.decoder_start_token_id = int(decoder_start_token_id)

    def predict_probs(self, questions: List[str], max_input_length: int = 512) -> List[List[float]]:
        prompts = [ROUTER_PROMPT.format(query=question) for question in questions]
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(self.device)

        decoder_input_ids = torch.full(
            (inputs["input_ids"].shape[0], 1),
            fill_value=self.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                decoder_input_ids=decoder_input_ids,
            )
            logits = outputs.logits[:, 0, :]
            label_logits = logits.index_select(dim=-1, index=self.label_token_ids)
            probs = F.softmax(label_logits.float(), dim=-1)
        return probs.detach().cpu().tolist()


def _build_exporter(router_model: str, checkpoint_dir: str, device: torch.device):
    router_model = str(router_model).strip().lower()
    if router_model == "distilbert":
        return DistilBertSidecar(checkpoint_dir=checkpoint_dir, device=device)
    if router_model == "t5-large":
        return T5Sidecar(checkpoint_dir=checkpoint_dir, device=device)
    raise ValueError(f"Unsupported router_model for probability export: {router_model}")


def _default_checkpoint(router_model: str) -> str:
    return os.path.join("route", "train", "checkpoints", router_model)


def _attach_probs(row, probs: List[float], source: str):
    probs = [float(x) for x in probs]
    if len(probs) != len(MODALITIES):
        raise ValueError(f"Expected {len(MODALITIES)} probabilities, got {len(probs)}")
    total = sum(max(0.0, x) for x in probs)
    if total <= 0:
        raise ValueError(f"Invalid probability vector: {probs}")
    probs = [max(0.0, x) / total for x in probs]

    ranked = sorted(probs, reverse=True)
    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))

    row["retrieval_probs"] = probs
    row["retrieval_probs_order"] = MODALITIES
    row["retrieval_probs_source"] = source
    row["retrieval_probs_pred"] = MODALITIES[pred_idx]
    row["retrieval_probs_conf"] = float(probs[pred_idx])
    row["retrieval_probs_margin"] = float(ranked[0] - ranked[1]) if len(ranked) > 1 else float(ranked[0])
    row["retrieval_probs_match_original"] = (
        str(row.get("retrieval", "")).strip().lower() == MODALITIES[pred_idx]
    )
    return row


def export_probabilities(
    router_model: str,
    checkpoint_dir: str,
    base_route_dir: str,
    output_dir: str,
    targets: Iterable[str],
    batch_size: int,
    max_input_length: int,
    device: torch.device,
):
    exporter = _build_exporter(router_model=router_model, checkpoint_dir=checkpoint_dir, device=device)
    router_output_dir = os.path.join(output_dir, router_model)
    os.makedirs(router_output_dir, exist_ok=True)

    summary = []
    for target in targets:
        src_path = os.path.join(base_route_dir, router_model, f"{target}.json")
        if not os.path.isfile(src_path):
            raise FileNotFoundError(f"Base route file not found: {src_path}")

        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        questions = [_clean_question(row) for row in data]
        enriched = []
        probs_pred_correct = 0
        original_match = 0

        for start in tqdm(
            range(0, len(questions), batch_size),
            desc=f"Exporting Bayes probs for {router_model}/{target}",
        ):
            batch_questions = questions[start : start + batch_size]
            batch_probs = exporter.predict_probs(batch_questions, max_input_length=max_input_length)
            for row, probs in zip(data[start : start + batch_size], batch_probs):
                enriched_row = dict(row)
                enriched_row = _attach_probs(
                    row=enriched_row,
                    probs=probs,
                    source=f"sidecar_{router_model}_probs",
                )
                enriched.append(enriched_row)

                gt = str(enriched_row.get("gt_retrieval", "")).strip().lower()
                if enriched_row["retrieval_probs_pred"] == gt:
                    probs_pred_correct += 1
                if enriched_row["retrieval_probs_match_original"]:
                    original_match += 1

        dst_path = os.path.join(router_output_dir, f"{target}.json")
        with open(dst_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=4, ensure_ascii=False)

        summary.append(
            {
                "target": target,
                "count": len(enriched),
                "prob_pred_acc": round(100.0 * probs_pred_correct / max(1, len(enriched)), 2),
                "match_original": round(100.0 * original_match / max(1, len(enriched)), 2),
                "output": dst_path,
            }
        )

    print(tabulate(summary, headers="keys", tablefmt="fancy_grid"))


def main():
    parser = argparse.ArgumentParser(
        description="Export probability-enriched route files for Bayes sidecar evaluation."
    )
    parser.add_argument("--router_model", type=str, required=True, choices=["distilbert", "t5-large"])
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--base_route_dir", type=str, default="route/results")
    parser.add_argument("--output_dir", type=str, default="route/results_bayes_probs")
    parser.add_argument(
        "--targets",
        type=str,
        default="mmlu,squad,natural_questions,hotpotqa,webqa",
        help="Comma-separated target list.",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.strip() or _default_checkpoint(args.router_model)
    targets = _parse_targets(args.targets)
    if not targets:
        raise ValueError("No targets provided.")

    export_probabilities(
        router_model=args.router_model,
        checkpoint_dir=checkpoint_dir,
        base_route_dir=args.base_route_dir,
        output_dir=args.output_dir,
        targets=targets,
        batch_size=int(args.batch_size),
        max_input_length=int(args.max_input_length),
        device=_resolve_device(args.device),
    )


if __name__ == "__main__":
    main()
