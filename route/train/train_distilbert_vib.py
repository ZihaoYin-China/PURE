import argparse
import contextlib
import json
import os
import random
from collections import Counter

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from route.train.vib_prototype_router import (
    DEFAULT_LABELS,
    VIBPrototypeRouter,
    load_router_tokenizer,
    resolve_device,
    save_router_checkpoint,
    set_global_seed,
)


DATASET_TO_ROUTE = {
    "mmlu": "no",
    "squad": "paragraph",
    "natural_questions": "paragraph",
    "hotpotqa": "document",
    "webqa": "image",
}


def _extract_query_text(question: str) -> str:
    text = str(question or "").strip()
    marker = "Classify the following query:"
    if marker not in text:
        return text

    tail = text.split(marker, 1)[-1].strip()
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if not lines:
        return text

    if lines[-1].lower().startswith("provide only"):
        lines = lines[:-1]
    if not lines:
        return text
    return lines[0]


def _normalize_label(row):
    for key in ["gt_retrieval", "source", "source_label"]:
        value = row.get(key)
        if isinstance(value, str):
            value = value.strip().lower()
            if value in DEFAULT_LABELS:
                return value
            if value in DATASET_TO_ROUTE:
                return DATASET_TO_ROUTE[value]
    return None


class RouterDataset(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def _load_records(input_path: str):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    records = []
    for row in raw_data:
        label = _normalize_label(row)
        question = _extract_query_text(row.get("question", ""))
        if not label or not question:
            continue
        records.append({
            "question": question,
            "labels": DEFAULT_LABELS.index(label),
            "label_text": label,
        })
    return records


def _stratified_split(records, train_size: float, seed: int):
    grouped = {label: [] for label in DEFAULT_LABELS}
    for row in records:
        grouped[row["label_text"]].append(dict(row))

    rng = random.Random(seed)
    train_rows = []
    eval_rows = []
    for label in DEFAULT_LABELS:
        items = grouped[label]
        rng.shuffle(items)
        if not items:
            continue

        if len(items) == 1:
            train_count = 1
        else:
            train_count = int(len(items) * train_size)
            train_count = max(1, min(train_count, len(items) - 1))

        train_rows.extend(items[:train_count])
        eval_rows.extend(items[train_count:])

    rng.shuffle(train_rows)
    rng.shuffle(eval_rows)
    return train_rows, eval_rows


def _build_collate_fn(tokenizer, max_input_length: int):
    def _collate(rows):
        encoded = tokenizer(
            [row["question"] for row in rows],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        )
        encoded["labels"] = torch.tensor([row["labels"] for row in rows], dtype=torch.long)
        return encoded

    return _collate


def _move_batch_to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def _resolve_mixed_precision(mixed_precision: str):
    mode = str(mixed_precision or "no").strip().lower()
    if mode == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            mode = "bf16"
        elif torch.cuda.is_available():
            mode = "fp16"
        else:
            mode = "no"
    if mode not in {"no", "fp16", "bf16"}:
        raise ValueError(f"Unsupported mixed precision mode: {mixed_precision}")
    return mode


def _autocast_context(device: torch.device, mixed_precision: str):
    if device.type != "cuda" or mixed_precision == "no":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if mixed_precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_ce = 0.0
    total_kl = 0.0
    total_proto = 0.0
    total_evi = 0.0
    correct = 0
    total = 0
    num_labels = len(DEFAULT_LABELS)
    confusion = torch.zeros((num_labels, num_labels), dtype=torch.long)

    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch_to_device(batch, device)
            outputs = model(**batch)
            preds = outputs["probs"].argmax(dim=-1)
            labels = batch["labels"]

            batch_size = labels.size(0)
            total += batch_size
            correct += (preds == labels).sum().item()
            total_loss += outputs["loss"].item() * batch_size
            total_ce += outputs["ce_loss"].item() * batch_size
            total_kl += outputs["kl_loss"].item() * batch_size
            total_proto += outputs["proto_loss"].item() * batch_size
            total_evi += outputs["evi_loss"].item() * batch_size
            for gold, pred in zip(labels.detach().cpu().tolist(), preds.detach().cpu().tolist()):
                confusion[int(gold), int(pred)] += 1

    denom = max(total, 1)
    per_label = {}
    macro_f1 = 0.0
    macro_recall = 0.0
    for idx, label_name in enumerate(DEFAULT_LABELS):
        tp = int(confusion[idx, idx].item())
        fp = int(confusion[:, idx].sum().item() - tp)
        fn = int(confusion[idx, :].sum().item() - tp)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall) / (precision + recall)
        per_label[label_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(confusion[idx, :].sum().item()),
        }
        macro_f1 += f1
        macro_recall += recall

    macro_f1 /= len(DEFAULT_LABELS)
    macro_recall /= len(DEFAULT_LABELS)
    return {
        "loss": total_loss / denom,
        "ce_loss": total_ce / denom,
        "kl_loss": total_kl / denom,
        "proto_loss": total_proto / denom,
        "evi_loss": total_evi / denom,
        "accuracy": correct / denom,
        "macro_f1": macro_f1,
        "macro_recall": macro_recall,
        "per_label": per_label,
        "count": total,
    }


def _compute_class_weights(rows, mode: str, clip_min: float, clip_max: float):
    mode = str(mode or "none").strip().lower()
    counts = Counter(row["label_text"] for row in rows)
    if mode == "none":
        return None
    if mode not in {"balanced", "sqrt_balanced"}:
        raise ValueError(f"Unsupported class_weight_mode: {mode}")

    total = float(sum(max(0, counts.get(label, 0)) for label in DEFAULT_LABELS))
    if total <= 0:
        return None

    weights = []
    k = float(len(DEFAULT_LABELS))
    for label in DEFAULT_LABELS:
        c = float(max(1, counts.get(label, 0)))
        inv_freq = total / (k * c)
        if mode == "sqrt_balanced":
            inv_freq = inv_freq ** 0.5
        weights.append(inv_freq)

    mean_w = sum(weights) / max(len(weights), 1)
    weights = [w / max(mean_w, 1e-8) for w in weights]
    weights = [min(max(w, clip_min), clip_max) for w in weights]
    mean_w = sum(weights) / max(len(weights), 1)
    weights = [w / max(mean_w, 1e-8) for w in weights]
    return weights


def main():
    parser = argparse.ArgumentParser(description="Train isolated VIB+Prototype+Evidential router.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--input_path", type=str, default="route/train/data/train_data_distilbert_4class.json")
    parser.add_argument("--checkpoint_dir", type=str, default="route/train/checkpoints/distilbert_vib")
    parser.add_argument("--init_checkpoint_dir", type=str, default="route/train/checkpoints/distilbert")
    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--num_train_epochs", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1)
    parser.add_argument("--prototype_temperature", type=float, default=1.0)
    parser.add_argument("--prototype_margin", type=float, default=0.2)
    parser.add_argument("--kl_weight", type=float, default=1e-3)
    parser.add_argument("--proto_weight", type=float, default=0.1)
    parser.add_argument("--evi_weight", type=float, default=0.2)
    parser.add_argument("--proto_logit_scale", type=float, default=0.5)
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default="none",
        choices=["none", "balanced", "sqrt_balanced"],
    )
    parser.add_argument("--class_weight_clip_min", type=float, default=0.25)
    parser.add_argument("--class_weight_clip_max", type=float, default=4.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument(
        "--select_metric",
        type=str,
        default="accuracy",
        choices=["accuracy", "macro_f1"],
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["auto", "fp16", "bf16", "no"])
    args = parser.parse_args()

    device = resolve_device(args.device)
    mixed_precision = _resolve_mixed_precision(args.mixed_precision)
    set_global_seed(args.seed)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    tokenizer = load_router_tokenizer(args.model_name)
    records = _load_records(args.input_path)
    train_rows, eval_rows = _stratified_split(records=records, train_size=args.train_size, seed=args.seed)
    train_dataset = RouterDataset(train_rows)
    eval_dataset = RouterDataset(eval_rows)
    label_counts = Counter(row["label_text"] for row in records)
    train_label_counts = Counter(row["label_text"] for row in train_rows)
    class_weights = _compute_class_weights(
        rows=train_rows,
        mode=args.class_weight_mode,
        clip_min=float(args.class_weight_clip_min),
        clip_max=float(args.class_weight_clip_max),
    )

    collator = _build_collate_fn(tokenizer=tokenizer, max_input_length=args.max_input_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    model = VIBPrototypeRouter(
        backbone_name=args.model_name,
        num_labels=len(DEFAULT_LABELS),
        latent_dim=args.latent_dim,
        hidden_dropout_prob=args.hidden_dropout_prob,
        prototype_temperature=args.prototype_temperature,
        prototype_margin=args.prototype_margin,
        kl_weight=args.kl_weight,
        proto_weight=args.proto_weight,
        evi_weight=args.evi_weight,
        proto_logit_scale=args.proto_logit_scale,
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
        label_names=DEFAULT_LABELS,
    ).to(device)

    if args.gradient_checkpointing and hasattr(model.backbone, "gradient_checkpointing_enable"):
        model.backbone.gradient_checkpointing_enable()
        if hasattr(model.backbone, "config") and hasattr(model.backbone.config, "use_cache"):
            model.backbone.config.use_cache = False

    warmstart_info = None
    if args.init_checkpoint_dir and os.path.isdir(args.init_checkpoint_dir):
        router_state_path = os.path.join(args.init_checkpoint_dir, "router_model.pt")
        if os.path.isfile(router_state_path):
            try:
                state = torch.load(router_state_path, map_location=device, weights_only=True)
            except TypeError:
                state = torch.load(router_state_path, map_location=device)
            current_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in state.items()
                if key in current_state and tuple(current_state[key].shape) == tuple(value.shape)
            }
            incompatible = model.load_state_dict(compatible_state, strict=False)
            warmstart_info = {
                "missing_keys": list(incompatible.missing_keys),
                "unexpected_keys": list(incompatible.unexpected_keys),
                "loaded_keys": sorted(compatible_state.keys()),
            }
            print(f"Warm-start VIB checkpoint: {args.init_checkpoint_dir}")
            print(f"Warm-start loaded keys    : {len(warmstart_info['loaded_keys'])}")
            print(f"Warm-start missing keys   : {len(warmstart_info['missing_keys'])}")
            print(f"Warm-start extra keys     : {len(warmstart_info['unexpected_keys'])}")
        elif getattr(model, "backbone_model_type", "") == "distilbert":
            warmstart_info = model.load_distilbert_classifier_checkpoint(
                checkpoint_dir=args.init_checkpoint_dir,
                device=device,
            )
            print(f"Warm-start checkpoint   : {args.init_checkpoint_dir}")
            print(f"Warm-start missing keys : {len(warmstart_info['missing_keys'])}")
            print(f"Warm-start extra keys   : {len(warmstart_info['unexpected_keys'])}")
        else:
            print(f"Warm-start checkpoint   : <skipped for backbone={model.backbone_model_type}>")
    else:
        print("Warm-start checkpoint   : <none>")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    use_grad_scaler = device.type == "cuda" and mixed_precision == "fp16"
    grad_scaler = torch.cuda.amp.GradScaler(enabled=use_grad_scaler)

    print("============================================================")
    print(f"Model                 : {args.model_name}")
    print(f"Checkpoint dir        : {args.checkpoint_dir}")
    print(f"Device                : {device}")
    print(f"Train examples        : {len(train_dataset)}")
    print(f"Eval examples         : {len(eval_dataset)}")
    print(f"Label counts          : {dict(label_counts)}")
    print(f"Train label counts    : {dict(train_label_counts)}")
    print(f"Class weight mode     : {args.class_weight_mode}")
    print(f"Class weights         : {class_weights if class_weights is not None else '<none>'}")
    print(f"Latent dim            : {args.latent_dim}")
    print(f"KL / Proto / Evi      : {args.kl_weight} / {args.proto_weight} / {args.evi_weight}")
    print(f"Proto logit scale     : {args.proto_logit_scale}")
    print(f"Label smoothing       : {args.label_smoothing}")
    print(f"Select metric         : {args.select_metric}")
    print(f"Mixed precision       : {mixed_precision}")
    print(f"Grad accumulation     : {args.gradient_accumulation_steps}")
    print(f"Grad checkpointing    : {args.gradient_checkpointing}")
    print("============================================================")

    best_score = -1.0
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    history = []

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Training epoch {epoch}/{args.num_train_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress, start=1):
            batch = _move_batch_to_device(batch, device)
            with _autocast_context(device, mixed_precision):
                outputs = model(**batch)
                loss = outputs["loss"]
                scaled_loss = loss / max(1, args.gradient_accumulation_steps)

            if use_grad_scaler:
                grad_scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = (
                step % max(1, args.gradient_accumulation_steps) == 0
                or step == len(train_loader)
            )
            if should_step:
                if use_grad_scaler:
                    grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_norm)
                if use_grad_scaler:
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            progress.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ce": f"{outputs['ce_loss'].item():.4f}",
                "proto": f"{outputs['proto_loss'].item():.4f}",
            })

        train_loss = running_loss / max(len(train_loader), 1)
        eval_metrics = _evaluate(model=model, data_loader=eval_loader, device=device)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "eval_loss": round(eval_metrics["loss"], 6),
            "eval_accuracy": round(eval_metrics["accuracy"], 6),
            "eval_macro_f1": round(eval_metrics["macro_f1"], 6),
            "eval_macro_recall": round(eval_metrics["macro_recall"], 6),
            "eval_ce_loss": round(eval_metrics["ce_loss"], 6),
            "eval_kl_loss": round(eval_metrics["kl_loss"], 6),
            "eval_proto_loss": round(eval_metrics["proto_loss"], 6),
            "eval_evi_loss": round(eval_metrics["evi_loss"], 6),
            "eval_per_label": {
                label: {
                    "precision": round(stats["precision"], 6),
                    "recall": round(stats["recall"], 6),
                    "f1": round(stats["f1"], 6),
                    "support": int(stats["support"]),
                }
                for label, stats in eval_metrics["per_label"].items()
            },
        }
        history.append(epoch_summary)
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"eval_loss={eval_metrics['loss']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f} "
            f"eval_macro_f1={eval_metrics['macro_f1']:.4f}"
        )

        select_score = eval_metrics["accuracy"]
        if args.select_metric == "macro_f1":
            select_score = eval_metrics["macro_f1"]

        if select_score >= best_score:
            best_score = select_score
            best_accuracy = eval_metrics["accuracy"]
            best_macro_f1 = eval_metrics["macro_f1"]
            save_router_checkpoint(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=args.checkpoint_dir,
                metadata={
                    "best_epoch": epoch,
                    "best_score": best_score,
                    "best_select_metric": args.select_metric,
                    "best_accuracy": best_accuracy,
                    "best_macro_f1": best_macro_f1,
                    "train_size": len(train_dataset),
                    "eval_size": len(eval_dataset),
                    "warmstart": warmstart_info,
                    "args": vars(args),
                    "history": history,
                },
            )

    summary = {
        "best_score": best_score,
        "best_select_metric": args.select_metric,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "history": history,
        "label_counts": dict(label_counts),
        "warmstart": warmstart_info,
        "args": vars(args),
    }
    with open(os.path.join(args.checkpoint_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("============================================================")
    print(f"Best select score     : {best_score:.4f} ({args.select_metric})")
    print(f"Best eval accuracy    : {best_accuracy:.4f}")
    print(f"Best eval macro_f1    : {best_macro_f1:.4f}")
    print(f"Saved checkpoint      : {args.checkpoint_dir}")
    print("============================================================")


if __name__ == "__main__":
    main()
