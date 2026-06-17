from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    T5ForConditionalGeneration,
    T5Tokenizer,
    set_seed,
)
from datasets import ClassLabel, Dataset
import argparse
import json
import logging
import os
import shutil

import numpy as np
import torch


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROUTER_LABELS = ["no", "paragraph", "document", "image"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-base")
    parser.add_argument("--input_dir", type=str, default="route/train/data/train_data_t5.json")
    parser.add_argument("--train_size", type=float, default=0.9)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=8)
    parser.add_argument("--generation_max_length", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="route/train/temp")
    parser.add_argument("--checkpoint_dir", type=str, default="route/train/checkpoints")
    parser.add_argument("--resume_from_checkpoint", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", type=str, default="auto", choices=["auto", "fp16", "bf16", "no"])
    parser.add_argument("--optim", type=str, default="adafactor", choices=["adafactor", "adamw_torch"])

    args = parser.parse_args()
    model_dir_name = args.model_name.split("/")[-1].replace("flan-", "")
    args.output_dir = os.path.join(args.output_dir, model_dir_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, model_dir_name)

    set_seed(args.seed)

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    logger.info("Loaded pretrained model: %s", args.model_name)

    label_token_ids = []
    for label in ROUTER_LABELS:
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Router label must be one token for constrained scoring: {label} -> {token_ids}")
        label_token_ids.append(token_ids[0])
    label_id_to_index = {token_id: index for index, token_id in enumerate(label_token_ids)}

    use_bf16 = False
    use_fp16 = False
    if args.mixed_precision == "bf16":
        use_bf16 = True
    elif args.mixed_precision == "fp16":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            logger.warning("fp16 is unstable for T5 on this GPU; using bf16 instead.")
            use_bf16 = True
        else:
            use_fp16 = True
    elif args.mixed_precision == "auto":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            use_bf16 = True
        elif torch.cuda.is_available():
            use_fp16 = True

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    with open(args.input_dir, "r") as file:
        dataset_rows = json.load(file)
    dataset = Dataset.from_list(dataset_rows)
    class_label = ClassLabel(names=ROUTER_LABELS)
    dataset = dataset.filter(lambda x: x["source"] in ROUTER_LABELS and x["gt_retrieval"] in ROUTER_LABELS)
    dataset = dataset.map(lambda x: {"source_label": class_label.str2int(x["source"])}, remove_columns=["source"])
    dataset = dataset.cast_column("source_label", class_label)
    split_dataset = dataset.train_test_split(train_size=args.train_size, stratify_by_column="source_label", seed=args.seed)

    def _preprocess_data(examples):
        model_inputs = tokenizer(
            examples["question"],
            max_length=args.max_input_length,
            truncation=True,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["gt_retrieval"],
                max_length=args.max_target_length,
                truncation=True,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = split_dataset["train"].map(
        _preprocess_data,
        batched=True,
        remove_columns=split_dataset["train"].column_names,
    )
    val_dataset = split_dataset["test"].map(
        _preprocess_data,
        batched=True,
        remove_columns=split_dataset["test"].column_names,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )

    def _preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        token_ids = torch.tensor(label_token_ids, device=logits.device)
        return logits[:, 0, :].index_select(dim=-1, index=token_ids)

    def _compute_metrics(eval_pred):
        predictions, labels = eval_pred
        pred_indices = np.argmax(predictions, axis=-1)
        true_indices = []
        for label_row in labels:
            first_token = None
            for token_id in label_row:
                if token_id != -100 and token_id != tokenizer.pad_token_id:
                    first_token = int(token_id)
                    break
            true_indices.append(label_id_to_index.get(first_token, -1))
        true_indices = np.asarray(true_indices)
        valid = true_indices >= 0
        matches = pred_indices[valid] == true_indices[valid]
        valid_count = int(np.sum(valid)) if len(valid) else 0
        total_count = int(matches.size)
        return {
            "accuracy": float(np.mean(matches)) if total_count else 0.0,
            "invalid_label_rate": float(1.0 - valid_count / len(valid)) if len(valid) else 0.0,
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        save_strategy="steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        save_total_limit=args.save_total_limit,
        fp16=use_fp16,
        bf16=use_bf16,
        optim=args.optim,
        logging_nan_inf_filter=False,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        predict_with_generate=False,
        report_to=args.report_to,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics,
        preprocess_logits_for_metrics=_preprocess_logits_for_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.checkpoint_dir)
    logger.info("Model saved to: %s", args.checkpoint_dir)

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)


if __name__ == "__main__":
    main()
