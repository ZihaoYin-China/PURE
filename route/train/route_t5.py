from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import torch.nn.functional as F

import argparse
import json
import os
import sys
from tqdm import tqdm
from tabulate import tabulate


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from route.gpt.prompt import ROUTER_PROMPT


ROUTER_LABELS = ["no", "paragraph", "document", "image"]

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _format_question(question):
    question = str(question)
    if "Classify the following query" in question:
        return question
    return ROUTER_PROMPT.format(query=question)


def route(questions, max_input_length=512):
    inputs = tokenizer(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
    ).to(device)

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id
    decoder_input_ids = torch.full(
        (len(questions), 1),
        decoder_start_token_id,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            decoder_input_ids=decoder_input_ids,
        )

    label_token_tensor = torch.tensor(label_token_ids, dtype=torch.long, device=device)
    label_logits = outputs.logits[:, 0, :].index_select(dim=-1, index=label_token_tensor)
    label_probs = F.softmax(label_logits.float(), dim=-1)
    best_indices = label_probs.argmax(dim=-1).tolist()
    confidences = label_probs.max(dim=-1).values.tolist()
    retrievals = [ROUTER_LABELS[index] for index in best_indices]
    return retrievals, confidences


def main(input_path, output_path, batch_size=128, max_input_length=512):
    overall_results = []

    for path in input_path:
        with open(path, "r") as file:
            data = json.load(file)

        count = {label: 0 for label in ROUTER_LABELS}
        correct_4class = 0
        correct_text_equiv = 0
        target = _target_from_path(path)
        retrieval_confs = 0.0

        questions = [_format_question(item["question"]) for item in data]
        for i in tqdm(
            range(0, len(questions), batch_size),
            desc=f"Routing {os.path.basename(path)} with {model.config._name_or_path}",
        ):
            batch_questions = questions[i : i + batch_size]
            batch_retrievals, batch_probabilities = route(
                batch_questions,
                max_input_length=max_input_length,
            )
            for j, (retrieval, probability) in enumerate(zip(batch_retrievals, batch_probabilities)):
                data[i + j]["retrieval"] = retrieval
                data[i + j]["retrieval_conf"] = probability
                gt = data[i + j]["gt_retrieval"].lower()
                if retrieval == gt:
                    correct_4class += 1
                if _route_match(target, retrieval, gt):
                    correct_text_equiv += 1
                count[retrieval] += 1
            retrieval_confs += sum(batch_probabilities)

        count["accuracy"] = round(correct_text_equiv / len(data), 4)
        count["accuracy_4class"] = round(correct_4class / len(data), 4)
        count["text_equiv_accuracy"] = round(correct_text_equiv / len(data), 4)
        count["text_equiv_applied"] = _is_text_equiv_target(target)
        count["avg_conf"] = round(retrieval_confs / len(data), 4)

        result_row = {"Path": os.path.basename(path)}
        result_row.update(count)
        overall_results.append(result_row)

        with open(os.path.join(output_path, os.path.basename(path)), "w") as outfile:
            json.dump(data, outfile, indent=4)

    print(tabulate(overall_results, headers="keys", tablefmt="fancy_grid"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default="route/train/checkpoints/t5-large")
    parser.add_argument("--input_dir", type=str, default="dataset/query")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="route/results")
    args = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained(args.checkpoint_dir)
    label_token_ids = []
    for label in ROUTER_LABELS:
        token_ids = tokenizer(label, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(f"Router label must be one token for constrained scoring: {label} -> {token_ids}")
        label_token_ids.append(token_ids[0])

    model = T5ForConditionalGeneration.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    model.eval()

    input_path = [os.path.join(args.input_dir, fname) for fname in os.listdir(args.input_dir) if fname.endswith(".json")]
    model_size = os.path.basename(args.checkpoint_dir)
    output_path = os.path.join(args.output_dir, model_size)
    os.makedirs(output_path, exist_ok=True)

    main(input_path, output_path, args.batch_size, args.max_input_length)
