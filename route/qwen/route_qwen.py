import os
import re
import sys
import json
import time
import requests
from tqdm import tqdm
from tabulate import tabulate

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpt.prompt import ROUTER_PROMPT

ALLOWED_METHODS = ["no", "paragraph", "document", "image"]
ALL_METHODS = ["no", "paragraph", "document", "image", "clip", "video", "error"]


def clean_response_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
    text = text.replace("```", "").strip()
    return text


def parse_retrieval(text: str) -> str:
    text = clean_response_text(text).lower()

    if not text:
        return "error"

    if text in ALLOWED_METHODS:
        return text

    patterns = [
        r"\b(no|paragraph|document|image)\b",
        r"category\s*[:：]?\s*(no|paragraph|document|image)\b",
        r"answer\s*[:：]?\s*(no|paragraph|document|image)\b",
        r"classification\s*[:：]?\s*(no|paragraph|document|image)\b",
        r"the\s+category\s+is\s+(no|paragraph|document|image)\b",
    ]

    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).lower()

    return "error"


def call_ollama_router(
    prompt_text,
    model_name,
    base_url,
    num_ctx=4096,
    num_predict=128,
    timeout=300,
):
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt_text
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        }
    }

    resp = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json=payload,
        timeout=timeout,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    message = data.get("message", {}) or {}

    raw_text = str(message.get("content", "") or "").strip()
    if raw_text:
        return raw_text

    thinking_text = str(message.get("thinking", "") or "").strip()
    if thinking_text:
        return thinking_text

    if "response" in data and data["response"]:
        return str(data["response"]).strip()

    raise RuntimeError(f"Unexpected Ollama response format: {str(data)[:500]}")


def route_with_qwen(
    target,
    output_path,
    model_name,
    base_url,
    max_retries=3,
    num_ctx=4096,
    num_predict=128,
    debug_max_errors=10,
):
    with open(target, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []
    count = {rm: 0 for rm in ALL_METHODS}
    correct = 0
    shown_error_count = 0

    for idx, item in enumerate(tqdm(dataset, desc=f"Routing {os.path.basename(target)} with {model_name}")):
        query = item["question"]
        prompt_text = ROUTER_PROMPT.format(query=query)
        prompt_text += "\n\nOutput only one lowercase word from: no, paragraph, document, image."

        retrieval = "error"
        raw_response = ""
        last_error = ""

        for _ in range(max_retries):
            try:
                raw_response = call_ollama_router(
                    prompt_text=prompt_text,
                    model_name=model_name,
                    base_url=base_url,
                    num_ctx=num_ctx,
                    num_predict=num_predict,
                )
                retrieval = parse_retrieval(raw_response)
                if retrieval != "error":
                    break
                last_error = f"parse_failed: raw_response={raw_response[:300]}"
            except Exception as e:
                last_error = repr(e)
                time.sleep(0.2)

        count[retrieval] += 1

        if retrieval == str(item.get("gt_retrieval", "")).lower():
            correct += 1

        item["retrieval"] = retrieval
        item["retrieval_raw"] = raw_response
        item["retrieval_error"] = last_error if retrieval == "error" else ""
        results.append(item)

        if retrieval == "error" and shown_error_count < debug_max_errors:
            print("\n[DEBUG ERROR SAMPLE]")
            print(f"idx={idx}")
            print(f"question={query[:200]}")
            print(f"raw_response={raw_response[:300]}")
            print(f"last_error={last_error}")
            shown_error_count += 1

    count["accuracy"] = round(correct / len(dataset), 4) if len(dataset) > 0 else 0.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    result_row = {"Path": os.path.basename(target)}
    result_row.update(count)
    return result_row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="dataset/query", help="Directory containing the input data")
    parser.add_argument("--output_dir", type=str, default="route/results/qwen", help="Directory to save results")
    parser.add_argument(
        "--model_name",
        type=str,
        default=os.getenv("QWEN_ROUTER_MODEL", "qwen3-vl:8b"),
        help="Ollama model name"
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama base URL"
    )
    parser.add_argument("--max_retries", type=int, default=3, help="Max retries for each sample")
    parser.add_argument("--num_ctx", type=int, default=int(os.getenv("QWEN_ROUTER_NUM_CTX", "4096")))
    parser.add_argument("--num_predict", type=int, default=int(os.getenv("QWEN_ROUTER_NUM_PREDICT", "128")))
    args = parser.parse_args()

    if os.path.isdir(args.input_dir):
        targets = [
            os.path.join(args.input_dir, fname)
            for fname in os.listdir(args.input_dir)
            if fname.endswith(".json")
        ]
        os.makedirs(args.output_dir, exist_ok=True)

        overall_results = []
        for target in targets:
            output_path = os.path.join(args.output_dir, os.path.basename(target))
            result_row = route_with_qwen(
                target=target,
                output_path=output_path,
                model_name=args.model_name,
                base_url=args.base_url,
                max_retries=args.max_retries,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
            )
            overall_results.append(result_row)

        print(tabulate(overall_results, headers="keys", tablefmt="fancy_grid"))

    elif os.path.isfile(args.input_dir) and args.input_dir.endswith(".json"):
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, os.path.basename(args.input_dir))

        result_row = route_with_qwen(
            target=args.input_dir,
            output_path=output_path,
            model_name=args.model_name,
            base_url=args.base_url,
            max_retries=args.max_retries,
            num_ctx=args.num_ctx,
            num_predict=args.num_predict,
        )
        print(tabulate([result_row], headers="keys", tablefmt="fancy_grid"))

    else:
        raise ValueError("Invalid input_dir. Please provide a valid JSON file or directory containing JSON files.")