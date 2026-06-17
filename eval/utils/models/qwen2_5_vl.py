import os
import re
import base64
import time
import requests

from utils.utils import get_scripts_for_videos


def _encode_image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _truncate_text(text, max_chars=3000):
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def _is_qwen3_model(model_name: str) -> bool:
    model_name = str(model_name).lower()
    return "qwen3" in model_name and "vl" in model_name


def _extract_ollama_fields(data):
    message = data.get("message", {}) or {}
    content = str(message.get("content", "") or "").strip()
    thinking = str(message.get("thinking", "") or "").strip()
    done_reason = str(data.get("done_reason", "") or "").strip()
    return content, thinking, done_reason


def _normalize_text(s: str) -> str:
    s = str(s or "").lower()
    s = re.sub(r"<think>|</think>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_mcq_choices(query: str):
    """
    Parse choices like:
    Choices:
    A) xxx
    B) yyy
    C) zzz
    D) www
    """
    text = str(query or "")
    choices = {}

    # line-based parse first
    for line in text.splitlines():
        m = re.match(r"^\s*([A-D])\)\s*(.+?)\s*$", line)
        if m:
            choices[m.group(1).upper()] = m.group(2).strip()

    if len(choices) == 4:
        return choices

    # fallback regex over whole text. MMLU rows often keep all choices on one line.
    matches = re.findall(
        r"([A-D])\)\s*(.*?)(?=(?:\s+[A-D]\))|$)",
        text,
        flags=re.DOTALL,
    )
    for letter, option_text in matches:
        option_text = option_text.strip()
        if option_text:
            option_text = option_text.split("Please respond with only")[0].strip()
            choices[letter.upper()] = option_text

    return choices


def _extract_letter_from_thinking_with_choices(thinking: str, query: str):
    """
    Use the reasoning text to infer which option it supports.
    """
    if not thinking:
        return ""

    choices = _parse_mcq_choices(query)
    if not choices:
        return ""

    thinking_norm = _normalize_text(thinking)

    # 1) direct option letter mention
    direct_patterns = [
        r"\boption\s*([A-D])\b",
        r"\bchoice\s*([A-D])\b",
        r"\banswer\s*[:：]?\s*([A-D])\b",
        r"\bfinal answer\s*[:：]?\s*([A-D])\b",
        r"\bthe correct answer is\s*([A-D])\b",
        r"\b([A-D])\)\b",
    ]
    for pat in direct_patterns:
        m = re.search(pat, thinking, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # 2) option text matching
    scores = {}
    for letter, option_text in choices.items():
        opt_norm = _normalize_text(option_text)
        if not opt_norm:
            continue

        score = 0

        # exact normalized option appears in reasoning
        if opt_norm in thinking_norm:
            score += 10

        # token overlap
        opt_tokens = set(opt_norm.split())
        think_tokens = set(thinking_norm.split())
        overlap = len(opt_tokens & think_tokens)
        score += overlap

        # stronger cues around "fit", "best", "correct"
        cue_patterns = [
            rf"{re.escape(opt_norm)}\s+seems\s+to\s+fit\s+best",
            rf"{re.escape(opt_norm)}\s+fits?\s+best",
            rf"{re.escape(opt_norm)}\s+is\s+correct",
            rf"{re.escape(opt_norm)}\s+would\s+fit",
            rf"{re.escape(opt_norm)}\s+makes\s+sense",
        ]
        for pat in cue_patterns:
            if re.search(pat, thinking_norm, flags=re.IGNORECASE):
                score += 20

        # "Option X: text"
        if re.search(rf"option\s+{letter.lower()}[^a-z0-9]+{re.escape(opt_norm)}", thinking_norm):
            score += 15

        scores[letter] = score

    if not scores:
        return ""

    best_letter = max(scores, key=scores.get)
    if scores[best_letter] > 0:
        return best_letter

    return ""


def _extract_answer_from_text(text: str, query: str = "", answer_mode: str = "generic"):
    if not text:
        return ""

    text = str(text).strip()

    if answer_mode == "mcq_letter":
        # first try explicit letter
        patterns = [
            r"final answer\s*(?:is)?\s*[:：]?\s*\(?([A-D])\)?[\).]?",
            r"answer\s*(?:is)?\s*[:：]?\s*\(?([A-D])\)?[\).]?",
            r"correct answer\s*(?:is)?\s*[:：]?\s*\(?([A-D])\)?[\).]?",
            r"option\s*([A-D])\b",
            r"\b([A-D])\b[\).]?\s*$",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()

        # then try infer from option text in reasoning
        inferred = _extract_letter_from_thinking_with_choices(text, query)
        if inferred:
            return inferred

        return ""

    if answer_mode == "exact_short":
        return _clean_exact_short_answer(text)

    # generic fallback
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        first = lines[0]
        first = re.sub(r"^(final answer|answer|response)\s*[:：]\s*", "", first, flags=re.IGNORECASE).strip()
        return first

    return ""


def _build_payload(
    model_name,
    messages,
    images_b64,
    temperature,
    num_predict,
    num_ctx,
    think=None,
):
    if images_b64:
        msgs = []
        for i, m in enumerate(messages):
            mm = dict(m)
            if i == len(messages) - 1 and mm.get("role") == "user":
                mm["images"] = images_b64
            msgs.append(mm)
    else:
        msgs = messages

    payload = {
        "model": model_name,
        "messages": msgs,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "num_ctx": num_ctx,
        },
    }

    if think is not None:
        payload["think"] = think

    return payload


def _get_env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _post_ollama_chat(base_url, payload, model_name):
    max_retries = max(1, _get_env_int("OLLAMA_HTTP_MAX_RETRIES", 6))
    timeout = max(1, _get_env_int("OLLAMA_HTTP_TIMEOUT", 600))
    retry_backoff = max(0.1, _get_env_float("OLLAMA_HTTP_RETRY_BACKOFF", 3.0))
    retry_backoff_max = max(retry_backoff, _get_env_float("OLLAMA_HTTP_RETRY_BACKOFF_MAX", 30.0))
    retry_status_codes = {502, 503, 504}

    last_error = None
    last_status_code = None
    last_response_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            if resp.status_code == 200:
                return resp.json()

            last_status_code = resp.status_code
            last_response_text = resp.text
            if resp.status_code not in retry_status_codes or attempt == max_retries:
                break
        except requests.RequestException as e:
            last_error = e
            if attempt == max_retries:
                break

        sleep_s = min(retry_backoff_max, retry_backoff * (2 ** (attempt - 1)))
        print(
            f"[WARN] Ollama request attempt {attempt}/{max_retries} failed "
            f"for model={model_name} at {base_url}. Retrying in {sleep_s:.1f}s..."
        )
        time.sleep(sleep_s)

    if last_status_code is not None:
        raise RuntimeError(
            f"Ollama HTTP {last_status_code}: {last_response_text}\n"
            f"base_url={base_url}, model={model_name}\n"
            f"retries={max_retries}\n"
            f"payload_preview={str(payload)[:1000]}"
        )

    raise RuntimeError(
        f"Ollama request failed: {last_error}\n"
        f"base_url={base_url}, model={model_name}\n"
        f"retries={max_retries}"
    ) from last_error


def _make_final_answer_only_messages(query, answer_mode="generic"):
    if answer_mode == "mcq_letter":
        user_text = (
            "Select the correct option for the multiple-choice question below.\n"
            "Reply with exactly one uppercase letter: A, B, C, or D.\n"
            "Do not explain. Do not think step by step. Do not add any other text.\n\n"
            f"{query}\n\n"
            "Answer:"
        )
    elif answer_mode == "exact_short":
        user_text = (
            "Answer the following question.\n"
            "Return only the exact short answer.\n"
            "Do not provide explanation.\n"
            "Do not provide analysis.\n"
            "Do not provide thinking.\n\n"
            f"{query}"
        )
    else:
        user_text = (
            "Answer the following query.\n"
            "Return only the final answer.\n"
            "Do not provide explanation.\n"
            "Do not provide analysis.\n"
            "Do not provide thinking.\n\n"
            f"{query}"
        )

    return [
        {
            "role": "system",
            "content": (
                "You are a strict multiple-choice answerer. "
                "Return only the final answer. "
                "For multiple-choice questions, output exactly one uppercase letter from A, B, C, D. "
                "Do not provide reasoning, analysis, acknowledgement, or thinking."
            ),
        },
        {
            "role": "user",
            "content": user_text,
        },
    ]


def _infer_answer_mode(query: str) -> str:
    q = str(query or "")
    if re.search(
        r"Please respond with only a single letter(?:\s*\([A-Z](?:\s*-\s*[A-Z])?\))?",
        q,
        flags=re.IGNORECASE,
    ):
        return "mcq_letter"
    if "Please respond with only the exact answer." in q:
        return "exact_short"
    return "generic"


def _clean_exact_short_answer(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    answer = lines[0]
    answer = re.sub(
        r"^(?:final answer|answer|response)\s*[:：]\s*",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    answer = re.sub(
        r"^(?:the exact short answer is|the exact answer is|exact answer is|the answer is|answer is|it is|it's)\s+",
        "",
        answer,
        flags=re.IGNORECASE,
    ).strip()
    answer = answer.strip(" \t\"'`")
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.sub(r"[\s\.;:,!?]+$", "", answer).strip()
    return answer


def _should_force_final_answer_pass(answer_mode: str) -> bool:
    if answer_mode == "mcq_letter":
        return _get_env_bool("QWEN_FORCE_FINAL_ANSWER_PASS_MCQ", False)
    if answer_mode == "exact_short":
        return _get_env_bool("QWEN_FORCE_FINAL_ANSWER_PASS_EXACT_SHORT", False)
    return False


def _run_final_answer_only_retry(
    *,
    base_url,
    model_name,
    query,
    answer_mode,
    images_b64,
    requested_num_predict,
    num_ctx,
    is_qwen3,
):
    retry_messages = _make_final_answer_only_messages(query, answer_mode=answer_mode)
    retry_num_predict = max(requested_num_predict, 32 if answer_mode == "mcq_letter" else 96)

    retry_payload = _build_payload(
        model_name=model_name,
        messages=retry_messages,
        images_b64=images_b64,
        temperature=0.0,
        num_predict=retry_num_predict,
        num_ctx=num_ctx,
        think=False if is_qwen3 else None,
    )
    retry_data = _post_ollama_chat(base_url, retry_payload, model_name)
    retry_content, retry_thinking, retry_done_reason = _extract_ollama_fields(retry_data)
    salvage_text = retry_content or retry_thinking
    retry_answer = _extract_answer_from_text(
        salvage_text,
        query=query,
        answer_mode=answer_mode,
    ) if salvage_text else ""
    return retry_data, retry_content, retry_thinking, retry_done_reason, retry_answer


def load_model(model_path):
    model = {
        "backend": "ollama",
        "model_name": model_path,
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    }
    processor = None
    tokenizer = None
    return model, processor, tokenizer


def inference(model, processor, tokenizer, query, **kwargs):
    """
    Compatible with:
    - qwen2.5vl:32b
    - qwen3-vl:8b
    """
    if not isinstance(model, dict) or model.get("backend") != "ollama":
        raise ValueError("Invalid Ollama model config.")

    base_url = model["base_url"].rstrip("/")
    model_name = model["model_name"]
    is_qwen3 = _is_qwen3_model(model_name)

    images_b64 = []

    if "retrieved_texts" in kwargs:
        retrieved_texts = kwargs["retrieved_texts"]
        retrieved_texts = [_truncate_text(t, max_chars=3000) for t in retrieved_texts[:1]]

        doc_text = "\n\n".join(
            [f"Relevant document {idx + 1}:\n{text}" for idx, text in enumerate(retrieved_texts)]
        )
        query = (
            "Answer the question using the retrieved document.\n"
            "Keep the answer short and exact.\n\n"
            f"{doc_text}\n\nQuestion:\n{query}"
        )

    elif "retrieved_images" in kwargs:
        retrieved_images = kwargs["retrieved_images"][:1]

        if kwargs.get("use_caption", False):
            img_metadata = kwargs.get("img_metadata", {})
            caption_texts = []
            for idx, image_path in enumerate(retrieved_images):
                caption = img_metadata.get(image_path, {}).get("caption", "")
                caption_texts.append(
                    f"Relevant image {idx + 1} caption:\n{_truncate_text(caption, max_chars=1000)}"
                )
                images_b64.append(_encode_image_to_base64(image_path))
            query = (
                "Considering the given image and caption,\n"
                + "\n\n".join(caption_texts)
                + "\n\n"
                + query
            )
        else:
            for image_path in retrieved_images:
                images_b64.append(_encode_image_to_base64(image_path))
            if kwargs.get("strict_image_grounding", False):
                query = (
                    "Answer the question using only evidence that is directly visible in the provided image. \n"
                    "If the image does not contain enough information to answer, output exactly NOT_ANSWERABLE. \n"
                    "Do not use outside knowledge.\n\n"
                    f"Question:\n{query}"
                )
            else:
                query = f"Considering the given image,\n{query}"

    elif "retrieved_videos" in kwargs:
        if kwargs.get("use_scripts", False):
            scripts = get_scripts_for_videos(
                kwargs["retrieved_videos"],
                kwargs.get("startend_times"),
            )
            scripts = [_truncate_text(s, max_chars=1500) for s in scripts[:1]]
            script_text = "\n\n".join(
                [f"Relevant video {idx + 1}:\n{script}" for idx, script in enumerate(scripts)]
            )
            query = f"Considering the given video script,\n{script_text}\n\n{query}"
        else:
            raise NotImplementedError(
                "Ollama backend currently does not support direct video input in this file. "
                "Please enable use_scripts=True or convert videos to text first."
            )

    user_message = {
        "role": "user",
        "content": query,
    }
    messages = [user_message]

    temperature = kwargs.get("temperature", 0.0)
    requested_num_predict = kwargs.get("max_new_tokens", 128)
    num_ctx = kwargs.get("num_ctx", 8192)
    answer_mode = _infer_answer_mode(query)

    if is_qwen3:
        first_num_predict = max(requested_num_predict, 256)
        first_think = False
    else:
        first_num_predict = requested_num_predict
        first_think = None

    payload = _build_payload(
        model_name=model_name,
        messages=messages,
        images_b64=images_b64,
        temperature=temperature,
        num_predict=first_num_predict,
        num_ctx=num_ctx,
        think=first_think,
    )

    data = _post_ollama_chat(base_url, payload, model_name)
    content, thinking, done_reason = _extract_ollama_fields(data)
    initial_answer = _extract_answer_from_text(content, query=query, answer_mode=answer_mode) if content else ""
    force_final_answer_pass = _should_force_final_answer_pass(answer_mode)

    if content and force_final_answer_pass:
        retry_data, retry_content, retry_thinking, retry_done_reason, retry_answer = _run_final_answer_only_retry(
            base_url=base_url,
            model_name=model_name,
            query=query,
            answer_mode=answer_mode,
            images_b64=images_b64,
            requested_num_predict=requested_num_predict,
            num_ctx=num_ctx,
            is_qwen3=is_qwen3,
        )

        if answer_mode == "mcq_letter":
            if re.fullmatch(r"[A-D]", retry_answer or ""):
                return retry_answer
            if re.fullmatch(r"[A-D]", initial_answer or ""):
                return initial_answer
        elif answer_mode == "exact_short":
            if retry_answer:
                return retry_answer
            if initial_answer:
                return initial_answer
        else:
            if retry_content:
                return retry_content.strip()
            if content:
                return content.strip()

        raise RuntimeError(
            "Empty refined response from Ollama.\n"
            f"answer_mode={answer_mode}, first_done_reason={done_reason}, "
            f"retry_done_reason={retry_done_reason}\n"
            f"first_response={data}\n"
            f"retry_response={retry_data}"
        )

    if content:
        if answer_mode == "mcq_letter" and re.fullmatch(r"[A-D]", initial_answer or ""):
            return initial_answer
        if answer_mode == "exact_short" and initial_answer:
            return initial_answer
        return content.strip()

    if thinking:
        salvaged = _extract_answer_from_text(thinking, query=query, answer_mode=answer_mode)
        if answer_mode == "mcq_letter":
            if re.fullmatch(r"[A-D]", salvaged or ""):
                return salvaged
        else:
            if salvaged:
                return salvaged

    if thinking or done_reason == "length":
        retry_data, retry_content, retry_thinking, retry_done_reason, retry_answer = _run_final_answer_only_retry(
            base_url=base_url,
            model_name=model_name,
            query=query,
            answer_mode=answer_mode,
            images_b64=images_b64,
            requested_num_predict=requested_num_predict,
            num_ctx=num_ctx,
            is_qwen3=is_qwen3,
        )

        if retry_content:
            if answer_mode == "mcq_letter" and re.fullmatch(r"[A-D]", retry_answer or ""):
                return retry_answer
            if answer_mode == "exact_short" and retry_answer:
                return retry_answer
            return retry_content.strip()

        salvage_text = retry_thinking or thinking
        if salvage_text:
            salvaged = retry_answer or _extract_answer_from_text(
                salvage_text,
                query=query,
                answer_mode=answer_mode,
            )
            if answer_mode == "mcq_letter":
                if re.fullmatch(r"[A-D]", salvaged or ""):
                    return salvaged
            else:
                if salvaged:
                    return salvaged

        raise RuntimeError(
            "Empty response from Ollama after retry.\n"
            f"first_done_reason={done_reason}, retry_done_reason={retry_done_reason}\n"
            f"first_response={data}\n"
            f"retry_response={retry_data}"
        )

    raise RuntimeError(f"Empty response from Ollama: {data}")
