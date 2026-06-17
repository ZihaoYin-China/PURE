import mimetypes
import json
import os
import re
import time
from types import SimpleNamespace

from utils.models.qwen2_5_vl import (
    _clean_exact_short_answer,
    _encode_image_to_base64,
    _extract_answer_from_text,
    _infer_answer_mode,
    _make_final_answer_only_messages,
    _should_force_final_answer_pass,
    _truncate_text,
)
from utils.utils import get_scripts_for_videos


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


def _is_data_inspection_failed(exc):
    text = str(exc or "").lower()
    return "data_inspection_failed" in text or "datainspectionfailed" in text


def _fallback_answer_for_blocked_output(answer_mode):
    override = (
        os.environ.get("GENERATOR_API_FALLBACK_TEXT")
        or os.environ.get("QWEN_API_DATA_INSPECTION_FALLBACK_TEXT")
    )
    if override is not None:
        return str(override)
    if answer_mode == "mcq_letter":
        return "A"
    if answer_mode == "complete_sentence":
        return "I don't know."
    return "unknown"


def _canonicalize_exact_short_answer(text):
    text = _clean_exact_short_answer(text)
    if not text:
        return text

    text = re.sub(
        r"^(?:approximately|about|around|roughly)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    small_numbers = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "11": "eleven",
        "12": "twelve",
        "13": "thirteen",
        "14": "fourteen",
        "15": "fifteen",
        "16": "sixteen",
        "17": "seventeen",
        "18": "eighteen",
        "19": "nineteen",
        "20": "twenty",
    }
    if re.fullmatch(r"\d+", text) and text in small_numbers:
        return small_numbers[text]
    return text


def _finalize_answer(answer, answer_mode):
    if answer_mode != "exact_short":
        return answer
    if _get_env_bool("QWEN_API_CANONICALIZE_EXACT_SHORT", False):
        return _canonicalize_exact_short_answer(answer)
    return _clean_exact_short_answer(answer)


API_PROVIDERS = {
    "qwen-api": {
        "label": "Qwen API",
        "env_prefix": "QWEN",
        "key_envs": ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "image_mode": "image",
        "supports_enable_thinking": True,
    },
    "dashscope": {
        "label": "DashScope API",
        "env_prefix": "QWEN",
        "key_envs": ("DASHSCOPE_API_KEY", "QWEN_API_KEY", "OPENAI_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "image_mode": "image",
        "supports_enable_thinking": True,
    },
    "openai": {
        "label": "OpenAI API",
        "env_prefix": "OPENAI",
        "key_envs": ("OPENAI_API_KEY",),
        "base_url": "https://api.openai.com/v1",
        "image_mode": "image",
        "supports_enable_thinking": False,
    },
    "gpt": {
        "label": "OpenAI API",
        "env_prefix": "OPENAI",
        "key_envs": ("OPENAI_API_KEY",),
        "base_url": "https://api.openai.com/v1",
        "image_mode": "image",
        "supports_enable_thinking": False,
    },
    "dmxapi": {
        "label": "DMXAPI",
        "env_prefix": "DMXAPI",
        "key_envs": ("DMXAPI_API_KEY", "DMX_API_KEY"),
        "base_url": "https://www.dmxapi.cn/v1",
        "image_mode": "image",
        "supports_enable_thinking": False,
        "auth_scheme": "",
        "force_requests": True,
    },
    "deepseek": {
        "label": "DeepSeek API",
        "env_prefix": "DEEPSEEK",
        "key_envs": ("DEEPSEEK_API_KEY",),
        "base_url": "https://api.deepseek.com",
        "image_mode": "caption",
        "supports_thinking": True,
        "supports_reasoning_effort": True,
        "supports_enable_thinking": False,
    },
    "glm": {
        "label": "GLM API",
        "env_prefix": "GLM",
        "key_envs": ("GLM_API_KEY", "ZHIPU_API_KEY", "BIGMODEL_API_KEY"),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "image_mode": "image",
        "image_url_format": "raw_base64",
        "supports_thinking": True,
        "supports_enable_thinking": False,
        "default_thinking_type": "enabled",
    },
    "zhipu": {
        "label": "Zhipu GLM API",
        "env_prefix": "GLM",
        "key_envs": ("GLM_API_KEY", "ZHIPU_API_KEY", "BIGMODEL_API_KEY"),
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "image_mode": "image",
        "image_url_format": "raw_base64",
        "supports_thinking": True,
        "supports_enable_thinking": False,
        "default_thinking_type": "enabled",
    },
    "openai-compatible": {
        "label": "OpenAI-compatible API",
        "env_prefix": "GENERATOR",
        "key_envs": ("GENERATOR_API_KEY", "OPENAI_COMPATIBLE_API_KEY"),
        "base_url": "https://api.openai.com/v1",
        "image_mode": "image",
        "supports_enable_thinking": False,
    },
}


def _provider_from_model_path(model_path):
    model_path = str(model_path or "").strip()
    model_lower = model_path.lower()
    for prefix in API_PROVIDERS:
        marker = f"{prefix}:"
        if model_lower.startswith(marker):
            return prefix, model_path[len(marker):]
    if model_lower.startswith("gpt-"):
        return "openai", model_path
    if model_lower.startswith("deepseek-"):
        return "deepseek", model_path
    if model_lower.startswith("glm-"):
        return "glm", model_path
    return "openai-compatible", model_path


def _first_env(names):
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _provider_env(provider_config, suffix):
    env_prefix = provider_config.get("env_prefix", "GENERATOR")
    return (
        os.environ.get(f"{env_prefix}_{suffix}")
        or os.environ.get(f"{env_prefix}_API_{suffix}")
        or os.environ.get(f"GENERATOR_{suffix}")
        or os.environ.get(f"GENERATOR_API_{suffix}")
    )


def _get_provider_float(provider_config, suffix, default):
    value = _provider_env(provider_config, suffix)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_provider_bool(provider_config, suffix, default):
    value = _provider_env(provider_config, suffix)
    if value is None or value == "":
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _get_provider_int(provider_config, suffix, default):
    value = _provider_env(provider_config, suffix)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


_IMAGE_METADATA_CACHE = None


def _load_image_metadata():
    global _IMAGE_METADATA_CACHE
    if _IMAGE_METADATA_CACHE is not None:
        return _IMAGE_METADATA_CACHE

    metadata = {}
    configured = os.environ.get("GENERATOR_API_IMAGE_METADATA")
    paths = []
    if configured:
        paths.extend([p for p in configured.split(os.pathsep) if p])
    paths.extend(
        [
            "dataset/WebQA/webqa_images.json",
            "dataset/visual_rag/images.json",
        ]
    )

    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                metadata.update(data)
        except Exception as exc:
            print(f"[WARN] Failed to load image metadata from {path}: {exc}")

    _IMAGE_METADATA_CACHE = metadata
    return metadata


def _caption_for_image(image_path, explicit_metadata=None):
    metadata = explicit_metadata or {}
    if image_path not in metadata:
        metadata = _load_image_metadata()

    info = metadata.get(image_path, {}) if isinstance(metadata, dict) else {}
    title = str(info.get("title", "") or "").strip()
    caption = str(info.get("caption", "") or "").strip()
    if title and caption and title.lower() not in caption.lower():
        return f"{title}. {caption}"
    return caption or title or os.path.basename(str(image_path))


def _image_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/jpeg"
    return f"data:{mime_type};base64,{_encode_image_to_base64(image_path)}"


def _image_url_for_provider(image_path, provider_config=None):
    provider_config = provider_config or {}
    image_url_format = (
        _provider_env(provider_config, "IMAGE_URL_FORMAT")
        or provider_config.get("image_url_format", "data_url")
    )
    image_url_format = str(image_url_format or "data_url").strip().lower()
    if image_url_format in {"raw", "raw_base64", "base64"}:
        return _encode_image_to_base64(image_path)
    return _image_data_url(image_path)


def _messages_for_query(query, image_paths=None, provider_config=None):
    image_paths = image_paths or []
    if not image_paths:
        return [{"role": "user", "content": query}]

    content = [{"type": "text", "text": query}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_url_for_provider(image_path, provider_config)},
            }
        )
    return [{"role": "user", "content": content}]


def _extract_choice_text(choice):
    message = getattr(choice, "message", None)
    if message is None:
        return "", ""
    content = str(getattr(message, "content", "") or "").strip()
    reasoning = str(getattr(message, "reasoning_content", "") or "").strip()
    return content, reasoning


class _RequestsChatCompletions:
    def __init__(self, api_key, base_url, timeout, auth_scheme="Bearer"):
        self.api_key = api_key
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.auth_scheme = str(auth_scheme or "").strip()

    def create(
        self,
        *,
        model,
        messages,
        temperature,
        max_tokens,
        extra_body=None,
        reasoning_effort=None,
    ):
        try:
            import requests
        except ImportError as exc:
            raise ImportError(
                "The OpenAI-compatible API backend requires either the openai package "
                "or requests. Install one of them in the active Python environment."
            ) from exc

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"{self.auth_scheme} {self.api_key}".strip(),
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        try:
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"OpenAI-compatible API HTTP {response.status_code}: {response.text[:1000]}"
            ) from exc

        data = response.json()
        choices = []
        for choice in data.get("choices", []) or []:
            message = choice.get("message", {}) or {}
            choices.append(
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=message.get("content", ""),
                        reasoning_content=message.get("reasoning_content", ""),
                    ),
                    finish_reason=choice.get("finish_reason", ""),
                )
            )
        return SimpleNamespace(choices=choices)


class _RequestsChat:
    def __init__(self, api_key, base_url, timeout, auth_scheme="Bearer"):
        self.completions = _RequestsChatCompletions(api_key, base_url, timeout, auth_scheme)


class _RequestsOpenAICompatibleClient:
    def __init__(self, *, api_key, base_url, timeout, auth_scheme="Bearer"):
        self.chat = _RequestsChat(api_key, base_url, timeout, auth_scheme)


def _chat_once(
    client,
    model_name,
    messages,
    *,
    temperature,
    max_tokens,
    extra_body=None,
    reasoning_effort=None,
):
    request_kwargs = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if extra_body:
        request_kwargs["extra_body"] = extra_body
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort
    completion = client.chat.completions.create(**request_kwargs)
    if not completion.choices:
        return "", "", "empty_choices"
    content, reasoning = _extract_choice_text(completion.choices[0])
    finish_reason = str(getattr(completion.choices[0], "finish_reason", "") or "")
    return content, reasoning, finish_reason


def _chat_with_retries(
    client,
    model_name,
    messages,
    *,
    temperature,
    max_tokens,
    provider_config,
    extra_body=None,
    reasoning_effort=None,
):
    provider_label = provider_config.get("label", "OpenAI-compatible API")
    max_retries = max(1, _get_provider_int(provider_config, "MAX_RETRIES", 6))
    retry_backoff = max(0.1, _get_provider_float(provider_config, "RETRY_BACKOFF", 3.0))
    retry_backoff_max = max(
        retry_backoff,
        _get_provider_float(provider_config, "RETRY_BACKOFF_MAX", 30.0),
    )

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            content, reasoning, finish_reason = _chat_once(
                client,
                model_name,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=extra_body,
                reasoning_effort=reasoning_effort,
            )
            if content or reasoning or finish_reason == "data_inspection_failed":
                return content, reasoning, finish_reason
            last_error = RuntimeError(f"empty response, finish_reason={finish_reason}")
        except Exception as exc:
            last_error = exc
            if _is_data_inspection_failed(exc) and _get_provider_bool(
                provider_config,
                "DATA_INSPECTION_FALLBACK",
                True,
            ):
                print(
                    f"[WARN] {provider_label} data inspection failed; "
                    "recording a fallback answer and continuing."
                )
                return "", "", "data_inspection_failed"

        if attempt == max_retries:
            break
        sleep_s = min(retry_backoff_max, retry_backoff * (2 ** (attempt - 1)))
        print(
            f"[WARN] {provider_label} request attempt {attempt}/{max_retries} failed "
            f"for model={model_name}: {last_error}. Retrying in {sleep_s:.1f}s..."
        )
        time.sleep(sleep_s)

    if _get_provider_bool(provider_config, "EMPTY_RESPONSE_FALLBACK", True):
        print(
            f"[WARN] {provider_label} returned an empty response after retries; "
            "recording a fallback answer and continuing."
        )
        return "", "", "empty_response"

    raise RuntimeError(
        f"{provider_label} request failed after {max_retries} attempts "
        f"for model={model_name}: {last_error}"
    ) from last_error


def _deepseek_default_thinking_type(model_name):
    model_name = str(model_name or "").lower()
    if model_name in {"deepseek-reasoner"} or model_name.endswith("-pro"):
        return "enabled"
    return None


def _deepseek_default_reasoning_effort(model_name):
    model_name = str(model_name or "").lower()
    if model_name in {"deepseek-reasoner"} or model_name.endswith("-pro"):
        return "medium"
    return None


def load_model(model_path):
    try:
        from openai import OpenAI
    except ImportError:
        OpenAI = _RequestsOpenAICompatibleClient

    provider, model_name = _provider_from_model_path(model_path)
    provider_config = dict(API_PROVIDERS[provider])
    if provider_config.get("force_requests"):
        OpenAI = _RequestsOpenAICompatibleClient
    provider_label = provider_config.get("label", "OpenAI-compatible API")

    api_key = _first_env(
        tuple(provider_config.get("key_envs", ()))
        + ("GENERATOR_API_KEY", "OPENAI_COMPATIBLE_API_KEY")
    )
    if not api_key:
        expected = ", ".join(provider_config.get("key_envs", ()))
        raise RuntimeError(
            f"Set one of {expected} for {provider_label} backend "
            f"or set GENERATOR_API_KEY."
        )
    if any(ord(ch) > 127 for ch in api_key) or "你的" in api_key:
        raise RuntimeError(
            f"{provider_label} API key must be the real ASCII API key, "
            "not a placeholder such as 'your-key'."
        )

    base_url = _provider_env(provider_config, "BASE_URL") or provider_config["base_url"]
    timeout = max(1.0, _get_provider_float(provider_config, "TIMEOUT", 120.0))
    image_mode = (_provider_env(provider_config, "IMAGE_MODE") or provider_config.get("image_mode", "image")).lower()

    enable_thinking = None
    if provider_config.get("supports_enable_thinking"):
        enable_thinking = _get_provider_bool(provider_config, "ENABLE_THINKING", False)

    thinking_type = None
    if provider_config.get("supports_thinking"):
        thinking_type = (
            _provider_env(provider_config, "THINKING")
            or _provider_env(provider_config, "THINKING_TYPE")
            or provider_config.get("default_thinking_type")
            or _deepseek_default_thinking_type(model_name)
        )
        if thinking_type is not None:
            thinking_type = str(thinking_type).strip().lower()
            if thinking_type in {"1", "true", "yes", "y", "on"}:
                thinking_type = "enabled"
            elif thinking_type in {"0", "false", "no", "n", "off"}:
                thinking_type = "disabled"

    reasoning_effort = None
    if provider_config.get("supports_reasoning_effort"):
        reasoning_effort = (
            _provider_env(provider_config, "REASONING_EFFORT")
            or _deepseek_default_reasoning_effort(model_name)
        )
        if reasoning_effort is not None:
            reasoning_effort = str(reasoning_effort).strip().lower()

    return {
        "backend": "openai_compatible_api",
        "provider": provider,
        "provider_config": provider_config,
        "client": OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            auth_scheme=provider_config.get("auth_scheme", "Bearer"),
        ) if OpenAI is _RequestsOpenAICompatibleClient else OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        ),
        "model_name": model_name,
        "base_url": base_url,
        "timeout": timeout,
        "image_mode": image_mode,
        "enable_thinking": enable_thinking,
        "thinking_type": thinking_type,
        "reasoning_effort": reasoning_effort,
    }, None, None


def inference(model, processor, tokenizer, query, **kwargs):
    if not isinstance(model, dict) or model.get("backend") not in {"qwen_api", "openai_compatible_api"}:
        raise ValueError("Invalid OpenAI-compatible API model config.")

    client = model["client"]
    model_name = model["model_name"]
    provider_config = model.get("provider_config", API_PROVIDERS["qwen-api"])
    enable_thinking = model.get("enable_thinking")
    extra_body = None
    if enable_thinking is not None:
        extra_body = {"enable_thinking": bool(enable_thinking)}
    thinking_type = model.get("thinking_type")
    if thinking_type:
        if extra_body is None:
            extra_body = {}
        extra_body["thinking"] = {"type": thinking_type}
    reasoning_effort = model.get("reasoning_effort")
    image_mode = str(model.get("image_mode", "image") or "image").lower()
    image_paths = []

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
        max_images = max(1, _get_provider_int(provider_config, "MAX_IMAGES", 1))
        retrieved_images = kwargs["retrieved_images"][:max_images]
        use_caption = kwargs.get("use_caption", False) or image_mode in {"caption", "both"}
        send_images = image_mode in {"image", "both"}

        caption_texts = []
        if use_caption:
            img_metadata = kwargs.get("img_metadata", {})
            for idx, image_path in enumerate(retrieved_images):
                caption = _caption_for_image(image_path, explicit_metadata=img_metadata)
                caption_texts.append(
                    f"Relevant image {idx + 1} caption:\n{_truncate_text(caption, max_chars=1000)}"
                )

        if send_images:
            image_paths = list(retrieved_images)

        if use_caption and send_images:
            query = (
                "Considering the given image and its retrieved caption,\n"
                + "\n\n".join(caption_texts)
                + "\n\n"
                + query
            )
        elif use_caption:
            query = (
                "Answer the question using the retrieved image caption as visual evidence.\n"
                "Keep the answer grounded in the caption; if the caption is insufficient, answer as best as possible.\n\n"
                + "\n\n".join(caption_texts)
                + "\n\nQuestion:\n"
                + query
            )
        elif kwargs.get("strict_image_grounding", False):
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
                "OpenAI-compatible API backend does not support direct video input. Use scripts instead."
            )

    answer_mode = _infer_answer_mode(query)
    if answer_mode == "mcq_letter" and not image_paths:
        messages = _make_final_answer_only_messages(query, answer_mode=answer_mode)
    else:
        messages = _messages_for_query(query, image_paths=image_paths, provider_config=provider_config)
    max_tokens = max(1, int(kwargs.get("max_new_tokens", 128)))
    temperature = kwargs.get("temperature", 0.0)
    if answer_mode == "mcq_letter":
        default_mcq_tokens = 256 if (reasoning_effort or thinking_type) else 32
        request_max_tokens = max(1, _get_provider_int(provider_config, "MCQ_MAX_TOKENS", default_mcq_tokens))
    else:
        request_max_tokens = max_tokens

    content, reasoning, finish_reason = _chat_with_retries(
        client,
        model_name,
        messages,
        temperature=temperature,
        max_tokens=request_max_tokens,
        provider_config=provider_config,
        extra_body=extra_body,
        reasoning_effort=reasoning_effort,
    )
    initial_text = content or reasoning
    initial_answer = (
        _extract_answer_from_text(initial_text, query=query, answer_mode=answer_mode)
        if initial_text else ""
    )

    if finish_reason in {"data_inspection_failed", "empty_response"}:
        return _fallback_answer_for_blocked_output(answer_mode)

    needs_final_answer_retry = _should_force_final_answer_pass(answer_mode)
    if answer_mode == "mcq_letter" and not initial_answer:
        needs_final_answer_retry = True

    if needs_final_answer_retry:
        retry_messages = _make_final_answer_only_messages(query, answer_mode=answer_mode)
        retry_extra_body = None
        if enable_thinking is not None:
            retry_extra_body = {"enable_thinking": False}
        elif provider_config.get("supports_thinking"):
            retry_extra_body = {"thinking": {"type": "disabled"}}
        retry_max_tokens = request_max_tokens if answer_mode == "mcq_letter" else max(max_tokens, 96)
        retry_content, retry_reasoning, retry_finish_reason = _chat_with_retries(
            client,
            model_name,
            retry_messages,
            temperature=0.0,
            max_tokens=retry_max_tokens,
            provider_config=provider_config,
            extra_body=retry_extra_body,
            reasoning_effort=reasoning_effort,
        )
        retry_answer = _extract_answer_from_text(
            retry_content or retry_reasoning,
            query=query,
            answer_mode=answer_mode,
        ) if (retry_content or retry_reasoning) else ""

        if retry_finish_reason in {"data_inspection_failed", "empty_response"}:
            if initial_answer:
                return _finalize_answer(initial_answer, answer_mode)
            return _fallback_answer_for_blocked_output(answer_mode)

        if retry_answer:
            return _finalize_answer(retry_answer, answer_mode)
        if initial_answer:
            return _finalize_answer(initial_answer, answer_mode)
        if answer_mode == "mcq_letter":
            msg = "MCQ response did not contain A-D after final-answer retry"
            if _get_provider_bool(provider_config, "INVALID_MCQ_FALLBACK", True):
                print(f"[WARN] {msg}; recording a fallback letter and continuing.")
                return _fallback_answer_for_blocked_output(answer_mode)
            raise RuntimeError(
                f"{msg}. first={str(content)[:200]!r}, retry={str(retry_content)[:200]!r}"
            )
        if retry_content:
            return retry_content.strip()
        raise RuntimeError(
            "Empty refined response from OpenAI-compatible API. "
            f"first_finish_reason={finish_reason}, retry_finish_reason={retry_finish_reason}"
        )

    if content:
        if answer_mode == "mcq_letter" and initial_answer:
            return initial_answer
        if answer_mode == "exact_short" and initial_answer:
            return _finalize_answer(initial_answer, answer_mode)
        return content.strip()

    if reasoning:
        salvaged = _extract_answer_from_text(reasoning, query=query, answer_mode=answer_mode)
        if salvaged:
            return _finalize_answer(salvaged, answer_mode)

    if _get_provider_bool(provider_config, "EMPTY_RESPONSE_FALLBACK", True):
        return _fallback_answer_for_blocked_output(answer_mode)
    raise RuntimeError(f"Empty response from OpenAI-compatible API. finish_reason={finish_reason}")
