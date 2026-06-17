import os
import random
import json
import pickle
import argparse
import gc
from tqdm import tqdm
import importlib

random.seed(42)

from retrieve.retrieve_text import BGETextRetriever
from retrieve.retrieve_image import InternImgRetriever
from retrieve.retrieve_image_bge import BGEImageRetriever
# from retrieve.retrieve_clip import InternClipRetriever
# from retrieve.retrieve_video import InternVidRetriever


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


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model_module = self._load_model_module()
        self.model, self.processor, self.tokenizer = self.model_module.load_model(model_path)

    def _load_model_module(self):
        model_lower = self.model_path.lower()

        # OpenAI-compatible remote API backends. This is opt-in only; local
        # Ollama/HF Qwen paths keep using qwen2_5_vl below.
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

        # InternVL family
        elif "internvl" in model_lower:
            module_name = "internvl2_5"

        # Any Qwen-VL-style Ollama/HF name:
        #   qwen2.5vl:32b
        #   qwen3-vl:8b
        #   Qwen/Qwen2.5-VL-7B-Instruct
        elif "qwen" in model_lower and "vl" in model_lower:
            module_name = "qwen2_5_vl"

        # Phi vision family
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
    elif data_type in ["natural_questions", "hotpotqa", "squad", "triviaqa"]:
        return f"{query} Please respond with only the exact answer."
    elif data_type in {"webqa", "lara", "visual_rag"}:
        return f"{query} Please respond in a complete sentence."
    else:
        raise ValueError(f"Invalid data type: {data_type}")


def get_text_feature_paths(target, modality):
    if modality == "paragraph":
        if target == "triviaqa":
            return ["eval/features/text/triviaqa.pkl"]
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
    if target == "visual_rag":
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


def load_pickles(paths):
    data = {}
    for path in paths:
        with open(path, "rb") as f:
            data.update(pickle.load(f))
    return data


def get_env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    return default


def has_response(row):
    response = row.get("response")
    return response is not None and str(response).strip() != ""


def row_key(row):
    return (
        str(row.get("source", "")),
        str(row.get("index", "")),
        str(row.get("question", "")),
    )


def merge_existing_results(data, existing):
    existing_by_key = {row_key(row): row for row in existing if has_response(row)}
    restored = 0
    for row in data:
        old_row = existing_by_key.get(row_key(row))
        if not old_row:
            continue
        row["retrieved"] = old_row.get("retrieved", [])
        row["response"] = old_row["response"]
        restored += 1
    return restored


def save_json_atomic(path, data):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


def clear_inactive_retrievers(active_name, enabled):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        default="qwen3-vl:8b",
        help="Path or name of the model checkpoint / Ollama model name",
    )

    parser.add_argument(
        "--router_model",
        type=str,
        default="distilbert",
        choices=["gpt", "qwen", "t5-large", "distilbert", "selfrag", "adaptive_rag", "crag"],
        help="Router model to use",
    )

    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=SUPPORTED_TARGETS,
        help="Target dataset for evaluation",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of top retrievals to use",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Weight for image caption or clip/video script features (0 to 1)",
    )

    parser.add_argument(
        "--nframes",
        type=str,
        default="1",
        help="Frame setting tag used in output filename, e.g. '1' or 'clip:8,video:32'",
    )
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
        "--route_dir",
        type=str,
        default=os.environ.get("ROUTE_DIR", "route/results"),
        help="Directory containing routed files, organized as route_dir/router/target.json.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=os.environ.get("RESULTS_ROOT", "eval/results"),
        help="Root directory for evaluation outputs.",
    )
    parser.add_argument(
        "--allow_nonvisual_image_retrieval",
        action="store_true",
        help=(
            "Allow image retrieval for non-visual targets. By default, image routes "
            "on non-visual targets are mapped to document retrieval."
        ),
    )
    parser.add_argument(
        "--strict_image_grounding",
        action="store_true",
        help=(
            "For image retrieval, require the model to answer only from visible image "
            "evidence and output NOT_ANSWERABLE when the image is irrelevant. This is "
            "intended for forced-image diagnostics, not the main benchmark setting."
        ),
    )
    parser.add_argument(
        "--bge_image_retrieval",
        action="store_true",
        help=(
            "Use BGE-based caption retrieval for images instead of InternVideo2 "
            "cross-modal retrieval. Query features come from --query_bge_dir and "
            "caption features from eval/features/image/<target>_bge_captions.pkl."
        ),
    )

    args = parser.parse_args()

    model_path = args.model_path
    router_model = args.router_model
    target = args.target
    top_k = args.top_k
    alpha = args.alpha
    query_bge_dir = args.query_bge_dir
    query_internvideo_dir = args.query_internvideo_dir
    model_name = model_path.split("/")[-1]
    nframes_tag = args.nframes.replace(",", "_").replace(":", "")
    output_dir = f"{args.output_root}/{model_name}/{router_model}"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{target}_top{top_k}_{alpha}_{nframes_tag}.json"
    partial_file = f"{output_file}.partial"
    save_every = max(1, get_env_int("EVAL_SAVE_EVERY", 25))
    resume_eval = get_env_bool("EVAL_RESUME", True)
    single_retriever_cache = get_env_bool("EVAL_SINGLE_RETRIEVER_CACHE", False)

    print(
        f"LVLM Model: {model_path}, Router Model: {router_model}, "
        f"Target: {target}, Top-k: {top_k}, Alpha: {alpha}, NFrames: {args.nframes}"
    )

    model = ModelLoader(model_path)

    retriever_paragraph = None
    retriever_document = None
    retriever_image = None

    target_file = os.path.join(args.route_dir, router_model, f"{target}.json")
    if not os.path.exists(target_file):
        raise FileNotFoundError(
            f"Route result file not found: {target_file}\n"
            f"Please run the router first."
        )

    with open(target_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if resume_eval:
        resume_file = partial_file if os.path.exists(partial_file) else output_file
        if os.path.exists(resume_file):
            with open(resume_file, "r", encoding="utf-8") as f:
                restored = merge_existing_results(data, json.load(f))
            if restored:
                print(f"[INFO] Resuming from {resume_file}: restored {restored}/{len(data)} rows.")

    if single_retriever_cache:
        print("[INFO] EVAL_SINGLE_RETRIEVER_CACHE=1: keeping only the active retriever in memory.")

    completed_since_save = 0
    for row in tqdm(data, desc=f"Evaluating {target} with {model_path} + {router_model}"):
        if resume_eval and has_response(row):
            continue

        query = reformat(row)
        modality = str(row.get("retrieval", "error")).lower()

        # 如果路由出错，使用随机回退，保持和原逻辑一致
        if modality == "error":
            if target == "webqa":
                modality = random.choice(["no", "paragraph", "document", "image"])
            else:
                modality = random.choice(["no", "paragraph", "document"])

        # By default, non-visual targets use target-aware masking. For strict
        # Fixed-Image baselines, this guard can be disabled explicitly.
        if (
            not args.allow_nonvisual_image_retrieval
            and target not in {"webqa", "visual_rag"}
            and modality == "image"
        ):
            modality = "document"

        retrieved = []

        if modality == "no":
            response = model.inference(query)

        elif modality in ["paragraph", "document"]:
            if modality == "paragraph":
                clear_inactive_retrievers("paragraph", single_retriever_cache)
                if retriever_paragraph is None:
                    retriever_paragraph = BGETextRetriever(
                        queryfeats_path=os.path.join(query_bge_dir, f"{target}.pkl"),
                        textfeats_path=get_text_feature_paths(target, modality),
                    )
                retrieved, _ = retriever_paragraph.retrieve(row["index"], top_k=top_k)

            else:
                clear_inactive_retrievers("document", single_retriever_cache)
                if retriever_document is None:
                    retriever_document = BGETextRetriever(
                        queryfeats_path=os.path.join(query_bge_dir, f"{target}.pkl"),
                        textfeats_path=get_text_feature_paths(target, modality),
                    )
                retrieved, _ = retriever_document.retrieve(row["index"], top_k=top_k)

            retrieved_texts = []
            for doc in retrieved:
                with open(doc, "r", encoding="utf-8", errors="ignore") as f:
                    retrieved_texts.append(f.read()[:3000])

            response = model.inference(
                query,
                retrieved_texts=retrieved_texts,
                max_new_tokens=128,
            )

        elif modality == "image":
            clear_inactive_retrievers("image", single_retriever_cache)
            if retriever_image is None:
                if args.bge_image_retrieval:
                    bge_query_path = os.path.join(query_bge_dir, f"{target}.pkl")
                    bge_caption_path = os.path.join(
                        "eval/features/image", f"{target}_bge_captions.pkl"
                    )
                    if not os.path.exists(bge_caption_path):
                        raise FileNotFoundError(
                            f"BGE caption features not found: {bge_caption_path}. "
                            f"Run preprocess/extract_webqa_caption_feats.py first."
                        )
                    retriever_image = BGEImageRetriever(
                        queryfeats_path=bge_query_path,
                        captionfeats_path=bge_caption_path,
                    )
                else:
                    query_img_feat = os.path.join(query_internvideo_dir, f"{target}.pkl")
                    if not os.path.exists(query_img_feat):
                        raise FileNotFoundError(
                            f"Image query features not found: {query_img_feat}. "
                            f"Current sample requires image retrieval, but this target has no prepared image query features."
                        )

                    imgfeats_path, imgcapfeats_path = get_image_feature_paths(target)
                    retriever_image = InternImgRetriever(
                        queryfeats_path=query_img_feat,
                        imgfeats_path=imgfeats_path,
                        imgcapfeats_path=imgcapfeats_path,
                        alpha=alpha,
                    )

            candidate_images = row.get("candidate_images") if target == "visual_rag" else None
            retrieved, _ = retriever_image.retrieve(row["index"], top_k=top_k, candidate_ids=candidate_images)

            response = model.inference(
                query,
                retrieved_images=retrieved,
                max_new_tokens=128,
                strict_image_grounding=args.strict_image_grounding,
            )

        else:
            raise ValueError(f"Invalid modality: {modality}")

        row["retrieved"] = retrieved
        row["response"] = response
        completed_since_save += 1

        if completed_since_save % save_every == 0:
            save_json_atomic(partial_file, data)
            print(f"[INFO] Saved partial results to: {partial_file}")

    if completed_since_save:
        save_json_atomic(partial_file, data)

    save_json_atomic(output_file, data)
    if os.path.exists(partial_file):
        os.remove(partial_file)

    print(f"Saved results to: {output_file}")
