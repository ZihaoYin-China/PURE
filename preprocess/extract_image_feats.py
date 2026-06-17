import os
import sys
import cv2
import json
import pickle
import warnings
import types
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _install_flash_attn_fallbacks():
    try:
        import flash_attn  # noqa: F401
        return
    except Exception:
        pass

    class _FallbackFusedMLP(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, heuristic=None, **kwargs):
            super().__init__()
            hidden_features = hidden_features or in_features
            out_features = out_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_features, out_features)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    class _FallbackDropoutAddRMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6, prenorm=True, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps
            self.prenorm = prenorm

        def forward(self, x, residual=None):
            if residual is None:
                residual = x
            y = x + residual
            variance = y.float().pow(2).mean(-1, keepdim=True)
            y = y.float() * torch.rsqrt(variance + self.eps)
            y = (self.weight * y).to(dtype=x.dtype)
            return y, residual

    def _unsupported(*args, **kwargs):
        raise RuntimeError(
            "flash-attn fallback stub was invoked at runtime. "
            "InternVideo tried to use flash-attn kernels instead of the expected non-flash path."
        )

    flash_attn_mod = types.ModuleType("flash_attn")
    flash_attn_interface_mod = types.ModuleType("flash_attn.flash_attn_interface")
    flash_attn_interface_mod.flash_attn_varlen_qkvpacked_func = _unsupported

    flash_attn_bert_padding_mod = types.ModuleType("flash_attn.bert_padding")
    flash_attn_bert_padding_mod.unpad_input = _unsupported
    flash_attn_bert_padding_mod.pad_input = _unsupported

    flash_attn_modules_mod = types.ModuleType("flash_attn.modules")
    flash_attn_modules_mlp_mod = types.ModuleType("flash_attn.modules.mlp")
    flash_attn_modules_mlp_mod.FusedMLP = _FallbackFusedMLP

    flash_attn_ops_mod = types.ModuleType("flash_attn.ops")
    flash_attn_ops_rms_mod = types.ModuleType("flash_attn.ops.rms_norm")
    flash_attn_ops_rms_mod.DropoutAddRMSNorm = _FallbackDropoutAddRMSNorm

    flash_attn_mod.flash_attn_interface = flash_attn_interface_mod
    flash_attn_mod.bert_padding = flash_attn_bert_padding_mod
    flash_attn_mod.modules = flash_attn_modules_mod
    flash_attn_mod.ops = flash_attn_ops_mod

    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.flash_attn_interface"] = flash_attn_interface_mod
    sys.modules["flash_attn.bert_padding"] = flash_attn_bert_padding_mod
    sys.modules["flash_attn.modules"] = flash_attn_modules_mod
    sys.modules["flash_attn.modules.mlp"] = flash_attn_modules_mlp_mod
    sys.modules["flash_attn.ops"] = flash_attn_ops_mod
    sys.modules["flash_attn.ops.rms_norm"] = flash_attn_ops_rms_mod


_install_flash_attn_fallbacks()


def _install_peft_fallback():
    try:
        import peft  # noqa: F401
        return
    except Exception:
        pass

    class _TaskType:
        CAUSAL_LM = "CAUSAL_LM"

    class _LoraConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _WrappedModel:
        def __init__(self, model):
            self.base_model = types.SimpleNamespace(
                model=types.SimpleNamespace(model=model.model)
            )

    def _get_peft_model(model, peft_config):
        return _WrappedModel(model)

    peft_mod = types.ModuleType("peft")
    peft_mod.get_peft_model = _get_peft_model
    peft_mod.LoraConfig = _LoraConfig
    peft_mod.TaskType = _TaskType
    sys.modules["peft"] = peft_mod


_install_peft_fallback()


def _install_open_clip_fallback():
    try:
        import open_clip  # noqa: F401
        return
    except Exception:
        pass

    class _DummyTokenizer:
        def __init__(self):
            self.encoder = {"<sot>": 0, "<eot>": 1}

        def __call__(self, text, context_length=77):
            if isinstance(text, str):
                ids = [0, 1]
                ids = ids + [0] * max(0, context_length - len(ids))
                return torch.tensor(ids[:context_length], dtype=torch.long)
            batch = []
            for _ in text:
                ids = [0, 1]
                ids = ids + [0] * max(0, context_length - len(ids))
                batch.append(ids[:context_length])
            return torch.tensor(batch, dtype=torch.long)

    def _get_tokenizer(model_name):
        return _DummyTokenizer()

    open_clip_mod = types.ModuleType("open_clip")
    open_clip_mod.get_tokenizer = _get_tokenizer
    sys.modules["open_clip"] = open_clip_mod


_install_open_clip_fallback()


def _resolve_internvideo_paths():
    """
    Resolve InternVideo2 root and multi_modality package path.

    Expected env:
      INTERNVIDEO_PATH=/Yin_zi_hao/code/InternVideo
    Then:
      internvideo_root = /Yin_zi_hao/code/InternVideo/InternVideo2
      mm_path          = /Yin_zi_hao/code/InternVideo/InternVideo2/multi_modality
    """
    base_path = os.getenv("INTERNVIDEO_PATH")
    if not base_path:
        raise EnvironmentError(
            "Environment variable `INTERNVIDEO_PATH` is not set.\n"
            "Example:\n"
            "export INTERNVIDEO_PATH=/Yin_zi_hao/code/InternVideo"
        )

    internvideo_root = os.path.join(base_path, "InternVideo2")
    mm_path = os.path.join(internvideo_root, "multi_modality")

    if not os.path.isdir(internvideo_root):
        raise FileNotFoundError(
            f"InternVideo2 root not found: {internvideo_root}\n"
            f"Please check INTERNVIDEO_PATH={base_path}"
        )

    if not os.path.isdir(mm_path):
        raise FileNotFoundError(
            f"multi_modality package path not found: {mm_path}\n"
            f"Please check your InternVideo2 repository layout."
        )

    return internvideo_root, mm_path


INTERNVIDEO_ROOT, INTERNVIDEO_MM_PATH = _resolve_internvideo_paths()

# Support both package-style imports (`multi_modality.*`) and the
# InternVideo config files historical top-level imports (`configs.*`).
if INTERNVIDEO_ROOT not in sys.path:
    sys.path.insert(0, INTERNVIDEO_ROOT)
if INTERNVIDEO_MM_PATH not in sys.path:
    sys.path.insert(0, INTERNVIDEO_MM_PATH)

try:
    from multi_modality.utils.config import Config, eval_dict_leaf
    from multi_modality.demo.utils import setup_internvideo2
except Exception as e:
    raise ImportError(
        "Failed to import InternVideo2 modules in package mode.\n"
        "Please also check /Yin_zi_hao/code/InternVideo/InternVideo2/multi_modality/demo/utils.py\n"
        "and make sure imports like:\n"
        "    from models.backbones.internvideo2 import ...\n"
        "are changed to:\n"
        "    from multi_modality.models.backbones.internvideo2 import ...\n"
        f"Original error: {repr(e)}"
    ) from e


device = os.getenv("INTERNVIDEO_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip().lower()
if device not in {"cuda", "cpu"}:
    raise ValueError(f"Unsupported INTERNVIDEO_DEVICE={device!r}; expected 'cuda' or 'cpu'.")

config = Config.from_file(
    os.path.join(INTERNVIDEO_MM_PATH, "demo", "internvideo2_stage2_config.py")
)
config = eval_dict_leaf(config)
config.device = device

config.model.vision_encoder.pretrained = os.path.join(
    INTERNVIDEO_MM_PATH, config.model.vision_encoder.pretrained
)
config.model.text_encoder.config = os.path.join(
    INTERNVIDEO_MM_PATH, config.model.text_encoder.config
)
config.pretrained_path = os.path.join(
    INTERNVIDEO_MM_PATH, config.pretrained_path
)

intern_model, _ = setup_internvideo2(config)
intern_model = intern_model.to(device)
intern_model.eval()

v_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def normalize(data: np.ndarray) -> np.ndarray:
    return (data.astype(np.float32) / 255.0 - v_mean) / v_std


def image2tensor(
    image: np.ndarray,
    target_size=(224, 224),
    device_obj: torch.device | str = "cuda",
) -> torch.Tensor:
    image = cv2.resize(image[:, :, ::-1], target_size)
    image_tensor = np.expand_dims(normalize(image), axis=(0, 1))  # [1,1,H,W,C]
    image_tensor = np.transpose(image_tensor, (0, 1, 4, 2, 3))   # [1,1,C,H,W]
    image_tensor = torch.from_numpy(image_tensor).to(
        device_obj, non_blocking=True
    ).float()
    return image_tensor


def _safe_makedirs_for_file(file_path: str):
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)


def extract_image_feats(
    input_path,
    output_path,
    num_splits=4,
    split_index=None,
    disable_prog=False,
):
    """
    Extract image features and save them as a pickle file.

    Args:
        input_path (str): Path to image metadata json.
        output_path (str): Path to save pickle output.
        num_splits (int): Number of splits to divide the total files into.
        split_index (int | None): Index of the split to process (0-based).
        disable_prog (bool): Whether to disable tqdm progress bars.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input json not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        image_metadata = json.load(f)

    if not isinstance(image_metadata, dict):
        raise ValueError(
            f"Expected metadata json to be a dict[path -> meta], got {type(image_metadata)}"
        )

    all_images = sorted(list(image_metadata.keys()))
    total_images = len(all_images)

    if num_splits <= 0:
        raise ValueError("num_splits must be a positive integer.")

    if split_index is not None and not (0 <= split_index < num_splits):
        raise ValueError("split_index must be between 0 and num_splits - 1.")

    split_size = (total_images + num_splits - 1) // num_splits

    if split_index is None:
        split_images = all_images
    else:
        split_start = split_index * split_size
        split_end = min(split_start + split_size, total_images)
        split_images = all_images[split_start:split_end]

    print(f"Input path: {input_path} | Output path: {output_path}")
    print(
        f"Processing {'all images' if split_index is None else f'split {split_index + 1}/{num_splits}'} "
        f"on device {device}..."
    )
    print(f"Total images in metadata: {total_images}")
    print(f"Images in current run: {len(split_images)}")

    # 仅在 split 0 或单次全量运行时抽 caption 特征，避免重复计算
    if split_index in (0, None):
        image_caps_feats = {}
        print("Extracting image caption text features...")

        with torch.no_grad():
            for image_path in tqdm(
                all_images,
                desc="Image-caption text feats",
                disable=disable_prog,
            ):
                image_meta = image_metadata.get(image_path, {})
                caption = image_meta.get("caption", "")

                if not isinstance(caption, str):
                    caption = str(caption)

                txt_feat = intern_model.get_txt_feat(caption).cpu().numpy().squeeze(0)
                image_caps_feats[image_path] = txt_feat

        base, ext = os.path.splitext(output_path)
        cap_output_path = f"{base}_imgcap{ext}"
        _safe_makedirs_for_file(cap_output_path)
        with open(cap_output_path, "wb") as f:
            pickle.dump(image_caps_feats, f)
        print(f"Saved image-caption text features to: {cap_output_path}")

    image_feats = {}
    size_t = config.get("size_t", 224)

    with torch.no_grad():
        for image_path in tqdm(
            split_images,
            desc=(
                "Processing all images"
                if split_index is None
                else f"Processing split {split_index + 1}/{num_splits}"
            ),
            disable=disable_prog,
        ):
            image = cv2.imread(image_path)
            if image is None:
                print(f"[ERROR] Unable to read image: {image_path}")
                continue

            image_tensor = image2tensor(
                image,
                target_size=(size_t, size_t),
                device_obj=device,
            )
            image_feature = intern_model.get_vid_feat(image_tensor).cpu().numpy().squeeze(0)
            image_feats[image_path] = image_feature

    if split_index is not None:
        base, ext = os.path.splitext(output_path)
        split_output_path = f"{base}_split{split_index + 1}{ext}"
    else:
        split_output_path = output_path

    _safe_makedirs_for_file(split_output_path)
    with open(split_output_path, "wb") as f:
        pickle.dump(image_feats, f)

    print(f"Saved image features to: {split_output_path}")
    print(f"Valid extracted image count: {len(image_feats)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract image features and save them as a pickle file."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the image metadata json file.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the pickle file.",
    )
    parser.add_argument(
        "--num_splits",
        type=int,
        default=1,
        help="Number of splits to divide the total files into.",
    )
    parser.add_argument(
        "--split_index",
        type=int,
        default=None,
        help="Index of the split to process (0-based).",
    )
    parser.add_argument(
        "--disable_prog",
        action="store_true",
        help="Disable progress bars.",
    )
    args = parser.parse_args()

    extract_image_feats(
        input_path=args.input_path,
        output_path=args.output_path,
        num_splits=args.num_splits,
        split_index=args.split_index,
        disable_prog=args.disable_prog,
    )
