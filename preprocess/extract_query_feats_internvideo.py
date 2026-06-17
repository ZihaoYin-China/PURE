import os
import sys
import pickle
import json
import warnings
import types
from tqdm import tqdm
import torch
from torch import nn

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
            "This means InternVideo tried to use flash-attn kernels instead of the expected non-flash path."
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
    env_root = os.getenv("INTERNVIDEO_PATH")
    if not env_root:
        raise EnvironmentError("Environment variable `INTERNVIDEO_PATH` is not set.")

    env_root = os.path.abspath(env_root)
    candidates = [
        (os.path.join(env_root, "InternVideo2"), os.path.join(env_root, "InternVideo2", "multi_modality")),
        (env_root, os.path.join(env_root, "multi_modality")),
    ]
    for internvideo2_root, multi_modality_root in candidates:
        if os.path.isdir(internvideo2_root) and os.path.isdir(multi_modality_root):
            return internvideo2_root, multi_modality_root

    raise FileNotFoundError(
        "Could not locate InternVideo2 paths from INTERNVIDEO_PATH. "
        f"Got INTERNVIDEO_PATH={env_root}. Expected either "
        f"{os.path.join(env_root, 'InternVideo2', 'multi_modality')} "
        "or <INTERNVIDEO_PATH>/multi_modality to exist."
    )


internvideo2_root, internvideo2_path = _resolve_internvideo_paths()
if internvideo2_root not in sys.path:
    sys.path.insert(0, internvideo2_root)
if internvideo2_path not in sys.path:
    sys.path.insert(0, internvideo2_path)

from utils.config import Config, eval_dict_leaf
from demo.utils import setup_internvideo2

device = os.getenv("INTERNVIDEO_DEVICE", "cuda").strip().lower()
if device not in {"cuda", "cpu"}:
    raise ValueError(f"Unsupported INTERNVIDEO_DEVICE={device!r}; expected 'cuda' or 'cpu'.")

config = Config.from_file(os.path.join(internvideo2_path, 'demo/internvideo2_stage2_config.py'))
config = eval_dict_leaf(config)
config.device = device
config.model.vision_encoder.pretrained = os.path.join(internvideo2_path, config.model.vision_encoder.pretrained)
config.model.text_encoder.config = os.path.join(internvideo2_path, config.model.text_encoder.config)
config.pretrained_path = os.path.join(internvideo2_path, config.pretrained_path)
intern_model, _ = setup_internvideo2(config)
intern_model.to(device)
intern_model.eval()

def extract_query_feats_internvideo(input, output_path):
    """
    Extract query features from the input JSON file and save them as a pickle file.
    Args:
        input (str): Path to the input JSON file.
        output_path (str): Path to save the pickle file.
    """
    with open(input, 'r') as f:
        data = json.load(f)

    id2feat = {}
    with torch.no_grad():
        for row in data:
            query_id = row['index']
            text_data = row['question']
            id2feat[query_id] = intern_model.get_txt_feat(text_data).squeeze(0).detach().cpu()

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, os.path.splitext(os.path.basename(input))[0] + '.pkl'), 'wb') as f:
        pickle.dump(id2feat, f)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description="Extract query features from InternVideo and save them as a pickle file.")
    parser.add_argument("--input_path", type=str, default="dataset/query", help="Path to the input directory containing JSON files.")
    parser.add_argument("--output_path", type=str, default="eval/features/query/internvideo", help="Path to save the output pickle files.")
    args = parser.parse_args()

    inputs = [os.path.join(args.input_path, input) for input in os.listdir(args.input_path)]

    for input in tqdm(inputs):
        extract_query_feats_internvideo(input, args.output_path)
