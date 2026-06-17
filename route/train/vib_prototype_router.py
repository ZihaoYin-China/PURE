import json
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, T5EncoderModel


DEFAULT_LABELS = ["no", "paragraph", "document", "image"]


def _infer_hidden_size(config) -> int:
    for attr in ("hidden_size", "d_model"):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Cannot infer hidden size from config type={type(config).__name__}")


def _load_backbone(backbone_name: str):
    config = AutoConfig.from_pretrained(backbone_name)
    model_type = str(getattr(config, "model_type", "")).strip().lower()

    if model_type == "t5":
        backbone = T5EncoderModel.from_pretrained(backbone_name)
    else:
        backbone = AutoModel.from_pretrained(backbone_name)

    hidden_size = _infer_hidden_size(backbone.config)
    return backbone, hidden_size, model_type


def _load_tokenizer(model_name_or_path: str):
    model_type = ""
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        model_type = str(getattr(config, "model_type", "")).strip().lower()
    except Exception:
        model_type = ""

    if not model_type and os.path.isdir(model_name_or_path):
        router_config_path = os.path.join(model_name_or_path, "router_config.json")
        tokenizer_config_path = os.path.join(model_name_or_path, "tokenizer_config.json")

        if os.path.isfile(router_config_path):
            try:
                with open(router_config_path, "r", encoding="utf-8") as f:
                    router_config = json.load(f)
                model_type = str(router_config.get("backbone_model_type", "")).strip().lower()
                if not model_type and router_config.get("backbone_name"):
                    cfg = AutoConfig.from_pretrained(router_config["backbone_name"])
                    model_type = str(getattr(cfg, "model_type", "")).strip().lower()
            except Exception:
                model_type = model_type or ""

        if not model_type and os.path.isfile(tokenizer_config_path):
            try:
                with open(tokenizer_config_path, "r", encoding="utf-8") as f:
                    tokenizer_config = json.load(f)
                tokenizer_class = str(tokenizer_config.get("tokenizer_class", "")).strip().lower()
                if "t5" in tokenizer_class:
                    model_type = "t5"
            except Exception:
                model_type = model_type or ""

        if not model_type and os.path.isfile(os.path.join(model_name_or_path, "spiece.model")):
            model_type = "t5"

    use_fast = model_type != "t5"
    try:
        return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=use_fast)
    except TypeError:
        if use_fast:
            return AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        raise


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    device = str(device or "").strip().lower()
    if device in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def label_to_id(label: str, label_names: List[str]) -> int:
    label = str(label or "").strip().lower()
    if label not in label_names:
        raise KeyError(f"Unknown label {label!r}; expected one of {label_names}")
    return label_names.index(label)


def dirichlet_mse_loss(alpha: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    num_classes = alpha.size(-1)
    targets = F.one_hot(labels, num_classes=num_classes).float()
    strength = alpha.sum(dim=-1, keepdim=True)
    mean = alpha / strength
    variance = alpha * (strength - alpha) / (strength * strength * (strength + 1.0))
    return ((targets - mean) ** 2 + variance).sum(dim=-1).mean()


class VIBPrototypeRouter(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_labels: int = 4,
        latent_dim: int = 128,
        hidden_dropout_prob: float = 0.1,
        prototype_temperature: float = 1.0,
        prototype_margin: float = 0.2,
        kl_weight: float = 1e-3,
        proto_weight: float = 0.1,
        evi_weight: float = 0.2,
        proto_logit_scale: float = 0.5,
        class_weights: Optional[List[float]] = None,
        label_smoothing: float = 0.0,
        label_names: Optional[List[str]] = None,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_labels = int(num_labels)
        self.latent_dim = int(latent_dim)
        self.hidden_dropout_prob = float(hidden_dropout_prob)
        self.prototype_temperature = float(prototype_temperature)
        self.prototype_margin = float(prototype_margin)
        self.kl_weight = float(kl_weight)
        self.proto_weight = float(proto_weight)
        self.evi_weight = float(evi_weight)
        self.proto_logit_scale = float(proto_logit_scale)
        self.label_smoothing = float(max(0.0, label_smoothing))
        self.label_names = list(label_names or DEFAULT_LABELS)

        self.backbone, hidden_size, self.backbone_model_type = _load_backbone(backbone_name)

        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.pre_classifier = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.mu_head = nn.Linear(hidden_size, self.latent_dim)
        self.logvar_head = nn.Linear(hidden_size, self.latent_dim)
        self.latent_norm = nn.LayerNorm(self.latent_dim)
        self.evidence_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.Tanh(),
            nn.Linear(self.latent_dim, self.num_labels),
        )
        self.prototypes = nn.Parameter(torch.randn(self.num_labels, self.latent_dim) * 0.02)
        if class_weights is not None:
            if len(class_weights) != self.num_labels:
                raise ValueError(
                    f"class_weights must have {self.num_labels} elements, got {len(class_weights)}"
                )
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
            class_weights_tensor = class_weights_tensor.clamp(min=1e-6)
            self.register_buffer("class_weights", class_weights_tensor)
        else:
            self.class_weights = None

    def mean_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        masked = hidden_states * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        return masked.sum(dim=1) / denom

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        classifier_hidden = F.relu(self.pre_classifier(pooled))
        classifier_hidden = self.dropout(classifier_hidden)
        classifier_logits = self.classifier(classifier_hidden)
        mu = self.mu_head(pooled)
        logvar = self.logvar_head(pooled).clamp(min=-8.0, max=8.0)
        return {
            "pooled": pooled,
            "classifier_hidden": classifier_hidden,
            "classifier_logits": classifier_logits,
            "mu": mu,
            "logvar": logvar,
        }

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def build_alpha(self, z: torch.Tensor, classifier_logits: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.latent_norm(z)
        z_unit = F.normalize(z, dim=-1)
        prototype_unit = F.normalize(self.prototypes, dim=-1)

        # Cosine geometry keeps prototype scores in a stable numeric range.
        proto_sim = torch.matmul(z_unit, prototype_unit.transpose(0, 1))
        proto_dist = 1.0 - proto_sim
        proto_logits = (
            self.proto_logit_scale
            * proto_sim
            / max(self.prototype_temperature, 1e-6)
        )

        logits = classifier_logits + proto_logits
        evidence_logits = self.evidence_head(z) + logits
        evidence = F.softplus(evidence_logits) + 1e-6
        alpha = evidence + 1.0
        probs = F.softmax(logits, dim=-1)
        dirichlet_mean = alpha / alpha.sum(dim=-1, keepdim=True)
        return {
            "z": z,
            "proto_dist": proto_dist,
            "proto_sim": proto_sim,
            "proto_logits": proto_logits,
            "evidence_logits": evidence_logits,
            "evidence": evidence,
            "alpha": alpha,
            "dirichlet_mean": dirichlet_mean,
            "probs": probs,
            "logits": logits,
        }

    def compute_loss(
        self,
        logits: torch.Tensor,
        alpha: torch.Tensor,
        proto_dist: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        ce_kwargs = {}
        if self.class_weights is not None:
            ce_kwargs["weight"] = self.class_weights
        if self.label_smoothing > 0:
            ce_kwargs["label_smoothing"] = self.label_smoothing
        ce_loss = F.cross_entropy(logits, labels, **ce_kwargs)
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        positive = proto_dist[torch.arange(labels.size(0), device=labels.device), labels]
        negative_mask = F.one_hot(labels, num_classes=self.num_labels).bool()
        negative = proto_dist.masked_fill(negative_mask, float("inf")).min(dim=-1).values
        proto_loss = F.relu(self.prototype_margin + positive - negative).mean()

        evi_loss = dirichlet_mse_loss(alpha, labels)
        total = ce_loss + self.kl_weight * kl_loss + self.proto_weight * proto_loss + self.evi_weight * evi_loss
        return {
            "loss": total,
            "ce_loss": ce_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "proto_loss": proto_loss.detach(),
            "evi_loss": evi_loss.detach(),
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode(input_ids=input_ids, attention_mask=attention_mask)
        z = self.reparameterize(encoded["mu"], encoded["logvar"])
        outputs = self.build_alpha(z=z, classifier_logits=encoded["classifier_logits"])
        outputs.update(encoded)
        if labels is not None:
            outputs.update(self.compute_loss(
                logits=outputs["logits"],
                alpha=outputs["alpha"],
                proto_dist=outputs["proto_dist"],
                mu=encoded["mu"],
                logvar=encoded["logvar"],
                labels=labels,
            ))
        return outputs

    def predict_batch(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        uncertainty = self.num_labels / outputs["alpha"].sum(dim=-1)
        pred_ids = outputs["probs"].argmax(dim=-1)
        conf = outputs["probs"].max(dim=-1).values
        return {
            "pred_ids": pred_ids,
            "probs": outputs["probs"],
            "conf": conf,
            "uncertainty": uncertainty,
            "logits": outputs["logits"],
            "alpha": outputs["alpha"],
            "dirichlet_mean": outputs["dirichlet_mean"],
            "proto_dist": outputs["proto_dist"],
            "proto_logits": outputs["proto_logits"],
            "classifier_logits": outputs["classifier_logits"],
            "evidence_logits": outputs["evidence_logits"],
            "mu": outputs["mu"],
        }

    def export_config(self) -> Dict[str, object]:
        class_weights = None
        if self.class_weights is not None:
            class_weights = [float(x) for x in self.class_weights.detach().cpu().tolist()]
        return {
            "backbone_name": self.backbone_name,
            "num_labels": self.num_labels,
            "latent_dim": self.latent_dim,
            "hidden_dropout_prob": self.hidden_dropout_prob,
            "prototype_temperature": self.prototype_temperature,
            "prototype_margin": self.prototype_margin,
            "kl_weight": self.kl_weight,
            "proto_weight": self.proto_weight,
            "evi_weight": self.evi_weight,
            "proto_logit_scale": self.proto_logit_scale,
            "class_weights": class_weights,
            "label_smoothing": self.label_smoothing,
            "label_names": self.label_names,
            "backbone_model_type": self.backbone_model_type,
        }

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "VIBPrototypeRouter":
        return cls(
            backbone_name=config["backbone_name"],
            num_labels=int(config["num_labels"]),
            latent_dim=int(config["latent_dim"]),
            hidden_dropout_prob=float(config["hidden_dropout_prob"]),
            prototype_temperature=float(config["prototype_temperature"]),
            prototype_margin=float(config["prototype_margin"]),
            kl_weight=float(config["kl_weight"]),
            proto_weight=float(config["proto_weight"]),
            evi_weight=float(config["evi_weight"]),
            proto_logit_scale=float(config.get("proto_logit_scale", 0.5)),
            class_weights=config.get("class_weights"),
            label_smoothing=float(config.get("label_smoothing", 0.0)),
            label_names=list(config.get("label_names", DEFAULT_LABELS)),
        )

    def load_distilbert_classifier_checkpoint(self, checkpoint_dir: str, device: Optional[torch.device] = None) -> Dict[str, List[str]]:
        state_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(f"Missing classifier checkpoint: {state_path}")

        try:
            state = torch.load(state_path, map_location=device or "cpu", weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location=device or "cpu")

        remapped_state = {}
        for key, value in state.items():
            if key.startswith("distilbert."):
                remapped_state[f"backbone.{key[len('distilbert.'):]}"] = value
            elif key in {
                "pre_classifier.weight",
                "pre_classifier.bias",
                "classifier.weight",
                "classifier.bias",
            }:
                remapped_state[key] = value

        incompatible = self.load_state_dict(remapped_state, strict=False)
        return {
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
        }


def save_router_checkpoint(
    model: VIBPrototypeRouter,
    tokenizer,
    checkpoint_dir: str,
    metadata: Optional[Dict[str, object]] = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "router_model.pt"))
    config = model.export_config()
    config["metadata"] = metadata or {}
    with open(os.path.join(checkpoint_dir, "router_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def load_router_checkpoint(checkpoint_dir: str, device: torch.device) -> VIBPrototypeRouter:
    config_path = os.path.join(checkpoint_dir, "router_config.json")
    state_path = os.path.join(checkpoint_dir, "router_model.pt")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    model = VIBPrototypeRouter.from_config(config)
    try:
        state = torch.load(state_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(state_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_router_tokenizer(checkpoint_dir: str):
    return _load_tokenizer(checkpoint_dir)
