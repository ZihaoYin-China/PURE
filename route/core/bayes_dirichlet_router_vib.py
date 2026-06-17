import numpy as np


MODALITIES = ["no", "paragraph", "document", "image"]


def _safe_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _normalize_probs(probs):
    if probs is None:
        return None
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape != (len(MODALITIES),):
        return None
    probs = np.clip(probs, 1e-8, None)
    total = float(probs.sum())
    if total <= 0:
        return None
    return probs / total


def _normalize_alpha(alpha):
    if alpha is None:
        return None
    alpha = np.asarray(alpha, dtype=np.float64)
    if alpha.shape != (len(MODALITIES),):
        return None
    return np.clip(alpha, 1.0, None)


class BayesDirichletRouterVIB:
    """
    VIB-aware Bayesian routing policy.

    Compared with the original Bayes router, this class can:
    1) consume VIB evidential outputs directly (Dirichlet mean / alpha / uncertainty)
    2) blend VIB probabilities with sidecar base-router probabilities
    3) adapt tau per query using VIB uncertainty / evidence strength
    """

    def __init__(
        self,
        alpha_prior=None,
        alpha_prior_by_target=None,
        tau=8.0,
        beta_cost=0.1,
        modality_costs=None,
        default_confidence=0.72,
        uncertainty_threshold=0.35,
        fallback_when_uncertain=True,
        decision_mode="mean",
        seed=42,
        vib_uncertainty_low=0.28,
        vib_uncertainty_high=0.45,
        vib_weight_low=0.15,
        vib_weight_high=0.85,
        dynamic_tau_min=0.35,
        dynamic_tau_max=1.15,
        evidence_saturation=8.0,
        vib_target_params=None,
        protect_base_modalities=None,
        allow_vib_modalities=None,
    ):
        if alpha_prior is None:
            alpha_prior = [1.0, 1.0, 1.0, 1.0]
        if len(alpha_prior) != len(MODALITIES):
            raise ValueError(
                f"alpha_prior must have {len(MODALITIES)} elements, got {len(alpha_prior)}"
            )

        self.alpha_prior = np.asarray(alpha_prior, dtype=np.float64)
        self.alpha_prior = np.clip(self.alpha_prior, 1e-6, None)
        self.alpha_init = self.alpha_prior.copy()
        self.alpha_prior_by_target = self._normalize_target_priors(alpha_prior_by_target)

        self.tau = float(tau)
        if self.tau < 0:
            raise ValueError("tau must be >= 0")

        self.beta_cost = float(beta_cost)
        self.default_confidence = float(default_confidence)
        self.default_confidence = float(np.clip(self.default_confidence, 1.0 / len(MODALITIES), 1.0))

        if modality_costs is None:
            modality_costs = [0.0, 0.25, 0.45, 0.60]
        if len(modality_costs) != len(MODALITIES):
            raise ValueError(
                f"modality_costs must have {len(MODALITIES)} elements, got {len(modality_costs)}"
            )
        self.modality_costs = np.asarray(modality_costs, dtype=np.float64)

        self.uncertainty_threshold = float(uncertainty_threshold)
        self.fallback_when_uncertain = bool(fallback_when_uncertain)

        decision_mode = str(decision_mode).strip().lower()
        if decision_mode not in {"mean", "thompson"}:
            raise ValueError("decision_mode must be 'mean' or 'thompson'")
        self.decision_mode = decision_mode

        self.vib_uncertainty_low = float(vib_uncertainty_low)
        self.vib_uncertainty_high = float(vib_uncertainty_high)
        self.vib_weight_low = float(np.clip(vib_weight_low, 0.0, 1.0))
        self.vib_weight_high = float(np.clip(vib_weight_high, 0.0, 1.0))
        self.dynamic_tau_min = float(max(0.0, dynamic_tau_min))
        self.dynamic_tau_max = float(max(self.dynamic_tau_min, dynamic_tau_max))
        self.evidence_saturation = float(max(1e-6, evidence_saturation))
        self.vib_target_params = self._normalize_vib_target_params(vib_target_params)
        self.protect_base_modalities = self._normalize_modalities(protect_base_modalities)
        self.allow_vib_modalities = self._normalize_modalities(allow_vib_modalities)

        self.rng = np.random.default_rng(seed=seed)

    def _normalize_modalities(self, modalities):
        if not modalities:
            return set()
        out = set()
        for item in modalities:
            key = str(item or "").strip().lower()
            if key in MODALITIES:
                out.add(key)
        return out

    def _normalize_vib_target_params(self, vib_target_params):
        if not vib_target_params:
            return {}
        out = {}
        for key, value in vib_target_params.items():
            if not isinstance(value, dict):
                continue
            target = str(key).lower()
            out[target] = {
                "vib_uncertainty_low": float(value.get("vib_uncertainty_low", self.vib_uncertainty_low)),
                "vib_uncertainty_high": float(value.get("vib_uncertainty_high", self.vib_uncertainty_high)),
                "vib_weight_low": float(np.clip(value.get("vib_weight_low", self.vib_weight_low), 0.0, 1.0)),
                "vib_weight_high": float(np.clip(value.get("vib_weight_high", self.vib_weight_high), 0.0, 1.0)),
            }
        return out

    def _resolve_vib_params(self, target=None):
        params = {
            "vib_uncertainty_low": self.vib_uncertainty_low,
            "vib_uncertainty_high": self.vib_uncertainty_high,
            "vib_weight_low": self.vib_weight_low,
            "vib_weight_high": self.vib_weight_high,
        }
        if target is not None:
            params.update(self.vib_target_params.get(str(target).lower(), {}))
        return params

    def _normalize_target_priors(self, alpha_prior_by_target):
        if not alpha_prior_by_target:
            return {}
        out = {}
        for key, value in alpha_prior_by_target.items():
            if value is None:
                continue
            if len(value) != len(MODALITIES):
                raise ValueError(
                    f"alpha_prior_by_target[{key}] must have {len(MODALITIES)} elements, got {len(value)}"
                )
            arr = np.asarray(value, dtype=np.float64)
            arr = np.clip(arr, 1e-6, None)
            out[str(key).lower()] = arr
        return out

    def _resolve_prior(self, target=None):
        if target is not None:
            key = str(target).lower()
            if key in self.alpha_prior_by_target:
                return self.alpha_prior_by_target[key].copy()
        return self.alpha_prior.copy()

    def _hard_label_to_probs(self, label, confidence=None):
        label = str(label or "").strip().lower()
        k = len(MODALITIES)

        if label not in MODALITIES:
            return np.full(k, 1.0 / k, dtype=np.float64)

        conf = _safe_float(confidence, default=self.default_confidence)
        if conf is None:
            conf = self.default_confidence
        conf = float(np.clip(conf, 1.0 / k, 1.0 - 1e-8))

        probs = np.full(k, (1.0 - conf) / (k - 1), dtype=np.float64)
        probs[MODALITIES.index(label)] = conf
        return probs / probs.sum()

    def _compute_vib_weight(self, vib_uncertainty, has_vib_probs, has_base_probs, target=None):
        if has_vib_probs and not has_base_probs:
            return 1.0
        if not has_vib_probs:
            return 0.0
        if not has_base_probs:
            return 1.0

        params = self._resolve_vib_params(target=target)
        vib_uncertainty_low = params["vib_uncertainty_low"]
        vib_uncertainty_high = params["vib_uncertainty_high"]
        vib_weight_low = params["vib_weight_low"]
        vib_weight_high = params["vib_weight_high"]

        vib_uncertainty = _safe_float(vib_uncertainty, None)
        if vib_uncertainty is None:
            return vib_weight_high

        low = min(vib_uncertainty_low, vib_uncertainty_high)
        high = max(vib_uncertainty_low, vib_uncertainty_high)
        if high <= low + 1e-8:
            return vib_weight_high
        if vib_uncertainty <= low:
            return vib_weight_high
        if vib_uncertainty >= high:
            return vib_weight_low

        ratio = (high - vib_uncertainty) / (high - low)
        return vib_weight_low + ratio * (vib_weight_high - vib_weight_low)

    def _compute_tau_scale(self, vib_uncertainty=None, vib_alpha=None, vib_weight=1.0, target=None):
        scales = []
        params = self._resolve_vib_params(target=target)
        vib_uncertainty_low = params["vib_uncertainty_low"]
        vib_uncertainty_high = params["vib_uncertainty_high"]

        vib_alpha = _normalize_alpha(vib_alpha)
        if vib_alpha is not None:
            evidence = np.clip(vib_alpha - 1.0, 0.0, None)
            evidence_mass = float(evidence.sum())
            strength_ratio = evidence_mass / (evidence_mass + self.evidence_saturation)
            scales.append(
                self.dynamic_tau_min
                + strength_ratio * (self.dynamic_tau_max - self.dynamic_tau_min)
            )

        vib_uncertainty = _safe_float(vib_uncertainty, None)
        if vib_uncertainty is not None:
            low = min(vib_uncertainty_low, vib_uncertainty_high)
            high = max(vib_uncertainty_low, vib_uncertainty_high)
            if high <= low + 1e-8:
                conf_ratio = 1.0
            else:
                conf_ratio = 1.0 - np.clip((vib_uncertainty - low) / (high - low), 0.0, 1.0)
            scales.append(
                self.dynamic_tau_min
                + conf_ratio * (self.dynamic_tau_max - self.dynamic_tau_min)
            )

        if not scales:
            return 1.0

        base_scale = float(np.mean(scales))
        vib_weight = float(np.clip(vib_weight, 0.0, 1.0))
        blended_scale = self.dynamic_tau_min + vib_weight * (base_scale - self.dynamic_tau_min)
        return float(np.clip(blended_scale, self.dynamic_tau_min, self.dynamic_tau_max))

    def posterior(
        self,
        retrieval,
        retrieval_conf=None,
        target=None,
        vib_probs=None,
        vib_alpha=None,
        vib_uncertainty=None,
        base_probs=None,
        vib_weight_override=None,
    ):
        alpha_prior = self._resolve_prior(target=target)

        vib_alpha = _normalize_alpha(vib_alpha)
        vib_probs = _normalize_probs(vib_probs)
        if vib_probs is None and vib_alpha is not None:
            vib_probs = vib_alpha / vib_alpha.sum()

        base_probs = _normalize_probs(base_probs)
        if vib_probs is None and base_probs is None:
            base_probs = self._hard_label_to_probs(retrieval, retrieval_conf)

        vib_weight = self._compute_vib_weight(
            vib_uncertainty=vib_uncertainty,
            has_vib_probs=vib_probs is not None,
            has_base_probs=base_probs is not None,
            target=target,
        )

        base_top = None
        vib_top = None
        if base_probs is not None:
            base_top = MODALITIES[int(np.argmax(base_probs))]
        if vib_probs is not None:
            vib_top = MODALITIES[int(np.argmax(vib_probs))]

        protected_by_base = False
        if (
            vib_probs is not None
            and base_probs is not None
            and self.protect_base_modalities
            and base_top in self.protect_base_modalities
        ):
            allow_vib = False
            if self.allow_vib_modalities:
                allow_vib = (
                    (base_top in self.allow_vib_modalities)
                    or (vib_top in self.allow_vib_modalities)
                )
            if not allow_vib:
                vib_weight = 0.0
                protected_by_base = True

        vib_weight_overridden = False
        vib_weight_override = _safe_float(vib_weight_override, None)
        if vib_weight_override is not None:
            vib_weight = float(np.clip(vib_weight_override, 0.0, 1.0))
            vib_weight_overridden = True

        if vib_probs is not None and base_probs is not None:
            router_probs = vib_weight * vib_probs + (1.0 - vib_weight) * base_probs
            probs_source = "base_protected" if protected_by_base else "hybrid_vib_base"
        elif vib_probs is not None:
            router_probs = vib_probs
            probs_source = "vib_only"
        else:
            router_probs = base_probs
            probs_source = "base_only"

        router_probs = _normalize_probs(router_probs)
        tau_scale = self._compute_tau_scale(
            vib_uncertainty=vib_uncertainty,
            vib_alpha=vib_alpha,
            vib_weight=vib_weight,
            target=target,
        )
        tau_effective = float(self.tau * tau_scale)

        alpha_post = alpha_prior + tau_effective * router_probs
        alpha_post = np.clip(alpha_post, 1e-8, None)

        return {
            "alpha_prior": alpha_prior,
            "posterior_alpha": alpha_post,
            "router_probs": router_probs,
            "vib_probs": vib_probs.tolist() if vib_probs is not None else None,
            "base_probs": base_probs.tolist() if base_probs is not None else None,
            "vib_weight": float(vib_weight),
            "tau_scale": float(tau_scale),
            "tau_effective": float(tau_effective),
            "probs_source": probs_source,
            "base_top_modality": base_top,
            "vib_top_modality": vib_top,
            "protected_by_base": bool(protected_by_base),
            "vib_weight_overridden": bool(vib_weight_overridden),
        }

    def decide(
        self,
        retrieval,
        retrieval_conf=None,
        target=None,
        vib_probs=None,
        vib_alpha=None,
        vib_uncertainty=None,
        base_probs=None,
        vib_weight_override=None,
    ):
        posterior = self.posterior(
            retrieval=retrieval,
            retrieval_conf=retrieval_conf,
            target=target,
            vib_probs=vib_probs,
            vib_alpha=vib_alpha,
            vib_uncertainty=vib_uncertainty,
            base_probs=base_probs,
            vib_weight_override=vib_weight_override,
        )

        alpha_post = posterior["posterior_alpha"]
        if self.decision_mode == "thompson":
            theta = self.rng.dirichlet(alpha_post)
        else:
            theta = alpha_post / alpha_post.sum()

        utility = theta - self.beta_cost * self.modality_costs
        selected_idx = int(np.argmax(utility))
        selected = MODALITIES[selected_idx]

        uncertainty = len(MODALITIES) / float(alpha_post.sum())
        target_key = str(target or "").lower()
        if (
            self.fallback_when_uncertain
            and uncertainty > self.uncertainty_threshold
            and target_key not in {"mmlu"}
        ):
            if selected == "no":
                if target_key == "webqa":
                    selected = "paragraph"
                else:
                    selected = "document"

        result = {
            "selected": selected,
            "selected_idx": MODALITIES.index(selected),
            "uncertainty": float(uncertainty),
            "theta": theta.tolist(),
            "utility": utility.tolist(),
        }
        result.update({
            "posterior_alpha": posterior["posterior_alpha"].tolist(),
            "alpha_prior": posterior["alpha_prior"].tolist(),
            "router_probs": posterior["router_probs"].tolist(),
            "vib_probs": posterior["vib_probs"],
            "base_probs": posterior["base_probs"],
            "vib_weight": posterior["vib_weight"],
            "tau_scale": posterior["tau_scale"],
            "tau_effective": posterior["tau_effective"],
            "probs_source": posterior["probs_source"],
            "base_top_modality": posterior["base_top_modality"],
            "vib_top_modality": posterior["vib_top_modality"],
            "protected_by_base": posterior["protected_by_base"],
            "vib_weight_overridden": posterior["vib_weight_overridden"],
        })
        return result

    def update_prior(self, selected_modality, reward, eta=1.0, rho=0.0):
        selected_modality = str(selected_modality or "").strip().lower()
        if selected_modality not in MODALITIES:
            return

        reward = float(np.clip(_safe_float(reward, 0.0), 0.0, 1.0))
        eta = float(max(0.0, eta))
        rho = float(np.clip(rho, 0.0, 1.0))

        if rho > 0.0:
            self.alpha_prior = (1.0 - rho) * self.alpha_prior + rho * self.alpha_init

        idx = MODALITIES.index(selected_modality)
        self.alpha_prior[idx] += eta * reward
        self.alpha_prior = np.clip(self.alpha_prior, 1e-6, None)

    def update_prior_with_penalty(
        self,
        selected_modality,
        reward,
        eta=1.0,
        penalty=0.5,
        rho=0.0,
        spread=0.25,
    ):
        selected_modality = str(selected_modality or "").strip().lower()
        if selected_modality not in MODALITIES:
            return

        reward = float(np.clip(_safe_float(reward, 0.0), 0.0, 1.0))
        eta = float(max(0.0, eta))
        penalty = float(max(0.0, penalty))
        rho = float(np.clip(rho, 0.0, 1.0))
        spread = float(np.clip(spread, 0.0, 1.0))

        if rho > 0.0:
            self.alpha_prior = (1.0 - rho) * self.alpha_prior + rho * self.alpha_init

        idx = MODALITIES.index(selected_modality)
        boost = eta * reward
        decay = penalty * (1.0 - reward)
        self.alpha_prior[idx] += boost
        self.alpha_prior[idx] = max(1e-6, self.alpha_prior[idx] - decay)

        if spread > 0.0:
            spill = (boost * spread) / max(1, len(MODALITIES) - 1)
            for j in range(len(MODALITIES)):
                if j != idx:
                    self.alpha_prior[j] += spill

        self.alpha_prior = np.clip(self.alpha_prior, 1e-6, None)
