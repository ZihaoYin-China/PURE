import numpy as np


MODALITIES = ["no", "paragraph", "document", "image"]


def _safe_float(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


class BayesDirichletRouter:
    """
    Bayesian routing policy over discrete modalities with a Dirichlet posterior.

    Core idea:
      posterior_alpha(q) = prior_alpha + tau * p_router(q)
    where p_router is a probability vector from the base router (or an inferred one).
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
            # Relative compute/retrieval burden (example defaults)
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

        self.rng = np.random.default_rng(seed=seed)

    def _normalize_target_priors(self, alpha_prior_by_target):
        if not alpha_prior_by_target:
            return {}
        out = {}
        for k, v in alpha_prior_by_target.items():
            if v is None:
                continue
            if len(v) != len(MODALITIES):
                raise ValueError(
                    f"alpha_prior_by_target[{k}] must have {len(MODALITIES)} elements, got {len(v)}"
                )
            arr = np.asarray(v, dtype=np.float64)
            arr = np.clip(arr, 1e-6, None)
            out[str(k).lower()] = arr
        return out

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
        probs = probs / probs.sum()
        return probs

    def posterior(self, retrieval, retrieval_conf=None, retrieval_probs=None):
        """
        Build a per-query posterior from either:
        1) full probabilities (retrieval_probs), or
        2) hard label + confidence.
        """
        if retrieval_probs is not None:
            probs = np.asarray(retrieval_probs, dtype=np.float64)
            if probs.shape != (len(MODALITIES),):
                raise ValueError(
                    f"retrieval_probs shape must be ({len(MODALITIES)},), got {probs.shape}"
                )
            probs = np.clip(probs, 1e-8, None)
            probs = probs / probs.sum()
        else:
            probs = self._hard_label_to_probs(retrieval, retrieval_conf)

        alpha_post = self.alpha_prior + self.tau * probs
        alpha_post = np.clip(alpha_post, 1e-8, None)
        return alpha_post, probs

    def decide(self, retrieval, retrieval_conf=None, retrieval_probs=None, target=None):
        if target is not None:
            target_key = str(target).lower()
            if target_key in self.alpha_prior_by_target:
                self.alpha_prior = self.alpha_prior_by_target[target_key].copy()
        alpha_post, probs = self.posterior(retrieval, retrieval_conf, retrieval_probs)

        if self.decision_mode == "thompson":
            theta = self.rng.dirichlet(alpha_post)
        else:
            theta = alpha_post / alpha_post.sum()

        utility = theta - self.beta_cost * self.modality_costs
        selected_idx = int(np.argmax(utility))
        selected = MODALITIES[selected_idx]

        # A simple uncertainty proxy:
        # larger sum(alpha) => lower posterior variance.
        uncertainty = len(MODALITIES) / float(alpha_post.sum())

        # Deterministic fallback when uncertainty is high.
        if self.fallback_when_uncertain and uncertainty > self.uncertainty_threshold:
            if selected == "no":
                if str(target or "").lower() == "webqa":
                    selected = "paragraph"
                else:
                    selected = "document"

        return {
            "selected": selected,
            "selected_idx": MODALITIES.index(selected),
            "uncertainty": float(uncertainty),
            "theta": theta.tolist(),
            "utility": utility.tolist(),
            "posterior_alpha": alpha_post.tolist(),
            "router_probs": probs.tolist(),
        }

    def update_prior(self, selected_modality, reward, eta=1.0, rho=0.0):
        """
        Lightweight online update:
          - optional forgetting toward initial prior by rho
          - add eta * reward count to selected modality
        """
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
        """
        Online update with optional penalty and spillover:
          - reward boosts selected modality by eta * reward
          - penalty reduces selected modality when reward is low
          - optional spillover distributes a small amount to other modalities
        """
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

    def state_dict(self):
        return {
            "modalities": MODALITIES,
            "alpha_prior": self.alpha_prior.tolist(),
            "alpha_init": self.alpha_init.tolist(),
            "alpha_prior_by_target": {
                k: v.tolist() for k, v in self.alpha_prior_by_target.items()
            },
            "tau": float(self.tau),
            "beta_cost": float(self.beta_cost),
            "modality_costs": self.modality_costs.tolist(),
            "default_confidence": float(self.default_confidence),
            "uncertainty_threshold": float(self.uncertainty_threshold),
            "fallback_when_uncertain": bool(self.fallback_when_uncertain),
            "decision_mode": self.decision_mode,
        }
