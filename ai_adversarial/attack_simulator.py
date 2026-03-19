"""
Adversarial ML Attack Simulator

Models three primary adversarial attack vectors against quantitative
trading ML systems:

1. Evasion attacks — perturb market signals at inference time to fool models
2. Data poisoning — corrupt training data to degrade alpha signal quality
3. Model inversion — reconstruct proprietary model features from API outputs

Each attack type produces a probability distribution over attack success,
enabling integration into the Monte Carlo portfolio risk pipeline.

References
----------
- Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
- Biggio et al., "Poisoning Attacks Against Support Vector Machines" (2012)
- Fredrikson et al., "Model Inversion Attacks That Exploit Confidence Information" (2015)
- Amid et al., "Adversarial Examples in the Financial Domain" (2022)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from core.scenario import AdversarialAIParams


# Attack surface multipliers by model architecture
ARCHITECTURE_EVASION_VULNERABILITY: dict[str, float] = {
    "gradient_boosting": 0.55,
    "neural_network": 0.80,
    "linear": 0.30,
    "random_forest": 0.45,
    "lstm": 0.75,
    "transformer": 0.85,
}

ARCHITECTURE_INVERSION_VULNERABILITY: dict[str, float] = {
    "gradient_boosting": 0.40,
    "neural_network": 0.70,
    "linear": 0.50,
    "random_forest": 0.35,
    "lstm": 0.65,
    "transformer": 0.75,
}

# Defense effectiveness multipliers (reduction in attack success probability)
DEFENSE_EFFECTIVENESS: dict[str, float] = {
    "adversarial_training": 0.35,
    "input_validation": 0.20,
    "ensemble_defense": 0.30,
    "differential_privacy": 0.45,
    "output_perturbation": 0.25,
    "rate_limiting": 0.15,
}


@dataclass
class AttackResult:
    """Results from adversarial attack simulation."""
    evasion_success_prob: float
    poisoning_success_prob: float
    inversion_success_prob: float
    overall_threat_score: float         # Weighted composite
    alpha_degradation_bps: float        # Expected alpha signal degradation (basis points)
    sharpe_impact: float                # Estimated Sharpe ratio change


class AdversarialAttackSimulator:
    """
    Simulates adversarial ML attacks against quantitative trading systems.

    Parameters
    ----------
    params : AdversarialAIParams
        Scenario parameters for the adversarial AI module.
    rng : np.random.Generator
        Seeded random number generator.

    Example
    -------
    >>> from core.scenario import AdversarialAIParams
    >>> import numpy as np
    >>> params = AdversarialAIParams()
    >>> sim = AdversarialAttackSimulator(params, rng=np.random.default_rng(42))
    >>> threat_score, sharpe_impact = sim.simulate()
    >>> 0.0 <= threat_score <= 1.0
    True
    """

    def __init__(self, params: AdversarialAIParams, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng
        self._evasion_base = ARCHITECTURE_EVASION_VULNERABILITY.get(
            params.model_architecture, 0.55
        )
        self._inversion_base = ARCHITECTURE_INVERSION_VULNERABILITY.get(
            params.model_architecture, 0.50
        )

    def _compute_defense_discount(self) -> float:
        """Compute combined defense effectiveness reduction (non-additive)."""
        discount = 0.0
        for defense in self.params.defense_mechanisms:
            effectiveness = DEFENSE_EFFECTIVENESS.get(defense, 0.10)
            discount = discount + effectiveness * (1.0 - discount)  # Non-additive stacking
        return float(np.clip(discount, 0.0, 0.90))

    def simulate_evasion_attack(self, n: int = 5_000) -> float:
        """
        Model FGSM/PGD-style evasion attacks on live inference.

        Finance-specific: market microstructure noise can mask adversarial
        perturbations, but also provides cover for adversary-injected noise.
        """
        base = self._evasion_base
        # Training data exposure amplifies evasion (adversary can tune attack)
        exposure_bonus = 0.15 * self.params.training_data_exposure
        raw_prob = self.rng.beta(
            a=10 * (base + exposure_bonus),
            b=10 * (1 - base - exposure_bonus),
            size=n,
        )
        discount = self._compute_defense_discount()
        return float(np.mean(raw_prob) * (1.0 - discount))

    def simulate_poisoning_attack(self, n: int = 5_000) -> float:
        """
        Model training data poisoning attacks.

        Higher training data exposure → adversary can inject more poison samples.
        Effect is a gradual degradation of model performance rather than
        a discrete failure event.
        """
        exposure = self.params.training_data_exposure
        base_poisoning_rate = exposure * 0.8   # Fraction of poison that survives cleaning
        success_prob = 1.0 - np.exp(-3.0 * base_poisoning_rate)
        noise = self.rng.uniform(-0.03, 0.03, size=n)
        discount = self._compute_defense_discount() * 0.7  # Poisoning harder to defend
        raw = np.clip(success_prob + noise, 0, 1) * (1.0 - discount)
        return float(np.mean(raw))

    def simulate_model_inversion(self, n: int = 5_000) -> float:
        """
        Model inversion attacks: recovering alpha signal structure from API outputs.

        Hedge fund ML APIs (if any are exposed) leak proprietary feature weights
        through prediction confidence scores. Model inversion recovers these.
        """
        base = self._inversion_base
        # Exposure increases inversion success (more queries → better reconstruction)
        exposure_factor = 1.0 + 0.5 * self.params.training_data_exposure
        adjusted_base = min(0.95, base * exposure_factor)
        raw_prob = self.rng.beta(
            a=8 * adjusted_base,
            b=8 * (1 - adjusted_base),
            size=n,
        )
        discount = self._compute_defense_discount()
        return float(np.mean(raw_prob) * (1.0 - discount * 0.5))

    def simulate(self) -> tuple[float, float]:
        """
        Run all attack simulations and return:
        - overall threat score in [0, 1]
        - estimated Sharpe ratio impact (negative = degradation)
        """
        attack_types = self.params.attack_types
        evasion_prob = self.simulate_evasion_attack() if "evasion" in attack_types else 0.0
        poisoning_prob = self.simulate_poisoning_attack() if "poisoning" in attack_types else 0.0
        inversion_prob = self.simulate_model_inversion() if "model_inversion" in attack_types else 0.0

        # Weighted composite: evasion and poisoning have higher financial impact
        overall = 0.40 * evasion_prob + 0.40 * poisoning_prob + 0.20 * inversion_prob

        # Sharpe impact: poisoning degrades model quality → lower alpha → lower Sharpe
        alpha_value = self.params.alpha_signal_value_usd
        alpha_degradation_bps = poisoning_prob * 30 + evasion_prob * 15  # basis points
        sharpe_impact = -float(poisoning_prob * 0.25 + evasion_prob * 0.10)

        return float(np.clip(overall, 0.0, 1.0)), sharpe_impact
