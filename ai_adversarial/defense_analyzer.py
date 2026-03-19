"""
Defense Analyzer

Evaluates the effectiveness and ROI of adversarial ML defense mechanisms
deployed against quantitative trading system attack vectors.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


DEFENSE_COSTS_ANNUAL_USD: dict[str, float] = {
    "adversarial_training": 180_000,
    "input_validation": 60_000,
    "ensemble_defense": 250_000,
    "differential_privacy": 120_000,
    "output_perturbation": 45_000,
    "rate_limiting": 30_000,
    "model_watermarking": 80_000,
}

DEFENSE_RISK_REDUCTION: dict[str, float] = {
    "adversarial_training": 0.35,
    "input_validation": 0.20,
    "ensemble_defense": 0.30,
    "differential_privacy": 0.45,
    "output_perturbation": 0.25,
    "rate_limiting": 0.15,
    "model_watermarking": 0.10,
}


@dataclass
class DefenseROI:
    """Return-on-investment analysis for a defense portfolio."""
    defense_name: str
    annual_cost_usd: float
    risk_reduction_fraction: float
    expected_annual_loss_avoided_usd: float
    roi_ratio: float       # Loss avoided / Cost; >1 = positive ROI
    payback_months: float


class DefenseAnalyzer:
    """
    Analyzes defense effectiveness and ROI for adversarial ML mitigations.

    Parameters
    ----------
    baseline_annual_loss_usd : float
        Expected annual loss from adversarial attacks without defenses.
    deployed_defenses : list[str]
        List of currently deployed defense mechanisms.
    """

    def __init__(
        self,
        baseline_annual_loss_usd: float = 5_000_000.0,
        deployed_defenses: list[str] | None = None,
    ) -> None:
        self.baseline_annual_loss_usd = baseline_annual_loss_usd
        self.deployed_defenses = deployed_defenses or []

    def analyze_all(self) -> list[DefenseROI]:
        """Compute ROI for all known defense mechanisms."""
        results = []
        for defense, cost in DEFENSE_COSTS_ANNUAL_USD.items():
            reduction = DEFENSE_RISK_REDUCTION.get(defense, 0.10)
            loss_avoided = self.baseline_annual_loss_usd * reduction
            roi = loss_avoided / cost if cost > 0 else 0.0
            payback = (cost / loss_avoided * 12) if loss_avoided > 0 else float("inf")

            results.append(DefenseROI(
                defense_name=defense,
                annual_cost_usd=cost,
                risk_reduction_fraction=reduction,
                expected_annual_loss_avoided_usd=loss_avoided,
                roi_ratio=roi,
                payback_months=payback,
            ))

        return sorted(results, key=lambda x: x.roi_ratio, reverse=True)

    def recommend(self, budget_usd: float = 500_000.0) -> list[str]:
        """Recommend the optimal defense portfolio within a budget constraint."""
        all_rois = self.analyze_all()
        selected = []
        remaining = budget_usd
        for defense_roi in all_rois:
            if defense_roi.annual_cost_usd <= remaining and defense_roi.roi_ratio > 1.0:
                selected.append(defense_roi.defense_name)
                remaining -= defense_roi.annual_cost_usd
        return selected
