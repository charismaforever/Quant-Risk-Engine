"""
PQC Migration Scheduler

Models the cost, timeline, and risk reduction curve of migrating from
classical to post-quantum cryptography across an enterprise. Supports
phased migration planning with asset prioritization.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MigrationPlan:
    """Output from migration scheduling optimization."""
    total_cost_usd: float
    timeline_years: float
    risk_reduction_curve: list[float]   # Annual risk reduction percentages
    recommended_order: list[str]        # Prioritized asset/system list
    completion_probability_by_year: list[float]


# Asset classes and their migration complexity weights
ASSET_MIGRATION_WEIGHTS = {
    "TLS/HTTPS infrastructure": 1.0,
    "VPN and remote access": 1.2,
    "Email encryption (S/MIME)": 0.8,
    "Database encryption at rest": 1.5,
    "Code signing certificates": 0.7,
    "HSM and key management": 2.0,
    "API authentication (JWT/OAuth)": 1.1,
    "Backup encryption": 0.9,
    "Trading platform comms": 1.8,
    "Customer data vaults": 1.6,
}


class MigrationScheduler:
    """
    Schedules PQC migration across enterprise asset classes.

    Uses a risk-weighted greedy scheduling approach: assets with the
    highest breach impact and lowest migration cost are prioritized first.

    Parameters
    ----------
    budget_usd : float
        Total PQC migration budget.
    target_algorithm : str
        Target NIST PQC algorithm (e.g. CRYSTALS-Kyber).
    years_available : float
        Migration window in years before quantum risk becomes critical.
    """

    def __init__(
        self,
        budget_usd: float = 2_000_000.0,
        target_algorithm: str = "CRYSTALS-Kyber",
        years_available: float = 5.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.budget_usd = budget_usd
        self.target_algorithm = target_algorithm
        self.years_available = years_available
        self.rng = rng or np.random.default_rng(42)

    def _estimate_asset_cost(self, asset: str, weight: float) -> float:
        """Estimate migration cost for a given asset type."""
        base = 150_000 * weight
        noise = self.rng.uniform(0.8, 1.3)
        return base * noise

    def compute_plan(self) -> MigrationPlan:
        """Generate an optimized migration plan."""
        asset_costs = {
            asset: self._estimate_asset_cost(asset, weight)
            for asset, weight in ASSET_MIGRATION_WEIGHTS.items()
        }

        # Sort by cost ascending (quick wins first — maximize coverage per dollar)
        prioritized = sorted(asset_costs.items(), key=lambda x: x[1])

        selected = []
        cumulative_cost = 0.0
        for asset, cost in prioritized:
            if cumulative_cost + cost <= self.budget_usd:
                selected.append(asset)
                cumulative_cost += cost

        total_assets = len(ASSET_MIGRATION_WEIGHTS)
        coverage_fraction = len(selected) / total_assets

        # Risk reduction curve: logistic adoption curve over the migration window
        years = list(range(1, int(self.years_available) + 2))
        risk_reduction_curve = [
            float(coverage_fraction / (1 + np.exp(-1.5 * (y - self.years_available / 2))))
            for y in years
        ]

        # Completion probability per year (assume normal completion time distribution)
        completion_probs = [
            float(np.clip(0.1 + (y / self.years_available) * 0.85, 0.0, 1.0))
            for y in years
        ]

        return MigrationPlan(
            total_cost_usd=cumulative_cost,
            timeline_years=self.years_available,
            risk_reduction_curve=risk_reduction_curve,
            recommended_order=selected,
            completion_probability_by_year=completion_probs,
        )
