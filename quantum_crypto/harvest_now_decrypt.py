"""
Harvest Now, Decrypt Later (HNDL) Attack Window Model

Models the risk that adversaries are currently harvesting encrypted data
to decrypt it when cryptographically-relevant quantum computers arrive.
This is the most pressing near-term quantum risk for financial institutions.

Key insight: HNDL attacks are happening NOW. The window of vulnerability
is: data_shelf_life + time_to_CRQC > migration_completion_date.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class HNDLExposure:
    """Quantified HNDL attack exposure."""
    window_years: float          # Years of active HNDL vulnerability
    data_at_risk_fraction: float # Fraction of sensitive data exposed
    expected_breach_cost_usd: float
    urgency_score: float         # [0, 1] how urgently migration is needed


class HNDLModel:
    """
    Models Harvest Now, Decrypt Later attack windows.

    Financial institutions are prime HNDL targets because:
    - Transaction records have multi-decade legal retention requirements
    - Client PII + financial data remains sensitive indefinitely
    - Competitive intelligence (M&A, position data) has indefinite value

    Parameters
    ----------
    data_shelf_life_years : float
        How many years the targeted data remains sensitive/valuable.
    years_to_crqc : float
        Expected years until a cryptographically-relevant quantum computer exists.
    migration_years_remaining : float
        Years until the organization completes PQC migration.
    annual_data_volume_gb : float
        Annual sensitive data production (proxy for exposure surface).
    """

    def __init__(
        self,
        data_shelf_life_years: float = 20.0,
        years_to_crqc: float = 7.0,
        migration_years_remaining: float = 3.0,
        annual_data_volume_gb: float = 500.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.data_shelf_life_years = data_shelf_life_years
        self.years_to_crqc = years_to_crqc
        self.migration_years_remaining = migration_years_remaining
        self.annual_data_volume_gb = annual_data_volume_gb
        self.rng = rng or np.random.default_rng(42)

    def mosca_inequality(self) -> bool:
        """
        Evaluate Mosca's Inequality:
        t_migration + t_shelf_life > t_quantum → act immediately.

        Returns True if the organization needs to act NOW.
        """
        return (self.migration_years_remaining + self.data_shelf_life_years) > self.years_to_crqc

    def compute_exposure(self, n_monte_carlo: int = 5_000) -> HNDLExposure:
        """
        Compute HNDL exposure with Monte Carlo uncertainty quantification.
        """
        # Sample CRQC arrival with uncertainty
        crqc_arrivals = self.rng.lognormal(
            mean=np.log(self.years_to_crqc), sigma=0.35, size=n_monte_carlo
        )
        migration_completions = self.rng.normal(
            loc=self.migration_years_remaining, scale=0.5, size=n_monte_carlo
        )
        migration_completions = np.clip(migration_completions, 0.1, None)

        # HNDL window = time between CRQC arrival and migration completion
        hndl_windows = crqc_arrivals - migration_completions
        vulnerable_fraction = float(np.mean(hndl_windows > 0))

        # Data at risk: proportional to shelf life overlap with HNDL window
        shelf_life_overlap = np.minimum(
            np.maximum(hndl_windows, 0), self.data_shelf_life_years
        )
        data_at_risk_fraction = float(np.mean(shelf_life_overlap / self.data_shelf_life_years))

        # Cost model: Ponemon Institute average breach cost proxy
        base_breach_cost = 4_450_000  # USD, 2023 Ponemon global average
        quantum_multiplier = 3.5      # Quantum breaches expected to be more severe
        expected_cost = (
            base_breach_cost * quantum_multiplier * data_at_risk_fraction * vulnerable_fraction
        )

        window_years = float(np.mean(np.maximum(hndl_windows, 0)))
        urgency = float(np.clip(vulnerable_fraction + 0.3 * data_at_risk_fraction, 0.0, 1.0))

        return HNDLExposure(
            window_years=window_years,
            data_at_risk_fraction=data_at_risk_fraction,
            expected_breach_cost_usd=expected_cost,
            urgency_score=urgency,
        )
