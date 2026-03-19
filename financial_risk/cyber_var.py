"""
CyberVaR Model

Standalone cyber-adjusted Value at Risk model that can be run independently
of the full simulation engine. Useful for quick sensitivity analysis and
scenario comparison.

Extends traditional VaR by treating cyber loss events as a separate
compound Poisson process superimposed on the market return distribution.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class CyberVaRResult:
    """CyberVaR computation result."""
    baseline_var: float           # VaR without cyber shocks
    cyber_var: float              # VaR with cyber shocks
    cyber_var_uplift: float       # Additional loss from cyber (cyber_var - baseline_var)
    cyber_cvar: float             # CVaR with cyber shocks
    cyber_event_contribution: float  # % of total VaR attributable to cyber risk


class CyberVaRModel:
    """
    Cyber-adjusted Value at Risk standalone model.

    Models cyber losses as a compound Poisson process:
    - N events per year ~ Poisson(λ)
    - Each event loss L ~ Pareto(α, x_m) [fat-tailed]

    The cyber loss process is added to the baseline GBM market returns
    to produce cyber-adjusted return paths.

    Parameters
    ----------
    portfolio_value_usd : float
        Total portfolio value for dollar-denominated loss reporting.
    cyber_event_frequency : float
        Annual expected number of material cyber events (λ for Poisson).
    cyber_loss_severity_mean : float
        Mean fractional loss per cyber event (e.g. 0.08 = 8% loss).
    confidence_level : float
        VaR confidence level.
    n_simulations : int
        Monte Carlo paths.
    """

    MARKET_MU = 0.07
    MARKET_SIGMA = 0.15

    def __init__(
        self,
        portfolio_value_usd: float = 500_000_000.0,
        cyber_event_frequency: float = 0.30,
        cyber_loss_severity_mean: float = 0.08,
        confidence_level: float = 0.99,
        n_simulations: int = 50_000,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.portfolio_value_usd = portfolio_value_usd
        self.cyber_event_frequency = cyber_event_frequency
        self.cyber_loss_severity_mean = cyber_loss_severity_mean
        self.confidence_level = confidence_level
        self.n_simulations = n_simulations
        self.rng = rng or np.random.default_rng(42)

    def compute(self) -> CyberVaRResult:
        """Compute cyber-adjusted VaR and CVaR."""
        alpha = 1.0 - self.confidence_level

        # Baseline market returns
        baseline = self.rng.normal(
            loc=self.MARKET_MU - 0.5 * self.MARKET_SIGMA**2,
            scale=self.MARKET_SIGMA,
            size=self.n_simulations,
        )
        baseline_var = float(np.percentile(baseline, 100 * alpha))

        # Cyber loss process: compound Poisson
        n_events = self.rng.poisson(self.cyber_event_frequency, size=self.n_simulations)

        # Pareto-distributed severity (α=2 → finite variance, heavier tail than normal)
        pareto_shape = 2.0
        pareto_scale = self.cyber_loss_severity_mean * (pareto_shape - 1) / pareto_shape
        individual_losses = self.rng.pareto(pareto_shape, size=self.n_simulations) * pareto_scale
        total_cyber_losses = n_events * individual_losses  # Simplified compound

        # Cyber-adjusted returns
        cyber_adjusted = baseline - total_cyber_losses
        cyber_var = float(np.percentile(cyber_adjusted, 100 * alpha))
        cyber_cvar = float(np.mean(cyber_adjusted[cyber_adjusted <= cyber_var]))

        uplift = cyber_var - baseline_var
        contribution = abs(uplift) / abs(cyber_var) if cyber_var != 0 else 0.0

        return CyberVaRResult(
            baseline_var=baseline_var,
            cyber_var=cyber_var,
            cyber_var_uplift=uplift,
            cyber_cvar=cyber_cvar,
            cyber_event_contribution=float(contribution),
        )
