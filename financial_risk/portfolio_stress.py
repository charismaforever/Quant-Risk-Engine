"""
Portfolio Stress Tester

Event-driven stress testing for cyber incident scenarios. Models the
P&L impact of specific, named cyber event types on a multi-asset
portfolio using historical analogs and expert elicitation.

Scenario library includes:
- Ransomware attack (Colonial Pipeline analog)
- SWIFT fraud event (Bangladesh Bank analog)
- Supply chain compromise (SolarWinds analog)
- Quantum cryptographic breach (hypothetical, forward-looking)
- AI model poisoning (novel, no historical analog)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class StressTestResult:
    """Result from a single stress scenario."""
    scenario_name: str
    portfolio_return_impact: float     # Fractional impact on portfolio return
    dollar_loss_usd: float
    recovery_months: float             # Estimated time to full recovery
    probability: float                 # Annual occurrence probability
    expected_annual_loss_usd: float   # probability * dollar_loss


# Stress scenario library
CYBER_STRESS_SCENARIOS = {
    "ransomware_major": {
        "return_impact_mu": -0.08,
        "return_impact_sigma": 0.03,
        "recovery_months_mu": 4.0,
        "annual_probability": 0.12,
        "description": "Major ransomware attack (Colonial Pipeline analog)",
    },
    "swift_fraud": {
        "return_impact_mu": -0.05,
        "return_impact_sigma": 0.02,
        "recovery_months_mu": 2.0,
        "annual_probability": 0.04,
        "description": "SWIFT/payment fraud event (Bangladesh Bank analog)",
    },
    "supply_chain_compromise": {
        "return_impact_mu": -0.12,
        "return_impact_sigma": 0.04,
        "recovery_months_mu": 9.0,
        "annual_probability": 0.06,
        "description": "Supply chain compromise (SolarWinds analog)",
    },
    "quantum_cryptographic_breach": {
        "return_impact_mu": -0.18,
        "return_impact_sigma": 0.06,
        "recovery_months_mu": 18.0,
        "annual_probability": 0.02,
        "description": "Quantum cryptographic breach (forward-looking)",
    },
    "ai_model_poisoning": {
        "return_impact_mu": -0.06,
        "return_impact_sigma": 0.025,
        "recovery_months_mu": 3.0,
        "annual_probability": 0.08,
        "description": "AI/ML trading model poisoning attack",
    },
    "ddos_market_disruption": {
        "return_impact_mu": -0.03,
        "return_impact_sigma": 0.01,
        "recovery_months_mu": 0.5,
        "annual_probability": 0.20,
        "description": "DDoS attack disrupting market access",
    },
}


class PortfolioStressTester:
    """
    Event-driven cyber stress tester for investment portfolios.

    Parameters
    ----------
    portfolio_value_usd : float
        Total portfolio value for dollar-loss computation.
    rng : np.random.Generator
        Seeded RNG.
    """

    def __init__(
        self,
        portfolio_value_usd: float = 500_000_000.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.portfolio_value_usd = portfolio_value_usd
        self.rng = rng or np.random.default_rng(42)

    def run_scenario(self, scenario_name: str) -> StressTestResult:
        """Run a single named stress scenario."""
        if scenario_name not in CYBER_STRESS_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. "
                             f"Available: {list(CYBER_STRESS_SCENARIOS.keys())}")

        params = CYBER_STRESS_SCENARIOS[scenario_name]

        # Sample return impact with uncertainty
        impact = float(self.rng.normal(
            loc=params["return_impact_mu"],
            scale=params["return_impact_sigma"],
        ))
        impact = min(impact, 0.0)  # Stress scenarios only produce losses

        recovery = float(self.rng.exponential(scale=params["recovery_months_mu"]))
        dollar_loss = abs(impact) * self.portfolio_value_usd
        annual_prob = params["annual_probability"]
        expected_annual_loss = annual_prob * dollar_loss

        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_return_impact=impact,
            dollar_loss_usd=dollar_loss,
            recovery_months=recovery,
            probability=annual_prob,
            expected_annual_loss_usd=expected_annual_loss,
        )

    def run_all_scenarios(self) -> list[StressTestResult]:
        """Run all scenarios in the library and sort by expected annual loss."""
        results = [
            self.run_scenario(name)
            for name in CYBER_STRESS_SCENARIOS
        ]
        return sorted(results, key=lambda r: r.expected_annual_loss_usd, reverse=True)

    def aggregate_expected_loss(self) -> float:
        """
        Compute total expected annual loss across all scenarios.
        Note: assumes scenarios are mutually exclusive for simplicity.
        """
        return sum(r.expected_annual_loss_usd for r in self.run_all_scenarios())
