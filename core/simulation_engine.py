"""
SimulationEngine — core orchestration layer for QuantumRiskEngine.

Runs all four domain modules as a composable Monte Carlo pipeline and
produces a unified SimulationResult containing cyber-adjusted risk metrics.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scenario import Scenario
from quantum_crypto.pqc_vulnerability import PQCVulnerabilityModel
from ai_adversarial.attack_simulator import AdversarialAttackSimulator
from threat_intel.ttp_engine import TTPEngine
from financial_risk.monte_carlo import CyberAdjustedMonteCarlo
from financial_risk.risk_metrics import RiskMetrics


@dataclass
class SimulationResult:
    """
    Aggregated output from a full QuantumRiskEngine simulation run.

    Attributes
    ----------
    scenario_name : str
        Name of the scenario that produced these results.
    quantum_risk_score : float
        Probability [0, 1] of a cryptographically-relevant quantum breach
        within the simulation horizon.
    adversarial_threat_score : float
        Estimated attack success probability against the ML trading stack.
    threat_likelihood : float
        Bayesian-updated posterior probability of a significant breach event
        in the given sector within one year.
    cyber_var : float
        Cyber-adjusted Value at Risk (negative = loss) at configured confidence level.
    cyber_cvar : float
        Cyber-adjusted Conditional VaR (Expected Shortfall) beyond VaR threshold.
    sharpe_impact : float
        Estimated Sharpe ratio degradation under adversarial ML scenario.
    elapsed_seconds : float
        Wall-clock time for the full simulation run.
    n_simulations : int
        Number of Monte Carlo paths used.
    """

    scenario_name: str
    quantum_risk_score: float
    adversarial_threat_score: float
    threat_likelihood: float
    cyber_var: float
    cyber_cvar: float
    sharpe_impact: float
    elapsed_seconds: float
    n_simulations: int

    def summary(self) -> str:
        """Return a formatted summary string for display."""
        lines = [
            f"\n{'='*60}",
            f"  QuantumRiskEngine — Results: {self.scenario_name}",
            f"{'='*60}",
            f"  QuantumRiskScore:       {self.quantum_risk_score:.2f}  "
            f"({'high' if self.quantum_risk_score > 0.5 else 'moderate' if self.quantum_risk_score > 0.25 else 'low'} PQC exposure)",
            f"  AdversarialThreat:      {self.adversarial_threat_score:.2f}  "
            f"({'high' if self.adversarial_threat_score > 0.6 else 'moderate'} ML attack surface)",
            f"  ThreatLikelihood:       {self.threat_likelihood:.2f}  "
            f"(sector-weighted posterior)",
            f"  Cyber-adjusted VaR:    {self.cyber_var:+.1%}  "
            f"({int((1 - 0.01)*100)}% confidence, 1-year)",
            f"  Cyber-adjusted CVaR:   {self.cyber_cvar:+.1%}  "
            f"(expected shortfall)",
            f"  Sharpe Impact:         {self.sharpe_impact:+.3f}  "
            f"(adversarial ML degradation)",
            f"  Simulations:           {self.n_simulations:,}",
            f"  Runtime:               {self.elapsed_seconds:.2f}s",
            f"{'='*60}\n",
        ]
        return "\n".join(lines)


class SimulationEngine:
    """
    Orchestrates the full four-module QuantumRiskEngine pipeline.

    Parameters
    ----------
    scenario : Scenario
        Fully parameterized scenario configuration.
    n_simulations : int
        Override for Monte Carlo path count (default: from scenario).
    seed : int
        Random seed for reproducibility.

    Example
    -------
    >>> from core.scenario import Scenario
    >>> from core.simulation_engine import SimulationEngine
    >>> scenario = Scenario.from_yaml("config/scenarios/hedge_fund_baseline.yaml")
    >>> engine = SimulationEngine(scenario)
    >>> results = engine.run()
    >>> print(results.summary())
    """

    def __init__(
        self,
        scenario: Scenario,
        n_simulations: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.scenario = scenario
        self.n_simulations = n_simulations or scenario.financial_risk.n_simulations
        self.seed = seed if seed is not None else scenario.random_seed
        self.rng = np.random.default_rng(self.seed)

    def run(self) -> SimulationResult:
        """Execute the full simulation pipeline and return results."""
        t0 = time.perf_counter()

        # --- Module 1: Quantum Crypto ---
        pqc_model = PQCVulnerabilityModel(
            params=self.scenario.quantum_crypto,
            rng=self.rng,
        )
        quantum_risk_score = pqc_model.compute_risk_score()

        # --- Module 2: Adversarial AI ---
        adversarial_sim = AdversarialAttackSimulator(
            params=self.scenario.adversarial_ai,
            rng=self.rng,
        )
        adversarial_threat_score, sharpe_impact = adversarial_sim.simulate()

        # --- Module 3: Threat Intelligence ---
        ttp_engine = TTPEngine(
            params=self.scenario.threat_intel,
            rng=self.rng,
        )
        threat_likelihood = ttp_engine.compute_posterior()

        # --- Module 4: Financial Risk (integration layer) ---
        mc = CyberAdjustedMonteCarlo(
            params=self.scenario.financial_risk,
            quantum_risk_score=quantum_risk_score,
            adversarial_threat_score=adversarial_threat_score,
            threat_likelihood=threat_likelihood,
            n_simulations=self.n_simulations,
            rng=self.rng,
        )
        portfolio_returns = mc.run()
        metrics = RiskMetrics(portfolio_returns, self.scenario.financial_risk.confidence_level)

        elapsed = time.perf_counter() - t0

        return SimulationResult(
            scenario_name=self.scenario.name,
            quantum_risk_score=quantum_risk_score,
            adversarial_threat_score=adversarial_threat_score,
            threat_likelihood=threat_likelihood,
            cyber_var=metrics.var(),
            cyber_cvar=metrics.cvar(),
            sharpe_impact=sharpe_impact,
            elapsed_seconds=elapsed,
            n_simulations=self.n_simulations,
        )


def cli_entry() -> None:
    """Command-line entry point: `qre --config <path>`"""
    parser = argparse.ArgumentParser(
        description="QuantumRiskEngine — run a cyber-adjusted risk simulation"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/scenarios/hedge_fund_baseline.yaml"),
        help="Path to YAML scenario configuration file",
    )
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=None,
        help="Override Monte Carlo simulation count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    args = parser.parse_args()

    scenario = Scenario.from_yaml(args.config)
    engine = SimulationEngine(
        scenario,
        n_simulations=args.n_simulations,
        seed=args.seed,
    )
    results = engine.run()
    print(results.summary())


if __name__ == "__main__":
    cli_entry()
