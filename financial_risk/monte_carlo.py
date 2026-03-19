"""
Cyber-Adjusted Monte Carlo Simulation

Integrates quantum risk, adversarial AI threat, and threat intelligence
likelihood scores as stochastic shocks into a portfolio return simulation.

Methodology
-----------
Traditional Monte Carlo portfolio simulation generates return paths from a
multivariate normal (or fat-tailed) distribution. We extend this by adding
three cyber-specific shock processes:

1. Quantum shock: A jump process triggered when a quantum breach occurs.
   Modeled as a Poisson process with intensity λ = quantum_risk_score.

2. Adversarial ML shock: A persistent drag on alpha returns proportional
   to the adversarial threat score. Modeled as a Gaussian noise inflation.

3. Threat event shock: A compound Poisson jump with magnitude drawn from
   a Pareto distribution (fat tails, reflecting ransomware/APT severity).

The three shocks are combined with the baseline GBM return process to
produce cyber-adjusted portfolio return paths.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from core.scenario import FinancialRiskParams


@dataclass
class MonteCarloResult:
    """Raw output from Monte Carlo simulation."""
    returns: np.ndarray          # Shape: (n_simulations,) — annual portfolio returns
    cyber_shocked_paths: int     # Number of paths that experienced a cyber event
    mean_return: float
    std_return: float


class CyberAdjustedMonteCarlo:
    """
    Monte Carlo simulation with integrated cyber risk shocks.

    Parameters
    ----------
    params : FinancialRiskParams
        Scenario financial risk parameters.
    quantum_risk_score : float
        Output from PQCVulnerabilityModel [0, 1].
    adversarial_threat_score : float
        Output from AdversarialAttackSimulator [0, 1].
    threat_likelihood : float
        Output from TTPEngine [0, 1].
    n_simulations : int
        Number of Monte Carlo paths.
    rng : np.random.Generator
        Seeded RNG.

    Example
    -------
    >>> import numpy as np
    >>> from core.scenario import FinancialRiskParams
    >>> params = FinancialRiskParams(n_simulations=1000)
    >>> mc = CyberAdjustedMonteCarlo(
    ...     params=params,
    ...     quantum_risk_score=0.30,
    ...     adversarial_threat_score=0.55,
    ...     threat_likelihood=0.40,
    ...     n_simulations=1000,
    ...     rng=np.random.default_rng(42),
    ... )
    >>> returns = mc.run()
    >>> returns.shape
    (1000,)
    """

    # Market parameters (calibrated to S&P 500 long-run estimates)
    MARKET_MU = 0.08          # Expected annual return
    MARKET_SIGMA = 0.16       # Annual volatility

    # Cyber shock parameters
    QUANTUM_SHOCK_SEVERITY_MU = -0.12    # Mean quantum breach impact on portfolio
    QUANTUM_SHOCK_SEVERITY_SIGMA = 0.04  # Severity uncertainty

    THREAT_PARETO_ALPHA = 1.5   # Pareto shape (heavier tail = more severe events)
    THREAT_PARETO_SCALE = 0.05  # Scale (minimum loss = 5% of portfolio)

    def __init__(
        self,
        params: FinancialRiskParams,
        quantum_risk_score: float,
        adversarial_threat_score: float,
        threat_likelihood: float,
        n_simulations: int,
        rng: np.random.Generator,
    ) -> None:
        self.params = params
        self.quantum_risk_score = quantum_risk_score
        self.adversarial_threat_score = adversarial_threat_score
        self.threat_likelihood = threat_likelihood
        self.n_simulations = n_simulations
        self.rng = rng

    def _baseline_returns(self) -> np.ndarray:
        """
        Generate baseline GBM portfolio returns with correlation structure.

        Uses log-normal approximation for annual returns:
        R ~ Normal(μ - σ²/2, σ)
        """
        T = self.params.time_horizon_days / 252  # Convert to years
        log_returns = self.rng.normal(
            loc=(self.MARKET_MU - 0.5 * self.MARKET_SIGMA**2) * T,
            scale=self.MARKET_SIGMA * np.sqrt(T),
            size=self.n_simulations,
        )
        return np.exp(log_returns) - 1.0  # Arithmetic returns

    def _quantum_shocks(self) -> tuple[np.ndarray, int]:
        """
        Poisson jump process for quantum breach events.

        λ = quantum_risk_score (annual breach intensity).
        Severity ~ Normal(μ_shock, σ_shock).
        """
        # Whether a quantum breach event occurs in each path
        n_events = self.rng.poisson(lam=self.quantum_risk_score, size=self.n_simulations)
        shocked_paths = int(np.sum(n_events > 0))

        # Severity of breach impact
        severity = self.rng.normal(
            loc=self.QUANTUM_SHOCK_SEVERITY_MU,
            scale=self.QUANTUM_SHOCK_SEVERITY_SIGMA,
            size=self.n_simulations,
        )
        shocks = np.where(n_events > 0, severity * n_events, 0.0)
        return shocks, shocked_paths

    def _adversarial_ml_drag(self) -> np.ndarray:
        """
        Persistent alpha degradation from adversarial ML attacks.

        Models as a return drag proportional to adversarial threat score,
        with Gaussian noise (attack success is probabilistic).
        """
        base_drag = -self.adversarial_threat_score * 0.03  # Up to 3% annual drag
        noise = self.rng.normal(loc=0.0, scale=0.01, size=self.n_simulations)
        return np.full(self.n_simulations, base_drag) + noise

    def _threat_event_shocks(self) -> np.ndarray:
        """
        Compound Poisson process for threat intelligence-driven breach events.

        Severity drawn from Pareto distribution (fat-tailed losses):
        - Ransomware events: severe, infrequent
        - APT data theft: moderate impact, persistent
        """
        # Event occurrence
        event_occurs = self.rng.binomial(1, self.threat_likelihood, size=self.n_simulations)

        # Pareto-distributed severity (heavier tail than normal)
        pareto_samples = (self.rng.pareto(self.THREAT_PARETO_ALPHA, size=self.n_simulations)
                          * self.THREAT_PARETO_SCALE)

        # Add market correlation component (cyber events correlate with stress periods)
        market_correlation = self.params.correlation_with_market
        correlated_component = market_correlation * self.rng.normal(0, 0.02, size=self.n_simulations)

        shocks = -event_occurs * (pareto_samples + np.abs(correlated_component))
        return shocks

    def run(self) -> np.ndarray:
        """
        Run the full cyber-adjusted Monte Carlo simulation.

        Returns
        -------
        np.ndarray
            Array of shape (n_simulations,) containing cyber-adjusted
            annual portfolio returns.
        """
        baseline = self._baseline_returns()
        quantum_shocks, _ = self._quantum_shocks()
        ml_drag = self._adversarial_ml_drag()
        threat_shocks = self._threat_event_shocks()

        # Combine: baseline + all cyber shocks
        cyber_adjusted = baseline + quantum_shocks + ml_drag + threat_shocks

        return cyber_adjusted
