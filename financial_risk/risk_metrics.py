"""
Risk Metrics — VaR, CVaR, Expected Shortfall, and Sharpe-adjusted measures.

Computes standard quantitative risk metrics from a vector of simulated
portfolio returns, with cyber-specific extensions.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FullRiskReport:
    """Complete risk metrics report from simulated return distribution."""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float    # Alias for CVaR at configured confidence
    mean_return: float
    volatility: float
    skewness: float
    excess_kurtosis: float
    sharpe_ratio: float          # Assumes risk-free rate of 5%
    max_drawdown_estimate: float
    probability_of_loss: float   # P(return < 0)


class RiskMetrics:
    """
    Computes quantitative risk metrics from Monte Carlo return simulations.

    Parameters
    ----------
    returns : np.ndarray
        Array of simulated portfolio returns (arithmetic, annual).
    confidence_level : float
        VaR/CVaR confidence level (e.g. 0.99 for 99%).
    risk_free_rate : float
        Annual risk-free rate for Sharpe ratio computation.

    Example
    -------
    >>> import numpy as np
    >>> returns = np.random.default_rng(42).normal(0.08, 0.16, 10000)
    >>> metrics = RiskMetrics(returns, confidence_level=0.99)
    >>> metrics.var() < 0
    True
    """

    RISK_FREE_RATE = 0.05  # 2024 approximate fed funds rate

    def __init__(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.99,
    ) -> None:
        self.returns = returns
        self.confidence_level = confidence_level
        self._alpha = 1.0 - confidence_level  # Lower tail probability

    def var(self) -> float:
        """
        Value at Risk at the configured confidence level.

        VaR_{α}(X) = -inf{x : P(X ≤ x) > α}
        Returns a negative value (loss convention).
        """
        return float(np.percentile(self.returns, 100 * self._alpha))

    def cvar(self) -> float:
        """
        Conditional Value at Risk (Expected Shortfall) at configured confidence.

        CVaR_{α}(X) = E[X | X ≤ VaR_{α}(X)]
        Returns a negative value. CVaR is always ≤ VaR (more conservative).
        """
        var_threshold = self.var()
        tail_returns = self.returns[self.returns <= var_threshold]
        if len(tail_returns) == 0:
            return var_threshold
        return float(np.mean(tail_returns))

    def sharpe_ratio(self) -> float:
        """
        Annualized Sharpe ratio: (E[R] - r_f) / σ(R).
        """
        excess = np.mean(self.returns) - self.RISK_FREE_RATE
        sigma = np.std(self.returns, ddof=1)
        return float(excess / sigma) if sigma > 0 else 0.0

    def full_report(self) -> FullRiskReport:
        """Compute the complete risk metrics report."""
        from scipy import stats as sp_stats

        mean_r = float(np.mean(self.returns))
        vol = float(np.std(self.returns, ddof=1))
        skew = float(sp_stats.skew(self.returns))
        kurt = float(sp_stats.kurtosis(self.returns))  # Excess kurtosis
        prob_loss = float(np.mean(self.returns < 0))

        # Estimate max drawdown from return distribution (simplified)
        sorted_returns = np.sort(self.returns)
        max_dd = float(np.percentile(sorted_returns, 1))  # 1st percentile proxy

        return FullRiskReport(
            var_95=float(np.percentile(self.returns, 5)),
            var_99=float(np.percentile(self.returns, 1)),
            cvar_95=float(np.mean(self.returns[self.returns <= np.percentile(self.returns, 5)])),
            cvar_99=self.cvar(),
            expected_shortfall=self.cvar(),
            mean_return=mean_r,
            volatility=vol,
            skewness=skew,
            excess_kurtosis=kurt,
            sharpe_ratio=self.sharpe_ratio(),
            max_drawdown_estimate=max_dd,
            probability_of_loss=prob_loss,
        )
