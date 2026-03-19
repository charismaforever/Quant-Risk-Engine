"""
Trading Algorithm Threat Model

Finance-specific adversarial threat surface analysis for ML-driven
quantitative trading strategies. Maps attack vectors to specific
trading strategy types and quantifies expected P&L impact.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


STRATEGY_ATTACK_SURFACES: dict[str, dict[str, float]] = {
    "momentum": {
        "signal_noise_injection": 0.70,
        "feature_manipulation": 0.60,
        "latency_exploitation": 0.80,
    },
    "mean_reversion": {
        "signal_noise_injection": 0.55,
        "feature_manipulation": 0.65,
        "latency_exploitation": 0.50,
    },
    "market_making": {
        "signal_noise_injection": 0.45,
        "feature_manipulation": 0.50,
        "latency_exploitation": 0.90,
    },
    "statistical_arbitrage": {
        "signal_noise_injection": 0.65,
        "feature_manipulation": 0.75,
        "latency_exploitation": 0.40,
    },
    "sentiment_alpha": {
        "signal_noise_injection": 0.85,
        "feature_manipulation": 0.80,
        "latency_exploitation": 0.30,
    },
}


@dataclass
class StrategyThreatProfile:
    """Adversarial threat profile for a trading strategy."""
    strategy_type: str
    overall_vulnerability: float
    highest_risk_vector: str
    estimated_annual_pnl_impact_usd: float
    attack_surface_scores: dict[str, float]


class TradingAlgoThreatModel:
    """
    Models adversarial threats specific to quantitative trading strategies.

    Sentiment-based alpha (NLP signals from news/social) has the highest
    adversarial exposure — adversaries can manipulate input text data.
    High-frequency market-making has extreme latency exploitation risk.

    Parameters
    ----------
    strategy_type : str
        One of: momentum, mean_reversion, market_making,
        statistical_arbitrage, sentiment_alpha.
    annual_pnl_usd : float
        Expected annual P&L for the strategy (for impact quantification).
    """

    def __init__(
        self,
        strategy_type: str = "statistical_arbitrage",
        annual_pnl_usd: float = 10_000_000.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.strategy_type = strategy_type
        self.annual_pnl_usd = annual_pnl_usd
        self.rng = rng or np.random.default_rng(42)
        self._surfaces = STRATEGY_ATTACK_SURFACES.get(
            strategy_type, STRATEGY_ATTACK_SURFACES["statistical_arbitrage"]
        )

    def profile(self) -> StrategyThreatProfile:
        """Generate a full threat profile for the trading strategy."""
        scores = {
            vector: self.rng.beta(10 * prob, 10 * (1 - prob))
            for vector, prob in self._surfaces.items()
        }
        overall = float(np.mean(list(scores.values())))
        highest = max(scores, key=lambda k: scores[k])
        pnl_impact = overall * self.annual_pnl_usd * 0.12  # Expect up to 12% P&L drag

        return StrategyThreatProfile(
            strategy_type=self.strategy_type,
            overall_vulnerability=overall,
            highest_risk_vector=highest,
            estimated_annual_pnl_impact_usd=pnl_impact,
            attack_surface_scores=scores,
        )
