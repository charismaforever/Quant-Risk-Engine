"""
TTP Engine — MITRE ATT&CK Statistical Frequency Modeling

Ingests MITRE ATT&CK framework data and applies Bayesian inference to
produce sector-specific posterior attack likelihood scores. Maps threat
actor TTPs to industry verticals relevant to hedge fund portfolio companies.

Model
-----
We treat each TTP as an independent Bernoulli event with unknown probability p.
We place a Beta(α, β) prior on p based on historical MITRE ATT&CK usage frequency
data, then update it with any observed incidents to produce a posterior.

The ThreatLikelihoodMatrix output is the expected value of the posterior
distribution, E[p] = α / (α + β), for each sector-threat actor pair.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from core.scenario import ThreatIntelParams
from .bayesian_updater import BayesianUpdater
from .sector_mapper import SECTOR_BASE_RATES, THREAT_ACTOR_PROFILES


@dataclass
class ThreatLikelihoodMatrix:
    """Output from TTP Engine."""
    overall_likelihood: float              # [0, 1] posterior breach probability
    sector_risk_score: float               # Sector-specific risk (vs. baseline)
    active_threat_actors: list[str]        # Threat actors with highest posterior
    top_ttps: list[str]                    # Most likely TTPs for this sector
    posterior_by_actor: dict[str, float]   # Per-actor posterior probabilities


class TTPEngine:
    """
    Statistical TTP engine using MITRE ATT&CK frequency data.

    Parameters
    ----------
    params : ThreatIntelParams
        Scenario parameters.
    rng : np.random.Generator
        Seeded RNG.

    Example
    -------
    >>> from core.scenario import ThreatIntelParams
    >>> import numpy as np
    >>> params = ThreatIntelParams()
    >>> engine = TTPEngine(params, rng=np.random.default_rng(42))
    >>> likelihood = engine.compute_posterior()
    >>> 0.0 <= likelihood <= 1.0
    True
    """

    def __init__(self, params: ThreatIntelParams, rng: np.random.Generator) -> None:
        self.params = params
        self.rng = rng
        self._sector_base_rate = SECTOR_BASE_RATES.get(params.sector, 0.12)
        self._updater = BayesianUpdater(rng=rng)

    def _compute_actor_posterior(self, actor: str) -> float:
        """
        Compute Bayesian posterior probability for a specific threat actor
        targeting this sector.
        """
        profile = THREAT_ACTOR_PROFILES.get(actor, {})
        sector_affinity = profile.get(self.params.sector, 0.20)

        # Prior: Beta(α, β) parameterized from sector affinity
        # Higher affinity → more concentrated prior around higher probabilities
        alpha_prior = max(1.0, 10.0 * sector_affinity)
        beta_prior = max(1.0, 10.0 * (1.0 - sector_affinity))

        # Simulate "observations" from MITRE ATT&CK incident reports
        n_incidents = self.rng.poisson(lam=5)
        n_targeted = self.rng.binomial(n=n_incidents, p=sector_affinity)

        posterior_mean = self._updater.update(
            alpha_prior=alpha_prior,
            beta_prior=beta_prior,
            n_successes=n_targeted,
            n_trials=n_incidents,
        )
        return posterior_mean

    def compute_posterior(self) -> float:
        """
        Compute the overall posterior threat likelihood for the scenario.

        Returns
        -------
        float
            Posterior probability [0, 1] of a significant breach within 1 year.
        """
        actor_posteriors = {
            actor: self._compute_actor_posterior(actor)
            for actor in self.params.threat_actors
        }

        if not actor_posteriors:
            return self._sector_base_rate

        # Combined likelihood: probability that at least one actor succeeds
        # P(at least one) = 1 - P(none succeed) = 1 - ∏(1 - p_i)
        prob_none = 1.0
        for p in actor_posteriors.values():
            prob_none *= (1.0 - p)
        combined = 1.0 - prob_none

        # Weight by sector base rate
        final = 0.6 * combined + 0.4 * self._sector_base_rate
        return float(np.clip(final, 0.0, 1.0))

    def build_likelihood_matrix(self) -> ThreatLikelihoodMatrix:
        """Build the full ThreatLikelihoodMatrix for this scenario."""
        actor_posteriors = {
            actor: self._compute_actor_posterior(actor)
            for actor in self.params.threat_actors
        }

        overall = self.compute_posterior()
        sector_risk = overall / max(self._sector_base_rate, 0.01)

        active_actors = sorted(actor_posteriors, key=lambda k: actor_posteriors[k], reverse=True)

        # Top TTPs for financial services (from MITRE ATT&CK frequency data)
        top_ttps = [
            "T1566 - Phishing",
            "T1190 - Exploit Public-Facing Application",
            "T1078 - Valid Accounts",
            "T1486 - Data Encrypted for Impact",
            "T1041 - Exfiltration Over C2 Channel",
        ]

        return ThreatLikelihoodMatrix(
            overall_likelihood=overall,
            sector_risk_score=sector_risk,
            active_threat_actors=active_actors,
            top_ttps=top_ttps,
            posterior_by_actor=actor_posteriors,
        )
