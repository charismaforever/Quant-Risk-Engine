"""
Bayesian Updater — Beta-Binomial conjugate prior update engine.

Implements conjugate Bayesian inference for Bernoulli/Binomial likelihoods
with Beta priors. The Beta-Binomial model is the natural choice for modeling
attack success probabilities: we have a Beta prior over p (unknown attack
success probability), observe k successes in n trials, and produce a
Beta posterior.

Posterior: Beta(α + k, β + (n - k))
Posterior mean: (α + k) / (α + β + n)
Posterior variance: Decreases as n grows (evidence accumulates).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class PosteriorResult:
    """Result from a Bayesian update."""
    alpha_posterior: float
    beta_posterior: float
    posterior_mean: float
    posterior_std: float
    credible_interval_95: tuple[float, float]  # 95% HDI


class BayesianUpdater:
    """
    Beta-Binomial conjugate Bayesian updater for attack probability estimation.

    Uses the conjugate prior relationship:
        Prior:     p ~ Beta(α, β)
        Likelihood: k | p ~ Binomial(n, p)
        Posterior:  p | k ~ Beta(α + k, β + n - k)

    This allows sequential updating as new threat intelligence arrives —
    each update's posterior becomes the next observation's prior.

    Parameters
    ----------
    rng : np.random.Generator
        Seeded RNG for sampling.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng(42)

    def update(
        self,
        alpha_prior: float,
        beta_prior: float,
        n_successes: int,
        n_trials: int,
    ) -> float:
        """
        Perform a single Beta-Binomial update and return posterior mean.

        Parameters
        ----------
        alpha_prior : float
            Prior Beta α parameter (pseudo-successes).
        beta_prior : float
            Prior Beta β parameter (pseudo-failures).
        n_successes : int
            Observed number of attack successes.
        n_trials : int
            Total observed trials.

        Returns
        -------
        float
            Posterior mean E[p | data] = (α + k) / (α + β + n).
        """
        alpha_post = alpha_prior + n_successes
        beta_post = beta_prior + (n_trials - n_successes)
        return alpha_post / (alpha_post + beta_post)

    def update_full(
        self,
        alpha_prior: float,
        beta_prior: float,
        n_successes: int,
        n_trials: int,
        n_samples: int = 10_000,
    ) -> PosteriorResult:
        """
        Perform update and return full posterior characterization.

        Parameters
        ----------
        n_samples : int
            Number of Monte Carlo samples from posterior for credible interval.
        """
        alpha_post = alpha_prior + n_successes
        beta_post = beta_prior + (n_trials - n_successes)

        posterior_mean = alpha_post / (alpha_post + beta_post)
        posterior_var = (
            (alpha_post * beta_post)
            / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
        )
        posterior_std = float(np.sqrt(posterior_var))

        samples = self.rng.beta(alpha_post, beta_post, size=n_samples)
        ci_low, ci_high = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))

        return PosteriorResult(
            alpha_posterior=alpha_post,
            beta_posterior=beta_post,
            posterior_mean=float(posterior_mean),
            posterior_std=posterior_std,
            credible_interval_95=(ci_low, ci_high),
        )

    def sequential_update(
        self,
        alpha_prior: float,
        beta_prior: float,
        observations: list[tuple[int, int]],
    ) -> PosteriorResult:
        """
        Sequentially update the posterior as new observations arrive.

        Parameters
        ----------
        observations : list of (n_successes, n_trials) tuples
            Each tuple is one batch of new threat intelligence data.
        """
        alpha, beta = alpha_prior, beta_prior
        for n_successes, n_trials in observations:
            alpha += n_successes
            beta += (n_trials - n_successes)

        return self.update_full(alpha_prior, beta_prior,
                                 int(alpha - alpha_prior), int(alpha + beta - alpha_prior - beta_prior))
