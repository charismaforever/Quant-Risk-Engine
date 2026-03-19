"""Unit tests for threat_intel.ttp_engine module."""

import numpy as np
import pytest

from core.scenario import ThreatIntelParams
from threat_intel.ttp_engine import TTPEngine


@pytest.fixture
def default_params():
    return ThreatIntelParams()


@pytest.fixture
def engine(default_params):
    return TTPEngine(default_params, rng=np.random.default_rng(42))


class TestTTPEngine:
    def test_posterior_in_unit_interval(self, engine):
        likelihood = engine.compute_posterior()
        assert 0.0 <= likelihood <= 1.0

    def test_posterior_is_float(self, engine):
        assert isinstance(engine.compute_posterior(), float)

    def test_more_threat_actors_increases_likelihood(self):
        few = ThreatIntelParams(threat_actors=["FIN7"])
        many = ThreatIntelParams(threat_actors=["FIN7", "Lazarus", "APT29", "APT41"])
        rng = np.random.default_rng(42)
        score_few = TTPEngine(few, rng).compute_posterior()
        rng = np.random.default_rng(42)
        score_many = TTPEngine(many, rng).compute_posterior()
        assert score_many >= score_few

    def test_likelihood_matrix_has_correct_actors(self, engine, default_params):
        matrix = engine.build_likelihood_matrix()
        for actor in default_params.threat_actors:
            assert actor in matrix.posterior_by_actor

    def test_likelihood_matrix_overall_matches_compute_posterior(self):
        params = ThreatIntelParams()
        rng = np.random.default_rng(42)
        engine = TTPEngine(params, rng)
        matrix = engine.build_likelihood_matrix()
        rng2 = np.random.default_rng(42)
        engine2 = TTPEngine(params, rng2)
        scalar = engine2.compute_posterior()
        # They use same seed so should match
        assert abs(matrix.overall_likelihood - scalar) < 1e-9

    def test_reproducible_with_same_seed(self, default_params):
        s1 = TTPEngine(default_params, np.random.default_rng(7)).compute_posterior()
        s2 = TTPEngine(default_params, np.random.default_rng(7)).compute_posterior()
        assert s1 == s2
