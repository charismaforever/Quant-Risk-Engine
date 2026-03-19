"""Unit tests for ai_adversarial.attack_simulator module."""

import numpy as np
import pytest

from core.scenario import AdversarialAIParams
from ai_adversarial.attack_simulator import AdversarialAttackSimulator


@pytest.fixture
def default_params():
    return AdversarialAIParams()


@pytest.fixture
def simulator(default_params):
    return AdversarialAttackSimulator(default_params, rng=np.random.default_rng(42))


class TestAdversarialAttackSimulator:
    def test_overall_score_in_unit_interval(self, simulator):
        score, _ = simulator.simulate()
        assert 0.0 <= score <= 1.0

    def test_sharpe_impact_is_nonpositive(self, simulator):
        _, sharpe_impact = simulator.simulate()
        assert sharpe_impact <= 0.0

    def test_evasion_prob_in_unit_interval(self, simulator):
        prob = simulator.simulate_evasion_attack()
        assert 0.0 <= prob <= 1.0

    def test_poisoning_prob_in_unit_interval(self, simulator):
        prob = simulator.simulate_poisoning_attack()
        assert 0.0 <= prob <= 1.0

    def test_inversion_prob_in_unit_interval(self, simulator):
        prob = simulator.simulate_model_inversion()
        assert 0.0 <= prob <= 1.0

    def test_more_defenses_reduces_threat(self):
        no_defense = AdversarialAIParams(defense_mechanisms=[])
        full_defense = AdversarialAIParams(defense_mechanisms=[
            "adversarial_training", "ensemble_defense", "differential_privacy",
            "input_validation", "output_perturbation",
        ])
        rng = np.random.default_rng(42)
        score_no_def, _ = AdversarialAttackSimulator(no_defense, rng).simulate()
        rng = np.random.default_rng(42)
        score_full_def, _ = AdversarialAttackSimulator(full_defense, rng).simulate()
        assert score_full_def < score_no_def

    def test_higher_exposure_increases_threat(self):
        low_exp = AdversarialAIParams(training_data_exposure=0.01, defense_mechanisms=[])
        high_exp = AdversarialAIParams(training_data_exposure=0.50, defense_mechanisms=[])
        rng = np.random.default_rng(42)
        score_low, _ = AdversarialAttackSimulator(low_exp, rng).simulate()
        rng = np.random.default_rng(42)
        score_high, _ = AdversarialAttackSimulator(high_exp, rng).simulate()
        assert score_high >= score_low

    def test_reproducible_with_same_seed(self, default_params):
        score1, _ = AdversarialAttackSimulator(
            default_params, np.random.default_rng(99)
        ).simulate()
        score2, _ = AdversarialAttackSimulator(
            default_params, np.random.default_rng(99)
        ).simulate()
        assert score1 == score2
