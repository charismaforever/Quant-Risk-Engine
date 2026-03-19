"""Integration tests for the full SimulationEngine pipeline."""

import numpy as np
import pytest
from pathlib import Path

from core.scenario import Scenario
from core.simulation_engine import SimulationEngine


@pytest.fixture
def baseline_scenario():
    return Scenario.from_yaml(
        Path(__file__).parent.parent.parent / "config/scenarios/hedge_fund_baseline.yaml"
    )


@pytest.fixture
def engine(baseline_scenario):
    return SimulationEngine(baseline_scenario, n_simulations=500, seed=42)


class TestSimulationEngineIntegration:
    def test_engine_runs_without_error(self, engine):
        result = engine.run()
        assert result is not None

    def test_all_scores_in_unit_interval(self, engine):
        result = engine.run()
        assert 0.0 <= result.quantum_risk_score <= 1.0
        assert 0.0 <= result.adversarial_threat_score <= 1.0
        assert 0.0 <= result.threat_likelihood <= 1.0

    def test_cyber_var_is_negative(self, engine):
        result = engine.run()
        assert result.cyber_var < 0

    def test_cyber_cvar_less_than_or_equal_var(self, engine):
        result = engine.run()
        assert result.cyber_cvar <= result.cyber_var

    def test_sharpe_impact_is_nonpositive(self, engine):
        result = engine.run()
        assert result.sharpe_impact <= 0.0

    def test_elapsed_seconds_positive(self, engine):
        result = engine.run()
        assert result.elapsed_seconds > 0

    def test_n_simulations_matches_requested(self, engine):
        result = engine.run()
        assert result.n_simulations == 500

    def test_scenario_name_in_result(self, engine, baseline_scenario):
        result = engine.run()
        assert result.scenario_name == baseline_scenario.name

    def test_summary_string_non_empty(self, engine):
        result = engine.run()
        summary = result.summary()
        assert len(summary) > 0
        assert "QuantumRiskEngine" in summary

    def test_well_defended_lower_risk_than_high_quantum(self):
        """Well-defended scenario should produce lower overall risk metrics."""
        base = Path(__file__).parent.parent.parent / "config/scenarios"
        defended = Scenario.from_yaml(base / "well_defended.yaml")
        aggressive = Scenario.from_yaml(base / "high_quantum_risk.yaml")

        r_defended = SimulationEngine(defended, n_simulations=500, seed=42).run()
        r_aggressive = SimulationEngine(aggressive, n_simulations=500, seed=42).run()

        assert r_defended.quantum_risk_score <= r_aggressive.quantum_risk_score

    def test_reproducibility_across_runs(self, baseline_scenario):
        r1 = SimulationEngine(baseline_scenario, n_simulations=200, seed=99).run()
        r2 = SimulationEngine(baseline_scenario, n_simulations=200, seed=99).run()
        assert r1.quantum_risk_score == r2.quantum_risk_score
        assert r1.cyber_var == r2.cyber_var
