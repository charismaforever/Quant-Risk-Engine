"""Unit tests for financial_risk.risk_metrics module."""

import numpy as np
import pytest

from financial_risk.risk_metrics import RiskMetrics


@pytest.fixture
def sample_returns():
    rng = np.random.default_rng(42)
    return rng.normal(loc=0.07, scale=0.15, size=10_000)


@pytest.fixture
def metrics(sample_returns):
    return RiskMetrics(sample_returns, confidence_level=0.99)


class TestRiskMetrics:
    def test_var_is_negative(self, metrics):
        assert metrics.var() < 0

    def test_cvar_less_than_or_equal_var(self, metrics):
        # CVaR (expected shortfall) must be ≤ VaR (more extreme tail)
        assert metrics.cvar() <= metrics.var()

    def test_sharpe_ratio_positive_for_positive_excess_return(self, sample_returns):
        # With mu=7% > rf=5%, Sharpe should be positive
        m = RiskMetrics(sample_returns)
        assert m.sharpe_ratio() > 0

    def test_full_report_var95_less_negative_than_var99(self, metrics):
        report = metrics.full_report()
        # 95% VaR is less extreme (smaller absolute loss) than 99% VaR
        assert report.var_95 >= report.var_99

    def test_full_report_cvar99_less_than_var99(self, metrics):
        report = metrics.full_report()
        assert report.cvar_99 <= report.var_99

    def test_probability_of_loss_in_unit_interval(self, metrics):
        report = metrics.full_report()
        assert 0.0 <= report.probability_of_loss <= 1.0

    def test_volatility_positive(self, metrics):
        report = metrics.full_report()
        assert report.volatility > 0

    def test_var_at_extreme_confidence_more_negative(self, sample_returns):
        m_99 = RiskMetrics(sample_returns, confidence_level=0.99)
        m_95 = RiskMetrics(sample_returns, confidence_level=0.95)
        assert m_99.var() <= m_95.var()
