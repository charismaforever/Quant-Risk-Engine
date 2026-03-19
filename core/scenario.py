"""
Scenario definition and loader for QuantumRiskEngine.

A Scenario is a fully parameterized description of a simulation run,
loaded from YAML configuration or constructed programmatically.
"""

from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QuantumCryptoParams:
    """Parameters for PQC vulnerability simulation."""
    encryption_standard: str = "RSA-2048"         # RSA-2048, ECC-256, AES-128, AES-256
    migration_target: str = "CRYSTALS-Kyber"       # Target NIST PQC algorithm
    years_to_cryptographically_relevant_qc: int = 7
    migration_budget_usd: float = 2_000_000.0
    hndl_data_sensitivity: str = "high"            # low, medium, high, critical
    simulation_horizon_years: int = 10


@dataclass
class AdversarialAIParams:
    """Parameters for adversarial ML attack modeling."""
    attack_types: list[str] = field(
        default_factory=lambda: ["evasion", "poisoning", "model_inversion"]
    )
    model_architecture: str = "gradient_boosting"  # gradient_boosting, neural_network, linear
    training_data_exposure: float = 0.15           # Fraction accessible to adversary
    defense_mechanisms: list[str] = field(
        default_factory=lambda: ["adversarial_training", "input_validation"]
    )
    alpha_signal_value_usd: float = 50_000_000.0


@dataclass
class ThreatIntelParams:
    """Parameters for statistical threat intelligence engine."""
    sector: str = "financial_services"             # financial_services, healthcare, tech, energy
    threat_actors: list[str] = field(
        default_factory=lambda: ["APT29", "FIN7", "Lazarus"]
    )
    prior_breach_probability: float = 0.12         # Annual base rate for sector
    mitre_framework_version: str = "14.1"
    geographic_region: str = "north_america"


@dataclass
class FinancialRiskParams:
    """Parameters for portfolio risk quantification."""
    portfolio_value_usd: float = 500_000_000.0
    confidence_level: float = 0.99                 # VaR/CVaR confidence level
    time_horizon_days: int = 252                   # 1 trading year
    n_simulations: int = 10_000
    cyber_shock_distribution: str = "pareto"       # pareto, gev, lognormal
    correlation_with_market: float = 0.25          # Cyber event market correlation
    asset_classes: list[str] = field(
        default_factory=lambda: ["equities", "fixed_income", "alternatives"]
    )


@dataclass
class Scenario:
    """
    Complete scenario specification for a QuantumRiskEngine simulation run.

    Example
    -------
    >>> scenario = Scenario.from_yaml("config/scenarios/hedge_fund_baseline.yaml")
    >>> scenario.name
    'hedge_fund_baseline'
    """

    name: str = "default_scenario"
    description: str = ""
    quantum_crypto: QuantumCryptoParams = field(default_factory=QuantumCryptoParams)
    adversarial_ai: AdversarialAIParams = field(default_factory=AdversarialAIParams)
    threat_intel: ThreatIntelParams = field(default_factory=ThreatIntelParams)
    financial_risk: FinancialRiskParams = field(default_factory=FinancialRiskParams)
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Scenario":
        """Load a Scenario from a YAML configuration file."""
        with open(path, "r") as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        return cls(
            name=raw.get("name", "unnamed"),
            description=raw.get("description", ""),
            quantum_crypto=QuantumCryptoParams(**raw.get("quantum_crypto", {})),
            adversarial_ai=AdversarialAIParams(**raw.get("adversarial_ai", {})),
            threat_intel=ThreatIntelParams(**raw.get("threat_intel", {})),
            financial_risk=FinancialRiskParams(**raw.get("financial_risk", {})),
            random_seed=raw.get("random_seed", 42),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize scenario to YAML."""
        import dataclasses
        with open(path, "w") as f:
            yaml.dump(dataclasses.asdict(self), f, default_flow_style=False)
