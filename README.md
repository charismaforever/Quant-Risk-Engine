# QuantumRiskEngine

> **A multi-domain simulation and modeling framework for next-generation risk.**
> Bridges post-quantum cryptography vulnerability modeling, adversarial AI threat simulation,
> statistical threat intelligence, and financial portfolio risk quantification.

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue.svg)](docs/)

---

## Overview

QuantumRiskEngine is a research-grade simulation platform designed for quantitative researchers
and security-risk practitioners who need to model the intersection of cybersecurity threats and
financial portfolio impact. The engine is composed of four domain modules, each independently
usable but architecturally designed to feed into a unified Monte Carlo simulation pipeline.

### Why This Matters for Finance

Traditional quantitative risk frameworks (VaR, CVaR, stress testing) treat cyber events as
binary shocks — on or off. QuantumRiskEngine models **attack probability distributions**,
**quantum migration timelines**, and **adversarial ML attack vectors** as continuous stochastic
processes that can be incorporated directly into portfolio risk models.

---

## Architecture

```
QuantumRiskEngine/
├── core/                        # Simulation orchestration engine
│   ├── simulation_engine.py     # Monte Carlo + scenario pipeline
│   ├── scenario.py              # Scenario dataclass definitions
│   └── reporter.py              # Output formatting utilities
│
├── quantum_crypto/              # Post-quantum cryptography vulnerability modeling
│   ├── pqc_vulnerability.py     # NIST PQC timeline simulations
│   ├── harvest_now_decrypt.py   # HNDL attack window estimation
│   └── migration_scheduler.py  # Crypto-agility migration cost models
│
├── ai_adversarial/              # Adversarial ML attack modeling
│   ├── attack_simulator.py      # Evasion, poisoning, model inversion
│   ├── trading_algo_threat.py   # Finance-specific ML threat surface
│   └── defense_analyzer.py     # Defense effectiveness scoring
│
├── threat_intel/                # Statistical threat intelligence engine
│   ├── ttp_engine.py            # MITRE ATT&CK frequency modeling
│   ├── bayesian_updater.py      # Posterior attack likelihood scoring
│   └── sector_mapper.py        # Industry vertical risk mapping
│
├── financial_risk/              # Portfolio risk quantification
│   ├── cyber_var.py             # Cyber-adjusted Value at Risk
│   ├── monte_carlo.py           # Correlated scenario simulation
│   ├── portfolio_stress.py      # Event-driven stress testing
│   └── risk_metrics.py         # CVaR, ES, Sharpe-adjusted metrics
│
├── notebooks/                   # Jupyter research notebooks
├── data/                        # Threat feeds and market data
├── tests/                       # pytest unit + integration tests
├── config/                      # YAML scenario configurations
└── docs/                        # MkDocs methodology documentation
```

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/charismaforever/QuantumRiskEngine.git
cd QuantumRiskEngine
pip install -e ".[dev]"

# Run a pre-built scenario
python -m core.simulation_engine --config config/scenarios/hedge_fund_baseline.yaml

# Launch the interactive dashboard
streamlit run dashboard/app.py

# Run all tests
pytest tests/ -v
```

---

## Module Summaries

### `quantum_crypto` — PQC Vulnerability Simulation

Models the cryptographic vulnerability surface of a portfolio company across quantum attack
timelines. Simulates "harvest now, decrypt later" (HNDL) risk windows and estimates time-to-breach
probability curves by asset class and encryption standard (RSA-2048, ECC-256, AES-128/256).
Integrates NIST PQC migration schedules (CRYSTALS-Kyber, Dilithium, SPHINCS+).

**Key output:** `QuantumRiskScore` — a time-indexed probability distribution of cryptographic
compromise across a migration timeline.

### `ai_adversarial` — Adversarial ML Threat Modeling

Models adversarial ML attacks targeting quantitative trading algorithms. Covers:
- **Evasion attacks** — perturbing market signals to fool inference-time models
- **Data poisoning** — corrupting training data to degrade alpha signal quality
- **Model inversion** — reconstructing proprietary model weights from outputs

Integrates with IBM Adversarial Robustness Toolbox (ART) for attack generation.

**Key output:** Attack success probability distributions and defense ROI scoring.

### `threat_intel` — Statistical TTP Engine

Ingests MITRE ATT&CK framework data and applies Bayesian inference to produce
sector-specific posterior attack likelihood scores. Maps threat actor TTPs to
industry verticals relevant to hedge fund portfolio companies (fintech, healthcare,
critical infrastructure, SaaS).

**Key output:** `ThreatLikelihoodMatrix` — Bayesian-updated probability scores by
sector, threat actor, and attack technique.

### `financial_risk` — Cyber-Adjusted Portfolio Risk

The integration layer. Incorporates outputs from all three upstream modules as
stochastic inputs to Monte Carlo portfolio simulations. Models:
- Cyber event shocks as fat-tailed distributions (Pareto, GEV)
- Cross-asset contagion from supply chain compromise events
- Cyber-adjusted VaR and CVaR with configurable confidence levels
- Sharpe ratio impact modeling under adversarial ML scenarios

**Key output:** Risk-adjusted return metrics blending financial and security risk.

---

## Example: Running a Scenario

```python
from core.simulation_engine import SimulationEngine
from core.scenario import Scenario

scenario = Scenario.from_yaml("config/scenarios/hedge_fund_baseline.yaml")
engine = SimulationEngine(scenario, n_simulations=10_000, seed=42)
results = engine.run()

print(results.summary())
# QuantumRiskScore:     0.34  (34% probability of PQC-relevant breach by 2027)
# AdversarialThreat:    0.61  (high ML attack surface for algo-trading stack)
# ThreatLikelihood:     0.48  (sector-weighted posterior, financial services)
# Cyber-adjusted VaR:  -8.2%  (99% confidence, 1-year horizon)
# Cyber-adjusted CVaR: -14.7% (expected shortfall beyond VaR threshold)
```

---

## Research Paper

See [`PAPER.md`](PAPER.md) for the full methodology, mathematical derivations,
and empirical validation against historical cyber incident data.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `scipy` | Numerical simulation, statistical distributions |
| `pandas` | Data manipulation, time-series handling |
| `qiskit` | Quantum circuit simulation for PQC modeling |
| `adversarial-robustness-toolbox` | Adversarial ML attack generation |
| `scikit-learn` | ML model wrappers and defense evaluation |
| `plotly`, `streamlit` | Interactive dashboard and visualization |
| `pyyaml` | Scenario configuration parsing |
| `pytest` | Unit and integration testing |

---

## Author

**Catrina Turner**
Principal & CEO, Imminent Flair | Cybersecurity × Quantitative Research
[imminent flair.com](https://imminentflair.com) · [GitHub: charismaforever](https://github.com/charismaforever)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
