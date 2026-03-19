# QuantumRiskEngine Documentation

**A multi-domain simulation framework for cyber-adjusted portfolio risk.**

QuantumRiskEngine bridges post-quantum cryptography vulnerability modeling,
adversarial AI threat simulation, statistical threat intelligence, and
financial portfolio risk quantification into a unified Monte Carlo pipeline.

---

## Modules

### [quantum_crypto](quantum_crypto.md)
Post-quantum cryptography vulnerability simulation. Models NIST PQC
migration timelines, HNDL attack windows, and quantum risk scores.

### [ai_adversarial](ai_adversarial.md)
Adversarial ML threat modeling for quantitative trading systems.
Covers evasion, poisoning, and model inversion attacks.

### [threat_intel](threat_intel.md)
Bayesian statistical TTP engine using MITRE ATT&CK frequency data.
Produces sector-specific posterior attack likelihood scores.

### [financial_risk](financial_risk.md)
Cyber-adjusted portfolio risk quantification. CyberVaR, CyberCVaR,
Monte Carlo simulation, and event-driven stress testing.

### [core](core.md)
Simulation orchestration engine. Composes all four modules into a
unified scenario pipeline with configurable parameters.

---

## Quickstart

```bash
pip install -e ".[dev]"
python -m core.simulation_engine --config config/scenarios/hedge_fund_baseline.yaml
```

## Research

See [PAPER.md](../PAPER.md) for the full mathematical methodology,
including derivations of CyberVaR, the Mosca-theorem quantum risk model,
and the Beta-Binomial Bayesian threat intelligence framework.

