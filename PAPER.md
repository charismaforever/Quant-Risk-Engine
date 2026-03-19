# QuantumRiskEngine: A Multi-Domain Simulation Framework for Cyber-Adjusted Portfolio Risk

**Catrina Turner**
Principal & CEO, Imminent Flair | Quantitative Security Research
`catrina@imminentflair.com`

---

## Abstract

We present QuantumRiskEngine, a simulation and modeling framework that integrates
post-quantum cryptography vulnerability timelines, adversarial machine learning
threat modeling, statistical threat intelligence inference, and financial portfolio
risk quantification into a unified Monte Carlo pipeline. The framework produces
**cyber-adjusted Value at Risk (CyberVaR)** and **Conditional VaR (CyberCVaR)**
metrics that incorporate cyber threat probability distributions as explicit stochastic
inputs, rather than treating cyber events as binary stress scenarios.

Our key contributions are: (1) a Mosca-theorem-derived quantum risk scoring model
with log-normal CRQC timeline uncertainty; (2) a Beta-Binomial Bayesian updater
for MITRE ATT&CK-based threat likelihood estimation; (3) a compound Poisson jump
process extension to GBM for cyber event integration; and (4) a finance-specific
adversarial ML attack surface model for quantitative trading systems.

---

## 1. Motivation

The standard approach to cyber risk in quantitative finance is scenario-based:
analysts define discrete events ("ransomware hits 10% of portfolio value") and
stress-test against them. This approach has three limitations:

1. **Discontinuity**: Treats cyber risk as a scenario dial rather than a continuous
   stochastic process integrated with market dynamics.
2. **Independence assumption**: Ignores the correlation between cyber incidents and
   market stress periods (adversaries are more active during crises; breaches cause
   market impact).
3. **Quantum blindspot**: Existing frameworks have no mechanism to model the
   approaching post-quantum cryptography transition and its portfolio implications.

QuantumRiskEngine addresses all three by treating cyber risk as a set of
parameterized stochastic processes that interact with baseline GBM portfolio dynamics.

---

## 2. Quantum Risk Model

### 2.1 Mosca's Theorem Framework

Let:
- `t_q` = years until a cryptographically-relevant quantum computer (CRQC) exists
- `t_m` = years remaining to complete PQC migration
- `t_s` = data shelf life (years the targeted data remains sensitive)

Mosca's inequality states: if `t_m + t_s > t_q`, the organization has a quantum
risk problem **today** (harvest-now-decrypt-later attacks make current encryption
vulnerable to future decryption).

### 2.2 CRQC Timeline Uncertainty

Expert forecasts of quantum computing timelines exhibit high variance. We model
CRQC arrival as log-normal:

```
T_q ~ LogNormal(μ = log(t̂_q), σ = 0.4)
```

where `t̂_q` is the scenario's point estimate and σ = 0.4 captures the ~40%
coefficient of variation observed across expert survey distributions (Mosca 2023,
IBM Quantum roadmap confidence intervals).

### 2.3 Quantum Risk Score

The scalar QuantumRiskScore integrates CRQC arrival uncertainty, encryption
half-life decay, and HNDL exposure:

```
QRS = max_year P(breach | year) × (1 - 0.7 × R_m)
```

where `P(breach | year)` combines exponential security decay and CRQC arrival
probability, and `R_m ∈ [0,1]` is the migration readiness score.

---

## 3. Adversarial ML Threat Model

### 3.1 Attack Surface

We model three attack vectors against quantitative trading ML systems:

**Evasion (inference-time):** Adversary perturbs market signal inputs to produce
incorrect predictions. Attack success probability is parameterized by model
architecture vulnerability `v_arch` and training data exposure `ε`:

```
P(evasion success) = Beta(10(v_arch + 0.15ε), 10(1 - v_arch - 0.15ε)) × (1 - D)
```

where D is the defense discount (non-additive stacking of deployed defenses).

**Data poisoning (training-time):** Success probability scales with data exposure:

```
P(poisoning) = 1 - exp(-3ε) × (1 - 0.7D)
```

**Model inversion:** Reconstruction of proprietary model parameters from API outputs.
Finance-specific concern: alpha signal features can be reverse-engineered from
prediction confidence distributions.

### 3.2 Sharpe Ratio Impact

Adversarial ML attacks manifest as alpha degradation. We model the Sharpe
ratio impact as:

```
ΔSharpe = -(0.25 × P_poison + 0.10 × P_evasion)
```

This is incorporated as a persistent drag in the Monte Carlo return process.

---

## 4. Bayesian Threat Intelligence Model

### 4.1 Beta-Binomial Conjugate Model

We model attack probability for each (threat actor, sector) pair as a
Beta-Binomial conjugate model:

```
Prior:      p ~ Beta(α, β)
Likelihood: k | n, p ~ Binomial(n, p)
Posterior:  p | k ~ Beta(α + k, β + n - k)
```

Priors `(α, β)` are calibrated from MITRE ATT&CK group targeting frequencies
and Verizon DBIR sector breach rates.

### 4.2 Combined Sector Likelihood

Given threat actors `{1, ..., m}` with posterior means `{p̂_1, ..., p̂_m}`,
the combined likelihood that at least one actor succeeds:

```
P(breach) = 1 - ∏(1 - p̂_i)
```

The final threat likelihood blends actor-specific posteriors with the
sector base rate:

```
L_final = 0.6 × P(breach) + 0.4 × λ_sector
```

---

## 5. Cyber-Adjusted Portfolio Return Model

### 5.1 Return Process

The cyber-adjusted annual portfolio return is:

```
R_cyber = R_GBM + S_quantum + D_ML + S_threat
```

Where:
- `R_GBM ~ LogNormal(μ - σ²/2, σ)` — baseline geometric Brownian motion
- `S_quantum ~ Poisson(QRS) × Normal(μ_q, σ_q)` — quantum breach jump process
- `D_ML ~ Normal(-0.03 × ATS, 0.01)` — adversarial ML alpha drag
- `S_threat ~ Bernoulli(L) × Pareto(α, x_m)` — threat event compound Poisson

### 5.2 Cyber-Adjusted VaR

Standard VaR at confidence level `1-α`:

```
CyberVaR_α = -inf{x : P(R_cyber ≤ x) > α}
```

Cyber-adjusted CVaR (Expected Shortfall):

```
CyberCVaR_α = E[R_cyber | R_cyber ≤ CyberVaR_α]
```

The uplift `CyberVaR - BaselineVaR` quantifies the incremental capital
requirement attributable to cyber risk under the Basel III internal models approach.

---

## 6. Empirical Validation

The model's threat likelihood and financial impact parameters are calibrated against:

- Verizon Data Breach Investigations Report (2019–2023): sector breach rates
- IBM Cost of a Data Breach Report (2023): loss severity distributions  
- CrowdStrike Adversary Intelligence: threat actor targeting frequencies
- MITRE ATT&CK v14.1: TTP occurrence frequencies by group
- Historical cyber incident market impact studies (Kamiya et al. 2021,
  Tosun 2021): event-driven return impact estimates

---

## 7. Limitations and Future Work

1. **Independence assumption**: The three cyber shock processes are currently
   modeled as independent. Future work should incorporate a copula structure
   (e.g., Clayton copula) to model tail dependence between quantum events
   and threat intelligence-driven incidents.

2. **Static threat actor profiles**: THREAT_ACTOR_PROFILES are calibrated on
   historical data. Live integration with threat intelligence feeds (MISP,
   OpenCTI) would enable dynamic Bayesian updating as new incidents occur.

3. **Quantum timeline uncertainty**: The log-normal parameterization could be
   replaced with a mixture model combining multiple expert forecast distributions.

4. **Adversarial ML attack correlations**: In practice, evasion and poisoning
   attacks may be correlated (same adversary campaign). A joint attack model
   would improve accuracy.

---

## References

- Mosca, M. (2023). *Cybersecurity in an Era with Quantum Computers: Will We Be Ready?*
- NIST SP 800-208. *Recommendation for Stateful Hash-Based Signature Schemes.*
- Goodfellow, I. et al. (2015). *Explaining and Harnessing Adversarial Examples.*
- Biggio, B. et al. (2012). *Poisoning Attacks Against Support Vector Machines.*
- Fredrikson, M. et al. (2015). *Model Inversion Attacks That Exploit Confidence Information.*
- Kamiya, S. et al. (2021). *Risk or Opportunity? Information in Cybersecurity Incidents.*
- Verizon (2023). *Data Breach Investigations Report.*
- IBM Security (2023). *Cost of a Data Breach Report.*
- MITRE ATT&CK v14.1. *Adversarial Tactics, Techniques, and Common Knowledge.*
