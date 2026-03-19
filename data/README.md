 Data Directory

## Structure

```
data/
├── raw/          # Unprocessed threat feeds, market data snapshots
└── processed/    # Cleaned, normalized datasets ready for module consumption
```

## Data Sources

| Source | Type | Update Frequency | Module |
|---|---|---|---|
| MITRE ATT&CK STIX | Threat actor TTPs | Quarterly | threat_intel |
| Verizon DBIR | Sector breach rates | Annual | threat_intel |
| NIST NVD | CVE severity scores | Daily | threat_intel |
| FRED (St. Louis Fed) | Risk-free rates | Daily | financial_risk |
| Yahoo Finance | Market return series | Daily | financial_risk |
| CISA KEV | Exploited vulnerabilities | Weekly | threat_intel |

## Populating Data

```bash
# Download MITRE ATT&CK enterprise data
python scripts/fetch_mitre_attack.py

# Fetch market return series (requires internet)
python scripts/fetch_market_data.py --start 2015-01-01
```

Raw data files are excluded from version control (see `.gitignore`).
Sample synthetic datasets for testing are in `data/processed/sample/`.
