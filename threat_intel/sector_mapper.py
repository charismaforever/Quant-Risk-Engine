"""
Sector Mapper — Industry vertical risk mapping and threat actor profiles.

Maps MITRE ATT&CK threat actor groups to sector-specific attack affinities
based on historical incident data and published threat intelligence reports.

Data sources:
- MITRE ATT&CK v14.1 group profiles
- Verizon DBIR 2023 industry breakdowns
- CrowdStrike Adversary Intelligence reports
- FS-ISAC Annual Threat Reports
"""

from __future__ import annotations

# Annual breach probability base rates by sector (Verizon DBIR 2023 proxies)
SECTOR_BASE_RATES: dict[str, float] = {
    "financial_services": 0.18,
    "healthcare": 0.22,
    "technology": 0.15,
    "energy": 0.14,
    "manufacturing": 0.16,
    "retail": 0.12,
    "government": 0.20,
    "education": 0.13,
    "telecommunications": 0.17,
}

# Threat actor affinity scores by sector [0, 1]
# Based on MITRE ATT&CK group targeting history
THREAT_ACTOR_PROFILES: dict[str, dict[str, float]] = {
    "APT29": {  # Cozy Bear — Russian SVR, espionage focus
        "financial_services": 0.55,
        "healthcare": 0.30,
        "technology": 0.70,
        "energy": 0.45,
        "government": 0.90,
        "telecommunications": 0.50,
    },
    "APT28": {  # Fancy Bear — Russian GRU, disruptive
        "financial_services": 0.45,
        "government": 0.85,
        "technology": 0.55,
        "energy": 0.60,
        "healthcare": 0.25,
        "telecommunications": 0.65,
    },
    "FIN7": {  # Financially motivated, POS/banking focus
        "financial_services": 0.88,
        "retail": 0.75,
        "healthcare": 0.40,
        "technology": 0.35,
        "government": 0.15,
        "energy": 0.20,
    },
    "FIN8": {  # POS intrusions, banking trojans
        "financial_services": 0.82,
        "retail": 0.70,
        "healthcare": 0.30,
        "technology": 0.25,
        "government": 0.10,
        "energy": 0.15,
    },
    "Lazarus": {  # North Korean DPRK — crypto theft, SWIFT attacks
        "financial_services": 0.92,
        "technology": 0.60,
        "energy": 0.35,
        "healthcare": 0.20,
        "government": 0.50,
        "telecommunications": 0.40,
    },
    "APT41": {  # Chinese MSS — espionage + financial crime
        "financial_services": 0.65,
        "healthcare": 0.70,
        "technology": 0.85,
        "telecommunications": 0.75,
        "energy": 0.55,
        "government": 0.60,
    },
    "Scattered Spider": {  # Social engineering, MFA bypass
        "financial_services": 0.75,
        "technology": 0.80,
        "telecommunications": 0.85,
        "retail": 0.60,
        "healthcare": 0.40,
        "government": 0.35,
    },
    "BlackCat": {  # ALPHV ransomware group
        "financial_services": 0.60,
        "healthcare": 0.75,
        "technology": 0.65,
        "energy": 0.70,
        "manufacturing": 0.80,
        "retail": 0.55,
    },
}

# Top TTPs by sector from MITRE ATT&CK frequency analysis
SECTOR_TOP_TTPS: dict[str, list[str]] = {
    "financial_services": [
        "T1566 - Phishing",
        "T1190 - Exploit Public-Facing Application",
        "T1078 - Valid Accounts",
        "T1059 - Command and Scripting Interpreter",
        "T1486 - Data Encrypted for Impact",
        "T1041 - Exfiltration Over C2 Channel",
        "T1110 - Brute Force",
    ],
    "healthcare": [
        "T1566 - Phishing",
        "T1486 - Data Encrypted for Impact",
        "T1078 - Valid Accounts",
        "T1133 - External Remote Services",
        "T1021 - Remote Services",
    ],
    "technology": [
        "T1190 - Exploit Public-Facing Application",
        "T1078 - Valid Accounts",
        "T1195 - Supply Chain Compromise",
        "T1566 - Phishing",
        "T1059 - Command and Scripting Interpreter",
    ],
}


def get_sector_risk_multiplier(sector: str, region: str = "north_america") -> float:
    """
    Return a risk multiplier based on sector and geographic region.

    North America and Western Europe financial services face higher threat
    actor interest due to SWIFT access and larger asset values.
    """
    region_multipliers = {
        "north_america": 1.20,
        "western_europe": 1.10,
        "apac": 1.05,
        "middle_east": 0.95,
        "latam": 0.90,
    }
    base = SECTOR_BASE_RATES.get(sector, 0.12)
    region_mult = region_multipliers.get(region, 1.0)
    return float(base * region_mult)
