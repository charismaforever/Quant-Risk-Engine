"""
Reporter — output formatting and export utilities.

Converts SimulationResult objects to JSON, CSV, and rich terminal output.
"""

from __future__ import annotations

import json
import csv
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simulation_engine import SimulationResult


class Reporter:
    """Formats and exports simulation results."""

    def __init__(self, result: "SimulationResult") -> None:
        self.result = result

    def to_dict(self) -> dict:
        return asdict(self.result)

    def to_json(self, path: str | Path | None = None) -> str:
        payload = json.dumps(self.to_dict(), indent=2)
        if path:
            Path(path).write_text(payload)
        return payload

    def to_csv(self, path: str | Path) -> None:
        data = self.to_dict()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            writer.writeheader()
            writer.writerow(data)

    def print_rich(self) -> None:
        """Print a rich terminal table if `rich` is available, else plain text."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            table = Table(title=f"QuantumRiskEngine — {self.result.scenario_name}", show_lines=True)
            table.add_column("Metric", style="bold cyan")
            table.add_column("Value", style="white")

            table.add_row("Quantum Risk Score", f"{self.result.quantum_risk_score:.3f}")
            table.add_row("Adversarial Threat Score", f"{self.result.adversarial_threat_score:.3f}")
            table.add_row("Threat Likelihood", f"{self.result.threat_likelihood:.3f}")
            table.add_row("Cyber-adjusted VaR", f"{self.result.cyber_var:+.2%}")
            table.add_row("Cyber-adjusted CVaR", f"{self.result.cyber_cvar:+.2%}")
            table.add_row("Sharpe Impact", f"{self.result.sharpe_impact:+.4f}")
            table.add_row("Simulations", f"{self.result.n_simulations:,}")
            table.add_row("Runtime (s)", f"{self.result.elapsed_seconds:.3f}")

            console.print(table)
        except ImportError:
            print(self.result.summary())
