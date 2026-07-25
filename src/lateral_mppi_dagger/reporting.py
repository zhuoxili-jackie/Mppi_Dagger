from __future__ import annotations

from pathlib import Path


def failure_report_path(report_path: str | Path) -> Path:
    """Derive a sibling failure report without changing the success target."""

    report = Path(report_path)
    return report.with_suffix(report.suffix + ".failure.json")
