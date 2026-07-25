from __future__ import annotations

from pathlib import Path

from lateral_mppi_dagger.reporting import failure_report_path


def test_failure_report_is_derived_from_selected_success_report() -> None:
    assert failure_report_path(
        Path("reports/42_low_load_mppi_smoke.json")
    ) == Path("reports/42_low_load_mppi_smoke.json.failure.json")
    assert failure_report_path(
        Path("reports/low_load_smoke")
    ) == Path("reports/low_load_smoke.failure.json")
