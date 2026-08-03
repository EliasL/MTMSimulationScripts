"""Generate participation plots after the exhaustive timing run completes."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


TIMING_STATUS = Path(
    "Plots/powerLaw/truncated_powerlaw_flowchart/full_global_min_timing.json"
)
PLOT_STATUS = Path(
    "Plots/numerical_parameter_justification/participation_plot_status.json"
)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(status: str, **extra) -> None:
    PLOT_STATUS.parent.mkdir(parents=True, exist_ok=True)
    PLOT_STATUS.write_text(
        json.dumps({"status": status, "updated_at": timestamp(), **extra}, indent=2)
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    write_status("waiting_for_global_min_timing")
    while True:
        if TIMING_STATUS.exists():
            timing = json.loads(TIMING_STATUS.read_text(encoding="utf-8"))
            if timing.get("status") == "complete":
                break
            if timing.get("status") == "failed":
                raise RuntimeError("The exhaustive timing run failed.")
        time.sleep(30)

    write_status("generating_participation_plots")
    from Plotting.numericalParameterJustification import generate_participation_plots

    generate_participation_plots()
    write_status("complete")


if __name__ == "__main__":
    main()
