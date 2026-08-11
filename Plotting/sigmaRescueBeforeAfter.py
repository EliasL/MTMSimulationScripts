"""Plot original versus validated sigma12 values for completed rescue tasks."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

from Management.sigmaRescue import _read_macro_rows


ROOT = Path("/private/tmp/sigma_rescue_before_after")
OUTPUT = Path("output/sigma_rescue_before_after_avg_sigma12.png")


def read_rows(path: Path) -> dict[int, tuple[float, float]]:
    _, rows = _read_macro_rows(path)
    result = {}
    for row in rows:
        step = int(row["load_step"])
        result[step] = (float(row["load"]), float(row["avg_sigma12"]))
    return result


def read_validated(path: Path) -> dict[int, tuple[float, float]]:
    with path.open(newline="") as stream:
        rows = csv.DictReader(stream)
        required = {"load_step", "load", "avg_sigma12"}
        if not required <= set(rows.fieldnames or ()):
            raise ValueError(f"Missing columns in {path}: {required}")
        return {
            int(row["load_step"]): (float(row["load"]), float(row["avg_sigma12"]))
            for row in rows
        }


CASES = {
    "L=150, seed=0": [
        ROOT / "L150/seg_0.21438_0.22466_validated_sigma.csv",
        ROOT / "L150/seg_0.22466_0.2347_validated_sigma.csv",
        ROOT / "L150/seg_0.2347_0.24431_validated_sigma.csv",
    ],
    "L=250, seed=0": [ROOT / "L250/prefix_validated_sigma.csv"],
}


def main() -> None:
    original = {
        "L=150, seed=0": read_rows(ROOT / "L150/original_macroData.csv"),
        "L=250, seed=0": read_rows(ROOT / "L250/original_macroData.csv"),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=False, constrained_layout=True)
    colors = ("tab:blue", "tab:orange", "tab:green")
    for axis, (label, validated_paths) in zip(axes, CASES.items()):
        before = original[label]
        before_steps = sorted(before)
        axis.plot(
            [before[step][0] for step in before_steps],
            [before[step][1] for step in before_steps],
            color="0.55",
            linewidth=0.8,
            label="original CSV (old sigma schema)",
            zorder=1,
        )
        invalid_total = 0
        for color, validated_path in zip(colors, validated_paths):
            validated = read_validated(validated_path)
            steps = sorted(validated)
            loads = [validated[step][0] for step in steps]
            values = [validated[step][1] for step in steps]
            invalid = [value == -1.0 for value in values]
            invalid_total += sum(invalid)
            axis.plot(
                loads,
                values,
                color=color,
                linewidth=1.4,
                label=f"validated: {validated_path.stem.replace('_validated_sigma', '')}",
                zorder=3,
            )
            if any(invalid):
                axis.scatter(
                    [load for load, bad in zip(loads, invalid) if bad],
                    [-1.0 for bad in invalid if bad],
                    color="crimson",
                    marker="x",
                    s=28,
                    label="invalid energy match (sigma = -1)" if invalid_total == sum(invalid) else None,
                    zorder=5,
                )
        axis.axhline(0.0, color="0.75", linewidth=0.7, zorder=0)
        axis.set_ylabel(r"$\langle\sigma_{12}\rangle$")
        axis.set_title(label)
        axis.grid(alpha=0.2)
        if invalid_total:
            axis.text(
                0.99,
                0.03,
                f"{invalid_total} row(s) marked invalid",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color="crimson",
            )
        axis.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("load / strain")
    fig.suptitle("Sigma-rescue comparison: original versus validated $\u03c3_{12}$", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
