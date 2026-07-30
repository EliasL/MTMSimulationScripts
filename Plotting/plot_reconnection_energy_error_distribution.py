"""Plot first- and second-order energy errors, including reconnection-conditioned steps.

Run from the repository root with::

    python -m Plotting.plot_reconnection_energy_error_distribution
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np


def _collect_values(
    csv_paths,
    event_only,
    no_event_only=False,
    pre_yield_min_gamma=None,
):
    from Management.updateCSV import read_macrodata_csv
    from Plotting.makePlots import compute_predicted_next_energy

    if event_only and no_event_only:
        raise ValueError("event_only and no_event_only are mutually exclusive.")
    if pre_yield_min_gamma is not None and (
        not np.isfinite(pre_yield_min_gamma) or pre_yield_min_gamma < 0.0
    ):
        raise ValueError("pre_yield_min_gamma must be None or nonnegative and finite.")

    values = {
        "first": {"pre": [], "post": []},
        "second": {"pre": [], "post": []},
    }
    for csv_path in csv_paths:
        prediction_df, prediction_info = compute_predicted_next_energy(csv_path)
        raw_df = read_macrodata_csv(csv_path)
        load = np.asarray(raw_df["load"], dtype=float)
        load_i = np.asarray(prediction_df["load_i"], dtype=float)
        step_energy_change = np.asarray(raw_df["total_energy_change"], dtype=float)[1:]
        drop_mask = step_energy_change < 0.0
        # Use the Cauchy shear-stress peak consistently with the energy
        # predictor. avg_P12 mixes material directions from independently
        # oriented element reference maps and is not conjugate to MTS2D's
        # left-multiplicative affine shear.
        stress = np.asarray(raw_df["avg_sigma12"], dtype=float)
        yield_load = float(load[int(np.nanargmax(stress))])

        if event_only or no_event_only:
            event_mask = np.asarray(raw_df["nr_total_edge_flips"], dtype=float)[1:] > 0
        for approximation, error_metric in (
            ("first", "prediction_error"),
            ("second", "second_order_prediction_error"),
        ):
            residual = np.abs(np.asarray(prediction_df[error_metric], dtype=float))
            residual /= float(prediction_info["reference_volume"])
            base_mask = np.isfinite(residual) & (residual > 0)
            if event_only:
                base_mask &= event_mask & (step_energy_change > 0.0)
            elif no_event_only:
                base_mask &= (~event_mask) & (step_energy_change >= 0.0)
            else:
                base_mask &= ~drop_mask

            for region in ("pre", "post"):
                region_mask = (
                    load_i < yield_load if region == "pre" else load_i >= yield_load
                )
                if region == "pre" and pre_yield_min_gamma is not None:
                    region_mask &= load_i >= pre_yield_min_gamma
                values[approximation][region].append(
                    residual[base_mask & region_mask]
                )

    return {
        approximation: {
            region: np.concatenate(chunks) if chunks else np.empty(0, dtype=float)
            for region, chunks in region_values.items()
        }
        for approximation, region_values in values.items()
    }


def _run_worker(
    csv_paths,
    event_only,
    output_path,
    no_event_only=False,
    pre_yield_min_gamma=None,
):
    values = _collect_values(
        csv_paths,
        event_only,
        no_event_only,
        pre_yield_min_gamma,
    )
    np.savez(
        output_path,
        first_pre=values["first"]["pre"],
        first_post=values["first"]["post"],
        second_pre=values["second"]["pre"],
        second_post=values["second"]["post"],
    )


def _format_count(count):
    exponent = int(np.floor(np.log10(count)))
    mantissa = float(count) / (10.0**exponent)
    return rf"$n={mantissa:.1f}\times 10^{{{exponent}}}$"


def _empty_values():
    return {
        "first": {"pre": np.empty(0, dtype=float), "post": np.empty(0, dtype=float)},
        "second": {"pre": np.empty(0, dtype=float), "post": np.empty(0, dtype=float)},
    }


def _merge_values(value_dicts):
    return {
        approximation: {
            region: (
                np.concatenate(
                    [
                        values[approximation][region]
                        for values in value_dicts
                        if values[approximation][region].size
                    ]
                )
                if any(values[approximation][region].size for values in value_dicts)
                else np.empty(0, dtype=float)
            )
            for region in ("pre", "post")
        }
        for approximation in ("first", "second")
    }


def _sample_id(csv_path):
    match = re.search(r"s(\d+)$", Path(csv_path).parent.name)
    return match.group(1) if match else "unknown"


def _system_size(csv_path):
    match = re.search(r",s(\d+)x\1(?:l|,)", Path(csv_path).parent.name)
    if match is None:
        raise ValueError(f"Could not infer the system size from {csv_path}.")
    return int(match.group(1))


def _validate_system_size(csv_paths, requested_size=None):
    sizes = {_system_size(path) for path in csv_paths}
    if len(sizes) != 1:
        raise ValueError(f"Expected one system size, found {sorted(sizes)}.")
    size = sizes.pop()
    if requested_size is not None and requested_size != size:
        raise ValueError(
            f"Requested L={requested_size}, but the CSV files contain L={size}."
        )
    return size


def _print_error_summary(values, pre_yield_min_gamma=None):
    """Print mean absolute errors for the same samples shown in the plot."""
    if pre_yield_min_gamma is None:
        filter_description = "same non-drop filters, no pre-yield gamma cutoff"
    else:
        filter_description = (
            "same non-drop filters, "
            f"pre-yield gamma >= {pre_yield_min_gamma:g}"
        )
    print(f"Mean absolute error per V0 ({filter_description}):")
    first_all = []
    second_all = []
    for condition, label in (("no", "No recon"), ("recon", "Recon")):
        for region in ("pre", "post"):
            first = values[condition]["first"][region]
            second = values[condition]["second"][region]
            if not first.size or not second.size:
                continue
            first_mean = float(np.mean(first))
            second_mean = float(np.mean(second))
            reduction = 100.0 * (first_mean - second_mean) / first_mean
            print(
                f"  {label}, {region}-yield: "
                f"1st={first_mean:.6e}, 2nd={second_mean:.6e}, "
                f"2nd-order reduction={reduction:+.3f}%"
            )
            first_all.append(first)
            second_all.append(second)
    if first_all:
        first_mean = float(np.mean(np.concatenate(first_all)))
        second_mean = float(np.mean(np.concatenate(second_all)))
        reduction = 100.0 * (first_mean - second_mean) / first_mean
        print(
            f"  All conditions: 1st={first_mean:.6e}, 2nd={second_mean:.6e}, "
            f"2nd-order reduction={reduction:+.3f}%"
        )


def _print_sample_summaries(condition, paths, individual_values):
    label = "No recon" if condition == "no" else "Recon"
    print(f"{label} sample means (second order, per V0):")
    for path in paths:
        sample = _sample_id(path)
        sample_values = individual_values[f"{condition}_s{sample}"]["second"]
        pre = sample_values["pre"]
        post = sample_values["post"]
        pre_text = "empty" if not pre.size else f"{np.mean(pre):.6e}"
        post_text = "empty" if not post.size else f"{np.mean(post):.6e}"
        print(
            f"  s{sample}: pre={pre_text} (n={pre.size}), "
            f"post={post_text} (n={post.size})"
        )


def _plot(
    values,
    output_path,
    approximation_order,
    show_reconnection_subsets=False,
):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    approximation = "first" if approximation_order == 1 else "second"
    base_specs = [
        ("No recon, pre-yield", "no", "pre", "#9ecae1"),
        ("No recon, post-yield", "no", "post", "#2171b5"),
        ("Recon, pre-yield", "recon", "pre", "#fdae6b"),
        ("Recon, post-yield", "recon", "post", "#e6550d"),
    ]
    entries = []
    for label, condition, region, color in base_specs:
        entries.append(
            (
                label,
                values[condition][approximation][region],
                color,
                "-",
            )
        )
    if show_reconnection_subsets:
        entries.extend(
            [
                (
                    "Flips + increase, pre-yield",
                    values["recon_events"][approximation]["pre"],
                    "#fdae6b",
                    "--",
                ),
                (
                    "Flips + increase, post-yield",
                    values["recon_events"][approximation]["post"],
                    "#e6550d",
                    "--",
                ),
                (
                    "Recon, no flips, pre-yield",
                    values["recon_no_events"][approximation]["pre"],
                    "#fdae6b",
                    ":",
                ),
                (
                    "Recon, no flips, post-yield",
                    values["recon_no_events"][approximation]["post"],
                    "#e6550d",
                    ":",
                ),
            ]
        )
    entries = [entry for entry in entries if entry[1].size]
    all_values = np.concatenate([entry[1] for entry in entries])
    edges = np.geomspace(float(np.min(all_values)), float(np.max(all_values)), 121)

    fig, ax = plt.subplots(figsize=(4.329, 2.808))
    plot_data = []
    for label, sample, color, linestyle in entries:
        histogram, _ = np.histogram(sample, bins=edges)
        probability = histogram.astype(float) / float(sample.size)
        plot_data.append((label, sample, probability, color, linestyle))
        ax.plot(
            np.repeat(edges, 2)[1:-1],
            np.repeat(probability, 2),
            color=color,
            linestyle=linestyle,
            linewidth=1.2 if linestyle == "-" else 1.0,
            label=f"{label} ({_format_count(sample.size)})",
        )

    ax.set_xscale("log")
    ax.set_xlabel(
        r"$|\Delta E_S|/V_0$"
    )
    ax.set_ylabel("Probability per logarithmic bin")
    ax.legend(loc="upper left", fontsize=8.0, handlelength=1.56, borderpad=0.26)
    fig.tight_layout()
    fig.savefig(output_path)
    print(f'Plot saved at: "{output_path}"')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--event-only", action="store_true")
    parser.add_argument("--csv-files", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--order", type=int, choices=(1, 2), default=2)
    parser.add_argument("--no-event-only", action="store_true")
    parser.add_argument("--show-reconnection-subsets", action="store_true")
    parser.add_argument("--pre-yield-min-gamma", type=float, default=None)
    parser.add_argument(
        "--no-reconnection-csv-files",
        nargs="+",
        type=Path,
        help="Non-reconnecting runs to pool in the main plot.",
    )
    parser.add_argument(
        "--reconnection-csv-files",
        nargs="+",
        type=Path,
        help="Reconnecting runs to pool in the main plot.",
    )
    parser.add_argument(
        "--system-size",
        type=int,
        help="Validate L and include it in the automatic output filename.",
    )
    parser.add_argument(
        "--generate-nonreconnecting-samples",
        action="store_true",
        help="Also generate one plot for each non-reconnecting sample.",
    )
    args = parser.parse_args()

    if args.worker:
        if not args.csv_files or args.output is None:
            raise ValueError("Worker mode requires --csv-files and --output.")
        _run_worker(
            args.csv_files,
            args.event_only,
            args.output,
            args.no_event_only,
            args.pre_yield_min_gamma,
        )
        return

    data_root = Path("Plots/energy_prediction_normal_data")
    gamma_tag = (
        "all_gamma"
        if args.pre_yield_min_gamma is None
        else f"gamma_ge_{args.pre_yield_min_gamma:g}".replace(".", "p")
    )
    if args.no_reconnection_csv_files is None:
        no_reconnection = sorted(
            data_root.glob(
                "simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsx1e-06s*/"
                "macroData.csv"
            )
        )
    else:
        no_reconnection = sorted(args.no_reconnection_csv_files)
    if args.reconnection_csv_files is None and args.no_reconnection_csv_files is None:
        reconnection = sorted(
            data_root.glob(
                "simpleShear,s100x100l0.15,1e-05,1.0PBCedgeFlipt3LBFGSEpsx1e-06s*/"
                "macroData.csv"
            )
        )
    else:
        reconnection = sorted(args.reconnection_csv_files or [])
    if not no_reconnection:
        raise FileNotFoundError(
            "No non-reconnecting macroData.csv files were found."
        )
    all_paths = [*no_reconnection, *reconnection]
    missing_paths = [path for path in all_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing CSV files: " + ", ".join(str(path) for path in missing_paths)
        )
    system_size = _validate_system_size(all_paths, args.system_size)
    if args.output is not None:
        output_path = args.output
    elif reconnection:
        output_path = Path(
            f"Plots/energy_error_cauchy_a_L{system_size}_"
            f"{len(no_reconnection)}samples_no_recon_"
            f"{len(reconnection)}samples_recon_{gamma_tag}_V0norm_120bins.pdf"
        )
    else:
        sample_word = "sample" if len(no_reconnection) == 1 else "samples"
        output_path = Path(
            f"Plots/energy_error_cauchy_a_L{system_size}_"
            f"{len(no_reconnection)}{sample_word}_no_recon_"
            f"{gamma_tag}_V0norm_120bins.pdf"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worker_env = os.environ.copy()
    worker_env["MPLBACKEND"] = "Agg"
    repo_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        worker_specs = {}
        conditions = [("no", no_reconnection, False, False)]
        if reconnection:
            conditions.append(("recon", reconnection, False, False))
        if args.show_reconnection_subsets:
            if not reconnection:
                raise ValueError(
                    "--show-reconnection-subsets requires reconnecting CSV files."
                )
            conditions.extend(
                [
                    ("recon_events", reconnection, True, False),
                    ("recon_no_events", reconnection, False, True),
                ]
            )
        for condition, paths, event_only, no_event_only in conditions:
            for path in paths:
                worker_specs[f"{condition}_s{_sample_id(path)}"] = (
                    [path],
                    event_only,
                    no_event_only,
                )
        processes = {}
        worker_outputs = {}
        for key, (paths, event_only, no_event_only) in worker_specs.items():
            worker_output = temp_dir / f"{key}.npz"
            worker_outputs[key] = worker_output
            command = [
                sys.executable,
                "-m",
                "Plotting.plot_reconnection_energy_error_distribution",
                "--worker",
                "--csv-files",
                *[str(path) for path in paths],
                "--output",
                str(worker_output),
            ]
            if event_only:
                command.append("--event-only")
            if no_event_only:
                command.append("--no-event-only")
            if args.pre_yield_min_gamma is not None:
                command.extend(
                    ["--pre-yield-min-gamma", str(args.pre_yield_min_gamma)]
                )
            processes[key] = subprocess.Popen(
                command,
                cwd=repo_root,
                env=worker_env,
            )
        for process in processes.values():
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)

        individual_values = {}
        for key, worker_output in worker_outputs.items():
            with np.load(worker_output) as data:
                individual_values[key] = {
                    "first": {
                        "pre": data["first_pre"].copy(),
                        "post": data["first_post"].copy(),
                    },
                    "second": {
                        "pre": data["second_pre"].copy(),
                        "post": data["second_post"].copy(),
                    },
                }

        values = {
            "no": _merge_values(
                [individual_values[f"no_s{_sample_id(path)}"] for path in no_reconnection]
            ),
            "recon": (
                _merge_values(
                    [
                        individual_values[f"recon_s{_sample_id(path)}"]
                        for path in reconnection
                    ]
                )
                if reconnection
                else _empty_values()
            ),
        }
        _print_sample_summaries("no", no_reconnection, individual_values)
        if reconnection:
            _print_sample_summaries("recon", reconnection, individual_values)
        if args.show_reconnection_subsets:
            values.update(
                {
                    "recon_events": _merge_values(
                        [
                            individual_values[f"recon_events_s{_sample_id(path)}"]
                            for path in reconnection
                        ]
                    ),
                    "recon_no_events": _merge_values(
                        [
                            individual_values[
                                f"recon_no_events_s{_sample_id(path)}"
                            ]
                            for path in reconnection
                        ]
                    ),
                }
            )
        _print_error_summary(values, args.pre_yield_min_gamma)
        _plot(
            values,
            output_path,
            args.order,
            show_reconnection_subsets=args.show_reconnection_subsets,
        )

        if not args.generate_nonreconnecting_samples:
            return

        for path in no_reconnection:
            sample = _sample_id(path)
            sample_output = Path(
                f"Plots/energy_error_cauchy_a_L{system_size}_no_recon_s{sample}_"
                f"{gamma_tag}_V0norm_120bins.pdf"
            )
            sample_values = {
                "no": individual_values[f"no_s{sample}"],
                "recon": _empty_values(),
            }
            _plot(
                sample_values,
                sample_output,
                args.order,
                show_reconnection_subsets=args.show_reconnection_subsets,
            )


if __name__ == "__main__":
    main()
