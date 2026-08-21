"""Compare evenly sampled reversible and irreversible energy drops.

The event classes are assigned from the default post-yield ``kappa_det``
threshold.  The paired ``Delta E_S`` values are restricted to the
intersection of the finite-positive Delta E_S ranges of the two classes.
Twenty target sizes are then sampled evenly in log(Delta E_S) across that
shared interval.  Each page of the final PDF contains the reversible plot on
the left and the irreversible plot on the right, from smaller to larger
Delta E_S.

Run from the repository root with::

    MPLCONFIGDIR=/tmp/mpl-cache .venv/bin/python -m \
        Plotting.plot_classified_energy_drops
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter

from Plotting.plot_single_energy_drop import (
    DEFAULT_CSV,
    ClassifiedDropPool,
    load_classified_drop_pool,
    load_drop_trace,
    plot_drop_trace,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / (
    "Plots/single_energy_drop/shared_es_region/even_overlap"
)
DEFAULT_PDF = DEFAULT_OUTPUT_ROOT / (
    "reversible_irreversible_20_shared_es_overlap.pdf"
)
EVENT_CLASSES = ("reversible", "irreversible")


def _merge_page_pdfs(page_pdfs: list[Path], pdf_path: Path) -> None:
    """Merge already-rendered pages while preserving their image bounds."""

    if not page_pdfs:
        raise ValueError("Cannot create a PDF without image pages.")

    ghostscript = shutil.which("gs")
    if ghostscript is not None:
        subprocess.run(
            [
                ghostscript,
                "-q",
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pdfwrite",
                f"-sOutputFile={pdf_path}",
                *(str(page_pdf) for page_pdf in page_pdfs),
            ],
            check=True,
        )
        return

    writer = PdfWriter()
    for page_pdf in page_pdfs:
        writer.append(PdfReader(str(page_pdf)))
    with pdf_path.open("wb") as output_file:
        writer.write(output_file)


def _shared_region_mask(pool: ClassifiedDropPool, event_class: str) -> np.ndarray:
    """Return the paired event mask for one class inside the shared region."""

    if event_class not in EVENT_CLASSES:
        raise ValueError(f"Unknown event class: {event_class!r}.")
    region_lo, region_hi = pool.shared_es_region
    es = pool.event_drops.es
    class_mask = (
        pool.split.is_rev
        if event_class == "reversible"
        else pool.split.is_irrev
    )
    return (
        class_mask
        & np.isfinite(es)
        & (es > 0)
        & (es >= region_lo)
        & (es <= region_hi)
    )


def _sample_evenly_by_es(
    pool: ClassifiedDropPool,
    event_class: str,
    count: int,
) -> list[tuple[int, float]]:
    """Sample class events at evenly spaced log Delta E_S targets.

    The returned list is sorted by the actual Delta E_S values.  Sampling in
    log space gives the small, middle, and large portions of the overlap
    comparable visual weight despite the broad energy-drop range.
    """

    if count < 1:
        raise ValueError("count must be at least 1.")
    mask = _shared_region_mask(pool, event_class)
    event_positions = np.flatnonzero(mask)
    if event_positions.size < count:
        raise ValueError(
            f"Only {event_positions.size} {event_class} events are available "
            f"in the shared Delta E_S region; requested {count}."
        )

    es = pool.event_drops.es[event_positions]
    order = np.argsort(es, kind="stable")
    sorted_positions = event_positions[order]
    sorted_es = es[order]
    region_lo, region_hi = pool.shared_es_region
    targets = np.geomspace(region_lo, region_hi, count)

    # Select one event for each ascending target, advancing through the
    # sorted population so the sampled events remain ordered and unique.
    selected_positions = []
    next_position = 0
    for target in targets:
        remaining = sorted_es[next_position:]
        distance = np.abs(np.log(remaining) - np.log(target))
        selected_position = next_position + int(np.argmin(distance))
        selected_positions.append(int(sorted_positions[selected_position]))
        next_position = selected_position + 1

    selected_positions = np.asarray(selected_positions, dtype=int)
    selected_es = pool.event_drops.es[selected_positions]
    if np.any(np.diff(selected_es) < 0):
        raise RuntimeError("Evenly sampled Delta E_S values are not ordered.")

    return [
        (
            int(pool.transition_indices[event_position]),
            float(pool.event_drops.es[event_position]),
        )
        for event_position in selected_positions
    ]


def _write_side_by_side_pdf(
    reversible_paths: list[Path],
    irreversible_paths: list[Path],
    pdf_path: Path,
    *,
    gap_pixels: int = 24,
) -> None:
    """Create one page per size with reversible left and irreversible right."""

    if len(reversible_paths) != len(irreversible_paths):
        raise ValueError("Both classes must have the same number of pages.")
    if not reversible_paths:
        raise ValueError("Cannot create a PDF without pages.")

    with TemporaryDirectory(prefix="energy-drop-comparison-") as temp_dir:
        temp_dir = Path(temp_dir)
        page_pdfs = []
        for page_number, (left_path, right_path) in enumerate(
            zip(reversible_paths, irreversible_paths),
            start=1,
        ):
            with Image.open(left_path) as left_image, Image.open(right_path) as right_image:
                left = left_image.convert("RGB")
                right = right_image.convert("RGB")
                if left.size != right.size:
                    raise ValueError(
                        "Reversible and irreversible plots must have the same "
                        f"size; got {left.size} and {right.size}."
                    )
                page = Image.new(
                    "RGB",
                    (left.width + gap_pixels + right.width, left.height),
                    "white",
                )
                page.paste(left, (0, 0))
                page.paste(right, (left.width + gap_pixels, 0))
                page_pdf = temp_dir / f"page_{page_number:03d}.pdf"
                page.save(page_pdf, "PDF", resolution=220.0)
                left.close()
                right.close()
            page_pdfs.append(page_pdf)

        _merge_page_pdfs(page_pdfs, pdf_path)


def generate_plots(
    csv_path: str | Path = DEFAULT_CSV,
    *,
    count: int = 20,
    pre_steps: int = 10,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    pdf_path: str | Path = DEFAULT_PDF,
    ylim_margin: float = 0.08,
) -> dict[str, list[Path]]:
    """Write individual class plots and one paired comparison PDF."""

    if count < 1:
        raise ValueError("count must be at least 1.")
    pool = load_classified_drop_pool(csv_path)
    output_root = Path(output_root).expanduser()
    pdf_path = Path(pdf_path).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    selected = {
        event_class: _sample_evenly_by_es(pool, event_class, count)
        for event_class in EVENT_CLASSES
    }
    image_paths: dict[str, list[Path]] = {event_class: [] for event_class in EVENT_CLASSES}

    for event_class in EVENT_CLASSES:
        output_dir = output_root / event_class
        output_dir.mkdir(parents=True, exist_ok=True)
        for number, (transition_index, delta_E_S) in enumerate(
            selected[event_class],
            start=1,
        ):
            trace = load_drop_trace(
                csv_path,
                pre_steps=pre_steps,
                drop_row=transition_index + 1,
                event_class=event_class,
                selection_label=f"{event_class} sample {number}/{count}",
            )
            if not np.isclose(trace.delta_E_S, delta_E_S, rtol=1e-12, atol=0.0):
                raise RuntimeError("Selected Delta E_S does not match the plotted trace.")
            image_path = output_dir / f"drop_{number:03d}.png"
            fig, _, _ = plot_drop_trace(
                trace,
                output_path=image_path,
                ylim_margin=ylim_margin,
            )
            plt.close(fig)
            image_paths[event_class].append(image_path)

    _write_side_by_side_pdf(
        image_paths["reversible"],
        image_paths["irreversible"],
        pdf_path,
    )

    print(
        f"kappa_det={pool.kappa_det:.8g}; "
        f"shared Delta E_S region=[{pool.shared_es_region[0]:.8g}, "
        f"{pool.shared_es_region[1]:.8g}]"
    )
    for event_class in EVENT_CLASSES:
        values = [value for _, value in selected[event_class]]
        print(
            f"{event_class}: {len(values)} samples, "
            f"Delta E_S=[{values[0]:.8g}, {values[-1]:.8g}]"
        )
        print(f"Wrote individual plots to {(output_root / event_class).resolve()}")
    print(f"Wrote paired multi-page PDF to {pdf_path.resolve()}")
    return image_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--pre-steps", type=int, default=10)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--ylim-margin", type=float, default=0.08)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    generate_plots(
        args.csv,
        count=args.count,
        pre_steps=args.pre_steps,
        output_root=args.output_root,
        pdf_path=args.pdf,
        ylim_margin=args.ylim_margin,
    )


if __name__ == "__main__":
    main()
