#!/usr/bin/env python3
"""Compare archived Ftotal fields with total T or the stored plastic factor T_p."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Plotting.element_tracking import read_matrix_component
from Plotting.vtuDataForSylvain import VTUData
from Plotting.dataFunctions import infer_strain_from_vtu


def read_matrix(data: VTUData, name: str) -> np.ndarray:
    return np.stack([
        read_matrix_component(data, name, i, j, symmetric=False)
        for i, j in ((1, 1), (1, 2), (2, 1), (2, 2))
    ], axis=-1).reshape(-1, 2, 2)


def frame_load(path: Path) -> float:
    load = infer_strain_from_vtu(path)
    if load is None or not np.isfinite(load):
        raise ValueError(f"Cannot extract a finite load from {path}")
    return float(load)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vtu_directory', type=Path)
    parser.add_argument('output_png', type=Path)
    parser.add_argument('--output-csv', type=Path)
    parser.add_argument('--comparison', choices=('T', 'T_p', 'both'), default='T')
    args = parser.parse_args()
    paths = sorted(args.vtu_directory.glob('*.vtu'), key=frame_load)
    if not paths:
        raise FileNotFoundError(f'No VTU files found in {args.vtu_directory}')
    names = ['Ftotal'] + (['T_p', 'T'] if args.comparison == 'both' else [args.comparison])
    rows = []
    for path in paths:
        data = VTUData(path)
        matrices = {'Ftotal': read_matrix(data, 'Ftotal')}
        # Legacy VTU T.. fields store T_p; total T = F_E @ T_p.
        plastic = read_matrix(data, 'T')
        if 'T_p' in names:
            matrices['T_p'] = plastic
        if 'T' in names:
            matrices['T'] = read_matrix(data, 'F_E') @ plastic
        if any(not np.all(np.isfinite(matrices[name])) for name in names):
            raise ValueError(f'Non-finite matrix values in {path}')
        if any(matrices[name].shape != matrices['Ftotal'].shape for name in names):
            raise ValueError(f'Matrix cell counts differ in {path}')
        row = {'gamma': frame_load(path)}
        for name in names:
            mean = matrices[name].mean(axis=0)
            row.update({f'{name}{i+1}{j+1}_mean': mean[i, j]
                        for i in range(2) for j in range(2)})
        rows.append(row)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    gamma = [row['gamma'] for row in rows]
    for ax, component in zip(axes.flat, ('11', '12', '21', '22')):
        for name in names:
            key = f'{name}{component}_mean'
            ax.plot(gamma, [row[key] for row in rows], label=name)
        ax.set_title(f'Component {component}')
        ax.set_ylabel('element-wise mean')
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    for ax in axes[-1]:
        ax.set_xlabel(r'$\gamma$')
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=220, bbox_inches='tight')
    fig.savefig(args.output_png.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)
    output_csv = args.output_csv or args.output_png.with_suffix('.csv')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {args.output_png}, {args.output_png.with_suffix(".pdf")}, {output_csv}')


if __name__ == '__main__':
    main()
