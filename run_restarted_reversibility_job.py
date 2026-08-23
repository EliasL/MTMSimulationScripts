"""Run edge-flip and Delaunay reversibility jobs from the edge-flip load-1 dump."""

import os
import sys
from pathlib import Path

from Management.configGenerator import ConfigGenerator
from runSimulations import run_many_locally

OUTPUT_ROOT = Path("/Volumes/data/MTS2D_output")


def make_configs():
    configs, labels = ConfigGenerator.generate(
        seed=0,
        rows=200,
        cols=200,
        usingPBC="true",
        experiment="reversibilityProtocolTest",
        reconnectionMethod=("edgeFlip", "delaunay"),
        reconnectRevert=1,
        reconnectEdgeLocking=0,
        startLoad=.15,
        loadIncrement=1e-5,
        maxLoad=3.0,
        nrThreads=2,
        minimizer="LBFGS",
        epsR=0.0,
        LBFGSEpsg=0.0,
        LBFGSEpsf=0.0,
        LBFGSEpsx=1e-6,
        energyDropThreshold=0.1,
    )

    return configs, labels


def main():
    configs, labels = make_configs()
    for config in configs:
        print(f"{config.reconnectionMethod}: {config.name}")

    if "--dry-run" in sys.argv[1:]:
        return
    if not OUTPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Output root not found: {OUTPUT_ROOT}")

    # The config also sets this through omp_set_num_threads(2); keep the
    # process environment explicit because both jobs run concurrently.
    os.environ["OMP_NUM_THREADS"] = "2"
    run_many_locally(
        configs,
        taskNames=labels,
        maxWorkers=2,
        build=True,
        outputPath=str(OUTPUT_ROOT),
    )


if __name__ == "__main__":
    main()
