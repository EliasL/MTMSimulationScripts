"""Run edge-flip and Delaunay reversibility jobs from the edge-flip load-1 dump."""

import os
import sys
from pathlib import Path

from Management.configGenerator import ConfigGenerator
from runSimulations import run_many_locally


DUMP = Path(
    "/Volumes/data/MTS2D_output/"
    "simpleShear,s200x200l0.15,1e-05,5.1PBCedgeFlipt5"
    "epsR1e-05LBFGSEpsg1e-08LBFGSEpsx1e-06s0/dumps/dump_l1.0.xml.gz"
)
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
        startLoad=1.0,
        loadIncrement=1e-5,
        maxLoad=5.1,
        nrThreads=2,
        minimizer="LBFGS",
        epsR=0.0,
        LBFGSEpsg=0.0,
        LBFGSEpsf=0.0,
        LBFGSEpsx=1e-6,
        energyDropThreshold=0.1,
    )

    for config in configs:
        if (
            config.nrThreads != 2
            or config.reconnectRevert != 1
            or config.reconnectEdgeLocking != 0
            or config.epsR != 0.0
            or config.LBFGSEpsg != 0.0
            or config.LBFGSEpsf != 0.0
            or config.LBFGSEpsx != 1e-6
        ):
            raise RuntimeError(f"Generated config does not match requested settings: {config.name}")
    return configs, labels


def main():
    configs, labels = make_configs()
    print(f"Restart dump: {DUMP}")
    for config in configs:
        print(f"{config.reconnectionMethod}: {config.name}")

    if "--dry-run" in sys.argv[1:]:
        return
    if not DUMP.is_file():
        raise FileNotFoundError(f"Restart dump not found: {DUMP}")
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
        dump=str(DUMP),
        outputPath=str(OUTPUT_ROOT),
        newOutput=True,
        overwriteSettings=True,
    )


if __name__ == "__main__":
    main()
