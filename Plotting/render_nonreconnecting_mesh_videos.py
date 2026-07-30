"""Render one mesh animation for each non-reconnecting L=100 sample."""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/simulationscripts-matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/simulationscripts-cache")

from Plotting.makeAnimations import makeAnimations


DATA_NAME = (
    "simpleShear,s100x100l0.15,1e-05,1.0PBCt3LBFGSEpsx1e-06s"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--video-name", default="mesh")
    parser.add_argument("--reuse-images", action="store_true")
    args = parser.parse_args()

    root = Path("Plots")
    runs = []
    for seed in args.seeds:
        if seed == 2:
            folder = root / "elastic_norm_data/no_reconnection_L100_s2"
            macro_data = (
                root / "energy_prediction_normal_data" / (DATA_NAME + "2")
                / "macroData.csv"
            ).resolve()
        else:
            folder = next((root / "animation_data").glob(DATA_NAME + str(seed)))
            macro_data = None
        folder = folder.resolve()
        runs.append((seed, folder, macro_data))

    for seed, folder, macro_data in runs:
        print(f"Rendering seed {seed} from {folder}", flush=True)
        makeAnimations(
            str(folder),
            macroData=str(macro_data) if macro_data else None,
            makeGIF=False,
            combineVideos=False,
            useTqdm=True,
            fps=30,
            seconds_per_unit_shear=15,
            allImages=False,
            minTime=7,
            reuseImages=args.reuse_images,
            X="load",
            videoNames=args.video_name,
            num_processes=args.num_processes,
        )
        print(
            f"Finished seed {seed}: {folder / f'{args.video_name}_video.mp4'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
