#!/bin/bash
# Upload only the worker, its plotting helpers, and the required MTMath files.

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 CLUSTER_HOST" >&2
    exit 2
fi

CLUSTER_HOST="$1"
REMOTE="elundheim@${CLUSTER_HOST}"
REMOTE_ROOT="${REMOTE}:~/simulation/SimulationScripts"

rsync -av -e "ssh -T" --exclude '__pycache__' --exclude '*.pyc' \
    ClusterJobs/ "${REMOTE_ROOT}/ClusterJobs/"
rsync -av -e "ssh -T" \
    Management/__init__.py Management/updateCSV.py \
    "${REMOTE_ROOT}/Management/"
rsync -av -e "ssh -T" \
    Plotting/__init__.py Plotting/dataFunctions.py \
    Plotting/energyDropCalculations.py Plotting/vtuDataForSylvain.py \
    "${REMOTE_ROOT}/Plotting/"
rsync -av -e "ssh -T" \
    MTMath/__init__.py MTMath/energyFunction.py MTMath/reduction.py MTMath/meshUtils.py \
    "${REMOTE_ROOT}/MTMath/"

echo "Uploaded postprocessing sources to ${REMOTE_ROOT}."
