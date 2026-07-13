#!/usr/bin/env bash

set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-/Volumes/data/remoteData/macro}"
OUTPUT="${OUTPUT:-${ROOT}/Plots/powerLaw/size_collapse}"
LOG_DIR="${OUTPUT}/logs"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Python environment not found or not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Size-scaling data directory not found: ${DATA_ROOT}" >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${ROOT}/.cache/matplotlib" "${ROOT}/.cache/fontconfig"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${ROOT}/.cache/matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT}/.cache/fontconfig}"

cd "${ROOT}"
echo "Starting/resuming sampled global-min size collapse at $(date)"
echo "Progress log: ${LOG_DIR}/global_min.log"

"${PYTHON}" -m Plotting.sizeScalingCollapse \
    --data-root "${DATA_ROOT}" \
    --output "${OUTPUT}" \
    --stage all \
    --xmin-strategy global_min \
    --xmin-accuracy 0.1 \
    --parallel-xmin \
    "$@" 2>&1 | tee -a "${LOG_DIR}/global_min.log"

echo "Finished sampled global-min size collapse at $(date)"
