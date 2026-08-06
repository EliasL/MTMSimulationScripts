#!/bin/bash
# Prepare and submit one Slurm array plus one dependent merge job per batch.
# This script only submits when it is explicitly run on a cluster login node.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 DATA_ROOT OUTPUT_ROOT [BATCH ...]" >&2
    echo "Example: $0 /data2/elundheim/MTS2D_output /data2/elundheim/MTS2D_postprocessed/sylvain_reversibility -2" >&2
    exit 2
fi

DATA_ROOT="$1"
OUTPUT_ROOT="$2"
shift 2
if [[ $# -eq 0 ]]; then
    set -- -2 -1
fi

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/simulation/reversibility-postprocess-venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found at ${PYTHON_BIN}." >&2
    echo "Run ClusterJobs/setup_reversibility_postprocessing_env.sh first." >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}/manifests"

for BATCH in "$@"; do
    MANIFEST="${OUTPUT_ROOT}/manifests/batch_${BATCH}.tsv"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/ClusterJobs/reversibility_postprocess.py" manifest \
        --data-root "${DATA_ROOT}" --manifest "${MANIFEST}" \
        --batch "${BATCH}" --allow-missing --allow-empty
    COUNT="$(wc -l < "${MANIFEST}")"
    if [[ "${COUNT}" -eq 0 ]]; then
        echo "No batch ${BATCH} jobs found under ${DATA_ROOT}; skipping."
        continue
    fi

    ARRAY_JOB_ID="$(sbatch --parsable --array="0-$((COUNT - 1))" \
        --export="ALL,DATA_ROOT=${DATA_ROOT},OUTPUT_ROOT=${OUTPUT_ROOT},MANIFEST=${MANIFEST},PYTHON_BIN=${PYTHON_BIN},PROJECT_ROOT=${PROJECT_ROOT}" \
        "${PROJECT_ROOT}/ClusterJobs/reversibility_postprocess.sbatch")"
    MERGE_JOB_ID="$(sbatch --parsable --dependency="afterok:${ARRAY_JOB_ID}" \
        --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT},MANIFEST=${MANIFEST},BATCH=${BATCH},PYTHON_BIN=${PYTHON_BIN},PROJECT_ROOT=${PROJECT_ROOT}" \
        "${PROJECT_ROOT}/ClusterJobs/reversibility_postprocess_merge.sbatch")"
    echo "batch=${BATCH} array_job=${ARRAY_JOB_ID} merge_job=${MERGE_JOB_ID} jobs=${COUNT}"
done
