#!/bin/bash
#SBATCH --job-name=sylvain-rev-post
#SBATCH --time=1-00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=sylvain-rev-post-%A_%a.out
#SBATCH --error=sylvain-rev-post-%A_%a.err

set -euo pipefail

: "${DATA_ROOT:?DATA_ROOT must be exported by the submit script}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT must be exported by the submit script}"
: "${MANIFEST:?MANIFEST must be exported by the submit script}"
: "${PYTHON_BIN:?PYTHON_BIN must be exported by the submit script}"
: "${PROJECT_ROOT:?PROJECT_ROOT must be exported by the submit script}"
: "${SLURM_ARRAY_TASK_ID:?This script must run as a Slurm array job}"

LINE_NUMBER=$((SLURM_ARRAY_TASK_ID + 1))
LINE="$(sed -n "${LINE_NUMBER}p" "${MANIFEST}")"
if [[ -z "${LINE}" ]]; then
    echo "No manifest entry at array index ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

IFS=$'\t' read -r BATCH JOB_NAME <<< "${LINE}"
JOB_DIR="${DATA_ROOT}/${JOB_NAME}"
JOB_OUTPUT="${OUTPUT_ROOT}/batch_${BATCH}/${JOB_NAME}.csv.gz"
mkdir -p "$(dirname -- "${JOB_OUTPUT}")"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
srun --cpu-bind=cores --distribution=block:block \
    "${PYTHON_BIN}" -u "${PROJECT_ROOT}/ClusterJobs/reversibility_postprocess.py" \
    process --batch "${BATCH}" --job-dir "${JOB_DIR}" --output "${JOB_OUTPUT}"
