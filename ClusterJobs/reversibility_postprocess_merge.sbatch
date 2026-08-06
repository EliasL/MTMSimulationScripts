#!/bin/bash
#SBATCH --job-name=sylvain-rev-merge
#SBATCH --time=0-02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=sylvain-rev-merge-%j.out
#SBATCH --error=sylvain-rev-merge-%j.err

set -euo pipefail

: "${OUTPUT_ROOT:?OUTPUT_ROOT must be exported by the submit script}"
: "${MANIFEST:?MANIFEST must be exported by the submit script}"
: "${BATCH:?BATCH must be exported by the submit script}"
: "${PYTHON_BIN:?PYTHON_BIN must be exported by the submit script}"
: "${PROJECT_ROOT:?PROJECT_ROOT must be exported by the submit script}"

OUTPUT_PATH="${OUTPUT_ROOT}/sylvain_batch_${BATCH}_reversibility_events.csv.gz"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
srun --cpu-bind=cores --distribution=block:block \
    "${PYTHON_BIN}" -u "${PROJECT_ROOT}/ClusterJobs/reversibility_postprocess.py" \
    merge --batch "${BATCH}" --manifest "${MANIFEST}" \
    --output-root "${OUTPUT_ROOT}" --output "${OUTPUT_PATH}"
