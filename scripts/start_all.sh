#!/bin/bash
WORKER_SCRIPT="./scripts/base_qsub.sh"
TAG="rp2-balanced+pacond"
MODALITIES_LIST=(
    "CFP"
)
PACOND_LIST=(
    "mixed"
    "healthy"
    "unhealthy"
)
for MOD in "${MODALITIES_LIST[@]}"; do
    for PACOND in "${PACOND_LIST[@]}"; do
        CURRENT_JOB_NAME="${TAG}_${MOD}_${PACOND}"
        qsub -N "$CURRENT_JOB_NAME" \
            -v MODALITIES="$MOD" \
            -v PACOND="$PACOND" \
            "$WORKER_SCRIPT"
    done
done