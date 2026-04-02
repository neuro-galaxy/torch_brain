#!/usr/bin/env bash
set -euo pipefail

# Edit these constants directly for your run.
REGIME="SS-SM"  # SS-SM | SS-DM | DS-DM
PATHS_CFG="default"
TEST_SUBJECT=1
TEST_SESSION=1
TASK="onset"

LABEL_MODE="binary"
MODEL="logistic"
PREPROCESSOR="laplacian_stft" # laplacian_stft_region_pool for DS-DM
OUTPUT_GROUP="release_2025_public"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXAMPLES_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"
RUN_DIR="${PROJECT_DIR}/outputs/${OUTPUT_GROUP}/${MODEL}_${PREPROCESSOR}/${REGIME}/${TASK}/sub${TEST_SUBJECT}_sess${TEST_SESSION}"

mkdir -p "${RUN_DIR}"
cd "${EXAMPLES_DIR}"

python -m neuroprobe_eval.run_eval \
  paths="${PATHS_CFG}" \
  dataset.provider=neuroprobe2025 \
  dataset.regime="${REGIME}" \
  dataset.label_mode="${LABEL_MODE}" \
  dataset.task="${TASK}" \
  dataset.test_subject="${TEST_SUBJECT}" \
  dataset.test_session="${TEST_SESSION}" \
  dataset.merge_val_into_test=true \
  model="${MODEL}" \
  preprocessor="${PREPROCESSOR}" \
  wandb.enabled=false \
  runtime.overwrite=true \
  runtime.verbose=true \
  hydra.run.dir="${RUN_DIR}"
