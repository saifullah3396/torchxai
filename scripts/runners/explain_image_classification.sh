#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export PYTHONPATH=$SCRIPT_DIR/../../src:$SCRIPT_DIR/../../external/torchfusionv2/src:$SCRIPT_DIR/../torchxai/src:

export TORCH_FUSION_OUTPUT_DIR=$SCRIPT_DIR/../output
export TORCH_FUSION_CACHE_DIR=$SCRIPT_DIR/../output
export DATA_ROOT_DIR=$SCRIPT_DIR/../data

# run the explainer script
LOG_LEVEL=DEBUG python3 -W ignore $SCRIPT_DIR/../../src/torchxai/runners/explain_image_classification.py --config-path ../../../cfg/ --config-name hydra +explainers=hydra "${@:1}"
# python3 -m debugpy --wait-for-client --listen 0.0.0.0:5678  $SCRIPT_DIR/../../src/torchxai/runners/explain_image_classification.py --config-path ../../../cfg/ --config-name hydra +explainers=hydra "${@:1}"
