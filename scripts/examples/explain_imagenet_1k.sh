#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export TORCH_XAI_OUTPUT_DIR=$SCRIPT_DIR/../output
export TORCH_XAI_CACHE_DIR=$SCRIPT_DIR/../output
export DATA_ROOT_DIR=$SCRIPT_DIR/../data

# declare the datasets, train_strategies and max_seq_lengths
MODEL="timm_model_resnet50"
declare -a datasets=(
    image_classification/imagenet_1k
)
declare -a checkpoints=(
    null # pretrained checkpoint, if available
)
declare -a batch_sizes=(
    2
    # 2
    # 2
    # 2
    # 2
    # 2
    # 2
    # 2
)
declare -a methods=(
    # saliency
    # deep_lift
    # input_x_gradient
    # guided_backprop
    # gradient_shap
    # integrated_gradients
    # deep_lift_shap
    # feature_ablation
    lime
    # kernel_shap
)

# set target script
RUN_SCRIPT=$SCRIPT_DIR/../runners/explain_image_classification.sh

# get length of an array
total_datasets=${#datasets[@]}
total_methods=${#methods[@]}

CONFIGS=()
# use for loop to read all values and indexes
for ((i = 0; i < ${total_datasets}; i++)); do
    for ((j = 0; j < ${total_methods}; j++)); do
        # echo $run_target
        CONFIGS+=("explain_${datasets[$i]}_${methods[$j]} $RUN_SCRIPT \
            +explainers/image_classification_explainer=with_preprocess \
            base/model_args=$MODEL \
            base/data_args=${datasets[$i]} \
            explanation_method=${methods[$j]} \
            n_test_samples=1000 \
            per_device_eval_batch_size=${batch_sizes[$j]} \
            checkpoint=${checkpoints[$i]} \
            test_model=false \
            visualize_explanations=true"
        )
    done
done

# # debug the configs by printing what we are about to run
total_configs=${#CONFIGS[@]}
echo $total_configs
for ((i = 0; i < ${total_configs}; i++)); do
    echo "Running script: ${CONFIGS[$i]}"
    # debug run the first script in configs
    set -- ${CONFIGS[$i]}
    EXP_NAME=$1
    SCRIPT="${@:2}"
    $SCRIPT
done
