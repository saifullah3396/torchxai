SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PYTHONPATH=$SCRIPTPATH/../src:$SCRIPTPATH/../
# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_grad/test_saliency.py $@
# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_grad/test_input_x_gradient.py $@
# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_grad/test_guided_backprop.py $@
# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_grad/test_integrated_gradients.py $@
# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_grad/test_deeplift.py $@
# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_grad/test_gradient_shap.py $@
# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_grad/test_deeplift_shap.py $@

# python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_perturbation/test_feature_ablation.py $@
python -W ignore $SCRIPTPATH/../tests/explanation_framework/explainers/_perturbation/test_lime.py $@