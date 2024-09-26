SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PYTHONPATH=$SCRIPTPATH/../src:$SCRIPTPATH/../

# basic test
python -W ignore $SCRIPTPATH/../tests/metrics/test_imports.py $@

# axiomatic
python -W ignore $SCRIPTPATH/../tests/metrics/axiomatic/test_completeness.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/axiomatic/test_non_sensitivity.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/axiomatic/test_input_invariance.py $@

# # complexity
python -W ignore $SCRIPTPATH/../tests/metrics/complexity/test_complexity.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/complexity/test_complexity_sundararajan.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/complexity/test_effective_complexity.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/complexity/test_sparseness.py $@

# # faithfulness
python -W ignore $SCRIPTPATH/../tests/metrics/faithfulness/test_monotonicity_corr.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/faithfulness/test_monotonicity.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/faithfulness/test_faithfulness_corr.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/faithfulness/test_faithfulness_estimate.py $@
python -W ignore $SCRIPTPATH/../tests/metrics/faithfulness/test_aopc.py $@