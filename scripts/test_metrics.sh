SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

export PYTHONPATH=$SCRIPTPATH/../src:$SCRIPTPATH/../tests

# axiomatic done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/axiomatic/test_completeness.py # done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/axiomatic/test_non_sensitivity.py # done

# complexity done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/complexity/test_complexity.py # done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/complexity/test_effective_complexity.py # done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/complexity/test_sparseness.py # done

# faithfulness done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/faithfulness/test_monotonicity_corr.py # done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/faithfulness/test_monotonicity.py # done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/faithfulness/test_faithfulness_corr.py # done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/faithfulness/test_faithfulness_estimate.py # done
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/faithfulness/test_infidelity.py # already done in captum
python -W ignore -m unittest -v $SCRIPTPATH/../tests/metrics/faithfulness/test_aopc.py # already done in captum