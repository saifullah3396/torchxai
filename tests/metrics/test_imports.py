import logging
from logging import getLogger

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


def test_axiomatic_imports() -> None:
    from torchxai.metrics import completeness  # noqa
    from torchxai.metrics import input_invariance  # noqa
    from torchxai.metrics import monotonicity_corr_and_non_sens  # noqa


def test_complexity_imports() -> None:
    from torchxai.metrics import complexity  # noqa
    from torchxai.metrics import effective_complexity  # noqa
    from torchxai.metrics import sparseness  # noqa


def test_faithfulness_imports() -> None:
    from torchxai.metrics import aopc  # noqa
    from torchxai.metrics import faithfulness_corr  # noqa
    from torchxai.metrics import faithfulness_estimate  # noqa
    from torchxai.metrics import infidelity  # noqa
    from torchxai.metrics import monotonicity  # noqa
    from torchxai.metrics import sensitivity_n  # noqa


def test_robustness_imports() -> None:
    from torchxai.metrics import sensitivity_max_and_avg  # noqa
