import logging
from logging import getLogger

from tests.metrics.base import MetricTestsBase

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_axiomatic_imports(self) -> None:
        from torchxai.metrics import completeness  # noqa
        from torchxai.metrics import monotonicity_corr_and_non_sens  # noqa

    def test_complexity_imports(self) -> None:
        from torchxai.metrics import complexity  # noqa
        from torchxai.metrics import effective_complexity  # noqa
        from torchxai.metrics import sparseness  # noqa

    def test_faithfulness_imports(self) -> None:
        from torchxai.metrics import aopc  # noqa
        from torchxai.metrics import faithfulness_corr  # noqa
        from torchxai.metrics import faithfulness_estimate  # noqa
        from torchxai.metrics import infidelity  # noqa
        from torchxai.metrics import monotonicity  # noqa

    def test_robustness_imports(self) -> None:
        from torchxai.metrics import sensitivity_max  # noqa
