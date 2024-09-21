from __future__ import annotations

import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torchfusion.core.utilities.logging import get_logger

from torchxai.explanation_framework.utils.h5io import HFIOSingleOutput

logger = get_logger()


class BatchComputationHandler:
    def __init__(
        self,
        metric_name: str,
        output_file: Union[str, Path],
        file_mode: str = "a",
    ) -> None:
        self._metric_name = metric_name
        self._output_file = output_file
        self._hfio = HFIOSingleOutput(self._output_file, mode=file_mode)

    def __enter__(self):
        with ExitStack() as stack:
            stack.enter_context(self._hfio)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stack.__exit__(exc_type, exc_val, exc_tb)

    def is_cached_for_sample_keys(self, sample_keys: List[str]) -> bool:
        return np.all(
            [
                self._hfio.load(f"{self._metric_name}_computed", sample_key)
                for sample_key in sample_keys
            ]
        )

    def _log_outputs(outputs: dict[Any]):
        pass

    def compute_and_save(
        self,
        sample_keys: List[str],
        force_recompute=False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], bool]:
        # If already computed, load the existing data
        if not force_recompute and self.is_cached_for_sample_keys(sample_keys):
            return self._load_outputs(sample_keys), False

        # Otherwise, perform the computation and save results
        start_time = time.time()
        outputs = self._compute_metric(**kwargs)
        time_taken = time.time() - start_time
        logger.debug(f"Time spent on computing {self._metric_name}: {time_taken:.4f}s")

        # Log outputs
        self._log_outputs(outputs)

        # Save computed outputs
        self._save_outputs(sample_keys, outputs, time_taken)

        # save sensitivity in output file
        for sample_key in sample_keys:
            # set flag that this sample's attribution and infidelity are computed
            self._hfio.save(f"{self._metric_name}_computed", True, sample_key)

        return outputs, True

    def _compute_metric(self, **kwargs) -> Tuple:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def _save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
    ) -> None:
        # Save time taken for computation
        for sample_key in sample_keys:
            self._hfio.save_attribute(  # save the time taken for each sample key divided by batch size
                f"time_taken_{self._metric_name}",
                time_taken / len(sample_keys),
                sample_key,
            )

    def _load_outputs(
        self, sample_keys: List[str]
    ) -> Tuple[Union[torch.Tensor, np.ndarray], ...]:
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )
