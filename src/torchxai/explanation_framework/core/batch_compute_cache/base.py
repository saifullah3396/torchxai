from __future__ import annotations

import time
from typing import Any, List, Tuple, Union

import numpy as np
import torch
from torchfusion.core.utilities.logging import get_logger

from torchxai.explanation_framework.core.utils.h5io import (
    HFIOMultiOutput,
    HFIOSingleOutput,
)

logger = get_logger()


class BatchComputeCache:
    """
    Base class for handling batch computation and caching results in an HDF5 file. For all computations, we save
    the results of each sample in a separate group in the HDF5 file.

    Attributes:
        metric_name (str): The name of the metric being computed.
        hf_sample_data_io (HFIOSingleOutputSample | HFIOMultiOutputSample):
            The HDF5 interface for saving/loading sample data.
    """

    def __init__(
        self,
        metric_name: str,
        hf_sample_data_io: Union[HFIOSingleOutput, HFIOMultiOutput],
    ) -> None:
        self.metric_name = metric_name
        self.hf_sample_data_io = hf_sample_data_io

    def is_cached_for_sample_keys(self, sample_keys: List[str]) -> bool:
        """
        Checks if computation is already done for the given sample keys.

        Args:
            sample_keys (List[str]): List of sample keys to check.

        Returns:
            bool: True if all sample keys have precomputed data, False otherwise.
        """
        return np.all(
            [
                self.hf_sample_data_io.load(f"{self.metric_name}_computed", sample_key)
                for sample_key in sample_keys
            ]
        )

    def compute_and_save(
        self,
        sample_keys: List[str],
        force_recompute=False,
        **kwargs,
    ) -> Tuple[Union[torch.Tensor, np.ndarray], bool]:
        """
        Computes the metric for the given sample keys and saves the outputs.

        Args:
            sample_keys (List[str]): List of sample keys for computation.

        Returns:
            Tuple[Union[torch.Tensor, np.ndarray], ...]: Loaded or computed outputs.
            bool: True if the outputs are computed, False if loaded from cache.
        """
        # If already computed, load the existing data
        if not force_recompute and self.is_cached_for_sample_keys(sample_keys):
            return self.load_outputs(sample_keys), False

        # Otherwise, perform the computation and save results
        start_time = time.time()
        outputs = self.compute_metric(**kwargs)
        time_taken = time.time() - start_time
        logger.info(f"Time spent on computing {self.metric_name}: {time_taken:.4f}s")

        # Save computed outputs
        self.save_outputs(sample_keys, outputs, time_taken)

        # save sensitivity in output file
        for sample_key in sample_keys:
            # set flag that this sample's attribution and infidelity are computed
            self.hf_sample_data_io.save(
                f"{self.metric_name}_computed", True, sample_key
            )

        return outputs, True

    def compute_metric(self, **kwargs) -> Tuple:
        """
        Abstract method to be implemented in subclasses for specific metric computation.

        Args:
            sample_keys (List[str]): List of sample keys for computation.

        Raises:
            NotImplementedError: If not implemented in a derived class.
        """
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )

    def save_outputs(
        self,
        sample_keys: List[str],
        outputs: dict[Any],
        time_taken: float,
    ) -> None:
        """
        Abstract method to be implemented in subclasses for saving outputs.

        Args:
            sample_keys (List[str]): List of sample keys.
            outputs (Tuple): Outputs to be saved.
            time_taken (float): Time taken for computation.

        Raises:
            NotImplementedError: If not implemented in a derived class.
        """
        # Save time taken for computation
        for sample_key in sample_keys:
            self.hf_sample_data_io.save_attribute(  # save the time taken for each sample key divided by batch size
                f"time_taken_{self.metric_name}",
                time_taken / len(sample_keys),
                sample_key,
            )

    def load_outputs(
        self, sample_keys: List[str]
    ) -> Tuple[Union[torch.Tensor, np.ndarray], ...]:
        """
        Abstract method to be implemented in subclasses for loading outputs.

        Args:
            sample_keys (List[str]): List of sample keys.

        Raises:
            NotImplementedError: If not implemented in a derived class.
        """
        raise NotImplementedError(
            "This method should be implemented in the derived class"
        )
