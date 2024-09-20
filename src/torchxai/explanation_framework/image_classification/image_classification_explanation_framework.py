from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch
import tqdm
from omegaconf import DictConfig
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.utilities.data_collators import BatchToTensorDataCollator
from torchfusion.core.utilities.logging import get_logger

from torchxai import *  # noqa
from torchxai.explanation_framework.core.batch_compute_cache.explanations.explanations import (
    EXPLANATIONS_KEY,
    ExplanationsBatchComputeCache,
)
from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.core.explanation_framework import (
    FusionExplanationFramework,
)
from torchxai.explanation_framework.core.utils.general import (
    pretty_classification_report,
)
from torchxai.explanation_framework.core.utils.h5io import HFDataset, HFIOSingleOutput

logger = get_logger()


class ImageClassificationExplanationFramework(FusionExplanationFramework):
    """
    A framework for document classification explanation using TorchFusion.

    Args:
        args (FusionArguments): Arguments for the framework.
        hydra_config (DictConfig): Configuration for the framework.
    """

    def _get_data_collators(
        self,
        tokenizer: Any,
    ) -> CollateFnDict:
        """
        Get data collators for the framework.

        Args:
            tokenizer (Any): The tokenizer to be used.

        Returns:
            CollateFnDict: A dictionary of data collators.
        """
        data_key_type_map = {
            DataKeys.INDEX: torch.long,
            DataKeys.LABEL: torch.long,
            DataKeys.IMAGE_FILE_PATH: None,
            DataKeys.IMAGE: torch.float,
        }
        collate_fn = BatchToTensorDataCollator(
            allowed_keys=list(data_key_type_map.keys())
        )
        return CollateFnDict(train=collate_fn, validation=collate_fn, test=collate_fn)

    def _evaluate_model(self) -> None:
        """
        Evaluate the model and log the classification report.
        """
        all_predicted_labels = []
        all_target_labels = []
        self._wrapped_model.configure(setup_for_explanation=False)
        for batch in tqdm.tqdm(self._test_dataloader_full):
            with torch.no_grad():
                # perform the model forward pass
                _, _, predicted_labels = self._wrapped_model(batch, device=self._device)

                # get model predictions from the outputs
                predicted_labels = predicted_labels.cpu()
                target_labels = batch[DataKeys.LABEL].cpu()

                # append the predicted and target labels
                all_target_labels.extend(target_labels)
                all_predicted_labels.extend(predicted_labels)

        # log the classification report
        pretty_classification_report(all_target_labels, all_predicted_labels)

    def _compute_and_save_explanations(
        self,
        sample_keys: List[str],
        explainer: FusionExplainer,
        model_inputs: Dict[str, Any],
        model_outputs: Any,
        runtime_config: DictConfig,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], bool]:
        """
        Compute and save explanations and infidelity scores.

        Args:
            sample_keys (List[str]): List of sample keys.
            explainer (TorchFusionExplainer): The explainer to be used.
            model_inputs (Dict[str, Any]): Model inputs.
            model_outputs (Any): Model outputs.
            runtime_config (DictConfig): Runtime configuration.
        """
        _, predicted_labels = model_outputs
        with HFIOSingleOutput(
            self._output_file.with_suffix(".explanation.h5")
        ) as hf_sample_data_io_explanations:
            # make a batch explanation computer
            explanations_computer = ExplanationsBatchComputeCache(
                None,
                hf_sample_data_io_explanations,
                explanation_reduce_fn=self._reduce_explanations,
            )
            all_dependent_metrics_cached = self._verify_depdendent_metrics_computed(
                sample_keys=sample_keys,
                required_metrics=list(runtime_config.metrics.keys()),
            )
            if not all_dependent_metrics_cached:
                logger.warning(
                    "Recomputing explanations for this batch as dependent metrics are not cached."
                )

            explanation_outputs, is_raw_explanation = (
                explanations_computer.compute_and_save(
                    sample_keys=sample_keys,
                    explainer=explainer,
                    model_inputs=model_inputs,
                    batch_target_labels=predicted_labels,
                    force_recompute=not all_dependent_metrics_cached,
                    train_baselines=self._train_baselines,
                )
            )

            explanations = explanation_outputs[EXPLANATIONS_KEY]
            for explanation in explanations:
                logger.debug(f"Explanations: {explanation.shape}")

            # change device to cpu
            if isinstance(explanations, tuple):
                explanations = tuple(x.detach().cpu() for x in explanations)
            else:
                explanations = explanations.detach().cpu()

            return explanations, is_raw_explanation

    def _reduce_explanations(self, explanations) -> None:
        if isinstance(explanations, tuple):
            return tuple(x.sum(-1) for x in explanations)
        else:
            return explanations.sum(-1)

    def _setup_train_baselines(
        self, runtime_config: DictConfig
    ) -> Dict[str, torch.Tensor]:
        """
        Setup baselines from the training data.

        Must be implemented by the derived class.
        """
        if self._train_dataloader is None:
            self._logger.error("Train dataloader is not set up.")
            raise ValueError("Train dataloader is not set up.")

        if self._output_file is None:
            self._logger.error("Output file is not set up.")
            raise ValueError("Output file is not set up.")

        baselines_cached = self._output_file.parent / "train_baselines.h5"

        # Check if baselines are already available
        logger.info(f"Load train baselines from {baselines_cached}")
        baselines = {}
        with HFDataset(baselines_cached) as hfio:
            for key in [DataKeys.IMAGE]:
                if hfio.key_exists(key):
                    baselines[key] = torch.from_numpy(hfio.load(key))

        # If all baselines are available, return them
        if all([len(value) > 0 for value in baselines.values()]):
            return baselines

        pbar = tqdm.tqdm(self._train_dataloader)
        for batch in pbar:
            with torch.no_grad():
                bsz = batch[DataKeys.IMAGE].shape[0]
                baselines[DataKeys.IMAGE] = batch[DataKeys.IMAGE]
                if (
                    bsz * len(baselines[DataKeys.IMAGE])
                    >= runtime_config.train_baselines_size
                ):
                    break

        for key, value in baselines.items():
            baselines[key] = torch.cat(value)

        logger.info(f"Saving train baselines to {baselines_cached}")
        with HFDataset(baselines_cached) as hfio:
            for key, data in baselines.items():
                hfio.save(key, data)

        return baselines

    def _visualize_explanations(
        self,
        batch: Dict[str, torch.Tensor],
        model_inputs: Dict[str, Any],
        explanations: torch.Tensor | Tuple[torch.Tensor],
    ) -> None:
        pass
