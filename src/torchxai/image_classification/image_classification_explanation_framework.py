from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import tqdm
from omegaconf import DictConfig
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.models.utilities.data_collators import BatchToTensorDataCollator
from torchfusion.core.utilities.logging import get_logger

from torchxai import *  # noqa
from torchxai.explanation_framework.explained_model.base import ExplainedModel
from torchxai.explanation_framework.explained_model.image_classification import (
    ExplainedModelForImageClassification,
)
from torchxai.explanation_framework.explanation_framework import (
    FusionExplanationFramework,
)
from torchxai.explanation_framework.utils.common import ExplanationParameters
from torchxai.explanation_framework.utils.h5io import HFDataset
from torchxai.image_classification.utils.evaluation import pretty_classification_report
from torchxai.image_classification.utils.visualization import (
    visualize_image_with_explanation,
)

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

    def _prepare_model_for_explanation(
        self, model: torch.nn.Module, runtime_config: DictConfig
    ) -> ExplainedModel:
        """
        Wrap the model for explanation.

        Must be implemented by the derived class.

        Args:
            model (torch.nn.Module): The model to wrap.
        """
        return ExplainedModelForImageClassification(
            model,
            segmentation_fn=runtime_config.segmentation_fn,
            segmentation_fn_kwargs=runtime_config.segmentation_fn_kwargs,
        )

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
                explanation_parameters = (
                    self._wrapped_model.prepare_explanation_parameters(
                        batch=batch, device=self._device
                    )
                )
                pred_probs = self._wrapped_model(explanation_parameters.model_inputs)

                # get model predictions from the outputs
                predicted_labels = pred_probs.argmax(-1).cpu()
                target_labels = batch[DataKeys.LABEL].cpu()

                # append the predicted and target labels
                all_target_labels.extend(target_labels)
                all_predicted_labels.extend(predicted_labels)

        # log the classification report
        pretty_classification_report(all_target_labels, all_predicted_labels)

    def _reduce_explanations(self, explanations) -> None:
        if isinstance(explanations, tuple):
            return tuple(x.sum(1).unsqueeze(1) for x in explanations)
        else:
            return explanations.sum(1).unsqueeze(1)

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
        explanation_parameters: ExplanationParameters,
        explanations: torch.Tensor | Tuple[torch.Tensor],
        model_outputs: torch.Tensor,
        runtime_config: DictConfig,
    ) -> None:
        visualize_image_with_explanation(
            explanation_parameters.model_inputs[0],
            explanations[0],
            runtime_config.explanation_method,
        )
