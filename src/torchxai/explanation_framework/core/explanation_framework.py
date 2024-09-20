from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import tqdm
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchfusion.core.args.args import FusionArguments
from torchfusion.core.constants import DataKeys
from torchfusion.core.data.datasets.dataset_metadata import FusionDatasetMetaData
from torchfusion.core.data.factory.batch_sampler import BatchSamplerFactory
from torchfusion.core.data.text_utils.tokenizers.hf_tokenizer import (
    HuggingfaceTokenizer,
)
from torchfusion.core.data.utilities.containers import CollateFnDict
from torchfusion.core.data.utilities.loaders import load_datamodule_from_args
from torchfusion.core.training.fusion_trainer import FusionTrainer
from torchfusion.core.training.utilities.constants import TrainingStage
from torchfusion.core.training.utilities.general import (
    initialize_torch,
    print_tf_from_loader,
    setup_logging,
)
from torchfusion.core.utilities.logging import get_logger

from torchxai import *  # noqa
from torchxai.explanation_framework.core.batch_compute_cache.base import (
    BatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.explanations.explanations import (
    EXPLANATIONS_KEY,
    ExplanationsBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.axiomatic.completeness import (
    CompletenessBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.axiomatic.monotonicity_corr_non_sens import (
    MonotonicityCorrNonSensitivityBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.complexity.complexity import (
    ComplexityBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.complexity.effective_complexity import (
    EffectiveComplexityBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.complexity.sparseness import (
    SparsenessBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.faithfulness.aopc import (
    AOPCBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.faithfulness.faithfulness_corr import (
    FaithfulnessCorrelationBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.faithfulness.faithfulness_estimate import (
    FaithfulnessEstimateBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.faithfulness.infidelity import (
    InfidelityBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.faithfulness.monotonicity import (
    MonotonicityBatchComputeCache,
)
from torchxai.explanation_framework.core.batch_compute_cache.metrics.robustness.sensitivity import (
    SensitivityBatchComputeCache,
)
from torchxai.explanation_framework.core.explainers.factory import ExplainerFactory
from torchxai.explanation_framework.core.explainers.torch_fusion_explainer import (
    FusionExplainer,
)
from torchxai.explanation_framework.core.model_forward_wrappers.base import (
    ModelForwardWrapper,
)
from torchxai.explanation_framework.core.utils.constants import (
    EXPLANATION_METRICS,
    RAW_EXPLANATION_DEPENDENT_METRICS,
)
from torchxai.explanation_framework.core.utils.general import generate_unique_sample_key
from torchxai.explanation_framework.core.utils.h5io import HFIOSingleOutput

logger = get_logger()


class FusionExplanationFramework(ABC):
    """
    Framework for setting up and running model explanations using TorchFusion.

    This framework handles data loading, model setup, device configuration,
    explanation generation, and result saving for model explanations.
    """

    def __init__(self, args: FusionArguments, hydra_config: DictConfig) -> None:
        """
        Initialize the TorchfusionExplanationFramework.

        Args:
            args (FusionArguments): The fusion arguments for the framework.
            hydra_config (DictConfig): The Hydra configuration.
        """
        self._args: FusionArguments = args
        self._hydra_config: DictConfig = hydra_config
        self._logger: logging.Logger = get_logger()

        # Initialize class variables
        # Output
        self._output_dir: Optional[str] = None
        self._output_file: Optional[Path] = None

        # Data
        self._train_dataloader: Optional[DataLoader] = None
        self._val_dataloader: Optional[DataLoader] = None
        self._test_dataloader: Optional[DataLoader] = None
        self._test_dataloader_full: Optional[DataLoader] = None
        self._tokenizer: Optional[Any] = None  # Replace Any with specific type if known
        self._dataset_metadata: Optional[Any] = (
            None  # Replace Any with specific type if known
        )
        self._train_baselines = None

        # Model
        self._device: Optional[torch.device] = None
        self._wrapped_model: Optional[ModelForwardWrapper] = None

    @property
    def dataset_labels(self) -> List[str]:
        """
        Get labels from the dataset metadata.

        Returns:
            List[str]: The labels of the dataset.
        """
        return self._dataset_metadata.get_labels()

    def _setup(self) -> str:
        """
        Initialize the framework, including torch setup and logging.

        Returns:
            str: The output directory path.
        """
        # Initialize torch with given arguments
        initialize_torch(
            self._args,
            seed=self._args.general_args.seed,
            deterministic=self._args.general_args.deterministic,
        )

        # Initialize logging directory and tensorboard logger
        output_dir, _ = setup_logging(
            output_dir=self._hydra_config.runtime.output_dir,
            setup_tb_logger=False,
        )

        return output_dir

    def _setup_datamodule(
        self, runtime_config: DictConfig
    ) -> Tuple[
        DataLoader, DataLoader, DataLoader, HuggingfaceTokenizer, FusionDatasetMetaData
    ]:
        """
        Setup the data module, including dataloaders and tokenizer.

        Args:
            runtime_config (DictConfig): The runtime configuration.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader, Any, Any]: The train, validation,
                test dataloaders, tokenizer, and dataset metadata.
        """
        # Load data module based on arguments
        datamodule = load_datamodule_from_args(
            self._args,
            stage=None,
        )

        # Setup batch sampler if needed
        batch_sampler_wrapper = BatchSamplerFactory.create(
            self._args.data_loader_args.train_batch_sampler.name,
            **self._args.data_loader_args.train_batch_sampler.kwargs,
        )

        # Setup custom data collators if required
        tokenizer = datamodule.get_tokenizer_if_available()
        datamodule._collate_fns = self._get_data_collators(tokenizer)

        # Load dataset metadata
        dataset_metadata = datamodule.get_dataset_metadata()

        # Setup dataloaders
        train_dataloader = datamodule.train_dataloader(
            self._args.data_loader_args.per_device_train_batch_size,
            dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_loader_args.pin_memory,
            shuffle_data=self._args.data_loader_args.shuffle_data,
            dataloader_drop_last=self._args.data_loader_args.dataloader_drop_last,
            batch_sampler_wrapper=batch_sampler_wrapper,
        )
        val_dataloader = datamodule.val_dataloader(
            self._args.data_loader_args.per_device_eval_batch_size,
            dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_loader_args.pin_memory,
        )

        self._test_dataloader_full = datamodule.test_dataloader(
            self._args.data_loader_args.per_device_eval_batch_size,  # for test we can use a larger batch
            dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
            pin_memory=self._args.data_loader_args.pin_memory,
        )

        # Setup test_dataloader based on runtime_config
        if runtime_config.start_idx is not None and runtime_config.end_idx is not None:
            test_dataloader = datamodule.test_dataloader_indices(
                runtime_config.start_idx,
                runtime_config.end_idx,
                self._args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_loader_args.pin_memory,
            )
        elif runtime_config.n_test_samples is not None:
            test_dataloader = datamodule.test_dataloader_equally_spaced_samples(
                runtime_config.n_test_samples,
                self._args.data_loader_args.per_device_eval_batch_size,
                dataloader_num_workers=self._args.data_loader_args.dataloader_num_workers,
                pin_memory=self._args.data_loader_args.pin_memory,
            )
        else:
            test_dataloader = self._test_dataloader_full

        # Print transforms before training run just for sanity check
        self._logger.debug("Final sanity check about transforms...")
        print_tf_from_loader(
            train_dataloader, stage=TrainingStage.train, log_level=logging.DEBUG
        )
        print_tf_from_loader(
            val_dataloader, stage=TrainingStage.validation, log_level=logging.DEBUG
        )
        print_tf_from_loader(
            test_dataloader, stage=TrainingStage.test, log_level=logging.DEBUG
        )

        return (
            train_dataloader,
            val_dataloader,
            test_dataloader,
            tokenizer,
            dataset_metadata,
        )

    def _setup_device(self) -> torch.device:
        """
        Setup the device (CPU or GPU) for computation.

        Returns:
            torch.device: The device to use for computation.
        """
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _setup_model(self) -> ModelForwardWrapper:
        """
        Setup the model for evaluation.

        Loads the model, moves it to the device, and sets it to evaluation mode.

        Returns:
            torch.nn.Module: The prepared model.
        """
        fusion_trainer = FusionTrainer(
            args=self._args,
            hydra_config=None,
        )
        model = fusion_trainer._setup_model(
            summarize=True,
            setup_for_train=False,
            strict=False,
            dataset_metadata=self._dataset_metadata,
        )
        model.torch_model.eval()
        model.torch_model.to(self._device)
        model.torch_model.zero_grad()

        return self._wrap_model(model.torch_model)

    def _setup_output_file(self, runtime_config: DictConfig) -> Path:
        """
        Setup the output file path based on the runtime configuration.

        Args:
            runtime_config (DictConfig): The runtime configuration.

        Returns:
            Path: The output file path.
        """
        if runtime_config.start_idx is not None and runtime_config.end_idx is not None:
            output_file = (
                Path(self._output_dir)
                / f"{runtime_config.explanation_method}_{runtime_config.start_idx}_{runtime_config.end_idx}.h5"
            )
        else:
            output_file = (
                Path(self._output_dir) / f"{runtime_config.explanation_method}.h5"
            )
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)
        return output_file

    @abstractmethod
    def _evaluate_model(self) -> None:
        """
        Evaluate the model.

        Must be implemented by the derived class.
        """
        raise NotImplementedError(
            "_evaluate_model must be implemented by the derived tass-specific class."
        )

    @abstractmethod
    def _get_data_collators(
        self,
        tokenizer: Any,
    ) -> CollateFnDict:
        """
        Get data collators for the data module.

        Must be implemented by the derived class.

        Args:
            tokenizer (Any): The tokenizer to use for data collators.

        Returns:
            CollateFnDict: A dictionary of collate functions.
        """
        raise NotImplementedError(
            "_get_data_collators must be implemented by the derived tass-specific class."
        )

    @abstractmethod
    def _setup_train_baselines(
        self, runtime_config: DictConfig
    ) -> Dict[str, torch.Tensor]:
        """
        Setup baselines from the training data.

        Must be implemented by the derived class.
        """
        raise NotImplementedError(
            "_setup_train_baselines must be implemented by the derived tass-specific class."
        )

    @abstractmethod
    def _visualize_explanations(
        self,
        batch: Dict[str, torch.Tensor],
        model_inputs: Dict[str, Any],
        explanations: torch.Tensor | Tuple[torch.Tensor],
    ) -> None:
        pass

    def _wrap_model(self, model: torch.nn.Module) -> ModelForwardWrapper:
        """
        Wrap the model for explanation.

        Must be implemented by the derived class.

        Args:
            model (torch.nn.Module): The model to wrap.
        """
        return ModelForwardWrapper(model)

    def _save_sample_metadata(
        self,
        sample_keys: List[str],
        batch: Mapping[str, torch.Tensor],
        model_outputs: Any,
    ):
        """
        Save sample metadata.

        Must be implemented by the derived class.

        Args:
            sample_keys (List[str]): Unique keys for the samples.
            batch (Mapping[str, torch.Tensor]): The batch of data.
            model_outputs (Any): The model outputs.
        """
        """
        Save sample metadata to an HDF5 file.

        Args:
            sample_keys (List[str]): List of sample keys.
            batch (Mapping[str, torch.Tensor]): Batch of input data.
            model_outputs (Any): Model outputs.
        """
        _, predicted_labels = model_outputs
        with HFIOSingleOutput(
            self._output_file.with_suffix(".metadata.h5")
        ) as hf_sample_data_io:
            for sample_key, sample_index, image_file_path, true_labels in zip(
                sample_keys,
                batch[DataKeys.INDEX],
                batch[DataKeys.IMAGE_FILE_PATH],
                batch[DataKeys.LABEL],
            ):
                hf_sample_data_io.save_attribute(
                    "sample_index", sample_index.item(), sample_key
                )
                hf_sample_data_io.save_attribute(
                    "image_file_path", image_file_path, sample_key
                )
                hf_sample_data_io.save_attribute(
                    "dataset_labels", self.dataset_labels, sample_key
                )
                hf_sample_data_io.save_attribute(
                    "predicted_labels",
                    predicted_labels.detach().cpu().tolist(),
                    sample_key,
                )
                hf_sample_data_io.save_attribute(
                    "expl_target_labels",
                    predicted_labels.detach().cpu().tolist(),
                    sample_key,
                )
                hf_sample_data_io.save_attribute(
                    "true_labels", true_labels.detach().cpu().tolist(), sample_key
                )

    def _verify_depdendent_metrics_computed(self, sample_keys, required_metrics):
        # now check for all dependent metrics if any of them require re-computation
        for metric in required_metrics:
            if metric not in RAW_EXPLANATION_DEPENDENT_METRICS:
                continue

            # see if there is a filx existing for this metric, if no file no metric is computed
            # therefore return false, meaning this metric needs to be computed
            metric_file_path = self._output_file.with_suffix(f".{metric}.h5")
            if not metric_file_path.exists():
                return False

            with HFIOSingleOutput(
                self._output_file.with_suffix(f".{metric}.h5"), mode="r"
            ) as hf_sample_data_io_metric:
                # open the file and see if the metric is already computed for this batch of sample keys
                if not BatchComputeCache(
                    metric_name=metric, hf_sample_data_io=hf_sample_data_io_metric
                ).is_cached_for_sample_keys(sample_keys):
                    # if any of the sample keys are not computed, return false
                    return False
        return True

    def _reduce_explanations(self, explanations) -> None:
        if isinstance(explanations, tuple):
            return tuple(x.sum(-1) for x in explanations)
        else:
            return explanations.sum(-1)

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
                ["default"],
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

    def _compute_and_save_metrics(
        self,
        sample_keys: List[str],
        explainer: FusionExplainer,
        explanations: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        reduced_explanations: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        model_inputs: Dict[str, Any],
        model_outputs: Any,
        runtime_config: DictConfig,
    ):
        """
        Compute explanation-dependent metrics.
        Args:
            sample_keys (List[str]): List of sample keys.
            explainer (TorchFusionExplainer): The explainer to be used.
            explanations (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Explanations for the samples.
            model_inputs (Dict[str, Any]): Inputs to the model.
            model_outputs (Any): Outputs from the model.
            runtime_config (DictConfig): Runtime configuration.
        Returns:
            None
        """

        # now check for all dependent metrics if any of them require re-computation
        _, predicted_labels = model_outputs
        for metric, metric_kwargs in runtime_config.metrics.items():
            if metric not in EXPLANATION_METRICS:
                raise ValueError(f"Invalid metric: {metric}")

            with HFIOSingleOutput(
                self._output_file.with_suffix(f".{metric}.h5")
            ) as hf_sample_data_io:
                metrics_dict = {
                    # axioamtic metrics
                    "completeness": CompletenessBatchComputeCache,
                    "monotonicity_corr_and_non_sens": MonotonicityCorrNonSensitivityBatchComputeCache,
                    # complexity metrics
                    "complexity": ComplexityBatchComputeCache,
                    "effective_complexity": EffectiveComplexityBatchComputeCache,
                    "sparseness": SparsenessBatchComputeCache,
                    # faithfulness metrics
                    "infidelity": InfidelityBatchComputeCache,
                    "faithfulness_corr": FaithfulnessCorrelationBatchComputeCache,
                    "faithfulness_estimate": FaithfulnessEstimateBatchComputeCache,
                    "aopc": AOPCBatchComputeCache,
                    "monotonicity": MonotonicityBatchComputeCache,
                    # robustness metrics
                    "sensitivity": SensitivityBatchComputeCache,
                }

                metric_computer = metrics_dict[metric](
                    hf_sample_data_io, **metric_kwargs
                )

                if metric in ["sensitivity"]:
                    # sensitivity requires the raw explanations
                    metric_scores = metric_computer.compute_and_save(
                        sample_keys=sample_keys,
                        wrapped_model=self._wrapped_model,
                        explainer=explainer,
                        model_inputs=model_inputs,
                        batch_target_labels=predicted_labels,
                    )
                else:
                    metric_scores = metric_computer.compute_and_save(
                        sample_keys=sample_keys,
                        wrapped_model=self._wrapped_model,
                        explanations=(
                            explanations
                            if metric in RAW_EXPLANATION_DEPENDENT_METRICS
                            else reduced_explanations
                        ),
                        model_inputs=model_inputs,
                        batch_target_labels=predicted_labels,
                    )
                logger.debug(f"{metric} scores: {metric_scores}")

    def _construct_model_forward_wrapper(self) -> ModelForwardWrapper:
        return ModelForwardWrapper(self._wrapped_model)

    def _generate_explanations(
        self,
        runtime_config: DictConfig,
    ) -> None:
        """
        Generate explanations for the test dataset.

        Args:
            runtime_config (DictConfig): The runtime configuration.
        """
        if self._test_dataloader is None:
            self._logger.error("Test dataloader is not set up.")
            raise ValueError("Test dataloader is not set up.")

        if self._output_file is None:
            self._logger.error("Output file is not set up.")
            raise ValueError("Output file is not set up.")

        self._logger.info(
            f"Generating attributions for {len(self._test_dataloader)} samples in test dataset "
            f"using attr method = {runtime_config.explanation_method} and saving to output file = {self._output_file}"
        )
        # Initialize attribution method
        explainer = ExplainerFactory.create(
            runtime_config.explanation_method, self._wrapped_model
        )
        pbar = tqdm.tqdm(self._test_dataloader)

        for batch in pbar:
            # Generate unique keys for each sample in the batch
            sample_keys = [
                generate_unique_sample_key(
                    sample_idx,
                    image_file_path,
                )
                for sample_idx, image_file_path in zip(
                    batch[DataKeys.INDEX], batch[DataKeys.IMAGE_FILE_PATH]
                )
            ]

            # Prepare model inputs and outputs
            model_inputs, predicted_scores, predicted_labels = self._wrapped_model(
                batch,
                device=self._device,
            )
            model_outputs = (predicted_scores, predicted_labels)

            # Save sample metadata
            self._save_sample_metadata(sample_keys, batch, model_outputs)

            # set the model to the state required for explanation
            self._wrapped_model.configure(setup_for_explanation=True)

            # compute the explanations
            explanations, is_raw_explanation = self._compute_and_save_explanations(
                sample_keys,
                explainer,
                model_inputs,
                model_outputs,
                runtime_config,
            )

            if is_raw_explanation:
                # compute reduced attributions, this is good for many cases such as to sum attributions over channels
                # or over embedding dimensions or words in a sentence
                reduced_explanations = self._reduce_explanations(explanations)
            else:
                reduced_explanations = explanations

            # visualize the explanations for debugging
            if runtime_config.visualize_explanations:
                self._visualize_explanations(
                    batch=batch,
                    model_inputs=model_inputs,
                    explanations=reduced_explanations,
                )

            # compute and save different explainability metrics
            self._compute_and_save_metrics(
                sample_keys=sample_keys,
                explainer=explainer,
                explanations=explanations,
                reduced_explanations=reduced_explanations,
                model_inputs=model_inputs,
                model_outputs=model_outputs,
                runtime_config=runtime_config,
            )

            # reset the model to its original state
            self._wrapped_model.configure(setup_for_explanation=False)

    def run(self, runtime_config: DictConfig) -> None:
        """
        Run the explanation framework with the given runtime configuration.

        This involves setting up logging, data modules, device, model,
        optionally evaluating the model, and generating explanations.

        Args:
            runtime_config (DictConfig): The runtime configuration.
        """
        # Setup logging
        self._output_dir = self._setup()

        # Setup dataset related components
        (
            self._train_dataloader,
            self._val_dataloader,
            self._test_dataloader,
            self._tokenizer,
            self._dataset_metadata,
        ) = self._setup_datamodule(runtime_config)

        # Setup device and model
        self._device = self._setup_device()
        self._wrapped_model = self._setup_model()
        self._model_forward_wrapper = self._construct_model_forward_wrapper()

        if runtime_config.test_model:
            self._evaluate_model()

        # Setup the output file
        self._output_file = self._setup_output_file(runtime_config)

        # setup training baselines that are used for computing attributions in some methods like deepshap
        self._train_baselines = self._setup_train_baselines(runtime_config)

        # Generate explanations
        self._generate_explanations(
            runtime_config,
        )
