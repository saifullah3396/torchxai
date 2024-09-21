"""
This module contains the `document_classification_explainer` script.

The `document_classification_explainer` script is used to explain document classifications using the `DocumentClassificationExplainer` class.

Usage:
    python document_classification_explainer.py [OPTIONS]

Options:
    -c, --config TEXT  Path to the configuration file. (default: config.yaml)
    --help             Show this message and exit.

Example:
    python document_classification_explainer.py -c config.yaml
    See: scripts/explainers/document_classification.sh for an example of how to run the script.
"""

from __future__ import annotations

import logging
from typing import Type

import hydra
import ignite.distributed as idist
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torchfusion.core.args.args import FusionArguments
from torchfusion.core.utilities.dataclasses.dacite_wrapper import from_dict
from torchfusion.core.utilities.logging import get_logger

from torchxai.image_classification.image_classification_explanation_framework import (
    ImageClassificationExplanationFramework,
)  # noqa

logger = get_logger()


def main(
    runtime_config: DictConfig,
    hydra_config: DictConfig,
    data_class: Type[FusionArguments] = FusionArguments,
) -> None:
    """
    Entry point function for the document classification explainer.
    Args:
        cfg (DictConfig): The configuration object.
        hydra_config (DictConfig): The Hydra configuration object.
        data_class (Type[FusionArguments], optional): The data class type. Defaults to FusionArguments.
    Returns:
        None
    """

    # setup logger
    logger = get_logger(hydra_config=hydra_config)

    # get the torch fusion arguments from the yaml config file
    args: FusionArguments = from_dict(
        data_class=data_class, data=OmegaConf.to_object(runtime_config)["args"]
    )

    # log the arguments
    logger.info("Starting torchfusion testing script with arguments:")
    logger.info(args)

    # run the document classification explainer
    try:
        explanation_framework = ImageClassificationExplanationFramework(
            args, hydra_config
        )
        return explanation_framework.run(runtime_config)
    except Exception as e:
        logging.exception(e)


@hydra.main(version_base=None, config_name="hydra")
def app(cfg: DictConfig) -> None:
    """
    Run the document classification explainer.
    Args:
        cfg (DictConfig): The configuration for the explainer.
    Returns:
        None
    """
    # get hydra config
    hydra_config = HydraConfig.get()

    # train and evaluate the model
    main(cfg, hydra_config)

    # wait for all processes to complete before exiting
    idist.barrier()


if __name__ == "__main__":
    app()
