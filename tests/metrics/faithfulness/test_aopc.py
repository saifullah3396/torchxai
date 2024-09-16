import logging
from logging import getLogger
from typing import Any, Optional, cast

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import Attribution
from torch import Tensor
from torch.nn import Module

from tests.helpers.basic import (
    assertAllTensorsAreAlmostEqualWithNan,
    assertTensorAlmostEqual,
)
from tests.metrics.base import MetricTestsBase
from torchxai.metrics._utils.common import _tuple_tensors_to_tensors
from torchxai.metrics.faithfulness.aopc import aopc

logging.basicConfig(level=logging.INFO)
logger = getLogger(__name__)


class Test(MetricTestsBase):
    def test_basic_single(self) -> None:
        aopc_desc_per_run = []
        aopc_asc_per_run = []
        aopc_rand_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            aopcs_desc, aopcs_asc, aopcs_rand = self.basic_model_assert(
                **self.basic_single_setup(),
                expected_desc=torch.tensor([[0.0000, 0.5000, 0.6667]]),
                expected_asc=torch.tensor([[0.0000, -0.5000, 0.0000]]),
                expected_rand=torch.tensor([[0.0000, 0.2000, 0.4667]]),
                expected_features=3,  # total features are 2 but aopc returns features + 1 since the fwd with no perturbation is also included
                max_examples_per_batch=max_examples_per_batch,
            )
            aopc_desc_per_run.append(aopcs_desc)
            aopc_asc_per_run.append(aopcs_asc)
            aopc_rand_per_run.append(aopcs_rand)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_desc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_asc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_rand_per_run]
        )

    def test_basic_batch(self) -> None:
        aopc_desc_per_run = []
        aopc_asc_per_run = []
        aopc_rand_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            aopcs_desc, aopcs_asc, aopcs_rand = self.basic_model_assert(
                **self.basic_batch_setup(),
                expected_desc=torch.tensor([[0.0000, 0.5000, 0.6667]] * 3),
                expected_asc=torch.tensor([[0.0000, -0.5000, 0.0000]] * 3),
                expected_rand=torch.tensor([[0.0000, 0.2000, 0.4667]] * 3),
                expected_features=3,  # total features are 2 but aopc returns features + 1 since the fwd with no perturbation is also included
                max_examples_per_batch=max_examples_per_batch,
            )
            aopc_desc_per_run.append(aopcs_desc)
            aopc_asc_per_run.append(aopcs_asc)
            aopc_rand_per_run.append(aopcs_rand)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_desc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_asc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_rand_per_run]
        )

    def test_basic_additional_forward_args1(self) -> None:
        aopc_desc_per_run = []
        aopc_asc_per_run = []
        aopc_rand_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            aopcs_desc, aopcs_asc, aopcs_rand = self.basic_model_assert(
                **self.basic_additional_forward_args_setup(),
                expected_desc=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                expected_asc=torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
                expected_rand=torch.tensor(
                    [[0.0000, 0.0000, -0.0167, -0.0500, -0.0800, -0.0750, -0.0643]]
                ),
                expected_features=7,  # total features are 6 but aopc returns features + 1 since the fwd with no perturbation is also included
                max_examples_per_batch=max_examples_per_batch,
            )
            aopc_desc_per_run.append(aopcs_desc)
            aopc_asc_per_run.append(aopcs_asc)
            aopc_rand_per_run.append(aopcs_rand)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_desc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_asc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_rand_per_run]
        )

    def test_classification_convnet_multi_targets(self) -> None:
        aopc_desc_per_run = []
        aopc_asc_per_run = []
        aopc_rand_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            aopcs_desc, aopcs_asc, aopcs_rand = self.basic_model_assert(
                **self.classification_convnet_multi_targets_setup(),
                expected_desc=torch.tensor(
                    [
                        [
                            0.0000,
                            7.5000,
                            12.6667,
                            22.0000,
                            31.0000,
                            37.0000,
                            41.2857,
                            44.5000,
                            47.0000,
                            49.0000,
                            50.6364,
                            52.0000,
                            53.1538,
                            54.1429,
                            55.0000,
                            55.7500,
                            56.4118,
                        ]
                    ]
                    * 20
                ),
                expected_asc=torch.tensor(
                    [
                        [
                            0.0000,
                            0.5000,
                            2.0000,
                            3.7500,
                            5.6000,
                            6.8333,
                            7.7143,
                            9.8750,
                            12.0000,
                            13.7000,
                            16.1818,
                            19.2500,
                            22.9231,
                            26.0714,
                            28.8000,
                            31.1875,
                            33.2941,
                        ]
                    ]
                    * 20
                ),
                expected_rand=torch.tensor(
                    [
                        [
                            0.0000,
                            3.1500,
                            5.4000,
                            7.6750,
                            9.6800,
                            12.5333,
                            16.1000,
                            19.3625,
                            22.8000,
                            26.2400,
                            29.5727,
                            32.5833,
                            35.2308,
                            37.5000,
                            39.4667,
                            41.1875,
                            42.7059,
                        ]
                    ]
                    * 20
                ),
                expected_features=17,  # total features are 16 but aopc returns features + 1 since the fwd with no perturbation is also included
                max_examples_per_batch=max_examples_per_batch,
                delta=1e-2,
            )
            aopc_desc_per_run.append(aopcs_desc)
            aopc_asc_per_run.append(aopcs_asc)
            aopc_rand_per_run.append(aopcs_rand)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_desc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_asc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_rand_per_run]
        )

    def test_classification_tpl_target(self) -> None:
        aopc_desc_per_run = []
        aopc_asc_per_run = []
        aopc_rand_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            aopcs_desc, aopcs_asc, aopcs_rand = self.basic_model_assert(
                **self.classification_tpl_target_setup(),
                expected_desc=torch.tensor(
                    [
                        [0.0000, 44.0000, 74.6667, 96.0000],
                        [0.0000, 96.0000, 181.3333, 256.0000],
                        [0.0000, 144.0000, 277.3333, 400.0000],
                        [0.0000, 192.0000, 373.3333, 544.0000],
                    ]
                ),
                expected_asc=torch.tensor(
                    [
                        [0.0000, 16.0000, 40.0000, 70.0000],
                        [0.0000, 64.0000, 138.6667, 224.0000],
                        [0.0000, 112.0000, 234.6667, 368.0000],
                        [0.0000, 160.0000, 330.6667, 512.0000],
                    ]
                ),
                expected_rand=torch.tensor(
                    [
                        [0.0000, 28.0000, 53.6000, 80.2000],
                        [0.0000, 76.8000, 154.6667, 236.0000],
                        [0.0000, 124.8000, 250.6667, 380.0000],
                        [0.0000, 172.8000, 346.6667, 524.0000],
                    ]
                ),
                expected_features=4,  # total features are 3 but aopc returns features + 1 since the fwd with no perturbation is also included
                max_examples_per_batch=max_examples_per_batch,
                delta=1.0e-3,
            )
            aopc_desc_per_run.append(aopcs_desc)
            aopc_asc_per_run.append(aopcs_asc)
            aopc_rand_per_run.append(aopcs_rand)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_desc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_asc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_rand_per_run]
        )

    def test_classification_tpl_target_w_baseline_perturb(self) -> None:
        aopc_desc_per_run = []
        aopc_asc_per_run = []
        aopc_rand_per_run = []
        for max_examples_per_batch in [
            1,
            None,
            40,
        ]:
            aopcs_desc, aopcs_asc, aopcs_rand = self.basic_model_assert(
                **self.classification_tpl_target_w_baseline_setup(),
                expected_desc=torch.tensor(
                    [
                        [0.0000, 32.0000, 50.6667, 60.0000],
                        [0.0000, 80.0000, 149.3333, 208.0000],
                        [0.0000, 128.0000, 245.3333, 352.0000],
                        [0.0000, 176.0000, 341.3333, 496.0000],
                    ]
                ),
                expected_asc=torch.tensor(
                    [
                        [0.0000, 0.0000, 10.6667, 30.0000],
                        [0.0000, 48.0000, 106.6667, 176.0000],
                        [0.0000, 96.0000, 202.6667, 320.0000],
                        [0.0000, 144.0000, 298.6667, 464.0000],
                    ]
                ),
                expected_rand=torch.tensor(
                    [
                        [0.0000, 12.8000, 26.1333, 41.6000],
                        [0.0000, 60.8000, 122.6667, 188.0000],
                        [0.0000, 108.8000, 218.6667, 332.0000],
                        [0.0000, 156.8000, 314.6667, 476.0000],
                    ]
                ),
                expected_features=4,  # total features are 3 but aopc returns features + 1 since the fwd with no perturbation is also included
                max_examples_per_batch=max_examples_per_batch,
                delta=1.0e-3,
            )
            aopc_desc_per_run.append(aopcs_desc)
            aopc_asc_per_run.append(aopcs_asc)
            aopc_rand_per_run.append(aopcs_rand)

        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_desc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_asc_per_run]
        )
        assertAllTensorsAreAlmostEqualWithNan(
            self, [x.float() for x in aopc_rand_per_run]
        )

    def basic_model_assert(
        self,
        expected_desc: Tensor,
        expected_asc: Tensor,
        expected_rand: Tensor,
        expected_features: int,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attribution_fn: Attribution,
        feature_masks: TensorOrTupleOfTensorsGeneric = None,
        baselines: BaselineType = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        max_examples_per_batch: int = None,
        multiply_by_inputs: bool = False,
        delta: float = 1e-4,
    ) -> Tensor:
        attributions = attribution_fn.attribute(
            inputs,
            additional_forward_args=additional_forward_args,
            target=target,
            baselines=baselines,
        )
        if multiply_by_inputs:
            attributions = cast(
                TensorOrTupleOfTensorsGeneric,
                tuple(attr / input for input, attr in zip(inputs, attributions)),
            )
        score = self.aopc_assert(
            expected_desc=expected_desc,
            expected_asc=expected_asc,
            expected_rand=expected_rand,
            expected_features=expected_features,
            model=model,
            inputs=inputs,
            attributions=attributions,
            baselines=baselines,
            feature_masks=feature_masks,
            additional_forward_args=additional_forward_args,
            target=target,
            max_examples_per_batch=max_examples_per_batch,
            delta=delta,
        )
        return score

    def aopc_assert(
        self,
        expected_desc: Tensor,
        expected_asc: Tensor,
        expected_rand: Tensor,
        expected_features: int,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        attributions: TensorOrTupleOfTensorsGeneric,
        baselines: Optional[BaselineType] = None,
        feature_masks: TensorOrTupleOfTensorsGeneric = None,
        additional_forward_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        max_examples_per_batch: int = None,
        delta: float = 1e-4,
    ) -> Tensor:
        aopcs_desc, aopcs_asc, aopcs_rand = aopc(
            forward_func=model,
            inputs=inputs,
            attributions=attributions,
            baselines=baselines,
            feature_masks=feature_masks,
            additional_forward_args=additional_forward_args,
            target=target,
            max_features_processed_per_example=max_examples_per_batch,
            seed=42,  # without generator the aopc random for each same input in batch will be different
        )
        attributions, _ = _tuple_tensors_to_tensors(attributions)
        for x in [aopcs_desc, aopcs_asc, aopcs_rand]:
            # match the batch size
            self.assertEqual(x.shape[0], attributions.shape[0])
            self.assertEqual(
                x.shape[1], expected_features
            )  # match the number of features
        assertTensorAlmostEqual(
            self, aopcs_desc.float(), expected_desc.float(), delta=delta
        )
        assertTensorAlmostEqual(
            self, aopcs_asc.float(), expected_asc.float(), delta=delta
        )
        assertTensorAlmostEqual(
            self, aopcs_rand.float(), expected_rand.float(), delta=delta
        )
        return aopcs_desc, aopcs_asc, aopcs_rand
