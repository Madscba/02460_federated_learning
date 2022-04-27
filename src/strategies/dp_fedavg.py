# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: https://arxiv.org/abs/1602.05629
"""

import numpy as np
import wandb
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from strategies.fedavg import FedAvg

from .aggregate import aggregate, weighted_loss_avg, save_final_global_model
from .strategy import Strategy
from privacy_opt import PrivacyAccount
from model import Net

DEPRECATION_WARNING = """
DEPRECATION WARNING: deprecated `eval_fn` return format

    loss, accuracy

move to

    loss, {"accuracy": accuracy}

instead. Note that compatibility with the deprecated return format will be
removed in a future release.
"""

DEPRECATION_WARNING_INITIAL_PARAMETERS = """
DEPRECATION WARNING: deprecated initial parameter type

    flwr.common.Weights (i.e., List[np.ndarray])

will be removed in a future update, move to

    flwr.common.Parameters

instead. Use

    parameters = flwr.common.weights_to_parameters(weights)

to easily transform `Weights` to `Parameters`.
"""

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


class DPFedAvg(FedAvg):
    """Configurable DP FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        num_rounds: int = 0,
        batch_size: int = 8,
        noise_multiplier: float = None,
        noise_scale: float = None,
        max_grad_norm: float = 1.1,
        target_delta: float = None,
        total_num_clients: int = 1000,
        test_file_path=None,
        model=Net,
        model_name="DP_FedAvg"
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        """
        super().__init__(
            model=model,
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            test_file_path=test_file_path)

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.num_rounds = num_rounds
        self.rounds = 0
        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.total_num_clients = total_num_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.batch_size = batch_size
        self.noise_multiplier = noise_multiplier
        self.noise_scale = noise_scale
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.epsilon = 0
        self.round=1
        self.privacy_account = None
        self.name = model_name
        self.model=model

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def set_privacy_account(self, results, rnd):
        num_examples = sum([fit_res.num_examples for _, fit_res in results]) 
        if not self.target_delta:
            self.target_delta = 0.1 * (1 / num_examples)
        C = len([fit_res.num_examples for _, fit_res in results]) 
        sensitivity = self.max_grad_norm / C
        self.noise_scale = self.noise_multiplier / sensitivity
        sample_rate = (self.fraction_fit * self.total_num_clients) / self.total_num_clients
        #steps = int(num_examples / self.batch_size) 
        self.privacy_account = PrivacyAccount(steps=rnd, sample_size=C, sample_rate=sample_rate,
                                              max_grad_norm=self.max_grad_norm, noise_multiplier=self.noise_multiplier,
                                              noise_scale=self.noise_scale, target_delta=self.target_delta)
                                              
    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        self.round=rnd
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results
        weights_results = [
                    (parameters_to_weights(fit_res.parameters), fit_res.num_examples) for client, fit_res in results]
        weights_aggregated = aggregate(weights_results)
        self.set_privacy_account(results=results,rnd=rnd)
        if self.noise_scale:
            sigma = self.privacy_account.noise_multiplier
            for w in weights_aggregated:
                w += np.random.normal(loc=0, scale=sigma, size=np.shape(w))
            self.epsilon = self.privacy_account.get_privacy_spent()
            wandb.log({'round': rnd, "epsilon": self.epsilon})

        loss_aggregated = weighted_loss_avg(
            [
                (fit_res.num_examples, fit_res.metrics['loss'])
                for _, fit_res in results
            ]
        )
        wandb.log({'round': rnd, 'train_loss_aggregated': loss_aggregated})
        self.rounds = save_final_global_model(weights_aggregated, self.name, self.rounds, self.num_rounds)
        return weights_to_parameters(weights_aggregated), {}