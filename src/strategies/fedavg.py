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
import wandb
from .aggregate import aggregate, weighted_loss_avg, save_final_global_model
from .strategy import Strategy
import numpy as np


import torch
from collections import OrderedDict
from model import Net
import json
import os

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
import time

class FedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        num_rounds: int = 200,
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
        test_file_path=None,
        model=Net,
        model_name="Fedavg",
        num_test_clients = 20
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
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_eval = fraction_eval
        self.num_rounds = num_rounds
        self.rounds = -1 # since it calls eval before training has even started thus the counter goes to 0
        self.min_fit_clients = min_fit_clients
        self.min_eval_clients = min_eval_clients
        self.min_available_clients = min_available_clients
        self.eval_fn = eval_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.round=1
        self.test_file_path=test_file_path
        self.name = model_name
        self.model=model
        self.num_test_clients = int(num_test_clients)
        self.t = time.time()
        self.best_loss = 10000000
        self.sampled_users = []

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        #return max(num_clients, self.min_eval_clients), self.min_available_clients
        #print("num_available_clients", num_available_clients)
        return min(10, num_available_clients), 10

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        if isinstance(initial_parameters, list):
            log(WARNING, DEPRECATION_WARNING_INITIAL_PARAMETERS)
            initial_parameters = weights_to_parameters(weights=initial_parameters)
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        self.rounds += 1

        if self.rounds <= 1:
            if self.rounds == 1:
                print("\nDuration of 1. round (including eval):", time.time()-self.t)
                print("The server sleep time should thus be this value + 5ish\n")

            self.t = time.time()

        if self.eval_fn:
            weights=parameters_to_weights(parameters)
            eval_res = self.eval_fn(state_dict=None,
                                    data_folder=self.test_file_path,
                                    parameters=weights,
                                    num_test_clients=self.num_test_clients,
                                    model=self.model,
                                    get_loss=True)

            acc, loss, num_observations  = eval_res
            sum_obs=np.sum(np.array(num_observations))
            test_acc=np.sum(np.array(acc)*np.array(num_observations))/sum_obs
            test_loss=np.sum(np.array(loss)*np.array(num_observations))/sum_obs
            test_acc_var=np.var(np.array(acc))
            test_loss_var=np.var(np.array(loss))
            wandb.log({'round':self.rounds,
                       'mean_global_test_loss':test_loss,
                       'mean_global_test_accuracy':test_acc,
                       'var_global_test_accuracy':test_acc_var,
                       'dist_global_test_accuracy':wandb.Histogram(np.array(acc))})
            if self.rounds == 1:
                print("duration of 1. global eval for {} clients:".format(self.num_test_clients), time.time() - self.t)

            if test_loss < self.best_loss and self.rounds >= int(self.num_rounds/2):
                self.best_loss = test_loss
                self.save_final_global_model(parameters)

        return None

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {'round':rnd}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_eval == 0.0:
            return []

        # Parameters and config
        config = {'round':rnd}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

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

        loss_aggregated = weighted_loss_avg(
            [
                (fit_res.num_examples, fit_res.metrics['loss'])
                for _, fit_res in results
            ]
        )
        
        wandb.log({'round':rnd, 'train_loss_aggregated':loss_aggregated})

        weights_aggregated = aggregate(weights_results)

        # only does something if its the final iteration: rounds == num_rounds.
        # The function counts aswell
        #self.rounds = save_final_global_model(weights_aggregated, self.name, self.rounds, self.num_rounds)
        return weights_to_parameters(weights_aggregated), {}

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        dis_metrics=np.array([
                (evaluate_res.num_examples, 
                evaluate_res.metrics['accuracy'],
                evaluate_res.metrics['ranked_pred'],
                evaluate_res.loss)
                for _, evaluate_res in results
            ])

        var_acc=np.var(dis_metrics[:,1])
        var_loss=np.var(dis_metrics[:,3])
        accuracy_aggregated = weighted_loss_avg(dis_metrics[:,[0,3]])
        loss_aggregated = weighted_loss_avg(dis_metrics[:,[0,1]])
        ranked_pred = weighted_loss_avg(dis_metrics[:,[0,2]])
        wandb.log({'round':rnd,
                   'test_accuracy_aggregated':accuracy_aggregated,
                   'test_loss_aggregated':loss_aggregated,
                   'var_test_accuracy_aggregated':var_acc,
                   'dist_test_accuracy_aggregated':wandb.Histogram(dis_metrics[:,1]),
                   'ranked_pred_test_accuracy':ranked_pred
                   })
        return loss_aggregated, {}

    def save_final_global_model(self, parameters):
        weights = parameters_to_weights(parameters)
        # import sys
        # import os
        #
        # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # sys.path.append(BASE_DIR)

        # import datetime
        # now = datetime.datetime.now()
        # day_hour_min = '{:02d}_{:02d}_{:02d}'.format(now.day, now.hour, now.minute)

        # this could maybe be simplified but i wont bother
        net = self.model()
        params_dict = zip(net.state_dict().keys(), weights)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # this step might not be necessary
        # net.load_state_dict(state_dict, strict=True)
        if "saved_models" not in os.listdir(): os.mkdir("saved_models")

        print("\nAt round:", self.rounds, "with test loss", self.best_loss)

        if len(self.sampled_users) > 0:
            with open("saved_models/" + self.name + "_users" + ".json", "w") as fp:
                json.dump(self.sampled_users, fp)

            print("Saving sampled users into:", "saved_models/" + self.name + "_users" + ".json")

        torch.save(state_dict, "saved_models/" + self.name + "_state_dict" + ".pt")
        print("Saving model at " "saved_models/" + self.name + "_state_dict" + ".pt")
