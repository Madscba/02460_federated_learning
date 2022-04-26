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
"""FAIR RESOURCE ALLOCATION IN FEDERATED LEARNING [Li et al., 2020] strategy.

Paper: https://openreview.net/pdf?id=ByexElSYDr
"""


from typing import Callable, Dict, List, Optional, Tuple
import wandb
import numpy as np
from model import Net
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
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate_qffl, weighted_loss_avg, save_final_global_model
from .fedavg import FedAvg
import time


class QFedAvg_manual(FedAvg):
    """Configurable QFedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        num_rounds=200,
        model_name="QFedAvg_manual",
        q_param: float = 0.2,
        qffl_learning_rate: float = 0.1,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 1,
        min_eval_clients: int = 1,
        min_available_clients: int = 1,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        test_file_path=None,
        num_test_clients = 20,
        model = Net
    ) -> None:
        super().__init__(
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
            test_file_path=test_file_path,
            num_test_clients = num_test_clients,
            model_name=model_name+"_"+str(q_param),
            num_rounds=num_rounds,
            model = model


        )
        self.learning_rate = qffl_learning_rate
        self.L = 1/qffl_learning_rate
        self.q_param = q_param
        self.pre_weights: Optional[Weights] = None
        self.eps = 1e-10
        self.model = model
        self.sampled_users = []
        #self.name = model_name+"_"+str(q_param)

    def __repr__(self) -> str:
        # pylint: disable=line-too-long
        rep = f"QffedAvg(learning_rate={self.learning_rate}, "
        rep += f"q_param={self.q_param}, pre_weights={self.pre_weights})"
        return rep

    # def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
    #     """Return the sample size and the required number of available
    #     clients."""
    #     num_clients = int(num_available_clients * self.fraction_fit)
    #     return max(num_clients, self.min_fit_clients), self.min_available_clients
    #
    # def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
    #     """Use a fraction of available clients for evaluation."""
    #     num_clients = int(num_available_clients * self.fraction_eval)
    #     return max(num_clients, self.min_eval_clients), self.min_available_clients

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        weights = parameters_to_weights(parameters)
        self.pre_weights = weights
        parameters = weights_to_parameters(weights)
        config = {"round":rnd}
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
        # Do not configure federated evaluation if fraction_eval is 0
        if self.fraction_eval == 0.0:
            return []

        # Parameters and config
        config = {"round":rnd}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        # Convert results

        def norm_grad_squared(list_of_grads: List[Weights]) -> float:
            # input: nested gradients
            # output: square of the L-2 norm
            n = 0.0
            for grad in list_of_grads:
                n += np.sum(np.square(grad))
            return n



        if self.pre_weights is None:
            raise Exception("QffedAvg pre_weights are None in aggregate_fit")

        weights_prev = self.pre_weights
        # eval_result = self.evaluate(weights_to_parameters(weights_prev))
        # if eval_result is not None:
        #     loss, _ = eval_result

        #t = time.time()

        ds = [0.0 for _ in range(len(weights_prev))]
        hs = 0.0

        train_losses = []
        for _, params in results:
            loss = params.metrics.get("loss_prior_to_training", None)
            train_losses.append(params.metrics['loss'])
            self.sampled_users.append(params.metrics['user'])


            if loss == None:
                print("\nplease enable qfed_client = True in client_main\n")
                raise ValueError


            weights_new = parameters_to_weights(params.parameters)

            weight_diff = [
                           (weight_prev - weight_new) * self.L for
                            weight_prev,  weight_new in zip(weights_prev, weights_new)
                          ]

            q_objective = np.float_power((loss + self.eps), self.q_param)
            q_objective_minus1 = np.float_power((loss + self.eps), (self.q_param-1))

            ds = [d + q_objective * grad for d, grad in zip(ds, weight_diff)]

            hs += (
                   self.q_param *
                   q_objective_minus1 *
                   norm_grad_squared(weight_diff) +
                   self.L *
                   q_objective
                  )

        weights_aggregated = [weight_prev - d/hs for weight_prev, d in zip(weights_prev, ds)]
        wandb.log({'round': self.rounds, 'train_loss_var': np.var(np.array(train_losses))})

        # safe the model at the final round and keep track of the number of
        #self.rounds = save_final_global_model(weights_aggregated, self.name, self.rounds, self.num_rounds)
        #print("qfed_manual agg time:", time.time() - t)
        return weights_to_parameters(weights_aggregated), {}

    # def aggregate_evaluate(
    #     self,
    #     rnd: int,
    #     results: List[Tuple[ClientProxy, EvaluateRes]],
    #     failures: List[BaseException],
    # ) -> Tuple[Optional[float], Dict[str, Scalar]]:
    #     """Aggregate evaluation losses using weighted average."""
    #     if not results:
    #         return None, {}
    #     # Do not aggregate if there are failures and failures are not accepted
    #     if not self.accept_failures and failures:
    #         return None, {}
    #     return (
    #         weighted_loss_avg(
    #             [
    #                 (evaluate_res.num_examples, evaluate_res.loss)
    #                 for _, evaluate_res in results
    #             ]
    #         ),
    #         {},
    #     )
    #
