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
"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import List, Tuple

import numpy as np

from flwr.common import Weights


def aggregate(results: List[Tuple[Weights, int]]) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: Weights = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime


def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    weights: Weights, deltas: List[Weights], hs_fll: List[Weights]
) -> Weights:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_weights = [(u - v) * 1.0 for u, v in zip(weights, updates)]
    return new_weights


def save_final_global_model(weights_aggregated, name, rounds, num_rounds):
    rounds += 1
    if rounds == num_rounds:
        import sys
        import os

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.append(BASE_DIR)
        import torch
        from collections import OrderedDict
        from model import Net

        import datetime
        now = datetime.datetime.now()
        day_hour_min = '{:02d}_{:02d}_{:02d}'.format(now.day, now.hour, now.minute)

        # this could maybe be simplified but i wont bother
        net = Net()
        params_dict = zip(net.state_dict().keys(), weights_aggregated)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        # this step might not be necessary
        # net.load_state_dict(state_dict, strict=True)
        if "saved_models" not in os.listdir(): os.mkdir("saved_models")
        torch.save(state_dict, "saved_models/" + name + "_state_dict_" + day_hour_min + ".pt")
        print("Saving model at " "saved_models/" + name + "_state_dict_" + day_hour_min + ".pt")
        torch.save(state_dict, "saved_models/" + name + "_state_dict" + ".pt")
        print("Saving model at " "saved_models/" + name + "_state_dict" + ".pt")

    return rounds
