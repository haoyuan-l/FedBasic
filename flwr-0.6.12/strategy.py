import flwr as fl
import torch
import collections
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.server.client_manager import ClientManager

class CustomFedNova(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], fl.common.Scalar]:

        if not results:
            return None, {}

        # Compute tau_effective from summation of local client tau: Eqn-6: Section 4.1
        local_tau = [res.metrics["tau"] for _, res in results]
        tau_eff = np.sum(local_tau)

        aggregate_parameters = []

        for _client, res in results:
            params = parameters_to_ndarrays(res.parameters)
            # compute the scale by which to weight each client's gradient
            # res.metrics["local_norm"] contains total number of local update steps
            # for each client
            # res.metrics["ratio"] contains the ratio of client dataset size
            # Below corresponds to Eqn-6: Section 4.1
            scale = tau_eff / float(res.metrics["local_norm"])
            scale *= float(res.metrics["ratio"])

            aggregate_parameters.append((params, scale))

        # Aggregate all client parameters with a weighted average using the scale
        agg_cum_gradient = aggregate(aggregate_parameters)

        # Aggregate custom metrics if aggregation function was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(agg_cum_gradient), metrics_aggregated