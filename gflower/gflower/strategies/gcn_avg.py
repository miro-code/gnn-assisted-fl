from flwr.server.strategy.fedavg import FedAvg
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.server.strategy import Strategy

class GCNAvg(FedAvg):
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
    ) -> None:
        super().__init__(
            fraction_fit =fraction_fit,
            fraction_evaluate =fraction_evaluate,
            min_fit_clients =min_fit_clients,
            min_evaluate_clients =min_evaluate_clients,
            min_available_clients =min_available_clients,
            evaluate_fn =evaluate_fn,
            on_fit_config_fn =on_fit_config_fn,
            on_evaluate_config_fn =on_evaluate_config_fn,
            accept_failures =accept_failures,
            initial_parameters =initial_parameters,
            fit_metrics_aggregation_fn =fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn =evaluate_metrics_aggregation_fn,
        )
        self.current_global_model = parameters_to_ndarrays(initial_parameters)
        self.angles = {}

    def initialize_parameters(self, client_manager: ClientManager
                                ) -> Optional[Parameters]:
        """initialize the angles and adjacency matrix and then call the default FedAvg implementation of initialize_parameters"""
        num_total_clients = client_manager.num_available()
        self.adjacency_matrix = []
        for i in range(num_total_clients):
            connections = []
            for j in range(num_total_clients):
                connections.append(False)
            self.adjacency_matrix.append(connections)
        return super().initialize_parameters(client_manager = client_manager)

    def _get_angle(self, parameters1, parameters2):
        """
        Returns the angle between two np vectors

        parameters1 : ndarray
        parameters2: ndarray

        Returns
        -------
        float 
            angle
        """
        parameters1 = np.concatenate([a.flatten() for a in parameters1]) 
        parameters2 = np.concatenate([a.flatten() for a in parameters2]) 

        p1_normed = parameters1 / np.linalg.norm(parameters1)
        p2_normed = parameters2 / np.linalg.norm(parameters2)
        return np.arccos(np.clip(np.dot(p1_normed, p2_normed), -1.0, 1.0))

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager: ClientManager
                      ) -> List[Tuple[ClientProxy, FitIns]]:
        """stores the global model and then calls the original FedAvg configure_fit
        """
        self.current_global_model = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round=server_round, parameters=parameters, client_manager=client_manager)

    def _get_update(self, original_model : List[np.ndarray], updated_model : List[np.ndarray]) -> List[np.ndarray]:
        """ extracts model update given the original and the new model
        """
        return [layer_new - layer_old for layer_old, layer_new in zip(original_model, updated_model)]

    def _reset_adjacency_matrix(self):
        """ 
        Sets all elements of the adjacency matrix to False
        """
        for i in range(len(self.adjacency_matrix)):
            for j in range(len(self.adjacency_matrix[i])):
                self.adjacency_matrix[i][j] = False

    def update_adjacency_matrix(self):
        """
        Updates the adjacency matrix based on the angles between the client updates
        two clients are defined as connected if their average angle over different rounds is smaller than the median
        """
        self._reset_adjacency_matrix()
        angle_list = list(self.angles.items())
        angle_list.sort(key = lambda x : np.mean(x[1]))
        for i in range(len(angle_list)//2):
            c1_str, c2_str = angle_list[i][0]
            try:
                c1 = int(c1_str)
                c2 = int(c2_str)
            except:
                raise ValueError("Client ids must be integers (with datatype str)")
            self.adjacency_matrix[c1][c2] = True
            self.adjacency_matrix[c2][c1] = True

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        #store all the pairwise angles between the updates
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                angle = self._get_angle(self._get_update(self.current_global_model, parameters_to_ndarrays(results[i][1].parameters)), self._get_update(self.current_global_model,parameters_to_ndarrays(results[j][1].parameters)))
                cid_i = results[i][0].cid
                cid_j = results[j][0].cid
                key = (min(cid_i, cid_j), max(cid_i, cid_j))
                existing_angles = self.angles.get(key, [])
                existing_angles.append(angle)
                self.angles[key] = existing_angles
        self.update_adjacency_matrix()

        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.current_global_model = parameters_aggregated
        return parameters_aggregated, metrics_aggregated