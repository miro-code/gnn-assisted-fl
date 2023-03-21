## Imports
import csv
import numbers
import os
import random
import pickle
from collections import OrderedDict, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import *


import flwr as fl
import ray
import gdown
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Metrics, Config, GetPropertiesIns, GetPropertiesRes, MetricsAggregationFn
from flwr.common.parameter import ndarrays_to_parameters,parameters_to_ndarrays
from flwr.common.typing import NDArrays, Parameters, Scalar
from flwr.server import ServerConfig
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from PIL import Image
from PIL.Image import Image as ImageType
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from tqdm import tqdm
from enum import IntEnum
from datetime import datetime,timezone
import json
from flwr.server.strategy.fedavg import FedAvg

from gflower.clients.client import FlowerRayClient, get_flower_client_generator
import gflower.servers.server
from gflower.servers.server import Server, NewHistory as History
import gflower.clients.client_manager as client_manager
from gflower.clients.client_manager import CustomClientManager
from gflower.strategies.strategy import DeterministicSampleFedAvg as FedAvgM
from gflower.clients.client_utils import (
    get_network_generator_cnn,
    get_model_parameters,
    aggregate_weighted_average,
    get_federated_evaluation_function,
    get_default_test_config,
    get_default_train_config,
)

from gflower.strategies.fedavg_angle import FedAvgAngle 
from gflower.strategies.gcn_avg import GCNAvg
# Add new seeds here for easy autocomplete
class Seeds(IntEnum):
    DEFAULT = 1337

np.random.seed(Seeds.DEFAULT)
random.seed(Seeds.DEFAULT)
torch.manual_seed(Seeds.DEFAULT)
torch.backends.cudnn.benchmark = False # type: ignore
torch.backends.cudnn.deterministic = True # type: ignore

def convert(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32): return int(o)  
    if isinstance(o, np.float32) or isinstance(o, np.float64): return float(o)  
    raise TypeError

def fit_client_seeded(client, params, conf, seed=Seeds.DEFAULT, **kwargs):
    """Wrapper to always seed client training."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    return client.fit(params, conf, **kwargs)

PathType = Optional[Union[Path, str]]

def get_device() -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    return device



home_dir = Path(os.getcwd())
devices_info_dir: Path = home_dir / "device_info"
statistical_utility: Path = home_dir / "statistical_utility.csv"
dataset_dir: Path = home_dir / "femnist"
data_dir: Path = dataset_dir / "data"
centralized_partition: Path = dataset_dir / "client_data_mappings" / "centralized"
centralized_mapping: Path = dataset_dir / "client_data_mappings" / "centralized" / "0"
federated_partition: Path = dataset_dir / "client_data_mappings" / "fed_natural"
(home_dir / "histories").mkdir(exist_ok=True,parents=True)


def save_history(hist, name):
  time = int(datetime.now(timezone.utc).timestamp())
  with open(home_dir / "histories" / f"hist_{time}_{name}.json", "w", encoding="utf-8") as f:
            json.dump(hist.__dict__, f, ensure_ascii=False, indent=4, default=convert)

def start_seeded_simulation(
    client_fn,
    num_clients,
    client_resources,
    server,
    config,
    strategy,
    name: str,
    seed: int = Seeds.DEFAULT,
):
    """Wrapper to always seed client selection."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    hist =  fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources=client_resources,
        server=server,
        config=config,
        strategy=strategy,
    )
    save_history(hist, name)
    ray.shutdown()
    return hist


np.random.seed(Seeds.DEFAULT)
random.seed(Seeds.DEFAULT)
torch.manual_seed(Seeds.DEFAULT)
network_generator_cnn = get_network_generator_cnn()
seed_net_cnn = network_generator_cnn()
seed_model_cnn_params: NDArrays = get_model_parameters(seed_net_cnn)

federated_evaluation_function = get_federated_evaluation_function(
    data_dir=data_dir,
    centralized_mapping=centralized_mapping,
    device=get_device(),
    batch_size=get_default_test_config()["batch_size"],
    num_workers=get_default_test_config()["num_workers"],
    model_generator=network_generator_cnn,
    criterion=nn.CrossEntropyLoss(),
)

client_generator = get_flower_client_generator(network_generator_cnn, data_dir, federated_partition)

default_parameters: Dict = {
    "train_config": get_default_train_config(),
    "test_config": get_default_test_config(),
    "num_total_clients": 3229,
    "num_clients_per_round": 4,
    "num_evaluate_clients": 1,
    "accept_failures": False,
    "min_fit_clients": 2,
    "min_available_clients": 2,
    "initial_parameters": ndarrays_to_parameters(seed_model_cnn_params),
    "client_generator": client_generator,
    "seed": Seeds.DEFAULT,
    "num_rounds": 10,
    "strategy": FedAvg,
    "fed_eval": True,
}


def run_fixed_fl(
    parameters=default_parameters,
    **kwargs
):
    parameters: Dict = {**parameters, **kwargs}

    on_fit_config_fn: Callable[[int], Dict[str, Scalar]] = lambda cid: parameters[
        "train_config"
    ]
    on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] = lambda cid: parameters["test_config"]  # type: ignore

    fraction_fit: float = (
        float(parameters["num_clients_per_round"]) / parameters["num_total_clients"]
    )
    fraction_evaluate: float = (
        float(parameters["num_evaluate_clients"]) / parameters["num_total_clients"]
    )

    client_resources = {
        "num_gpus": 1.0 / parameters["num_clients_per_round"] if get_device() == "cuda" else 0.0,  # maximum amount of resources that a client can take
        "num_cpus": 1,
    }

    strategy = parameters["strategy"](
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=parameters["min_fit_clients"],
        min_available_clients=parameters["min_available_clients"],
        on_fit_config_fn=on_fit_config_fn,
        on_evaluate_config_fn=on_evaluate_config_fn,
        initial_parameters=parameters["initial_parameters"],
        accept_failures=parameters["accept_failures"],
        evaluate_fn=federated_evaluation_function
        if parameters["fed_eval"] is True
        else None,
        fit_metrics_aggregation_fn=aggregate_weighted_average,
        evaluate_metrics_aggregation_fn=aggregate_weighted_average,
        
    )
    client_manager = SimpleClientManager()
    server = Server(
        client_manager=client_manager,
        strategy=strategy,
    )
    return start_seeded_simulation(
        client_fn=parameters["client_generator"],
        num_clients=parameters["num_total_clients"],
        client_resources=client_resources,
        server=server,
        config=ServerConfig(num_rounds=parameters["num_rounds"]),
        strategy=strategy,
        seed=parameters["seed"],
        name=f"fixed_fl_run"
    )

run_fixed_fl(num_clients_per_round = 10)