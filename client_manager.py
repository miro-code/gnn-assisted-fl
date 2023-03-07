#!/usr/bin/env python
# -*-coding:utf-8 -*-

# @File    :   client.py
# @Time    :   2023/01/21 11:36:46
# @Author  :   Alexandru-Andrei Iacob
# @Contact :   aai30@cam.ac.uk
# @Author  :   Lorenzo Sani
# @Contact :   ls985@cam.ac.uk, lollonasi97@gmail.com
# @Version :   1.0
# @License :   (C)Copyright 2023, Alexandru-Andrei Iacob, Lorenzo Sani
# @Desc    :   None

from typing import Optional, List
import random

from logging import INFO, WARNING

from flwr.common.logger import log
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class CustomClientManager(SimpleClientManager):
    """
    Client manager that samples the same clients every time
    """
    def __init__(self, criterion: Criterion, seed: int) -> None:
        super().__init__()
        self.criterion = criterion
        self.seed = seed

    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        seed : int = 1337
    ) -> List[ClientProxy]:
        random.seed(seed)
        return super().sample(num_clients, min_num_clients, criterion)