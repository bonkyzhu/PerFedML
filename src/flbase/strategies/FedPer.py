'''
# File: FedPer.py
# Project: strategies
# Created Date: 2021-12-19 4:28
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2022-06-09 7:53
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''
from collections import OrderedDict, Counter
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass

from .FedAvg import FedAvgClient, FedAvgServer
from ..models.CNN import *
from ..models.MLP import *
from ..models.RNN import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch


class FedPerClient(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)


class FedPerServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        self.server_model_state_dict = deepcopy(self.clients_dict[0].get_params())
        self.server_side_client.set_params(self.server_model_state_dict, exclude_keys=set())
        self.exclude_layer_keys = set()
        for key in self.server_model_state_dict:
            for ekey in exclude:
                if ekey in key:
                    self.exclude_layer_keys.add(key)
        # FedPer do not aggregate head as implemented in the FedBaBu's code
        # head_key = [name for name in self.server_side_client.model.state_dict().keys() if 'fc' in name or 'prototype' in name]
        head_key = [name for name in self.server_side_client.model.state_dict().keys() if 'prototype' in name]
        self.exclude_layer_keys.update(head_key)
        if len(self.exclude_layer_keys) > 0:
            print(f"FedPerServer: the following keys will not be aggregate:\n ", self.exclude_layer_keys)
        freeze_layers = []
        for param in self.server_side_client.model.named_parameters():
            if param[1].requires_grad == False:
                freeze_layers.append(param[0])
        if len(freeze_layers) > 0:
            print("FedPerServer: the following layers will not be updated:", freeze_layers)
