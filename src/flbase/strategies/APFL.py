'''
# File: APFL.py
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
from ..server import Server
from ..client import Client
from ..models.CNN import *
from ..models.MLP import *
from ..models.RNN import *
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
from .FedAvg import FedAvgClient, FedAvgServer

class APFLClient(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()
        self.local_model = deepcopy(self.model)
        self.global_model = deepcopy(self.model)
    
    def weighted_aggreagte(self):
        _alpha = self.client_config['apfl_alpha'] 
        _ans_model = deepcopy(self.model)
        weight_keys = list(_ans_model.state_dict().keys())
        _ans_model_sd = OrderedDict()

        local_sd = self.local_model.state_dict()
        global_sd = self.global_model.state_dict()

        for key in weight_keys:
            _ans_model_sd[key] =  _alpha * local_sd[key] + (1-_alpha) * global_sd[key]

        _ans_model.load_state_dict(_ans_model_sd)
        return _ans_model
    

    def training(self, round, num_epochs):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        setup_seed(round)
        # train mode
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")

        # train global model
        self.model = nn.DataParallel(self.model).cuda()
        optimizer = setup_optimizer(self.model, self.client_config, round)
        epoches_iter = tqdm(range(num_epochs))
        for i in epoches_iter:
            epoches_iter.set_description("Processing Client %s" % self.cid)
            self.model.train()
            for j, (x, y) in enumerate(self.trainloader):
                # x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                yhat = self.model.forward(x)
                loss = self.criterion(yhat, y)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()

        ## save trained global model
        self.global_model = deepcopy(self.model).module.to(self.device)
        self.model = nn.DataParallel(deepcopy(self.local_model)).cuda()
        
        # train local model
        optimizer = setup_optimizer(self.model, self.client_config, round)
        epoches_iter = tqdm(range(num_epochs))
        for i in epoches_iter:
            epoches_iter.set_description("Processing Client %s" % self.cid)
            self.model.train()
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
                # x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                yhat = self.model.forward(x)
                loss = self.criterion(yhat, y)
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize

            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)

        ## save trained local model
        self.local_model = deepcopy(self.model).module.to(self.device)
        self.model = self.model.module.to(self.device)
        ## APFL aggregate
        self.local_model = self.weighted_aggreagte()
        ## save local model to self.model for testing
        self.model = deepcopy(self.local_model)

        self.new_state_dict = self.global_model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq


class APFLServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)