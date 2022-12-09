'''
# File: FedDyn.py
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

class FedDynClient(FedAvgClient):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self._initialize_model()
        # initialize feddyn term
        self.feddyn_linear_term = deepcopy(self.model)
        tmp_sd = {}; model_sd = self.model.state_dict()
        for key in model_sd.keys():
            tmp_sd[key] = torch.zeros_like(model_sd[key])
        self.feddyn_linear_term.load_state_dict(tmp_sd)
    
    def _add_prox_term(self):
        for _param, _init_param in zip(
            self.model.module.parameters(), self.global_model.parameters()
        ):
            if _param.grad is not None:
                _param.grad.data.add_(
                    (_param.data - _init_param.data.to(_param.device)) * self.client_config['feddyn_alpha'] 
                )
    
    def _add_linear_term(self):
        for _param, linear_param in zip(
            self.model.module.parameters(), self.feddyn_linear_term.parameters()
        ):
            if _param.grad is not None:
                _param.grad.data.add_(-linear_param.to(_param.device))
    
    def add_feddyn_grad_term(self):
        self._add_linear_term()
        self._add_prox_term()
    
    def update_feddyn_linear_term(self):
        '''
        feddyn linear term is theta L, update after local training
        '''
        prox_loss = {}
        local_sd = self.model.state_dict()
        glo_sd = self.global_model.state_dict()
        alpha = self.client_config['feddyn_alpha'] 

        for _layer_name in local_sd:
            prox_loss[_layer_name] = (local_sd[_layer_name] - glo_sd[_layer_name]) * -alpha
            
        # calculate feddyn theta L
        feddyn_linear_term_sd = {
            _layer_name: self.feddyn_linear_term.state_dict()[_layer_name] + prox_loss[_layer_name]
            for _layer_name in self.feddyn_linear_term.state_dict()
        }
        self.feddyn_linear_term.load_state_dict(feddyn_linear_term_sd)

    def training(self, round, num_epochs):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        setup_seed(round)
        # train mode
        self.global_model = deepcopy(self.model)
        self.model.train()
        # tracking stats
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")
        self.model = nn.DataParallel(self.model).cuda()
        optimizer = setup_optimizer(self.model, self.client_config, round)
        

        # training starts
        epoches_iter = tqdm(range(num_epochs))
        for i in epoches_iter:
            epoches_iter.set_description("Processing Client %s" % self.cid)
            # setup_seed(2022)
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
                # forward pass
                # x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                yhat = self.model.forward(x)
                loss = self.criterion(yhat, y)
                # backward pass
                # model.zero_grad safer and memory-efficient
                self.model.zero_grad(set_to_none=True)
                loss.backward()
                self.add_feddyn_grad_term()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()
                # stats
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
                
            
            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)

        self.model = self.model.module.to(self.device)
        self.update_feddyn_linear_term()
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq


class FedDynServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        # initialize feddyn_h
        model_template = self.clients_dict[0].model
        self.feddyn_h_sd = {}; model_sd = model_template.state_dict()
        for key in model_sd.keys():
            self.feddyn_h_sd[key] = torch.zeros_like(model_sd[key])
        self.fedavg_model_sd = self.server_model_state_dict
    
    def aggregate(self, client_uploads, round):
        server_lr = self.server_config['learning_rate'] * (self.server_config['lr_decay_per_round'] ** (round - 1))
        num_participants = len(client_uploads)
        update_direction_state_dict = None
        exclude_layer_keys = self.exclude_layer_keys
        with torch.no_grad():
            for idx, client_state_dict in enumerate(client_uploads):
                client_update = linear_combination_state_dict(client_state_dict,
                                                              self.server_model_state_dict,
                                                              1.0,
                                                              -1.0,
                                                              exclude=exclude_layer_keys
                                                              )
                if idx == 0:
                    update_direction_state_dict = client_update
                else:
                    update_direction_state_dict = linear_combination_state_dict(update_direction_state_dict,
                                                                                client_update,
                                                                                1.0,
                                                                                1.0,
                                                                                exclude=exclude_layer_keys
                                                                                )
            # new global model
            _alpha = self.server_config['feddyn_alpha'] 
            self.feddyn_h_sd = linear_combination_state_dict(self.feddyn_h_sd, 
                                                        update_direction_state_dict,
                                                        1.0,
                                                        - _alpha / num_participants,
                                                        exclude=exclude_layer_keys
                                                        )
            self.fedavg_model_sd = linear_combination_state_dict(self.fedavg_model_sd,
                                                            update_direction_state_dict,
                                                            1.0,
                                                            server_lr / num_participants,
                                                            exclude=exclude_layer_keys
                                                            )
            self.server_model_state_dict = linear_combination_state_dict(self.fedavg_model_sd,
                                                            self.feddyn_h_sd,
                                                            1.0,
                                                            - (1 / _alpha),
                                                            exclude=exclude_layer_keys
                                                            )