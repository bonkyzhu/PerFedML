'''
# File: FedROD.py
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
from ..strategies.FedAvg import FedAvgServer
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
import torch.nn as nn


class FedRODClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        self.is_on_server = False
        # prepare for balanced softmax loss
        temp = [self.count_by_class[cls] if cls in self.count_by_class.keys() else 1e-12 for cls in range(client_config['num_classes'])]
        count_by_class_full = torch.tensor(temp).to(self.device)
        self.sample_per_class = count_by_class_full / torch.sum(count_by_class_full)

        self._initialize_model()

    def set_on_server(self):
        self.is_on_server = True

    def _initialize_model(self):
        # parse the model from config file
        self.model = eval(self.client_config['model'])(self.client_config).to(self.device)
        # separate base and head
        g_head = deepcopy(self.model.prototype)
        self.p_head = deepcopy(g_head)
        self.model.prototype = None
        self.model = ModelWrapper(self.model, g_head, self.model.config)
        # this is needed if the criterion has stateful tensors.
        self.criterion = self.criterion.to(self.device)
        # self.trainloader_list = [(x.to(self.device), y.to(self.device)) for (x,y) in list(self.trainloader)]

    def training(self, round, num_epochs):
        """
            Note that in order to use the latest server side model the `set_params` method should be called before `training` method.
        """
        setup_seed(round)
        # train mode
        self.model.train()
        # tracking stats
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")
        
        self.model = nn.DataParallel(self.model).cuda()
        optimizer = setup_optimizer(self.model, self.client_config, round)
        optimizer_lr = optimizer.param_groups[0]['lr']

        p_head_optimizer = torch.optim.SGD([self.p_head], lr=optimizer_lr)
        epoches_iter = tqdm(range(num_epochs))
        for i in epoches_iter:
            epoches_iter.set_description("Processing Client %s" % self.cid)
            epoch_loss, correct = 0.0, 0
            for j, (x, y) in enumerate(self.trainloader):
                # forward pass
                # x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
                embedding, out_g = self.model(x, return_embedding=True)
                loss_bsm = balanced_softmax_loss(y, out_g, self.sample_per_class)
                self.model.zero_grad(set_to_none=True)
                loss_bsm.backward()
                torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.model.parameters()), max_norm=10)
                optimizer.step()

                out_p = torch.matmul(embedding.detach().to(self.device), self.p_head.T)
                out = out_p + out_g.detach().to(self.device)
                loss = self.criterion(out, y.to(self.device))
                p_head_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=[self.p_head], max_norm=10)
                p_head_optimizer.step()

                predicted = out.data.max(1)[1]
                correct += predicted.eq(y.data.to(self.device)).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize
            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        
        self.model = self.model.module.to(self.device)
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq

    def upload(self):
        return self.new_state_dict

    def testing(self, round, testloader=None):
        self.model.eval()
        if testloader is None:
            testloader = self.testloader
        test_count_per_class = Counter(testloader.dataset.targets.numpy())
        all_classes_sorted = sorted(test_count_per_class.keys())
        test_count_per_class = torch.tensor([test_count_per_class[cls] * 1.0 for cls in all_classes_sorted])
        num_classes = len(all_classes_sorted)
        test_correct_per_class = torch.tensor([0] * num_classes)

        weight_per_class_dict = {'uniform': torch.tensor([1.0] * num_classes),
                                 'validclass': torch.tensor([0.0] * num_classes),
                                 'labeldist': torch.tensor([0.0] * num_classes)}
        for cls in self.label_dist.keys():
            weight_per_class_dict['labeldist'][cls] = self.label_dist[cls]
            weight_per_class_dict['validclass'][cls] = 1.0
        # start testing
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                # forward pass
                x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                feature_embedding, out_g = self.model(x, return_embedding=True)
                if self.is_on_server:
                    out = out_g
                else:
                    out_p = torch.matmul(feature_embedding, self.p_head.T)
                    out = out_p + out_g
                # stats
                predicted = out.data.max(1)[1]
                classes_shown_in_this_batch = torch.unique(y).cpu().numpy()
                for cls in classes_shown_in_this_batch:
                    test_correct_per_class[cls] += ((predicted == y) * (y == cls)).sum().item()
        acc_by_critertia_dict = {}
        for k in weight_per_class_dict.keys():
            acc_by_critertia_dict[k] = (((weight_per_class_dict[k] * test_correct_per_class).sum()) /
                                        ((weight_per_class_dict[k] * test_count_per_class).sum())).item()

        self.test_acc_dict[round] = {'acc_by_criteria': acc_by_critertia_dict,
                                     'correct_per_class': test_correct_per_class,
                                     'weight_per_class': weight_per_class_dict}


class FedRODServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        # set correct status so that it use the global model to perform evaluation
        self.server_side_client.set_on_server()

# https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification


def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    # If Only One Classes, Use Only Logits 
    if (max(sample_per_class)-1<1e-5).item() is True:
        return F.cross_entropy(input=logits, target=labels, reduction=reduction)
    
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
