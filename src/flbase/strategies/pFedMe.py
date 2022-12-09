'''
# File: pFedMe.py
# Project: flbase
# Created Date: 2021-12-11 2:08
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
from ..optimizers.fedoptimizer import pFedMeOptimizer
from ..strategies.FedAvg import FedAvgServer
from ..utils import setup_optimizer, linear_combination_state_dict, setup_seed
from ...utils import autoassign, save_to_pkl, access_last_added_element
import time
import torch
import torch.nn as nn


class pFedMeClient(Client):
    def __init__(self, criterion, trainset, testset, client_config, cid, device, **kwargs):
        super().__init__(criterion, trainset, testset,
                         client_config, cid, device, **kwargs)
        # proximal term coefficients
        self.lambda_pfedme = client_config['lambda_pfedme']
        self.is_on_server = False
        self._initialize_model()
        self.ptrain_loss_dict = OrderedDict()
        self.ptrain_acc_dict = OrderedDict()
        self.local_steps_pfedme = client_config['local_steps_pfedme']

    def set_on_server(self):
        self.is_on_server = True
        self.personalized_model = None

    def _initialize_model(self):
        """
            self.model holds a local copy of the server side model
        """
        # parse the model from config file
        self.model = eval(self.client_config['model'])(self.client_config).to(self.device)
        self.personalized_model = deepcopy(self.model)
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
        self.personalized_model.train()
        # tracking stats
        self.num_rounds_particiapted += 1
        loss_seq = []
        acc_seq = []
        if self.trainloader is None:
            raise ValueError("No trainloader is provided!")
        optimizer = setup_optimizer(self.personalized_model,
                                    self.client_config, round)
        lgmodel_lr = max(optimizer.param_groups[0]['lr'] * 0.5, 0.01)
        pmodel_lr = max(optimizer.param_groups[0]['lr'] * 0.5, 0.01)
        optimizer = None
        # here intentionally use self.personalized_model.parameters() instead of self.model.parameters()
        # to differentiate the impl of pFedMe, i.e., every round the theta is intailized to the latest server side model
        # which is counter-intuitive to me
        for param, new_param in zip(self.personalized_model.parameters(), self.model.parameters()):
            param.data = new_param.data.clone()

        self.personalized_model = nn.DataParallel(self.personalized_model).cuda()
        personalized_model_optimizer = pFedMeOptimizer(params=filter(lambda p: p.requires_grad, self.personalized_model.parameters()),
                                                       lr=pmodel_lr, lamda=self.lambda_pfedme, mu=1e-5)
        

        epoches_iter = tqdm(range(num_epochs))
        for i in epoches_iter:
            epoches_iter.set_description("Processing Client %s" % self.cid)
            epoch_loss, correct = 0.0, 0
            # for j, (x, y) in enumerate(self.trainloader):
            for j, (x, y) in enumerate(self.trainloader):
                # x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
                x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

                # approximately solve the pmodel theta
                for i_local in range(self.local_steps_pfedme):
                    self.personalized_model.zero_grad(set_to_none=True)
                    personalized_model_optimizer.zero_grad()
                    yhat = self.personalized_model.forward(x)
                    loss = self.criterion(yhat, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=filter(lambda p: p.requires_grad, self.personalized_model.parameters()), max_norm=10)
                    # finding aproximate theta
                    # self.model.parameters() here is the latest W
                    # actual update personalized_model
                    personalized_params = personalized_model_optimizer.step(list(self.model.parameters()), self.device)

                # stats for personalized model derived from the moreau envelope problem
                predicted = yhat.data.max(1)[1]
                correct += predicted.eq(y.data).sum().item()
                epoch_loss += loss.item() * x.shape[0]  # rescale to bacthsize

                # update w after finding aproximate theta
                with torch.no_grad():
                    for theta, w in zip(personalized_params, self.model.parameters()):
                        w = w.to(self.device)
                        w.data = w.data - self.lambda_pfedme * lgmodel_lr * (w.data - theta.data.to(self.device))

            epoch_loss /= len(self.trainloader.dataset)
            epoch_accuracy = correct / len(self.trainloader.dataset)
            loss_seq.append(epoch_loss)
            acc_seq.append(epoch_accuracy)
        
        self.personalized_model = self.personalized_model.module.to(self.device)
        self.new_state_dict = self.model.state_dict()
        self.train_loss_dict[round] = loss_seq
        self.train_acc_dict[round] = acc_seq

    def upload(self):
        # only update global model
        return self.new_state_dict

    def testing(self, round, testloader=None):
        if self.is_on_server:
            self.model.eval()
            net = self.model
        else:
            self.personalized_model.eval()
            net = self.personalized_model
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
                yhat = net.forward(x)
                # stats
                predicted = yhat.data.max(1)[1]
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


class pFedMeServer(FedAvgServer):
    def __init__(self, server_config, clients_dict, exclude, **kwargs):
        super().__init__(server_config, clients_dict, exclude, **kwargs)
        # set correct status so that it use the global model to perform evaluation
        self.server_side_client.set_on_server()

    def run(self, **kwargs):
        self.run_id = kwargs["run_id"]
        if self.server_config['use_tqdm']:
            round_iterator = tqdm(range(self.rounds + 1, self.server_config['num_rounds'] + 1), desc="Round Progress")
        else:
            round_iterator = range(self.rounds + 1, self.server_config['num_rounds'] + 1)
        # round index begin with 1
        for r in round_iterator:
            setup_seed(r)
            selected_indices = self.select_clients(self.server_config['participate_ratio'])
            if self.server_config['drop_ratio'] > 0:
                # mimic the stragler issues; simply drop them
                self.active_clients_indicies = np.random.choice(selected_indices, int(
                    len(selected_indices) * (1 - self.server_config['drop_ratio'])), replace=False)
            else:
                self.active_clients_indicies = selected_indices
            # active clients download weights from the server
            tqdm.write(f"Round:{r} - Active clients:{self.active_clients_indicies}:")
            from datetime import datetime
            timestamp = datetime.now().strftime("%m.%d  %H:%M")
            tqdm.write(f"Starting Time: {timestamp}")
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                client.set_params(self.server_model_state_dict, self.exclude_layer_keys)

            # clients perform local training
            train_start = time.time()
            client_uploads = []
            for cid in self.active_clients_indicies:
                client = self.clients_dict[cid]
                # update both global and local models
                client.training(r, client.client_config['num_epochs'])
                client_uploads.append(client.upload())
            
            train_time = time.time() - train_start
            print(f" Training time:{train_time:.3f} seconds")
            # collect training stats
            # average train loss and acc over active clients, where each client uses the latest local models
            self.collect_stats(stage="train", round=r, active_only=True)

            # get new server model
            # agg_start = time.time()
            self.aggregate(client_uploads, round=r)
            # agg_time = time.time() - agg_start
            # print(f" Aggregation time:{agg_time:.3f} seconds")
            # collect testing stats
            if (r - 1) % self.server_config['test_every'] == 0:
                test_start = time.time()
                self.testing(round=r, active_only=True)
                test_time = time.time() - test_start
                print(f" Testing time:{test_time:.3f} seconds")
                self.collect_stats(stage="test", round=r, active_only=True)
                print(" avg_test_acc:", self.gfl_test_acc_dict[r]['acc_by_criteria'])
                print(" pfl_avg_test_acc:", self.average_pfl_test_acc_dict[r])
                if len(self.gfl_test_acc_dict) >= 2:
                    current_key = r
                    if self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform'] > best_test_acc:
                        best_test_acc = self.gfl_test_acc_dict[current_key]['acc_by_criteria']['uniform']
                        self.server_model_state_dict_best_so_far = deepcopy(self.server_model_state_dict)
                        tqdm.write(f" Best test accuracy:{float(best_test_acc):5.3f}. Best server model is updatded and saved at {kwargs['filename']}!")
                        if 'filename' in kwargs:
                            torch.save(self.server_model_state_dict_best_so_far, 
                                        f"{kwargs['filename']}.{self.run_id}")
                else:
                    best_test_acc = self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
            # wandb monitoring
            if kwargs['use_wandb']:
                stats = {"avg_train_loss": self.average_train_loss_dict[r],
                         "avg_train_acc": self.average_train_acc_dict[r],
                         "gfl_test_acc_uniform": self.gfl_test_acc_dict[r]['acc_by_criteria']['uniform']
                         }

                for criteria in self.average_pfl_test_acc_dict[r].keys():
                    stats[f'pfl_test_acc_{criteria}'] = self.average_pfl_test_acc_dict[r][criteria]

                wandb.log(stats)


    def save(self, filename, keep_clients_model=False):
        if not keep_clients_model:
            for client in self.clients_dict.values():
                client.model = None
                client.personalized_model = None
                client.trainloader = None
                client.trainset = None
                client.new_state_dict = None

        self.server_side_client.trainloader = None
        self.server_side_client.trainset = None
        self.server_side_client.testloader = None
        self.server_side_client.testset = None
        save_to_pkl(self, filename, self.run_id)