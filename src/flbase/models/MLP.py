'''
# File: CNNMnist.py
# Project: models
# Created Date: 2021-12-16 5:11
# Author: Yutong Dai yutongdai95@gmail.com
# -----
# Last Modified: 2022-06-09 6:23
# Modified By: Yutong Dai yutongdai95@gmail.com
# 
# This code is published under the MIT License.
# -----
# HISTORY:
# Date      	By 	Comments
# ----------	---	----------------------------------------------------------
'''
from ..model import Model
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(Model):
    def __init__(self, config):
        dim = config['dim']
        num_cls = config['num_classes']
        W = None
        self.normalize = config['normalize']
        self.return_embedding = config['return_embedding']
        super().__init__(config)
        self.fc1 = nn.Linear(2, dim * 8)
        self.fc2 = nn.Linear(dim * 8, dim * 4)
        self.fc3 = nn.Linear(dim * 4, dim * 2)
        self.fc4 = nn.Linear(dim * 2, dim)
        if W is None:
            temp = nn.Linear(dim, num_cls, bias=False).state_dict()['weight']
            self.prototype = nn.Parameter(temp)
        else:
            self.prototype = W

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        feature_embedding = self.fc4(x)
        if self.normalize:
            feature_embedding_norm = torch.norm(feature_embedding, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            feature_embedding = torch.div(feature_embedding, feature_embedding_norm)
            prototype_norm = torch.norm(self.prototype, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            normalized_prototype = torch.div(self.prototype, prototype_norm)
            logits = torch.matmul(feature_embedding, normalized_prototype.T)
        else:
            logits = torch.matmul(feature_embedding, self.prototype.T)

        if self.return_embedding:
            return feature_embedding, logits
        return logits
