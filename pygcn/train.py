from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.models import GCN
class NetworkInstance:
    def __init__(self, **kwargs):
        self.params = {}
        for key, value in kwargs.items():
            #print(key)
            self.params[key] = value
        self.make_model()
    def make_model(self):


        np.random.seed(self.params["seed"])
        torch.cuda.manual_seed(self.params["seed"])

        # Model and optimizer
        self.model = GCN(nfeat=self.params["nfeat"],
                    nhid=self.params["n_hidden"],
                    nclass=self.params["nclass"],
                    dropout=self.params["dropout"])
        self.optimizer = optim.Adam(self.model.parameters(),
                            lr=10 ** self.params["lr"], weight_decay=10 ** self.params["weight_decay"])
        self.model.cuda()