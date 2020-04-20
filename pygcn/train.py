from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

from pygcn.utils import load_data, accuracy
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

        # Load data
        # self.adj, self.features, self.labels, self.idx_train, self.idx_val, self.idx_test = load_data()
        # adjT = torch.transpose(adj, 0, 1)

        # Model and optimizer
        self.model = GCN(nfeat=self.params["nfeat"],
                    nhid=self.params["n_hidden"],
                    nclass=self.params["nclass"],
                    dropout=self.params["dropout"])
        self.optimizer = optim.Adam(self.model.parameters(),
                            lr=10 ** self.params["lr"], weight_decay=10 ** self.params["weight_decay"])
        self.model.cuda()

#     def train(self, epoch):
#         t = time.time()
#         self.model.train()
#         self.optimizer.zero_grad()
#         output = self.model(features, adj)
#         loss_train = F.nll_loss(output[idx_train], labels[idx_train])
#         acc_train = accuracy(output[idx_train], labels[idx_train])
#         loss_train.backward()
#         optimizer.step()

#         if not args.fastmode:
#             # Evaluate validation set performance separately,
#             # deactivates dropout during validation run.
#             model.eval()
#             output = model(features, adj)

#         loss_val = F.nll_loss(output[idx_val], labels[idx_val])
#         acc_val = accuracy(output[idx_val], labels[idx_val])
#         print('Epoch: {:04d}'.format(epoch+1),
#             'loss_train: {:.4f}'.format(loss_train.item()),
#             'acc_train: {:.4f}'.format(acc_train.item()),
#             'loss_val: {:.4f}'.format(loss_val.item()),
#             'acc_val: {:.4f}'.format(acc_val.item()),
#             'time: {:.4f}s'.format(time.time() - t))


# def test():
#     model.eval()
#     output = model(features, adj)
#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:",
#           "loss= {:.4f}".format(loss_test.item()),
#           "accuracy= {:.4f}".format(acc_test.item()))

def do():
    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    test()
