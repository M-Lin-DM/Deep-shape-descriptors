from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import sys

sys.path.append('../')

from chamfer import ChamferDistance
from networks import PDLNet


class DeepLatent(nn.Module):
    def __init__(self, latent_length, n_points_per_cloud=1024, chamfer_weight=0.1):
        super(DeepLatent, self).__init__()
        self.latent_length = latent_length
        self.pdl_net = PDLNet(latent_length, n_points_per_cloud)
        self.chamfer_dist = ChamferDistance()
        self.L2_dist = nn.MSELoss()
        self.chamfer_weight = chamfer_weight

    def forward(self, pc, pc_gt, latent):
        # self.pc = deepcopy(pc)  # this prevents the gradient from reaching back to the input tensor. But is it equivalent to using requires_grad=False?
        # self.pc_gt = deepcopy(pc_gt)

        predicted_noise = self.pdl_net(pc, latent)  # note that the first arg is not used!
        pc_est = pc - predicted_noise
        loss = self.compute_loss(pc_est, pc_gt)
        return loss, pc_est

    def compute_loss(self, pc_est, pc_gt):
        loss_chamfer = self.chamfer_dist(pc_gt, pc_est)
        loss_L2 = self.L2_dist(pc_gt, pc_est)
        loss = self.chamfer_weight * loss_chamfer + (1 - self.chamfer_weight)*loss_L2
        return loss, loss_chamfer, loss_L2
