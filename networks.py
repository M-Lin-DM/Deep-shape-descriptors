import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *

class PDLNet(nn.Module):
    # [ B * N * (3+z) ] -> # [ B * N * 3 ] 
    def __init__(self, size_z, n_points_per_cloud):
        super(PDLNet, self).__init__()
        size_kernel = 1
        size_pad = 0

        self.size_z = size_z
        self.num_point = n_points_per_cloud
        self.conv1 = torch.nn.Conv1d(pc_dim + self.size_z, 128, size_kernel, padding=size_pad)  # note the conv slides across the N (dim=2) axis.
        self.conv2 = torch.nn.Conv1d(128, 32, size_kernel, padding=size_pad)  # NOTE: the use of conv with kernel size=1 is EQUIVALENT to using a shared MLP at all points
        self.conv3 = torch.nn.Conv1d(32, 8, size_kernel, padding=size_pad)
        
        self.conv4 = torch.nn.Conv1d(8 + self.size_z, 128, size_kernel, padding=size_pad)
        self.conv5 = torch.nn.Conv1d(128, 32, size_kernel, padding=size_pad)
        self.conv6 = torch.nn.Conv1d(32, pc_dim, size_kernel, padding=size_pad)
        
        self.ln0 = nn.LayerNorm((self.size_z, n_points_per_cloud))
        self.ln1 = nn.LayerNorm((128, n_points_per_cloud))
        self.ln2 = nn.LayerNorm((32, n_points_per_cloud))
        self.ln3 = nn.LayerNorm((8, n_points_per_cloud))
        self.ln4 = nn.LayerNorm((128, n_points_per_cloud))
        self.ln5 = nn.LayerNorm((32, n_points_per_cloud))
        self.ln6 = nn.LayerNorm((pc_dim, n_points_per_cloud))

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, z):
        # The shape of the input is (batch, channels(ie 3+128), N)
        # x_shape=(batch_size, 3, N)
        # z_shape=(batch_size, latentsize, N)
        z = self.ln0(z)
        x = torch.cat([z, x], 1)
        x = self.dropout(F.relu(self.ln1(self.conv1(x))))
        x = self.dropout(F.relu(self.ln2(self.conv2(x))))
        x = self.dropout(F.relu(self.ln3(self.conv3(x))))
        
        x = torch.cat([z, x], 1)
        x = self.dropout(F.relu(self.ln4(self.conv4(x))))
        x = self.dropout(F.relu(self.ln5(self.conv5(x))))
        x1 = self.dropout(self.conv6(x))
        return x1



