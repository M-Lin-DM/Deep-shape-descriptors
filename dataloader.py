import os
import glob
import random
import numpy as np
import tqdm
import torch
import math
from copy import deepcopy
# from scipy.stats import multivariate_normal
from torch.utils.data import Dataset

from config import *


def get_mean_dis(pc):
    center = np.sum(pc)
    mean_dis = np.mean(pc - center)
    print(mean_dis)
    return mean_dis

@torch.no_grad()
def add_gaussian_noise(pc, sigma=0.02):
    # pc [N*3]
    noise = torch.randn_like(pc)
    return pc + noise * sigma


class Flowers(Dataset):
    def __init__(self, root, device, sigma=0.04):
        self.sigma = sigma
        self.device = device
        self.data = np.load(os.path.join(root, DATASET_DIR, 'pc_list.npy'))
        self.indices = range(self.data.shape[0])

    def __getitem__(self, index, is_online=True):
        pc_gt = self.data[index]  # numpy array
        pc_gt = torch.FloatTensor(pc_gt).to(self.device)  # convert to tensor
        pc = add_gaussian_noise(pc_gt, sigma=self.sigma)  # shape=(N, 3)
        pc_gt = pc_gt.transpose(0, 1)  #
        pc = pc.transpose(0, 1)

        return pc, pc_gt, index  # (3,N), (3,N), (1,)

    def __len__(self):
        return len(self.data)
