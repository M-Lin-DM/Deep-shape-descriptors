import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from dataloader import *
from deeplatent import *
from networks import *
from utils import *
from config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = Flowers(DATASET_DIR, device, sigma)  # returns an entire point cloud [N, 3] as the training instance
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
shape_batch, shape_gt_batch, latent_indices = next(iter(loader))

print(f"shape of model input: {shape_batch.shape}")
# checkpoint_dir = config.save_dir

num_total_instance = len(dataset)
num_batch = len(loader)

print(f"num_batch {num_batch}")

model = DeepLatent(latent_length=latent_size, n_points_per_cloud=N, chamfer_weight=0.1)

# initialize all latent vectors in the dataset
latent_vecs = []
for i in range(len(dataset)):
    vec = (torch.ones(latent_size).normal_(0, 0.9).to(device))  # draw values from normal distribution with mean and std
    vec.requires_grad = True  #vec = torch.nn.Parameter(vec)  # make it a parameter to be optimized
    latent_vecs.append(vec)

optimizer = optim.Adam([
    {
        "params": model.parameters(), "lr": lr,
    },
    {
        "params": latent_vecs, "lr": lr * 0.5,
    }
]
)

# if resume:
#     model, latent_vecs, optimizer = load_checkpoint(os.path.join(checkpoint_dir, 'model_best'), model, optimizer)

model.to(device)
min_loss = float('inf')

for epoch in range(epochs):
    print(f"epoch {epoch}")
    training_loss = 0.0
    model.train()
    for index, (shape_batch, shape_gt_batch, latent_indices) in enumerate(loader):  # in my code I need a step to select a fixed number of sample points if the pcs have different sizes.
        # latent_indices is a batch of indices in the dataset (batch_size,). each index corresponds to one entire point cloud
        # shape_batch: shape=(batch_size, 3, N)
        shape_batch.requires_grad = False
        shape_gt_batch.requires_grad = False

        lat_inds = latent_indices.cpu().detach().numpy()
        latent_inputs = torch.ones((lat_inds.shape[0], latent_size), dtype=torch.float32, requires_grad=True).to(
            device)  # ones with shape=(batchsize, latentvectorsize). Notice how requires_grad=True was used since we want the gradient to connect back to the latent_vecs list of params
        i = 0
        for ind in lat_inds:  #
            latent_inputs[i] *= latent_vecs[ind]  # extract the latent vectors corresponding to this batch. Why cant you just say latent_inputs = latent_vecs[lats_inds]????
            i += 1

        latent_repeat = latent_inputs.unsqueeze(-1).expand(-1, -1, shape_batch.size()[-1])  #  shape=(batchsize, latentvectorsize, N)  Any dimension of size 1 can be expanded(ie repeated n times) to an arbitrary value without allocating new memory. Here he expands the singleton dim=2 to be
        shape_batch = shape_batch.to(device)
        shape_gt_batch = shape_gt_batch.to(device)
        loss, chamfer, l2 = model(shape_batch, shape_gt_batch, latent_repeat)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        training_loss += loss.item()

        print("Epoch:[%d|%d], Batch:[%d|%d]  loss: %f , chamfer: %f, l2: %f" % (
        epoch, epochs, index, num_batch, loss.item() / batch_size, chamfer.item() / batch_size,
        l2.item() / batch_size))

    training_loss_epoch = training_loss / len(dataset)  # loss per training instance

    # if training_loss_epoch < min_loss:
    #     min_loss = training_loss_epoch
    #     print('New best performance! saving')
    #     save_name = os.path.join(CHECKPOINT_DIR, 'model_best')
    #     save_checkpoint(save_name, model, latent_vecs, optimizer)
    #
    # if (epoch + 1) % log_interval == 0:
    #     save_name = os.path.join(CHECKPOINT_DIR, 'model_routine')
    #     save_checkpoint(save_name, model, latent_vecs, optimizer)

save_name = os.path.join(CHECKPOINT_DIR, 'model_best')
save_checkpoint(save_name, model, latent_vecs, optimizer)
