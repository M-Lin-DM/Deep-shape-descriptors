import torch
import torch.optim as optim
import numpy as np
from config import *
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import umap
from sklearn.cluster import KMeans


# @torch.no_grad()

def add_gaussian_noise(pc, sigma=0.02):
    # pc [N*3]
    noise = torch.randn_like(pc)
    return pc + noise * sigma


def center_and_rescale(dat):
    # centers points on the origin and makes std in each dim = 1
    means = np.mean(dat, 0)  # this is litterally the center of mass, assuming all particles have equal mass

    dat2 = (dat - means[None, :]) / 250
    return dat2


def save_checkpoint(save_name, model, z, optimizer):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    z_ts = torch.stack(z, dim=0)  # Concatenates a sequence of tensors along a new dimension.
    torch.save(z_ts, save_name + '_latents.pt')
    torch.save(state, save_name + '.pth')
    print('model saved to {}'.format(save_name))


def load_checkpoint(save_name, model, optimizer):
    z_ts = torch.load(save_name + '_latents.pt')
    z_lst = []
    for i in range(z_ts.size()[0]):
        z_lst.append(z_ts[i, :])
    if model is None:
        pass
    else:
        model_CKPT = torch.load(save_name + '.pth')
        model.load_state_dict(model_CKPT['state_dict'])  # load weights into model
        # model.cuda()
        # optimizer = optim.Adam(model.parameters())
        print('loading checkpoint!')
        # optimizer.load_state_dict(model_CKPT['optimizer'])

    return model, z_lst, optimizer


def pc_batch_to_data_matrices_list(pc_batch):
    # Args: pc_batch shape=(batch_size, spatial dims, number of points in cloud)
    data_matricies = []
    for pc in pc_batch:
        dat = pc.squeeze().transpose(0, 1)
        data_matricies.append(dat.cpu().detach().numpy())

    return data_matricies


def contruct_latent_repeat_tensor(shape_batch, latent_indices, latent_vecs, device='cuda', use_noise=True):
    lat_inds = latent_indices.cpu().detach().numpy()
    latent_inputs = torch.ones((lat_inds.shape[0], latent_size), dtype=torch.float32, requires_grad=True).to(
        device)  # ones with shape=(batchsize, latentvectorsize). Notice how requires_grad=True was used since we want the gradient to connect back to the latent_vecs list of params
    i = 0
    for ind in lat_inds:  #
        latent_inputs[i] *= latent_vecs[
            ind]  # extract the latent vectors corresponding to this batch. Why cant you just say latent_inputs = latent_vecs[lats_inds]????
        i += 1

    if use_noise:
        latent_inputs = add_gaussian_noise(latent_inputs, sigma=0.01)

    latent_repeat = latent_inputs.unsqueeze(-1).expand(-1, -1, shape_batch.size()[
        -1])  # shape=(batchsize, latentvectorsize, N)  Any dimension of size 1 can be expanded(ie repeated n times) to an arbitrary value without allocating new memory. Here he expands the singleton dim=2 to be

    return latent_repeat


def embed_tsne(data_matrix_norm, initial_pos=None):
    if initial_pos is not None:
        emb = TSNE(n_components=3, perplexity=30, early_exaggeration=12, n_iter=250, n_iter_without_progress=50,
                   init=initial_pos, n_jobs=-1, metric="euclidean", learning_rate=10).fit_transform(data_matrix_norm)
    else:
        emb = TSNE(n_components=3, perplexity=2, early_exaggeration=2, n_iter=300, n_iter_without_progress=50,
                   init="random", n_jobs=-1, metric="euclidean", learning_rate=10).fit_transform(data_matrix_norm)

    emb = center_and_rescale(emb)
    print('done tsne emb')

    return emb


def scatter3D(emb, colors):
    fig = go.Figure(data=[go.Scatter3d(
        name='training images',
        x=emb[:, 0],
        y=emb[:, 1],
        z=emb[:, 2],
        mode='markers',
        marker=dict(
            size=4, color=colors, symbol='circle')
    )])

    fig.show()


def embed_umap(data_matrix_norm):
    reducer_emb = umap.UMAP(n_neighbors=7,  # <-- make dependent on global metrics in df for greater diversity
                            min_dist=0.1,
                            n_components=3,
                            metric='euclidean')
    emb = reducer_emb.fit_transform(data_matrix_norm)
    emb = center_and_rescale(emb)
    print('done emb')

    return emb


def kmeans_clustering(dat, n_clusters):
    # Instantiate the KMeans algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Fit the algorithm to the data
    kmeans.fit(dat)

    return kmeans, kmeans.labels_, kmeans.cluster_centers_
