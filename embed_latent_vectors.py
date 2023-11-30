import numpy as np
import matplotlib.pyplot as plt

from dataloader import *
from deeplatent import *
from utils import *
from config import *

load_file = latent_filename
device = "cuda" if torch.cuda.is_available() else "cpu"


save_name = os.path.join(CHECKPOINT_DIR, load_file)
z_ts = torch.load(save_name + '_latents.pt').cpu().detach().numpy()
print(z_ts.shape)

emb = embed_tsne(z_ts, initial_pos=None)
# emb = embed_umap(z_ts)

kmeans, labels, cluster_centers = kmeans_clustering(z_ts, 350)
scatter3D(emb[:, :3], labels)

dataset = Flowers(DATASET_DIR, device, sigma)  # returns an entire point cloud [N, 3] as the training instance
# shape_batch, shape_gt_batch, latent_indices = next(iter(dataset))

for c in range(10):
    cluster_ind = np.where(labels == c)[0]  # numpy array of indices of data points sharing a certain cluster ID
    # print(len(cluster_ind))

    pc, pc_gt, index = dataset[cluster_ind]  # NOTE: for some reason, this transposes the last 2 dims. shapes= List[(N,2)]
    pc_gt = pc_gt.cpu().detach().numpy()  # shape=(num points, cluster size, 2) Not sure why..
    pc_gt = pc_gt.transpose(1, 0, 2)  # shape=(batch, #points in cloud, 2)
    print(pc_gt.shape)

    num_images = 6  # number of images to plot
    plt.figure(figsize=(15 * num_images, 15))
    plt.axis('off')

    for j, dat in enumerate(pc_gt[:num_images]):
        # print(dat.shape)
        plt.subplot(1, num_images, j+1)
        plt.scatter(dat[:, 0], dat[:, 1], color='black', s=4)
        plt.axis('off')
    #
    # plt.show()
    plt.savefig(f"C:\\Users\\MrLin\\OneDrive\\Documents\\Experiments\\Deep shape descriptor\\fig\\cluster_{c}.png",
                bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close()