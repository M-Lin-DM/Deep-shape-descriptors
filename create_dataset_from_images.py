import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from utils import center_and_rescale

DATASET_DIR = r"D:\Datasets\VAE_zeroshot\data_full\unprocessed\point clouds"
train_img_dir = r"D:\Datasets\VAE_zeroshot\data_full\unprocessed\train"
train_img_path = Path(train_img_dir)
img_files = list(train_img_path.glob('*.png'))  # will list the entire path


N = 1000  # number of points to sample from the list of extracted pixel coords
pc_list = []
for j, image_path in enumerate(img_files):
    print(j)
    if j < 5000:
        image = Image.open(image_path).convert('1')
        img_arr = np.array(image)

        # print(np.sum(~img_arr))
        # Find the coordinates of all pixels with the value=0
        row_indices, col_indices = np.where(img_arr == 0)
        if len(row_indices) < N:  # removes small shapes and ensures we have a uniform array
            continue

        # Combine row and column indices into a single array of coordinates
        coordinates = np.vstack((row_indices, col_indices)).T
        coordinates = center_and_rescale(coordinates)

        # sample fixed number of points N
        sample_inds = np.random.choice(len(coordinates), size=(N,), replace=False)
        pc_list.append(coordinates[sample_inds])

#         print(len(coordinates[sample_inds]))
#         plt.imshow(img_arr)


np.save(f'{DATASET_DIR}\\pc_list.npy', pc_list, allow_pickle=True)