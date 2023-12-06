DATASET_DIR = r"D:\Datasets\VAE_zeroshot\data_full\unprocessed\point clouds"
sigma = 0.06  # std of noise added to each point. 0.08
batch_size = 8
latent_size = 128  # a large latent size will lead to sparse data manifold in latent vector space, but perhaps better denoising performance. Smaller dim should encourage more connected latent space that can be interpolated.
lr = 0.005
epochs = 10
N = 1000
pc_dim = 2  # dimensionality of the point cloud embedding space
log_interval = 10
sigma_z = 0.0  # sigma of noisd added to the latents (optionally)
CHECKPOINT_DIR = r"C:\Users\MrLin\OneDrive\Documents\Experiments\Deep shape descriptor\SAVED MODELS"
model_filename = 'MODEL_znoise'
latent_filename = 'LATENTS_znoise'
bottleneck_dim = 16


hyperparams = dict(epochs=epochs, batch_size=batch_size, lr=lr, sigma=sigma, latent_size=latent_size, sigma_z=sigma_z, model_filename=model_filename, latent_filename=latent_filename)
