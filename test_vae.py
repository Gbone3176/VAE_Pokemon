import torch
from model import VAE
from utils import data_utils
import matplotlib.pyplot as plt
import numpy as np


def save_image_tensor(image_tensor, title, path):
    """Render a CHW tensor as an RGB image and save it."""
    plt.figure()
    plt.title(title)
    plt.imshow(image_tensor.permute(1, 2, 0).numpy())
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ==============Hpyer-Parameters==================
ckpt_path = "output_dir_lr0.0005_epoch10000_latent16_hidden1024/checkpoint_9600.pt" # path to checkpoint
img_index = 1  # index of the encoded image in dataset for grid visualization
step = 8       #  expected step for grid visualizing
grid_bound = 4.5  # value of the rangeing for grid search
selectd_dim = [2, 10] # the index of the two dim that want to apply grid search 

# ==============Two-Image Interpolation Settings==================
img_indices = (0, 1)  # dataset indices for the two anchor images
interp_steps = 20  # number of interpolation samples (including endpoints)
# ===============================================================
# ================================================


# ==============Load Checkpoints==================
state_dict = torch.load(ckpt_path)
# Trace settings from the ckpt
linear_shape = state_dict['encoder.1.weight'].shape  # (1024, 4800) = (1024, 40 * 40 * 3)
input_size = int((linear_shape[1] / 3) ** 0.5)
hidden_size = int(linear_shape[0])
latent_size = state_dict['encoder.9.weight'].shape[0] // 2
print(input_size)
model = VAE(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size)
model.load_state_dict(state_dict=state_dict, strict=True)

assert selectd_dim[0] < latent_size and selectd_dim[1] < latent_size, "selectd_dimesion should smaller than the latent_size"
# ================================================

with torch.no_grad():
    model.eval()
    dataset = data_utils.figure_dataset("./figure")
    assert img_index < len(dataset), "img_index is out of range"
    assert len(img_indices) == 2, "img_indices must contain two entries"
    assert max(img_indices) < len(dataset), "One of img_indices is out of range"
    assert interp_steps >= 2, "interp_steps should be at least 2"

    img = dataset[img_index]
    save_image_tensor(img, "raw input", "visualize_of_input_image.png")

    img_a = dataset[img_indices[0]]
    img_b = dataset[img_indices[1]]
    save_image_tensor(img_a, f"image @ index {img_indices[0]}", f"raw_img_{img_indices[0]}.png")
    save_image_tensor(img_b, f"image @ index {img_indices[1]}", f"raw_img_{img_indices[1]}.png")
    
    # ===Start grid search for the checkpoint with init latent space from the input image===
    img = img.unsqueeze(0) # batching
    z, mu, _  = model.encode(img)
    
    # PyTorch has a deprecated issue for torch.meshgrid, check here: https://pytorch.org/docs/stable/generated/torch.meshgrid.html#torch-meshgrid
    latent_grid = torch.tensor(np.array(np.meshgrid(       # create a gird for VAE grid visualize
        np.linspace(-grid_bound, grid_bound , num=step),
        np.linspace(-grid_bound, grid_bound , num=step),
    )))
    

    # latent_grid_shape ==  2, 10, 10
    latent_grid = latent_grid.reshape(2, step * step)
    latent_grid = latent_grid.permute(1, 0)
    print(latent_grid.shape)
    
    z_tensor = torch.repeat_interleave(mu, step * step, 0) # 100, 16
    
    z_tensor[:, selectd_dim[0]] += latent_grid[:, 0]
    z_tensor[:, selectd_dim[1]] += latent_grid[:, 1]
    
    print(z_tensor.shape)
    z_img = model.decode(z_tensor)  # no batch, may OOM
    print(z_img.shape)
    
    plt.figure(figsize=(20, 20), dpi=300)  # Increase figure size and set dpi for higher resolution
    
    for i in range(step * step):
        plt.subplot(step, step, i + 1)
        plt.imshow(z_img[i].permute(1,2,0).numpy())
        plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)  # Set spacing between subplots to zero
    
    file_name = ckpt_path.replace('/',"-")
    plt.savefig(f"grid_img_{file_name}_step{step}_bound{grid_bound}_dim{selectd_dim}.png")

    # ===Latent interpolation between two images===
    img_pair = torch.stack([img_a, img_b], dim=0)
    _, mu_pair, _ = model.encode(img_pair)
    mu_a = mu_pair[0]
    mu_b = mu_pair[1]

    # Linear interpolation in full latent space
    alphas = torch.linspace(0.0, 1.0, steps=interp_steps).unsqueeze(1)
    interp_latents = (1 - alphas) * mu_a + alphas * mu_b
    decoded_interp = model.decode(interp_latents)

    plt.figure(figsize=(2 * interp_steps, 4), dpi=200)
    for i in range(interp_steps):
        plt.subplot(1, interp_steps, i + 1)
        plt.imshow(decoded_interp[i].permute(1, 2, 0).numpy())
        plt.axis('off')
        plt.title(f"t={i/(interp_steps - 1):.2f}")
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0)
    interp_file = f"latent_interp_{file_name}_idx{img_indices[0]}_{img_indices[1]}_steps{interp_steps}.png"
    plt.savefig(interp_file)
    plt.close()
    
    # ===2D grid interpolation on selected two dimensions===
    # Grid where top-left (0,0) is img_a and bottom-right (step-1, step-1) is img_b
    # Use diagonal position to determine overall interpolation weight
    interp_latents_2d = []
    for i in range(step):
        for j in range(step):
            # Calculate interpolation weight based on diagonal position
            # (0,0) -> alpha=0 (img_a), (step-1, step-1) -> alpha=1 (img_b)
            alpha = (i + j) / (2 * (step - 1)) if step > 1 else 0
            
            # Interpolate all dimensions
            lat = (1 - alpha) * mu_a + alpha * mu_b
            interp_latents_2d.append(lat)
    
    interp_latents_2d = torch.stack(interp_latents_2d, dim=0)
    decoded_interp_2d = model.decode(interp_latents_2d)
    
    # Visualize in grid: top-left is img_a, bottom-right is img_b
    plt.figure(figsize=(20, 20), dpi=300)
    for i in range(step):
        for j in range(step):
            idx = i * step + j
            plt.subplot(step, step, idx + 1)
            plt.imshow(decoded_interp_2d[idx].permute(1, 2, 0).numpy())
            plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    interp_2d_file = f"2d_interp_{file_name}_idx{img_indices[0]}_{img_indices[1]}_dim{selectd_dim}_step{step}.png"
    plt.savefig(interp_2d_file)
    plt.close()
        

