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

# ==============Four-Image Interpolation Settings==================
# Four corner images: top-left, top-right, bottom-left, bottom-right
# img_indices_4 = (0, 1, 3, 2)  # dataset indices for the four corner images
img_indices_4 = (9, 10, 12, 11)  # dataset indices for the four corner images
grid_steps = 10  # grid resolution for 2D interpolation
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
    assert len(img_indices_4) == 4, "img_indices_4 must contain four entries"
    assert max(img_indices_4) < len(dataset), "One of img_indices_4 is out of range"
    assert grid_steps >= 2, "grid_steps should be at least 2"

    # Load four corner images
    img_tl = dataset[img_indices_4[0]]  # top-left
    img_tr = dataset[img_indices_4[1]]  # top-right
    img_bl = dataset[img_indices_4[2]]  # bottom-left
    img_br = dataset[img_indices_4[3]]  # bottom-right
    
    # Save the four corner images
    save_image_tensor(img_tl, f"Top-Left (idx {img_indices_4[0]})", f"vis/exp2-PrarmsInterpolation/corner_tl_{img_indices_4[0]}.png")
    save_image_tensor(img_tr, f"Top-Right (idx {img_indices_4[1]})", f"vis/exp2-PrarmsInterpolation/corner_tr_{img_indices_4[1]}.png")
    save_image_tensor(img_bl, f"Bottom-Left (idx {img_indices_4[2]})", f"vis/exp2-PrarmsInterpolation/corner_bl_{img_indices_4[2]}.png")
    save_image_tensor(img_br, f"Bottom-Right (idx {img_indices_4[3]})", f"vis/exp2-PrarmsInterpolation/corner_br_{img_indices_4[3]}.png")
    
    # ===Four-corner bilinear interpolation in latent space===
    # Encode the four corner images
    img_corners = torch.stack([img_tl, img_tr, img_bl, img_br], dim=0)
    _, mu_corners, _ = model.encode(img_corners)
    mu_tl, mu_tr, mu_bl, mu_br = mu_corners[0], mu_corners[1], mu_corners[2], mu_corners[3]
    
    # Create 2D bilinear interpolation grid
    interp_latents_grid = []
    for i in range(grid_steps):
        for j in range(grid_steps):
            # Normalized position in [0, 1]
            alpha_i = i / (grid_steps - 1) if grid_steps > 1 else 0  # vertical (top to bottom)
            alpha_j = j / (grid_steps - 1) if grid_steps > 1 else 0  # horizontal (left to right)
            
            # Bilinear interpolation:
            # top edge: interpolate between top-left and top-right
            lat_top = (1 - alpha_j) * mu_tl + alpha_j * mu_tr
            # bottom edge: interpolate between bottom-left and bottom-right
            lat_bottom = (1 - alpha_j) * mu_bl + alpha_j * mu_br
            # final: interpolate between top and bottom
            lat = (1 - alpha_i) * lat_top + alpha_i * lat_bottom
            
            interp_latents_grid.append(lat)
    
    interp_latents_grid = torch.stack(interp_latents_grid, dim=0)
    decoded_grid = model.decode(interp_latents_grid)
    
    # Visualize the grid: corners should match the four input images
    plt.figure(figsize=(20, 20), dpi=300)
    for i in range(grid_steps):
        for j in range(grid_steps):
            idx = i * grid_steps + j
            plt.subplot(grid_steps, grid_steps, idx + 1)
            plt.imshow(decoded_grid[idx].permute(1, 2, 0).numpy())
            plt.axis('off')
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
    
    file_name = ckpt_path.replace('/', "-")
    grid_file = f"vis/exp2-PrarmsInterpolation/4img_bilinear_interp_{file_name}_idx{img_indices_4}_steps{grid_steps}.png"
    plt.savefig(grid_file)
    plt.close()
    
    print(f"Bilinear interpolation grid saved to: {grid_file}")
        

