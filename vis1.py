import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from model import VAE
from utils.data_utils import figure_dataset

def generate_images_from_checkpoints():
    """
    从多个checkpoint生成图片并拼接成一行
    """
    # 设置
    checkpoint_dir = "output_dir_lr0.0005_epoch10000_latent16_hidden1024"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 模型参数（与训练时保持一致）
    input_size = 40
    hidden_size = 1024
    latent_size = 16
    
    # 选择12个检查点（均匀采样）
    checkpoint_epochs = [0, 200, 400, 800, 1800, 2800, 3800, 4800, 5800, 6800, 7800]
    checkpoint_files = [f"checkpoint_{epoch}.pt" for epoch in checkpoint_epochs]
    randomseed = 2
    # 从数据集中随机抽取一张图片
    torch.manual_seed(randomseed)
    dataset = figure_dataset("figure")
    random_idx = torch.randint(0, len(dataset), (1,)).item()
    sample_img = dataset[random_idx].unsqueeze(0).to(device)  # shape: (1, 3, 40, 40)
    print(f"从数据集中选择第 {random_idx} 张图片作为输入")
    
    # 使用第一个checkpoint的encoder获取z向量
    first_model = VAE(input_size=input_size, hidden_size=hidden_size, latent_size=latent_size)
    first_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, checkpoint_files[-1]), map_location=device))
    first_model = first_model.to(device)
    first_model.eval()
    
    with torch.no_grad():
        z_sample, _, _ = first_model.encode(sample_img)  # 从真实图片编码得到z
    print(f"获得潜在向量 z, shape: {z_sample.shape}")
    
    # 保存input_shape，后续复用
    saved_input_shape = first_model.input_shape.copy()
    
    # 创建图像网格 (1行：原图 + N个生成图)
    num_cols = 1 + len(checkpoint_epochs)  # 原图 + checkpoint数量
    fig, axes = plt.subplots(1, num_cols, figsize=(2 * num_cols, 2))
    
    # 显示原始图片
    original_img = sample_img.squeeze(0).cpu().numpy()
    original_img = np.transpose(original_img, (1, 2, 0))
    original_img = np.clip(original_img, 0, 1)
    axes[0].imshow(original_img)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis('off')
    
    for idx, (epoch, ckpt_file) in enumerate(zip(checkpoint_epochs, checkpoint_files)):
        ckpt_path = os.path.join(checkpoint_dir, ckpt_file)
        
        if not os.path.exists(ckpt_path):
            print(f"警告: {ckpt_path} 不存在，跳过")
            continue
        
        # 加载模型
        model = VAE(
            input_size=input_size,
            hidden_size=hidden_size,
            latent_size=latent_size
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model = model.to(device)
        model.eval()
        
        # 生成图片
        with torch.no_grad():
            # 直接设置input_shape，避免重复调用encode
            model.input_shape = saved_input_shape
            # 使用之前提取的z_sample进行解码
            generated_img = model.decode(z_sample)  # shape: (1, 3, 40, 40)
        
        # 转换为numpy并调整维度 (C, H, W) -> (H, W, C)
        img_np = generated_img.squeeze(0).cpu().numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
        img_np = np.clip(img_np, 0, 1)  # 确保在[0,1]范围
        
        # 显示图片
        axes[idx + 1].imshow(img_np)  # +1 因为第0列是原图
        axes[idx + 1].set_title(f"Epoch {epoch}", fontsize=10)
        axes[idx + 1].axis('off')
        
        print(f"已生成 Epoch {epoch} 的图片")
    
    plt.tight_layout()
    save_path = f"vis/exp1-TimeInterpolation/generated_images{randomseed}_progression.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n拼接图片已保存为: {save_path}")
    plt.show()


if __name__ == "__main__":
    generate_images_from_checkpoints()
