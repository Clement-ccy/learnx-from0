import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import math

class MultiGaussianModel(nn.Module):
    def __init__(self, num_gaussians, target_h, target_w):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # 1. 位置: 随机分布在图像范围内
        # 范围: [-W/2, W/2] 和 [-H/2, H/2]
        self.pos = nn.Parameter(torch.rand(num_gaussians, 2) * torch.tensor([target_w, target_h]) - torch.tensor([target_w/2, target_h/2]))
        
        # 2. 颜色: 随机 RGB
        self.color = nn.Parameter(torch.rand(num_gaussians, 3))
        
        # 3. 缩放: 初始较小，避免一开始就糊成一团
        self.scale = nn.Parameter(torch.rand(num_gaussians, 2) * 5.0 + 1.0)
        
        # 4. 不透明度: 初始较低
        self.opacity_logit = nn.Parameter(torch.ones(num_gaussians) * -2.0) # sigmoid(-2) ≈ 0.12

    def forward(self):
        return {
            'pos': self.pos,
            'color': torch.sigmoid(self.color),
            'scale': torch.abs(self.scale),
            'opacity': torch.sigmoid(self.opacity_logit)
        }

def render_multi_gaussians(params, H, W):
    """
    并行渲染多个 2D 高斯
    """
    num_gaussians = params['pos'].shape[0]
    
    # 1. 生成像素网格 (H, W, 2)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H).float(), 
        torch.arange(W).float(),
        indexing='ij'
    )
    # grid: (H, W, 2)
    grid = torch.stack([x_coords, y_coords], dim=-1)
    
    # 2. 调整中心坐标
    # params['pos']: (N, 2) -> center: (N, 2)
    centers = params['pos'] + torch.tensor([W/2, H/2])
    
    # 3. 计算距离 (广播机制)
    # grid: (H, W, 1, 2)
    # centers: (1, 1, N, 2)
    # delta: (H, W, N, 2)
    delta = grid.unsqueeze(2) - centers.reshape(1, 1, num_gaussians, 2)
    dist_sq = torch.sum(delta**2, dim=-1) # (H, W, N)
    
    # 4. 计算高斯强度
    # scale: (N, 2) -> mean_scale: (N,)
    # radius: (1, 1, N)
    radius = params['scale'].mean(dim=1).reshape(1, 1, num_gaussians)
    
    # power: (H, W, N)
    power = torch.exp(-0.5 * dist_sq / (radius**2 + 1e-6))
    
    # 5. 混合 (简化版：加法混合，不考虑遮挡)
    # 真正的 3DGS 需要排序和 Alpha Blending，但在 2D 拟合中，加法混合也能出效果，且更快
    # alpha: (H, W, N)
    alpha = (params['opacity'].reshape(1, 1, num_gaussians) * power)
    
    # contribution: (H, W, N, 3)
    # color: (1, 1, N, 3)
    contribution = alpha.unsqueeze(-1) * params['color'].reshape(1, 1, num_gaussians, 3)
    
    # sum over gaussians: (H, W, 3)
    final_image = torch.sum(contribution, dim=2)
    
    # 限制在 0-1
    return torch.clamp(final_image, 0.0, 1.0)

if __name__ == "__main__":
    # 1. 加载目标图像 (这里我们生成一个简单的笑脸作为测试)
    # H, W = 100, 100
    # gt_image = torch.zeros(H, W, 3)
    # # 脸
    # mask_face = ((torch.arange(W)-50)**2 + (torch.arange(H).reshape(-1,1)-50)**2) < 1600
    # gt_image[mask_face] = torch.tensor([1.0, 1.0, 0.0]) # 黄色
    # # 左眼
    # mask_eye_l = ((torch.arange(W)-35)**2 + (torch.arange(H).reshape(-1,1)-35)**2) < 50
    # gt_image[mask_eye_l] = torch.tensor([0.0, 0.0, 0.0])
    # # 右眼
    # mask_eye_r = ((torch.arange(W)-65)**2 + (torch.arange(H).reshape(-1,1)-35)**2) < 50
    # gt_image[mask_eye_r] = torch.tensor([0.0, 0.0, 0.0])
    # # 嘴巴
    # mask_mouth = ((torch.arange(W)-50)**2/400 + (torch.arange(H).reshape(-1,1)-65)**2/50) < 1
    # gt_image[mask_mouth] = torch.tensor([1.0, 0.0, 0.0])

    # 使用真实图像
    img = Image.open("IP for Palpitating Forever.png").convert("RGB")
    gt_image = torch.from_numpy(np.array(img)).float() / 255.0
    H, W = gt_image.shape[:2]

    # 2. 初始化模型 (100 个高斯)
    model = MultiGaussianModel(100, H, W)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    
    print("Start Fitting...")
    
    for epoch in range(501):
        optimizer.zero_grad()
        
        params = model()
        rendered = render_multi_gaussians(params, H, W)
        
        loss = torch.mean((rendered - gt_image)**2)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    # 3. 显示结果
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(gt_image.numpy())
    ax[0].set_title("Target Image")
    ax[1].imshow(rendered.detach().numpy())
    ax[1].set_title(f"Fitted with {model.num_gaussians} Gaussians")
    plt.show()