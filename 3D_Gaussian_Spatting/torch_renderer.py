import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

class GaussianModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始化一个高斯球
        # 我们希望它能自动移动到目标位置，并变成目标颜色
        
        # 1. 位置: 随机初始化
        self.position = nn.Parameter(torch.tensor([0.0, 0.0, 5.0]))
        
        # 2. 颜色: 随机初始化 (RGB)
        self.color = nn.Parameter(torch.rand(3))
        
        # 3. 缩放: 初始为一个单位球
        self.scale = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
        
        # 4. 旋转: 初始为不旋转 (欧拉角)
        self.rotation = nn.Parameter(torch.zeros(3))
        
        # 5. 不透明度: 初始为 0.5 (Logit 空间，经过 Sigmoid 后变成 0.5)
        self.opacity_logit = nn.Parameter(torch.tensor([0.0]))

    def forward(self):
        # 返回处理后的参数 (例如应用激活函数)
        return {
            'pos': self.position,
            'color': torch.sigmoid(self.color), # 限制在 0-1
            'scale': torch.abs(self.scale), # 必须为正
            'rot': self.rotation,
            'opacity': torch.sigmoid(self.opacity_logit) # 限制在 0-1
        }

def compute_covariance_3d_torch(scale, rotation):
    # 构建缩放矩阵 S
    S = torch.diag(scale)
    
    # 构建旋转矩阵 R (欧拉角 -> 矩阵)
    x, y, z = rotation[0], rotation[1], rotation[2]
    
    # 注意：这里需要手动构建 PyTorch 的旋转矩阵
    # 为了简化，我们先只写 Z 轴旋转
    # 实际应该写完整的 XYZ 旋转
    cos_z = torch.cos(z)
    sin_z = torch.sin(z)
    R_z = torch.stack([
        torch.stack([cos_z, -sin_z, torch.tensor(0.0)]),
        torch.stack([sin_z, cos_z, torch.tensor(0.0)]),
        torch.stack([torch.tensor(0.0), torch.tensor(0.0), torch.tensor(1.0)])
    ])
    
    R = R_z # 简化版
    
    # Sigma = R @ S @ S.T @ R.T
    Sigma = R @ S @ S.t() @ R.t()
    return Sigma

def render_single_gaussian_torch(params, target_h, target_w):
    """
    极简版的可微渲染器：只渲染一个高斯球
    """
    # 1. 准备画布 (H, W, 3)
    # 注意：为了可微，我们不能先创建一个全 0 的 Tensor 然后赋值
    # 我们需要生成网格坐标，然后并行计算所有像素的值
    
    y_coords, x_coords = torch.meshgrid(
        torch.arange(target_h).float(), 
        torch.arange(target_w).float(),
        indexing='ij'
    )
    
    # 2. 投影 (简化版：假设正交投影，直接忽略 Z，只看 XY)
    # 这是一个巨大的简化，为了先跑通优化循环
    # 实际上应该用 EWA Splatting
    
    center_x = params['pos'][0] + target_w / 2
    center_y = params['pos'][1] + target_h / 2
    
    # 3. 计算每个像素到中心的距离平方
    dx = x_coords - center_x
    dy = y_coords - center_y
    dist_sq = dx**2 + dy**2
    
    # 4. 高斯衰减
    # radius = scale.mean() * 10 (随便拍的一个系数)
    radius = params['scale'].mean() * 20.0
    power = torch.exp(-0.5 * dist_sq / (radius**2 + 1e-6))
    
    # 5. 最终图像
    # Image = Color * Opacity * Power
    # 扩展维度以便广播: (H, W, 1) * (3,)
    pixel_alpha = (params['opacity'] * power).unsqueeze(-1)
    pixel_color = params['color'] * pixel_alpha
    
    return pixel_color

# --- 训练循环 ---
if __name__ == "__main__":
    # 1. 创建目标图像 (Ground Truth)
    # 一个位于 (30, 30) 的绿色圆
    H, W = 64, 64
    gt_image = torch.zeros(H, W, 3)
    for y in range(H):
        for x in range(W):
            if (x-40)**2 + (y-20)**2 < 100: # 半径 10
                gt_image[y, x] = torch.tensor([0.0, 1.0, 0.0]) # 绿色
                
    # 2. 初始化模型
    model = GaussianModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    print("Start Training...")
    losses = []
    
    for epoch in range(100):
        optimizer.zero_grad()
        
        # Forward
        params = model()
        rendered_image = render_single_gaussian_torch(params, H, W)
        
        # Loss
        loss = torch.mean((rendered_image - gt_image)**2)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
    print("Training Finished!")
    print(f"Final Position: {model.position.data}")
    print(f"Final Color: {torch.sigmoid(model.color).data}")
    
    # 可视化 Loss
    plt.plot(losses)
    plt.title("Training Loss")
    plt.show()
    
    # 可视化结果对比
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(gt_image.detach().numpy())
    ax[0].set_title("Ground Truth")
    ax[1].imshow(rendered_image.detach().numpy())
    ax[1].set_title("Optimized Result")
    plt.show()