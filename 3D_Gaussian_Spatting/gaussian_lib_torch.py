import torch
import math

def build_rotation(r):
    """
    构建旋转矩阵 (Batch)
    r: (N, 4) 四元数 [w, x, y, z] 或 (N, 3) 欧拉角
    这里为了简单，我们假设输入是四元数，因为四元数在优化时更稳定
    """
    # 归一化四元数
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    
    R = torch.zeros((q.shape[0], 3, 3), device=q.device)
    
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    """
    构建 3D 协方差矩阵的分解部分 L = R * S
    Sigma = L * L^T = R * S * S^T * R^T
    """
    L = torch.zeros((s.shape[0], 3, 3), device=s.device)
    R = build_rotation(r)
    
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    
    L = R @ L
    return L

def project_gaussians(means3D, scales, quats, viewmatrix, projmatrix, H, W, tan_fovx, tan_fovy):
    """
    将 3D 高斯投影到 2D (Batch)
    这是整个渲染器中最核心、计算量最大的部分
    """
    # 1. 转换到相机坐标系
    # means3D: (N, 3)
    # viewmatrix: (4, 4) World -> Camera
    
    # 齐次坐标 (N, 4)
    p_hom = torch.cat((means3D, torch.ones_like(means3D[:, :1])), dim=1)
    # (N, 4) @ (4, 4) -> (N, 4)
    p_cam = (p_hom @ viewmatrix.transpose(0, 1))[:, :3] 
    
    # 2. 剔除视锥体外的高斯 (简单的 Z 剔除)
    # z < 0.2 的高斯不渲染
    # DEBUG: 暂时禁用剔除，或者放宽条件
    mask = torch.ones_like(p_cam[:, 2], dtype=torch.bool) # p_cam[:, 2] > -999.0
    
    # 3. 构建 3D 协方差
    # L = R * S
    L = build_scaling_rotation(scales, quats)
    # Sigma = L @ L.T
    cov3D = L @ L.transpose(1, 2)
    
    # 4. 投影 3D 协方差到 2D (EWA Splatting)
    # J 矩阵 (雅可比)
    # J = [ fx/z   0    -(fx*x)/(z*z) ]
    #     [ 0    fy/z   -(fy*y)/(z*z) ]
    
    x = p_cam[:, 0]
    y = p_cam[:, 1]
    z = p_cam[:, 2]
    
    # DEBUG: 强制深度为正，防止投影反转
    z = torch.abs(z) + 1e-6
    
    # 焦距 f = W / (2 * tan(fov/2))
    fx = W / (2 * tan_fovx)
    fy = H / (2 * tan_fovy)
    
    # 构建 J 矩阵 (N, 2, 3)
    J = torch.zeros((means3D.shape[0], 2, 3), device=means3D.device)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -(fx * x) / (z * z)
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -(fy * y) / (z * z)
    
    # 视图变换的旋转部分 W (3, 3)
    W_mat = viewmatrix[:3, :3] # (3, 3)
    
    # T = J @ W (N, 2, 3)
    # 注意：W 是所有高斯共用的，需要广播
    T = J @ W_mat.unsqueeze(0)
    
    # cov2D = T @ cov3D @ T.T
    cov2D = T @ cov3D @ T.transpose(1, 2)
    
    # 加上低通滤波 (防止协方差奇异)
    cov2D[:, 0, 0] += 0.3
    cov2D[:, 1, 1] += 0.3
    
    # 5. 计算屏幕坐标
    # u = fx * x / z + cx
    # v = fy * y / z + cy
    means2D = torch.zeros_like(means3D[:, :2])
    means2D[:, 0] = (x * fx / z) + W / 2
    means2D[:, 1] = (y * fy / z) + H / 2
    
    return means2D, cov2D, mask, p_cam[:, 2] # 返回 depth 用于排序