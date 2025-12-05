import numpy as np
import math
import matplotlib.pyplot as plt
from gaussian_utils import compute_covariance_3d, compute_2d_covariance, get_rotation_matrix_from_euler
from simple_rasterizer import compute_inverted_covariance, get_bounding_box

class Gaussian:
    def __init__(self, position, scale, rotation, color, opacity):
        self.position = np.array(position) # World Space (x, y, z)
        self.scale = np.array(scale)
        self.rotation = np.array(rotation) # Euler angles
        self.color = np.array(color)
        self.opacity = opacity

def sort_gaussians(gaussians, view_matrix):
    """
    对高斯球进行深度排序 (从近到远? 还是从远到近?)
    提示：Alpha Blending 公式通常是从前向后(Front-to-Back)或从后向前(Back-to-Front)
    这里我们采用标准的 Painter's Algorithm (画家算法)，即从远到近画。
    """
    # 1. 计算每个高斯球在相机坐标系下的 Z 值 (Depth)
    # World Pos -> Camera Pos: P_cam = View_Matrix * P_world
    
    dtype = [('index', int), ('depth', float)]
    depths = []
    
    for i, g in enumerate(gaussians):
        # 简单的矩阵乘法: P_cam = R * P_world + t (这里简化为只有旋转，假设相机在原点)
        # 实际项目中 View Matrix 是 4x4，包含平移
        # 这里为了复用之前的代码，我们假设 view_matrix 只是 3x3 旋转
        # 且相机在原点，看向 Z 正方向
        
        pos_cam = view_matrix @ g.position
        depth = pos_cam[2] # Z value
        depths.append((i, depth))
    
    # TODO: 对 depths 进行排序
    # 如果是从远到近画 (Back-to-Front)，应该按深度降序还是升序？
    # 提示：深度大 = 远
    # 使用 numpy 的 argsort 进行排序，然后反转得到降序
    # 或者直接用 Python 的 sort
    
    # 方法 1: Python sort
    depths.sort(key=lambda x: x[1], reverse=True)
    sorted_indices = [x[0] for x in depths]
    
    return sorted_indices

def render_scene(gaussians, view_matrix, focal_x, focal_y, width, height):
    canvas = np.zeros((height, width, 3))
    
    # 1. 排序
    sorted_indices = sort_gaussians(gaussians, view_matrix)
    
    # 2. 按顺序渲染
    for idx in sorted_indices:
        g = gaussians[idx]
        
        # --- 3D 部分 ---
        # 计算 3D 协方差
        R = get_rotation_matrix_from_euler(*g.rotation)
        cov3d = compute_covariance_3d(g.scale, R)
        
        # 投影到 2D
        pos_cam = view_matrix @ g.position
        cov2d = compute_2d_covariance(cov3d, view_matrix, focal_x, focal_y, pos_cam)
        
        # --- 2D 光栅化部分 (简化版，直接画在 canvas 上) ---
        # 注意：这里为了演示简单，我们还是用 simple_rasterizer 里的逻辑
        # 但实际上应该在这里实现 Alpha Blending
        # 为了不让代码太复杂，我们直接调用一个修改版的绘制函数
        rasterize_with_blending(canvas, g, cov2d, pos_cam, width, height, focal_x, focal_y)
        
    return canvas

def rasterize_with_blending(canvas, gaussian, cov2d, pos_cam, width, height, fx, fy):
    """
    将单个高斯混合到画布上 (Painter's Algorithm: 直接覆盖/混合)
    """
    # 计算屏幕中心坐标
    # u = fx * x / z + cx (假设 cx = width / 2)
    center_u = fx * pos_cam[0] / pos_cam[2] + width / 2
    center_v = fy * pos_cam[1] / pos_cam[2] + height / 2
    
    inv_cov = compute_inverted_covariance(cov2d)
    if inv_cov is None: return

    min_x, max_x, min_y, max_y = get_bounding_box(center_u, center_v, cov2d, width, height)
    
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            delta = np.array([x - center_u, y - center_v])
            power = math.exp(-0.5 * (delta @ inv_cov @ delta))
            alpha = gaussian.opacity * power
            
            if alpha > 0.05:
                # Painter's Algorithm (Back-to-Front):
                # New_Color = Alpha * Src_Color + (1 - Alpha) * Dest_Color
                # 这就是标准的 Alpha Blending 公式
                
                pixel_color = canvas[y, x]
                new_pixel_color = alpha * gaussian.color + (1 - alpha) * pixel_color
                canvas[y, x] = new_pixel_color

# --- 测试代码 ---
if __name__ == "__main__":
    # 场景：三个球，红绿蓝，前后排列
    scene = [
        # 红色：最远 (Z=15)
        Gaussian([0, 0, 15], [2, 0.5, 0.5], [0, 0, math.pi/4], [1, 0, 0], 1.0),
        # 绿色：中间 (Z=10)
        Gaussian([0, 0, 10], [0.5, 2, 0.5], [0, 0, 0], [0, 1, 0], 0.8),
        # 蓝色：最近 (Z=5)
        Gaussian([0, 0, 5], [0.5, 0.5, 2], [0, 0, 0], [0, 0, 1], 0.5),
    ]
    
    view_mat = np.eye(3)
    img = render_scene(scene, view_mat, 500, 500, 400, 400)
    
    plt.imshow(img)
    plt.title("Multi-Gaussian Scene (Sorted)")
    plt.show()