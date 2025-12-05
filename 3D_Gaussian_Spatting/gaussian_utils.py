import numpy as np
import math

def get_rotation_matrix_from_euler(x, y, z):
    """
    简单的辅助函数：将欧拉角(弧度)转换为旋转矩阵 R
    这里为了简单直观，先使用欧拉角，实际项目中通常使用四元数(Quaternion)
    """
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(x), -math.sin(x)],
                    [0, math.sin(x), math.cos(x)]])
                    
    R_y = np.array([[math.cos(y), 0, math.sin(y)],
                    [0, 1, 0],
                    [-math.sin(y), 0, math.cos(y)]])
                 
    R_z = np.array([[math.cos(z), -math.sin(z), 0],
                    [math.sin(z), math.cos(z), 0],
                    [0, 0, 1]])
                    
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def compute_covariance_3d(scale, rotation_matrix):
    """
    计算 3D 协方差矩阵 Sigma
    
    参数:
    scale: (3,) numpy array, 代表 x, y, z 三个轴的长度
    rotation_matrix: (3, 3) numpy array, 旋转矩阵 R
    
    返回:
    Sigma: (3, 3) 协方差矩阵
    """
    # 1. 构建缩放矩阵 S (对角矩阵)
    # S 是一个 3x3 矩阵，对角线上是 scale 的值，其他地方是 0
    S = np.diag(scale)
    R = rotation_matrix
    
    # 2. 计算协方差矩阵 Sigma = R * S * S^T * R^T
    # 提示：在 numpy 中，矩阵乘法可以使用 @ 符号，转置使用 .T
    
    # TODO: 请根据公式实现下面这行代码
    Sigma = R @ S @ S.T @ R.T
    
    return Sigma

def compute_2d_covariance(covariance_3d, view_matrix, focal_x, focal_y, view_space_pos):
    """
    计算 2D 投影协方差矩阵 (EWA Splatting)
    
    参数:
    covariance_3d: (3, 3) 3D 协方差矩阵
    view_matrix: (3, 3) 视图矩阵的旋转部分 (World to Camera Rotation)
    focal_x, focal_y: 相机焦距
    view_space_pos: (3,) 高斯中心在相机坐标系下的位置 (x, y, z)
    
    返回:
    cov2d: (2, 2) 2D 协方差矩阵
    """
    # 1. 提取相机坐标系下的坐标
    x, y, z = view_space_pos
    
    # 2. 构建雅可比矩阵 J (2x3)
    # J 描述了 3D 位置微小变化如何影响 2D 屏幕坐标
    # 公式:
    # J = [ fx/z   0    -(fx*x)/(z*z) ]
    #     [ 0    fy/z   -(fy*y)/(z*z) ]
    
    # TODO: 请实现 J 矩阵的构建
    J = np.zeros((2, 3))
    # 第一行：u 对 x, y, z 的偏导数
    J[0, 0] = focal_x / z
    J[0, 1] = 0
    J[0, 2] = -(focal_x * x) / (z * z)
    
    # 第二行：v 对 x, y, z 的偏导数
    J[1, 0] = 0
    J[1, 1] = focal_y / z
    J[1, 2] = -(focal_y * y) / (z * z)
    
    # 3. 计算变换矩阵 T = J @ W
    # W 是 view_matrix (3x3)
    W = view_matrix
    T = J @ W
    
    # 4. 计算 2D 协方差: Sigma_2D = T @ Sigma_3D @ T.T
    cov2d = T @ covariance_3d @ T.T
    
    return cov2d

# --- 测试代码 ---
if __name__ == "__main__":
    # 定义一个高斯球
    # 1. 大小：X轴长一些(5)，Y和Z轴短一些(1) -> 像一个雪茄
    my_scale = np.array([5.0, 1.0, 1.0])
    
    # 2. 旋转：绕 Z 轴旋转 45 度 (pi/4)
    my_rotation = get_rotation_matrix_from_euler(0, 0, math.pi/4)
    
    # 3. 计算协方差
    cov = compute_covariance_3d(my_scale, my_rotation)
    
    print("Scale:\n", my_scale)
    print("Rotation Matrix:\n", my_rotation)
    print("Calculated Covariance Matrix:\n", cov)

    # --- 测试 2D 投影 ---
    print("\n--- Testing 2D Projection ---")
    # 假设相机在原点，看向 Z 轴正方向
    # 高斯球在相机前方 Z=10 的位置
    view_pos = np.array([0.0, 0.0, 10.0]) 
    
    # 简单的相机参数
    focal_x = 500.0
    focal_y = 500.0
    
    # 视图矩阵 (这里假设是单位矩阵，即世界坐标系 = 相机坐标系)
    view_matrix = np.eye(3)
    
    cov2d = compute_2d_covariance(cov, view_matrix, focal_x, focal_y, view_pos)
    print("Projected 2D Covariance:\n", cov2d)