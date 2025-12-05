import numpy as np
import math

def compute_inverted_covariance(cov2d):
    """
    计算 2D 协方差矩阵的逆矩阵
    cov2d: (2, 2)
    """
    # 1. 计算行列式 det = ad - bc
    det = np.linalg.det(cov2d)
    
    # 保护机制：如果行列式太小（矩阵奇异），则无法求逆
    if det < 1e-6:
        return None
        
    # 2. 计算逆矩阵
    # inv = [[d, -b], [-c, a]] / det
    inv_cov = np.linalg.inv(cov2d)
    return inv_cov

def get_bounding_box(center_u, center_v, cov2d, image_width, image_height):
    """
    计算 2D 高斯的包围盒 (3倍标准差)
    """
    # 提取对角线元素 (方差)
    var_x = cov2d[0, 0]
    var_y = cov2d[1, 1]
    
    # 标准差 = sqrt(方差)
    radius_x = 3.0 * math.sqrt(var_x)
    radius_y = 3.0 * math.sqrt(var_y)
    
    # 计算边界 (取整)
    min_x = max(0, int(center_u - radius_x))
    max_x = min(image_width, int(center_u + radius_x) + 1)
    
    min_y = max(0, int(center_v - radius_y))
    max_y = min(image_height, int(center_v + radius_y) + 1)
    
    return min_x, max_x, min_y, max_y

def rasterize_single_gaussian(image, center_u, center_v, cov2d, color, opacity):
    """
    光栅化单个高斯到图像上 (Additive Blending)
    
    参数:
    image: (H, W, 3) 画布
    center_u, center_v: 高斯中心坐标
    cov2d: (2, 2) 协方差矩阵
    color: (3,) RGB 颜色
    opacity: 不透明度 (0.0 - 1.0)
    """
    H, W, _ = image.shape
    
    # 1. 计算逆协方差矩阵
    inv_cov = compute_inverted_covariance(cov2d)
    if inv_cov is None:
        return # 忽略奇异矩阵
        
    # 2. 计算包围盒
    min_x, max_x, min_y, max_y = get_bounding_box(center_u, center_v, cov2d, W, H)
    
    # 3. 遍历包围盒内的像素
    for y in range(min_y, max_y):
        for x in range(min_x, max_x):
            # 像素坐标相对于高斯中心的偏移
            delta = np.array([x - center_u, y - center_v])
            
            # TODO: 计算高斯强度 (Power)
            # 公式: power = exp(-0.5 * delta.T @ inv_cov @ delta)
            power = math.exp(-0.5 * (delta @ inv_cov @ delta))
            
            # 最终 Alpha = 基础不透明度 * 高斯强度
            alpha = opacity * power
            
            # 简单的叠加混合 (Additive Blending) - 仅用于演示
            # 实际 3DGS 使用的是 Alpha Blending (从前向后或从后向前)
            if alpha > 0.05: # 忽略太淡的像素
                image[y, x] += color * alpha

    return image

# --- 测试代码 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 创建一个黑色画布 100x100
    canvas = np.zeros((100, 100, 3))
    
    # 定义一个高斯
    center = (50, 50)
    # 一个向右上倾斜的椭圆
    cov = np.array([[100.0, 50.0], 
                    [50.0, 100.0]]) 
    color = np.array([1.0, 0.0, 0.0]) # 红色
    opacity = 1.0
    
    rasterize_single_gaussian(canvas, center[0], center[1], cov, color, opacity)
    
    # 显示结果 (需要 matplotlib)
    # 如果没有 GUI，可以保存为图片
    try:
        plt.imshow(canvas)
        plt.title("Single 2D Gaussian")
        plt.show()
    except:
        print("无法显示图像，尝试保存为 'result.png'")
        # 简单的保存逻辑 (需要 PIL 或 cv2，这里为了简单略过，只打印成功)
        print("Rasterization finished.")