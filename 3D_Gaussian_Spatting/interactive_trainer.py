import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import math
from PIL import Image
from gaussian_lib_torch import project_gaussians

# 检查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class InteractiveGaussianModel(nn.Module):
    def __init__(self, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians
        
        # 初始化参数 (全部在 GPU 上)
        # DEBUG: 扩大分布范围到 [-5, 5]，让它们铺满屏幕
        self.means3D = nn.Parameter((torch.rand(num_gaussians, 3, device=device) - 0.5) * 10.0)
        
        # DEBUG: 颜色随机
        self.colors = nn.Parameter(torch.rand(num_gaussians, 3, device=device))
        
        # DEBUG: 缩放初始值设大一点 (1.0)，确保能看见
        self.scales = nn.Parameter(torch.ones(num_gaussians, 3, device=device) * 1.0)
        
        # 旋转: 随机四元数
        quats = torch.randn(num_gaussians, 4, device=device)
        self.quats = nn.Parameter(quats / quats.norm(dim=1, keepdim=True))
        
        # DEBUG: 不透明度设高一点 (sigmoid(2.0) ≈ 0.88)
        self.opacity_logit = nn.Parameter(torch.ones(num_gaussians, device=device) * 2.0)

    def forward(self):
        return self.means3D, torch.sigmoid(self.colors), torch.abs(self.scales), \
               self.quats / self.quats.norm(dim=1, keepdim=True), torch.sigmoid(self.opacity_logit)

class CameraController:
    def __init__(self):
        self.yaw = 0.0
        self.pitch = 0.0
        self.distance = 10.0
        self.last_mouse_pos = None
        self.is_dragging = False
        
    def handle_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.is_dragging = True
            self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.is_dragging = False
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_dragging and self.last_mouse_pos is not None:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.yaw -= dx * 0.01
                self.pitch -= dy * 0.01
                self.last_mouse_pos = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # 注意：OpenCV 的滚轮事件处理可能因系统而异
            # 这里简单处理，如果 flags > 0 放大，< 0 缩小
            if flags > 0:
                self.distance *= 0.9
            else:
                self.distance *= 1.1

    def get_view_matrix(self):
        # 计算相机位置
        cam_x = self.distance * math.sin(self.yaw) * math.cos(self.pitch)
        cam_y = self.distance * math.sin(self.pitch)
        cam_z = self.distance * math.cos(self.yaw) * math.cos(self.pitch)
        
        eye = np.array([cam_x, cam_y, cam_z])
        target = np.array([0.0, 0.0, 0.0])
        up = np.array([0.0, 1.0, 0.0]) # Y-up
        
        # 构建 LookAt 矩阵
        z_axis = eye - target
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        view_mat = np.eye(4)
        view_mat[:3, 0] = x_axis
        view_mat[:3, 1] = y_axis
        view_mat[:3, 2] = z_axis
        view_mat[:3, 3] = -np.dot(np.array([x_axis, y_axis, z_axis]), eye)
        
        return torch.tensor(view_mat, dtype=torch.float32, device=device)

def render(model, view_matrix, H, W, fov_y):
    means3D, colors, scales, quats, opacities = model()
    
    tan_fovx = math.tan(fov_y * 0.5 * W / H)
    tan_fovy = math.tan(fov_y * 0.5)
    
    # 1. 投影 (Batch)
    means2D, cov2D, mask, depths = project_gaussians(
        means3D, scales, quats, view_matrix, None, H, W, tan_fovx, tan_fovy
    )
    
    # DEBUG: 打印投影信息 (只打印一次)
    if torch.rand(1) < 0.01: # 1% 的概率打印，避免刷屏
        print(f"\n--- Debug Info ---")
        print(f"Means3D range: X[{means3D[:,0].min():.2f}, {means3D[:,0].max():.2f}], Z[{means3D[:,2].min():.2f}, {means3D[:,2].max():.2f}]")
        print(f"Depths range: [{depths.min():.2f}, {depths.max():.2f}]")
        print(f"Means2D range: U[{means2D[:,0].min():.2f}, {means2D[:,0].max():.2f}], V[{means2D[:,1].min():.2f}, {means2D[:,1].max():.2f}]")
        print(f"Valid points: {mask.sum()} / {mask.numel()}")
    
    # 2. 简单的光栅化 (Additive Blending for speed)
    # 注意：这里为了实时性，我们仍然使用加法混合
    # 真正的排序+Alpha Blending 在 Python 里很难做到实时
    
    # 过滤掉视锥体外的高斯
    valid_means2D = means2D[mask]
    valid_cov2D = cov2D[mask]
    valid_colors = colors[mask]
    valid_opacities = opacities[mask]
    
    # 创建画布
    canvas = torch.zeros((H, W, 3), device=device)
    
    # 并行光栅化 (简化版)
    # 我们只画每个高斯中心的一个点，或者用很小的核
    # 为了演示效果，我们这里只画点云
    
    u = valid_means2D[:, 0].long()
    v = valid_means2D[:, 1].long()
    
    # 边界检查
    valid_indices = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u = u[valid_indices]
    v = v[valid_indices]
    c = valid_colors[valid_indices]
    
    # 简单的 Scatter Add
    # canvas[v, u] += c # 这种写法在有重复索引时是不确定的
    # 正确做法是用 scatter_add，但为了代码简单，我们先这样写
    # 或者我们只显示前 10000 个点
    
    # DEBUG: 打印有多少个点在屏幕内
    # if u.numel() > 0:
    #     print(f"Points in screen: {u.numel()}, Max U: {u.max()}, Max V: {v.max()}")
    
    # 为了能看到东西，我们把点画大一点 (3x3)
    for i in range(-1, 2):
        for j in range(-1, 2):
            uu = torch.clamp(u + i, 0, W-1)
            vv = torch.clamp(v + j, 0, H-1)
            # 注意：这里用 accumulate=False，避免过曝
            # 或者我们除以一个系数
            canvas.index_put_((vv, uu), c * 0.5, accumulate=True)
            
    return torch.clamp(canvas, 0.0, 1.0)

if __name__ == "__main__":
    # 1. 加载图像
    img_path = "IP.png"
    try:
        gt_img_pil = Image.open(img_path).convert('RGB')
        gt_img_pil = gt_img_pil.resize((600, 450)) # 缩小一点以加快速度
        gt_img = torch.tensor(np.array(gt_img_pil) / 255.0, dtype=torch.float32, device=device)
        H, W = gt_img.shape[:2]
    except Exception as e:
        print(f"Error loading image: {e}")
        exit()

    # 2. 初始化
    # DEBUG: 增加高斯数量到 10000，提高学习率到 0.1
    model = InteractiveGaussianModel(100000)
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    cam_ctrl = CameraController()
    
    # 3. OpenCV 窗口
    cv2.namedWindow("Interactive Training")
    cv2.setMouseCallback("Interactive Training", cam_ctrl.handle_mouse)
    
    print("Start Interactive Training...")
    print("Controls:")
    print("  [Space]: Toggle Training/View Mode")
    print("  [Mouse Left]: Rotate Camera")
    print("  [Mouse Wheel]: Zoom")
    print("  [q]: Quit")
    
    step = 0
    is_training = True
    
    # 固定一个训练视角 (正视图)
    training_cam = CameraController()
    training_cam.distance = 5.0 # 拉近相机，让物体看起来更大
    training_cam.yaw = 0.0
    training_cam.pitch = 0.0
    training_view_mat = training_cam.get_view_matrix()
    
    while True:
        if is_training:
            optimizer.zero_grad()
            
            # 训练模式：始终使用固定的正视图
            rendered = render(model, training_view_mat, H, W, math.pi/3)
            
            loss = torch.mean((rendered - gt_img)**2)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            status_text = f"Training... Loss: {current_loss:.4f}"
        else:
            # 查看模式：使用鼠标控制的视角，不更新参数
            with torch.no_grad():
                view_mat = cam_ctrl.get_view_matrix()
                rendered = render(model, view_mat, H, W, math.pi/3)
            status_text = "View Mode (Not Training)"

        # 显示
        if step % 5 == 0: # 稍微提高刷新率
            img_np = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # 在图像上绘制状态文字
            cv2.putText(img_np, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Interactive Training", img_np)
            
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_training = not is_training
            print(f"\nSwitched to {'Training' if is_training else 'View'} Mode")
            
        step += 1
        if is_training:
            print(f"Step {step}, Loss: {current_loss:.4f}", end='\r')
        else:
            print(f"Step {step}, View Mode", end='\r')
        
    cv2.destroyAllWindows()