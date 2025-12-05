# 从零实现 3D Gaussian Splatting

这是一个用于学习 3D Gaussian Splatting 原理的教学项目。我们将从最基础的数学原理开始，逐步构建一个纯 Python 的渲染器，最终迁移到 PyTorch 进行优化。

## 学习路线图

### Phase 1: 数学基础与数据结构
- 目标：理解如何用数学公式描述一个“3D 高斯球”，以及它如何投影到屏幕上。
- 关键概念：协方差矩阵 (Covariance Matrix)、旋转矩阵 (Rotation Matrix)、透视投影 (Perspective Projection)。

### Phase 2: 纯 Python 渲染器
- 目标：不依赖深度学习框架，手动实现将 3D 高斯画到图片上的过程。
- 关键概念：光栅化 (Rasterization)、Alpha Blending、深度排序。

### Phase 3: 优化循环
- 目标：利用 PyTorch 的自动微分，让高斯球自己“移动”和“变形”来拟合目标图像。
- 关键概念：Loss 函数、梯度下降、自适应密度控制 (Adaptive Density Control)。