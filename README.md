# learnx-from0

一个词怎样找到另一个词的信息？一团三维高斯怎样落到照片上？一个网络又怎样记住物体的形状？

这里从这些具体问题出发，把例子、公式、代码和图像放进同一篇 Notebook。边读边运行，改一个参数，再看看结果发生了什么。

## 挑一个问题开始

| 主题 | 会动手做什么？ | 篇数 |
| --- | --- | --- |
| [Transformer](transformer/01_注意力与多头机制.ipynb) | 算一次注意力，搭好网络，教它把数字倒着写，再尝试德英翻译 | 3 篇主线 + 1 篇选读 |
| [3D Gaussian Splatting](3d-gaussian-spatting/01_相机与高斯投影.ipynb) | 把空间里的“软气球”投到屏幕上，再让二维软画笔学着画图 | 3 篇主线 + 1 篇点云选读 |
| [Diffusion Model](diffusion-model/01_forward_process.ipynb) | 给一张图逐渐加上雪花，观察噪声调度怎样改变它 | 1 篇前向扩散 |
| [DeepSDF](deepsdf/01_SDF基础与单物体拟合.ipynb) | 学习空间中的距离，拼出表面，再为新物体寻找形状编码 | 3 篇渐进实验 |
| [RVC](retrieval-based-voice-conversion/00_声音转换学习地图.ipynb) | 从波形、频谱、音高到声音转换，先整理一条学习路线 | 1 篇学习计划 |
| [现代控制与 MPC](modern-control/00_学习地图与小车建模.ipynb) | 用小车停靠串起 PID、状态空间、LQR、Kalman 滤波与约束 MPC，比较扰动和模型误差下的表现 | 6 篇递进实验 |

每个主题的第一篇都有导航，正文和实现都在 Notebook 里。可以按顺序走，也可以带着某个问题回来看一节。

## 先跑起来

安装 [uv](https://docs.astral.sh/uv/getting-started/installation/) 后，进入想学的主题目录：

```bash
cd transformer  # 或 modern-control、3d-gaussian-spatting、diffusion-model、deepsdf
uv sync --python 3.11.15
uv run jupyter lab
```

`uv sync` 把依赖装进这个主题的 `.venv`，`uv run` 用它启动 Jupyter。同一主题的几篇笔记共用环境，不同主题各有自己的依赖。

打开 Notebook 后，从第一格顺序运行。想重新开始一次实验，就重启内核，再从头运行。

### 使用 VS Code

安装 Python 和 Jupyter 扩展，打开主题文件夹，在 Notebook 右上角选择“选择内核 → Python 环境”，选中该主题的 `.venv/bin/python`。

列表找不到环境时，先执行 `Python: Select Interpreter`，手动指定该路径，再回到 Notebook 选择内核。Transformer 和 DeepSDF 使用不同版本的 PyTorch，分别选择各自目录中的环境即可。

### 第一次训练

先降低训练步数或批量，把数据、损失和图像看完整，再增加训练预算。笔记中的设备选择使用 CUDA 或 CPU；Mac 上当前默认使用 CPU。其他设备配置可参考 [PyTorch 安装说明](https://pytorch.org/get-started/locally/)。

## 几个实验的准备事项

- **DeepSDF 新物体重建**：先运行第二篇的训练与保存单元，第三篇会加载 `deepsdf/artifacts/auto_decoder.pt`。
- **[德英翻译](transformer/04_德英翻译实战.ipynb)**：按篇内命令安装可选依赖和语言模型。`RUN_TRANSLATION=False` 用于阅读，改为 `True` 后准备语料并训练。
- **[交互点云](3d-gaussian-spatting/04_交互点云与实现边界.ipynb)**：在支持 OpenCV 窗口的本机内核运行，开启 `RUN_INTERACTIVE` 后可旋转视角、观察点云颜色拟合。
- **RVC**：目前是一页学习计划，直接阅读即可。
- **现代控制与 MPC**：安装依赖后全部实验离线运行，只需 CPU；六篇各自独立运行，无需下载数据或读取前篇产物。最后两篇使用 CVXPY + OSQP 求解小规模二次规划。

## 怎样用这些笔记复习

先读问题，猜一猜结果，再运行代码。看输出时找一个具体对象：注意力表的一行、投影后的一片椭圆、重建图的一条零等值线。能顺着代码解释它怎样产生，就把这一步接起来了。

公式推导和实现细节放在对应实验旁，部分内容可以折叠。静态示意图嵌在 Notebook 中，运行会产生实验图表与日志；保留有助于阅读的输出，环境、下载语料和训练权重留在本地。
