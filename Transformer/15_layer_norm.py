"""
15_layer_norm.py - Layer Normalization 和 Residual Connection 实现

目标：实现 Add & Norm 模块
"""

import torch
import torch.nn as nn

print("=" * 80)
print("🎯 LayerNorm 与 SublayerConnection 实现与测试")
print("=" * 80)


class LayerNorm(nn.Module):
    """
    自定义 Layer Normalization 实现 (为了理解原理)
    公式: y = (x - mean) / (std + eps) * gamma + beta
    """
    
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # gamma (缩放)
        self.b_2 = nn.Parameter(torch.zeros(features)) # beta (平移)
        self.eps = eps
        
        print(f"✅ LayerNorm初始化: features={features}")

    def forward(self, x):
        # 在最后一个维度上计算均值和方差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        
        # 归一化
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    残差连接 + 层归一化
    Output = x + Sublayer(Norm(x))  (Pre-Norm 结构，更稳定)
    或者
    Output = Norm(x + Sublayer(x))  (Post-Norm 结构，原始论文)
    
    这里我们实现原始论文的 Post-Norm: Norm(x + Sublayer(x))
    """
    
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ SublayerConnection初始化: size={size}, dropout={dropout}")

    def forward(self, x, sublayer):
        """
        Args:
            x: 输入
            sublayer: 一个函数/层（比如 Attention 或 FFN）
        """
        # 原始论文结构: x + Dropout(Sublayer(x)) -> Norm
        return self.norm(x + self.dropout(sublayer(x)))


# =============================================================================
# 测试代码
# =============================================================================
print("\n\n【测试】LayerNorm 功能验证")
print("-" * 80)

features = 4
ln = LayerNorm(features)

# 创建一个分布不均匀的输入
x = torch.tensor([[1.0, 2.0, 3.0, 100.0]], dtype=torch.float32)
print(f"输入: {x}")
print(f"输入均值: {x.mean():.4f}, 标准差: {x.std():.4f}")

output = ln(x)
print(f"输出: {output}")
print(f"输出均值: {output.mean():.4f}, 标准差: {output.std():.4f}")

# 验证输出均值接近0，标准差接近1
assert torch.abs(output.mean()) < 1e-3, "均值应接近0"
assert torch.abs(output.std() - 1.0) < 1e-3, "标准差应接近1"
print("✅ LayerNorm 归一化效果验证通过！")


print("\n\n【测试】SublayerConnection (Add & Norm) 功能验证")
print("-" * 80)

size = 4
dropout = 0.0
connection = SublayerConnection(size, dropout)

x = torch.zeros(1, size) # 输入全0
sublayer = lambda x: torch.ones(1, size) # 子层总是输出全1

# 预期：
# sublayer(x) = [1, 1, 1, 1]
# x + sublayer(x) = [1, 1, 1, 1]
# Norm([1, 1, 1, 1]) -> 全0 (因为方差为0，除以eps) * gamma + beta -> 0
# 等等，如果输入全一样，std是0，除以eps会得到0吗？
# (1-1)/eps = 0. 正确。

output = connection(x, sublayer)
print(f"输入 x: {x}")
print(f"子层输出: {sublayer(x)}")
print(f"Add & Norm 输出: {output}")

print("\n" + "=" * 80)
print("✅ LayerNorm 和 SublayerConnection 实现完成！")
print("=" * 80)