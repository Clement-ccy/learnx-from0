"""
13_position_wise_ffn.py - Position-wise Feed-Forward Network 实现

目标：实现Transformer中的前馈神经网络层
结构：Linear(d_model -> d_ff) -> ReLU -> Linear(d_ff -> d_model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

print("=" * 80)
print("🎯 PositionWiseFeedForward 类实现与测试")
print("=" * 80)


class PositionWiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network
    对序列中的每个位置独立地应用相同的全连接网络
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: 输入/输出维度 (例如 512)
            d_ff: 中间层维度 (例如 2048，通常是 d_model 的 4 倍)
            dropout: Dropout比率
        """
        super().__init__()
        
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ PositionWiseFeedForward初始化:")
        print(f"   d_model={d_model}, d_ff={d_ff}")
        print(f"   结构: Linear({d_model}->{d_ff}) -> ReLU -> Dropout -> Linear({d_ff}->{d_model})")
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, d_model]
        """
        # 1. 第一层线性变换 + ReLU
        # x @ w_1.T + b_1
        inter = self.w_1(x)
        inter = F.relu(inter)
        
        # 2. Dropout
        inter = self.dropout(inter)
        
        # 3. 第二层线性变换
        # inter @ w_2.T + b_2
        output = self.w_2(inter)
        
        return output


# =============================================================================
# 测试代码
# =============================================================================
print("\n\n【测试】FFN 功能验证")
print("-" * 80)

d_model = 512
d_ff = 2048
batch_size = 2
seq_len = 10

ffn = PositionWiseFeedForward(d_model, d_ff, dropout=0.0)

x = torch.randn(batch_size, seq_len, d_model)
print(f"输入形状: {x.shape}")

output = ffn(x)
print(f"输出形状: {output.shape}")

# 验证形状不变
assert output.shape == x.shape, "输出形状应该与输入形状相同"
print("✅ 形状验证通过！")

# 验证Position-wise特性（可选）
# 我们可以验证改变序列长度不影响参数数量，也不报错
x_short = torch.randn(batch_size, 5, d_model)
output_short = ffn(x_short)
print(f"变长序列输入形状: {x_short.shape} -> 输出: {output_short.shape}")
print("✅ 变长序列处理验证通过！")

print("\n" + "=" * 80)
print("✅ PositionWiseFeedForward 实现完成！")
print("=" * 80)