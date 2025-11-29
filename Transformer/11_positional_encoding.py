"""
11_positional_encoding.py - 位置编码实现与可视化

目标：实现正弦/余弦位置编码，并可视化其热力图
"""

import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("🎯 PositionalEncoding 类实现与可视化")
print("=" * 80)


class PositionalEncoding(nn.Module):
    """
    实现正弦/余弦位置编码
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        Args:
            d_model: 嵌入维度
            max_len: 预计算的最大序列长度
            dropout: Dropout比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建一个足够大的矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置索引 [0, 1, ..., max_len-1] -> [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母项 10000^(2i/d_model)
        # 使用对数空间计算更数值稳定: exp(log(10000) * -2i / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 填充矩阵
        # 偶数位用 sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 奇数位用 cos
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 增加 batch 维度: [1, max_len, d_model]
        # 这样可以直接广播加到输入 batch 上
        pe = pe.unsqueeze(0)
        
        # 注册为 buffer (不是模型参数，不需要梯度更新，但会随模型保存)
        self.register_buffer('pe', pe)
        
        print(f"✅ PositionalEncoding初始化: d_model={d_model}, max_len={max_len}")
        
    def forward(self, x):
        """
        Args:
            x: 输入嵌入向量 [batch_size, seq_len, d_model]
        """
        # x.size(1) 是当前序列长度
        # 我们只取前 seq_len 个位置编码
        # 广播加法: [batch, seq, d] + [1, seq, d]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 可视化测试
# =============================================================================
print("\n\n【测试】可视化位置编码矩阵")
print("-" * 80)

d_model = 128
max_len = 100
pe_module = PositionalEncoding(d_model, max_len, dropout=0.0)

# 获取位置编码矩阵 (去掉batch维度)
pe_matrix = pe_module.pe.squeeze(0).numpy()

print(f"位置编码矩阵形状: {pe_matrix.shape}")

# 绘图
plt.figure(figsize=(12, 8))
plt.imshow(pe_matrix, cmap='RdBu', aspect='auto')
plt.title("Positional Encoding Matrix Heatmap")
plt.xlabel("Embedding Dimension (Depth)")
plt.ylabel("Position (Sequence Length)")
plt.colorbar(label="Value")
plt.grid(False)

# 保存图片
save_path = "positional_encoding_heatmap.png"
plt.savefig(save_path)
print(f"✅ 热力图已保存至: {save_path}")
print("   观察图表：")
print("   - X轴是维度，Y轴是位置")
print("   - 你会看到明显的波纹图案，这就是正弦波")
print("   - 左侧（低频）变化慢，右侧（高频）变化快")

# =============================================================================
# 功能测试
# =============================================================================
print("\n\n【测试】前向传播功能")
print("-" * 80)

batch_size = 2
seq_len = 50
x = torch.zeros(batch_size, seq_len, d_model) # 全0输入，方便看加法结果

output = pe_module(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")

# 验证输出是否等于位置编码
expected = pe_module.pe[:, :seq_len, :]
assert torch.allclose(output, expected), "输出应该等于位置编码（因为输入是0）"
print("✅ 输出验证通过！")

print("\n" + "=" * 80)
print("✅ PositionalEncoding 实现完成！")
print("=" * 80)