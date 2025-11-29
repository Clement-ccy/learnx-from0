"""
09_multi_head_attention.py - Multi-Head Attention 类实现

目标：实现Transformer的核心组件 Multi-Head Attention
关键点：维度变换 (Reshape & Transpose) 以实现并行计算
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("=" * 80)
print("🎯 MultiHeadAttention 类实现与测试")
print("=" * 80)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention 模块
    
    将输入分割成多个头，分别计算注意力，然后合并。
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Args:
            d_model: 输入和输出的维度 (例如 512)
            num_heads: 头的数量 (例如 8)
            dropout: Dropout比率
        """
        super().__init__()
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) 必须能被 num_heads ({num_heads}) 整除"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # 每个头的维度
        
        # 定义 Q, K, V 的线性投影层
        # 注意：我们使用一个大的线性层来一次性计算所有头，然后再分割
        # 也可以写成 num_heads 个小的线性层，但这样效率更高
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        print(f"✅ MultiHeadAttention初始化:")
        print(f"   d_model={d_model}, num_heads={num_heads}, d_head={self.d_head}")
        
    def forward(self, q, k, v, mask=None):
        """
        前向传播
        
        Args:
            q, k, v: 输入张量，通常在Self-Attention中 q=k=v=x
                     形状: [batch_size, seq_len, d_model]
            mask: 掩码张量 [batch_size, 1, seq_len, seq_len] 或 [batch_size, 1, 1, seq_len]
                  (维度1是为了广播给所有头)
        """
        batch_size = q.size(0)
        
        # 1. 线性投影 + 分头 (Split Heads)
        # 输入: [batch, seq_len, d_model]
        # 变换: [batch, seq_len, num_heads, d_head] -> [batch, num_heads, seq_len, d_head]
        
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        
        # 2. 缩放点积注意力 (Scaled Dot-Product Attention)
        # Q @ K^T
        # [batch, heads, seq, d_head] @ [batch, heads, d_head, seq] -> [batch, heads, seq, seq]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        # 应用掩码
        if mask is not None:
            # mask形状需要能广播到 scores
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        # [batch, heads, seq, seq] @ [batch, heads, seq, d_head] -> [batch, heads, seq, d_head]
        context = torch.matmul(attn_weights, V)
        
        # 3. 合并头 (Concat Heads)
        # [batch, heads, seq, d_head] -> [batch, seq, heads, d_head] -> [batch, seq, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 4. 最终线性投影
        output = self.w_o(context)
        
        return output, attn_weights


# =============================================================================
# 测试代码
# =============================================================================
print("\n\n【测试】MultiHeadAttention 功能验证")
print("-" * 80)

# 参数设置
batch_size = 2
seq_len = 5
d_model = 16  # 总维度
num_heads = 4 # 头数
d_head = d_model // num_heads # 4

print(f"参数: batch={batch_size}, seq={seq_len}, d_model={d_model}, heads={num_heads}")

# 创建模型
mha = MultiHeadAttention(d_model, num_heads, dropout=0.0)

# 创建输入 (Self-Attention: q=k=v)
x = torch.randn(batch_size, seq_len, d_model)
print(f"输入 x 形状: {x.shape}")

# 前向传播
output, attn = mha(x, x, x)

print(f"\n输出 output 形状: {output.shape}  (应为 [{batch_size}, {seq_len}, {d_model}])")
print(f"注意力 attn 形状: {attn.shape}    (应为 [{batch_size}, {num_heads}, {seq_len}, {seq_len}])")

# 验证输出形状
assert output.shape == (batch_size, seq_len, d_model), "输出形状错误！"
assert attn.shape == (batch_size, num_heads, seq_len, seq_len), "注意力权重形状错误！"

print("\n✅ 形状验证通过！")

# 打印第一个样本的第一个头的注意力矩阵
print("\n示例注意力矩阵 (Batch 0, Head 0):")
print(attn[0, 0].detach())
print(f"行和检查: {attn[0, 0].sum(dim=-1).detach()} (应全为1)")

print("\n" + "=" * 80)
print("✅ MultiHeadAttention 实现完成！")
print("=" * 80)