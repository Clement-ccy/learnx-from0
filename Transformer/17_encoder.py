"""
17_encoder.py - Encoder Layer 和 Encoder 实现

目标：组装 Transformer 的 Encoder 部分
包含：
1. EncoderLayer: 单个编码器层 (Attention + FFN)
2. Encoder: N 层堆叠的编码器
"""

import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F

# 为了方便运行，我们把之前定义的组件复制过来
# 在实际项目中，你应该从其他文件 import

# --- 组件复用开始 ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        Q = self.w_q(q).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        K = self.w_k(k).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = self.w_v(v).view(batch_size, -1, self.num_heads, self.d_head).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.w_o(context)

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, sublayer):
        return self.norm(x + self.dropout(sublayer(x)))
        
def clones(module, N):
    "生成 N 个相同的层"
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
# --- 组件复用结束 ---


print("=" * 80)
print("🎯 Encoder Layer 与 Encoder 实现与测试")
print("=" * 80)


class EncoderLayer(nn.Module):
    """
    EncoderLayer 由两个子层组成：Self-Attention 和 FeedForward
    """
    
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # 两个子层连接器 (Add & Norm)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
        
        print(f"✅ EncoderLayer初始化: size={size}")

    def forward(self, x, mask):
        """
        Args:
            x: 输入 [batch, seq_len, d_model]
            mask: 掩码
        """
        # 1. Self-Attention 子层
        # 注意：SublayerConnection 接受一个函数 sublayer(x)
        # 所以我们用 lambda 包装 self_attn
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # 2. Feed-Forward 子层
        x = self.sublayer[1](x, self.feed_forward)
        
        return x


class Encoder(nn.Module):
    """
    Encoder 是 N 个 EncoderLayer 的堆叠
    """
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size) # 最后的 LayerNorm
        
        print(f"✅ Encoder初始化: {N} 层堆叠")
        
    def forward(self, x, mask):
        """
        依次通过每一层
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# =============================================================================
# 测试代码
# =============================================================================
print("\n\n【测试】EncoderLayer 功能验证")
print("-" * 80)

d_model = 512
heads = 8
d_ff = 2048
dropout = 0.1

# 实例化组件
attn = MultiHeadAttention(d_model, heads)
ff = PositionWiseFeedForward(d_model, d_ff)
layer = EncoderLayer(d_model, attn, ff, dropout)

# 创建输入
x = torch.randn(2, 10, d_model) # batch=2, seq=10
mask = torch.ones(2, 1, 1, 10) # 简单的全1 mask

output = layer(x, mask)
print(f"EncoderLayer 输入: {x.shape}")
print(f"EncoderLayer 输出: {output.shape}")
assert output.shape == x.shape
print("✅ EncoderLayer 形状验证通过！")


print("\n\n【测试】完整 Encoder (N=6) 功能验证")
print("-" * 80)

N = 6
encoder = Encoder(layer, N) # 传入一个层，它会自动复制6份

output_enc = encoder(x, mask)
print(f"Encoder (6层) 输入: {x.shape}")
print(f"Encoder (6层) 输出: {output_enc.shape}")
assert output_enc.shape == x.shape
print("✅ Encoder 堆叠验证通过！")

print("\n" + "=" * 80)
print("✅ Encoder 实现完成！")
print("=" * 80)