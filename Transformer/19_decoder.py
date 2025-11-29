"""
19_decoder.py - Decoder Layer 和 Decoder 实现

目标：组装 Transformer 的 Decoder 部分
包含：
1. DecoderLayer: 单个解码器层 (Masked Self-Attention + Cross-Attention + FFN)
2. Decoder: N 层堆叠的解码器
"""

import torch
import torch.nn as nn
import copy
import math
import torch.nn.functional as F
import numpy as np

# --- 组件复用 (与 17_encoder.py 相同) ---
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
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "生成上三角掩码，用于遮蔽未来位置"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
# --- 组件复用结束 ---


print("=" * 80)
print("🎯 Decoder Layer 与 Decoder 实现与测试")
print("=" * 80)


class DecoderLayer(nn.Module):
    """
    DecoderLayer 由三个子层组成：
    1. Masked Self-Attention
    2. Cross-Attention (Source-Target Attention)
    3. FeedForward
    """
    
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        # 三个子层连接器
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
        print(f"✅ DecoderLayer初始化: size={size}")

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        Args:
            x: Decoder 输入 (目标序列) [batch, tgt_len, d_model]
            memory: Encoder 输出 (源序列) [batch, src_len, d_model]
            src_mask: 源序列掩码 (Padding Mask)
            tgt_mask: 目标序列掩码 (Look-ahead Mask + Padding Mask)
        """
        m = memory
        
        # 1. Masked Self-Attention (只能看自己之前的词)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        
        # 2. Cross-Attention (查询 Encoder 输出)
        # Q = x (Decoder), K = m (Encoder), V = m (Encoder)
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        
        # 3. Feed-Forward
        x = self.sublayer[2](x, self.feed_forward)
        
        return x


class Decoder(nn.Module):
    """
    Decoder 是 N 个 DecoderLayer 的堆叠
    """
    
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
        print(f"✅ Decoder初始化: {N} 层堆叠")
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# =============================================================================
# 测试代码
# =============================================================================
print("\n\n【测试】Mask 生成功能")
print("-" * 80)

mask = subsequent_mask(5)
print(f"Look-ahead Mask (size=5):\n{mask[0]}")
print("True 表示可见，False 表示被遮蔽")


print("\n\n【测试】DecoderLayer 功能验证")
print("-" * 80)

d_model = 512
heads = 8
d_ff = 2048
dropout = 0.1

# 实例化组件
self_attn = MultiHeadAttention(d_model, heads)
src_attn = MultiHeadAttention(d_model, heads)
ff = PositionWiseFeedForward(d_model, d_ff)
layer = DecoderLayer(d_model, self_attn, src_attn, ff, dropout)

# 创建输入
tgt = torch.randn(2, 10, d_model) # 目标序列 (batch=2, len=10)
memory = torch.randn(2, 15, d_model) # 源序列 (batch=2, len=15) - 长度可以不同！

# 创建掩码
src_mask = torch.ones(2, 1, 1, 15) # 源序列全可见
tgt_mask = subsequent_mask(10) # 目标序列遮蔽未来

output = layer(tgt, memory, src_mask, tgt_mask)
print(f"DecoderLayer 输入 (tgt): {tgt.shape}")
print(f"DecoderLayer 输出: {output.shape}")
assert output.shape == tgt.shape
print("✅ DecoderLayer 形状验证通过！")


print("\n\n【测试】完整 Decoder (N=6) 功能验证")
print("-" * 80)

N = 6
decoder = Decoder(layer, N)

output_dec = decoder(tgt, memory, src_mask, tgt_mask)
print(f"Decoder (6层) 输入: {tgt.shape}")
print(f"Decoder (6层) 输出: {output_dec.shape}")
assert output_dec.shape == tgt.shape
print("✅ Decoder 堆叠验证通过！")

print("\n" + "=" * 80)
print("✅ Decoder 实现完成！")
print("=" * 80)