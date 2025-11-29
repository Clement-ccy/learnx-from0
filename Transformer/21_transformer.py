"""
21_transformer.py - 完整 Transformer 模型实现

目标：组装 Encoder, Decoder, Generator, Embedding, PositionalEncoding
包含：
1. Transformer: 主模型类
2. make_model: 构建模型的辅助函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import numpy as np

# --- 组件复用 (为了完整性，这里包含了所有之前的组件) ---
# 实际项目中建议拆分文件

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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.pe: torch.Tensor
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class Generator(nn.Module):
    "标准的线性层 + Softmax"
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size
    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# --- 组件复用结束 ---


print("=" * 80)
print("🎯 完整 Transformer 模型实现与测试")
print("=" * 80)


class Transformer(nn.Module):
    """
    标准的 Encoder-Decoder 架构
    """
    
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
        print("✅ Transformer 模型初始化完成")
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Args:
            src: 源序列 [batch, src_len]
            tgt: 目标序列 [batch, tgt_len]
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
        """
        # 1. 编码
        memory = self.encode(src, src_mask)
        
        # 2. 解码
        output = self.decode(memory, src_mask, tgt, tgt_mask)
        
        return output
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    构建模型的辅助函数：一次性实例化所有组件
    """
    c = copy.deepcopy
    
    # 实例化基础组件
    attn = MultiHeadAttention(d_model, h)
    ff = PositionWiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    
    # 构建模型
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab)
    )
    
    # 初始化参数 (Xavier Initialization)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return model


# =============================================================================
# 测试代码
# =============================================================================
print("\n\n【测试】构建完整模型")
print("-" * 80)

src_vocab = 100
tgt_vocab = 100
N = 2 # 简化层数
d_model = 512

model = make_model(src_vocab, tgt_vocab, N=N, d_model=d_model)
print(f"模型结构:\n{model}")


print("\n\n【测试】前向传播")
print("-" * 80)

# 模拟输入
batch_size = 2
src_len = 10
tgt_len = 10

src = torch.randint(0, src_vocab, (batch_size, src_len))
tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

print(f"源序列 src: {src.shape}")
print(f"目标序列 tgt: {tgt.shape}")

# 创建简单的掩码
src_mask = torch.ones(batch_size, 1, 1, src_len)
tgt_mask = torch.ones(batch_size, 1, tgt_len, tgt_len) # 这里简化了，实际需要上三角mask

# 运行模型
output = model(src, tgt, src_mask, tgt_mask)
print(f"Decoder 输出: {output.shape}")

# 运行 Generator
probs = model.generator(output)
print(f"Generator 输出 (Logits): {probs.shape}")

assert probs.shape == (batch_size, tgt_len, tgt_vocab)
print("✅ 最终输出形状验证通过！")

print("\n" + "=" * 80)
print("🎉 恭喜！你已经成功从零实现了完整的 Transformer！")
print("=" * 80)