"""
Transformer基础概念验证代码
目标：用简单的代码示例理解张量、矩阵乘法、词嵌入和注意力机制
"""

import torch
import torch.nn as nn
import math
import numpy as np

print("=" * 60)
print("🎯 Transformer基础概念验证")
print("=" * 60)


# =============================================================================
# 1️⃣ 张量（Tensor）基础
# =============================================================================
print("\n【1】张量（Tensor）- 数据的容器")
print("-" * 60)

# 标量（0维）
scalar = torch.tensor(3.14)
print(f"标量: {scalar}, 形状: {scalar.shape}, 维度: {scalar.ndim}")

# 向量（1维）
vector = torch.tensor([1, 2, 3, 4])
print(f"向量: {vector}, 形状: {vector.shape}, 维度: {vector.ndim}")

# 矩阵（2维）
matrix = torch.tensor([[1, 2, 3],
                       [4, 5, 6]])
print(f"矩阵:\n{matrix}\n形状: {matrix.shape}, 维度: {matrix.ndim}")

# 3维张量（批次句子）
# 模拟：2个句子，每句3个词，每词4维向量
batch_sentences = torch.randn(2, 3, 4)
print(f"批次句子形状: {batch_sentences.shape}")
print(f"含义: [批次大小=2, 序列长度=3, 词向量维度=4]")
print(f"第1个句子的第1个词的向量:\n{batch_sentences[0, 0, :]}")


# =============================================================================
# 2️⃣ 矩阵乘法 - 信息变换
# =============================================================================
print("\n\n【2】矩阵乘法 - 信息的变换")
print("-" * 60)

# 例子：食谱转换器
print("例子：配方 → 营养成分")
recipe = torch.tensor([2.0, 100.0, 50.0])  # [鸡蛋数, 面粉克数, 糖克数]
print(f"输入配方: {recipe}")

# 转换矩阵：每种成分对应的营养价值
# 行1：蛋白质 [每个鸡蛋6g, 每100g面粉10g, 糖0g]
# 行2：碳水化合物 [鸡蛋0g, 面粉75g, 糖100g]
transform = torch.tensor([[6.0, 0.1, 0.0],      # 蛋白质
                          [0.0, 0.75, 1.0]])    # 碳水化合物
print(f"转换矩阵形状: {transform.shape}")

nutrition = torch.matmul(transform, recipe)
print(f"输出营养成分: {nutrition}")
print(f"解释: 蛋白质={nutrition[0]:.1f}g, 碳水={nutrition[1]:.1f}g")

# 在神经网络中
print("\n在神经网络中：")
batch_size, seq_len, d_model = 2, 3, 4
x = torch.randn(batch_size, seq_len, d_model)
W = torch.randn(d_model, 8)  # 4维 → 8维
y = torch.matmul(x, W)
print(f"输入: {x.shape} → 权重: {W.shape} → 输出: {y.shape}")


# =============================================================================
# 3️⃣ 词嵌入（Embedding）
# =============================================================================
print("\n\n【3】词嵌入（Embedding）- 词的数字化表示")
print("-" * 60)

# 模拟小词汇表
vocab_size = 10  # 10个词
embedding_dim = 4  # 每个词用4维向量表示

# 创建嵌入层
embedding = nn.Embedding(vocab_size, embedding_dim)
print(f"嵌入矩阵形状: {embedding.weight.shape}")
print(f"含义: {vocab_size}个词，每个词{embedding_dim}维向量")

# 输入句子（词的索引）
sentence_indices = torch.tensor([2, 5, 1, 8])  # "第2词 第5词 第1词 第8词"
print(f"\n输入句子（索引）: {sentence_indices}")

# 查表得到词向量
word_vectors = embedding(sentence_indices)
print(f"词向量矩阵形状: {word_vectors.shape}")
print(f"第一个词（索引2）的向量:\n{word_vectors[0]}")

# 相似度示例：看看哪些词向量相似
print("\n词向量之间的余弦相似度：")
vec_0 = word_vectors[0]  # 第1个词
vec_1 = word_vectors[1]  # 第2个词
vec_2 = word_vectors[2]  # 第3个词

cos_sim_01 = torch.cosine_similarity(vec_0, vec_1, dim=0)
cos_sim_02 = torch.cosine_similarity(vec_0, vec_2, dim=0)
print(f"词0和词1的相似度: {cos_sim_01:.3f}")
print(f"词0和词2的相似度: {cos_sim_02:.3f}")


# =============================================================================
# 4️⃣ 注意力机制 - 不同相似度计算方法对比
# =============================================================================
print("\n\n【4】注意力机制 - 相似度计算方法对比")
print("-" * 60)

# 模拟Query和Key
d_k = 64  # 向量维度
query = torch.randn(1, 1, d_k)  # [batch, 1个query, 维度]
keys = torch.randn(1, 5, d_k)   # [batch, 5个key, 维度]

print(f"Query形状: {query.shape}")
print(f"Keys形状: {keys.shape}")


# --- 方法1：标准点积（Transformer使用）---
print("\n方法1️⃣：缩放点积（Scaled Dot-Product）⭐")
scores_scaled = torch.matmul(query, keys.transpose(-2, -1)) / math.sqrt(d_k)    # 关于为什么用transpose(-2, -1)可以参考矩阵乘法的规则,并用-2和-1是为了适应更高维度的张量（防广播机制）
attention_scaled = torch.softmax(scores_scaled, dim=-1)
print(f"缩放前的分数: {torch.matmul(query, keys.transpose(-2, -1))}")
print(f"缩放后的分数: {scores_scaled}")
print(f"Softmax注意力权重: {attention_scaled}")
print(f"权重和: {attention_scaled.sum():.4f} (应该=1)")


# --- 方法2：不缩放的点积 ---
print("\n方法2️⃣：不缩放的点积（对比：为什么需要缩放？）")
scores_unscaled = torch.matmul(query, keys.transpose(-2, -1))
attention_unscaled = torch.softmax(scores_unscaled, dim=-1)
print(f"未缩放分数: {scores_unscaled}")
print(f"Softmax注意力权重: {attention_unscaled}")
print("⚠️ 注意：分数过大会导致softmax过于集中（梯度消失）")


# --- 方法3：余弦相似度 ---
print("\n方法3️⃣：余弦相似度")
query_norm = query / query.norm(dim=-1, keepdim=True)
keys_norm = keys / keys.norm(dim=-1, keepdim=True)
scores_cosine = torch.matmul(query_norm, keys_norm.transpose(-2, -1))
attention_cosine = torch.softmax(scores_cosine, dim=-1)
print(f"余弦相似度分数: {scores_cosine}")
print(f"Softmax注意力权重: {attention_cosine}")
print("✅ 余弦相似度固定在[-1, 1]范围内")


# --- 可视化对比 ---
print("\n📊 三种方法的注意力权重对比：")
print(f"{'方法':<15} | {'权重分布':<50} | {'最大权重'}")
print("-" * 80)
print(f"缩放点积      | {[f'{x:.3f}' for x in attention_scaled[0, 0].tolist()]} | {attention_scaled.max():.3f}")
print(f"未缩放点积    | {[f'{x:.3f}' for x in attention_unscaled[0, 0].tolist()]} | {attention_unscaled.max():.3f}")
print(f"余弦相似度    | {[f'{x:.3f}' for x in attention_cosine[0, 0].tolist()]} | {attention_cosine.max():.3f}")


# =============================================================================
# 5️⃣ 完整的简化注意力示例
# =============================================================================
print("\n\n【5】完整的简化注意力示例")
print("-" * 60)

# 模拟句子："cat sat mat"（猫坐垫子）
seq_len = 3
d_model = 4

# 简化：直接使用词向量作为Q, K, V
embeddings = torch.tensor([
    [1.0, 0.5, 0.3, 0.2],  # "cat" 的向量
    [0.3, 0.8, 0.5, 0.4],  # "sat" 的向量  
    [0.5, 0.3, 1.0, 0.6],  # "mat" 的向量
])

print("句子: 'cat sat mat'")
print(f"词向量:\n{embeddings}")

# 在实际Transformer中，Q、K、V是通过线性变换得到的
# 这里简化：直接用词向量
Q = embeddings
K = embeddings
V = embeddings

# 计算注意力
print("\n计算注意力：")
scores = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(d_model)
print(f"注意力分数矩阵:\n{scores}")
print("\n解释：每行表示一个词对所有词（包括自己）的注意力分数")
print(f"  第0行：'cat'对['cat', 'sat', 'mat']的分数")
print(f"  第1行：'sat'对['cat', 'sat', 'mat']的分数")
print(f"  第2行：'mat'对['cat', 'sat', 'mat']的分数")

attention_weights = torch.softmax(scores, dim=-1)
print(f"\nSoftmax后的注意力权重:\n{attention_weights}")
print("每行和=1，表示概率分布")

# 加权求和
output = torch.matmul(attention_weights, V)
print(f"\n输出（加权后的词表示）:\n{output}")
print("\n解释：每个词的新表示 = 融合了其他相关词的信息")


# =============================================================================
# 6️⃣ 维度影响实验
# =============================================================================
print("\n\n【6】维度对注意力的影响实验")
print("-" * 60)

print("实验：观察不同维度下，是否需要缩放")

for d in [4, 16, 64, 256]:
    q = torch.randn(1, 1, d)
    k = torch.randn(1, 5, d)
    
    # 未缩放
    scores_raw = torch.matmul(q, k.transpose(-2, -1))
    att_raw = torch.softmax(scores_raw, dim=-1)
    
    # 缩放
    scores_scaled = scores_raw / math.sqrt(d)
    att_scaled = torch.softmax(scores_scaled, dim=-1)
    
    print(f"\n维度 d={d}:")
    print(f"  未缩放分数范围: [{scores_raw.min():.2f}, {scores_raw.max():.2f}]")
    print(f"  缩放后分数范围: [{scores_scaled.min():.2f}, {scores_scaled.max():.2f}]")
    print(f"  未缩放最大注意力: {att_raw.max():.4f}")
    print(f"  缩放后最大注意力: {att_scaled.max():.4f}")
    if att_raw.max() > 0.9:
        print(f"  ⚠️  未缩放导致注意力过于集中！")


print("\n" + "=" * 60)
print("✅ 基础概念验证完成！")
print("=" * 60)
print("\n💡 关键要点回顾：")
print("1. 张量是多维数组，形状决定了数据的组织方式")
print("2. 矩阵乘法是信息变换的核心操作")
print("3. 词嵌入将离散的词转换为连续的向量")
print("4. 注意力机制让每个词能够关注其他相关的词")
print("5. 缩放点积是最优选择：快速、有效、数学优雅")
print("6. 除以√d_k防止维度增加导致的梯度消失问题")