"""
Self-Attention 完整执行流程示例
通过一个具体例子，逐步展示数据如何流动：输入 → Q,K,V → 注意力分数 → 输出
"""

import torch
import torch.nn as nn
import math
import numpy as np

print("=" * 80)
print("🎯 Self-Attention 完整执行流程示例")
print("=" * 80)

# =============================================================================
# 设置：一个简单的句子
# =============================================================================
print("\n【步骤 0】准备输入数据")
print("-" * 80)

# 句子："I love AI"（3个词）
sentence = ["I", "love", "AI"]
seq_len = 3  # 序列长度
d_model = 4  # 每个词的向量维度（简化为4维便于理解）

print(f"句子: {sentence}")
print(f"序列长度: {seq_len}")
print(f"词向量维度: {d_model}")

# 模拟词嵌入：每个词都用一个4维向量表示
# 在实际应用中，这些向量是通过Embedding层学习得到的
X = torch.tensor([
    [1.0, 0.0, 0.5, 0.2],  # "I" 的向量
    [0.0, 1.0, 0.3, 0.8],  # "love" 的向量
    [0.5, 0.5, 1.0, 0.1],  # "AI" 的向量
], dtype=torch.float32)

print(f"\n输入矩阵 X 的形状: {X.shape}  # [序列长度, 词向量维度]")
print("输入矩阵 X (每行是一个词的向量):")
print(X)
print("\n解释：")
print(f"  X[0] = {X[0].tolist()} ← 'I' 的向量")
print(f"  X[1] = {X[1].tolist()} ← 'love' 的向量")
print(f"  X[2] = {X[2].tolist()} ← 'AI' 的向量")


# =============================================================================
# 步骤 1：创建 Q, K, V 投影矩阵
# =============================================================================
print("\n\n【步骤 1】创建 Q, K, V 投影矩阵")
print("-" * 80)

d_k = d_model  # Query和Key的维度（这里简化为和d_model相同）
d_v = d_model  # Value的维度

print("💡 为什么需要投影矩阵？")
print("   虽然Q, K, V都来自同一个输入X，但我们希望它们在不同的")
print("   '表示空间'中学习，就像从不同角度观察同一个事物。")
print()

# 创建权重矩阵（随机初始化，实际训练中会学习）
torch.manual_seed(42)  # 固定随机种子，保证结果可复现

W_q = torch.randn(d_model, d_k) * 0.5  # Query权重矩阵
W_k = torch.randn(d_model, d_k) * 0.5  # Key权重矩阵  
W_v = torch.randn(d_model, d_v) * 0.5  # Value权重矩阵

print(f"W_q (Query权重) 形状: {W_q.shape}  # [d_model, d_k] = [{d_model}, {d_k}]")
print(f"W_k (Key权重) 形状:   {W_k.shape}  # [d_model, d_k] = [{d_model}, {d_k}]")
print(f"W_v (Value权重) 形状: {W_v.shape}  # [d_model, d_v] = [{d_model}, {d_v}]")

print("\nW_q 矩阵内容:")
print(W_q)


# =============================================================================
# 步骤 2：计算 Q, K, V
# =============================================================================
print("\n\n【步骤 2】通过矩阵乘法计算 Q, K, V")
print("-" * 80)

Q = torch.matmul(X, W_q)  # [seq_len, d_model] @ [d_model, d_k] = [seq_len, d_k]
K = torch.matmul(X, W_k)  # [seq_len, d_model] @ [d_model, d_k] = [seq_len, d_k]
V = torch.matmul(X, W_v)  # [seq_len, d_model] @ [d_model, d_v] = [seq_len, d_v]

print(f"Q (Query矩阵) 形状: {Q.shape}  # [seq_len, d_k] = [{seq_len}, {d_k}]")
print("Q 矩阵内容 (每行是一个词的Query向量):")
print(Q)
print(f"\n解释：Q[0] = {Q[0].tolist()[:2]}... 是'I'的Query向量")
print(f"     Q[1] = {Q[1].tolist()[:2]}... 是'love'的Query向量")
print(f"     Q[2] = {Q[2].tolist()[:2]}... 是'AI'的Query向量")

print(f"\n\nK (Key矩阵) 形状: {K.shape}  # [seq_len, d_k] = [{seq_len}, {d_k}]")
print("K 矩阵内容 (每行是一个词的Key向量):")
print(K)

print(f"\n\nV (Value矩阵) 形状: {V.shape}  # [seq_len, d_v] = [{seq_len}, {d_v}]")
print("V 矩阵内容 (每行是一个词的Value向量):")
print(V)

print("\n💡 理解Q, K, V的角色：")
print("   - Query(Q): '我在寻找什么信息？' (查询者)")
print("   - Key(K):   '我能提供什么信息？' (被查询者的标识)")
print("   - Value(V): '这是我要传递的具体信息' (实际内容)")


# =============================================================================
# 步骤 3：计算注意力分数（相似度）
# =============================================================================
print("\n\n【步骤 3】计算注意力分数（Query和Key的相似度）")
print("-" * 80)

# 点积：Q @ K^T
scores = torch.matmul(Q, K.transpose(0, 1))  # [seq_len, d_k] @ [d_k, seq_len] = [seq_len, seq_len]

print(f"分数矩阵 (QK^T) 形状: {scores.shape}  # [seq_len, seq_len] = [{seq_len}, {seq_len}]")
print("未缩放的分数矩阵:")
print(scores)
print("\n🔍 矩阵含义（每个位置表示一个词对另一个词的相似度）:")
print(f"              {'I':>8}  {'love':>8}  {'AI':>8}")
print(f"  I     {scores[0, 0]:>8.3f}  {scores[0, 1]:>8.3f}  {scores[0, 2]:>8.3f}  ← 'I'对所有词的分数")
print(f"  love  {scores[1, 0]:>8.3f}  {scores[1, 1]:>8.3f}  {scores[1, 2]:>8.3f}  ← 'love'对所有词的分数")
print(f"  AI    {scores[2, 0]:>8.3f}  {scores[2, 1]:>8.3f}  {scores[2, 2]:>8.3f}  ← 'AI'对所有词的分数")

# 缩放（除以sqrt(d_k)）
scaled_scores = scores / math.sqrt(d_k)

print(f"\n缩放后的分数 (除以√{d_k} ≈ {math.sqrt(d_k):.2f}):")
print(scaled_scores)
print("\n💡 为什么要缩放？")
print("   防止维度增大时，点积值过大导致softmax梯度消失")


# =============================================================================
# 步骤 4：应用 Softmax 得到注意力权重
# =============================================================================
print("\n\n【步骤 4】应用 Softmax 将分数转换为概率分布")
print("-" * 80)

attention_weights = torch.softmax(scaled_scores, dim=-1)  # 在最后一维上做softmax

print(f"注意力权重矩阵形状: {attention_weights.shape}  # [{seq_len}, {seq_len}]")
print("注意力权重矩阵 (每行和为1):")
print(attention_weights)

print("\n🔍 注意力权重解读（概率分布）:")
print(f"              {'I':>8}  {'love':>8}  {'AI':>8}  {'行和':>8}")
row_sums = attention_weights.sum(dim=-1)
for i, word in enumerate(sentence):
    weights_str = "  ".join([f"{attention_weights[i, j]:>8.3f}" for j in range(seq_len)])
    print(f"  {word:<4}  {weights_str}  {row_sums[i]:>8.3f}")

print("\n💡 解释：")
print(f"   'I' 最关注 '{sentence[int(torch.argmax(attention_weights[0]).item())]}' (权重={attention_weights[0].max():.3f})")
print(f"   'love' 最关注 '{sentence[int(torch.argmax(attention_weights[1]).item())]}' (权重={attention_weights[1].max():.3f})")
print(f"   'AI' 最关注 '{sentence[int(torch.argmax(attention_weights[2]).item())]}' (权重={attention_weights[2].max():.3f})")


# =============================================================================
# 步骤 5：加权求和得到输出
# =============================================================================
print("\n\n【步骤 5】用注意力权重对Value进行加权求和")
print("-" * 80)

output = torch.matmul(attention_weights, V)  # [seq_len, seq_len] @ [seq_len, d_v] = [seq_len, d_v]

print(f"输出矩阵形状: {output.shape}  # [seq_len, d_v] = [{seq_len}, {d_v}]")
print("输出矩阵内容:")
print(output)

print("\n🔍 详细计算过程（以'love'为例）:")
print(f"\n   'love'的输出 = 加权求和所有词的Value:")
print(f"   output[1] = {attention_weights[1, 0]:.3f} * V[0]  (权重 × 'I'的value)")
print(f"             + {attention_weights[1, 1]:.3f} * V[1]  (权重 × 'love'的value)")
print(f"             + {attention_weights[1, 2]:.3f} * V[2]  (权重 × 'AI'的value)")
print(f"\n   计算结果:")

manual_calc = (attention_weights[1, 0] * V[0] + 
               attention_weights[1, 1] * V[1] + 
               attention_weights[1, 2] * V[2])
print(f"   手动计算: {manual_calc.tolist()}")
print(f"   矩阵计算: {output[1].tolist()}")
print(f"   验证一致: {torch.allclose(manual_calc, output[1])}")


# =============================================================================
# 步骤 6：对比输入和输出
# =============================================================================
print("\n\n【步骤 6】对比输入和输出，理解Self-Attention的作用")
print("-" * 80)

print("对比每个词的输入和输出向量：\n")

for i, word in enumerate(sentence):
    print(f"词: '{word}'")
    print(f"  输入向量:  {X[i].tolist()}")
    print(f"  输出向量:  {output[i].tolist()}")
    
    # 计算变化
    diff = output[i] - X[i]
    print(f"  变化:      {diff.tolist()}")
    print(f"  变化幅度:  {torch.norm(diff).item():.4f}")
    print()

print("💡 Self-Attention做了什么？")
print("   1. 每个词的输出 = 融合了其他相关词的信息")
print("   2. 注意力权重决定了融合的比例")
print("   3. 输出向量包含了更丰富的上下文信息")


# =============================================================================
# 步骤 7：完整流程总结
# =============================================================================
print("\n\n【步骤 7】完整数据流总结")
print("=" * 80)

print("""
完整的Self-Attention计算流程：

输入: X                     形状: [3, 4]   (3个词，每词4维)
  ↓
投影变换:
  Q = X @ W_q              形状: [3, 4]   (Query矩阵)
  K = X @ W_k              形状: [3, 4]   (Key矩阵)
  V = X @ W_v              形状: [3, 4]   (Value矩阵)
  ↓
计算相似度:
  scores = Q @ K^T         形状: [3, 3]   (每个词对其他词的分数)
  ↓
缩放:
  scaled_scores = scores / √d_k
  ↓
归一化:
  attention = softmax(scaled_scores)  形状: [3, 3]   (注意力权重，每行和为1)
  ↓
加权求和:
  output = attention @ V   形状: [3, 4]   (融合上下文后的输出)

关键点：
✓ Q, K, V 都来自同一个输入 X（这就是"Self"的含义）
✓ 注意力权重矩阵是方阵 [seq_len, seq_len]
✓ 每个词都可以关注包括自己在内的所有词
✓ 输出维度和输入维度相同 [seq_len, d_model]
""")


# =============================================================================
# 步骤 8：可视化注意力热力图（简化版）
# =============================================================================
print("\n\n【步骤 8】注意力热力图（文本版）")
print("-" * 80)

print("\n注意力权重可视化（数字越大，关注度越高）:")
print(f"\n{'':>8}", end="")
for word in sentence:
    print(f"{word:>8}", end="")
print()

for i, word in enumerate(sentence):
    print(f"{word:>8}", end="")
    for j in range(seq_len):
        weight = attention_weights[i, j].item()
        # 用符号表示权重大小
        if weight > 0.5:
            symbol = "████"
        elif weight > 0.3:
            symbol = "███ "
        elif weight > 0.2:
            symbol = "██  "
        else:
            symbol = "█   "
        print(f"{symbol}", end="")
    print(f"  (最关注: {sentence[int(torch.argmax(attention_weights[i]).item())]})")


print("\n" + "=" * 80)
print("✅ Self-Attention 执行流程示例完成！")
print("=" * 80)

print("\n🎓 核心要点回顾:")
print("1. Self-Attention让每个词能看到序列中的所有词（包括自己）")
print("2. Q, K, V通过不同的投影矩阵从同一输入生成")
print("3. 注意力权重 = softmax(QK^T/√d_k) 表示词之间的相关性")
print("4. 输出 = 注意力权重加权求和Value，融合上下文信息")
print("5. 输入和输出形状相同，但输出包含了更丰富的语义")