# Encoder Layer 数据流向详解

## 🎯 核心结构
一个 Encoder Layer 由两个子层组成：
1.  **Multi-Head Self-Attention**
2.  **Position-wise Feed-Forward Network**

每个子层后面都接了一个 **Add & Norm**。

---

## 🔄 数据流向步骤

假设输入为 `x`，形状为 `[batch, seq_len, d_model]`。

### 第一步：Self-Attention 子层
1.  **输入**: `x`
2.  **Attention 计算**:
    *   `x` 同时作为 Q, K, V 输入到 Multi-Head Attention。
    *   `attn_output = MultiHeadAttention(x, x, x)`
    *   *作用*：捕捉序列内部的依赖关系（上下文信息）。
3.  **Add & Norm**:
    *   `x = LayerNorm(x + attn_output)`
    *   *作用*：残差连接防止梯度消失，归一化稳定分布。

### 第二步：Feed-Forward 子层
1.  **输入**: 上一步的输出 `x`
2.  **FFN 计算**:
    *   `ffn_output = FeedForward(x)`
    *   *作用*：对每个位置的特征进行非线性变换和提取。
3.  **Add & Norm**:
    *   `x = LayerNorm(x + ffn_output)`
    *   *作用*：再次稳定训练。

### 最终输出
*   输出形状仍为 `[batch, seq_len, d_model]`。
*   这意味着 Encoder Layer 的输入和输出形状完全一致，可以像积木一样无限堆叠！

---

## 📝 伪代码预览

```python
class EncoderLayer(nn.Module):
    def forward(self, x, mask):
        # 1. Self-Attention + Add & Norm
        # 注意：这里用 lambda 封装是为了配合 SublayerConnection
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        
        # 2. Feed-Forward + Add & Norm
        x = self.sublayer[1](x, self.feed_forward)
        
        return x
```

---

## 🎓 关键点检查

1.  **为什么 Self-Attention 的输入全是 x？**
    *   因为这是 **Self**-Attention，查询(Q)、键(K)、值(V)都来自同一个序列（即上一层的输出）。

2.  **Mask 在这里有什么用？**
    *   在 Encoder 中，Mask 主要用于**Padding Mask**。如果句子长度不一，我们需要把填充的部分（pad token）遮住，不让 Attention 关注到它们。

3.  **Add & Norm 是在子层前还是后？**
    *   原始论文是在**后** (Post-Norm): `Norm(x + Sublayer(x))`
    *   现在的实践（如 GPT-2/3）常放在**前** (Pre-Norm): `x + Sublayer(Norm(x))`
    *   我们在 `15_layer_norm.py` 中实现的是 Post-Norm，为了忠实于原论文。