# Multi-Head Attention 维度变换流程图解

## 🎯 目标
将输入 `[batch, seq_len, d_model]` 转换为适合并行计算的形状，计算注意力后，再还原回原形状。

假设参数：
- `batch_size = 2`
- `seq_len = 10` (句子长度)
- `d_model = 512` (总维度)
- `num_heads = 8` (头数)
- `d_head = 64` (每个头的维度 = 512 / 8)

---

## 🔄 阶段 1：分割 (Split) - 准备并行计算

我们希望把 `d_model` (512) 拆分成 `num_heads` (8) × `d_head` (64)。

### 1. 线性投影 (Linear Projection)
输入 `x` 经过 `W_q` 线性层。
- 输入: `[batch, seq_len, d_model]` → `[2, 10, 512]`
- 操作: `x @ W_q`
- 输出: `[batch, seq_len, d_model]` → `[2, 10, 512]`
*(注意：这里虽然还是512，但逻辑上已经包含了8个头的信息)*

### 2. 维度重塑 (Reshape / View)
把最后一维拆开。
- 输入: `[batch, seq_len, d_model]`
- 操作: `.view(batch, seq_len, num_heads, d_head)`
- 输出: `[batch, seq_len, num_heads, d_head]` → `[2, 10, 8, 64]`

### 3. 维度转置 (Transpose) ⭐ 关键步骤
为了让 `num_heads` 独立并行，我们需要把它放到前面去，和 `batch` 维度在一起。
这样 PyTorch 就会把 `(batch, num_heads)` 看作是一个大的"批次"，一次性并行处理所有头。
- 输入: `[batch, seq_len, num_heads, d_head]`
- 操作: `.transpose(1, 2)` (交换第1和第2维)
- 输出: `[batch, num_heads, seq_len, d_head]` → `[2, 8, 10, 64]`

**现在，Q, K, V 都在这个形状下准备好了！**

---

## ⚡ 阶段 2：注意力计算 (Attention)

在这个阶段，我们只关心最后两个维度 `[seq_len, d_head]`。前面的 `[batch, num_heads]` 只是作为独立的样本处理。

### 1. 计算分数 (Scores)
- Q: `[..., seq_len, d_head]` → `[..., 10, 64]`
- K.T: `[..., d_head, seq_len]` → `[..., 64, 10]`
- 操作: `matmul(Q, K.T)`
- 输出: `[batch, num_heads, seq_len, seq_len]` → `[2, 8, 10, 10]`
*(看！我们得到了 8 个 10x10 的注意力矩阵，每个头一个！)*

### 2. 加权求和 (Weighted Sum)
- Attention: `[..., seq_len, seq_len]`
- V: `[..., seq_len, d_head]`
- 操作: `matmul(Attention, V)`
- 输出: `[batch, num_heads, seq_len, d_head]` → `[2, 8, 10, 64]`

---

## 🔄 阶段 3：合并 (Concat) - 还原形状

现在我们需要把 8 个头的结果拼回去。

### 1. 维度转置 (Transpose Back)
先把 `seq_len` 换回来。
- 输入: `[batch, num_heads, seq_len, d_head]`
- 操作: `.transpose(1, 2)`
- 输出: `[batch, seq_len, num_heads, d_head]` → `[2, 10, 8, 64]`

### 2. 维度重塑 (Reshape / Contiguous View)
把最后两维合并。
- 输入: `[batch, seq_len, num_heads, d_head]`
- 操作: `.contiguous().view(batch, seq_len, d_model)`
- 输出: `[batch, seq_len, d_model]` → `[2, 10, 512]`
*(注意：transpose后内存不连续，必须先调 `.contiguous()` 才能调 `.view()`)*

### 3. 最终线性投影 (Final Linear)
混合所有头的信息。
- 输入: `[batch, seq_len, d_model]`
- 操作: `x @ W_o`
- 输出: `[batch, seq_len, d_model]` → `[2, 10, 512]`

---

## 📝 代码逻辑预览

```python
def forward(self, x):
    batch_size = x.size(0)
    
    # 1. 线性投影 + 分头
    # [batch, seq, d_model] -> [batch, seq, heads, d_head] -> [batch, heads, seq, d_head]
    Q = self.w_q(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)
    K = self.w_k(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)
    V = self.w_v(x).view(batch_size, -1, self.heads, self.d_head).transpose(1, 2)
    
    # 2. 注意力计算 (PyTorch会自动广播 batch 和 heads 维度)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
    attn = torch.softmax(scores, dim=-1)
    context = torch.matmul(attn, V) 
    # context形状: [batch, heads, seq, d_head]
    
    # 3. 合并头
    # [batch, heads, seq, d_head] -> [batch, seq, heads, d_head] -> [batch, seq, d_model]
    context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
    
    # 4. 最终输出
    output = self.w_o(context)
    return output
```

---

## 🎓 检查点

1.  **为什么要 `transpose(1, 2)`？**
    为了让 `heads` 维度和 `batch` 维度相邻，这样 `matmul` 就可以把它们当作一个大的批次并行处理，而只在最后两维（`seq`, `d_head`）进行矩阵乘法。

2.  **为什么要 `.contiguous()`？**
    `transpose` 操作只是改变了步长（stride），并没有改变内存中的实际存储顺序。`.view` 需要内存是连续的，所以必须先调用 `.contiguous()` 重新整理内存。