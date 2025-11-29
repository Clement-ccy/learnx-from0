# Decoder 原理详解 - 为什么需要"戴眼罩"？

## 🎯 核心区别
Encoder 和 Decoder 最大的区别在于：
*   **Encoder**：一次性看到整个句子（"上帝视角"）。它可以利用上下文（前后文）来理解每个词。
*   **Decoder**：在推理（生成）时，是一个词一个词生成的（自回归）。它**只能看到已经生成的词**，绝对不能看到未来的词。

---

## 1️⃣ Masked Self-Attention (遮蔽自注意力)

### 💡 为什么需要 Mask？
在训练时，为了并行计算，我们是一次性把整个目标句子（比如 "I love AI"）喂给 Decoder 的。
但是，如果我们不加限制：
*   当模型预测 "I" 的下一个词时，它可以通过 Self-Attention 直接"偷看"到后面的 "love"。
*   这就像考试时把答案直接放在题目旁边，模型根本不需要学习，直接抄答案就行了。
*   结果：训练时 Loss 很低，但推理时（没有未来信息）完全不会生成。

### 🛠️ 怎么做 Mask？
我们需要一个**上三角矩阵**（Upper Triangular Matrix）作为 Mask。

假设句子是 "A B C"：

| | A | B | C |
|---|---|---|---|
| **生成 A 时** | ✅ 看 A | ❌ 遮住 | ❌ 遮住 |
| **生成 B 时** | ✅ 看 A | ✅ 看 B | ❌ 遮住 |
| **生成 C 时** | ✅ 看 A | ✅ 看 B | ✅ 看 C |

在 Attention 分数矩阵中，我们将被遮住的位置（上三角部分）填为 **负无穷大 (-inf)**。
这样经过 Softmax 后，这些位置的概率就变成了 **0**。

---

## 2️⃣ Cross-Attention (Encoder-Decoder Attention)

这是 Decoder 的第二个 Attention 层，也是连接 Encoder 和 Decoder 的桥梁。

### 🔄 数据流向
*   **Query (Q)**: 来自 **Decoder**（上一层 Masked Self-Attention 的输出）。
    *   代表："我现在生成到了这一步，我想查询源句子里的相关信息"。
*   **Key (K) & Value (V)**: 来自 **Encoder**（最后一层的输出）。
    *   代表："这是源句子的所有信息，请随意查询"。

### 💡 直觉理解
想象你在做翻译（中译英）：
1.  **Masked Self-Attention**: 你看着已经写下的英文单词（比如 "I"），思考语法结构，觉得下一个词应该是动词。
2.  **Cross-Attention**: 你带着"找动词"的意图（Query），去读中文原句（Key/Value），把注意力集中在"爱"这个字上。
3.  **FFN**: 综合信息，决定写下 "love"。

---

## 3️⃣ Decoder Layer 结构总结

一个 Decoder Layer 包含 **三个** 子层：

1.  **Masked Self-Attention**
    *   输入：Decoder 上一层的输出
    *   Mask：Look-ahead Mask (遮住未来)
2.  **Cross-Attention**
    *   输入：Q=Decoder, K=Encoder, V=Encoder
    *   Mask：Padding Mask (遮住源句子的填充部分)
3.  **Feed-Forward Network**
    *   输入：Cross-Attention 的输出

每个子层后都有 **Add & Norm**。

---

## 🎓 理解检查

1.  **训练和推理时，Decoder 的行为一样吗？**
    *   **训练时**：并行计算。输入是完整的目标句子，通过 Mask 模拟"看不到未来"。
    *   **推理时**：串行计算。生成一个词，把它加到输入里，再生成下一个词。

2.  **Cross-Attention 需要 Mask 吗？**
    *   **需要**，但只需要 Padding Mask（遮住源句子的 pad token），不需要 Look-ahead Mask（因为源句子是完全可见的）。

---

## 🚀 下一步

现在我们来实现 `DecoderLayer` 和 `Decoder`。代码结构和 Encoder 很像，只是多了一个 Attention 层。