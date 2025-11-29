"""
23_train_reverse_task.py - 训练 Transformer 执行反转任务

目标：验证 Transformer 模型能否学会将输入序列反转
任务：输入 "1 2 3 4 5" -> 输出 "5 4 3 2 1"
优点：无需外部数据集，无需 torchtext，立即可运行
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 导入我们实现的 Transformer 模型
from transformer_model import make_model

print("=" * 80)
print("🎯 Transformer 反转任务训练 (Reverse Task)")
print("=" * 80)

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# =============================================================================
# 1. 数据生成器
# =============================================================================
def generate_random_batch(batch_size, seq_len, vocab_size):
    """
    生成随机整数序列及其反转序列
    """
    # 生成随机数据 [batch, seq_len]
    # 范围 [1, vocab_size-1] (0留给padding)
    data = torch.from_numpy(np.random.randint(1, vocab_size, size=(batch_size, seq_len))).long()
    
    # 源序列: data
    src = data
    
    # 目标序列: data的反转
    # 注意：我们需要添加 <bos> (开始) 和 <eos> (结束) 标记吗？
    # 为了简化，我们只用 <bos> 作为解码器的起始输入
    # src: [1, 2, 3]
    # tgt_input: [<bos>, 3, 2, 1] (作为Decoder输入)
    # tgt_out:   [3, 2, 1, <eos>] (作为标签)
    
    # 这里我们简化处理：
    # src: [1, 2, 3]
    # tgt: [0, 3, 2, 1] (0作为起始符)
    
    tgt = torch.zeros((batch_size, seq_len + 1), dtype=torch.long)
    tgt[:, 0] = 0 # <bos>
    tgt[:, 1:] = torch.flip(data, [1]) # 反转
    
    return src, tgt

# =============================================================================
# 2. 构建模型
# =============================================================================
VOCAB_SIZE = 11 # 0-10 (0是<bos>, 1-10是数字)
SEQ_LEN = 10
BATCH_SIZE = 32
D_MODEL = 64 # 小模型
N_HEAD = 2
N_LAYER = 2

# 注意：src_vocab 和 tgt_vocab 都要包含所有可能的数字
model = make_model(VOCAB_SIZE, VOCAB_SIZE, N=N_LAYER, d_model=D_MODEL, d_ff=256, h=N_HEAD, dropout=0.1)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================================================================
# 3. 辅助函数
# =============================================================================
def subsequent_mask(size):
    "生成上三角掩码"
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def make_std_mask(src, tgt, pad=0):
    "生成 Mask"
    # src_mask: [batch, 1, 1, src_len]
    # 这里假设没有 padding (除了可能的 0，但我们把 0 当作 bos)
    # 实际上我们的随机数据没有 padding，所以全是 True
    src_mask = (src != -1).unsqueeze(1).unsqueeze(2) # 全 True
    
    # tgt_mask: [batch, 1, tgt_len, tgt_len]
    tgt_mask = (tgt != -1).unsqueeze(1).unsqueeze(2) # 全 True
    tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
    
    return src_mask, tgt_mask

# =============================================================================
# 4. 训练循环
# =============================================================================
print("\n开始训练...")

# 增加训练轮数和每轮的步数
EPOCHS = 15
STEPS_PER_EPOCH = 100  # 每轮训练 100 个 batch
# 总共更新次数 = 15 * 100 = 1500 次 (之前只有 20 次)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for i in range(STEPS_PER_EPOCH):
        # 生成一批数据
        src, tgt = generate_random_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        
        tgt_input = tgt[:, :-1]
        tgt_real = tgt[:, 1:]
        
        src_mask, tgt_mask = make_std_mask(src, tgt_input)
        
        # 前向传播
        out = model(src, tgt_input, src_mask, tgt_mask)
        
        # Generator
        logits = model.generator(out)
        
        # 计算损失
        loss = criterion(logits.contiguous().view(-1, VOCAB_SIZE),
                         tgt_real.contiguous().view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / STEPS_PER_EPOCH
    print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")

print("训练完成！")

# =============================================================================
# 5. 推理测试
# =============================================================================
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len-1):
        out = model.decode(memory, src_mask, 
                           ys, 
                           subsequent_mask(ys.size(1)).type_as(src.data))
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

print("\n【推理测试】")
model.eval()
src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]).to(DEVICE)
src_mask = (src != -1).unsqueeze(1).unsqueeze(2)

print(f"输入序列: {src[0].cpu().numpy()}")
out = greedy_decode(model, src, src_mask, max_len=SEQ_LEN+1, start_symbol=0)
print(f"模型输出: {out[0].cpu().numpy()[1:]}") # 去掉起始符

expected = np.flip(src[0].cpu().numpy())
print(f"预期输出: {expected}")

if np.array_equal(out[0].cpu().numpy()[1:], expected):
    print("✅ 测试通过！模型成功学会了反转序列！")
else:
    print("❌ 测试失败。模型还需要更多训练。")