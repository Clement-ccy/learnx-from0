"""
22_train_translation.py - 使用 Multi30k 数据集训练 Transformer

目标：在德语->英语翻译任务上训练我们实现的 Transformer 模型
"""

import torch
import torchtext
torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List

# 导入我们实现的 Transformer 模型
from transformer_model import make_model

print("=" * 80)
print("🎯 Transformer 翻译任务训练 (Multi30k: De -> En)")
print("=" * 80)

# 设置随机种子
torch.manual_seed(0)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# =============================================================================
# 1. 数据预处理
# =============================================================================
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# 定义分词器
tokenizers = {}
tokenizers[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
tokenizers[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# 构建词表的辅助函数
def yield_tokens(data_iter: Iterable, language: str) -> Iterable[List[str]]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for data_sample in data_iter:
        yield tokenizers[language](data_sample[language_index[language]])

# 定义特殊符号
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# 构建词表
vocab_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # 训练集迭代器
    # 注意：Multi30k API 可能需要 root 参数，这里尝试不加，如果报错再加
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    
    # 创建词表
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(train_iter, ln),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )
    
    # 设置默认索引为 <unk>
    vocab_transform[ln].set_default_index(UNK_IDX)

print(f"德语词表大小: {len(vocab_transform[SRC_LANGUAGE])}")
print(f"英语词表大小: {len(vocab_transform[TGT_LANGUAGE])}")

# =============================================================================
# 2. 数据加载器 (DataLoader)
# =============================================================================
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

# 文本转换流水线：分词 -> 数值化 -> 添加BOS/EOS
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(tokenizers[ln], # 分词
                                               vocab_transform[ln], # 数值化
                                               tensor_transform) # 添加特殊token

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    # 填充 (Padding)
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

BATCH_SIZE = 16 # 显存不够可以调小
train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# =============================================================================
# 3. 构建模型
# =============================================================================
SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

print(f"初始化模型参数: SRC_VOCAB={SRC_VOCAB_SIZE}, TGT_VOCAB={TGT_VOCAB_SIZE}")
transformer = make_model(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE,
                         N=NUM_ENCODER_LAYERS,
                         d_model=EMB_SIZE,
                         d_ff=FFN_HID_DIM,
                         h=NHEAD,
                         dropout=0.1)

print(f"Generator 输出层权重形状: {transformer.generator.proj.weight.shape}")
transformer = transformer.to(DEVICE)

# 初始化参数
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

# 定义损失函数 (忽略 padding)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# 定义优化器
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# =============================================================================
# 4. 训练循环
# =============================================================================
def subsequent_mask(size):
    "生成上三角掩码，用于遮蔽未来位置"
    import numpy as np
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    # Source Padding Mask: [batch, 1, 1, src_len]
    src_mask = (src != PAD_IDX).unsqueeze(1).unsqueeze(2)
    
    # Target Padding Mask: [batch, 1, tgt_len, 1] -> 广播
    tgt_pad_mask = (tgt != PAD_IDX).unsqueeze(1).unsqueeze(2)
    
    # Target Look-ahead Mask: [1, 1, tgt_len, tgt_len]
    look_ahead_mask = subsequent_mask(tgt_seq_len).to(DEVICE)
    
    # 组合 Padding Mask 和 Look-ahead Mask
    tgt_mask = tgt_pad_mask & look_ahead_mask
    
    return src_mask, tgt_mask

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    count = 0
    
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        # tgt_input: 去掉最后一个词 (作为输入)
        tgt_input = tgt[:, :-1]
        
        # tgt_out: 去掉第一个词 (作为标签)
        tgt_out = tgt[:, 1:]

        src_mask, tgt_mask = create_mask(src, tgt_input)

        # 前向传播
        out = model(src, tgt_input, src_mask, tgt_mask)
        
        # Generator: 将 Decoder 输出映射到词表大小
        logits = model.generator(out)

        optimizer.zero_grad()

        # 计算损失
        # logits: [batch, seq_len, vocab_size] -> [batch*seq_len, vocab_size]
        # tgt_out: [batch, seq_len] -> [batch*seq_len]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        loss.backward()
        optimizer.step()
        losses += loss.item()
        count += 1

    return losses / count if count > 0 else 0

# =============================================================================
# 5. 开始训练
# =============================================================================
NUM_EPOCHS = 5 # 演示用，实际可能需要更多

print(f"\n开始训练 ({NUM_EPOCHS} epochs)...")

for epoch in range(1, NUM_EPOCHS+1):
    loss = train_epoch(transformer, optimizer)
    print(f"Epoch: {epoch}, Loss: {loss:.4f}")

print("\n训练完成！")

# =============================================================================
# 6. 简单推理测试
# =============================================================================
def translate(model, src_sentence):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(1, -1).to(DEVICE)
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(1, 1, 1, num_tokens) == 0).to(DEVICE) # 全1 mask
    
    memory = model.encode(src, src_mask)
    
    # 开始解码，初始输入只有 <bos>
    ys = torch.ones(1, 1).fill_(BOS_IDX).type(torch.long).to(DEVICE)
    
    for i in range(20): # 最大长度20
        tgt_mask = (subsequent_mask(ys.size(1)).type(torch.bool)).to(DEVICE)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        
        # Generator 输出概率
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS_IDX:
            break
            
    # 将 token ID 转回单词
    tgt_tokens = []
    for tok in ys.flatten():
        tgt_tokens.append(vocab_transform[TGT_LANGUAGE].lookup_token(tok))
        
    return " ".join(tgt_tokens)

print("\n【推理测试】")
sentence = "Eine Gruppe von Menschen steht vor einem Iglu ."
print(f"德语: {sentence}")
print(f"翻译: {translate(transformer, sentence)}")