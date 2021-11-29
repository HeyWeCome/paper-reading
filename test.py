from collections import defaultdict

import torch
from torch import nn

users = defaultdict(list)
users[1].append([1, 2, 3, 4, 5])
users[2].append([6, 7, 8])
max_length = 0  # 序列的最长维度
max_num = 0  # 序列中最大的数字

for index in users.keys():
    user_sequence = users[index][0]  # 序列真正的长度
    length = len(user_sequence)
    if max_length < length:
        max_length = length

print("最长为", max_length)

train_data = defaultdict(list)  # 训练的数据

embedding = torch.nn.Embedding(num_embeddings=9999, embedding_dim=64)
for index in users.keys():
    user_sequence = users[index][0]  # 序列真正的长度
    for item in user_sequence:
        trans_item = embedding(torch.tensor(item))
        train_data[index] = trans_item
    index = index+1

print(train_data[1].shape)

# transformer_model = nn.TransformerDecoderLayer(d_model=64, nhead=16, num_encoder_layers=12)
# # print(transformer_model)
# # src = torch.rand((10, 32, 64))
# # print(src)
# # tgt = torch.rand((10, 32, 64))
# # print(tgt)
# out = transformer_model(train_data[1], train_data[1])
# print("结果", out.shape)

decoder_layer = nn.TransformerDecoderLayer(d_model=64, nhead=8)
memory = torch.rand(1, 10, 64)
tgt = torch.rand(2, 10, 64)
out = decoder_layer(tgt, memory)
print(memory.shape)
print(tgt.shape)
print(out.shape)

