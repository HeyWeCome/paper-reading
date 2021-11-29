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

# print(train_data[1].unsqueeze(dim=0).unsqueeze(dim=0).shape)
temp = train_data[1].unsqueeze(dim=0).unsqueeze(dim=0)

encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
out = transformer_encoder(temp)
print(temp.reshape(-1))
print(out.reshape(-1))

