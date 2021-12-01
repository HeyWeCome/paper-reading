import torch

x = torch.rand(256, 20, 20)
y = torch.rand(256, 20, 64)
print(x.shape)
print(y.shape)
print((x+y).shape)