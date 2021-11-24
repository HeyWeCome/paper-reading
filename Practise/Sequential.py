# 当模型的前向计算为简单串联各个层的计算时，Sequential类可以通过更加简单的方式定义模型。
# 这正是Sequential类的目的：它可以接收一个子模块的有序字典（OrderedDict）或者一系列子模块作为参数来逐一添加Module的实例，
# 而模型的前向计算就是将这些实例按添加的顺序逐一计算。

import torch
from torch import nn
from collections import OrderedDict


class MySequential(nn.Module):
    def __init__(self, *args):
        super(MySequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):  # 如果传入的是一个OrderedDict
            for key, module in args[0].items():
                self.add_module(key, module)  # add_module 方法会将module添加进self._modules(一个OrderedDict)
        else:  # 传入的是一些模型
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, input):
        # self._modules返回的是一个 OrderedDict, 保证会按照成员添加时的顺序便利成员
        for module in self._modules.values():
            input = module(input)
        return input


net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.rand(2, 784)
print(X)
print(net)
print(net(X))
