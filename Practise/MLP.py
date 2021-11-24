import torch
from torch import nn
from collections import OrderedDict

class MLP(nn.Module):
    # 声明带有模型参数的层，这里声明两个全连接层
    def __init__(self, **kwargs):
        # 调用MLP父类Module的构造函数来进行必要的初始化，这样可以在构造实例的时候可以指定其他函数
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)  # 隐藏层
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 18)

    # 定义模型的前向计算，即如何根据输入x返回所需要的模型输出
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)


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


# 以上的MLP类中无须定义反向传播函数。系统将通过自动求梯度而自动生成反向传播所需的backward函数。
X = torch.rand(2, 784)

# 简洁实现MLP
# net = MLP()
# print(net)
# print(net(X))

# 简洁实现Sequential
net = MySequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
print(X)
print(net)
print(net(X))
