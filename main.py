# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from collections import defaultdict

import sklearn.cluster
import torch
from torch import nn as nn
import numpy as np
from sklearn import preprocessing

users = defaultdict(list)  # list对应[]
item_count = defaultdict(int)  # int对应0

def read_from_taobao(source):
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')  # ['1', '2268318', '2520377', 'pv', '1511544070']
            uid = int(conts[0])
            iid = int(conts[1])
            # if conts[3] != 'pv':
            #     continue
            item_count[iid] += 1  # 字典中物品ID对应的数值加1
            users[uid].append(iid)  # 用户的字典中添加交互的物品和时间戳


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # sklearn.cluster.KMeans(n_clusters=8,  # 簇的个数，即你想聚成几类
    #                        init='k-means++',  # 初始簇中心的获取方法
    #                        n_init=10,  # 获取初始簇中心的更迭次数，为了弥补初始质心的影响，算法默认会初始10次质心，实现算法，然后返回最好的结果
    #                        max_iter=300,  # 最大迭代次数（因为kmeans算法的实现需要迭代）
    #                        tol=0.0001,  # 容忍度，即kmeans运行准则收敛的条件
    #                        precompute_distances='auto',  # 是否需要提前计算距离，这个参数会在空间和时间之间做权衡, auto 会默认在数据样本大于featurs*samples 的数量大于12e6 的时候False,False 时核心实现的方法是利用Cpython 来实现的
    #                        verbose=0,  # 不修改
    #                        random_state=None,  # 随机生成簇中心的状态条件
    #                        copy_x=True,  # 对是否修改数据的一个标记，如果True，即复制了就不会修改数据
    #                        n_jobs=1,  # 并行设置
    #                        algorithm='auto'  # kmeans的实现算法，有：‘auto’, ‘full’, ‘elkan’, 其中 'full’表示用EM方式实现
    #                        )

    read_from_taobao('UserBehavior.csv')
    data = users[1]
    print(data)
    processed_data = preprocessing.scale(data)

    # num = torch.tensor([1, 2, 3, 4])
    #
    # embedding = nn.Embedding(num_embeddings=1, embedding_dim=64)
    # result = embedding(data)

    k = 3  # 假如我要聚类为3个clusters
    [centroid, label, inertia] = sklearn.cluster.k_means(processed_data.reshape(-1, 1), k)  # centroid: 中心点， inertial： 质心
    print("中心点", centroid)
    print("质点", inertia)
    # print([centroid, label, inertia])
