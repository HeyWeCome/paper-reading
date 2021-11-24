import math
from collections import defaultdict


class MostPopular:
    def __init__(self, source):
        # 相当于this.train_graph = XXX, 就是一个类的属性
        self.train_graph = self.read(source + '_train.txt')
        self.test_graph = self.read(source + '_test.txt')

    #  相当于形成一个对象数组，用户所有交互过的数据都在这个列表里面
    def read(self, source):
        graph = {}
        with open(source, 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                user_id = int(conts[0])
                item_id = int(conts[1])
                if user_id not in graph:
                    graph[user_id] = []
                graph[user_id].append(item_id)
        return graph

    # 接下来跑评估函数
    def evaluate(self, N=50):
        item_count = defaultdict(int)  # 创建一个dict字典,用来记录每个物品被交互过的次数
        for user in self.train_graph.keys():
            for item in self.train_graph[user]:
                item_count[item] += 1
        item_list = list(item_count.items())
        item_list.sort(key=lambda x: x[1], reverse=True)  # 根据物品被交互过的次数排序
        item_pop = set()  # set() 函数创建一个无序不重复元素集,删除重复数据,还可以计算交集、差集、并集等。
        for i in range(N):
            item_pop.add(item_list[i][0])  # 将交互数量最多的Top50商品放到列表中

        # 定义三个指标
        total_recall = 0.0  # 召回率，正确预测出正样本占实际正样本的概率
        total_ndcg = 0.0  # 归一化折损累计增益
        total_hitrate = 0  # 命中率，举个简单的例子，三个用户在测试集中的商品个数分别是10，12，8，模型得到的top-10推荐列表中，分别有6个，5个，4个在测试集中，那么此时HR的值是(6+5+4)/(10+12+8) = 0.5。
        for user in self.test_graph.keys():
            recall = 0
            dcg = 0.0
            item_list = self.test_graph[user]
            item_list = item_list[int(len(item_list) * 0.8):]
            for no, item_id in enumerate(item_list):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
                if item_id in item_pop:
                    recall += 1
                    dcg += 1.0 / math.log(no+2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
            total_recall += recall * 1.0 / len(item_list)
            if recall > 0:
                total_ndcg += dcg / idcg
                total_hitrate += 1
        total = len(self.test_graph)
        recall = total_recall / total
        ndcg = total_ndcg / total
        hitrate = total_hitrate * 1.0 / total
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}


if __name__ == "__main__":
    data = MostPopular('../Data/ComiRec/Preprocess/data/taobao_data/taobao')
    print(data.evaluate(20))

#  在淘宝数据集上，跑出来的结果为：
#  Metrics@20：{'recall': 0.003953526598784269, 'ndcg': 0.020654048922330904, 'hitrate': 0.05423943979196953}
#  Metrics@50：{'recall': 0.00734890055426121, 'ndcg': 0.036028407655913954, 'hitrate': 0.09309158664182313}
