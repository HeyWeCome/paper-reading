import argparse
import math
import os
import shutil
import sys
from collections import defaultdict

import numpy as np
import torch
import faiss
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')  # 训练集还是测试集
parser.add_argument('--dataset', type=str, default='book', help='book | taobao')  # 数据集
parser.add_argument('--random_seed', type=int, default=19)  # 随机种子，用来保持随机数保持一致
parser.add_argument('--embedding_dim', type=int, default=64)  # embedding的维度
parser.add_argument('--hidden_size', type=int, default=64)  # 隐藏层的维度
parser.add_argument('--num_interest', type=int, default=4)  # 兴趣点的个数：论文中说可以是4或者2
parser.add_argument('--model_type', type=str, default='none', help='DNN | GRU4REC | ..')  # 模型的类型，是DNN还是GRU4REC等等
parser.add_argument('--learning_rate', type=float, default=0.001, help='')  # 学习率是多少
parser.add_argument('--max_iter', type=int, default=1000, help='(k)')  # 最大迭代次数
parser.add_argument('--patience', type=int,
                    default=50)  # patience是等待训练停止的“无改善时期”的数量。使用时patience = 50，可以忍受在50个epoch内没有改进
parser.add_argument('--coef', default=None)  #
parser.add_argument('--topN', type=int, default=50)

best_metric = 0


def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask


# 加载商品的种类信息
def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate


# 计算多样性
def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n - 1) * n / 2)
    return diversity


# 计算各种指标
def evaluate_full(sess, test_data, model, model_path, batch_size, item_cate_map, save=True, coef=None):
    topN = args.topN
    item_embs = model.output_item(sess)
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, args.embedding_dim, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0
    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)

        user_embs = model.output_user(sess, [hist_item, hist_mask])

        if len(user_embs.shape) == 2:
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                true_item_set = set(iid_list)
                for no, iid in enumerate(I[i]):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(I[i], item_cate_map)
        else:
            ni = user_embs.shape[1]
            user_embs = np.reshape(user_embs, [-1, user_embs.shape[-1]])
            D, I = gpu_index.search(user_embs, topN)
            for i, iid_list in enumerate(item_id):
                recall = 0
                dcg = 0.0
                item_list_set = set()
                item_cor_list = []
                if coef is None:
                    item_list = list(
                        zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    item_list.sort(key=lambda x: x[1], reverse=True)
                    for j in range(len(item_list)):
                        if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                            item_list_set.add(item_list[j][0])
                            item_cor_list.append(item_list[j][0])
                            if len(item_list_set) >= topN:
                                break
                else:
                    origin_item_list = list(
                        zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                    origin_item_list.sort(key=lambda x: x[1], reverse=True)
                    item_list = []
                    tmp_item_set = set()
                    for (x, y) in origin_item_list:
                        if x not in tmp_item_set and x in item_cate_map:
                            item_list.append((x, y, item_cate_map[x]))
                            tmp_item_set.add(x)
                    cate_dict = defaultdict(int)
                    for j in range(topN):
                        max_index = 0
                        max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                        for k in range(1, len(item_list)):
                            if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                                max_index = k
                                max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                            elif item_list[k][1] < max_score:
                                break
                        item_list_set.add(item_list[max_index][0])
                        item_cor_list.append(item_list[max_index][0])
                        cate_dict[item_list[max_index][2]] += 1
                        item_list.pop(max_index)

                true_item_set = set(iid_list)
                for no, iid in enumerate(item_cor_list):
                    if iid in true_item_set:
                        recall += 1
                        dcg += 1.0 / math.log(no + 2, 2)
                idcg = 0.0
                for no in range(recall):
                    idcg += 1.0 / math.log(no + 2, 2)
                total_recall += recall * 1.0 / len(iid_list)
                if recall > 0:
                    total_ndcg += dcg / idcg
                    total_hitrate += 1
                if not save:
                    total_diversity += compute_diversity(list(item_list_set), item_cate_map)

        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}


# def get_model(dataset, model_type, item_count, batch_size, maxlen):
# if model_type == 'DNN':
#     model = Model_DNN(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
# elif model_type == 'GRU4REC':
#     model = Model_GRU4REC(item_count, args.embedding_dim, args.hidden_size, batch_size, maxlen)
# elif model_type == 'MIND':
#     relu_layer = True if dataset == 'book' else False
#     model = Model_MIND(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen, relu_layer=relu_layer)
# elif model_type == 'ComiRec-DR':
#     model = Model_ComiRec_DR(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
# elif model_type == 'ComiRec-SA':
#     model = Model_ComiRec_SA(item_count, args.embedding_dim, args.hidden_size, batch_size, args.num_interest, maxlen)
# else:
#     print ("Invalid model_type : %s", model_type)
#     return
# return model


def get_exp_name(dataset, model_type, batch_size, lr, maxlen, save=True):
    extr_name = input('Please input the experiment name: ')
    para_name = '_'.join([dataset, model_type, 'b' + str(batch_size), 'lr' + str(lr), 'd' + str(args.embedding_dim),
                          'len' + str(maxlen)])
    exp_name = para_name + '_' + extr_name

    while os.path.exists('runs/' + exp_name) and save:
        flag = input('The exp name already exists. Do you want to cover? (y/n)')
        if flag == 'y' or flag == 'Y':
            shutil.rmtree('runs/' + exp_name)
            break
        else:
            extr_name = input('Please input the experiment name: ')
            exp_name = para_name + '_' + extr_name

    return exp_name


# 定义训练
def train(
        train_file,
        valid_file,
        test_file,
        cate_file,
        item_count,
        dataset="book",
        batch_size=128,
        maxlen=100,
        test_iter=50,
        model_type='DNN',
        lr=0.001,
        max_iter=100,
        patience=20
):
    exp_name = get_exp_name(dataset, model_type, batch_size, lr, maxlen)

    best_model_path = "best_model/" + exp_name + '/'
    gpu_options = tf.GPUOptions(allow_growth=True)

    writer = SummaryWriter('runs/' + exp_name)

    item_cate_map = load_item_cate(cate_file)


if __name__ == '__main__':
    print(sys.argv)
    args = parser.parse_args()
    SEED = args.random_seed
