import numpy as np
import torch
from collections import defaultdict
from multiprocessing import Process, Queue


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}

    print(f"Loading data from {fname}...")
    with open('data/%s.txt' % fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])

    return [user_train, user_valid, user_test, usernum, itemnum]


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():
        user = np.random.randint(1, usernum + 1)
        while len(user_train.get(user, [])) <= 1:
            user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0:
                neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function,
                        args=(User, usernum, itemnum, batch_size, maxlen, self.result_queue, np.random.randint(2e9)))
            )
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


# 评估函数
def evaluate(model, dataset, args, is_test=False):
    [train, valid, test, usernum, itemnum] = dataset

    NDCG_5 = 0.0
    HT_5 = 0.0
    NDCG_10 = 0.0
    HT_10 = 0.0
    valid_user = 0.0

    if usernum > 10000:
        users = np.random.choice(range(1, usernum + 1), 10000, replace=False)
    else:
        users = range(1, usernum + 1)

    for u in users:
        # 如果训练集或测试集为空，则跳过
        if len(train.get(u, [])) < 1 or len(test.get(u, [])) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1

        # 如果是最终测试，历史序列需要包含 valid 集中的那一个 item
        if is_test:
            seq[idx] = valid[u][0]
            idx -= 1

        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)

        # 确定目标 item 和需要避开的已交互 item
        if is_test:
            item_idx = [test[u][0]]
            rated.add(valid[u][0])
        else:
            item_idx = [valid[u][0]]

        # 采样 100 个负样本
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        # 模型预测
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        # 获取正样本（即 item_idx[0]）的排名
        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1
        # 计算 @5 指标
        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        # 计算 @10 指标
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1

    return {
        'HR@5': HT_5 / valid_user, 'NDCG@5': NDCG_5 / valid_user,
        'HR@10': HT_10 / valid_user, 'NDCG@10': NDCG_10 / valid_user
    }