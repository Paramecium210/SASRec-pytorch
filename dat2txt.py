# 转换脚本示例
import os
import pandas as pd

# 读取原始数据，保留全部4列：UserID, MovieID, Rating, Timestamp
df = pd.read_csv('./data/ml-1m/ratings.dat', sep='::', header=None,
                 names=['user', 'item', 'rating', 'timestamp'], engine='python')

# 按 UserID 升序、Timestamp 升序排序
# utils.py 的 data_partition 依赖此顺序做 train/valid/test 划分
# 不排序会导致信息泄露和序列上下文错误
df = df.sort_values(['user', 'timestamp'], ascending=[True, True])

# 对 user_id 和 item_id 重新连续编号（从1开始）
# ML-1M 的 item_id 范围是 1~3952 但实际只有 3706 个物品，中间有 246 个空缺
# utils.py 用 max(item_id) 作为 itemnum，不重编号会导致：
#   - Embedding 有 246 个永远不会被训练的"幽灵槽位"
#   - 负采样 random_neq(1, itemnum+1) 会采到幽灵 item，污染训练信号
user_map = {u: i + 1 for i, u in enumerate(sorted(df['user'].unique()))}
item_map = {it: i + 1 for i, it in enumerate(sorted(df['item'].unique()))}
df['user'] = df['user'].map(user_map)
df['item'] = df['item'].map(item_map)

# 只保留 user_id 和 item_id 两列输出
os.makedirs('data', exist_ok=True)
df[['user', 'item']].to_csv('data/ml-1m.txt', sep=' ', header=False, index=False)

print("数据已保存至 data/ml-1m.txt")
print(f"总行数: {len(df)}")
print(f"用户数: {df['user'].nunique()}，user_id 范围: 1 ~ {df['user'].max()}")
print(f"物品数: {df['item'].nunique()}，item_id 范围: 1 ~ {df['item'].max()}（连续，无空缺）")