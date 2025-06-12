import os
import numpy as np
import torch

# ========== 参数 ==========
feature_dir = 'datasets/test_images'  # 存放 .npy 特征的目录
threshold = 0.98

# ========== 加载特征 ==========
print("🔄 正在加载特征...")
features = []
ids = []
names = []

for name in os.listdir(feature_dir):
    if not name.endswith('.npy') or not name[0] == '0':
        continue
    path = os.path.join(feature_dir, name)
    feat = np.load(path)
    features.append(feat)
    names.append(name)
    ids.append(name.split('_')[0])  # 默认ID为文件名的前缀，如 0001_xxx.npy

features = np.vstack(features).astype(np.float32)
ids = np.array(ids)

print(f"✅ 共加载 {len(features)} 条特征")

# ========== GPU 加速计算余弦相似度 ==========
print("⚡ 使用 GPU 加速计算相似度矩阵...")
with torch.no_grad():
    features_tensor = torch.tensor(features, dtype=torch.float32).cuda()
    features_tensor = torch.nn.functional.normalize(features_tensor, dim=1)
    similarity_matrix = features_tensor @ features_tensor.T
    similarity_matrix = similarity_matrix.cpu().numpy()

print("✅ 相似度矩阵计算完成")

# ========== 构建同一身份矩阵 ==========
print("🔍 构建身份匹配矩阵...")
id_matrix = ids[:, None] == ids[None, :]  # [m, m] 的布尔矩阵

# ========== 分析 ==========
same_id = id_matrix
diff_id = ~id_matrix
high_sim = similarity_matrix > threshold
low_sim = ~high_sim

# ========== 去除对角线 ==========
np.fill_diagonal(same_id, False)
np.fill_diagonal(diff_id, False)
np.fill_diagonal(high_sim, False)
np.fill_diagonal(low_sim, False)

# ========== 统计 ==========
A = np.sum(high_sim & same_id)
B = np.sum(high_sim & diff_id)
C = np.sum(low_sim & same_id)
D = np.sum(low_sim & diff_id)

print("\n📊 统计结果 (阈值 > {:.2f})：".format(threshold))
print(f"1. 相似度高 且 同一身份   ：{A}")
print(f"2. 相似度高 且 不同身份 ：{B}")
print(f"3. 相似度低 且 同一身份 ：{C}")
print(f"4. 相似度低 且 不同身份 ：{D}")
