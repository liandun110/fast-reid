import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ========== 参数 ==========
seq_path = 'datasets/MOT16/train/MOT16-04'
img_dir = os.path.join(seq_path, 'img1')
gt_file = os.path.join(seq_path, 'gt/gt.txt')
feature_dir = 'datasets/test_images'  # 存放 .npy 特征的路径
topk = 10  # 当前未使用，但保留

# ========== 加载 GT ==========
gt_df = pd.read_csv(gt_file, header=None)
gt_df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
gt_df = gt_df[(gt_df['class'] == 1) & (gt_df['id'] > 0)]

# ========== 加载特征库 ==========
print("🔄 正在加载特征库...")
feature_db = {}
for name in os.listdir(feature_dir):
    if name.endswith('.npy'):
        feature = np.load(os.path.join(feature_dir, name))
        feature_db[name] = feature
print(f"✅ 加载特征 {len(feature_db)} 条")

# ========== 视频帧映射 ==========
frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
frame_index_map = {i + 1: f for i, f in enumerate(frame_files)}
current_frame_idx = 1

# ========== 相似度绘图函数 ==========
def plot_similarity_curve(query_id, query_feat):
    same_id_scores = []
    diff_id_top_scores = []
    frame_ids = sorted(gt_df['frame'].unique())

    print(f"\n📈 正在绘制 ReID 曲线（Query ID = {query_id}）:")
    for idx, frame_id in enumerate(frame_ids):
        print(f"  - 处理帧 {frame_id} ({idx + 1}/{len(frame_ids)})...")
        frame_gt = gt_df[gt_df['frame'] == frame_id]
        scores = []
        ids = []

        for _, row in frame_gt.iterrows():
            tid = int(row['id'])
            frame_str = frame_index_map[frame_id].split('.')[0]
            crop_name = f"{tid:04d}_c1s1_{frame_str}_00"
            npy_path = os.path.join(feature_dir, crop_name + '.npy')
            if not os.path.exists(npy_path):
                continue
            feat = np.load(npy_path).reshape(1, -1)
            score = cosine_similarity(query_feat, feat)[0][0]
            scores.append(score)
            ids.append(tid)

        same_score = 0
        diff_top_score = 0
        for tid, score in zip(ids, scores):
            if tid == query_id:
                same_score = max(same_score, score)
            else:
                diff_top_score = max(diff_top_score, score)

        same_id_scores.append(same_score)
        diff_id_top_scores.append(diff_top_score)

    print("✅ 相似度数据计算完毕，正在绘图...\n")

    # 绘图
    plt.figure(figsize=(10, 4))
    plt.plot(frame_ids, same_id_scores, label=f'Query ID {query_id}', color='red')
    plt.plot(frame_ids, diff_id_top_scores, label='Top Non-matching ID', color='blue')
    plt.xlabel('Frame ID')
    plt.ylabel('Cosine Similarity')
    plt.title('ReID Similarity Over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== 鼠标点击事件 ==========
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global current_frame_idx
        frame_gt = gt_df[gt_df['frame'] == current_frame_idx]
        for _, row in frame_gt.iterrows():
            x1, y1 = int(row['x']), int(row['y'])
            x2, y2 = x1 + int(row['w']), y1 + int(row['h'])
            if x1 <= x <= x2 and y1 <= y <= y2:
                track_id = int(row['id'])
                frame_str = frame_index_map[current_frame_idx].split('.')[0]
                crop_name = f"{track_id:04d}_c1s1_{frame_str}_00"
                npy_path = os.path.join(feature_dir, crop_name + '.npy')
                if not os.path.exists(npy_path):
                    print(f"❌ 特征文件缺失: {npy_path}")
                    return
                query_feat = np.load(npy_path).reshape(1, -1)

                # 👉 绘制相似度曲线
                plot_similarity_curve(track_id, query_feat)
                return

# ========== 主循环 ==========
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", on_mouse)

while True:
    frame_name = frame_index_map[current_frame_idx]
    frame_path = os.path.join(img_dir, frame_name)
    img = cv2.imread(frame_path)

    # 显示目标框
    frame_gt = gt_df[gt_df['frame'] == current_frame_idx]
    for _, row in frame_gt.iterrows():
        x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
        pid = int(row['id'])
        color = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f'ID {pid}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    cv2.putText(img, f'FRAME: {frame_name.split(".")[0]}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    cv2.imshow("Video", img)
    key = cv2.waitKey(30)

    if key == 27:  # ESC
        break
    elif key == ord('a'):
        current_frame_idx = max(1, current_frame_idx - 1)
    elif key == ord('d'):
        current_frame_idx = min(len(frame_files), current_frame_idx + 1)

cv2.destroyAllWindows()
