import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# ========== 参数 ==========
seq_path = 'datasets/MOT16/train/MOT16-04'
img_dir = os.path.join(seq_path, 'img1')
gt_file = os.path.join(seq_path, 'gt/gt.txt')
crop_dir = 'datasets/MOT16-04-ReID'
feature_dir = 'datasets/test_images'
topk = 10

# ========== 加载 GT ==========
gt_df = pd.read_csv(gt_file, header=None)
gt_df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis']
gt_df = gt_df[(gt_df['class'] == 1) & (gt_df['id'] > 0)]

# ========== 加载特征库 ==========
print("🔄 正在加载特征库...")
feature_db = {}
for name in os.listdir(feature_dir):
    if not name.endswith('.npy'):
        continue
    feature = np.load(os.path.join(feature_dir, name))
    feature_db[name] = feature
print(f"✅ 加载特征 {len(feature_db)} 条")

# ========== 视频帧映射 ==========
frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
frame_index_map = {i + 1: f for i, f in enumerate(frame_files)}
current_frame_idx = 1

# ========== 鼠标回调 ==========
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

                # 计算相似度
                all_names, all_feats = zip(*feature_db.items())
                all_feats = np.vstack(all_feats)
                sims = cosine_similarity(query_feat, all_feats)[0]
                top_indices = sims.argsort()[-topk:][::-1]

                # 显示匹配结果
                result_imgs = []
                for idx in top_indices:
                    name = all_names[idx]
                    sim = sims[idx]
                    show_id = name.split('_')[0]
                    frame_id = name.split('_')[2]

                    img_path = os.path.join(crop_dir, name.replace('.npy', '.jpg'))
                    if os.path.exists(img_path):
                        img = cv2.imread(img_path)
                        img = cv2.resize(img, (128, 256))

                        # ↑ 添加顶部空白用于显示文字
                        img = cv2.copyMakeBorder(img, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

                        # 写文字：ID + Frame + 相似度
                        text1 = f"ID:{int(show_id)} F:{int(frame_id)}"
                        text2 = f"S:{sim:.2f}"
                        cv2.putText(img, text1, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        cv2.putText(img, text2, (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                        result_imgs.append(img)

                if result_imgs:
                    gallery = cv2.hconcat(result_imgs)
                    cv2.imshow("Top-K Matches", gallery)
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

    if key == 27:  # ESC退出
        break
    elif key == ord('a'):
        current_frame_idx = max(1, current_frame_idx - 1)
    elif key == ord('d'):
        current_frame_idx = min(len(frame_files), current_frame_idx + 1)

cv2.destroyAllWindows()
