import os
import cv2
import pandas as pd
import random
from tqdm import tqdm

def visualize_mot16_gt(
    seq_path='datasets/MOT16/train/MOT16-04',
    output_name='mot16_gt_visualization',
    draw_gt=True,
    fps=30
):
    # ========== 路径配置 ==========
    img_dir = os.path.join(seq_path, 'img1')
    gt_file = os.path.join(seq_path, 'gt/gt.txt')
    suffix = 'with_gt' if draw_gt else 'no_gt'
    output_video = f'{output_name}_{suffix}.mp4'

    # ========== 读取 Ground Truth ==========
    gt_df = pd.read_csv(gt_file, header=None)
    gt_df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'visibility']
    gt_df = gt_df[(gt_df['class'] == 1) & (gt_df['id'] > 0)]  # 只保留有效的行人目标

    # ========== 视频参数 ==========
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    sample_img = cv2.imread(os.path.join(img_dir, frame_files[0]))
    height, width = sample_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    # ========== ID颜色映射 ==========
    color_map = {}
    def get_color_for_id(track_id):
        if track_id not in color_map:
            random.seed(track_id)
            color_map[track_id] = (
                random.randint(30, 255),
                random.randint(30, 255),
                random.randint(30, 255)
            )
        return color_map[track_id]

    # ========== 视频生成 ==========
    for frame_idx, frame_name in tqdm(enumerate(frame_files, 1), total=len(frame_files)):
        frame_path = os.path.join(img_dir, frame_name)
        img = cv2.imread(frame_path)

        # 始终绘制帧信息在右上角
        label_text = f'FRAME: {frame_name.split(".")[0]}'
        text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = width - text_size[0] - 10  # 右上角对齐
        text_y = 30
        cv2.putText(img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if draw_gt:
            frame_gt = gt_df[gt_df['frame'] == frame_idx]
            for _, row in frame_gt.iterrows():
                x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
                pid = int(row['id'])
                color = get_color_for_id(pid)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                id_label = f'ID {pid:03d}'
                cv2.putText(img, id_label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        video_writer.write(img)

    video_writer.release()
    print(f'✅ 视频已保存：{output_video}')

# 示例调用
visualize_mot16_gt(draw_gt=True)
visualize_mot16_gt(draw_gt=False)
