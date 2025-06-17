"""
MOT16提供的检测框准确率较低，因为是很多年前的了。本代码可生成一份新的det/det_yolov8x.txt
"""

import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

# 输入输出路径
seq_path = 'datasets/MOT16/train/MOT16-04'
img_dir = os.path.join(seq_path, 'img1')
output_det_file = os.path.join(seq_path, 'det', 'det_yolov8x.txt')
os.makedirs(os.path.dirname(output_det_file), exist_ok=True)

# 加载模型（假设你使用的是 yolov8x.pt，替换成你自己的模型）
model = YOLO('yolov8x.pt')  # 或 yolov11x.pt，如果你有

# 写入文件
with open(output_det_file, 'w') as f_out:
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    for frame_id, img_name in tqdm(enumerate(frame_files, 1), total=len(frame_files)):
        img_path = os.path.join(img_dir, img_name)
        results = model(img_path, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w = x2 - x1
            h = y2 - y1
            conf = float(box.conf)
            cls = int(box.cls)

            # 仅保留行人类别（YOLO中通常是 class==0，需根据模型训练时的类别顺序确认）
            if cls == 0:
                line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                f_out.write(line)

print(f"✅ 新的检测结果已保存至 {output_det_file}")
