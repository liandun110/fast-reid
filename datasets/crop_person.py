import os
import cv2
import argparse
import pandas as pd
from tqdm import tqdm

def save_cropped_pedestrians(args):
    # 读取检测文件
    det_path = os.path.join(args.seq_path, 'det/det_yolov8x.txt')
    if not os.path.exists(det_path):
        raise FileNotFoundError(f"检测文件不存在: {det_path}")
    
    # 读取检测结果，使用合适的列名
    det_df = pd.read_csv(det_path, header=None)
    det_df.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'conf', 'class', 'vis', 'direction']
    
    # 确定图像目录
    img_dir = os.path.join(args.seq_path, 'img1')  # MOT16标准目录结构为img1
    if not os.path.exists(img_dir):
        img_dir = os.path.join(args.seq_path, 'images')  # 兼容可能的其他目录名
    
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")
    
    # 获取所有帧文件
    frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    total_frames = len(frame_files)
    
    print(f"开始处理序列: {args.seq_path}")
    print(f"总帧数: {total_frames}")
    print(f"检测文件: {det_path}")
    
    # 创建输出目录
    output = os.path.join(args.seq_path, 'person_crops')
    os.makedirs(output, exist_ok=True)
    
    # 处理每一帧
    for frame_file in tqdm(frame_files):
        # 从文件名中提取帧号（例如：000001.jpg -> 1）
        frame_name = os.path.splitext(frame_file)[0]
        try:
            frame_id = int(frame_name)  # 直接转换为整数，假设文件名是纯数字
        except ValueError:
            # 如果文件名不是纯数字，尝试更复杂的提取逻辑
            frame_id = None
            for part in frame_name.split('_'):
                if part.isdigit():
                    frame_id = int(part)
                    break
            if frame_id is None:
                print(f"警告: 无法从文件名 {frame_file} 中提取帧号")
                continue
        
        frame_path = os.path.join(img_dir, frame_file)
        img = cv2.imread(frame_path)
        
        if img is None:
            print(f"警告: 无法读取图片 {frame_path}")
            continue
        
        # 获取当前帧的所有检测结果
        frame_dets = det_df[det_df['frame'] == frame_id]
        
        # 为每个检测结果生成唯一ID（使用帧号和检测索引组合）
        for det_idx, row in frame_dets.iterrows():
            # 使用帧号和检测索引生成唯一ID
            pid = int(f"{frame_id}{det_idx % 1000:03d}")
            x, y, w, h = map(int, [row['x'], row['y'], row['w'], row['h']])
            conf = float(row['conf'])
            
            # 裁剪行人区域，确保不越界
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            cropped = img[y1:y2, x1:x2]
            
            # 跳过空裁剪结果
            if cropped.size == 0:
                continue
            
            # 构建输出文件名
            person_id_str = f"{pid:06d}"  # 使用6位ID以避免冲突
            filename = f"{person_id_str}_c1s1_{frame_name}_{conf:.4f}.jpg"
            save_path = os.path.join(output, filename)
            
            # 保存裁剪结果
            cv2.imwrite(save_path, cropped)
    
    print(f"✅ 处理完成，共裁剪出 {len(os.listdir(output))} 个行人图像")
    print(f"保存路径: {output}")

def get_parser():
    parser = argparse.ArgumentParser(description="根据YOLOv8检测结果裁剪行人图像")
    parser.add_argument("--seq_path", default="datasets/yisuo/人脸追踪02", help="图像序列路径")
    parser.add_argument("--min_conf", type=float, default=0.5, help="最小置信度阈值")
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()
    save_cropped_pedestrians(args)