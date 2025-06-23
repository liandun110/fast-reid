import os
import cv2
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def person_detect(seq_path, model_path='yolov8x.pt', output_dir=None):
    """
    使用YOLOv8模型对MOT16格式的序列进行行人检测，输出YOLO格式的检测框
    
    参数:
        seq_path (str): 序列路径，包含img1子目录
        model_path (str): YOLO模型路径，默认为'yolov8x.pt'
        output_dir (str): 输出文件夹路径，默认为None（自动在seq_path下创建det目录）
    
    返回:
        str: 生成的检测文件路径
    """
    img_dir = os.path.join(seq_path, 'img1')
    
    # 输入输出路径处理
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"图像目录不存在: {img_dir}")

    # 确定输出文件路径
    if output_dir is None:
        output_dir = os.path.join(seq_path, 'det')
    os.makedirs(output_dir, exist_ok=True)
    output_det_file = os.path.join(output_dir, 'det_yolov8x.txt')
    
    # 加载模型
    model = YOLO(model_path)
    
    # 处理所有帧
    with open(output_det_file, 'w') as f_out:
        frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        total_frames = len(frame_files)
        
        print(f"开始处理序列: {seq_path}")
        print(f"总帧数: {total_frames}")
        print(f"模型: {model_path}")
        print(f"输出路径: {output_det_file}")
        
        # 用于生成唯一ID的计数器
        detection_id = 0
        
        for frame_id, img_name in tqdm(enumerate(frame_files, 1), total=total_frames):
            img_path = os.path.join(img_dir, img_name)
            
            # 读取图像获取宽高
            img = cv2.imread(img_path)
            img_h, img_w = img.shape[:2]
            
            # 模型推理
            results = model(img_path, verbose=False)[0]
            
            # 处理检测框
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf)
                cls = int(box.cls)
                
                # 转换为YOLO格式的相对坐标
                x_center = (x1 + w/2) / img_w
                y_center = (y1 + h/2) / img_h
                width = w / img_w
                height = h / img_h
                
                # 仅保留行人类别（YOLO中通常是class==0）
                if cls == 0:
                    line = f"{frame_id},-1,{x_center:.6f},{y_center:.6f},{width:.6f},{height:.6f},{conf:.6f},-1,-1,{detection_id}\n"
                    f_out.write(line)
                    detection_id += 1  # 递增ID
    
    print(f"✅ 新的检测结果已保存至 {output_det_file}")
    return output_det_file


if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用YOLOv8进行行人检测')
    parser.add_argument('--seq_path', type=str, required=True, help='序列路径，包含img1子目录')
    parser.add_argument('--model_path', type=str, default='datasets/yolov8x.pt', help='YOLO模型路径')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    
    args = parser.parse_args()
    
    # 调用检测函数
    person_detect(
        seq_path=args.seq_path,
        model_path=args.model_path,
        output_dir=args.output_dir
    )