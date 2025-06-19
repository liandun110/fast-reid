import os
import cv2
from ultralytics import YOLO
from tqdm import tqdm

def person_detect(seq_path, model_path='yolov8x.pt', output_dir=None):
    """
    使用YOLOv8模型对MOT16格式的序列进行行人检测
    
    参数:
        model_path (str): YOLO模型路径，默认为'yolov8x.pt'
        output_dir (str): 输出文件夹路径，默认为None（自动在seq_path下创建det目录）
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
    model = YOLO('datasets/yolov8x.pt')
    
    # 处理所有帧
    with open(output_det_file, 'w') as f_out:
        frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        total_frames = len(frame_files)
        
        print(f"开始处理序列: {seq_path}")
        print(f"总帧数: {total_frames}")
        print(f"模型: {model_path}")
        print(f"输出路径: {output_det_file}")
        
        for frame_id, img_name in tqdm(enumerate(frame_files, 1), total=total_frames):
            img_path = os.path.join(img_dir, img_name)
            
            # 模型推理
            results = model(img_path, verbose=False)[0]
            
            # 处理检测框
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                conf = float(box.conf)
                cls = int(box.cls)
                
                # 仅保留行人类别（YOLO中通常是class==0）
                if cls == 0:
                    line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                    f_out.write(line)
    
    print(f"✅ 新的检测结果已保存至 {output_det_file}")
    return output_det_file

# 示例调用
if __name__ == "__main__":
    # 请替换为实际路径
    seq_path = 'datasets/yisuo/人脸追踪02/'
    person_detect(seq_path)    