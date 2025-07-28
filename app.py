from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
from glob import glob
import re

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 基础数据集路径
BASE_DATASET_DIR = "datasets/yisuo"

def load_npy_feature(file_path):
    """加载npy格式的特征文件"""
    try:
        return np.load(file_path)
    except Exception as e:
        app.logger.error(f"加载特征文件失败 {file_path}: {str(e)}")
        return None

def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    # 确保为一维向量
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    if vec1 is None or vec2 is None:
        return -1.0
    if vec1.ndim != 1 or vec2.ndim != 1:
        return -1.0
    if len(vec1) != len(vec2):
        return -1.0
        
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

def get_feature_path_by_id(detection_id, source_camera_dir, frame_id=1):
    """根据检测ID、源摄像头目录和帧ID生成特征文件路径"""
    frame_name = f"{frame_id:06d}"
    # 构造源特征文件路径
    pattern = f"{detection_id:06d}_c1s1_{frame_name}_*.npy"
    source_dir = os.path.join(BASE_DATASET_DIR, source_camera_dir, "reid_features")
    
    # 查找匹配的文件
    matches = glob(os.path.join(source_dir, pattern))
    if matches:
        return matches[0]  # 返回第一个匹配的文件
    return None

def get_all_camera_dirs():
    """获取所有摄像头目录"""
    camera_dirs = []
    if os.path.exists(BASE_DATASET_DIR):
        # 匹配所有"人脸追踪xx"格式的目录
        for item in os.listdir(BASE_DATASET_DIR):
            item_path = os.path.join(BASE_DATASET_DIR, item)
            if os.path.isdir(item_path) and re.match(r'^人脸追踪\d+$', item):
                camera_dirs.append(item)
    return sorted(camera_dirs)

@app.route('/api/find_all_similar', methods=['POST'])
def find_all_similar():
    """接收选中的特征ID和源摄像头，返回所有摄像头的相似特征结果"""
    data = request.json
    if not data or 'detection_id' not in data or 'source_camera_dir' not in data:
        return jsonify({"error": "缺少detection_id或source_camera_dir参数"}), 400
    
    detection_id = data['detection_id']
    source_camera_dir = data['source_camera_dir']
    
    try:
        detection_id = int(detection_id)
    except ValueError:
        return jsonify({"error": "detection_id必须是整数"}), 400
    
    # 1. 获取选中特征的文件路径并加载
    source_feature_path = get_feature_path_by_id(detection_id, source_camera_dir)
    if not source_feature_path or not os.path.exists(source_feature_path):
        return jsonify({"error": f"未找到ID为{detection_id}的特征文件"}), 404
    
    source_feature = load_npy_feature(source_feature_path)
    if source_feature is None:
        return jsonify({"error": "无法解析源特征文件"}), 500
    
    app.logger.info(f"已成功加载源特征：{source_feature_path}")
    
    # 2. 获取所有摄像头目录
    all_camera_dirs = get_all_camera_dirs()
    if not all_camera_dirs:
        return jsonify({"error": "未找到任何摄像头目录"}), 404
    
    similar_persons = []
    
    # 3. 遍历每个摄像头目录查找相似特征
    for camera_dir in all_camera_dirs:
        # 跳过源摄像头本身
        if camera_dir == source_camera_dir:
            continue

        print("正在从相机{}中查找相似行人".format(camera_dir))
            
        # 特征文件目录
        feature_dir = os.path.join(BASE_DATASET_DIR, camera_dir, "reid_features")
        if not os.path.exists(feature_dir):
            continue
            
        # 获取该摄像头下的所有特征文件
        target_files = glob(os.path.join(feature_dir, "*.npy"))
        if not target_files:
            continue
        
        # 查找该摄像头下最相似的特征
        max_similarity = -1.0
        most_similar_file = None
        
        for file_path in target_files:
            target_feature = load_npy_feature(file_path)
            if target_feature is None:
                continue
            
            # 计算相似度
            similarity = cosine_similarity(source_feature, target_feature)
            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_file = file_path
        
        # 如果找到有效相似特征
        if most_similar_file and max_similarity > 0.5:  # 过滤低相似度结果
            # 解析文件名信息
            file_name = os.path.basename(most_similar_file)
            print("在该相机中找到最相似的行人：{}".format(file_name))
            name_match = re.match(r'^(\d{6})_c1s1_(\d{6})_[\d.]+.npy$', file_name)
            if name_match:
                target_detection_id = name_match.group(1)
                frame_id = name_match.group(2)
                
                # 构建裁剪图URL（假设裁剪图存储在对应摄像头的crop_images目录）
                crop_image_name = file_name[:-4] + '.jpg'
                crop_image_url = f"datasets/yisuo/{camera_dir}/person_crops/{crop_image_name}"
                
                # 提取摄像头ID（从目录名"人脸追踪02"中提取"02"）
                camera_id_match = re.match(r'^人脸追踪(\d+)$', camera_dir)
                camera_id = camera_id_match.group(1) if camera_id_match else camera_dir

                similar_person_dict = {
                    "camera_id": camera_id,
                    "camera_dir": camera_dir,
                    "detection_id": target_detection_id,
                    "frame_id": frame_id,
                    "crop_image_url": crop_image_url,
                    "similarity": round(max_similarity, 6),
                    "similarity_percent": f"{max_similarity * 100:.2f}%"
                }
                print(similar_person_dict)
                
                similar_persons.append(similar_person_dict)
    
    # 按相似度排序
    similar_persons.sort(key=lambda x: x["similarity"], reverse=True)
    
    return jsonify({
        "source_id": detection_id,
        "source_camera": source_camera_dir,
        "similar_persons": similar_persons
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)