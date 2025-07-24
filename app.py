from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入CORS
import os
import numpy as np
from glob import glob
import re

app = Flask(__name__)
CORS(app)  # 允许所有跨域请求（开发环境用，生产环境需限制来源）

# 配置特征文件目录
REID_FEATURES_DIR = "datasets/yisuo/人脸追踪02/reid_features"
# 确保目录存在
os.makedirs(REID_FEATURES_DIR, exist_ok=True)

def load_npy_feature(file_path):
    """加载npy格式的特征文件"""
    try:
        return np.load(file_path)
    except Exception as e:
        app.logger.error(f"加载特征文件失败 {file_path}: {str(e)}")
        return None

def cosine_similarity(vec1, vec2):
    # 确保为一维向量
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()

    """计算两个向量的余弦相似度"""
    if vec1 is None or vec2 is None:
        print("向量为空")
        return -1.0
    if vec1.ndim != 1 or vec2.ndim != 1:
        print("维度错误")
        return -1.0
    if len(vec1) != len(vec2):
        return -1.0
        
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        print("模长为0")
        return 0.0

    result = float(dot_product / (norm1 * norm2))
    return result

def get_feature_path_by_id(detection_id, frame_id=1):
    """根据检测ID和帧ID生成特征文件路径"""
    # 构造源特征文件路径（人脸追踪01的特征）
    frame_name = f"{frame_id:06d}"
    # 假设置信度格式匹配（实际应用中可能需要更精确的匹配）
    pattern = f"{detection_id:06d}_c1s1_{frame_name}_*.npy"
    source_dir = "datasets/yisuo/人脸追踪01/reid_features"
    
    # 查找匹配的文件
    matches = glob(os.path.join(source_dir, pattern))
    if matches:
        return matches[0]  # 返回第一个匹配的文件
    return None

@app.route('/api/find_similar', methods=['POST'])
def find_similar():
    print("函数调用")  # 现在会在POST请求时打印
    """接收选中的特征ID，返回最相似的特征结果"""
    data = request.json
    if not data or 'detection_id' not in data:
        return jsonify({"error": "缺少detection_id参数"}), 400
    else:
        print("接收到前端信息")
    
    detection_id = data['detection_id']
    try:
        detection_id = int(detection_id)
    except ValueError:
        return jsonify({"error": "detection_id必须是整数"}), 400
    
    # 1. 获取选中特征的文件路径并加载
    source_feature_path = get_feature_path_by_id(detection_id)
    if not source_feature_path or not os.path.exists(source_feature_path):
        return jsonify({"error": f"未找到ID为{detection_id}的特征文件"}), 404
    
    source_feature = load_npy_feature(source_feature_path)
    if source_feature is None:
        return jsonify({"error": "无法解析源特征文件"}), 500
    else:
        print("已成功加载所选行人的reid特征：{}".format(source_feature_path))
    
    # 2. 遍历目标目录中的所有特征文件
    target_files = glob(os.path.join(REID_FEATURES_DIR, "*.npy"))
    if not target_files:
        return jsonify({"error": "目标特征目录中没有文件"}), 404
    else:
        print("找到{}条候选特征".format(len(target_files)))
    
    max_similarity = -1.0
    most_similar_file = None
    similar_results = []
    
    for file_path in target_files:
        # 加载目标特征
        target_feature = load_npy_feature(file_path)
        if target_feature is None:
            continue
        
        # 计算相似度
        similarity = cosine_similarity(source_feature, target_feature)
        
        # 提取文件名中的ID（从文件名格式中解析）
        file_name = os.path.basename(file_path)
        id_match = re.match(r'^(\d{6})_', file_name)
        target_id = id_match.group(1) if id_match else "未知ID"
        
        # 保存结果
        similar_results.append({
            "file_name": file_name,
            "target_id": target_id,
            "similarity": round(similarity, 6)
        })
        
        # 更新最大相似度
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_file = file_name

    # 3. 解析最相似文件的元信息并在后端打印（核心新增逻辑）
    if most_similar_file:
        # 解析文件名：假设格式为 {detection_id:06d}_c1s1_{frame_id:06d}_{conf}.npy
        # 例如：000002_c1s1_000005_0.9765.npy
        file_name = most_similar_file
        name_match = re.match(r'^(\d{6})_c1s1_(\d{6})_[\d.]+.npy$', file_name)
        if name_match:
            similar_detection_id = name_match.group(1)  # 提取detectionID
            frame_id = name_match.group(2)              # 提取帧号
        else:
            similar_detection_id = "未知"
            frame_id = "未知"
        
        # 提取视频名：从特征文件路径中解析（假设路径格式为 .../视频名/reid_features/文件名）
        # 例如：REID_FEATURES_DIR = "datasets/yisuo/人脸追踪02/reid_features"
        # 则视频名为 "人脸追踪02"
        video_name = os.path.basename(os.path.dirname(REID_FEATURES_DIR))
        
        # 在后端打印信息
        print("\n===== 最相似特征信息 =====")
        print(f"最相似特征所在视频：{video_name}")
        print(f"所在帧号：{frame_id}")
        print(f"对应detectionID：{similar_detection_id}")
        print(f"相似度：{round(max_similarity, 6)} ({max_similarity * 100:.2f}%)")
        print(f"特征文件：{most_similar_file}")
        print("==========================\n")
    
    # 4. 返回结果（包含最相似和前N个结果）
    return jsonify({
        "source_id": detection_id,
        "most_similar": {
            "file_name": most_similar_file,
            "similarity": round(max_similarity, 6),
            "similarity_percent": f"{max_similarity * 100:.2f}%"
        },
        "top_results": sorted(similar_results, key=lambda x: x["similarity"], reverse=True)[:5]  # 返回前5个
    })

if __name__ == '__main__':
    # 开发环境使用，生产环境需配置WSGI服务器
    print("后端主函数")
    app.run(host='0.0.0.0', port=5000, debug=True)