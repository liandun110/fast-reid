import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def load_features_and_images(folder_path):
    """
    从指定文件夹加载图片和对应的特征
    :param folder_path: 文件夹路径
    :return: 图片路径列表，特征数组
    """
    image_paths = sorted(glob.glob(os.path.join(folder_path, '*.png')) + glob.glob(os.path.join(folder_path, '*.jpg')))
    print("图像数：{}".format(len(image_paths)))
    feature_paths = sorted(glob.glob(os.path.join(folder_path, '*.npy')))
    features = []
    for feat_path in feature_paths:
        feat = np.load(feat_path)
        features.append(feat)
    features = np.array(features).squeeze()
    return image_paths, features

def find_top_n_similar(features, image_paths, top_n):
    """
    为每张图片找到最相似的top_n张图片
    :param features: 特征数组
    :param image_paths: 图片路径列表
    :param top_n: 要查找的相似图片数量
    :return: 包含每张图片对应的top_n张相似图片信息的列表
    """
    top_n_results = []
    num_images = len(image_paths)
    for i in range(num_images):
        similarity_scores = cosine_similarity([features[i]], features)[0]
        top_n_indices = np.argsort(similarity_scores)[-(top_n + 1):][::-1][1:]  # 排除自身
        top_n_scores = similarity_scores[top_n_indices]
        top_n_images = [image_paths[idx] for idx in top_n_indices]
        top_n_results.append((image_paths[i], top_n_images, top_n_scores))
    return top_n_results

def extract_id_from_filename(filename):
    """
    从文件名中提取ID信息
    例如: 1501_c6s4_001877_00.jpg -> 1501
    """
    basename = os.path.basename(filename)
    try:
        # 查找第一个下划线的位置
        first_underscore = basename.find('_')
        if first_underscore != -1:
            # 提取下划线前的部分作为ID
            pid = basename[:first_underscore]
            return pid
    except:
        pass
    
    # 如果无法提取，返回默认值
    return "Unkwonw"


def add_green_border(image, border_width=4):
    """
    为图像添加绿色边框
    :param image: 输入图像
    :param border_width: 边框宽度
    :return: 添加边框后的图像
    """
    # 创建一个绿色边框
    border_color = [0, 255, 0]  # BGR格式的绿色
    bordered_image = cv2.copyMakeBorder(
        image,
        top=border_width,
        bottom=border_width,
        left=border_width,
        right=border_width,
        borderType=cv2.BORDER_CONSTANT,
        value=border_color
    )
    return bordered_image


def visualize_results(top_n_results, top_n, save_dir='visualization_results'):
    """
    可视化结果，显示每张图片和其最相似的top_n张图片，并显示相似度得分和ID。
    相同ID的图像添加绿色边框，并保存结果到指定目录。
    :param top_n_results: 每张图片对应的top_n张相似图片信息
    :param top_n: 相似图像数量
    :param save_dir: 结果保存路径
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i, (query_image, top_n_images, top_n_scores) in enumerate(top_n_results):
        # 减小图像尺寸和间距
        fig, axes = plt.subplots(1, top_n + 1, figsize=(2 * (top_n + 1), 2))
        fig.subplots_adjust(wspace=0.1, hspace=0)  # 减小子图间水平间距

        # 查询图像
        query_img = cv2.cvtColor(cv2.imread(query_image), cv2.COLOR_BGR2RGB)
        query_id = extract_id_from_filename(query_image)
        axes[0].imshow(query_img)
        axes[0].set_title(f'Query\nID: {query_id}', fontsize=8)
        axes[0].axis('off')

        for j, (similar_image, score) in enumerate(zip(top_n_images, top_n_scores)):
            sim_img = cv2.imread(similar_image)
            similar_id = extract_id_from_filename(similar_image)

            if similar_id == query_id:
                sim_img = add_green_border(sim_img)

            sim_img_rgb = cv2.cvtColor(sim_img, cv2.COLOR_BGR2RGB)
            axes[j + 1].imshow(sim_img_rgb)
            axes[j + 1].set_title(f'ID: {similar_id}\nScore: {score:.2f}', fontsize=8)
            axes[j + 1].axis('off')

        # 保存结果
        query_basename = os.path.splitext(os.path.basename(query_image))[0]
        save_path = os.path.join(save_dir, f'{query_basename}_similarity_results.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"结果已保存至: {save_path}")
        plt.close()


if __name__ == '__main__':
    folder_path = 'datasets/test_images'  # 请替换为实际的文件夹路径
    top_n = 10  # 可以修改这个值来指定要显示的相似图片数量
    save_dir = 'visualization_results'  # 保存结果的目录
    
    image_paths, features = load_features_and_images(folder_path)
    top_n_results = find_top_n_similar(features, image_paths, top_n)
    visualize_results(top_n_results, top_n, save_dir)
