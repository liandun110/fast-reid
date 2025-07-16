import numpy as np
import cv2

def transform_point(point, homography_matrix):
    """
    使用单应矩阵转换点坐标
    
    参数:
        point[tuple]: 原始点坐标，格式为(x, y)
        homography_matrix[ndarray]: 3x3的单应矩阵
        
    返回:
        变换后的点坐标，格式为(x, y)
    """
    # 将点转换为适合OpenCV处理的格式
    src_x, src_y = point
    src_point = np.array([[[src_x, src_y]]], dtype=np.float32)
    
    # 应用透视变换
    dst_point = cv2.perspectiveTransform(src_point, homography_matrix)
    
    # 提取结果并返回
    return (dst_point[0][0][0], dst_point[0][0][1])

if __name__ == "__main__":
    # 测试用例
    H = [
            [
                -0.4171420884831931,
                0.28049145461229275,
                970.4659835459827
            ],
            [
                -0.49451926155046566,
                0.5436427509591759,
                1122.5386902763921
            ],
            [
                -0.00046553582230021804,
                0.00043734716206250994,
                1.0
            ]
    ]  # 说明：该矩阵是非常敏感的。如果仅仅保留1位小数，则计算结果可能偏差100像素。
    
    # 转换为numpy数组以便OpenCV处理
    homography_matrix = np.array(H, dtype=np.float32)
    
    # 原始点坐标
    original_point = (947.20, 714.67)
    print(f"原始点坐标: {original_point}")
    
    # 计算对应点
    transformed_point = transform_point(original_point, homography_matrix)
    print(f"变换后的点坐标: ({transformed_point[0]:.2f}, {transformed_point[1]:.2f})")