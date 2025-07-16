import sys
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog)
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QPoint, QSize


# 定义常量，便于统一修改
MAX_IMAGE_SIZE = 900


class ImageLabel(QLabel):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # 保存主窗口实例
        self.setAlignment(Qt.AlignCenter)
        self.setMouseTracking(True)
        self.points = []  # Stores points in YOLO format (normalized coordinates)
        self.current_point = None
        self.image = None
        self.pixmap = None
        self.scale_factor = 1.0  # 图像缩放因子
        self.original_size = QSize(0, 0)  # 原始图像尺寸
        self.display_size = QSize(0, 0)  # 显示图像尺寸
        
    def set_image(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is not None:
            self.original_size = QSize(self.image.shape[1], self.image.shape[0])
            print(f"原始图像分辨率: {self.original_size.width()}x{self.original_size.height()}")
            
            # 使用常量计算缩放后的尺寸
            max_size = MAX_IMAGE_SIZE
            width, height = self.original_size.width(), self.original_size.height()
            
            if width > height:
                # 宽边是长边
                self.scale_factor = max_size / width
                new_width = max_size
                new_height = int(height * self.scale_factor)
            else:
                # 高边是长边
                self.scale_factor = max_size / height
                new_height = max_size
                new_width = int(width * self.scale_factor)
                
            self.display_size = QSize(new_width, new_height)
            self.setFixedSize(self.display_size)  # 设置Label大小为缩放后的大小
            
            # 缩放图像
            self.image = cv2.resize(self.image, (new_width, new_height))
            print(f"显示图像分辨率: {new_width}x{new_height}")
            
            self.update_display()
            
    def update_display(self):
        if self.image is not None:
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_img = QImage(self.image.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            self.pixmap = QPixmap.fromImage(q_img)
            self.setPixmap(self.pixmap)
            
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.image is not None:
            x = event.pos().x()
            y = event.pos().y()
            
            if 0 <= x < self.image.shape[1] and 0 <= y < self.image.shape[0]:
                # 计算原始图像上的坐标
                original_x = x / self.scale_factor
                original_y = y / self.scale_factor
                
                # 转换为YOLO格式（归一化坐标，基于原始图像尺寸）
                norm_x = original_x / self.original_size.width()
                norm_y = original_y / self.original_size.height()
                
                self.current_point = (norm_x, norm_y)
                self.points.append(self.current_point)
                
                # 输出原始图像上的坐标和归一化坐标
                print(f"选择的点坐标 (原始图像): ({original_x:.2f}, {original_y:.2f})，归一化坐标: ({norm_x:.4f}, {norm_y:.4f})")
                self.update()

                # 查找对应点
                if self.main_window.homography is not None:
                    if self == self.main_window.map_label:
                        other_label = self.main_window.monitor_label
                        transformation_matrix = np.linalg.inv(self.main_window.homography)
                    else:
                        other_label = self.main_window.map_label
                        transformation_matrix = self.main_window.homography

                    # 将归一化坐标转换为原始图像坐标
                    src_x = norm_x * self.original_size.width()
                    src_y = norm_y * self.original_size.height()
                    src_point = np.array([[src_x, src_y]], dtype=np.float32).reshape(-1, 1, 2)

                    # 计算对应点
                    dst_point = cv2.perspectiveTransform(src_point, transformation_matrix)
                    dst_x = dst_point[0][0][0]
                    dst_y = dst_point[0][0][1]

                    # 将对应点转换为归一化坐标
                    dst_norm_x = dst_x / other_label.original_size.width()
                    dst_norm_y = dst_y / other_label.original_size.height()

                    # 添加对应点
                    other_label.current_point = (dst_norm_x, dst_norm_y)
                    other_label.points.append(other_label.current_point)
                    print("变换矩阵为：{}".format(np.round(transformation_matrix, decimals=1).tolist()))
                    print("对应的点坐标（原始图像）：（{}, {}），归一化坐标：（{}, {}）".format(dst_x, dst_y, dst_norm_x, dst_norm_y))
                    other_label.update()
                
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self.pixmap is None or self.image is None:
            return
            
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw points on current image
        pen = QPen(QColor(255, 0, 0), 5)
        painter.setPen(pen)
        
        for i, point in enumerate(self.points):
            # Convert normalized coordinates to display coordinates
            norm_x, norm_y = point
            
            # 计算原始图像上的坐标
            original_x = norm_x * self.original_size.width()
            original_y = norm_y * self.original_size.height()
            
            # 计算显示图像上的坐标
            display_x = original_x * self.scale_factor
            display_y = original_y * self.scale_factor
            
            scaled_point = QPoint(int(display_x), int(display_y))
            painter.drawEllipse(scaled_point, 5, 5)
            painter.drawText(scaled_point + QPoint(10, -10), str(i + 1))
        
        # Draw current point (if any)
        if self.current_point:
            norm_x, norm_y = self.current_point
            original_x = norm_x * self.original_size.width()
            original_y = norm_y * self.original_size.height()
            display_x = original_x * self.scale_factor
            display_y = original_y * self.scale_factor
            
            scaled_point = QPoint(int(display_x), int(display_y))
            pen = QPen(QColor(0, 0, 255), 7)
            painter.setPen(pen)
            painter.drawEllipse(scaled_point, 7, 7)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("地图与监控画面点对应标注工具")
        self.setGeometry(100, 100, 1200, 600)
        
        # Initialize UI
        self.init_ui()
        
        # Transformation matrix
        self.homography = None
        
    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Map image area
        self.map_label = ImageLabel(self)  # 传入主窗口实例
        self.map_label.set_image('/home/moonlet/projects/fast-reid/datasets/yisuo/map.png')
        
        # Right side: Monitor image and control panel
        right_layout = QVBoxLayout()
        
        # Monitor image area
        self.monitor_label = ImageLabel(self)  # 传入主窗口实例
        self.monitor_label.set_image('/home/moonlet/projects/fast-reid/datasets/yisuo/人脸追踪01/人脸追踪1_First_Frame.png')
        
        # Control buttons area
        control_panel = QVBoxLayout()
        
        self.calc_button = QPushButton("计算对应关系")
        self.calc_button.setFixedWidth(MAX_IMAGE_SIZE)  # 使用常量设置按钮宽度
        self.calc_button.clicked.connect(self.calculate_homography)
        
        self.save_button = QPushButton("保存对应关系")
        self.save_button.setFixedWidth(MAX_IMAGE_SIZE)  # 使用常量设置按钮宽度
        self.save_button.clicked.connect(self.save_homography)
        self.save_button.setEnabled(False)
        
        self.clear_button = QPushButton("清除所有点及对应关系")
        self.clear_button.setFixedWidth(MAX_IMAGE_SIZE)  # 使用常量设置按钮宽度
        self.clear_button.clicked.connect(self.clear_points)

        self.find_button = QPushButton("查找对应点(清除现有点)")
        self.find_button.setFixedWidth(MAX_IMAGE_SIZE)  # 使用常量设置按钮宽度
        self.find_button.clicked.connect(self.find_corresponding_points)
        self.find_button.setEnabled(False)
        
        self.status_label = QLabel("请在地图和监控画面中选择至少4对对应点")
        self.status_label.setFixedWidth(MAX_IMAGE_SIZE)  # 使用常量设置标签宽度
        
        control_panel.addWidget(self.calc_button)
        control_panel.addWidget(self.save_button)
        control_panel.addWidget(self.clear_button)
        control_panel.addWidget(self.find_button)
        control_panel.addWidget(self.status_label)
        control_panel.addStretch()
        
        # Add monitor label and control panel to right layout
        right_layout.addWidget(self.monitor_label)
        right_layout.addLayout(control_panel)
        
        # Add to main layout
        main_layout.addWidget(self.map_label)
        main_layout.addLayout(right_layout)
        
    def calculate_homography(self):
        map_points = self.map_label.points
        monitor_points = self.monitor_label.points
        
        if len(map_points) != len(monitor_points):
            self.status_label.setText("错误: 地图和监控画面中的点数不一致")
            return
            
        if len(map_points) < 4:
            self.status_label.setText("错误: 至少需要4对点来计算变换关系")
            return
            
        # Convert normalized points back to original image coordinates for OpenCV
        map_width, map_height = self.map_label.original_size.width(), self.map_label.original_size.height()
        monitor_width, monitor_height = self.monitor_label.original_size.width(), self.monitor_label.original_size.height()
        
        src_points = np.array([(p[0] * monitor_width, p[1] * monitor_height) for p in monitor_points], dtype=np.float32)
        dst_points = np.array([(p[0] * map_width, p[1] * map_height) for p in map_points], dtype=np.float32)
        
        # Calculate homography matrix
        self.homography, _ = cv2.findHomography(src_points, dst_points)
        
        if self.homography is not None:
            self.status_label.setText(f"计算成功! 共使用了{len(map_points)}对点")
            self.save_button.setEnabled(True)
            self.find_button.setEnabled(True)
        else:
            self.status_label.setText("计算失败，请尝试选择不同的点")
            
    def save_homography(self):
        if self.homography is None:
            self.status_label.setText("没有可保存的对应关系")
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "保存对应关系", "", "JSON Files (*.json);;All Files (*)", options=options)
            
        if file_name:
            # Ensure file extension is .json
            if not file_name.lower().endswith('.json'):
                file_name += '.json'
                
            # Prepare data to save
            data = {
                "homography": self.homography.tolist(),
                "map_points": self.map_label.points,  # Already in YOLO format
                "monitor_points": self.monitor_label.points,  # Already in YOLO format
                "map_image_size": [self.map_label.original_size.width(), self.map_label.original_size.height()],
                "monitor_image_size": [self.monitor_label.original_size.width(), self.monitor_label.original_size.height()]
            }
            print("保存的变换矩阵为：{}".format(np.round(self.homography, 1).tolist()))
            
            # Save to file
            with open(file_name, 'w') as f:
                json.dump(data, f, indent=4)
                
            self.status_label.setText(f"对应关系已保存到: {file_name}")
            
    def clear_points(self):
        self.map_label.points = []
        self.map_label.current_point = None
        self.monitor_label.points = []
        self.monitor_label.current_point = None
        self.homography = None
        self.save_button.setEnabled(False)
        self.find_button.setEnabled(False)
        self.map_label.update()
        self.monitor_label.update()
        self.status_label.setText("已清除所有点及对应关系")

    def find_corresponding_points(self):
        self.map_label.points = []
        self.map_label.current_point = None
        self.monitor_label.points = []
        self.monitor_label.current_point = None
        self.map_label.update()
        self.monitor_label.update()
        self.status_label.setText("请在地图或监控画面中选择一个点")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())