import sys
import os
import glob
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, QFileDialog, 
                            QHBoxLayout, QVBoxLayout, QSlider, QGridLayout, QGroupBox,
                            QMessageBox, QListWidget)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QFont, QImage
from PyQt5.QtCore import Qt, pyqtSignal, QPoint

class ClickableLabel(QLabel):
    clicked = pyqtSignal(QPoint)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: #222;")
        self.setAlignment(Qt.AlignCenter)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)

class PersonSearchApp(QWidget):
    def __init__(self):
        super().__init__()
        self.seq_path_left = None
        self.seq_path_right = None
        self.detections_left = {}
        self.detections_right = {}
        self.current_query = None
        self.candidate_persons = []
        self.current_left_frame = 1
        self.current_right_frame = 1
        self.original_img_size = {}  # 存储原始图像尺寸 (width, height)
        
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle('跨视频行人搜索系统 (YOLO坐标格式)')
        self.resize(1400, 900)
        
        # 主布局
        main_layout = QVBoxLayout()
        
        # 序列选择区域
        seq_select_layout = QHBoxLayout()
        
        # 左侧序列选择
        left_seq_group = QGroupBox("左侧查询视频")
        left_seq_layout = QVBoxLayout()
        self.btn_load_left = QPushButton('加载左侧序列')
        self.btn_load_left.clicked.connect(lambda: self.load_sequence('left'))
        left_seq_layout.addWidget(self.btn_load_left)
        self.left_seq_label = QLabel('未加载')
        left_seq_layout.addWidget(self.left_seq_label)
        left_seq_group.setLayout(left_seq_layout)
        seq_select_layout.addWidget(left_seq_group)
        
        # 右侧序列选择
        right_seq_group = QGroupBox("右侧检索视频")
        right_seq_layout = QVBoxLayout()
        self.btn_load_right = QPushButton('加载右侧序列')
        self.btn_load_right.clicked.connect(lambda: self.load_sequence('right'))
        right_seq_layout.addWidget(self.btn_load_right)
        self.right_seq_label = QLabel('未加载')
        right_seq_layout.addWidget(self.right_seq_label)
        right_seq_group.setLayout(right_seq_layout)
        seq_select_layout.addWidget(right_seq_group)
        
        main_layout.addLayout(seq_select_layout)
        
        # 视频显示区域
        video_layout = QHBoxLayout()
        
        # 左侧视频
        left_video_group = QGroupBox("查询视频 (点击选择行人)")
        left_video_layout = QVBoxLayout()
        self.left_image_label = ClickableLabel()
        self.left_image_label.setMinimumSize(640, 480)
        self.left_image_label.clicked.connect(self.handle_left_click)
        left_video_layout.addWidget(self.left_image_label)
        
        # 左侧控制
        left_control = QHBoxLayout()
        self.btn_prev_left = QPushButton('上一帧')
        self.btn_prev_left.clicked.connect(lambda: self.change_frame('left', -1))
        left_control.addWidget(self.btn_prev_left)
        self.btn_next_left = QPushButton('下一帧')
        self.btn_next_left.clicked.connect(lambda: self.change_frame('left', 1))
        left_control.addWidget(self.btn_next_left)
        self.slider_left = QSlider(Qt.Horizontal)
        self.slider_left.valueChanged.connect(lambda v: self.slider_moved('left', v))
        left_control.addWidget(self.slider_left)
        left_video_layout.addLayout(left_control)
        left_video_group.setLayout(left_video_layout)
        video_layout.addWidget(left_video_group)
        
        # 右侧视频
        right_video_group = QGroupBox("检索视频")
        right_video_layout = QVBoxLayout()
        self.right_image_label = ClickableLabel()
        self.right_image_label.setMinimumSize(640, 480)
        right_video_layout.addWidget(self.right_image_label)
        
        # 右侧控制
        right_control = QHBoxLayout()
        self.btn_prev_right = QPushButton('上一帧')
        self.btn_prev_right.clicked.connect(lambda: self.change_frame('right', -1))
        right_control.addWidget(self.btn_prev_right)
        self.btn_next_right = QPushButton('下一帧')
        self.btn_next_right.clicked.connect(lambda: self.change_frame('right', 1))
        right_control.addWidget(self.btn_next_right)
        self.slider_right = QSlider(Qt.Horizontal)
        self.slider_right.valueChanged.connect(lambda v: self.slider_moved('right', v))
        right_control.addWidget(self.slider_right)
        right_video_layout.addLayout(right_control)
        right_video_group.setLayout(right_video_layout)
        video_layout.addWidget(right_video_group)
        
        main_layout.addLayout(video_layout)
        
        # 结果显示区域
        result_layout = QHBoxLayout()
        
        # 查询结果
        query_group = QGroupBox("查询人物")
        query_layout = QVBoxLayout()
        self.query_image_label = QLabel()
        self.query_image_label.setAlignment(Qt.AlignCenter)
        self.query_image_label.setFixedSize(200, 200)
        self.query_image_label.setStyleSheet("border: 2px solid gray;")
        query_layout.addWidget(self.query_image_label)
        self.query_info_label = QLabel("未选择查询人物")
        query_layout.addWidget(self.query_info_label)
        query_group.setLayout(query_layout)
        result_layout.addWidget(query_group)
        
        # 候选结果
        candidate_group = QGroupBox("候选人物 (点击跳转)")
        candidate_layout = QVBoxLayout()
        self.candidate_list = QListWidget()
        self.candidate_list.itemClicked.connect(self.handle_candidate_click)
        candidate_layout.addWidget(self.candidate_list)
        candidate_group.setLayout(candidate_layout)
        result_layout.addWidget(candidate_group)
        
        main_layout.addLayout(result_layout)
        self.setLayout(main_layout)
        
        # 初始状态
        self.update_controls()
    
    def load_sequence(self, side):
        dir_path = QFileDialog.getExistingDirectory(self, f'选择{side}侧视频序列')
        if not dir_path:
            return
            
        # 检查必要目录
        required_dirs = ['img1', 'det', 'person_crops', 'reid_features']
        missing_dirs = [d for d in required_dirs if not os.path.exists(os.path.join(dir_path, d))]
        
        if missing_dirs:
            QMessageBox.warning(self, "警告", f"缺少必要目录: {', '.join(missing_dirs)}")
            return
        
        if side == 'left':
            self.seq_path_left = dir_path
            self.left_seq_label.setText(os.path.basename(dir_path))
            self.load_detections('left')
        else:
            self.seq_path_right = dir_path
            self.right_seq_label.setText(os.path.basename(dir_path))
            self.load_detections('right')
        
        self.load_frame(side, 1)
        self.update_controls()
    
    def load_frame(self, side, frame_id):
        seq_path = self.seq_path_left if side == 'left' else self.seq_path_right
        if not seq_path:
            return
        
        img_path = os.path.join(seq_path, 'img1', f"{frame_id:06d}.jpg")
        
        if os.path.exists(img_path):
            # 读取图像并存储原始尺寸
            img = cv2.imread(img_path)
            self.original_img_size[side] = (img.shape[1], img.shape[0])  # (width, height)
            
            # 转换为QPixmap并显示
            pixmap = self.cv2_to_pixmap(img)
            
            if side == 'left':
                self.current_left_frame = frame_id
                self.current_left_img_path = img_path  # 添加此行
                self.slider_left.setMaximum(self.get_total_frames('left'))
                self.slider_left.setValue(frame_id-1)
                self.left_image_label.setPixmap(pixmap.scaled(
                    self.left_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.current_right_frame = frame_id
                self.current_right_img_path = img_path  # 可选
                self.slider_right.setMaximum(self.get_total_frames('right'))
                self.slider_right.setValue(frame_id-1)
                self.right_image_label.setPixmap(pixmap.scaled(
                    self.right_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
    
    def cv2_to_pixmap(self, cv_img):
        """将OpenCV图像转换为QPixmap"""
        height, width, channel = cv_img.shape
        bytes_per_line = 3 * width
        q_img = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        return QPixmap.fromImage(q_img)
    
    def get_total_frames(self, side):
        seq_path = self.seq_path_left if side == 'left' else self.seq_path_right
        if not seq_path:
            return 0
        img_dir = os.path.join(seq_path, 'img1')
        frames = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        return frames
    
    def load_detections(self, side):
        seq_path = self.seq_path_left if side == 'left' else self.seq_path_right
        if not seq_path:
            return
        
        det_file = os.path.join(seq_path, 'det', 'det_yolov8x.txt')
        
        if os.path.exists(det_file):
            detections = {}
            with open(det_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 10:  # 确保有足够字段
                        continue
                    
                    try:
                        frame_id = int(parts[0])
                        detection_id = int(parts[9])
                        # YOLO格式: frame_id,-1,x_center,y_center,width,height,conf,-1,-1,detection_id
                        bbox = list(map(float, parts[2:6]))  # x_center, y_center, width, height (归一化)
                        conf = float(parts[6])
                        
                        if frame_id not in detections:
                            detections[frame_id] = []
                        
                        detections[frame_id].append({
                            'id': detection_id,
                            'bbox': bbox,  # 存储归一化坐标
                            'conf': conf
                        })
                    except Exception as e:
                        print(f"解析检测行错误: {line.strip()} | {str(e)}")
                        continue
            
            if side == 'left':
                self.detections_left = detections
            else:
                self.detections_right = detections

    def handle_left_click(self, pos):
        if not self.seq_path_left or not self.detections_left:
            QMessageBox.warning(self, "警告", "请先加载左侧视频序列!")
            return
            
        if self.current_left_frame not in self.detections_left:
            QMessageBox.information(self, "提示", "当前帧没有检测到行人")
            return
            
        pixmap = self.left_image_label.pixmap()
        if not pixmap:
            return
        
        # 获取UI中图像的显示区域和缩放比例
        img_size = pixmap.size()
        label_size = self.left_image_label.size()
        
        # 计算缩放比例和偏移 (保持宽高比居中显示)
        w_ratio = label_size.width() / img_size.width()
        h_ratio = label_size.height() / img_size.height()
        scale = min(w_ratio, h_ratio)
        
        offset_x = (label_size.width() - img_size.width() * scale) / 2
        offset_y = (label_size.height() - img_size.height() * scale) / 2
        
        # 检查点击是否在图像区域内
        if not (offset_x <= pos.x() < label_size.width() - offset_x and 
                offset_y <= pos.y() < label_size.height() - offset_y):
            QMessageBox.information(self, "提示", "请点击图像区域内")
            return
        
        # 转换为归一化坐标 (相对于原始图像)
        norm_x = (pos.x() - offset_x) / (img_size.width() * scale)
        norm_y = (pos.y() - offset_y) / (img_size.height() * scale)
        
        # 查找点击的行人 (使用归一化坐标比较)
        frame_dets = self.detections_left[self.current_left_frame]
        selected_person = None
        min_dist = float('inf')
        
        for det in frame_dets:
            # YOLO格式: [x_center, y_center, width, height] 都是归一化坐标
            x_center, y_center, width, height = det['bbox']
            
            # 计算点击点到bbox中心的距离
            dist = ((norm_x - x_center)**2 + (norm_y - y_center)**2)**0.5
            
            # 选择距离最近且点击点在bbox内的检测框
            if dist < min_dist and (abs(norm_x - x_center) <= width/2 and 
                                    abs(norm_y - y_center) <= height/2):
                min_dist = dist
                selected_person = det
        
        if selected_person:
            self.current_query = {
                'id': selected_person['id'],
                'frame': self.current_left_frame,
                'conf': selected_person['conf'],
                'seq_path': self.seq_path_left,
                'bbox': selected_person['bbox']  # 存储归一化坐标
            }
            self.show_query_person()
            self.draw_selection_effect(selected_person['bbox'], offset_x, offset_y, scale)
            
            # 修改ReID特征路径查找方式，匹配demo.py中的格式
            crop_dir = os.path.join(self.seq_path_left, 'person_crops')
            crop_files = glob.glob(os.path.join(crop_dir, f"{selected_person['id']:06d}_*.jpg"))
            
            if crop_files:
                # 获取第一个匹配的裁剪图片文件名
                crop_file = os.path.basename(crop_files[0])
                base_name = os.path.splitext(crop_file)[0]  # 去掉扩展名
                reid_path = os.path.join(self.seq_path_left, 'reid_features', f"{base_name}.npy")
                print(f"选中行人的ReID特征路径: {reid_path}")
            else:
                print(f"警告: 未找到ID为{selected_person['id']}的裁剪图片")
        else:
            QMessageBox.information(self, "提示", "未检测到点击位置有行人\n请尝试点击行人身体中心区域")

    def draw_selection_effect(self, norm_bbox, offset_x, offset_y, scale):
        """在UI上绘制选中行人的效果"""
        # 从原图重新读取图像，防止覆盖错误
        img = cv2.imread(self.current_left_img_path)
        if img is None:
            return
        
        height, width = img.shape[:2]

        # 转换归一化bbox为绝对坐标
        x_center, y_center, w, h = norm_bbox
        abs_x = int(x_center * width)
        abs_y = int(y_center * height)
        abs_w = int(w * width)
        abs_h = int(h * height)

        # 画框
        cv2.rectangle(img,
                    (abs_x - abs_w // 2, abs_y - abs_h // 2),
                    (abs_x + abs_w // 2, abs_y + abs_h // 2),
                    (0, 255, 0), 3)
        cv2.putText(img, f"ID: {self.current_query['id']}",
                    (abs_x - abs_w // 2 + 10, abs_y - abs_h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 显示
        pixmap = self.cv2_to_pixmap(img)
        self.left_image_label.setPixmap(pixmap.scaled(
            self.left_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_query_person(self):
        if not self.current_query:
            return
        
        # 加载裁剪图像
        crop_dir = os.path.join(self.current_query['seq_path'], 'person_crops')
        crop_path = os.path.join(crop_dir, f"{self.current_query['id']:06d}_*.jpg")
        crop_files = glob.glob(crop_path)
        
        if crop_files:
            pixmap = QPixmap(crop_files[0]).scaled(
                200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.query_image_label.setPixmap(pixmap)
        else:
            self.query_image_label.clear()
        
        # 显示信息
        info = f"ID: {self.current_query['id']}\n"
        info += f"帧号: {self.current_query['frame']}\n"
        info += f"置信度: {self.current_query['conf']:.2f}"
        self.query_info_label.setText(info)

    def update_candidate_list(self):
        self.candidate_list.clear()
        for i, person in enumerate(self.candidate_persons[:20]):  # 只显示前20个结果
            item_text = f"ID: {person['id']} | 相似度: {person['similarity']:.4f} | 帧: {person['frame']}"
            self.candidate_list.addItem(item_text)

    def handle_candidate_click(self, item):
        if not self.seq_path_right:
            return
        
        selected_idx = self.candidate_list.row(item)
        if 0 <= selected_idx < len(self.candidate_persons):
            person = self.candidate_persons[selected_idx]
            self.load_frame('right', person['frame'])
            self.highlight_selected_person('right', person['id'])

    def highlight_selected_person(self, side, person_id):
        if side == 'left':
            frame_id = self.current_left_frame
            seq_path = self.seq_path_left
            detections = self.detections_left
        else:
            frame_id = self.current_right_frame
            seq_path = self.seq_path_right
            detections = self.detections_right
        
        if frame_id not in detections:
            return
        
        # 查找指定人物
        selected_det = None
        for det in detections[frame_id]:
            if det['id'] == person_id:
                selected_det = det
                break
        
        if selected_det:
            # 加载原始图像
            img_path = os.path.join(seq_path, 'img1', f"{frame_id:06d}.jpg")
            if not os.path.exists(img_path):
                return
            
            img = cv2.imread(img_path)
            height, width = img.shape[:2]
            
            # 转换归一化坐标到绝对坐标
            x_center, y_center, w, h = selected_det['bbox']
            abs_x = int(x_center * width)
            abs_y = int(y_center * height)
            abs_w = int(w * width)
            abs_h = int(h * height)
            
            # 绘制边框和ID
            cv2.rectangle(img, 
                         (abs_x - abs_w//2, abs_y - abs_h//2),
                         (abs_x + abs_w//2, abs_y + abs_h//2),
                         (0, 255, 0), 3)
            
            cv2.putText(img, f"ID: {person_id}", 
                       (abs_x - abs_w//2 + 10, abs_y - abs_h//2 + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # 显示图像
            pixmap = self.cv2_to_pixmap(img)
            if side == 'left':
                self.left_image_label.setPixmap(pixmap.scaled(
                    self.left_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.right_image_label.setPixmap(pixmap.scaled(
                    self.right_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def change_frame(self, side, delta):
        current_frame = self.current_left_frame if side == 'left' else self.current_right_frame
        total_frames = self.get_total_frames(side)
        new_frame = max(1, min(total_frames, current_frame + delta))
        self.load_frame(side, new_frame)

    def slider_moved(self, side, value):
        frame_id = value + 1  # 滑块从0开始，帧号从1开始
        self.load_frame(side, frame_id)

    def update_controls(self):
        has_left = self.seq_path_left is not None
        has_right = self.seq_path_right is not None
        
        self.btn_prev_left.setEnabled(has_left)
        self.btn_next_left.setEnabled(has_left)
        self.slider_left.setEnabled(has_left)
        
        self.btn_prev_right.setEnabled(has_right)
        self.btn_next_right.setEnabled(has_right)
        self.slider_right.setEnabled(has_right)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = PersonSearchApp()
    window.show()
    sys.exit(app.exec_())