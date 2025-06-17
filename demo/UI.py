import sys
import os
import cv2
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, 
                            QPushButton, QFileDialog, QHBoxLayout,
                            QVBoxLayout, QSlider, QGridLayout, 
                            QProgressBar, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from ultralytics import YOLO

class DetectionThread(QThread):
    progress_updated = pyqtSignal(int, int)  # current, total
    detection_finished = pyqtSignal(str)  # output file path
    detection_failed = pyqtSignal(str)  # error message

    def __init__(self, seq_path, model_path):
        super().__init__()
        self.seq_path = seq_path
        self.model_path = model_path
        self._is_running = True

    def run(self):
        try:
            img_dir = os.path.join(self.seq_path, 'img1')
            output_dir = os.path.join(self.seq_path, 'det')
            os.makedirs(output_dir, exist_ok=True)
            output_det_file = os.path.join(output_dir, 'det_yolov8x.txt')

            model = YOLO(self.model_path)
            frame_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
            total_frames = len(frame_files)

            with open(output_det_file, 'w') as f_out:
                for frame_id, img_name in enumerate(frame_files, 1):
                    if not self._is_running:
                        break

                    img_path = os.path.join(img_dir, img_name)
                    results = model(img_path, verbose=False)[0]

                    for box in results.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        w = x2 - x1
                        h = y2 - y1
                        conf = float(box.conf)
                        cls = int(box.cls)

                        if cls == 0:  # 行人类别
                            line = f"{frame_id},-1,{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1\n"
                            f_out.write(line)

                    self.progress_updated.emit(frame_id, total_frames)

            if self._is_running:
                self.detection_finished.emit(output_det_file)
        except Exception as e:
            self.detection_failed.emit(str(e))

    def stop(self):
        self._is_running = False

class DualFramePlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.frame_files_left = []
        self.frame_files_right = []
        self.current_idx_left = 0
        self.current_idx_right = 0
        self.playing_left = False
        self.playing_right = False
        self.timer_left = QTimer()
        self.timer_right = QTimer()
        
        # 检测相关
        self.detection_thread = None
        self.detection_model_path = 'yolov8x.pt'  # 默认模型路径
        
        self.init_ui()
        self.timer_left.timeout.connect(self.next_frame_left)
        self.timer_right.timeout.connect(self.next_frame_right)
        
    def init_ui(self):
        # 主布局
        main_layout = QVBoxLayout()
        
        # 图像显示区域 (使用网格布局)
        grid_layout = QGridLayout()
        
        # 左侧帧序列
        self.left_label = QLabel("左侧帧序列")
        self.left_label.setAlignment(Qt.AlignCenter)
        self.left_label.setStyleSheet("font-weight: bold;")
        grid_layout.addWidget(self.left_label, 0, 0)
        
        self.image_label_left = QLabel()
        self.image_label_left.setAlignment(Qt.AlignCenter)
        self.image_label_left.setMinimumSize(640, 480)
        grid_layout.addWidget(self.image_label_left, 1, 0)
        
        # 右侧帧序列
        self.right_label = QLabel("右侧帧序列")
        self.right_label.setAlignment(Qt.AlignCenter)
        self.right_label.setStyleSheet("font-weight: bold;")
        grid_layout.addWidget(self.right_label, 0, 1)
        
        self.image_label_right = QLabel()
        self.image_label_right.setAlignment(Qt.AlignCenter)
        self.image_label_right.setMinimumSize(640, 480)
        grid_layout.addWidget(self.image_label_right, 1, 1)
        
        main_layout.addLayout(grid_layout)
        
        # 检测控制区域
        detection_layout = QHBoxLayout()
        
        self.btn_detect = QPushButton('运行行人检测(左侧)')
        self.btn_detect.clicked.connect(self.run_detection)
        detection_layout.addWidget(self.btn_detect)
        
        self.btn_stop_detect = QPushButton('停止检测')
        self.btn_stop_detect.clicked.connect(self.stop_detection)
        self.btn_stop_detect.setEnabled(False)
        detection_layout.addWidget(self.btn_stop_detect)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setAlignment(Qt.AlignCenter)
        detection_layout.addWidget(self.progress_bar)
        
        main_layout.addLayout(detection_layout)
        
        # 控制按钮区域 (左侧)
        left_control_layout = QHBoxLayout()
        
        self.btn_load_left = QPushButton('加载左侧帧序列')
        self.btn_load_left.clicked.connect(lambda: self.load_frames('left'))
        left_control_layout.addWidget(self.btn_load_left)
        
        self.btn_play_left = QPushButton('播放左侧')
        self.btn_play_left.clicked.connect(lambda: self.toggle_play('left'))
        self.btn_play_left.setEnabled(False)
        left_control_layout.addWidget(self.btn_play_left)
        
        self.btn_prev_left = QPushButton('上一帧')
        self.btn_prev_left.clicked.connect(lambda: self.prev_frame('left'))
        self.btn_prev_left.setEnabled(False)
        left_control_layout.addWidget(self.btn_prev_left)
        
        self.btn_next_left = QPushButton('下一帧')
        self.btn_next_left.clicked.connect(lambda: self.next_frame('left'))
        self.btn_next_left.setEnabled(False)
        left_control_layout.addWidget(self.btn_next_left)
        
        main_layout.addLayout(left_control_layout)
        
        # 左侧进度条
        self.slider_left = QSlider(Qt.Horizontal)
        self.slider_left.valueChanged.connect(lambda v: self.slider_moved(v, 'left'))
        main_layout.addWidget(self.slider_left)
        
        # 控制按钮区域 (右侧)
        right_control_layout = QHBoxLayout()
        
        self.btn_load_right = QPushButton('加载右侧帧序列')
        self.btn_load_right.clicked.connect(lambda: self.load_frames('right'))
        right_control_layout.addWidget(self.btn_load_right)
        
        self.btn_play_right = QPushButton('播放右侧')
        self.btn_play_right.clicked.connect(lambda: self.toggle_play('right'))
        self.btn_play_right.setEnabled(False)
        right_control_layout.addWidget(self.btn_play_right)
        
        self.btn_prev_right = QPushButton('上一帧')
        self.btn_prev_right.clicked.connect(lambda: self.prev_frame('right'))
        self.btn_prev_right.setEnabled(False)
        right_control_layout.addWidget(self.btn_prev_right)
        
        self.btn_next_right = QPushButton('下一帧')
        self.btn_next_right.clicked.connect(lambda: self.next_frame('right'))
        self.btn_next_right.setEnabled(False)
        right_control_layout.addWidget(self.btn_next_right)
        
        main_layout.addLayout(right_control_layout)
        
        # 右侧进度条
        self.slider_right = QSlider(Qt.Horizontal)
        self.slider_right.valueChanged.connect(lambda v: self.slider_moved(v, 'right'))
        main_layout.addWidget(self.slider_right)
        
        # 同步播放控制
        sync_layout = QHBoxLayout()
        self.btn_sync_play = QPushButton('同步播放')
        self.btn_sync_play.clicked.connect(self.sync_play)
        self.btn_sync_play.setEnabled(False)
        sync_layout.addWidget(self.btn_sync_play)
        
        self.btn_sync_stop = QPushButton('同步停止')
        self.btn_sync_stop.clicked.connect(self.sync_stop)
        self.btn_sync_stop.setEnabled(False)
        sync_layout.addWidget(self.btn_sync_stop)
        
        main_layout.addLayout(sync_layout)
        
        self.setLayout(main_layout)
        self.setWindowTitle('双帧序列播放器(带行人检测)')
        self.resize(1200, 800)
    
    def load_frames(self, side):
        """选择包含帧序列的文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, f'选择{side}侧帧序列文件夹')
        if dir_path:
            frame_files = sorted([
                os.path.join(dir_path, f) 
                for f in os.listdir(dir_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            if frame_files:
                if side == 'left':
                    self.frame_files_left = frame_files
                    self.current_idx_left = 0
                    self.slider_left.setRange(0, len(self.frame_files_left)-1)
                    self.btn_play_left.setEnabled(True)
                    self.btn_prev_left.setEnabled(True)
                    self.btn_next_left.setEnabled(True)
                    self.btn_detect.setEnabled(True)
                else:
                    self.frame_files_right = frame_files
                    self.current_idx_right = 0
                    self.slider_right.setRange(0, len(self.frame_files_right)-1)
                    self.btn_play_right.setEnabled(True)
                    self.btn_prev_right.setEnabled(True)
                    self.btn_next_right.setEnabled(True)
                
                # 如果两侧都加载了序列，启用同步控制
                if self.frame_files_left and self.frame_files_right:
                    self.btn_sync_play.setEnabled(True)
                    self.btn_sync_stop.setEnabled(True)
                
                self.show_frame(side)
    
    def show_frame(self, side):
        """显示当前帧"""
        if side == 'left':
            if 0 <= self.current_idx_left < len(self.frame_files_left):
                pixmap = QPixmap(self.frame_files_left[self.current_idx_left])
                self.image_label_left.setPixmap(
                    pixmap.scaled(self.image_label_left.size(), 
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation))
                self.slider_left.setValue(self.current_idx_left)
        else:
            if 0 <= self.current_idx_right < len(self.frame_files_right):
                pixmap = QPixmap(self.frame_files_right[self.current_idx_right])
                self.image_label_right.setPixmap(
                    pixmap.scaled(self.image_label_right.size(), 
                                Qt.KeepAspectRatio,
                                Qt.SmoothTransformation))
                self.slider_right.setValue(self.current_idx_right)
    
    def toggle_play(self, side):
        """切换播放/暂停状态"""
        if side == 'left':
            self.playing_left = not self.playing_left
            self.btn_play_left.setText('暂停左侧' if self.playing_left else '播放左侧')
            
            if self.playing_left:
                self.timer_left.start(100)  # 100ms = 10fps
            else:
                self.timer_left.stop()
        else:
            self.playing_right = not self.playing_right
            self.btn_play_right.setText('暂停右侧' if self.playing_right else '播放右侧')
            
            if self.playing_right:
                self.timer_right.start(100)  # 100ms = 10fps
            else:
                self.timer_right.stop()
    
    def next_frame(self, side):
        """显示下一帧"""
        if side == 'left':
            if self.frame_files_left:
                self.current_idx_left = (self.current_idx_left + 1) % len(self.frame_files_left)
                self.show_frame('left')
        else:
            if self.frame_files_right:
                self.current_idx_right = (self.current_idx_right + 1) % len(self.frame_files_right)
                self.show_frame('right')
    
    def prev_frame(self, side):
        """显示上一帧"""
        if side == 'left':
            if self.frame_files_left:
                self.current_idx_left = (self.current_idx_left - 1) % len(self.frame_files_left)
                self.show_frame('left')
        else:
            if self.frame_files_right:
                self.current_idx_right = (self.current_idx_right - 1) % len(self.frame_files_right)
                self.show_frame('right')
    
    def slider_moved(self, value, side):
        """滑块拖动事件"""
        if side == 'left':
            if not self.timer_left.isActive():  # 防止播放时拖动冲突
                self.current_idx_left = value
                self.show_frame('left')
        else:
            if not self.timer_right.isActive():  # 防止播放时拖动冲突
                self.current_idx_right = value
                self.show_frame('right')
    
    def next_frame_left(self):
        self.next_frame('left')
    
    def next_frame_right(self):
        self.next_frame('right')
    
    def sync_play(self):
        """同步播放两侧序列"""
        self.playing_left = True
        self.playing_right = True
        self.btn_play_left.setText('暂停左侧')
        self.btn_play_right.setText('暂停右侧')
        self.timer_left.start(100)
        self.timer_right.start(100)
    
    def sync_stop(self):
        """同步停止两侧序列"""
        self.playing_left = False
        self.playing_right = False
        self.btn_play_left.setText('播放左侧')
        self.btn_play_right.setText('播放右侧')
        self.timer_left.stop()
        self.timer_right.stop()
    
    def run_detection(self):
        """运行行人检测"""
        if not self.frame_files_left:
            QMessageBox.warning(self, "警告", "请先加载左侧帧序列!")
            return
        
        # 获取序列目录 (假设帧序列在img1子目录中)
        seq_dir = os.path.dirname(os.path.dirname(self.frame_files_left[0]))
        
        # 检查YOLO模型文件
        if not os.path.exists(self.detection_model_path):
            model_path, _ = QFileDialog.getOpenFileName(
                self, "选择YOLO模型文件", "", "PyTorch模型 (*.pt)")
            if model_path:
                self.detection_model_path = model_path
            else:
                return
        
        # 禁用相关按钮
        self.btn_detect.setEnabled(False)
        self.btn_stop_detect.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # 创建并启动检测线程
        self.detection_thread = DetectionThread(seq_dir, self.detection_model_path)
        self.detection_thread.progress_updated.connect(self.update_detection_progress)
        self.detection_thread.detection_finished.connect(self.detection_completed)
        self.detection_thread.detection_failed.connect(self.detection_failed)
        self.detection_thread.start()
    
    def stop_detection(self):
        """停止检测过程"""
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait()
            self.progress_bar.setValue(0)
            QMessageBox.information(self, "信息", "检测已停止")
        
        self.btn_detect.setEnabled(True)
        self.btn_stop_detect.setEnabled(False)
    
    def update_detection_progress(self, current, total):
        """更新检测进度条"""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
    
    def detection_completed(self, output_file):
        """检测完成处理"""
        self.btn_detect.setEnabled(True)
        self.btn_stop_detect.setEnabled(False)
        QMessageBox.information(self, "完成", f"行人检测完成!\n结果已保存至:\n{output_file}")
    
    def detection_failed(self, error_msg):
        """检测失败处理"""
        self.btn_detect.setEnabled(True)
        self.btn_stop_detect.setEnabled(False)
        QMessageBox.critical(self, "错误", f"检测过程中发生错误:\n{error_msg}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = DualFramePlayer()
    player.show()
    sys.exit(app.exec_())