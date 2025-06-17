import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, 
                            QPushButton, QFileDialog, QHBoxLayout,
                            QVBoxLayout, QSlider, QGridLayout)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

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
        self.setWindowTitle('双帧序列播放器')
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = DualFramePlayer()
    player.show()
    sys.exit(app.exec_())