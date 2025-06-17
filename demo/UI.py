import sys
import os
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, 
                            QPushButton, QFileDialog, QHBoxLayout,
                            QVBoxLayout, QSlider)
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QTimer

class FramePlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.frame_files = []
        self.current_idx = 0
        self.playing = False
        self.timer = QTimer()
        
        self.init_ui()
        self.timer.timeout.connect(self.next_frame)
        
    def init_ui(self):
        # 主布局
        layout = QVBoxLayout()
        
        # 图像显示区域
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        layout.addWidget(self.image_label)
        
        # 控制按钮区域
        control_layout = QHBoxLayout()
        
        self.btn_load = QPushButton('加载帧序列')
        self.btn_load.clicked.connect(self.load_frames)
        control_layout.addWidget(self.btn_load)
        
        self.btn_play = QPushButton('播放')
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)
        control_layout.addWidget(self.btn_play)
        
        self.btn_prev = QPushButton('上一帧')
        self.btn_prev.clicked.connect(self.prev_frame)
        self.btn_prev.setEnabled(False)
        control_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton('下一帧')
        self.btn_next.clicked.connect(self.next_frame)
        self.btn_next.setEnabled(False)
        control_layout.addWidget(self.btn_next)
        
        layout.addLayout(control_layout)
        
        # 进度条
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.slider_moved)
        layout.addWidget(self.slider)
        
        self.setLayout(layout)
        self.setWindowTitle('帧序列播放器')
        self.resize(800, 600)
    
    def load_frames(self):
        """选择包含帧序列的文件夹"""
        dir_path = QFileDialog.getExistingDirectory(self, '选择帧序列文件夹')
        if dir_path:
            self.frame_files = sorted([
                os.path.join(dir_path, f) 
                for f in os.listdir(dir_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ])
            
            if self.frame_files:
                self.current_idx = 0
                self.slider.setRange(0, len(self.frame_files)-1)
                self.btn_play.setEnabled(True)
                self.btn_prev.setEnabled(True)
                self.btn_next.setEnabled(True)
                self.show_frame()
    
    def show_frame(self):
        """显示当前帧"""
        if 0 <= self.current_idx < len(self.frame_files):
            pixmap = QPixmap(self.frame_files[self.current_idx])
            self.image_label.setPixmap(
                pixmap.scaled(self.image_label.size(), 
                            Qt.KeepAspectRatio,
                            Qt.SmoothTransformation))
            self.slider.setValue(self.current_idx)
    
    def toggle_play(self):
        """切换播放/暂停状态"""
        self.playing = not self.playing
        self.btn_play.setText('暂停' if self.playing else '播放')
        
        if self.playing:
            self.timer.start(100)  # 100ms = 10fps
        else:
            self.timer.stop()
    
    def next_frame(self):
        """显示下一帧"""
        if self.frame_files:
            self.current_idx = (self.current_idx + 1) % len(self.frame_files)
            self.show_frame()
    
    def prev_frame(self):
        """显示上一帧"""
        if self.frame_files:
            self.current_idx = (self.current_idx - 1) % len(self.frame_files)
            self.show_frame()
    
    def slider_moved(self, value):
        """滑块拖动事件"""
        if not self.timer.isActive():  # 防止播放时拖动冲突
            self.current_idx = value
            self.show_frame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = FramePlayer()
    player.show()
    sys.exit(app.exec_())