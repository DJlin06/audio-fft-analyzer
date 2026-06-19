"""
基于快速傅里叶变换(FFT)的音频信号频谱分析仪 —— 主程序入口

模块划分：
- fft_module.py : 手写Cooley-Tukey FFT/IFFT核心算法
- audio_io.py   : 音频读取、分帧、加窗处理
- gui.py        : PyQt5 + pyqtgraph 实时交互式可视化界面
"""

import sys
from PyQt5.QtWidgets import QApplication
import pyqtgraph as pg
from gui import SpectrumAnalyzer

def main():
    app = QApplication(sys.argv)
    win = SpectrumAnalyzer()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    print(sys.executable)
    from PyQt5.QtCore import QLibraryInfo
    print(QLibraryInfo.location(QLibraryInfo.PluginsPath))
    main()
