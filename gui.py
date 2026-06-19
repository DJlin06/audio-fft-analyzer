"""
GUI Module
基于 PyQt5 + pyqtgraph 的实时音频频谱分析仪界面。
支持文件播放与麦克风实时输入两种模式。
"""

import os
import sys
import numpy as np
import pyqtgraph as pg
import sounddevice as sd
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QFileDialog, QCheckBox,
    QApplication,
)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

import fft_module
import audio_io


class SpectrumAnalyzer(QMainWindow):
    """
    主窗口：包含控制面板、时域波形图、频域频谱图。
    支持两种工作模式：
    - file: 加载音频文件并播放，画面跟随播放进度
    - mic:  实时采集麦克风输入，画面实时刷新
    """

    # 用于跨线程安全地请求停止播放/录音
    stop_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于FFT的音频频谱分析仪")
        self.resize(1200, 800)

        # 工作模式: "file" | "mic"
        self.mode = "file"

        # 音频数据（文件模式）
        self.fs = 8000
        self.signal = np.zeros(self.fs, dtype=np.float32)
        self.duration = 0.0
        self.is_playing = False
        self.current_sample = 0

        # 麦克风缓冲区（实时模式）使用 list 避免 np.concatenate 高频开销
        self.mic_buffer = []
        self.is_recording = False

        # 分析参数
        self.frame_size = 1024
        self.hop_size = 256          # 75% 重叠，减少频谱抖动
        self.n_fft = 1024
        self.window_name = "hamming"
        self.window = audio_io.get_window(self.window_name, self.frame_size)
        self.log_scale = False       # 默认线性幅度
        self.log_freq_axis = False   # 默认线性频率轴
        self.fft_display_mode = "split"  # "handwritten_only" | "split" | "overlay"

        # 播放/录音流
        self._stream = None

        # 文件模式：音频callback独立的播放位置
        self._play_pos = 0

        # 防重入标志，避免 stop() 重复执行导致 WASAPI 死锁
        self._closing = False

        # 跨线程停止信号连接
        self.stop_requested.connect(self.stop)

        # UI 构建
        self._init_ui()
        self._init_plots()
        self._init_timer()

    def _init_ui(self):
        """初始化控制面板与布局。"""
        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        # --- 第一行：操作按钮 ---
        hbox_btn = QHBoxLayout()

        self.btn_load = QPushButton("加载音频")
        self.btn_load.clicked.connect(self.load_audio)
        hbox_btn.addWidget(self.btn_load)

        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        hbox_btn.addWidget(self.btn_play)

        self.btn_record = QPushButton("开始录音")
        self.btn_record.clicked.connect(self.start_recording)
        hbox_btn.addWidget(self.btn_record)

        self.btn_stop_record = QPushButton("停止录音")
        self.btn_stop_record.clicked.connect(self.stop_recording)
        self.btn_stop_record.setEnabled(False)
        hbox_btn.addWidget(self.btn_stop_record)

        hbox_btn.addStretch()
        vbox.addLayout(hbox_btn)

        # --- 第二行：参数设置 ---
        hbox_params = QHBoxLayout()

        hbox_params.addWidget(QLabel("采样率:"))
        self.combo_fs = QComboBox()
        self.combo_fs.addItems(["8000", "11025", "16000", "22050", "44100", "48000"])
        self.combo_fs.setCurrentText(str(self.fs))
        self.combo_fs.currentTextChanged.connect(self.change_sample_rate)
        hbox_params.addWidget(self.combo_fs)

        hbox_params.addWidget(QLabel("窗函数:"))
        self.combo_window = QComboBox()
        self.combo_window.addItems(["hamming", "rectangular"])
        self.combo_window.currentTextChanged.connect(self.change_window)
        hbox_params.addWidget(self.combo_window)

        hbox_params.addWidget(QLabel("FFT显示:"))
        self.combo_fft_mode = QComboBox()
        self.combo_fft_mode.addItems(["仅手写FFT", "分图对比", "叠图对比"])
        self.combo_fft_mode.setCurrentText("分图对比")
        self.combo_fft_mode.currentTextChanged.connect(self.change_fft_display_mode)
        hbox_params.addWidget(self.combo_fft_mode)

        self.chk_log = QCheckBox("对数幅度(dB)")
        self.chk_log.stateChanged.connect(self.toggle_log_scale)
        hbox_params.addWidget(self.chk_log)

        self.chk_log_freq = QCheckBox("对数频率轴")
        self.chk_log_freq.stateChanged.connect(self.toggle_log_freq_axis)
        hbox_params.addWidget(self.chk_log_freq)

        hbox_params.addStretch()
        vbox.addLayout(hbox_params)

        # --- 第三行：输入信号指标 ---
        hbox_info = QHBoxLayout()

        self.lbl_info = QLabel("未加载音频")
        hbox_info.addWidget(self.lbl_info)

        self.lbl_peaks = QLabel("主频: --")
        hbox_info.addWidget(self.lbl_peaks)

        hbox_info.addStretch()
        vbox.addLayout(hbox_info)

        # --- 绘图区 ---
        self.plot_time = pg.PlotWidget(title="时域波形")
        self.plot_freq = pg.PlotWidget(title="频域频谱（手写FFT）")
        self.plot_freq_np = pg.PlotWidget(title="频域频谱（numpy.fft 对比）")
        self.plot_freq_np.setVisible(True)
        vbox.addWidget(self.plot_time)
        vbox.addWidget(self.plot_freq)
        vbox.addWidget(self.plot_freq_np)

    def _init_plots(self):
        """配置 pyqtgraph 绘图参数。"""
        # 时域波形
        self.plot_time.setLabel("left", "幅度")
        self.plot_time.setLabel("bottom", "时间", units="s")
        self.plot_time.setXRange(0, self.frame_size / self.fs, padding=0)
        self.plot_time.setYRange(-1.2, 1.2)
        self.curve_time = self.plot_time.plot(pen="y")

        # 频域频谱
        self.plot_freq.setLabel("left", "幅度")
        self.plot_freq.setLabel("bottom", "频率", units="Hz")
        self.plot_freq.setXRange(0, self.fs / 2, padding=0)
        self.plot_freq.setYRange(0, 1.0)
        self.curve_freq = self.plot_freq.plot(pen="c")

        # numpy.fft 对比频谱图（独立窗口）
        self.plot_freq_np.setLabel("left", "幅度")
        self.plot_freq_np.setLabel("bottom", "频率", units="Hz")
        self.plot_freq_np.setXRange(0, self.fs / 2, padding=0)
        self.plot_freq_np.setYRange(0, 1.0)
        self.curve_freq_np = self.plot_freq_np.plot(pen="r")

        # 叠图模式下的 numpy.fft 对比曲线（初始隐藏）
        self.curve_freq_overlay = self.plot_freq.plot(
            pen=pg.mkPen("r", style=Qt.DotLine)
        )
        self.curve_freq_overlay.setVisible(False)

        # 预计算频率轴（单边谱）
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)

    def _init_timer(self):
        """初始化定时器，用于驱动频谱的帧级更新。
        限制最小间隔 30ms，避免高采样率（如 48000Hz）下刷新过快导致 GUI 卡死。"""
        interval_ms = max(30, int(1000 * self.hop_size / self.fs))
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(interval_ms)

    # ------------------------------------------------------------------
    # 文件模式控制逻辑
    # ------------------------------------------------------------------

    def load_audio(self):
        """打开文件对话框，读取音频文件（支持 WAV/MP4/M4A/MP3/FLAC 等）。"""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择音频文件",
            "",
            "音频文件 (*.wav *.mp4 *.m4a *.mp3 *.flac *.ogg *.aiff);;所有文件 (*)",
        )
        if not path:
            return
        try:
            target_fs = int(self.combo_fs.currentText())
            self.fs, self.signal = audio_io.read_audio(path, target_fs=target_fs)
            self.duration = len(self.signal) / self.fs
            self.current_sample = 0
            self._play_pos = 0
            self.mode = "file"
            # 若之前录音改变了采样率，加载文件后恢复为选定的目标采样率并更新参数
            self._update_analysis_params()
            self.lbl_info.setText(
                f"已加载: {os.path.basename(path)} | fs={self.fs}Hz | "
                f"时长={self.duration:.2f}s | 点数={len(self.signal)}"
            )
            # 预显示第一帧
            self.update_frame(force_sample=0)
            # 恢复按钮状态
            self.btn_play.setEnabled(True)
            self.btn_record.setEnabled(True)
        except Exception as e:
            self.lbl_info.setText(f"加载失败: {e}")

    def toggle_play(self):
        """播放/暂停切换（仅文件模式）。支持播放到末尾后自动从头重复。"""
        if self.is_playing:
            self.stop()
        else:
            # 若已播放到末尾，自动重置到开头以便重复播放
            if self.current_sample >= len(self.signal):
                self.current_sample = 0
                self._play_pos = 0
            self.play_file()

    def play_file(self):
        """启动音频文件播放与定时器。"""
        if len(self.signal) == 0:
            return
        self.mode = "file"
        self.is_playing = True
        self.btn_play.setText("暂停")
        self.btn_record.setEnabled(False)

        # 音频callback独立的播放位置
        self._play_pos = self.current_sample

        def callback(outdata, frames, time_info, status):
            end = self._play_pos + frames
            if end > len(self.signal):
                remain = len(self.signal) - self._play_pos
                outdata[:remain, 0] = self.signal[self._play_pos:]
                outdata[remain:, 0] = 0.0
                self._play_pos = len(self.signal)
                self.stop_requested.emit()
            else:
                outdata[:, 0] = self.signal[self._play_pos:end]
                self._play_pos = end

        self._stream = sd.OutputStream(
            samplerate=self.fs,
            channels=1,
            dtype=np.float32,
            callback=callback,
        )
        self._stream.start()
        self.timer.start()

    def _update_analysis_params(self):
        """当采样率变化时，重新计算频率轴、时域范围和定时器间隔。"""
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.fs)
        self.plot_time.setXRange(0, self.frame_size / self.fs, padding=0)
        self._set_freq_axis_mode()
        # 限制最小间隔 30ms，避免高采样率下刷新过快卡死 GUI
        interval_ms = max(30, int(1000 * self.hop_size / self.fs))
        self.timer.setInterval(interval_ms)

    def _set_freq_axis_mode(self):
        """根据 log_freq_axis 状态统一设置频谱图 x 轴范围与刻度模式。"""
        self.plot_freq.setLogMode(x=self.log_freq_axis)
        if hasattr(self, "plot_freq_np"):
            self.plot_freq_np.setLogMode(x=self.log_freq_axis)
        if self.log_freq_axis:
            # 对数轴从 10 Hz 开始，避免 log(0) 越界
            self.plot_freq.setXRange(10, self.fs / 2, padding=0)
            if hasattr(self, "plot_freq_np"):
                self.plot_freq_np.setXRange(10, self.fs / 2, padding=0)
        else:
            self.plot_freq.setXRange(0, self.fs / 2, padding=0)
            if hasattr(self, "plot_freq_np"):
                self.plot_freq_np.setXRange(0, self.fs / 2, padding=0)

    def stop(self):
        """停止播放或录音，恢复 UI 状态。
        Windows WASAPI 下，后台线程调用 stop/close 可能与 COM STA 冲突导致死锁，
        因此在主线程使用 abort() 强制终止，并配合 processEvents() 让底层 callback 退出。
        """
        if self._closing:
            return
        self._closing = True

        self.is_playing = False
        self.is_recording = False
        self.btn_play.setText("播放")
        self.btn_play.setEnabled(True)
        self.btn_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        self.timer.stop()

        # 让 Qt 处理完正在排队的事件，减少 PortAudio callback 与 GUI 的竞争
        QApplication.processEvents()

        stream = self._stream
        self._stream = None

        if stream is not None:
            try:
                # abort() 立即停止且不等待缓冲区排空，比 stop() 更不容易阻塞
                stream.abort()
                stream.close()
            except Exception:
                pass

        self._closing = False

    # ------------------------------------------------------------------
    # 实时麦克风模式控制逻辑
    # ------------------------------------------------------------------

    def start_recording(self):
        """
        启动麦克风实时输入与频谱显示。
        为避免 sd.query_devices() 在某些 Windows 驱动下阻塞 GUI 主线程，
        直接尝试打开 InputStream，失败时自动回退到备选参数。
        """
        self.mode = "mic"
        self.is_recording = True
        self.mic_buffer = []  # 使用 list 避免 np.concatenate 高频开销

        self.btn_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)
        self.btn_play.setEnabled(False)

        def callback(indata, frames, time_info, status):
            # 自动选择能量较大的声道，避免某个声道静音导致无数据
            if indata.ndim > 1 and indata.shape[1] >= 2:
                e0 = np.mean(np.abs(indata[:, 0]))
                e1 = np.mean(np.abs(indata[:, 1]))
                mono = indata[:, 0 if e0 >= e1 else 1].astype(np.float32)
            else:
                mono = indata[:, 0].astype(np.float32)

            # 用 list.extend 追加，性能远高于 np.concatenate
            self.mic_buffer.extend(mono.tolist())
            # 限制总长度，防止内存无限增长
            max_len = self.frame_size * 4
            if len(self.mic_buffer) > max_len:
                del self.mic_buffer[:-max_len]

        # 备选配置列表：(samplerate, channels, label)
        # 优先尝试用户设备原生参数 48000Hz/2ch，再回退通用参数
        candidates = [
            (48000, 2, "48000Hz 立体声"),
            (44100, 2, "44100Hz 立体声"),
            (8000, 1, "8000Hz 单声道"),
            (self.fs, 1, f"{self.fs}Hz 单声道"),
        ]

        last_err = None
        for sr, ch, label in candidates:
            try:
                self._stream = sd.InputStream(
                    samplerate=sr,
                    channels=ch,
                    dtype=np.float32,
                    callback=callback,
                )
                self._stream.start()
                # 若实际采样率与当前分析参数不同，自适应更新
                if sr != self.fs:
                    self.fs = sr
                    self._update_analysis_params()
                    self.combo_fs.setCurrentText(str(sr))
                self.lbl_info.setText(f"实时输入中 | {label}")
                self.timer.start()
                return  # 成功启动，直接返回
            except Exception as e:
                last_err = e
                continue  # 尝试下一组参数

        # 所有配置均失败
        self.lbl_info.setText(f"录音启动失败: {last_err}")
        self.stop()

    def stop_recording(self):
        """停止麦克风输入。"""
        self.stop()
        self.lbl_info.setText("录音已停止")

    # ------------------------------------------------------------------
    # 公共控制
    # ------------------------------------------------------------------

    def change_window(self, name: str):
        """切换窗函数。"""
        self.window_name = name
        self.window = audio_io.get_window(name, self.frame_size)
        if not self.is_playing and not self.is_recording:
            self.update_frame(force_sample=self.current_sample)

    def toggle_log_scale(self, state):
        """切换线性/对数幅度刻度。"""
        self.log_scale = (state == Qt.Checked)
        if self.log_scale:
            self.plot_freq.setLabel("left", "幅度 (dB)")
        else:
            self.plot_freq.setLabel("left", "幅度")
        if not self.is_playing and not self.is_recording:
            self.update_frame(force_sample=self.current_sample)

    def toggle_log_freq_axis(self, state):
        """切换线性/对数频率轴刻度。"""
        self.log_freq_axis = (state == Qt.Checked)
        self._set_freq_axis_mode()
        if not self.is_playing and not self.is_recording:
            self.update_frame(force_sample=self.current_sample)

    def change_sample_rate(self, text: str):
        """修改目标采样率，已加载的音频会自动重采样。"""
        new_fs = int(text)
        if new_fs == self.fs:
            return
        self.stop()
        if self.duration > 0:
            self.signal = audio_io._resample_anti_alias(self.signal, self.fs, new_fs)
            self.duration = len(self.signal) / new_fs
        else:
            self.signal = np.zeros(new_fs, dtype=np.float32)
            self.duration = 0.0
        self.fs = new_fs
        self.current_sample = 0
        self._play_pos = 0
        self._update_analysis_params()
        if self.duration > 0:
            self.lbl_info.setText(
                f"已加载音频 | fs={self.fs}Hz | 时长={self.duration:.2f}s | 点数={len(self.signal)}"
            )
        else:
            self.lbl_info.setText(f"采样率已改为 {self.fs} Hz")
        self.update_frame(force_sample=0)

    def change_fft_display_mode(self, text: str):
        """切换 FFT 显示模式：仅手写 / 分图 / 叠图。"""
        mode_map = {
            "仅手写FFT": "handwritten_only",
            "分图对比": "split",
            "叠图对比": "overlay",
        }
        self.fft_display_mode = mode_map.get(text, "split")
        if self.fft_display_mode == "handwritten_only":
            self.plot_freq.setTitle("频域频谱（手写FFT）")
            self.plot_freq_np.setVisible(False)
            self.curve_freq_overlay.setVisible(False)
        elif self.fft_display_mode == "split":
            self.plot_freq.setTitle("频域频谱（手写FFT）")
            self.plot_freq_np.setVisible(True)
            self.curve_freq_overlay.setVisible(False)
        elif self.fft_display_mode == "overlay":
            self.plot_freq.setTitle("频域频谱（手写FFT + numpy.fft 叠图）")
            self.plot_freq_np.setVisible(False)
            self.curve_freq_overlay.setVisible(True)
        if not self.is_playing and not self.is_recording:
            self.update_frame(force_sample=self.current_sample)

    # ------------------------------------------------------------------
    # 核心更新
    # ------------------------------------------------------------------

    def update_frame(self, force_sample=None):
        """
        定时器槽函数：提取当前帧，计算FFT，更新绘图。
        根据当前模式（file / mic）选择不同的数据源。
        """
        # ----- 数据源选择 -----
        if self.mode == "mic" and self.is_recording:
            # 实时模式：取缓冲区最新一帧
            buf_len = len(self.mic_buffer)
            if buf_len < self.frame_size:
                return  # 缓冲区尚未填满，跳过本次刷新
            frame = np.array(self.mic_buffer[-self.frame_size:], dtype=np.float32)
            # 消费掉 hop_size 个旧样本，实现滚动效果
            self.mic_buffer = self.mic_buffer[self.hop_size:]
        else:
            # 文件模式
            if force_sample is not None:
                self.current_sample = force_sample
            elif self.is_playing:
                self.current_sample = self._play_pos

            if self.current_sample + self.frame_size > len(self.signal):
                self.stop()
                return

            frame = self.signal[
                self.current_sample : self.current_sample + self.frame_size
            ].copy()

        # ----- 加窗 -----
        frame *= self.window

        # ----- 手写FFT + 归一化 -----
        X = fft_module.fft(frame)
        amp = np.abs(X[: self.n_fft // 2 + 1])

        # 单边谱归一化
        amp[1:-1] *= 2.0
        amp /= self.n_fft
        amp /= np.mean(self.window)

        # ----- 对数/线性转换 -----
        if self.log_scale:
            amp_db = 20.0 * np.log10(amp + 1e-12)
            self.plot_freq.setYRange(-100, 0)
            y_data = amp_db
        else:
            self.plot_freq.setYRange(0, 1.2)
            y_data = amp

        # ----- 更新图形 -----
        t = np.arange(self.frame_size) / self.fs
        self.curve_time.setData(t, frame)
        self.curve_freq.setData(self.freqs, y_data)

        # ----- numpy.fft 对比 -----
        if self.fft_display_mode != "handwritten_only":
            X_np = np.fft.fft(frame, n=self.n_fft)
            amp_np = np.abs(X_np[: self.n_fft // 2 + 1])
            amp_np[1:-1] *= 2.0
            amp_np /= self.n_fft
            amp_np /= np.mean(self.window)
            if self.log_scale:
                y_np = 20.0 * np.log10(amp_np + 1e-12)
            else:
                y_np = amp_np

            if self.fft_display_mode == "split":
                if self.log_scale:
                    self.plot_freq_np.setYRange(-100, 0)
                else:
                    self.plot_freq_np.setYRange(0, 1.2)
                self.curve_freq_np.setData(self.freqs, y_np)
                self.plot_freq_np.setVisible(True)
                self.curve_freq_overlay.setVisible(False)
            elif self.fft_display_mode == "overlay":
                self.curve_freq_overlay.setData(self.freqs, y_np)
                self.curve_freq_overlay.setVisible(True)
                self.plot_freq_np.setVisible(False)
        else:
            self.plot_freq_np.setVisible(False)
            self.curve_freq_overlay.setVisible(False)

        # ----- 实时检测主要频率成分 -----
        peaks = []
        # 跳过直流分量 (索引 0) 和奈奎斯特频率 (索引 -1)，避免边界干扰
        for i in range(1, len(amp) - 1):
            if amp[i] > amp[i - 1] and amp[i] > amp[i + 1] and amp[i] > 0.01:
                peaks.append((self.freqs[i], amp[i]))
        # 按幅度降序取前 3 个
        peaks.sort(key=lambda p: p[1], reverse=True)
        top_peaks = peaks[:3]
        if top_peaks:
            peak_text = " | ".join([f"{f:.0f}Hz" for f, _ in top_peaks])
            self.lbl_peaks.setText(f"主频: {peak_text}")
        else:
            self.lbl_peaks.setText("主频: --")

        # ----- 文件模式下非播放状态推进指针 -----
        if self.mode == "file" and force_sample is None and not self.is_playing:
            self.current_sample += self.hop_size
            if self.current_sample + self.frame_size > len(self.signal):
                self.current_sample = 0


if __name__ == "__main__":
    app = pg.mkQApp()
    win = SpectrumAnalyzer()
    win.show()
    sys.exit(app.exec_())
