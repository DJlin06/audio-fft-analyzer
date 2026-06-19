# 基于 FFT 的音频频谱分析仪

> 信号与系统课程设计项目 —— 手写 FFT 算法 + 实时音频频谱可视化

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)

## 项目简介

本课题实现了一个完整的**实时音频频谱分析仪**，核心亮点是从零手写 **Cooley-Tukey 基-2 DIT FFT 算法**（禁止调用 `numpy.fft` / `scipy.fft`），并提供 PyQt5 图形界面进行实时频谱可视化。

### 主要特性

- **手写 FFT 算法**：基于 Cooley-Tukey 基-2 按时间抽取 (DIT) 的迭代实现，含位反转重排、蝶形运算、自动补零
- **双模式工作**：
  - 📁 **文件模式**：加载 WAV/MP3/MP4/FLAC 等音频文件，播放画面跟随进度
  - 🎤 **麦克风模式**：实时采集麦克风输入，毫秒级频谱刷新
- **GUI 可视化界面**（PyQt5 + pyqtgraph）：
  - 时域波形 + 频域频谱图
  - 手写 FFT 与 numpy.fft 分图 / 叠图对比
  - 线性 / 对数幅度 (dB) 切换
  - 线性 / 对数频率轴切换
  - Hamming 窗 / 矩形窗可选
  - 实时主频峰值检测
- **扎实的信号处理**：
  - 零填充 (Zero-padding)：非 2 的幂长度输入自动补零
  - 抗混叠重采样：`scipy.signal.resample_poly` 多相滤波
  - 幅度谱归一化 + 单边谱窗口补偿
  - 75% 重叠帧移，减少频谱抖动
- **完整测试覆盖**：单元测试 + 端到端流水线测试 + 复杂度对比

---

## 项目结构

```
demo/
├── fft_module.py          # FFT 核心算法（手写 Cooley-Tukey DIT-FFT + IFFT + 幅度/功率谱）
├── audio_io.py            # 音频 I/O 模块（读取/分帧/加窗/重采样）
├── gui.py                 # 主 GUI 程序（PyQt5 + pyqtgraph）
├── test_fft.py            # FFT 模块单元测试（5项测试 + 对比图生成）
├── test_end_to_end.py     # 端到端流水线测试
├── test_audio.wav         # 测试用音频文件
├── fft_comparison.png     # 手写 FFT vs numpy.fft 对比图
├── test_audio_spectrum.png# 测试音频频谱图
├── 信号与系统课设报告.docx # 课程设计报告
└── requirements.txt       # Python 依赖清单
```

---

## 快速开始

### 环境要求

- Python 3.8+
- Windows / macOS / Linux

### 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：若要播放 MP4/M4A/MP3 格式，还需安装 [ffmpeg](https://ffmpeg.org/download.html) 并将其加入系统 PATH。

### 运行 GUI 频谱分析仪

```bash
python gui.py
```

### 运行测试

```bash
# FFT 模块单元测试
python test_fft.py

# 端到端流水线测试
python test_end_to_end.py
```

```bash
# 也可直接运行 FFT 模块内置验证
python fft_module.py
```

---

## 使用说明

| 操作 | 说明 |
|------|------|
| **文件模式** | 点击「加载音频」选择文件 → 点击「播放」开始分析 |
| **麦克风模式** | 点击「开始录音」→ 实时采集并分析 → 点击「停止录音」 |
| **切换对比** | 下拉菜单选择「仅手写FFT」「分图对比」「叠图对比」 |
| **对数幅度** | 勾选「对数幅度(dB)」以 dB 尺度查看频谱 |
| **对数频率** | 勾选「对数频率轴」以对数坐标显示频率轴 |
| **窗函数** | 下拉菜单切换 Hamming 窗 / 矩形窗 |
| **采样率** | 支持 8000 / 11025 / 16000 / 22050 / 44100 / 48000 Hz |

---

## 手写 FFT 算法原理

### 算法：Cooley-Tukey 基-2 DIT-FFT（迭代实现）

**1. 补零至 2 的整数次幂**

FFT 算法要求输入长度为 2 的整数次幂。对于不满足的输入，自动在末尾补零。补零不改变频谱形状，但增加了频域采样密度（平滑显示）。

**2. 位反转重排 (Bit-Reversal Permutation)**

DIT-FFT 要求输入为位反转顺序，输出为自然顺序。例如 N=8 时，索引 3 (011₂) 与索引 6 (110₂) 交换。

**3. 多级蝶形运算**

进行 log₂(N) 级蝶形运算，每级合并相邻的短 DFT 为更长 DFT：

```
G[k] = 上半部分 m 点 DFT 结果
H[k] = 下半部分 m 点 DFT 结果

蝶形公式：
  X[k]     = G[k] + W_m^k · H[k]
  X[k+m]   = G[k] - W_m^k · H[k]

其中 W_m^k = exp(-j·2πk/m) 为旋转因子
```

**4. IFFT 逆变换**

利用 DFT 共轭对称性实现：`ifft(X) = conj(fft(conj(X))) / N`

### 计算复杂度

- 手写 FFT：**O(N log N)**
- 直接 DFT（按定义）：O(N²)
- 当 N=2048 时，FFT 加速比约为 **200 倍以上**

---

## 单元测试项

| 测试 | 内容 | 验证标准 |
|------|------|----------|
| 测试 1 | 2 的整数次幂长度输入 (N=2~4096) | 与 numpy.fft 误差 < 1e-10 |
| 测试 2 | 非 2 的整数次幂长度输入（自动补零） | 补零后与 numpy.fft 误差 < 1e-10 |
| 测试 3 | IFFT 可逆性 (FFT → IFFT = 原信号) | 恢复误差 < 1e-10 |
| 测试 4 | 幅度谱峰值检测准确性 | 峰值频率偏差 < 5 Hz |
| 测试 5 | 复杂度定性对比（FFT vs 直接 DFT） | 加速比随 N 增大显著增长 |

---

## 依赖

| 包 | 用途 |
|----|------|
| `numpy` | 数值计算（验证对比、窗函数生成） |
| `PyQt5` | GUI 框架 |
| `pyqtgraph` | 高性能实时绘图 |
| `sounddevice` | 音频播放与麦克风采集 |
| `soundfile` | 音频文件读写 (WAV/FLAC/AIFF) |
| `scipy` | 信号处理（抗混叠重采样） |
| `pydub` | MP3/MP4/M4A 解码（可选，需 ffmpeg） |
| `matplotlib` | 测试脚本中的静态图表生成 |

---

## 许可证

MIT License — 详见 LICENSE 文件。
