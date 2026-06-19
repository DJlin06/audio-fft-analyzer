"""
端到端测试脚本
生成测试音频 -> 读取 -> 分帧加窗 -> 手写FFT -> 输出频谱图
"""

import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")  # 无头环境
import matplotlib.pyplot as plt

# Windows 下设置中文字体，避免中文标签显示为方块
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

import fft_module
import audio_io


def generate_test_wav(path: str, fs: int = 8000, duration: float = 2.0):
    """生成包含 500Hz + 1500Hz 正弦波的测试WAV文件。"""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 双频信号，幅度分别为 0.8 和 0.4
    signal = 0.8 * np.sin(2 * np.pi * 500 * t) + 0.4 * np.sin(2 * np.pi * 1500 * t)
    # 加汉宁窗形包络避免咔嗒声（可选）
    signal = signal * 0.9  # 留一定裕量
    # 转 int16
    data = (signal * 32767).astype(np.int16)
    wavfile.write(path, fs, data)
    print(f"已生成测试音频: {path} | fs={fs}Hz | duration={duration}s")


def test_pipeline(wav_path: str):
    """完整处理流水线测试。"""
    print("\n=== 端到端流水线测试 ===")

    # 1. 读取音频
    fs, signal = audio_io.read_audio(wav_path, target_fs=8000)
    print(f"读取音频: fs={fs}Hz, 点数={len(signal)}, 时长={len(signal)/fs:.2f}s")

    # 2. 频域分辨率
    n_fft = 1024
    delta_f = audio_io.frequency_resolution(fs, n_fft)
    print(f"频域分辨率 Δf = fs/N = {fs}/{n_fft} = {delta_f:.2f} Hz")
    print("物理意义: 相邻两条频谱线间隔为 {:.2f} Hz，".format(delta_f) +
          "能分辨的最小频率差约为此值。增大N可提高频率分辨率，但时间分辨率下降。")

    # 3. 分帧加窗
    frame_size = 1024
    hop_size = 256
    window = audio_io.get_window("hamming", frame_size)
    frames = audio_io.frame_signal(signal, frame_size, hop_size, window)
    print(f"分帧结果: 共{len(frames)}帧, 帧长={frame_size}, 帧移={hop_size}, "
          f"重叠率={100*(1-hop_size/frame_size):.0f}%")

    # 4. 取中间一帧进行FFT分析
    mid_frame = frames[len(frames) // 2]
    X = fft_module.fft(mid_frame)
    amp = np.abs(X[:n_fft // 2 + 1])
    amp[1:-1] *= 2.0 / n_fft
    amp[0] /= n_fft
    amp[-1] /= n_fft

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)

    # 5. 与 numpy.fft.fft 对比（仅验证）
    X_np = np.fft.fft(mid_frame, n=n_fft)
    amp_np = np.abs(X_np[:n_fft // 2 + 1])
    amp_np[1:-1] *= 2.0 / n_fft
    amp_np[0] /= n_fft
    amp_np[-1] /= n_fft
    err = np.max(np.abs(amp - amp_np))
    print(f"幅度谱与numpy对比: max_err={err:.3e}")

    # 6. 绘制并保存频谱图
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # 时域波形
    t = np.arange(len(mid_frame)) / fs
    axes[0].plot(t, mid_frame)
    axes[0].set_title("时域波形（单帧，Hamming窗）")
    axes[0].set_xlabel("时间 [s]")
    axes[0].set_ylabel("幅度")
    axes[0].grid(True)

    # 频域幅度谱
    axes[1].plot(freqs, amp, label="手写FFT", color="blue")
    axes[1].plot(freqs, amp_np, label="numpy.fft", linestyle="--", color="red", alpha=0.7)
    axes[1].set_title("频域幅度谱（单面谱，归一化）")
    axes[1].set_xlabel("频率 [Hz]")
    axes[1].set_ylabel("幅度")
    axes[1].legend()
    axes[1].grid(True)
    axes[1].set_xlim(0, 2500)

    plt.tight_layout()
    out_png = wav_path.replace(".wav", "_spectrum.png")
    plt.savefig(out_png, dpi=150)
    print(f"频谱图已保存: {out_png}")
    plt.close()

    # 7. 检测峰值频率（应接近 500Hz 和 1500Hz）
    peaks = [] 
    for target in [500, 1500]:
        idx = np.argmin(np.abs(freqs - target))
        # 在附近小范围搜索最大值
        search_range = slice(max(0, idx - 5), min(len(freqs), idx + 6))
        peak_idx = search_range.start + np.argmax(amp[search_range])
        peaks.append((freqs[peak_idx], amp[peak_idx]))
    print(f"检测到的峰值频率: {peaks[0][0]:.1f}Hz (幅度={peaks[0][1]:.3f}), "
          f"{peaks[1][0]:.1f}Hz (幅度={peaks[1][1]:.3f})")


if __name__ == "__main__":
    wav_path = "test_audio.wav"
    generate_test_wav(wav_path)
    test_pipeline(wav_path)
    print("\n=== 测试完成 ===")
