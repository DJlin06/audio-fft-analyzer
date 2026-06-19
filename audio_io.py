"""
Audio I/O Module
音频读取、分帧、加窗、重采样处理模块。
支持格式：WAV/FLAC/AIFF（通过 soundfile）以及 MP4/M4A/MP3（通过 pydub + ffmpeg）。
"""

import os
import subprocess
import numpy as np
from scipy import signal

# 优先使用 soundfile（支持任意位深度和浮点格式）
import soundfile as sf

# pydub 用于处理 ffmpeg 格式（MP4 等）
try:
    from pydub import AudioSegment
    _HAS_PYDUB = True
except Exception:
    _HAS_PYDUB = False


def _has_ffmpeg() -> bool:
    """检测系统是否安装了 ffmpeg（pydub 解码 MP4 必需）。"""
    if not _HAS_PYDUB:
        return False
    try:
        # creationflags=0x08000000 避免 Windows 下弹出命令行黑框
        subprocess.check_output(
            ["ffmpeg", "-version"],
            stderr=subprocess.STDOUT,
            creationflags=0x08000000,
        )
        return True
    except Exception:
        return False


_HASS_FFMPEG = _has_ffmpeg()


def read_audio(path: str, target_fs: int = 8000) -> tuple:
    """
    通用音频读取接口，自动识别格式并统一转换为 float32 单声道信号。

    支持格式：
    - 原生支持（libsndfile）：WAV, FLAC, AIFF, OGG 等
    - ffmpeg 转码支持（需系统安装 ffmpeg）：MP4, M4A, MP3 等

    参数:
        path: 音频文件路径。
        target_fs: 目标采样率，默认 8000 Hz。
    返回:
        (fs, data): 实际采样率（重采样后等于 target_fs）与 float32 单声道信号。
    抛出:
        RuntimeError: 当文件无法读取且 ffmpeg 未安装时给出友好提示。
    """
    # 1. 尝试 soundfile 直接读取（支持 WAV/FLAC/AIFF/OGG，含 24-bit / float32）
    try:
        data, fs = sf.read(path, dtype="float32")
    except Exception as sf_err:
        # 2. soundfile 失败，尝试 pydub + ffmpeg（MP4 / M4A / MP3）
        if not _HASS_FFMPEG:
            raise RuntimeError(
                f"无法读取文件 '{path}'。\n"
                f"soundfile 报错: {sf_err}\n"
                f"系统未检测到 ffmpeg，因此无法解码 MP4/M4A/MP3 等格式。\n"
                f"请安装 ffmpeg 并确保其加入系统 PATH。"
            ) from sf_err
        data, fs = _read_via_pydub(path)

    # 统一转单声道：默认使用第一声道，避免左右声道反相抵消导致数据为空
    if data.ndim > 1:
        data = data[:, 0]

    # 重采样至 target_fs（带抗混叠滤波）
    if fs != target_fs:
        data = _resample_anti_alias(data, fs, target_fs)
        fs = target_fs

    return fs, data


def _read_via_pydub(path: str) -> tuple:
    """使用 pydub + ffmpeg 读取音频，返回 (data, fs)。"""
    audio = AudioSegment.from_file(path)
    fs = audio.frame_rate
    # 转换为单声道
    audio = audio.set_channels(1)
    # 获取原始样本数组
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    # pydub 的 sample_width 以字节为单位：2 对应 16-bit，4 对应 32-bit
    max_val = float(1 << (audio.sample_width * 8 - 1))
    data = samples / max_val
    return data, fs


def _resample_anti_alias(data: np.ndarray, orig_fs: int, target_fs: int) -> np.ndarray:
    """
    使用 scipy.signal.resample_poly 进行多相滤波重采样。
    相比简单的 np.interp，resample_poly 自带抗混叠低通滤波，
    能有效避免高频分量在降采样时混叠到低频。
    """
    if orig_fs == target_fs:
        return data
    g = np.gcd(orig_fs, target_fs)
    up = target_fs // g
    down = orig_fs // g
    return signal.resample_poly(data, up, down)


def hamming_window(n: int) -> np.ndarray:
    """
    汉明窗 (Hamming Window)。
    w[n] = 0.54 - 0.46 * cos(2π * n / (N-1)), n = 0,1,...,N-1
    """
    if n == 1:
        return np.ones(1)
    n_arr = np.arange(n)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n_arr / (n - 1))


def rectangular_window(n: int) -> np.ndarray:
    """矩形窗（无窗，相当于直接截断）。"""
    return np.ones(n, dtype=np.float32)


WINDOW_MAP = {
    "hamming": hamming_window,
    "rectangular": rectangular_window,
}


def get_window(name: str, n: int) -> np.ndarray:
    """根据名称获取窗函数。"""
    if name not in WINDOW_MAP:
        raise ValueError(f"不支持的窗类型: {name}，可选: {list(WINDOW_MAP.keys())}")
    return WINDOW_MAP[name](n)


def frame_signal(
    signal: np.ndarray,
    frame_size: int,
    hop_size: int,
    window: np.ndarray = None,
) -> list:
    """
    对信号进行分帧处理，可选加窗。

    参数:
        signal: 输入时域信号。
        frame_size: 每帧样本数。
        hop_size: 帧移（相邻两帧起点之间的距离）。
        window: 窗函数向量，长度需等于 frame_size。为 None 时不加窗。
    返回:
        frames: 帧列表，每个元素为一帧信号（已乘窗）。
    """
    if window is not None and len(window) != frame_size:
        raise ValueError("窗长度必须与帧长相等")

    frames = []
    n = len(signal)
    start = 0
    while start + frame_size <= n:
        frame = signal[start : start + frame_size].copy()
        if window is not None:
            frame *= window
        frames.append(frame)
        start += hop_size

    # 若最后一帧不足，补零处理
    if start < n:
        frame = np.zeros(frame_size, dtype=signal.dtype)
        remain = n - start
        frame[:remain] = signal[start:]
        if window is not None:
            frame *= window
        frames.append(frame)

    return frames


def frequency_resolution(fs: int, n_fft: int) -> float:
    """
    计算频域分辨率 Δf = fs / N。

    物理意义:
    Δf 表示相邻两条频谱线之间的频率间隔，决定了频谱分析仪能够分辨的
    最小频率差异。N 越大（或 fs 越小），Δf 越小，频率分辨能力越强。
    但时间分辨率会相应降低（时频测不准原理）。
    """
    return fs / n_fft
