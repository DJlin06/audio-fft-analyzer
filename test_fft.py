"""
FFT 模块独立单元测试

使用正弦波、多频合成信号、非 2 的整数次幂长度输入验证手写 FFT 正确性。
对比 numpy.fft.fft 误差，测试 IFFT 可逆性，验证幅度谱峰值检测。
"""

import numpy as np
import fft_module


def test_power_of_two():
    """测试1: 标准 2 的整数次幂长度输入。"""
    print("=" * 60)
    print("测试 1: 2 的整数次幂长度输入")
    print("=" * 60)
    all_pass = True
    for bits in range(1, 13):
        N = 1 << bits
        t = np.linspace(0, 1, N, endpoint=False)
        # 多频合成信号：3 Hz 正弦 + 7 Hz 余弦
        x = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)

        X_mine = fft_module.fft(x)
        X_np = np.fft.fft(x, n=N)

        err = np.max(np.abs(X_mine - X_np))
        status = "PASS" if err < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  N={N:5d}: max_err={err:.3e} [{status}]")
    return all_pass


def test_non_power_of_two():
    """测试2: 非 2 的整数次幂长度输入（验证自动补零）。"""
    print("\n" + "=" * 60)
    print("测试 2: 非 2 的整数次幂长度输入（自动补零）")
    print("=" * 60)
    all_pass = True
    test_lengths = [50, 100, 300, 500, 1000, 1500, 2000]
    for N in test_lengths:
        t = np.linspace(0, 1, N, endpoint=False)
        x = np.sin(2 * np.pi * 5 * t) + 0.3 * np.cos(2 * np.pi * 13 * t)

        X_mine = fft_module.fft(x)
        N_padded = fft_module._next_power_of_two(N)
        X_np = np.fft.fft(x, n=N_padded)

        err = np.max(np.abs(X_mine - X_np))
        status = "PASS" if err < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  N={N:4d} -> padded={N_padded:4d}: max_err={err:.3e} [{status}]")
    return all_pass


def test_ifft():
    """测试3: IFFT 可逆性（FFT -> IFFT 应恢复原信号）。"""
    print("\n" + "=" * 60)
    print("测试 3: IFFT 可逆性")
    print("=" * 60)
    all_pass = True
    for bits in range(1, 13):
        N = 1 << bits
        t = np.linspace(0, 1, N, endpoint=False)
        x = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)

        X = fft_module.fft(x)
        x_recovered = fft_module.ifft(X)

        err = np.max(np.abs(x - x_recovered))
        status = "PASS" if err < 1e-10 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  N={N:5d}: max_err={err:.3e} [{status}]")
    return all_pass


def test_amplitude_spectrum():
    """测试4: 幅度谱峰值检测准确性。"""
    print("\n" + "=" * 60)
    print("测试 4: 幅度谱峰值检测准确性")
    print("=" * 60)
    fs = 8000
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # 双频信号：500 Hz (0.8) + 1200 Hz (0.4)
    x = 0.8 * np.sin(2 * np.pi * 500 * t) + 0.4 * np.sin(2 * np.pi * 1200 * t)

    amp = fft_module.amplitude_spectrum(x)
    # amplitude_spectrum 内部会补零到 2 的幂，因此频率轴必须基于补零后的长度
    N_fft = (len(amp) - 1) * 2
    freqs = np.fft.rfftfreq(N_fft, d=1.0 / fs)

    # 简单局部峰值检测
    peaks = []
    for i in range(1, len(amp) - 1):
        if amp[i] > amp[i - 1] and amp[i] > amp[i + 1] and amp[i] > 0.01:
            peaks.append((freqs[i], amp[i]))
    peaks.sort(key=lambda p: p[1], reverse=True)
    top2 = peaks[:2]

    print(f"  理论峰值: 500 Hz (幅值≈0.8), 1200 Hz (幅值≈0.4)")
    if len(top2) >= 2:
        print(
            f"  检测结果: {top2[0][0]:.1f} Hz (幅值={top2[0][1]:.3f}), "
            f"{top2[1][0]:.1f} Hz (幅值={top2[1][1]:.3f})"
        )
        err1 = abs(top2[0][0] - 500)
        err2 = abs(top2[1][0] - 1200)
        status = "PASS" if err1 < 5 and err2 < 5 else "FAIL"
        print(f"  频率偏差: {err1:.1f} Hz, {err2:.1f} Hz [{status}]")
    else:
        print("  检测到峰值数不足，可能信号或参数有误。")
        status = "FAIL"
    return status == "PASS"


def test_complexity():
    """测试5: 复杂度定性验证（N 越大，手写 FFT 应明显快于直接 DFT 实现）。"""
    print("\n" + "=" * 60)
    print("测试 5: 复杂度定性对比（手写 FFT vs 直接 DFT）")
    print("=" * 60)
    import time

    def direct_dft(x):
        """直接按定义计算的 O(N^2) DFT，仅用于对比。"""
        N = len(x)
        n = np.arange(N)
        k = n.reshape((N, 1))
        return np.dot(np.exp(-2j * np.pi * k * n / N), x)

    for bits in range(6, 12):
        N = 1 << bits
        x = np.random.randn(N)

        t0 = time.perf_counter()
        _ = fft_module.fft(x)
        t_fft = time.perf_counter() - t0

        t0 = time.perf_counter()
        _ = direct_dft(x)
        t_dft = time.perf_counter() - t0

        ratio = t_dft / t_fft
        print(f"  N={N:5d}: FFT={t_fft*1e3:7.3f} ms, DFT={t_dft*1e3:9.3f} ms, 加速比={ratio:.1f}x")
    print("  注: 随着 N 增大，FFT 应显著快于直接 DFT，体现 O(N log N) 优势。")
    return True


def plot_comparison():
    """生成手写 FFT 与 numpy.fft 对比图，供报告使用。"""
    print("\n" + "=" * 60)
    print("生成对比图: fft_comparison.png")
    print("=" * 60)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Windows 下设置中文字体，避免中文标签显示为方块
        plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:
        print(f"  matplotlib 不可用，跳过绘图: {e}")
        return

    N = 1024
    t = np.linspace(0, 1, N, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.cos(2 * np.pi * 120 * t)

    X_mine = fft_module.fft(x)
    X_np = np.fft.fft(x, n=N)

    freqs = np.fft.fftfreq(N, d=1.0 / N)
    half = N // 2

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # 幅度谱对比
    axes[0].plot(freqs[:half], np.abs(X_mine[:half]), label="手写 FFT", color="blue")
    axes[0].plot(freqs[:half], np.abs(X_np[:half]), label="numpy.fft",
                 linestyle="--", color="red", alpha=0.7)
    axes[0].set_title("幅度谱对比（单面谱）")
    axes[0].set_xlabel("频率 [Hz]")
    axes[0].set_ylabel("幅度")
    axes[0].legend()
    axes[0].grid(True)

    # 误差
    err = np.abs(X_mine - X_np)
    axes[1].plot(freqs[:half], err[:half], color="green")
    axes[1].set_title(f"绝对误差 (max={np.max(err):.3e})")
    axes[1].set_xlabel("频率 [Hz]")
    axes[1].set_ylabel("误差")
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("fft_comparison.png", dpi=150)
    print("  已保存: fft_comparison.png")


if __name__ == "__main__":
    results = []
    results.append(test_power_of_two())
    results.append(test_non_power_of_two())
    results.append(test_ifft())
    results.append(test_amplitude_spectrum())
    results.append(test_complexity())
    plot_comparison()

    print("\n" + "=" * 60)
    if all(results):
        print("全部测试通过 [PASS]")
    else:
        print("存在失败项 [FAIL]")
    print("=" * 60)
