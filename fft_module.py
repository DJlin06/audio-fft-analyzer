"""
FFT Core Module
基于Cooley-Tukey算法的基2-按时间抽取(DIT) FFT手写实现。
严禁在核心算法中调用 numpy.fft.fft 或 scipy.fft 等现成FFT函数。
"""

import numpy as np


def _is_power_of_two(n: int) -> bool:
    """判断整数是否为2的整数次幂。"""
    return n > 0 and (n & (n - 1)) == 0


def _next_power_of_two(n: int) -> int:
    """计算不小于n的最小2的整数次幂。"""
    if n <= 0:
        return 1
    if _is_power_of_two(n):
        return n
    return 1 << (n - 1).bit_length()


def _bit_reverse(k: int, bits: int) -> int:
    """
    计算k的bits位二进制位反转结果。
    例：bits=4, k=3 (0011) -> 12 (1100)
    """
    rev = 0
    for _ in range(bits):
        rev = (rev << 1) | (k & 1)
        k >>= 1
    return rev


def _zero_pad_to_power_of_two(x: np.ndarray) -> np.ndarray:
    """
    将输入序列补零(Zero-padding)至2的整数次幂。
    补零不改变信号的频谱形状，但增加了频谱采样点数，使频谱显示更平滑。
    """
    n = len(x)
    n_padded = _next_power_of_two(n)
    if n_padded == n:
        return x
    padded = np.zeros(n_padded, dtype=x.dtype)
    padded[:n] = x
    return padded


def fft(x: np.ndarray) -> np.ndarray:
    """
    基于Cooley-Tukey算法的基2-按时间抽取(DIT) FFT迭代实现。

    算法原理：
    1. 将输入序列按二进制位反转重排。
    2. 进行 log2(N) 级蝶形运算。
    3. 每一级的蝶形运算合并相邻的短DFT结果为更长的DFT。

    蝶形运算公式（对于m点DFT合并为2m点DFT）：
        X[k]       = G[k] + W_{2m}^k * H[k]
        X[k + m]   = G[k] - W_{2m}^k * H[k]
    其中 G[k]、H[k] 为上下两半的m点DFT，W_{2m}^k = exp(-j*2π*k/(2m))。

    参数:
        x: 输入序列，长度自动补零至最近的2的整数次幂。
    返回:
        X: 复数频谱，长度N（补零后长度）。
    """
    # 补零至2的整数次幂，这是FFT算法能够基2分解的前提
    x = _zero_pad_to_power_of_two(np.asarray(x)) 
    N = len(x)

    if N == 1:
        return x.copy()

    # 步骤1：位反转重排 (Bit-Reversal Permutation)
    # Cooley-Tukey DIT-FFT要求输入为位反转顺序，输出为自然顺序
    bits = N.bit_length() - 1
    X = np.zeros(N, dtype=np.complex128)
    for i in range(N):
        j = _bit_reverse(i, bits)
        X[j] = complex(x[i])

    # 步骤2：多级蝶形运算 (Butterfly Operations)
    # stage 表示当前合并的DFT长度：2, 4, 8, ..., N
    for stage in range(1, bits + 1):
        m = 1 << stage           # 当前级DFT长度 m = 2^stage
        half_m = m >> 1          # 半长
        # 旋转因子 W_m^k = exp(-j * 2π * k / m), k=0,1,...,half_m-1
        # 利用欧拉公式：exp(-jθ) = cos(θ) - j*sin(θ)
        W_m = np.exp(-2j * np.pi / m)

        # 每组蝶形运算处理 m 个点，共 N/m 组
        for k in range(0, N, m):
            W = 1.0 + 0.0j       # W_m^0 = 1
            for j in range(half_m):
                # 蝶形对：idx_top (上端), idx_bottom (下端)
                idx_top = k + j
                idx_bottom = idx_top + half_m

                top = X[idx_top]
                bottom = X[idx_bottom] * W

                # 蝶形运算核心：加与减
                X[idx_top] = top + bottom
                X[idx_bottom] = top - bottom

                # 更新旋转因子：W = W * W_m (即 W_m^{j+1})
                W *= W_m

    return X


def ifft(X: np.ndarray) -> np.ndarray:
    """
    基于FFT的逆变换(IFFT)实现。
    利用性质：ifft(X) = conj(fft(conj(X))) / N
    """
    X = np.asarray(X)
    N = len(X)
    # 共轭 -> FFT -> 共轭 -> 除以N
    x = fft(np.conj(X))
    return np.conj(x) / N


def amplitude_spectrum(x: np.ndarray) -> np.ndarray:
    """
    计算归一化幅度谱（单面谱，仅正频率部分）。
    对于实信号，频谱具有共轭对称性，因此只取前 N//2 + 1 个点。
    幅度乘以 2/N 进行归一化（除直流和奈奎斯特分量外）。
    """
    X = fft(x)
    N = len(X)
    # 取前半部分（含直流）
    half = N // 2 + 1
    amp = np.abs(X[:half])
    # 归一化（补偿单面谱能量）
    amp[1:-1] *= 2.0 / N
    amp[0] /= N
    if N % 2 == 0:
        amp[-1] /= N
    return amp


def power_spectrum(x: np.ndarray) -> np.ndarray:
    """计算功率谱密度估计（幅度谱平方）。"""
    return amplitude_spectrum(x) ** 2


def verify_fft(max_n: int = 12, tol: float = 1e-10) -> bool:
    """
    单元测试：使用正弦波验证自制FFT与numpy.fft.fft的一致性。
    对比误差应在数值精度范围内（默认 < 1e-10）。
    """
    import math
    all_pass = True
    for bits in range(1, max_n + 1):
        N = 1 << bits
        t = np.linspace(0, 1, N, endpoint=False)
        # 构造多频正弦波叠加信号
        x = np.sin(2 * np.pi * 3 * t) + 0.5 * np.cos(2 * np.pi * 7 * t)

        X_mine = fft(x)
        X_np = np.fft.fft(x, n=N)  # 仅用于验证对比，非核心算法调用

        err = np.max(np.abs(X_mine - X_np))
        status = "PASS" if err < tol else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"N={N:5d}: max_err={err:.3e} [{status}]")
    return all_pass


if __name__ == "__main__":
    print("=" * 50)
    print("FFT模块单元测试：对比自制FFT与numpy.fft.fft")
    print("=" * 50)
    ok = verify_fft(max_n=12)
    print("=" * 50)
    print("全部通过" if ok else "存在失败项")
