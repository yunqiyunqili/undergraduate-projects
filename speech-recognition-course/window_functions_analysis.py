# 完成32点输出的矩形窗、汉宁窗、汉明窗时频域图，并对三个窗进⾏分析。

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft

# 定义窗
N = 32 # 32点
nn = np.arange(0, N) # 时域横坐标范围
NN = np.arange(-128, 128) # 频域横坐标范围

# 矩形窗
'''
矩形窗是使⽤ np.ones 函数⽣成的。
该函数返回⼀个全为1的数组，其形状与信号 x 相同。表⽰在信号中选取⼀个连续的时间窗⼝，其幅度为1。
'''
w_rect = np.ones(N) # 矩形窗函数在相应序列点时域上的幅度响应
W_rect = fft(w_rect, 256) # 矩形窗函数在相应序列点频域上的幅度响应
plt.subplot(2,1,1)
plt.stem(nn,w_rect)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Rectangular Windowed Signal')
plt.subplot(2,1,2)
plt.plot(NN, abs(np.fft.fftshift(W_rect)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Rectangular Windowed Signal Spectrum')
plt.show()

# 汉宁窗
'''
使⽤ numpy 库中的 hanning 函数⽣成汉宁窗。
'''
w_hann = np.hanning(N) # 汉宁窗函数在相应序列点时域上的幅度响应
W_hann = fft(w_hann, 256) # 汉宁窗函数在相应序列点频域上的幅度响应
plt.subplot(2,1,1)
plt.stem(nn,w_hann)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Hanning Windowed Signal')
plt.subplot(2,1,2)
plt.plot(NN, abs(np.fft.fftshift(W_hann)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Hanning Windowed Signal Spectrum')
plt.show()

# 汉明窗
'''
使⽤ numpy 库中的 hamming 函数⽣成汉明窗。
'''
w_hamm = np.hamming(N) # 汉明窗函数在相应序列点时域上的幅度响应
W_hamm = fft(w_hamm, 256) # 汉明窗函数在相应序列点频域上的幅度响应
plt.subplot(2,1,1)
plt.stem(nn,w_hamm)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Hamming Windowed Signal')
plt.subplot(2,1,2)
plt.plot(NN, abs(np.fft.fftshift(W_hamm)))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Hamming Windowed Signal Spectrum')
plt.show()