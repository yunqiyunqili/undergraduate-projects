# 短时能量求取及画图分析  &  语谱图的绘制及分析
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
import wave

from YujiazhongFenzhenWin import preemphasis,enframe


# 短时能量求取及画图分析
# 加载音频文件
filename = 'sample.wav'
y, sr = librosa.load(filename)  # 将音频信号y和采样率sr分别赋值
# 预加重
y_pre = preemphasis(y)
# 设置分帧的参数，其中窗长为25ms，帧移为10ms
win_len = int(sr * 0.025)  # 窗长，25ms
inc = int(sr * 0.01)  # 帧移，10ms
win = np.hamming(win_len)  # 加窗函数
# 使用YujiazhongFenzhenWin中自己编写的分帧函数enframe将信号分帧，并加窗
frames = enframe(y_pre, win, inc)
# 计算每一帧的短时能量
energy = np.sum(frames ** 2, axis=1)
# 画出短时能量图
plt.subplot(2, 1, 1)  # 创建一个带有两个子图的画布，第一个子图
plt.plot(energy, color='blue')  # 绘制能量曲线
plt.ylabel('Energy')  # 设置y轴标签
plt.xlabel('Frame Index')  # 设置x轴标签
plt.title('Short-time Energy')  # 设置图标题


# 语谱图的绘制及分析
# 打开音频文件，获取音频文件参数
f = wave.open(r"sample.wav", "rb")
params = f.getparams()  # 获取音频文件的参数
nchannels, sampwidth, framerate, nframes = params[:4]  # 声道数、每个样本点的字节数、采样率和样本点数

# 读取音频文件数据并进行归一化处理
str_data = f.readframes(nframes)  # 读取音频数据，返回一个字节字符串
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data = wave_data*1.0/(max(abs(wave_data)))  # 归一化处理

plt.subplot(2, 1, 2)
plt.specgram(wave_data,Fs = framerate, scale_by_freq = True, sides = 'default')
'''
其中 wave_data 为归一化后的音频数据，
Fs 为采样率，scale_by_freq 表示是否按比例缩放频谱，
sides 为频谱的取值范围，
'default' 表示频谱的左右两边均绘制
'''
plt.ylabel('Frequency (Hz)')  # 设置y轴标签
plt.xlabel('Time (s)')  # 设置x轴标签
plt.title('Spectrogram')  # 设置图标题
plt.tight_layout()
plt.show()  # 显示图像
