'''
调⽤归⼀化、分帧等函数，继续实现：
短时能量求取及画图分析
语谱图的绘制及分析
'''
# 短时能量求取及画图分析 & 语谱图的绘制及分析
import numpy as np
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal
from plot_frames import preemphasis
from enframe import enframe
import wave

# 短时能量求取及画图分析
# 加载⾳频⽂件
filename = 'sample.wav'
y, sr = librosa.load(filename) # 将⾳频信号y和采样率sr分别赋值
# 预加重
y_pre = preemphasis(y)
# 设置分帧的参数，其中窗⻓为25ms，帧移为10ms
win_len = int(sr * 0.025) # 窗⻓，25ms
inc = int(sr * 0.01) # 帧移，10ms
win = np.hamming(win_len) # 加窗函数
# 使⽤YujiazhongFenzhenWin中⾃⼰编写的分帧函数enframe将信号分帧，并加窗
frames = enframe(y_pre, win, inc)
# 计算每⼀帧的短时能量
energy = np.sum(frames ** 2, axis=1)
# 画出短时能量图
plt.subplot(2, 1, 1) # 创建⼀个带有两个⼦图的画布，第⼀个⼦图
plt.plot(energy, color='blue') # 绘制能量曲线
plt.ylabel('Energy') # 设置y轴标签
plt.xlabel('Frame Index') # 设置x轴标签
plt.title('Short-time Energy') # 设置图标题
# 语谱图的绘制及分析
# 打开⾳频⽂件，获取⾳频⽂件参数
f = wave.open(r"sample.wav", "rb")
params = f.getparams() # 获取⾳频⽂件的参数
nchannels, sampwidth, framerate, nframes = params[:4] # 声道数、每个样本点的字节数、采样率和样本点数
# 读取⾳频⽂件数据并进⾏归⼀化处理
str_data = f.readframes(nframes) # 读取⾳频数据，返回⼀个字节字符串
wave_data = np.fromstring(str_data, dtype=np.short)
wave_data = wave_data*1.0/(max(abs(wave_data))) # 归⼀化处理
plt.subplot(2, 1, 2)
plt.specgram(wave_data,Fs = framerate, scale_by_freq = True, sides = 'default')
'''
其中 wave_data 为归⼀化后的⾳频数据，
Fs 为采样率，scale_by_freq 表⽰是否按⽐例缩放频谱，
sides 为频谱的取值范围，
'default' 表⽰频谱的左右两边均绘制
'''
plt.ylabel('Frequency (Hz)') # 设置y轴标签
plt.xlabel('Time (s)') # 设置x轴标签
plt.title('Spectrogram') # 设置图标题
plt.tight_layout()
plt.show() # 显⽰图像