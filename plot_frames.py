'''
实验验证：⾃⼰录制1秒语⾳：根握分帧的语⾳，输⼈帧号(或者⾃⼰选定⼀个帧号），绘制
连续两帧语⾳信号
•实验验证提⽰：
•⾃⾏设计窗⻓和帧移的点数
•读取wav⽂件scipy.io. wavfile.read(）
•⾳频需要归⼀化：最⼤值：np.absolute(wave data).max0
•画第帧的输⼊提⽰：i=input('please input first frame number(i)")
实例：录制语⾳sample.wav
'''

from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from enframe import enframe

# 读取⾳频⽂件
filename = 'sample.wav'
sample_rate, signal = wavfile.read(filename)
# 归⼀化
signal = signal / np.abs(signal).max()
# 设计分帧的参数
frame_length = int(sample_rate * 0.025) # 25ms
frame_step = int(sample_rate * 0.01) # 10ms
w_rect = np.ones(frame_length) # 矩形窗
w_hann = np.hanning(frame_length) # 汉宁窗
w_hamm = np.hamming(frame_length) # 汉明窗
# 分帧，可选择不同的加窗⽅式
frames = enframe(signal, w_hann, frame_step)
# 打印帧数和每帧的⻓度
print(f"Number of frames: {frames.shape[0]}")
print(f"Length of each frame: {frames.shape[1]}")
# 选择某帧和后⼀帧，并绘制波形图
frame_idx = int(input('please input first frame number(i): '))
'''
在绘制连续两帧语⾳信号的波形图时，可以根据⾃⼰的需要选择要显⽰的帧号，
即输⼊ i 的值，尝试不同的帧号，以观察连续两帧语⾳信号的差异。
'''
plt.figure(figsize=(12, 3))
# frame_idx帧的波形图，frame_idx表⽰帧号，frame_step表⽰帧移，frame_length表⽰帧⻓
plt.plot(np.arange(frame_idx*frame_step, frame_idx*frame_step+frame_length), frames[frame_idx])
'''
np.arange()⽣成了表⽰时间轴的数组，
该数组的起始值是frame_idx*frame_step，步⻓是1，⻓度是frame_length.
frames[frame_idx]表⽰第frame_idx帧的语⾳信号。
'''
# frame_idx+1帧的波形图
plt.plot(np.arange((frame_idx+1)*frame_step, (frame_idx+1)*frame_step+frame_length), frames[frame_idx+1]) # 与上同理
plt.show()