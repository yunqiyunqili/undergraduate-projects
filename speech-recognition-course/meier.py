import librosa
import numpy as np
import matplotlib.pyplot as plt

# 读取音频文件
audio_path = 'sample.wav'
y, sr = librosa.load(audio_path)  # y：音频信号的时域数据，sr：采样率

# 计算梅尔滤波器组数
n_fft = 2048  # FFT点数，即音频信号被分为多少个FFT窗口
n_mels = 128  # 梅尔滤波器组数
fmin = 0  # 最小频率值
fmax = sr // 2  # 最大频率值（设为采样率的一半，表示不考虑超过采样率一半的频率）
mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)

# 计算功率谱
S = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2)  # 对音频信号进行短时傅里叶变换，得到每个时刻的频谱信息S
power = np.abs(S) ** 2  # 每个时刻的功率谱

# 计算梅尔倒谱系数
n_mfcc = 20
mfcc = librosa.feature.mfcc(S=librosa.power_to_db(power), n_mfcc=n_mfcc, mel_basis=mel_basis)
'''
librosa.power_to_db()：将功率谱转换为分贝尺度的值，
n_mfcc：需要计算的MFCC个数，
mel_basis是上一步计算得到的梅尔滤波器矩阵。
'''

print(mfcc)


# 可视化梅尔倒谱系数
plt.figure(figsize=(10, 4))
librosa.display.specshow(mfcc, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()