import matplotlib.pyplot as plt
import numpy as np
from Func_play_audio import play_audio #导入播放音频函数
from Func_read_audio import read_audio #导入读取音频函数
from Func_write_audio import write_audio #导入写入音频函数


(framerate,wave_data,nframes) = read_audio("../F215.wav") #读取音频"F215.wav"的framerate采样频率,wave_data频率扫描波,nframes采样点数
time = np.arange(0, nframes) * (1.0 / framerate) # 持续时间

plt.figure()
plt.subplot(3, 1, 1)
plt.plot(time, wave_data[:, 0])#绘制F215.wav波形图
plt.xlabel("time/s",fontsize=14)
plt.ylabel("amplitude",fontsize=14) #幅度
plt.title("waveform",fontsize=14)
plt.grid()  # 标尺

wave_data_rev=wave_data[::-1] #F215.wav逆序
plt.subplot(3, 1, 2)
plt.plot(time, wave_data_rev[:, 0])#绘制F215.wav逆序波形图
plt.xlabel("time/s",fontsize=14)
plt.ylabel("amplitude",fontsize=14)
plt.title("waveform_rev",fontsize=14)
plt.grid()  # 标尺

write_audio(framerate, nframes, wave_data_rev, '../wave_data_rev.wav') #写入F215.wav逆序
play_audio('../F215.wav') #播放F215.wav
play_audio('../wave_data_rev.wav') #播放F215.wav逆序

#正弦函数 noise=0.2*np.sin(2*np.pi*f1*t)其中f1=440Hz，波形长度同F215.wav
noise=[0.2*np.sin(2*np.pi*440*x/framerate)for x in range (nframes)]
write_audio(44000, nframes, noise, '../noise.wav') #写入noise.wav
play_audio('../noise.wav') #播放noise.wav


plt.subplot(3, 1, 3)
plt.plot(time,noise) #绘制noise.wav波形图
plt.xlabel("time/s",fontsize=14)
plt.ylabel("amplitude",fontsize=14)
plt.title("waveform_sin",fontsize=14)
plt.grid()  # 标尺

plt.tight_layout()  # 紧密布局
plt.show()




