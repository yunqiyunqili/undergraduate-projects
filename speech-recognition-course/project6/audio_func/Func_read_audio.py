import scipy.io as wavfile
import numpy as np
from scipy.io import wavfile

def read_audio(wave_path):

    framerate, wave_data = wavfile.read(wave_path)  # 读入音频文件
    nframes = len(wave_data)  # 采样点数
    wave_data = wave_data / np.max(wave_data)  # 音频波形数据
    wave_data_reserve = wave_data[::-1]  # 波形倒序

    # wave_data.shape = (-1, 2)   # -1的意思就是没有指定,根据另一个维度的数量进行分割
    return framerate,wave_data,nframes

