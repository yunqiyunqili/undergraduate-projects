import wave
import struct

def write_audio(framerate,nframes,waveTest,wave_new_path):


    # 打开WAV音频用来写操作
    f = wave.open(wave_new_path, "w")

    #f.setnchannels(1)           # 配置声道数
    #f.setsampwidth(2)           # 配置量化位数
    #f.setframerate(framerate)   # 配置取样频率
    comptype = "NONE" #comptype压缩类型，
    compname = "not compressed" #compname压缩类型描述

    # 也可以用setparams一次性配置所有参数
    f.setparams((1, 2, framerate, nframes,comptype, compname))
    #struct.pack()将采样数据打包成字节数据，writeframes()将字节数据写入波形
    for s in waveTest:
        f.writeframes(struct.pack('h',int(s * 1000))) #1000个采样点
    f.close()