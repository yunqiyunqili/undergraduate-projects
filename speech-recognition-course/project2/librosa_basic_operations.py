'''
基础部分：基于librosa完成代码
'''
import matplotlib.pyplot as plt
import numpy as np

# librosa.load() 读取⾳频⽂件
'''
librosa.load()
librosa.load(path, sr=22050, mono=True, offset=0.0, duration=None)
读取⾳频⽂件。默认采样率是22050，如果要保留⾳频的原始采样率，使⽤sr = None。
参数：
    path ：⾳频⽂件的路径。
    sr ：采样率，如果为“None”使⽤⾳频⾃⾝的采样率
    mono ：bool，是否将信号转换为单声道
    offset ：float，在此时间之后开始阅读（以秒为单位）
    持续时间：float，仅加载这么多的⾳频（以秒为单位）
返回：
    y ：⾳频时间序列
    sr ：⾳频的采样率
'''
y, sr = librosa.load('sample.wav')
print(' y ：⾳频时间序列:', y,'sr ：⾳频的采样率',sr)

# librosa.get_duration() 计算时间序列、特征矩阵、⽂件的持续时间（以秒为单位）
'''
librosa.get_duration()
librosa.get_duration(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, center=True, filename=None)
计算时间序列的的持续时间（以秒为单位）
参数：
    y ：⾳频时间序列
    sr ：y的⾳频采样率
    S ：STFT矩阵或任何STFT衍⽣的矩阵（例如，⾊谱图或梅尔频谱图）。根据频谱图输⼊计算的持续时间仅在达到帧分辨率之前才是准确的。如果需要⾼精度，则最好直接使⽤⾳频时间序
    n_fft ：S的 FFT窗⼝⼤⼩
    hop_length ：S列之间的⾳频样本数
    center ：布尔值
        如果为True，则S [:, t]的中⼼为y [t * hop_length]
        如果为False，则S [:, t]从y[t * hop_length]开始
    filename ：如果提供，则所有其他参数都将被忽略，并且持续时间是直接从⾳频⽂件中计算得出的。
返回：
    d：持续时间（以秒为单位）
'''
time = librosa.get_duration(filename='sample.wav')
print('time',time)

# librosa.get_samplerate() 读取采样率
'''
librosa.get_samplerate(path) 读取采样率
参数：
    path ：⾳频⽂件的路径
返回：⾳频⽂件的采样率
'''
sample_rate= librosa.get_samplerate('sample.wav')
print('sample_rate',sample_rate)

# librosa.output.write_wav() 将时间序列输出为.wav⽂件
'''
librosa.output.write_wav()
librosa.output.write_wav(path, y, sr, norm=False)
将时间序列输出为.wav⽂件
参数：
    path：保存输出wav⽂件的路径
    y ：⾳频时间序列。
    sr ：y的采样率
    norm：bool，是否启⽤幅度归⼀化。将数据缩放到[-1，+1]范围。
在0.8.0以后的版本，librosa都会将这个函数删除，推荐⽤下⾯的函数：
import soundfile
soundfile.write(file, data, samplerate)
参数：
    file：保存输出wav⽂件的路径
    data：⾳频数据
    samplerate：采样率
'''
import soundfile
soundfile.write('sample_output.wav', y, sr)

# librosa.feature.zero_crossing_rate() 计算⼀段⾳频序列的过零率
'''
librosa.feature.zero_crossing_rate()
librosa.feature.zero_crossing_rate(y, frame_length = 2048, hop_length = 512, center = True)
计算⾳频时间序列的过零率。
参数：
    y ：⾳频时间序列
    frame_length ：帧⻓
    hop_length ：帧移
    center：bool，如果为True，则通过填充y的边缘来使帧居中。
返回：
    zcr：zcr[0，i]是第i帧中的过零率
'''
zcr = librosa.feature.zero_crossing_rate(y, frame_length= 2048, hop_length= 512, center = True)
print('过零率',zcr)

# librosa.display.waveplot() 绘制波形的幅度包络线
'''
librosa.display.waveplot()
librosa.display.waveplot(y, sr=22050, x_axis='time', offset=0.0, ax=None)
绘制波形的幅度包络线
参数：
    y ：⾳频时间序列
    sr ：y的采样率
    x_axis ：str {'time'，'off'，'none'}或None，如果为“时间”，则在x轴上给定时间刻度线。
    offset：⽔平偏移（以秒为单位）开始波形图
'''
import librosa.display
librosa.display.waveshow(y, sr=sr)
plt.title('Amplitude envelope of the waveform')
plt.show()

# librosa.stft() 短时傅⽴叶变换（STFT），返回⼀个复数矩阵D(F，T)
'''
librosa.stft()
librosa.stft(y, n_fft=2048, hop_length=None, win_length=None, window='hann', center=True, pad_mode='reflect')
短时傅⽴叶变换（STFT），返回⼀个复数矩阵D(F，T)
参数：
    y：⾳频时间序列
    n_fft：FFT窗⼝⼤⼩，n_fft=hop_length+overlapping
    hop_length：帧移，如果未指定，则默认win_length / 4。
    win_length：每⼀帧⾳频都由window（）加窗。窗⻓win_length，然后⽤零填充以匹配N_FFT。默认win_length=n_fft。
    window：字符串，元组，数字，函数 shape =（n_fft, )
        窗⼝（字符串，元组或数字）；
        窗函数，例如scipy.signal.hanning 
        ⻓度为n_fft的向量或数组
    center：bool 
        如果为True，则填充信号y，以使帧 D [:, t]以y [t * hop_length]为中⼼。
        如果为False，则D [:, t]从y [t * hop_length]开始
    dtype：D的复数值类型。默认值为64-bit complex复数
    pad_mode：如果center = True，则在信号的边缘使⽤填充模式。默认情况下，STFT使⽤reflection padding。
返回：
    STFT矩阵，shape =（1 + n_fft/2，t）
'''
stft = librosa.stft(y)
print('stft矩阵',stft)

# librosa.istft() 短时傅⽴叶逆变换（ISTFT），将复数值D(f,t)频谱矩阵转换为时间序列y，窗函数、帧移等参数应与stft相同
'''
librosa.istft()
librosa.istft(stft_matrix, hop_length=None, win_length=None, window='hann', center=True, length=None)
短时傅⽴叶逆变换（ISTFT），将复数值D(f,t)频谱矩阵转换为时间序列y，窗函数、帧移等参数应与stft相同
参数：
    stft_matrix ：经过STFT之后的矩阵
    hop_length ：帧移，默认为
    win_length ：窗⻓，默认为n_fft
    window：字符串，元组，数字，函数或shape = (n_fft, )
        窗⼝（字符串，元组或数字）
        窗函数，例如scipy.signal.hanning 
        ⻓度为n_fft的向量或数组
    center：bool 
        如果为True，则假定D具有居中的帧
        如果False，则假定D具有左对⻬的帧
    length：如果提供，则输出y为零填充或剪裁为精确⻓度⾳频
返回：
    y ：时域信号
'''
istft = librosa.istft(stft)
print('istft矩阵', istft)

# librosa.amplitude_to_db() 将幅度频谱转换为dB标度频谱。也就是对S取对数。与这个函数相反的是librosa.db_to_amplitude(S)
'''
librosa.amplitude_to_db(S, ref=1.0)
将幅度频谱转换为dB标度频谱。也就是对S取对数。与这个函数相反的是librosa.db_to_amplitude(S)
参数：
    S ：输⼊幅度
    ref ：参考值，幅值 abs(S) 相对于ref进⾏缩放，
返回：
    dB为单位的S
'''
S = np.abs(librosa.stft(y))
print('幅度频谱转换为dB标度频谱',librosa.amplitude_to_db(S ** 2))

# librosa.core.power_to_db() 将功率谱(幅值平⽅)转换为dB单位，与这个函数相反的是 librosa.db_to_power(S)
'''
librosa.core.power_to_db(S, ref=1.0)
将功率谱(幅值平⽅)转换为dB单位，与这个函数相反的是 librosa.db_to_power(S)
参数：
    S：输⼊功率
    ref ：参考值，振幅abs(S)相对于ref进⾏缩放，
返回：
    dB为单位的S
'''
S = np.abs(librosa.stft(y))
print('将功率谱(幅值平⽅)转换为dB单位',librosa.power_to_db(S ** 2))

# librosa.display.specshow() 绘制频谱图 
# 绘制功率谱和转换为dB单位的功率谱
'''
librosa.display.specshow(data, x_axis=None, y_axis=None, sr=22050, hop_length=512)
绘制频谱图
参数：
    data：要显⽰的矩阵
    sr ：采样率
    hop_length ：帧移
    x_axis 、y_axis ：x和y轴的范围
    频率类型
        'linear'，'fft'，'hz'：频率范围由FFT窗⼝和采样率确定
        'log'：频谱以对数刻度显⽰
        'mel'：频率由mel标度决定
    时间类型
        time：标记以毫秒，秒，分钟或⼩时显⽰。值以秒为单位绘制。
        s：标记显⽰为秒。
        ms：标记以毫秒为单位显⽰。
    所有频率类型均以Hz为单位绘制
'''
plt.figure()
plt.subplot(2, 1, 1)
librosa.display.specshow(S ** 2, sr=sr, y_axis='log') # 绘制功率谱
plt.colorbar()
plt.title('Power spectrogram')
plt.subplot(2, 1, 2)
# 相对于峰值功率计算dB, 那么其他的dB都是负的，注意看后边cmp值
librosa.display.specshow(librosa.power_to_db(S ** 2, ref=np.max), sr=sr, y_axis='log', x_axis='time') # 绘制对数功率谱
plt.colorbar(format='%+2.0f dB')
plt.title('Log-Power spectrogram')
plt.set_cmap("autumn")
plt.tight_layout()
plt.show()
#绘制幅值转dB的线性频率功率谱和对数频率功率谱
plt.figure()
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
plt.subplot(2, 1, 1)
librosa.display.specshow(D, y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('Linear-frequency power spectrogram') # 线性频率功率谱
plt.subplot(2, 1, 2)
librosa.display.specshow(D, y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Log-frequency power spectrogram') # 对数频率功率谱
plt.show()

# librosa.filters.mel() 创建⼀个滤波器组矩阵以将FFT合并成Mel频率
'''
librosa.filters.mel(sr, n_fft, n_mels=128, fmin=0.0, fmax=None, htk=False, norm=1)
创建⼀个滤波器组矩阵以将FFT合并成Mel频率
参数：
    sr ：输⼊信号的采样率
    n_fft ：FFT组件数
    n_mels ：产⽣的梅尔带数
    fmin ：最低频率（Hz）
    fmax：最⾼频率（以Hz为单位）。如果为None，则使⽤fmax = sr / 2.0
    norm：{None，1，np.inf} [标量]
    如果为1，则将三⾓mel权重除以mel带的宽度（区域归⼀化）。否则，保留所有三⾓形的峰值为1.0
返回：Mel变换矩阵
'''
melfb = librosa.filters.mel(sr=22050, n_fft=2048)
plt.figure()
librosa.display.specshow(melfb, x_axis='linear')
plt.ylabel('Mel filter')
plt.title('Mel filter bank')
plt.colorbar()
plt.tight_layout()
plt.show()

# librosa.feature.melspectrogram() 求取Mel频谱
'''
librosa.feature.melspectrogram(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window='hann', center=True, pad_mode=
计算Mel频谱
如果提供了频谱图输⼊S，则通过mel_f.dot（S）将其直接映射到mel_f上。
如果提供了时间序列输⼊y，sr，则⾸先计算其幅值频谱S，然后通过mel_f.dot（S ** power）将其映射到mel scale上 。默认情况下，power= 2在功率谱上运⾏。
参数：
    y ：⾳频时间序列
    sr ：采样率
    S ：频谱
    n_fft ：FFT窗⼝的⻓度
    hop_length ：帧移
    win_length ：窗⼝的⻓度为win_length，默认win_length = n_fft
        window ：字符串，元组，数字，函数或shape =（n_fft, )
        窗⼝规范（字符串，元组或数字）；看到scipy.signal.get_window 
        窗⼝函数，例如 scipy.signal.hanning ⻓度为n_fft的向量或数组
    center：bool 
        如果为True，则填充信号y，以使帧 t以y [t * hop_length]为中⼼。
        如果为False，则帧t从y [t * hop_length]开始
    power：幅度谱的指数。例如1代表能量，2代表功率，等等
    n_mels：滤波器组的个数 1288
    fmax：最⾼频率
返回：Mel频谱shape=(n_mels, t)
'''
# ⽅法⼀：使⽤时间序列求Mel频谱
print('Mel频谱',librosa.feature.melspectrogram(y=y, sr=sr))
# ⽅法⼆：使⽤stft频谱求Mel频谱
D = np.abs(librosa.stft(y)) ** 2 # stft频谱
S = librosa.feature.melspectrogram(S=D) # 使⽤stft频谱求Mel频谱
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()

# 提取Log-Mel Spectrogram 特征
# 提取 mel spectrogram feature
melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=512, n_mels=128)
logmelspec = librosa.power_to_db(melspec) # 转换到对数刻度
print('Log-Mel Spectrogram 特征',logmelspec.shape)

# librosa.feature.mfcc() 提取MFCC系数
'''
librosa.feature.mfcc(y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm='ortho', **kwargs)
提取MFCC系数
参数：
    y：⾳频数据
    sr：采样率
    S：np.ndarray，对数功能梅尔谱图
    n_mfcc：int>0，要返回的MFCC数量
    dct_type：None, or {1, 2, 3} 离散余弦变换（DCT）类型。默认情况下，使⽤DCT类型2。
    norm： None or ‘ortho’ 规范。如果dct_type为2或3，则设置norm =’ortho’使⽤正交DCT基础。 标准化不⽀持dct_type = 1。
返回：
    M： MFCC序列
'''
# 提取 MFCC feature
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# 可视化 MFCC feature
print('MFCC feature',mfccs.shape) # (40, 65)
librosa.display.specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()
