import glob
import os
import warnings
import librosa
import librosa.display
import numpy as np

warnings.filterwarnings('ignore')

# 情绪标签的字典
emotions_dict_EmoDB = {
    'W': '0',
    'L': '1',
    'E': '2',
    'A': '3',
    'F': '4',
    'T': '5',
    'N': '6'}


def feature_melspectrogram(waveform, sr, fft=1024, winlen=512, window='hamming', hop=256, mels=128, ):
    '''
    计算音频波形的梅尔频谱图特征
    :param waveform: 音频波形数据
    :param sr: 音频的采样率
    :param fft:  FFT（快速傅里叶变换）的大小，默认为 1024
    :param winlen: 窗口长度，默认为 512
    :param window: 窗口函数类型，默认为 'hamming'。
    :param hop: 跳跃长度，默认为 256
    :param mels: 梅尔滤波器的数量，默认为 128
    :return: 梅尔频谱图特征
    '''
    melspectrogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_fft=fft, win_length=winlen,
                                                    window=window, hop_length=hop, n_mels=mels, fmax=sr / 2)
    melspectrogram = librosa.power_to_db(melspectrogram, ref=np.max)
    return melspectrogram


def feature_mfcc(waveform, sr, n_mfcc=40, fft=1024, winlen=512, window='hamming', mels=128):
    """
    计算音频波形的 MFCC 特征。
    :param waveform: 音频波形数据。
    :param sr: 音频的采样率（采样频率）。
    :param n_mfcc: MFCC 系数的数量，默认为 40。
    :param fft: FFT（快速傅里叶变换）的大小，默认为 1024。
    :param winlen: 窗口长度，默认为 512。
    :param window: 窗口函数类型，默认为 'hamming'。
    :param mels: 梅尔滤波器的数量，默认为 128。
    :return: MFCC 特征。
    """
    mfc_coefficients = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=fft, win_length=winlen,
                                            window=window, n_mels=mels, fmax=sr / 2)
    return mfc_coefficients

def feature_zero_crossing_rate(waveform, hop_length=512):
    '''
    计算音频波形的过零率特征。
    :param waveform: 音频波形数据。
    :param hop_length: 跳跃长度，默认为 512。
    :return: 过零率特征。
    '''
    zero_crossing_rate = librosa.feature.zero_crossing_rate(waveform, hop_length=hop_length)
    return zero_crossing_rate


def feature_pitch(waveform, sr, hop_length=512):
    """
    计算音频波形的声调特征。
    :param waveform: 音频波形数据。
    :param sr: 音频的采样率（采样频率）。
    :param hop_length: 跳跃长度，默认为 512。
    :return: 声调特征。
    """
    pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr, hop_length=hop_length)
    pitch = np.nanmean(pitches)
    return pitch




def get_features(waveforms, features, sr):
    """
    获取音频波形的特征。

    :param waveforms: 音频波形列表。
    :param features: 特征列表。
    :param sr: 音频的采样率（采样频率）。
    :return: 特征列表。
    """
    file_count = 0
    for waveform in waveforms:
        mfccs = feature_mfcc(waveform, sr)  # MFCC
        features.append(mfccs)
        '''
        zero_crossing = feature_zero_crossing_rate(waveform)  # 过零率
        pitch = feature_pitch(waveform, sr)  # 声调
        zero_crossing_rate = zero_crossing[:, :mfccs.shape[1]]
        pitch_reshaped = np.tile(np.array([[pitch]]), (1, mfccs.shape[1]))
        features.append(np.concatenate((mfccs, zero_crossing_rate,pitch_reshaped), axis=0))
        '''
        file_count += 1
        print('\r' + f' Processed {file_count}/{len(waveforms)} waveforms', end='')
    return features


def get_waveforms(file, sr):
    '''
    加载音频文件并返回音频波形数据。

    :param file: 音频文件路径。
    :param sr: 音频的采样率（采样频率）。
    :return: 音频波形数据。
    '''
    waveform, _ = librosa.load(file, duration=3, offset=0, sr=sr)  # 使用 librosa 库加载音频文件，并指定相关参数
    waveform_homo = np.zeros((int(sr * 3, )))  # 创建一个大小为 sr * 3 的零数组
    waveform_homo[:len(waveform)] = waveform  # 将加载的音频波形复制到零数组中，以保持长度为 sr * 3
    return waveform_homo  # 返回处理后的音频波形数据



def load_data(data_path, sr):
    '''
    加载音频数据和情感标签。

    :param data_path: 音频文件路径。
    :param sr: 音频的采样率（采样频率）。
    :return: 音频数据和情感标签。
    '''
    emotions = []  # 存储情感标签的列表
    waveforms = []  # 存储音频数据的列表
    file_count = 0  # 文件计数器，用于跟踪处理的文件数量
    for file in glob.glob(data_path):  # 遍历指定路径下的所有音频文件
        file_name = os.path.basename(file)  # 获取音频文件的基本名称（不含路径）
        emotion = int(emotions_dict_EmoDB[file_name[5]])  # 从文件名中提取情感标签，并转换为整数类型
        waveform = get_waveforms(file, sr)  # 使用get_waveforms函数加载音频文件并获取音频数据
        waveforms.append(waveform)  # 将音频数据添加到waveforms列表中
        emotions.append(emotion)  # 将情感标签添加到emotions列表中
        file_count += 1  # 增加文件计数器
        print('\r' + f' Processed {file_count}/{535} audio samples', end='')  # 打印处理进度信息
    return waveforms, emotions  # 返回音频数据和情感标签的列表

