import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from features import get_features, load_data

sample_rate = 16000  # 设置采样率
emotions_label_EmoDB = {  # 定义情绪标签的映射字典
    'W': '0',
    'L': '1',
    'E': '2',
    'A': '3',
    'F': '4',
    'T': '5',
    'N': '6'}

data_path = './wav/*.wav'  # 定义音频文件路径


waveforms, emotions = load_data(data_path, sample_rate)  # 使用load_data函数加载音频数据和情绪标签
print(f'Waveforms set: {len(waveforms)} samples')  # 打印加载的音频波形样本数
print(f'Waveform signal length: {len(waveforms[5])}')  # 打印单个音频波形信号的长度
print(f'Emotions set: {len(emotions)} sample labels')  # 打印加载的情绪标签样本数

train_set, valid_set, test_set = [], [], []
X_train, X_valid, X_test = [], [], []
y_train, y_valid, y_test = [], [], []

waveforms = np.array(waveforms)

for emotion_num in range(len(emotions_label_EmoDB)):  # 遍历情绪标签的索引
    emotion_indices = [index for index, emotion in enumerate(emotions) if emotion == emotion_num]  # 根据给定的情绪标签编号（emotion_num），在emotions列表中找到所有与该情绪标签对应的索引值

    np.random.seed(69)  # 随机打乱索引顺序
    emotion_indices = np.random.permutation(emotion_indices)

    dim = len(emotion_indices)  # 计算索引的维度

    # 划分训练集、验证集和测试集的索引
    train_indices = emotion_indices[:int(0.6 * dim)]
    valid_indices = emotion_indices[int(0.6 * dim):int(0.8 * dim)]
    test_indices = emotion_indices[int(0.8 * dim):]

    # 将训练集的波形数据添加到X_train列表中
    X_train.append(waveforms[train_indices, :])
    print(len(X_train))

    # 创建训练集的情绪标签，并添加到y_train列表中
    # y_train列表中存储了与训练样本对应的情绪标签，每个情绪标签的数值与情绪标签编号（emotion_num）相同，并且与训练样本的数量相匹配。
    y_train.append(np.array([emotion_num] * len(train_indices), dtype=np.int32))

    # 验证集
    X_valid.append(waveforms[valid_indices, :])
    y_valid.append(np.array([emotion_num] * len(valid_indices), dtype=np.int32))

    # 测试集
    X_test.append(waveforms[test_indices, :])
    y_test.append(np.array([emotion_num] * len(test_indices), dtype=np.int32))

# 将X_train列表/X_valid列表/X_test列表中的波形数据沿指定轴进行连接
X_train = np.concatenate(X_train, axis=0)
X_valid = np.concatenate(X_valid, axis=0)
X_test = np.concatenate(X_test, axis=0)

# 将y_train列表/y_valid列表/y_test列表中的情绪标签沿指定轴进行连接
y_train = np.concatenate(y_train, axis=0)
y_valid = np.concatenate(y_valid, axis=0)
y_test = np.concatenate(y_test, axis=0)

# 打印训练集、验证集和测试集的形状信息
print(f'X_train:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test:{X_test.shape}, y_test:{y_test.shape}')

features_train, features_valid, features_test = [], [], []
print('Train waveforms:')
features_train = get_features(X_train, features_train, sample_rate)  # 训练集的波形数据特征提取信息
print('\n\nValidation waveforms:')
features_valid = get_features(X_valid, features_valid, sample_rate)  # 验证集的波形数据特征提取信息
print('\n\nTest waveforms:')
features_test = get_features(X_test, features_test, sample_rate)  # 测试集的波形数据特征提取信息

# 打印总的特征集信息，包括训练集、验证集和测试集的样本数
print(
    f'\n\nFeatures set: {len(features_train) + len(features_test) + len(features_valid)} total, '
    f'{len(features_train)} train, {len(features_valid)} validation, {len(features_test)} test samples')
# 打印特征矩阵的形状信息
print(
    f'Features (MFC coefficient matrix) shape: {len(features_train[0])} mel frequency coefficients x '
    f'{len(features_train[0][1])} time steps')


# 保存无增强的特征与标签，作为可直接训练测试文件
print(len(features_train))  # 打印训练集特征的数量
print(features_train[1].shape)  # 打印第一个训练样本的特征矩阵形状

# 在特征矩阵的第二个维度上添加一个维度，变成三维矩阵
X_train = np.expand_dims(features_train, 1)
X_valid = np.expand_dims(features_valid, 1)
X_test = np.expand_dims(features_test, 1)
scaler = StandardScaler()  # 创建一个StandardScaler对象

'''
X_train = scaler.fit_transform(X_train)  # 对训练集进行标准化处理
print(X_train.shape)  # 打印标准化后的训练集特征矩阵的形状
'''

# N 样本数，C 通道数，这里为 1，H 高度，即特征向量长度，W 宽度，即时间步数。
N, C, H, W = X_train.shape
X_train = np.reshape(X_train, (N, -1))  # 将 X_train 重新排列为二维矩阵，形状为 (N, C*H*W)，为了方便后续的特征标准化处理
X_train = scaler.fit_transform(X_train)  # 对 X_train 进行标准化处理，使得每个特征的均值为 0，方差为 1。
X_train = np.reshape(X_train, (N, C, H, W))  # 重新排列为原来的形状 (N, C, H, W)。

N, C, H, W = X_valid.shape
X_valid = np.reshape(X_valid, (N, -1))
X_valid = scaler.transform(X_valid)
X_valid = np.reshape(X_valid, (N, C, H, W))

N, C, H, W = X_test.shape
X_test = np.reshape(X_test, (N, -1))
X_test = scaler.transform(X_test)
X_test = np.reshape(X_test, (N, C, H, W))

print(f'X_train scaled:{X_train.shape}, y_train:{y_train.shape}')
print(f'X_valid scaled:{X_valid.shape}, y_valid:{y_valid.shape}')
print(f'X_test scaled:{X_test.shape}, y_test:{y_test.shape}')

# 将特征矩阵和标签保存到文件中。
featuresFileName = './feature/mel40_feat_emodb.pkl'
features = {'X_train': X_train, 'X_valid': X_valid, 'X_test': X_test,
            'y_train': y_train, 'y_valid': y_valid, 'y_test': y_test}
print("preservation features at %s" % featuresFileName)
with open(featuresFileName, 'wb') as f:
    pickle.dump(features, f)
