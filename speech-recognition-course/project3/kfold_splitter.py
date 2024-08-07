# 对数据集进⾏交叉验证分类（可以把这个功能封装为类），输出trainset⻓度和validationset⻓度。
import random
from mel_spectrogram_processing import MelSpectrogramDataset

class KFoldSplitter:
    ''' 为了实现交叉验证，我们可以先将数据集进⾏随机划分，
    然后每次选取其中⼀部分作为验证集，
    其余部分作为训练集。
    '''
    def __init__(self, dataset, num_folds=5, shuffle=True):
        self.dataset = dataset
        self.num_folds = num_folds
        self.shuffle = shuffle
    def split(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)
        fold_size = len(self.dataset) // self.num_folds
        for i in range(self.num_folds):
            start = i * fold_size
            end = (i+1) * fold_size if i < self.num_folds-1 else len(self.dataset)
            val_indices = indices[start:end]
            train_indices = list(set(indices) - set(val_indices))
            yield train_indices, val_indices

dataset = MelSpectrogramDataset("Boyin_mel_save/kanghui") # mel谱图数据集
splitter = KFoldSplitter(dataset, num_folds=5, shuffle=True) # 创建交叉验证实例
# 循环迭代器返回的每⼀个训练集和验证集的索引
for fold, (train_indices, val_indices) in enumerate(splitter.split()):
    ''' 这⾥使⽤了 splitter.split() 返回的迭代器，
    每次迭代都会返回⼀个元组，其中包含训练集和验证集的索引，
    enumerate() 函数⽤于将迭代器中的每个元素与其对应的索引进⾏关联，并返回⼀个元组。
    '''
    trainset = [dataset[i] for i in train_indices] # 获取训练集（将训练集的索引 train_indices 与数据集实例 dataset 关联起来，获取训练集数据。）
    valset = [dataset[i] for i in val_indices] # 获取验证集（将验证集的索引 val_indices 与数据集实例 dataset 关联起来，获取验证集数据。）
    print(f"Fold {fold+1}: trainset length: {len(trainset)}, validationset length: {len(valset)}") # 输出当前交叉验证的折数 fold，以及训练集和验证