'''
1）将mel谱图的读取封装为类
step1: 选取1个播⾳员的谱图⽂件夹，读取第⼀张谱图，显⽰图，打印图⽚的分辨率和标签，⽂件夹⻓度
的读取等功能
step2: 选取2个播⾳员的谱图⽂件夹，将数据集合并，并打印合并后数据集的⻓度。
step3: 选取1个播⾳员的谱图⽂件夹，⽣成label⽂件夹。
2）提⾼任务：
对数据集进⾏交叉验证分类（可以把这个功能封装为类），输出trainset⻓度和validationset⻓度。
'''

# 将mel谱图的读取封装为类
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
class MelSpectrogramDataset(Dataset):
    '''该类封装了读取谱图的功能'''
    def __init__(self, root_dir):
        '''初始化类MelSpectrogramDataset'''
        self.root_dir = root_dir
        self.file_list = sorted(os.listdir(self.root_dir))
    def __len__(self):
        '''返回谱图⽂件夹中⽂件的数量'''
        return len(self.file_list)
    def __getitem__(self, index):
        '''根据索引读取⽂件并返回⽂件和标签。'''
        file_path = os.path.join(self.root_dir, self.file_list[index])
        label = file_path.split("/")[-1][:-4] # ⽂件夹名即标签，通过⽂件路径获取标签
        image = cv2.imread(file_path) # 读取图⽚
        return image, label # 返回图⽚和标签
    
# 1、选取1个播⾳员的谱图⽂件夹，读取第⼀张谱图，显⽰图，打印图⽚的分辨率和标签，⽂件夹⻓度的读取等功能
dataset = MelSpectrogramDataset("Boyin_mel_save/haixia")
image, label = dataset[0] # 获取第⼀张图⽚和对应的标签
print(f"Image shape: {image.shape}, Label: {label}") # 打印图⽚尺⼨和标签
cv2.imshow("Mel Spectrogram", image) # 展⽰图⽚
print(f"Length of haixia_dataset: {len(dataset)}") # 打印Boyin_mel_save/haixia/⽂件夹的⻓度
# cv2.waitKey(0) # 等待按键
cv2.destroyAllWindows() # 关闭所有窗⼝

# 2、选取2个播⾳员的谱图⽂件夹，将数据集合并，并打印合并后数据集的⻓度。
kanghui_dataset = MelSpectrogramDataset("Boyin_mel_save/kanghui")
haixia_dataset = MelSpectrogramDataset("Boyin_mel_save/haixia")
merged_dataset = np.concatenate((kanghui_dataset, haixia_dataset))
print(f"Merged dataset length: {len(merged_dataset)}")

# 3、选取1个播⾳员的谱图⽂件夹，⽣成label⽂件夹。
if not os.path.exists("labels"):
    os.mkdir("labels")
with open("labels/haixia.txt", "w") as f:
    for i in range(len(haixia_dataset)):
        label = haixia_dataset[i][1]
        f.write(f"{label}\n")