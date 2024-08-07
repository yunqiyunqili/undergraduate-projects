import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
# from Project3 import k_fold
from PIL import Image

class MelSpectrogramDataset(Dataset):
    '''
    该类封装了读取谱图的功能
    在我们定义的MelSpectrogramDataset类中，我们重载了__len__和__getitem__函数，使得我们可以通过类实例的索引访问数据集中的图像和标签。
__len__函数返回数据集的长度（即数据集中包含的样本数量），__getitem__函数通过指定的索引值获取对应的数据样本。因此，当我们调用dataset[1]时，__getitem__函数会被调用，返回索引为0的样本，即第一张图像和对应的标签。而当我们调用len(dataset)时，__len__函数会被调用，返回数据集的长度。
    '''
    def __init__(self, root_dir, label_dir, transform=None):
        '''初始化类MelSpectrogramDataset'''
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.transform = transform
        self.file_list = sorted(os.listdir(os.path.join(self.root_dir,self.label_dir)))

    def __len__(self):
        '''返回谱图文件夹中文件的数量'''
        return len(self.file_list)

    def __getitem__(self, index):
        '''根据索引读取文件并返回文件和标签。'''
        file_path = os.path.join(self.root_dir,self.label_dir)
        file_path = os.path.join(file_path, self.file_list[index])
        label = int(self.label_dir)
        image = cv2.imread(file_path)  # 读取图片
        if self.transform:
            # 将numpy数组转换为PIL.Image格式
            image = Image.fromarray(image)
            image = self.transform(image)
            image = np.transpose(image, (1, 2, 0))
        return image, label  # 返回图片和标签


root_dir1 = "Boyin_mel/train"
label_dir1 = '0'
root_dir2 = "Boyin_mel/val"
label_dir2 = '1'
# 选取1个播音员的谱图文件夹，读取第一张谱图，显示图，打印图片的分辨率和标签，文件夹长度的读取
dataset = MelSpectrogramDataset(root_dir2,label_dir2)
image, label = dataset[0]  # 获取第一张图片和对应的标签
print(f"Image shape: {image.shape}, Label: {label}")  # 打印图片尺寸和标签
plt.imshow(image)  # 展示图片
plt.show()
print(f"Length of haixia_dataset: {len(dataset)}")  # 打印Boyin_mel_save/0/文件夹的长度

# 选取2个播音员的谱图文件夹，将数据集合并，并打印合并后数据集的长度
haixia_dataset = MelSpectrogramDataset(root_dir1,label_dir1 )
kanghui_dataset = MelSpectrogramDataset(root_dir2,label_dir2)
merged_dataset = np.concatenate((kanghui_dataset, haixia_dataset))
print(f"Merged dataset length: {len(merged_dataset)}")

# 选取1个播音员的谱图文件夹，生成label文件夹
if not os.path.exists("labels"):
    os.mkdir("labels")
with open("labels/haixia.txt", "w") as f:
    for i in range(len(haixia_dataset)):
        label = haixia_dataset[i][1]
        f.write(f"{label}\n")

# 裁剪图片操作的情况 （使用transform）
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片的大小调整为 224x224
    transforms.ToTensor() # 将图片转换为 Tensor
])
# 选取1个播音员的谱图文件夹，读取第一张谱图，显示图，打印图片的分辨率和标签，文件夹长度的读取
dataset = MelSpectrogramDataset(root_dir1,label_dir1,transform=transform)
image, label = dataset[0]  # 获取第一张图片和对应的标签
print(f"Image shape: {image.shape}, Label: {label}")  # 打印图片尺寸和标签
plt.imshow(image)  # 展示图片
plt.show()