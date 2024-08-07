# 任务⼆：完成康辉、海霞的声纹识别
# 对刚刚搭建的讲义中的vgg16模型，完成模型的训练，验证，并⽣成⼏个pth模型

import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn

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
        label = file_path.split("/")[-1][:-4] # 文件夹名即标签，通过文件路径获取标签
        label = int(self.label_dir)
        image = cv2.imread(file_path)  # 读取图片
       # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            # 将numpy数组转换为PIL.Image格式
            image = Image.fromarray(image)
            image = self.transform(image)
            image = np.transpose(image, (1, 2, 0))
        #image = image.transpose((2, 1, 0))  # 将最后一个维度移动到第二个维度
        return image, label  # 返回图片和标签

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),  # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )



        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),

            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),

            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2),

        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x

vgg16=VGG16()


# 创建变换函数
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图片的大小调整为 224x224
    transforms.ToTensor() # 将图片转换为 Tensor
])


# 加载数据集
root_dir1 = "Boyin_mel/train"
label_dir1 = '0'
root_dir2 = "Boyin_mel/val"
label_dir2 = '1'

#      **************分别划分train 0***********
# 定义数据集路径和目录
kanghui_train_dataset = MelSpectrogramDataset(root_dir1, label_dir2, transform=transform)
haixia_train_dataset = MelSpectrogramDataset(root_dir1, label_dir1, transform=transform)
kanghui_val_dataset = MelSpectrogramDataset(root_dir2, label_dir2, transform=transform)
haixia_val_dataset = MelSpectrogramDataset(root_dir2, label_dir1, transform=transform)
train_dataset = kanghui_train_dataset + haixia_train_dataset
val_dataset = kanghui_val_dataset + haixia_val_dataset
test_data_size = len(val_dataset)
image, label = train_dataset[0]  # 获取第一张图片和对应的标签
print(f"Image shape: {image.shape}, Label: {label}")  # 打印图片尺寸和标签
print("Tensor shape:", transform(Image.fromarray(np.uint8(image))).shape)
print(f" kanghui_trainset length: {len(kanghui_train_dataset)}, kanghui_validationset length: {len(kanghui_val_dataset)}")
print(f" haixia_trainset length: {len(kanghui_train_dataset)}, haixia_validationset length: {len(kanghui_val_dataset)}")
print(f" trainset length: {len(train_dataset)}, validationset length: {test_data_size}")

# 定义训练参数
batch_size = 16
num_epochs = 10  # 训练轮次 1
learning_rate = 0.01


# 创建训练集和验证集的 DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 创建网络模型
# tudui = TuduiNet()


# 损失函数
loss = nn.CrossEntropyLoss()
# 创建随机梯度下降法优化器（神经网络参数，学习率）
optim = torch.optim.SGD(vgg16.parameters(), lr=learning_rate)

# 记录训练次数
total_train_step=0
# 记录测试次数
total_test_step=0


for i in range(num_epochs):
    print('-----第{}轮训练开始-----'.format(i+1))
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.float()
        imgs=imgs.permute(0, 3, 1, 2)
        print(imgs.shape)
        outputs = vgg16(imgs)
        print(outputs)
        print(targets)
        #targets = torch.tensor([int(targets[1]), ], dtype=torch.int64)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        total_train_step+=1
        print("训练次数:{}，loss：{}".format(total_train_step,result_loss))

    torch.save(vgg16,'pth/tudui_{}.pth'.format(i+100))
    # 测试步骤开始
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_loader:
            imgs, targets = data
            # targets=torch.tensor([int(targets[1]), ], dtype=torch.int64)
            # imgs = imgs.float()
            imgs = imgs.permute(0, 3, 1, 2)
            outputs = vgg16(imgs)
            result_loss = loss(outputs, targets)
            total_test_loss = total_test_loss + result_loss
            print(total_test_loss)
            accuracy = (outputs.argmax(1) == targets).sum()
            print(accuracy)
            total_accuracy = total_accuracy + accuracy
            print(total_accuracy)
    print('整体验证集上的loss：{}'.format(total_test_loss))
    print('整体验证集上的accuracy：{}'.format(total_accuracy / test_data_size))



