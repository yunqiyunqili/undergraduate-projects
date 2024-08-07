# 扩展任务⼀：⾃⼰写⼀个VGG16的⽹络
# 按照下图⾃⼰写⼀个VGG16的⽹络
import torch
import torch.nn as nn

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        # 定义卷积层部分
        self.conv_layers = nn.Sequential(
            # 第1个卷积层：输⼊通道为3，输出通道为64，卷积核⼤⼩为3，填充为1，添加BatchNorm层和ReLU激活函数，后⾯紧跟⼀个最⼤池化层，池化核⼤⼩为2，步⻓为2。
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), # 添加BatchNorm2d，通过对每个⼩批量的数据在每个神经元的输出上做归⼀化，可以加速⽹络的训练，并且使得⽹络对初始参数的选择更加鲁棒
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第2个卷积层：输⼊通道为64，输出通道为128，卷积核⼤⼩为3，填充为1，添加BatchNorm层和ReLU激活函数，后⾯紧跟⼀个最⼤池化层，池化核⼤⼩为2，步⻓为2。
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第3个卷积层：输⼊通道为128，输出通道为256，卷积核⼤⼩为3，填充为1，添加BatchNorm层和ReLU激活函数，后⾯紧跟⼀个最⼤池化层，池化核⼤⼩为2，步⻓为2。
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第4个卷积层：输⼊通道为256，输出通道为512，卷积核⼤⼩为3，填充为1，添加BatchNorm层和ReLU激活函数，后⾯紧跟⼀个最⼤池化层，池化核⼤⼩为2，步⻓为2。
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 第5个卷积层：输⼊通道为512，输出通道为512，卷积核⼤⼩为3，填充为1，添加BatchNorm层和ReLU激活函数，后⾯紧跟⼀个最⼤池化层，池化核⼤⼩为2，步⻓为2。
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512), # 添加BatchNorm层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        # 全连接层：三个线性层，中间添加了ReLU激活函数和Dropout正则化。
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), # 线性层
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096), # 线性层
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2), # 线性层
            )
    # 前向传播函数
    def forward(self, x):
        x = self.conv_layers(x) # 输⼊x通过卷积层部分得到特征图
        x = x.reshape(x.size(0), -1) # 压缩成1维向量
        x = self.fc_layers(x) # 通过全连接层输出
        return x

vgg16=VGG16() # 创建了VGG16模型的实例
print(vgg16)
image=torch.randn(1,3,224,224) # 创建⼀张随机的输⼊图像
vgg16.eval() # 将模型设置为评估模式，并关闭梯度计算
with torch.no_grad(): # 对输⼊图像进⾏推理，并输出模型的预测结果
    output=vgg16(image)
print(output) # 输出模型的预测结果
print(output.argmax(1)) # 输出预测结果的类别