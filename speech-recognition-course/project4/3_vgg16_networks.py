
import torch
import torch.nn as nn
import torchvision

'''
法一：预训练的vgg16，添加一层add_module()：
#在数据集上取得比较好的效果的网络参数
vgg16_true=torchvision.models.vgg16(pretrained=True)
#添加一层add_module()：
vgg16_true.classifier.add_module('add_linear',nn.Linear(1000,2)) #名称,神经网络线性层(放在classifier层中），从1000变到10
#可以打印模型，看一下模型结构
print(vgg16_true)
print(vgg16_true)
image=torch.randn(1,3,224,224)
vgg16_true.eval()
with torch.no_grad():
    output=vgg16_true(image)
print(output)
print(output.argmax(1))
'''

'''
法二：预训练的vgg16，直接修改预训练网络最后一层
vgg16_true=torchvision.models.vgg16(pretrained=True)
vgg16_true.classifier[6]=nn.Linear(4096,2) # 直接修改预训练网络最后一层，使模型不会输出1000类，而是输出2类
print(vgg16_true)
print(vgg16_true)
image=torch.randn(1,3,224,224)
vgg16_true.eval()
with torch.no_grad():
    output=vgg16_true(image)
print(output)
print(output.argmax(1))
'''

# 按照讲义搭建的vgg16
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
print(vgg16)
image=torch.randn(1,3,224,224)
vgg16.eval()
with torch.no_grad():
    output=vgg16(image)
print(output)
print(output.argmax(1))

