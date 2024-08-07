# 完成某个mel谱图输⼊后的测试
# 分别输⼊海霞和康辉的mel谱图测试，注意康辉的mel谱图的label设置为0.，海霞为1.
from PIL import Image
from torchvision import transforms
import torch

# Kanghui mel谱图
file_path = 'kanghui.png'
image = Image.open(file_path)
print(image.size)
image = image.convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图⽚的⼤⼩调整为 224x224
    transforms.ToTensor() # 将图⽚转换为 Tensor
])
image = transform(image)
print(image.shape)
model = torch.load('pth/tudui_15.pth')
#print(model)
image=torch.reshape(image,(1,3,224,224))
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1)) # output.argmax(1)为0，是康辉的label，表明康辉mel谱图分类正确

# haixia mel 
file_path = 'haixia.png'
image = Image.open(file_path)
print(image.size)
image = image.convert('RGB')
transform = transforms.Compose([
    transforms.Resize((224, 224)), # 将图⽚的⼤⼩调整为 224x224
    transforms.ToTensor() # 将图⽚转换为 Tensor
])
image = transform(image)
print(image.shape)
model = torch.load('pth/tudui_15.pth')
#print(model)
image=torch.reshape(image,(1,3,224,224))
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))