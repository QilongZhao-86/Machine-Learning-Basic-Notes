# 机器学习
---
这个笔记基础知识来源于李宏毅的课程Machine Learning基于2025年版本的课程视频整理而成，内容包括监督学习、无监督学习、神经网络等基础知识。笔记中包含了课程中的重要概念、算法和数学推导，旨在帮助读者更好地理解机器学习的基本原理和应用。
课程地址： https://www.bilibili.com/video/BV1TAtwzTE1S?spm_id_from=333.788.videopod.sections&vd_source=61bcc59902591a3dc6dd5817f3f53a86
同时也会包含本人在**计算机视觉**&**具身智能**等方向的一些学习笔记和心得体会。
欢迎大家交流讨论，共同进步！
作者：Qilong Zhao
github：https://github.com/QilongZhao-86

## Pytorch基础
Pytorch是一个流行的深度学习框架，广泛应用于机器学习和人工智能领域。以下是Pytorch的一些基础知识：
1. **张量（Tensor）**：Pytorch的核心数据结构是张量，类似于NumPy的数组。张量可以在CPU和GPU上进行计算，支持多维数组操作。
2. **自动微分（Autograd）**：Pytorch提供了自动微分功能，可以自动计算梯度，方便进行反向传播和优化。
3. **神经网络模块（torch.nn）**：Pytorch提供了丰富的神经网络模块，可以方便地构建和训练神经网络模型。
4. **优化器（torch.optim）**：Pytorch提供了多种优化算法，如SGD、Adam等，用于更新模型参数以最小化损失函数。
5. **数据加载（torch.utils.data）**：Pytorch提供了数据加载和预处理的工具，可以方便地处理大规模数据集。
6. **训练循环**：Pytorch允许用户自定义训练循环，灵活控制训练过程。
7. **模型保存与加载**：Pytorch提供了简单的接口用于保存和加载模型参数，方便模型的持久化和部署。
8. **社区支持**：Pytorch拥有活跃的社区，提供了丰富的资源和教程，帮助用户解决问题和学习新技术。
通过掌握以上基础知识，用户可以使用Pytorch构建和训练各种机器学习模型，应用于图像识别、自然语言处理等领域。  

## 语法笔记
### tensor的创建
```python   
import torch
# 创建一个未初始化的3x4张量
x = torch.empty(3, 4)
print(x)
# 创建一个随机初始化的3x4张量
x = torch.rand(3, 4)
print(x)
# 创建一个全零的3x4张量，数据类型为long
x = torch.zeros(3, 4, dtype=torch.long)
print(x)
# 直接从数据创建张量
x = torch.tensor([3.14, 2.71])
print(x)
```
### transpose和索引
```python
torch.transpose(a_t, 0, 1)
import torch
# 创建一个2x3的张量 
a = torch.tensor([[1, 2, 3], [4, 5, 6]])
# 转置张量
a_t = torch.transpose(a, 0, 1)
print(a_t)
```
### mean和sum row是1 column是0 row是行 column是列
```python
import torch
# 创建一个2x3的张量
a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
# 计算所有元素的均值
mean_all = torch.mean(a)
print(mean_all)
# 计算每列的均值
mean_dim0 = torch.mean(a, dim=0)
print(mean_dim0)
# 计算每行的均值
mean_dim1 = torch.mean(a, dim=1)
print(mean_dim1)
# 计算所有元素的和
sum_all = torch.sum(a)
print(sum_all)
# 计算每列的和
sum_dim0 = torch.sum(a, dim=0)
print(sum_dim0)
# 计算每行的和
sum_dim1 = torch.sum(a, dim=1)
print(sum_dim1)
```
### tensor的运算
```python
import torch
# 创建两个张量
x = torch.rand(3, 4)
y = torch.rand(3, 4)
# 张量加法
print(x + y)
print(torch.add(x, y))
# 创建一个结果张量用于存储加法结果
result = torch.empty(3, 4)
torch.add(x, y, out=result)
print(result)
# 张量原地加法
y.add_(x)
print(y)
# 张量索引
print(x[:, 1])
# 改变张量形状
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # 自动计算行数
print(x.size(), y.size(), z.size())
# 获取单个元素的值
x = torch.randn(1)
print(x)
print(x.item())
# 张量的乘法
x = torch.randn(2, 3)
y = torch.randn(3, 4)
print(torch.mm(x, y))
#张量和标量的运算
x = torch.randn(3, 4)
print(x)
print(x + 2)
print(x * 3)
# 张量的逐元素乘法
y = torch.randn(3, 4)
print(x * y)
print(torch.mul(x, y))
# 特征值的计算
eigenvalues = torch.linalg.eigvals(matrix)
# 只取实部
eigenvalues = eigenvalues.real
# 排序特征值
eigenvalues, _ = torch.sort(eigenvalues)
```
### 广播机制
```python
import torch
# 创建一个3x4的张量
x = torch.rand(3, 4)
# 创建一个1x4的张量
y = torch.rand(1, 4)
# 广播机制进行加法运算
print(x + y)
```
### GPU加速
```python
import torch
# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # 使用GPU
    y = torch.ones_like(x, device=device)  # 在GPU上创建张量
    x = x.to(device)                       # 将张量移动到GPU
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # 将结果移动回CPU并转换数据类型
```
### 自动微分
```python
import torch
# 创建一个张量并设置requires_grad=True以启用自动微分
x = torch.ones(2, 2, requires_grad=True)
print(x)
# 对张量进行一些操作
y = x + 2
print(y)
z = y * y * 3
out = z.mean()
print(z, out)
# 反向传播
out.backward()
# 输出梯度
print(x.grad)
```
### 神经网络模块
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# 定义一个简单的神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)
# 定义一个损失函数和优化器
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# 训练循环
for epoch in range(2):  # 多次循环遍历数据集
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()  # 清零梯度
        outputs = net(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 优化参数
        running_loss += loss.item()
        if i % 2000 == 1999:    # 每2000个小批量输出一次loss
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')
```
