# 根据example/data中的train、dev、test数据集，使用ANN进行文本分类，如下：
# 导入已经编写好的model文件夹中的ANNmodel.py文件

import sys
sys.path.append('../')
from model.ANNmodel import ANNmodel
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读取数据
X_train = pd.read_csv('../../example/data/train.csv')
X_test = pd.read_csv('../../example/data/test.csv')
X_dev = pd.read_csv('../../example/data/dev.csv')
y_train = pd.read_csv('../../example/data/y_train.csv')
y_test = pd.read_csv('../../example/data/y_test.csv')
y_dev = pd.read_csv('../../example/data/y_dev.csv')

# 数据预处理
scaler = StandardScaler()  # 标准化
X_train = scaler.fit_transform(X_train)  # 训练标准化
X_test = scaler.transform(X_test)  # 测试集和验证集使用训练集的标准化
X_dev = scaler.transform(X_dev)  # 测试集和验证集使用训练集的标准化

# 转换为tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
X_dev = torch.tensor(X_dev, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.long)
y_test = torch.tensor(y_test.values, dtype=torch.long)
y_dev = torch.tensor(y_dev.values, dtype=torch.long)
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)
X_dev = X_dev.to(device)
y_dev = y_dev.to(device)


# 定义模型
model = ANNmodel(input_dim=X_train.shape[1],output_dim=2)
model.to(device)

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器


# 将数据带进入dataloader中
train_data = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# 训练模型
for epoch in range(1000):
    for i,data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()  # 梯度清零
        output = model(inputs)  # 前向传播
        loss = criterion(output, labels.squeeze())  # 计算损失

        # 反向传播
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    if epoch % 100 == 0:
        print('epoch: {}, loss: {}'.format(epoch, loss.item()))


# # 分成batch训练
# batch_size = 64
# for epoch in range(1000):
#     for i in range(0, X_train.size(0), batch_size):
#         optimizer.zero_grad()  # 梯度清零
#         output = model(X_train[i:i+batch_size])  # 前向传播
#         loss = criterion(output, y_train[i:i+batch_size].squeeze())  # 计算损失
#
#         # 反向传播
#         loss.backward()  # 反向传播
#         optimizer.step()
#     if epoch % 100 == 0:
#         print('epoch: {}, loss: {}'.format(epoch, loss.item()))

# # 训练模型
# for epoch in range(2000):
#     optimizer.zero_grad()  # 梯度清零
#     output = model(X_train)  # 前向传播
#     loss = criterion(output, y_train.squeeze())  # 计算损失
#
#     # 反向传播
#     loss.backward()  # 反向传播
#     optimizer.step()
#     if epoch % 100 == 0:
#         print('epoch: {}, loss: {}'.format(epoch, loss.item()))
#
# 验证模型
output = model(X_dev)
_, predicted = torch.max(output, 1)
total = y_dev.size(0)
correct = (predicted == y_dev.squeeze()).sum().item()
print('Accuracy of the network on the dev set: %d %%' % (100 * correct / total))

# 测试模型
output = model(X_test)
_, predicted = torch.max(output, 1)
total = y_test.size(0)
correct = (predicted == y_test.squeeze()).sum().item()
print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))
