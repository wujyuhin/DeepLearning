# 根据example/data中的train、dev、test数据集，使用ANN进行文本分类，如下：
# 导入已经编写好的model文件夹中的ANNmodel.py文件
import numpy as np
from model.ANNmodel import ANNmodel
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset
from example.word2vec.word2vec_model import collate, HotelDataset
import pytorch_lightning as pl

# 定义路径
data_path = '../data/ChineseNlpCorpus_processed.csv'
stopwords_path = '../data/stopwords.txt'
model_path = '../word2vec/word2vec.pkl'
# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================  在train_loader中使用word2vec =================================================
# 导入数据(导入)\划分数据集
data = pd.read_csv(data_path)
X = data['review']  # 形如：pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])
y = data['label']  # 形如：pd.DataFrame([1,0])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
train_data = HotelDataset(X_train, y_train)
model = Word2Vec.load(model_path)
# train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, collate_fn=collate(model, device))
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True, collate_fn=collate(model,device))

# ===============================  先word2vec再生成train_loader =================================================
X2 = pd.read_csv('../data/train.csv')  # 已经转成向量,形如: pd.DataFrame([[1,2,3,4,5],[2,3,4,5,6]])
y2 = pd.read_csv('../data/y_train.csv')  # 形如: pd.DataFrame([1,0])
X_train2 = torch.tensor(np.array(X2), dtype=torch.float32)
y_train2 = torch.tensor(y2.values, dtype=torch.long).squeeze()
X_train2 = X_train2.to(device)
y_train2 = y_train2.to(device)
train_data2 = HotelDataset(X_train2, y_train2)
train_loader2 = torch.utils.data.DataLoader(train_data2, batch_size=64, shuffle=True, drop_last=True, collate_fn=None)
# =============================================================================================================


#  ===================================  训练模型(简单板块)  ====================================================
# 定义模型
# model = ANNmodel(input_dim=128, output_dim=2)
# model.to(device)
# criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
# optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
# for epoch in range(100):
#     model.train()
#     for i, data in enumerate(train_loader2):
#         inputs, labels = data
#         optimizer.zero_grad()  # 梯度清零
#         output = model(inputs)  # 前向传播 input_dim=(128,1), output_dim=2
#         loss = criterion(output, labels.squeeze())  # 计算损失 labels.squeeze()将标签的维度从(64,1)变为(64,)
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
#     if epoch % 100 == 0:
#         print('epoch: {}, loss: {}'.format(epoch, loss.item()))
# ================================== pytorch_lightning training  =====================================================
model = ANNmodel(input_dim=128, output_dim=2)
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, train_loader)








# 在训练模型的过程中，我们可以使用验证集来验证模型的性能，以及调整超参数,并且画出学习曲线和验证曲线
# 画出学习曲线和验证曲线
# import matplotlib.pyplot as plt
# import numpy as np

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="my-awesome-project",
#
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 0.02,
#     "architecture": "CNN",
#     "dataset": "CIFAR-100",
#     "epochs": 1000,
#     }
# )
# losses = []
# accs = []
# for epoch in range(1000):
#     for i,data in enumerate(train_loader):
#         inputs, labels = data
#         optimizer.zero_grad()  # 梯度清零
#         output = model(inputs)  # 前向传播
#         loss = criterion(output, labels.squeeze())  # 计算损失
#
#         # 反向传播
#         loss.backward()  # 反向传播
#         optimizer.step()  # 更新参数
#     # if epoch % 100 == 0:
#     #     print('epoch: {}, loss: {}'.format(epoch, loss.item()))
#     #     losses.append(loss.item())
#     #     # 验证模型
#     #     output = model(X_dev)
#     #     _, predicted = torch.max(output, 1)
#     #     total = y_dev.size(0)
#     #     correct = (predicted == y_dev.squeeze()).sum().item()
#     #     acc = 100 * correct / total
#     #     accs.append(acc)
#     #     print('Accuracy of the network on the dev set: %d %%' % acc)
#     losses.append(loss.item())
#     output = model(X_dev)
#     _, predicted = torch.max(output, 1)
#     acc = (predicted == y_dev.squeeze()).sum().item() / y_dev.size(0)
#     wandb.log({"acc": acc, "loss": loss})
#
#
# wandb.finish()

# plt.plot(np.arange(0, 1000, 100), losses, label='train loss')
# plt.plot(np.arange(0, 1000, 100), accs, label='dev acc')
# plt.xlabel('epoch')
# plt.ylabel('loss/acc')
# plt.legend()
# plt.show()


#
# 验证模型
# output = model(X_dev)
# _, predicted = torch.max(output, 1)
# total = y_dev.size(0)
# correct = (predicted == y_dev.squeeze()).sum().item()
# print('Accuracy of the network on the dev set: %d %%' % (100 * correct / total))
#
# # 测试模型
# output = model(X_test)
# _, predicted = torch.max(output, 1)
# total = y_test.size(0)
# correct = (predicted == y_test.squeeze()).sum().item()
# print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))

#
# # 基于以上的代码，编写五折交叉验证的代码，如下：
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score
#
# # 读取数据
# X = pd.read_csv('../../example/data/train.csv')
# y = pd.read_csv('../../example/data/y_train.csv')
#
# # 数据预处理
# scaler = StandardScaler()  # 标准化
# X = scaler.fit_transform(X)  # 训练标准化
# X = torch.tensor(X, dtype=torch.float32)
# y = torch.tensor(y.values, dtype=torch.long)
# X = X.to(device)
# y = y.to(device)
#
#
#
# # 五折交叉验证
# accs = []
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# for fold, train_index, test_index in enumerate(kf.split(X)):
#     # 定义模型
#     model = ANNmodel(input_dim=X.shape[1], output_dim=2)  # ================================== 如果更换模型，需要修改此处
#     model.to(device)
#     criterion = nn.CrossEntropyLoss()  # ===================================================== 如果更换损失函数，需要修改此处
#     optimizer = optim.Adam(model.parameters(), lr=0.001)  # ================================== 如果更换优化器，需要修改此处
#     # 第n折数据切分
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     train_data = torch.utils.data.TensorDataset(X_train, y_train)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
#
#     # 训练模型
#     for epoch in range(1000):
#         for i,data in enumerate(train_loader):
#             inputs, labels = data
#             optimizer.zero_grad()  # 梯度清零
#             output = model(inputs)  # 前向传播
#             loss = criterion(output, labels.squeeze())  # 计算损失
#
#             # 反向传播
#             loss.backward()  # 反向传播
#             optimizer.step()  # 更新参数
#     with torch.no_grad():
#         output = model(X_test)
#         _, predicted = torch.max(output, 1)
#         # 计算acc
#         total = y_test.size(0)
#         correct = (predicted == y_test.squeeze()).sum().item()
#         acc = 100 * correct / total
#         accs.append(acc)
#         print('Accuracy of the network on the test set: %d %%' % acc)
#
# print('Average accuracy of the network on the test set: %d %%' % np.mean(accs))

# 测试模型
# output = model(X_dev)
# _, predicted = torch.max(output, 1)
# total = y_dev.size(0)
# correct = (predicted == y_dev.squeeze()).sum().item()
# print('Accuracy of the network on the test set: %d %%' % (100 * correct / total))


# 将数据带进入dataloader中
# train_data = torch.utils.data.TensorDataset(X_train, y_train)
# class HotelDataset(Dataset):
#     def __init__(self, x_train, y_train):
#         self.x_train = x_train
#         self.y_train = y_train
#
#     def __getitem__(self, index):
#         return self.x_train[index], self.y_train[index]
#
#     def __len__(self):
#         return len(self.x_train)
#
#
# class collate:
#     def __init__(self, model, device):
#         self.model = model
#         self.device = device
#
#     def __call__(self, data):
#         X = []
#         y = []
#         for text in data:
#             vec = []
#             for word in text[0]:
#                 try:
#                     vec.append(self.model.wv[word])
#                 except:
#                     pass
#             if len(vec) == 0:
#                 vec = np.array([0] * 128)
#                 X.append(vec)
#             else:
#                 X.append(sum(vec) / len(vec))
#             y.append(text[1])
#         return torch.tensor(np.array(X)).to(self.device), torch.tensor(np.array(y)).to(self.device)
