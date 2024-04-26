from typing import Any

import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ELMOClassificationModel(pl.LightningModule):
    """
    该类用于执行两件事情, 第一个建立双向 GRU 将预训练的词向量再训练, 最后得到
    的结果是一个向量表示一个句子或者是一条评论, 第二个是利用多层感知机(MLP)定
    义文本分类模型, 进行文本分类.
    """

    def __init__(self, input_size, hidden_size, batch_size, output_size):
        super(ELMOClassificationModel, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size

        # 定义双向 GRU.
        self.GRU = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True)

        # 定义多层感知机 MLP.
        self.Linear = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.ReLU = nn.ReLU()
        self.Predict = nn.Linear(in_features=hidden_size, out_features=output_size)

        # 将预测值经过 Softmax 转化为概率.
        self.Softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_text, hidden=None):
        """
        该方法用于实现双向 GRU 训练词向量并进行文本分类,
        即更新词向量与分类预测同时进行.
        :param input_text: batch_size 条评论, 形状为(seq_len, batch_size, input_size).
        :param hidden: GRU 的初始隐藏层.
        :return:
        """
        # 更新词向量的双向 GRU 模型初次运行时, 需要一个初始化的隐藏层.
        # if hidden is None:
        #     hidden = self.default_hidden()

        # 利用双向 GRU 训练词向量.
        # sentences, _ = self.GRU(input_text, hidden)

        # torch.max() 得到两个变量, 一个是 values, 一个是 indices, 这里只选取 values.
        # sentences, _ = torch.max(sentences, 0)

        # 对输入的每一条评论进行分类预测.
        # predict_text = self.Predict(self.ReLU(self.Linear(sentences)))
        # predict_text = self.Softmax(predict_text)
        # return predict_text, hidden

        sentences, _ = self.GRU(input_text)
        sentences, _ = torch.max(sentences, 0)
        predict_text = self.Predict(self.ReLU(self.Linear(sentences)))
        return predict_text


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss',loss,prog_bar=True)
        self.log('train_accuracy',self.accuracy(y_hat,y),prog_bar=True)
        return {'loss': loss, 'accuracy': self.accuracy(y_hat, y)}

    def accuracy(self, y_hat, y):
        return torch.sum(torch.argmax(y_hat, dim=1) == y).item() / len(y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss,prog_bar=True)
        self.log('val_accuracy', self.accuracy(y_hat, y),prog_bar=True)
        return {'loss': loss, 'accuracy': self.accuracy(y_hat, y)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss,prog_bar=True)
        self.log('test_accuracy', self.accuracy(y_hat, y),prog_bar=True)
        return {'loss': loss, 'accuracy': self.accuracy(y_hat, y)}

