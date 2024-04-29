from typing import Any

import torch.nn as nn
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ELMOClassificationModelwithBert(pl.LightningModule):
    """
    该类用于执行两件事情, 第一个建立双向 GRU 将预训练的词向量再训练, 最后得到
    的结果是一个向量表示一个句子或者是一条评论, 第二个是利用多层感知机(MLP)定
    义文本分类模型, 进行文本分类.
    """

    def __init__(self, input_size, hidden_size, batch_size, output_size, bert_model):
        super(ELMOClassificationModelwithBert, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.bert_model = bert_model
        self.validation_step_outputs = []

        # 定义双向 GRU.
        self.GRU = nn.GRU(input_size=bert_model.pooler.dense.out_features, hidden_size=hidden_size, bidirectional=True)

        # 定义多层感知机 MLP.
        self.BertLinear = nn.Linear(in_features=self.bert_model.pooler.dense.out_features, out_features=self.input_size)
        self.Linear = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.ReLU = nn.ReLU()
        self.Predict = nn.Linear(in_features=hidden_size, out_features=output_size)
        # 将预测值经过 Softmax 转化为概率.
        self.Softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, hidden=None):

        with torch.no_grad():
            out = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        # out.last_hidden_state.shape     # [batch_size, seq_length, 默认值1024]
        # 需要转成[seq_length,batch_size, input_size] 以适应GRU的输入
        # GRU_input = self.BertLinear(out.last_hidden_state.permute(1, 0, 2))
        GRU_input = out.last_hidden_state.permute(1, 0, 2)  # [:,:, :self.input_size]
        # 利用双向 GRU 训练词向量.
        sentences, _ = self.GRU(GRU_input)
        # sentences, _ = torch.max(sentences, 0) # [seq_length,batch_size, input_size] --> [batch_size,input_size]
        predict_text = self.Predict(self.ReLU(self.Linear(sentences[0,:,:])))
        return predict_text

    def configure_optimizers(self):
        return torch.optim.Adam(filter(lambda p:p.requires_grad,self.parameters()), lr=0.001)


    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y = batch
        y_hat = self(input_ids, token_type_ids, attention_mask)
        loss = self.loss(y_hat, y)
        self.log('train_loss',loss,prog_bar=True)
        self.log('train_accuracy',self.accuracy(y_hat,y),prog_bar=True)
        return loss
        # return {'loss': loss, 'accuracy': self.accuracy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y = batch
        y_hat = self(input_ids, token_type_ids, attention_mask)
        result = {'loss': self.loss(y_hat, y), 'accuracy': self.accuracy(y_hat, y)}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self):
        avg_loss = torch.tensor([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_accuracy = torch.tensor([x['accuracy'] for x in self.validation_step_outputs]).mean()
        self.log('avg_val_loss', avg_loss, prog_bar=True)
        self.log('avg_val_acc', avg_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y = batch
        y_hat = self(input_ids, token_type_ids, attention_mask)
        loss = self.loss(y_hat, y)
        self.log('test_loss', loss,prog_bar=True)
        self.log('test_accuracy', self.accuracy(y_hat, y),prog_bar=True)
        return {'loss': loss, 'accuracy': self.accuracy(y_hat, y)}

    def accuracy(self, y_hat, y):
        return torch.sum(torch.argmax(y_hat, dim=1) == y) / len(y)
