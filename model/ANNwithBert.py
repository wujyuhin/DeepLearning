import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Any

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

# 调用gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ANNwithBert(pl.LightningModule):
    def __init__(self, input_dim, output_dim,bert_model):
        super(ANNwithBert, self).__init__()
        self.input_dim = input_dim
        self.bert_model = bert_model
        self.validation_step_outputs = []
        # 网络
        self.bert_fc = nn.Linear(bert_model.pooler.dense.out_features, input_dim)
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, output_dim)

        self.dropout = nn.Dropout(0.2)
        # softmax
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, token_type_ids, attention_mask):
        with torch.no_grad():
            out = self.bert_model(input_ids=input_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        # if self.input_dim == out.last_hidden_state.shape[2]:
        #     x = out.last_hidden_state[:, 0, :]  # bert-base-uncased :[batch_size, seq_length, 768]
        # else:
        #     x = self.bert_fc(out.last_hidden_state[:, 0, :])
        x = self.bert_fc(out.last_hidden_state[:, 0, :])
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # 分类
        x = self.softmax(x)
        return x


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y = batch
        y_hat = self(input_ids, token_type_ids, attention_mask)
        loss = self.loss(y_hat, y)
        # loss键与值需要存在
        self.log('train_loss',loss,prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_accuracy',self.accuracy(y_hat,y),prog_bar=True)
        # return {'loss': loss, 'train_accuracy': self.accuracy(y_hat, y)}
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, y = batch
        y_hat = self(input_ids, token_type_ids, attention_mask)
        loss = self.loss(y_hat, y)
        result = {'loss': loss, 'val_accuracy': self.accuracy(y_hat, y)}
        self.validation_step_outputs.append(result)
        return result

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        avg_accuracy = torch.stack([x['val_accuracy'] for x in self.validation_step_outputs]).mean()
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
        return torch.sum(torch.argmax(y_hat, dim=1) == y) / torch.tensor(len(y))
