import numpy as np
import wandb
import pandas as pd
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from model.ELMO import ELMOClassificationModel
import torch.nn.utils.rnn as rnn_utils
import torch
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer
from model.ELMOwithBert import ELMOClassificationModelwithBert
from model.metric import PrintAccuracyAndLossCallback
from model.ANNwithBert import ANNwithBert

class HotelDataset(Dataset):
    """
    定义数据集类
    """
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return len(self.x_train)


def collator_fn(data):
    """
    定义自己的 collate_fn 方法, 用于处理变长数据.
    :param data:  data = pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])
    :return: (torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)), torch.tensor(labels)
    """
    sents = [i[0].replace(" ","") for i in data]
    labels = [i[1] for i in data]
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                       add_special_tokens=True,  # add_special_tokens 会给输入加上[cls], [sep] 等特殊token
                                       padding="max_length",
                                       return_tensors='pt',
                                       max_length=256,
                                       truncation=True)  # 512 是默认值
    input_ids = data['input_ids']
    token_type_ids = data['token_type_ids']
    attention_mask = data['attention_mask']
    # labels必须是tensor，因为后面的forward函数中的input_ids, token_type_ids, attention_mask都是tensor
    return input_ids, token_type_ids, attention_mask, torch.tensor(labels)


if __name__ == '__main__':
    # 定义路径
    # data_path = '../../data/ChineseNlpCorpus/_processed.csv'
    data_path = '../data/ChineseNlpCorpus_processed.csv'
    # 数据导入
    data = pd.read_csv(data_path)
    # bert模型导入和配置
    model_name = 'bert-base-uncased'
    # model_name = 'uerchinese_roberta_L-2_H-128'
    tokenizer = BertTokenizer.from_pretrained(model_name)  # 加载分词器
    pre_trained = BertModel.from_pretrained(model_name)  # 加载模型
    for parameter in pre_trained.parameters():  # 冻结bert模型参数
        parameter.requires_grad_(False)

    # ================================  1.在train_loader中使用bert     =================================================
    # 切分训练集和验证集、测试集
    X = data['review']  # 形如：pd.DataFrame(['我 是 一个 好人', '你 是 一个 坏人'])
    y = data['label']  # 形如：pd.DataFrame([1,0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_dev.reset_index(drop=True, inplace=True)
    y_dev.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
    # 数据集和数据加载器
    train_data = HotelDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=False, collate_fn=collator_fn)
    dev_data = HotelDataset(X_dev, y_dev)
    dev_loader = DataLoader(dataset=dev_data, batch_size=2000, shuffle=True, drop_last=False, collate_fn=collator_fn)
    test_data = HotelDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, drop_last=False, collate_fn=collator_fn)

    #  =====================================  模型训练 pytorch_lightning  ============================================
    # model = ELMOClassificationModelwithBert(input_size=768, hidden_size=64, batch_size=64, output_size=2,
    #                                         bert_model=pre_trained)
    model = ANNwithBert(input_dim=128, output_dim=2, bert_model=pre_trained)
    trainer = pl.Trainer(max_epochs=500)
    trainer.fit(model, train_loader, dev_loader)
    trainer.test(model, test_loader)

    # ====================================== 训练模型【简单板块】  ======================================================
    # from torch import nn, optim
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = ELMOClassificationModelwithBert(input_size=128, hidden_size=64, batch_size=64, output_size=2, bert_model=pre_trained)
    # model.to(device)
    # criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    # optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
    # for epoch in range(100):
    #     model.train()
    #     for i, data in enumerate(train_loader):
    #         inputs, labels = data
    #         optimizer.zero_grad()  # 梯度清零
    #         output = model(inputs)  # 前向传播 input_dim=(128,1), output_dim=2
    #         loss = criterion(output, labels.squeeze())  # 计算损失 labels.squeeze()将标签的维度从(64,1)变为(64,)
    #         loss.backward()  # 反向传播
    #         optimizer.step()  # 更新参数
    #     if epoch % 10 == 0:
    #         print('epoch: {}, loss: {}'.format(epoch, loss.item()))

    #  ===============================  先bert再生成train_loader =================================================
    # # 使用bert进行文本索引化
    # sents = [i for i in X_train]
    # labels = [i for i in y_train]
    # data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
    #                                    add_special_tokens=True,  # add_special_tokens 会给输入加上[cls], [sep] 等特殊token
    #                                    padding="max_length",
    #                                    return_tensors='pt',
    #                                    max_length=256,
    #                                    truncation=True)  # 512 是默认值
    # input_ids = data['input_ids']
    # token_type_ids = data['token_type_ids']
    # attention_mask = data['attention_mask']
    # train_data = HotelDataset(list(zip(input_ids, token_type_ids, attention_mask)), labels)
    # train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, drop_last=False, collate_fn=None)
